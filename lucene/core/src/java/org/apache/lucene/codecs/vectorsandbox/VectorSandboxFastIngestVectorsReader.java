/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.codecs.vectorsandbox;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;

/**
 * Reads vectors from the index segments along with index data structures supporting KNN search.
 *
 * @lucene.experimental
 */
public final class VectorSandboxFastIngestVectorsReader extends KnnVectorsReader {

  private static final long SHALLOW_SIZE =
      RamUsageEstimator.shallowSizeOfInstance(VectorSandboxFastIngestVectorsFormat.class);

  private final FieldInfos fieldInfos;
  private final KnnVectorsReader wrapped;
  private final Map<String, FieldEntry> fields = new HashMap<>();
  private final Map<String, Boolean> delegated = new HashMap<>();
  private final IndexInput vectorData;

  VectorSandboxFastIngestVectorsReader(SegmentReadState state, KnnVectorsReader wrapped)
      throws IOException {
    this.fieldInfos = state.fieldInfos;
    this.wrapped = wrapped;
    int versionMeta = readMetadata(state);
    boolean success = false;
    try {
      vectorData =
          openDataInput(
              state,
              versionMeta,
              VectorSandboxFastIngestVectorsFormat.VECTOR_DATA_EXTENSION,
              VectorSandboxFastIngestVectorsFormat.VECTOR_DATA_CODEC_NAME);
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  private int readMetadata(SegmentReadState state) throws IOException {
    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            VectorSandboxFastIngestVectorsFormat.META_EXTENSION);
    int versionMeta = -1;
    try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
      Throwable priorE = null;
      try {
        versionMeta =
            CodecUtil.checkIndexHeader(
                meta,
                VectorSandboxFastIngestVectorsFormat.META_CODEC_NAME,
                VectorSandboxFastIngestVectorsFormat.VERSION_START,
                VectorSandboxFastIngestVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix);
        readFields(meta, state.fieldInfos);
      } catch (Throwable exception) {
        priorE = exception;
      } finally {
        CodecUtil.checkFooter(meta, priorE);
      }
    }
    return versionMeta;
  }

  private static IndexInput openDataInput(
      SegmentReadState state, int versionMeta, String fileExtension, String codecName)
      throws IOException {
    String fileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, fileExtension);
    IndexInput in = state.directory.openInput(fileName, state.context);
    boolean success = false;
    try {
      int versionVectorData =
          CodecUtil.checkIndexHeader(
              in,
              codecName,
              VectorSandboxVamanaVectorsFormat.VERSION_START,
              VectorSandboxVamanaVectorsFormat.VERSION_CURRENT,
              state.segmentInfo.getId(),
              state.segmentSuffix);
      if (versionMeta != versionVectorData) {
        throw new CorruptIndexException(
            "Format versions mismatch: meta="
                + versionMeta
                + ", "
                + codecName
                + "="
                + versionVectorData,
            in);
      }
      CodecUtil.retrieveChecksum(in);
      success = true;
      return in;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(in);
      }
    }
  }

  private void readFields(ChecksumIndexInput meta, FieldInfos infos) throws IOException {
    for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
      FieldInfo info = infos.fieldInfo(fieldNumber);
      if (info == null) {
        throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
      }
      FieldEntry fieldEntry = readField(meta);
      validateFieldEntry(info, fieldEntry);
      fields.put(info.name, fieldEntry);

      boolean delegated = readDelegated(meta);
      this.delegated.put(info.name, delegated);
    }
  }

  private void validateFieldEntry(FieldInfo info, FieldEntry fieldEntry) {
    int dimension = info.getVectorDimension();
    if (dimension != fieldEntry.dimension) {
      throw new IllegalStateException(
          "Inconsistent vector dimension for field=\""
              + info.name
              + "\"; "
              + dimension
              + " != "
              + fieldEntry.dimension);
    }
  }

  private VectorSimilarityFunction readSimilarityFunction(DataInput input) throws IOException {
    int similarityFunctionId = input.readInt();
    if (similarityFunctionId < 0
        || similarityFunctionId >= VectorSimilarityFunction.values().length) {
      throw new CorruptIndexException(
          "Invalid similarity function id: " + similarityFunctionId, input);
    }
    return VectorSimilarityFunction.values()[similarityFunctionId];
  }

  private VectorEncoding readVectorEncoding(DataInput input) throws IOException {
    int encodingId = input.readInt();
    if (encodingId < 0 || encodingId >= VectorEncoding.values().length) {
      throw new CorruptIndexException("Invalid vector encoding id: " + encodingId, input);
    }
    return VectorEncoding.values()[encodingId];
  }

  private FieldEntry readField(IndexInput input) throws IOException {
    VectorEncoding vectorEncoding = readVectorEncoding(input);
    VectorSimilarityFunction similarityFunction = readSimilarityFunction(input);
    return new FieldEntry(input, vectorEncoding, similarityFunction);
  }

  private boolean readDelegated(IndexInput input) throws IOException {
    int delegated = input.readInt();
    if (delegated != 0 && delegated != 1) {
      throw new CorruptIndexException("Invalid delegated value: " + delegated, input);
    }

    return delegated == 1;
  }

  @Override
  public long ramBytesUsed() {
    return VectorSandboxFastIngestVectorsReader.SHALLOW_SIZE
        + RamUsageEstimator.sizeOfMap(
            fields, RamUsageEstimator.shallowSizeOfInstance(FieldEntry.class));
  }

  @Override
  public void checkIntegrity() throws IOException {
    CodecUtil.checksumEntireFile(vectorData);
  }

  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    if (delegated.get(field)) {
      return this.wrapped.getFloatVectorValues(field);
    }

    FieldEntry fieldEntry = fields.get(field);
    if (fieldEntry.vectorEncoding != VectorEncoding.FLOAT32) {
      throw new IllegalArgumentException(
          "field=\""
              + field
              + "\" is encoded as: "
              + fieldEntry.vectorEncoding
              + " expected: "
              + VectorEncoding.FLOAT32);
    }

    return OffHeapFloatVectorValues.load(
        fieldEntry.ordToDoc,
        fieldEntry.vectorEncoding,
        fieldEntry.dimension,
        fieldEntry.vectorDataOffset,
        fieldEntry.vectorDataLength,
        vectorData);
  }

  @Override
  public ByteVectorValues getByteVectorValues(String field) throws IOException {
    if (delegated.get(field)) {
      return this.wrapped.getByteVectorValues(field);
    }

    FieldEntry fieldEntry = fields.get(field);
    if (fieldEntry.vectorEncoding != VectorEncoding.BYTE) {
      throw new IllegalArgumentException(
          "field=\""
              + field
              + "\" is encoded as: "
              + fieldEntry.vectorEncoding
              + " expected: "
              + VectorEncoding.FLOAT32);
    }

    return OffHeapByteVectorValues.load(
        fieldEntry.ordToDoc,
        fieldEntry.vectorEncoding,
        fieldEntry.dimension,
        fieldEntry.vectorDataOffset,
        fieldEntry.vectorDataLength,
        vectorData);
  }

  @Override
  public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    if (delegated.get(field)) {
      this.wrapped.search(field, target, knnCollector, acceptDocs);
      return;
    }

    // FIXME: do brute force
    throw new UnsupportedOperationException();
  }

  @Override
  public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    if (delegated.get(field)) {
      this.wrapped.search(field, target, knnCollector, acceptDocs);
      return;
    }

    // FIXME: do brute force
    throw new UnsupportedOperationException();
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(vectorData);
    this.wrapped.close();
  }

  static class FieldEntry implements Accountable {

    private static final long SHALLOW_SIZE =
        RamUsageEstimator.shallowSizeOfInstance(FieldEntry.class);
    final VectorSimilarityFunction similarityFunction;
    final VectorEncoding vectorEncoding;
    final long vectorDataOffset;
    final long vectorDataLength;
    final int dimension;
    final int size;
    final OrdToDocDISIReaderConfiguration ordToDoc;

    FieldEntry(
        IndexInput input,
        VectorEncoding vectorEncoding,
        VectorSimilarityFunction similarityFunction)
        throws IOException {
      this.similarityFunction = similarityFunction;
      this.vectorEncoding = vectorEncoding;
      vectorDataOffset = input.readVLong();
      vectorDataLength = input.readVLong();
      dimension = input.readVInt();
      size = input.readInt();
      ordToDoc = OrdToDocDISIReaderConfiguration.fromStoredMeta(input, size);
    }

    int size() {
      return size;
    }

    @Override
    public long ramBytesUsed() {
      return SHALLOW_SIZE + RamUsageEstimator.sizeOf(ordToDoc);
    }
  }
}
