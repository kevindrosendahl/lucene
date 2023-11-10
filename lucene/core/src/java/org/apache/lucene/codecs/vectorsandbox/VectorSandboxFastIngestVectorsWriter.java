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

import static org.apache.lucene.codecs.vectorsandbox.VectorSandboxVamanaVectorsFormat.DIRECT_MONOTONIC_BLOCK_SHIFT;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.RamUsageEstimator;

/**
 * Writes vector values and knn graphs to index segments.
 *
 * @lucene.experimental
 */
public final class VectorSandboxFastIngestVectorsWriter extends KnnVectorsWriter {

  private final SegmentWriteState segmentWriteState;
  private final IndexOutput meta, vectorData;
  private final KnnVectorsWriter wrapped;
  private final List<FieldWriter<?>> fields = new ArrayList<>();
  private boolean finished;

  VectorSandboxFastIngestVectorsWriter(SegmentWriteState state, KnnVectorsWriter wrapped)
      throws IOException {
    segmentWriteState = state;
    this.wrapped = wrapped;
    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            VectorSandboxFastIngestVectorsFormat.META_EXTENSION);

    String vectorDataFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            VectorSandboxFastIngestVectorsFormat.VECTOR_DATA_EXTENSION);

    boolean success = false;
    try {
      meta = state.directory.createOutput(metaFileName, state.context);
      vectorData = state.directory.createOutput(vectorDataFileName, state.context);

      CodecUtil.writeIndexHeader(
          meta,
          VectorSandboxFastIngestVectorsFormat.META_CODEC_NAME,
          VectorSandboxFastIngestVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          vectorData,
          VectorSandboxFastIngestVectorsFormat.VECTOR_DATA_CODEC_NAME,
          VectorSandboxFastIngestVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  @Override
  public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    FieldWriter<?> newField = FieldWriter.create(fieldInfo, segmentWriteState.infoStream);
    fields.add(newField);
    return newField;
  }

  @Override
  public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    for (FieldWriter<?> field : fields) {
      writeField(field, maxDoc);
    }
  }

  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    wrapped.finish();

    if (meta != null) {
      // write end of fields marker
      meta.writeInt(-1);
      CodecUtil.writeFooter(meta);
    }
    if (vectorData != null) {
      CodecUtil.writeFooter(vectorData);
    }
  }

  @Override
  public long ramBytesUsed() {
    long total = 0;
    for (FieldWriter<?> field : fields) {
      total += field.ramBytesUsed();
    }
    return total;
  }

  private void writeField(FieldWriter<?> fieldData, int maxDoc) throws IOException {
    long vectorDataOffset = vectorData.alignFilePointer(Float.BYTES);
    // write vector values
    switch (fieldData.fieldInfo.getVectorEncoding()) {
      case BYTE -> writeByteVectors(fieldData);
      case FLOAT32 -> writeFloat32Vectors(fieldData);
    }
    long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;

    writeMeta(
        fieldData.fieldInfo,
        maxDoc,
        vectorDataOffset,
        vectorDataLength,
        fieldData.docsWithField,
        false);
  }

  private void writeFloat32Vectors(FieldWriter<?> fieldData) throws IOException {
    final ByteBuffer buffer =
        ByteBuffer.allocate(fieldData.dim * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (Object v : fieldData.vectors) {
      buffer.asFloatBuffer().put((float[]) v);
      vectorData.writeBytes(buffer.array(), buffer.array().length);
    }
  }

  private void writeByteVectors(FieldWriter<?> fieldData) throws IOException {
    for (Object v : fieldData.vectors) {
      byte[] vector = (byte[]) v;
      vectorData.writeBytes(vector, vector.length);
    }
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    wrapped.mergeOneField(fieldInfo, mergeState);
    writeMeta(fieldInfo, 0, 0, 0, new DocsWithFieldSet(), true);
  }

  private void writeMeta(
      FieldInfo field,
      int maxDoc,
      long vectorDataOffset,
      long vectorDataLength,
      DocsWithFieldSet docsWithField,
      boolean delegated)
      throws IOException {
    meta.writeInt(field.number);
    meta.writeInt(field.getVectorEncoding().ordinal());
    meta.writeInt(field.getVectorSimilarityFunction().ordinal());
    meta.writeVLong(vectorDataOffset);
    meta.writeVLong(vectorDataLength);
    meta.writeVInt(field.getVectorDimension());

    // write docIDs
    int count = docsWithField.cardinality();
    meta.writeInt(count);
    OrdToDocDISIReaderConfiguration.writeStoredMeta(
        DIRECT_MONOTONIC_BLOCK_SHIFT, meta, vectorData, count, maxDoc, docsWithField);

    meta.writeInt(delegated ? 1 : 0);
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(meta, vectorData);
    this.wrapped.close();
  }

  private abstract static class FieldWriter<T> extends KnnFieldVectorsWriter<T> {

    private final FieldInfo fieldInfo;
    private final int dim;
    private final DocsWithFieldSet docsWithField;
    private final List<T> vectors;
    private int lastDocID = -1;

    static FieldWriter<?> create(FieldInfo fieldInfo, InfoStream infoStream) throws IOException {
      int dim = fieldInfo.getVectorDimension();
      return switch (fieldInfo.getVectorEncoding()) {
        case BYTE -> new VectorSandboxFastIngestVectorsWriter.FieldWriter<byte[]>(
            fieldInfo, infoStream) {
          @Override
          public byte[] copyValue(byte[] value) {
            return ArrayUtil.copyOfSubArray(value, 0, dim);
          }
        };
        case FLOAT32 -> new VectorSandboxFastIngestVectorsWriter.FieldWriter<float[]>(
            fieldInfo, infoStream) {
          @Override
          public float[] copyValue(float[] value) {
            return ArrayUtil.copyOfSubArray(value, 0, dim);
          }
        };
      };
    }

    @SuppressWarnings("unchecked")
    FieldWriter(FieldInfo fieldInfo, InfoStream infoStream) throws IOException {
      this.fieldInfo = fieldInfo;
      this.dim = fieldInfo.getVectorDimension();
      this.docsWithField = new DocsWithFieldSet();
      vectors = new ArrayList<>();
    }

    @Override
    public void addValue(int docID, T vectorValue) throws IOException {
      if (docID == lastDocID) {
        throw new IllegalArgumentException(
            "VectorValuesField \""
                + fieldInfo.name
                + "\" appears more than once in this document (only one value is allowed per field)");
      }
      assert docID > lastDocID;
      T copy = copyValue(vectorValue);
      docsWithField.add(docID);
      vectors.add(copy);
      lastDocID = docID;
    }

    @Override
    public long ramBytesUsed() {
      if (vectors.size() == 0) {
        return 0;
      }
      return docsWithField.ramBytesUsed()
          + (long) vectors.size()
              * (RamUsageEstimator.NUM_BYTES_OBJECT_REF + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER)
          + (long) vectors.size()
              * fieldInfo.getVectorDimension()
              * fieldInfo.getVectorEncoding().byteSize;
    }
  }
}
