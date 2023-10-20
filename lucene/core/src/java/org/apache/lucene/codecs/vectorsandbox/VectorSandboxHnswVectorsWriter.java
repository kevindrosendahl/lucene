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

import static org.apache.lucene.codecs.vectorsandbox.VectorSandboxHnswVectorsFormat.DIRECT_MONOTONIC_BLOCK_SHIFT;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
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
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraph.NodesIterator;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.apache.lucene.util.hnsw.OnHeapHnswGraph;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.packed.DirectMonotonicWriter;
import org.apache.lucene.util.vectors.RandomAccessVectorValues;

/**
 * Writes vector values and knn graphs to index segments.
 *
 * @lucene.experimental
 */
public final class VectorSandboxHnswVectorsWriter extends KnnVectorsWriter {

  private final SegmentWriteState segmentWriteState;
  private final IndexOutput meta, vectorData, vectorIndex;
  private final int M;
  private final int beamWidth;

  private final List<FieldWriter<?>> fields = new ArrayList<>();
  private boolean finished;

  VectorSandboxHnswVectorsWriter(SegmentWriteState state, int M, int beamWidth) throws IOException {
    this.M = M;
    this.beamWidth = beamWidth;
    segmentWriteState = state;
    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix,
            VectorSandboxHnswVectorsFormat.META_EXTENSION);

    String vectorDataFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            VectorSandboxHnswVectorsFormat.VECTOR_DATA_EXTENSION);

    String indexDataFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            VectorSandboxHnswVectorsFormat.VECTOR_INDEX_EXTENSION);
    boolean success = false;
    try {
      meta = state.directory.createOutput(metaFileName, state.context);
      vectorData = state.directory.createOutput(vectorDataFileName, state.context);
      vectorIndex = state.directory.createOutput(indexDataFileName, state.context);

      CodecUtil.writeIndexHeader(
          meta,
          VectorSandboxHnswVectorsFormat.META_CODEC_NAME,
          VectorSandboxHnswVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          vectorData,
          VectorSandboxHnswVectorsFormat.VECTOR_DATA_CODEC_NAME,
          VectorSandboxHnswVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          vectorIndex,
          VectorSandboxHnswVectorsFormat.VECTOR_INDEX_CODEC_NAME,
          VectorSandboxHnswVectorsFormat.VERSION_CURRENT,
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
    FieldWriter<?> newField =
        FieldWriter.create(fieldInfo, M, beamWidth, segmentWriteState.infoStream);
    fields.add(newField);
    return newField;
  }

  @Override
  public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    for (FieldWriter<?> field : fields) {
      if (sortMap == null) {
        writeField(field, maxDoc);
      } else {
        throw new UnsupportedOperationException();
      }
    }
  }

  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;

    if (meta != null) {
      // write end of fields marker
      meta.writeInt(-1);
      CodecUtil.writeFooter(meta);
    }
    if (vectorData != null) {
      CodecUtil.writeFooter(vectorData);
      CodecUtil.writeFooter(vectorIndex);
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
    // TODO: write full fidelity vectors if quantizing
    long vectorDataOffset = vectorData.alignFilePointer(Float.BYTES);
    long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;

    // write graph
    long vectorIndexOffset = vectorIndex.getFilePointer();
    int[][] graphLevelNodeOffsets = writeGraph(fieldData);
    long vectorIndexLength = vectorIndex.getFilePointer() - vectorIndexOffset;

    writeMeta(
        fieldData.fieldInfo,
        maxDoc,
        vectorDataOffset,
        vectorDataLength,
        vectorIndexOffset,
        vectorIndexLength,
        fieldData.docsWithField,
        fieldData.getGraph(),
        graphLevelNodeOffsets);
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    throw new UnsupportedOperationException();
  }

  private int[][] writeGraph(FieldWriter<?> fieldData) throws IOException {
    var graph = fieldData.getGraph();
    if (graph == null) {
      return new int[0][0];
    }

    var encoding = fieldData.fieldInfo.getVectorEncoding();
    var vectorBuffer =
        encoding == VectorEncoding.FLOAT32 ? ByteBuffer.allocate(fieldData.dim * Float.SIZE)
            .order(ByteOrder.LITTLE_ENDIAN) : null;

    // offsets[level][nodeIdx] is the offset into the index file for that node in that level.
    // note that nodeIdx is not the ordinal, but rather that ordinal's position in the sorted
    // list of ordinals in that level.
    int[][] offsets = new int[graph.numLevels()][];

    for (int level = 0; level < graph.numLevels(); level++) {
      int[] sortedNodes = getSortedNodes(graph.getNodesOnLevel(level));
      offsets[level] = new int[sortedNodes.length];
      int nodeOffsetId = 0;
      long offsetStart = vectorIndex.getFilePointer();

      for (int node : sortedNodes) {
        var offset = Math.toIntExact(vectorIndex.getFilePointer() - offsetStart);
        offsets[level][nodeOffsetId++] = offset;

        // Write the full fidelity vector
        var vector = fieldData.vectors.get(node);
        switch (encoding) {
          case BYTE -> {
            byte[] v = (byte[]) vector;
            vectorIndex.writeBytes(v, v.length);
          }
          case FLOAT32 -> {
            vectorBuffer.asFloatBuffer().put((float[]) vector);
            vectorIndex.writeBytes(vectorBuffer.array(), vectorBuffer.array().length);
          }
        }

        // Encode neighbors as vints.
        NeighborArray neighbors = graph.getNeighbors(level, node);
        int size = neighbors.size();
        int[] nnodes = neighbors.node();
        Arrays.sort(nnodes, 0, size);

        for (int i = size - 1; i > 0; --i) {
          nnodes[i] -= nnodes[i - 1];
        }
        for (int i = 0; i < size; i++) {
          vectorIndex.writeVInt(nnodes[i]);
        }

        if (encoding == VectorEncoding.FLOAT32) {
          vectorIndex.alignFilePointer(Float.BYTES);
        }
      }
    }

    return offsets;
  }

  public static int[] getSortedNodes(NodesIterator nodesOnLevel) {
    int[] sortedNodes = new int[nodesOnLevel.size()];
    for (int n = 0; nodesOnLevel.hasNext(); n++) {
      sortedNodes[n] = nodesOnLevel.nextInt();
    }
    Arrays.sort(sortedNodes);
    return sortedNodes;
  }

  private void writeMeta(
      FieldInfo field,
      int maxDoc,
      long vectorDataOffset,
      long vectorDataLength,
      long vectorIndexOffset,
      long vectorIndexLength,
      DocsWithFieldSet docsWithField,
      HnswGraph graph,
      int[][] graphLevelNodeOffsets)
      throws IOException {
    meta.writeInt(field.number);
    meta.writeInt(field.getVectorEncoding().ordinal());
    meta.writeInt(field.getVectorSimilarityFunction().ordinal());
    meta.writeVLong(vectorDataOffset);
    meta.writeVLong(vectorDataLength);
    meta.writeVLong(vectorIndexOffset);
    meta.writeVLong(vectorIndexLength);
    meta.writeVInt(field.getVectorDimension());

    // write docIDs
    int count = docsWithField.cardinality();
    meta.writeInt(count);
    OrdToDocDISIReaderConfiguration.writeStoredMeta(
        DIRECT_MONOTONIC_BLOCK_SHIFT, meta, vectorData, count, maxDoc, docsWithField);
    meta.writeVInt(M);

    if (graph == null) {
      meta.writeVInt(0);
      return;
    }

    // Write graph nodes on each level
    meta.writeVInt(graph.numLevels());
    long valueCount = 0;
    for (int level = 0; level < graph.numLevels(); level++) {
      NodesIterator nodesOnLevel = graph.getNodesOnLevel(level);
      valueCount += nodesOnLevel.size();
      if (level > 0) {
        int[] nol = new int[nodesOnLevel.size()];
        int numberConsumed = nodesOnLevel.consume(nol);
        Arrays.sort(nol);
        assert numberConsumed == nodesOnLevel.size();
        meta.writeVInt(nol.length); // number of nodes on a level
        for (int i = nodesOnLevel.size() - 1; i > 0; --i) {
          nol[i] -= nol[i - 1];
        }
        for (int n : nol) {
          assert n >= 0 : "delta encoding for nodes failed; expected nodes to be sorted";
          meta.writeVInt(n);
        }
      } else {
        assert nodesOnLevel.size() == count : "Level 0 expects to have all nodes";
      }
    }

    // Include node offset data
    long start = vectorIndex.getFilePointer();
    meta.writeLong(start);
    meta.writeVInt(DIRECT_MONOTONIC_BLOCK_SHIFT);
    final DirectMonotonicWriter memoryOffsetsWriter =
        DirectMonotonicWriter.getInstance(
            meta, vectorIndex, valueCount, DIRECT_MONOTONIC_BLOCK_SHIFT);
    long cumulativeOffsetSum = 0;
    for (int[] levelOffsets : graphLevelNodeOffsets) {
      for (int v : levelOffsets) {
        memoryOffsetsWriter.add(cumulativeOffsetSum);
        cumulativeOffsetSum += v;
      }
    }
    memoryOffsetsWriter.finish();
    meta.writeLong(vectorIndex.getFilePointer() - start);
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(meta, vectorData, vectorIndex);
  }

  private abstract static class FieldWriter<T> extends KnnFieldVectorsWriter<T> {

    private final FieldInfo fieldInfo;
    private final int dim;
    private final DocsWithFieldSet docsWithField;
    private final List<T> vectors;
    private final HnswGraphBuilder hnswGraphBuilder;

    private int lastDocID = -1;
    private int node = 0;

    static FieldWriter<?> create(FieldInfo fieldInfo, int M, int beamWidth, InfoStream infoStream)
        throws IOException {
      int dim = fieldInfo.getVectorDimension();
      return switch (fieldInfo.getVectorEncoding()) {
        case BYTE -> new VectorSandboxHnswVectorsWriter.FieldWriter<byte[]>(fieldInfo, M, beamWidth,
            infoStream) {
          @Override
          public byte[] copyValue(byte[] value) {
            return ArrayUtil.copyOfSubArray(value, 0, dim);
          }
        };
        case FLOAT32 ->
            new VectorSandboxHnswVectorsWriter.FieldWriter<float[]>(fieldInfo, M, beamWidth,
                infoStream) {
              @Override
              public float[] copyValue(float[] value) {
                return ArrayUtil.copyOfSubArray(value, 0, dim);
              }
            };
      };
    }

    @SuppressWarnings("unchecked")
    FieldWriter(FieldInfo fieldInfo, int M, int beamWidth, InfoStream infoStream)
        throws IOException {
      this.fieldInfo = fieldInfo;
      this.dim = fieldInfo.getVectorDimension();
      this.docsWithField = new DocsWithFieldSet();
      vectors = new ArrayList<>();
      RAVectorValues<T> raVectors = new RAVectorValues<>(vectors, dim);
      RandomVectorScorerSupplier scorerSupplier =
          switch (fieldInfo.getVectorEncoding()) {
            case BYTE -> RandomVectorScorerSupplier.createBytes(
                (RandomAccessVectorValues<byte[]>) raVectors,
                fieldInfo.getVectorSimilarityFunction());
            case FLOAT32 -> RandomVectorScorerSupplier.createFloats(
                (RandomAccessVectorValues<float[]>) raVectors,
                fieldInfo.getVectorSimilarityFunction());
          };
      hnswGraphBuilder =
          HnswGraphBuilder.create(scorerSupplier, M, beamWidth, HnswGraphBuilder.randSeed);
      hnswGraphBuilder.setInfoStream(infoStream);
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
      docsWithField.add(docID);
      vectors.add(copyValue(vectorValue));
      hnswGraphBuilder.addGraphNode(node);
      node++;
      lastDocID = docID;
    }

    OnHeapHnswGraph getGraph() {
      if (vectors.size() > 0) {
        return hnswGraphBuilder.getGraph();
      } else {
        return null;
      }
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
          * fieldInfo.getVectorEncoding().byteSize
          + hnswGraphBuilder.getGraph().ramBytesUsed();
    }
  }

  private static class RAVectorValues<T> implements RandomAccessVectorValues<T> {

    private final List<T> vectors;
    private final int dim;

    RAVectorValues(List<T> vectors, int dim) {
      this.vectors = vectors;
      this.dim = dim;
    }

    @Override
    public int size() {
      return vectors.size();
    }

    @Override
    public int dimension() {
      return dim;
    }

    @Override
    public T vectorValue(int targetOrd) throws IOException {
      return vectors.get(targetOrd);
    }

    @Override
    public RandomAccessVectorValues<T> copy() throws IOException {
      return this;
    }
  }
}
