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

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.IntStream;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.VamanaGraphProvider;
import org.apache.lucene.codecs.vectorsandbox.VectorSandboxVamanaVectorsFormat.PQRerank;
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
import org.apache.lucene.search.KnnCollector.Reranker;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.ScalarQuantizer;
import org.apache.lucene.util.iouring.IoUring;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.apache.lucene.util.pq.PQVectorScorer;
import org.apache.lucene.util.pq.ProductQuantization;
import org.apache.lucene.util.pq.ProductQuantization.Codebook;
import org.apache.lucene.util.vamana.OrdinalTranslatedKnnCollector;
import org.apache.lucene.util.vamana.RandomAccessVectorValues;
import org.apache.lucene.util.vamana.RandomVectorScorer;
import org.apache.lucene.util.vamana.VamanaGraph;
import org.apache.lucene.util.vamana.VamanaGraph.ArrayNodesIterator;
import org.apache.lucene.util.vamana.VamanaGraph.NodesIterator;
import org.apache.lucene.util.vamana.VamanaGraphSearcher;
import org.apache.lucene.util.vamana.VamanaGraphSearcher.CachedNode;

/**
 * Reads vectors from the index segments along with index data structures supporting KNN search.
 *
 * @lucene.experimental
 */
public final class VectorSandboxVamanaVectorsReader extends KnnVectorsReader
    implements QuantizedVectorsReader, VamanaGraphProvider {

  private static final ExecutorService PARALLEL_READ_EXECUTOR;

  static {
    if (System.getenv("VAMANA_PARALLEL_RERANK_THREADS") == null
        || System.getenv("VAMANA_PARALLEL_RERANK_THREADS").equals("null")
        || System.getenv("VAMANA_PARALLEL_RERANK_THREADS").equals("unbounded")) {
      PARALLEL_READ_EXECUTOR = Executors.newCachedThreadPool();
    } else {
      int numThreads = Integer.parseInt(System.getenv("VAMANA_PARALLEL_RERANK_THREADS"));
      PARALLEL_READ_EXECUTOR = Executors.newFixedThreadPool(numThreads);
    }
  }

  private static final int NODE_CACHE_DEGREE =
      System.getenv("VAMANA_CACHE_DEGREE") == null
              || System.getenv("VAMANA_CACHE_DEGREE").equals("null")
          ? -1
          : Integer.parseInt(System.getenv("VAMANA_CACHE_DEGREE"));

  private static final long SHALLOW_SIZE =
      RamUsageEstimator.shallowSizeOfInstance(VectorSandboxVamanaVectorsFormat.class);

  private final FieldInfos fieldInfos;
  public final Map<String, FieldEntry> fields = new HashMap<>();
  public final Map<String, byte[][]> pqVectors = new HashMap<>();
  public final Map<String, Map<Integer, CachedNode>> nodeCaches = new HashMap<>();
  private final IndexInput vectorData;
  private final IndexInput vectorIndex;
  private final IndexInput quantizedVectorData;
  private final PQRerank pqRerank;
  private final VectorSandboxScalarQuantizedVectorsReader quantizedVectorsReader;
  private final IoUring.FileFactory uringFactory;
  private final FileChannel channel;
  private final ThreadLocal<IoUring> uring;

  VectorSandboxVamanaVectorsReader(SegmentReadState state, PQRerank pqRerank) throws IOException {
    this.fieldInfos = state.fieldInfos;
    this.pqRerank = pqRerank;
    int versionMeta = readMetadata(state);
    boolean success = false;
    try {
      readPqVectors(state, versionMeta);
      vectorData =
          openDataInput(
              state,
              versionMeta,
              VectorSandboxVamanaVectorsFormat.VECTOR_DATA_EXTENSION,
              VectorSandboxVamanaVectorsFormat.VECTOR_DATA_CODEC_NAME);
      vectorIndex =
          openDataInput(
              state,
              versionMeta,
              VectorSandboxVamanaVectorsFormat.VECTOR_INDEX_EXTENSION,
              VectorSandboxVamanaVectorsFormat.VECTOR_INDEX_CODEC_NAME);

      // FIXME: read from env var
      boolean mlock =
          (System.getenv("VAMANA_MLOCK") != null && !System.getenv("VAMANA_MLOCK").equals("null"))
              && Boolean.parseBoolean(System.getenv("VAMANA_MLOCK"));
      if (mlock) {
        vectorIndex.mlock();
      }

      if (pqRerank == PQRerank.IO_URING) {
        Path vectorDataFileName =
            ((FSDirectory) state.directory)
                .getDirectory()
                .resolve(
                    IndexFileNames.segmentFileName(
                        state.segmentInfo.name,
                        state.segmentSuffix,
                        VectorSandboxVamanaVectorsFormat.VECTOR_DATA_EXTENSION));
        uringFactory = IoUring.factory(vectorDataFileName);
        uring = ThreadLocal.withInitial(() -> uringFactory.create(100));
      } else {
        uringFactory = null;
        uring = null;
      }

      if (pqRerank == PQRerank.PARALLEL_FILE_IO) {
        Path vectorDataFileName =
            ((FSDirectory) state.directory)
                .getDirectory()
                .resolve(
                    IndexFileNames.segmentFileName(
                        state.segmentInfo.name,
                        state.segmentSuffix,
                        VectorSandboxVamanaVectorsFormat.VECTOR_DATA_EXTENSION));
        channel = FileChannel.open(vectorDataFileName, StandardOpenOption.READ);
      } else {
        channel = null;
      }

      if (NODE_CACHE_DEGREE > 0) {
        buildNodeCaches();
      }

      if (fields.values().stream().anyMatch(FieldEntry::hasQuantizedVectors)) {
        quantizedVectorData =
            openDataInput(
                state,
                versionMeta,
                VectorSandboxScalarQuantizedVectorsFormat.QUANTIZED_VECTOR_DATA_EXTENSION,
                VectorSandboxScalarQuantizedVectorsFormat.QUANTIZED_VECTOR_DATA_CODEC_NAME);
        quantizedVectorsReader = new VectorSandboxScalarQuantizedVectorsReader(quantizedVectorData);
      } else {
        quantizedVectorData = null;
        quantizedVectorsReader = null;
      }
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
            VectorSandboxVamanaVectorsFormat.META_EXTENSION);
    int versionMeta = -1;
    try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
      Throwable priorE = null;
      try {
        versionMeta =
            CodecUtil.checkIndexHeader(
                meta,
                VectorSandboxVamanaVectorsFormat.META_CODEC_NAME,
                VectorSandboxVamanaVectorsFormat.VERSION_START,
                VectorSandboxVamanaVectorsFormat.VERSION_CURRENT,
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

  private void readPqVectors(SegmentReadState state, int versionMeta) throws IOException {
    try (var pqData =
        openDataInput(
            state,
            versionMeta,
            VectorSandboxVamanaVectorsFormat.PQ_DATA_EXTENSION,
            VectorSandboxVamanaVectorsFormat.PQ_DATA_CODEC_NAME)) {
      for (var entry : fields.entrySet()) {
        var field = entry.getKey();
        var fieldEntry = entry.getValue();

        if (fieldEntry.pq == null) {
          continue;
        }

        int M = fieldEntry.pq.M();
        pqData.seek(fieldEntry.pqDataOffset);

        byte[][] encoded = new byte[fieldEntry.size][];
        for (int i = 0; i < encoded.length; i++) {
          encoded[i] = new byte[M];
          pqData.readBytes(encoded[i], 0, M);
        }

        pqVectors.put(field, encoded);
      }
    }
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

    if (fieldEntry.hasQuantizedVectors()) {
      int byteSize =
          switch (info.getVectorEncoding()) {
            case BYTE -> Byte.BYTES;
            case FLOAT32 -> Float.BYTES;
          };
      long vectorBytes = Math.multiplyExact((long) dimension, byteSize);
      long numBytes = Math.multiplyExact(vectorBytes, fieldEntry.size);
      if (numBytes != fieldEntry.vectorDataLength) {
        throw new IllegalStateException(
            "Vector data length "
                + fieldEntry.vectorDataLength
                + " not matching size="
                + fieldEntry.size
                + " * dim="
                + dimension
                + " * byteSize="
                + byteSize
                + " = "
                + numBytes);
      }
      VectorSandboxScalarQuantizedVectorsReader.validateFieldEntry(
          info, fieldEntry.dimension, fieldEntry.size, fieldEntry.quantizedVectorDataLength);
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

  private FieldEntry readField(IndexInput meta) throws IOException {
    VectorEncoding vectorEncoding = readVectorEncoding(meta);
    VectorSimilarityFunction similarityFunction = readSimilarityFunction(meta);
    return new FieldEntry(meta, vectorEncoding, similarityFunction);
  }

  private void buildNodeCaches() throws IOException {
    for (var field : fields.keySet()) {
      var graph = getGraph(field);
      var cache = new HashMap<Integer, CachedNode>();
      cacheNeighbors(
          cache, graph, getFloatVectorValues(field), graph.entryNode(), NODE_CACHE_DEGREE);
      nodeCaches.put(field, cache);
    }
  }

  private void cacheNeighbors(
      Map<Integer, CachedNode> cache,
      VamanaGraph graph,
      RandomAccessVectorValues<float[]> vectors,
      int ord,
      int distance)
      throws IOException {
    graph.seek(ord);
    NodesIterator neighbors = graph.getNeighbors();
    int[] neighborCopy = new int[neighbors.size()];
    int i = 0;
    while (neighbors.hasNext()) {
      neighborCopy[i++] = neighbors.nextInt();
    }

    cache.put(ord, new CachedNode(vectors.copy().vectorValue(ord), neighborCopy));

    if (distance > 0) {
      for (int neighbor : neighborCopy) {
        cacheNeighbors(cache, graph, vectors, neighbor, distance - 1);
      }
    }
  }

  @Override
  public long ramBytesUsed() {
    return VectorSandboxVamanaVectorsReader.SHALLOW_SIZE
        + RamUsageEstimator.sizeOfMap(
            fields, RamUsageEstimator.shallowSizeOfInstance(FieldEntry.class));
  }

  @Override
  public void checkIntegrity() throws IOException {
    CodecUtil.checksumEntireFile(vectorData);
    CodecUtil.checksumEntireFile(vectorIndex);
    if (quantizedVectorsReader != null) {
      quantizedVectorsReader.checkIntegrity();
    }
  }

  @Override
  public OffHeapFloatVectorValues getFloatVectorValues(String field) throws IOException {
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
    FieldEntry fieldEntry = fields.get(field);

    if (fieldEntry.size() == 0
        || knnCollector.k() == 0
        || fieldEntry.vectorEncoding != VectorEncoding.FLOAT32) {
      return;
    }
    if (fieldEntry.hasQuantizedVectors()) {
      InGraphOffHeapQuantizedByteVectorValues vectorValues =
          InGraphOffHeapQuantizedByteVectorValues.load(fieldEntry, vectorIndex);
      RandomVectorScorer scorer =
          new ScalarQuantizedRandomVectorScorer(
              fieldEntry.similarityFunction, fieldEntry.scalarQuantizer, vectorValues, target);
      VamanaGraphSearcher.search(
          scorer,
          new OrdinalTranslatedKnnCollector(knnCollector, vectorValues::ordToDoc),
          getGraph(fieldEntry),
          // FIXME: support filtered
          //          vectorValues.getAcceptOrds(acceptDocs));
          acceptDocs,
          null);
    } else {
      RandomAccessVectorValues<float[]> vectorValues =
          fieldEntry.inGraphVectors
              ? InGraphOffHeapFloatVectorValues.load(fieldEntry, vectorIndex)
              : OffHeapFloatVectorValues.load(
                  fieldEntry.ordToDoc,
                  fieldEntry.vectorEncoding,
                  fieldEntry.dimension,
                  fieldEntry.vectorDataOffset,
                  fieldEntry.vectorDataLength,
                  vectorData);

      RandomVectorScorer scorer =
          RandomVectorScorer.createFloats(vectorValues, fieldEntry.similarityFunction, target);

      KnnCollector collector =
          new OrdinalTranslatedKnnCollector(knnCollector, vectorValues::ordToDoc);
      Map<Integer, float[]> cached = new HashMap<>();
      if (pqVectors.containsKey(field)) {
        byte[][] encoded = pqVectors.get(field);
        scorer = new PQVectorScorer(fieldEntry.pq, fieldEntry.similarityFunction, encoded, target);

        if (pqRerank == PQRerank.CACHED) {
          KnnCollector wrapped = collector;
          collector =
              new KnnCollector() {
                @Override
                public boolean earlyTerminated() {
                  return wrapped.earlyTerminated();
                }

                @Override
                public void incVisitedCount(int count) {
                  wrapped.incVisitedCount(count);
                }

                @Override
                public long visitedCount() {
                  return wrapped.visitedCount();
                }

                @Override
                public long visitLimit() {
                  return wrapped.visitLimit();
                }

                @Override
                public int k() {
                  return wrapped.k();
                }

                @Override
                public boolean collect(int docId, float similarity) {
                  return wrapped.collect(docId, similarity);
                }

                @Override
                public float minCompetitiveSimilarity() {
                  return wrapped.minCompetitiveSimilarity();
                }

                @Override
                public TopDocs topDocs() {
                  return wrapped.topDocs();
                }

                @Override
                public void cacheNode(int ordinal) {
                  try {
                    float[] vector = vectorValues.copy().vectorValue(ordinal);
                    cached.put(ordinal, vector);
                  } catch (Exception e) {
                    throw new RuntimeException(e);
                  }
                }

                @Override
                public void cacheNode(int ordinal, float[] vector) {
                  cached.put(ordinal, vector);
                }

                @Override
                public void rerank(Reranker reranker) {
                  wrapped.rerank(reranker);
                }
              };
        }
      }

      VamanaGraphSearcher.search(
          scorer,
          collector,
          getGraph(fieldEntry),
          // FIXME: support filtered
          //          vectorValues.getAcceptOrds(acceptDocs));
          acceptDocs,
          this.nodeCaches.get(field));

      if (pqVectors.containsKey(field)) {
        IoUring uring = null;
        Reranker reranker =
            switch (pqRerank) {
              case NONE -> new NoneReranker();
              case SEQUENTIAL -> {
                var exactScorer =
                    RandomVectorScorer.createFloats(
                        vectorValues, fieldEntry.similarityFunction, target);
                yield new SequentialReranker(exactScorer);
              }
              case CACHED -> new CachedReranker(fieldEntry.similarityFunction, target, cached);
              case PARALLEL_FILE_IO -> new ParallelFileIoReranker(
                  fieldEntry.similarityFunction, target, channel, fieldEntry.vectorDataOffset);
              case PARALLEL_MMAP -> new ParallelMmapReranker(
                  fieldEntry.similarityFunction, vectorValues, target);
              case IO_URING -> {
                uring = this.uring.get();
                yield new IoUringReranker(
                    fieldEntry.similarityFunction, uring, target, fieldEntry.vectorDataOffset);
              }
            };

        collector.rerank(reranker);
        // reusing thread local
        //        if (uring != null) {
        //          uring.close();
        //        }
      }
    }
  }

  @Override
  public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    FieldEntry fieldEntry = fields.get(field);

    if (fieldEntry.size() == 0
        || knnCollector.k() == 0
        || fieldEntry.vectorEncoding != VectorEncoding.BYTE) {
      return;
    }

    InGraphOffHeapByteVectorValues vectorValues =
        InGraphOffHeapByteVectorValues.load(fieldEntry, vectorIndex);
    RandomVectorScorer scorer =
        RandomVectorScorer.createBytes(vectorValues, fieldEntry.similarityFunction, target);
    VamanaGraphSearcher.search(
        scorer,
        new OrdinalTranslatedKnnCollector(knnCollector, vectorValues::ordToDoc),
        getGraph(fieldEntry),
        // FIXME: support filtered
        //        vectorValues.getAcceptOrds(acceptDocs));
        acceptDocs,
        null);
  }

  @Override
  public VamanaGraph getGraph(String field) throws IOException {
    FieldInfo info = fieldInfos.fieldInfo(field);
    if (info == null) {
      throw new IllegalArgumentException("No such field '" + field + "'");
    }
    FieldEntry entry = fields.get(field);
    if (entry != null && entry.vectorIndexLength > 0) {
      return getGraph(entry);
    } else {
      return VamanaGraph.EMPTY;
    }
  }

  private OffHeapVamanaGraph getGraph(FieldEntry entry) throws IOException {
    return new OffHeapVamanaGraph(entry, vectorIndex);
  }

  @Override
  public void close() throws IOException {
    if (uringFactory != null) {
      uringFactory.close();
    }
    IOUtils.close(vectorData, vectorIndex, quantizedVectorData);
  }

  @Override
  public QuantizedByteVectorValues getQuantizedVectorValues(String field) throws IOException {
    FieldEntry fieldEntry = fields.get(field);
    if (fieldEntry == null || fieldEntry.hasQuantizedVectors() == false) {
      return null;
    }
    assert quantizedVectorsReader != null && fieldEntry.quantizedOrdToDoc != null;
    return InGraphOffHeapQuantizedByteVectorValues.load(fieldEntry, vectorIndex);
  }

  @Override
  public ScalarQuantizer getQuantizationState(String fieldName) {
    FieldEntry field = fields.get(fieldName);
    if (field == null || field.hasQuantizedVectors() == false) {
      return null;
    }
    return field.scalarQuantizer;
  }

  public static class FieldEntry implements Accountable {

    private static final long SHALLOW_SIZE =
        RamUsageEstimator.shallowSizeOfInstance(FieldEntry.class);
    final VectorSimilarityFunction similarityFunction;
    final VectorEncoding vectorEncoding;
    final long vectorDataOffset;
    final long vectorDataLength;
    final long vectorIndexOffset;
    final long vectorIndexLength;
    final int M;
    final boolean inGraphVectors;
    final int entryNode;
    final int dimension;
    final int size;
    final DirectMonotonicReader.Meta offsetsMeta;
    final long offsetsOffset;
    final int offsetsBlockShift;
    final long offsetsLength;
    final OrdToDocDISIReaderConfiguration ordToDoc;

    final float configuredQuantile, lowerQuantile, upperQuantile;
    final long quantizedVectorDataOffset, quantizedVectorDataLength;
    final ScalarQuantizer scalarQuantizer;
    final boolean isQuantized;
    final OrdToDocDISIReaderConfiguration quantizedOrdToDoc;

    final long pqDataOffset;
    final long pqDataLength;
    final int pqFactor;
    public final ProductQuantization pq;

    FieldEntry(
        IndexInput meta, VectorEncoding vectorEncoding, VectorSimilarityFunction similarityFunction)
        throws IOException {
      this.similarityFunction = similarityFunction;
      this.vectorEncoding = vectorEncoding;
      this.isQuantized = meta.readByte() == 1;
      boolean hasGraph = meta.readVInt() == 1;
      // Has int8 quantization
      if (isQuantized) {
        configuredQuantile = Float.intBitsToFloat(meta.readInt());
        lowerQuantile = Float.intBitsToFloat(meta.readInt());
        upperQuantile = Float.intBitsToFloat(meta.readInt());
        quantizedVectorDataOffset = meta.readVLong();
        quantizedVectorDataLength = meta.readVLong();
        scalarQuantizer = new ScalarQuantizer(lowerQuantile, upperQuantile, configuredQuantile);
      } else {
        configuredQuantile = -1;
        lowerQuantile = -1;
        upperQuantile = -1;
        quantizedVectorDataOffset = -1;
        quantizedVectorDataLength = -1;
        scalarQuantizer = null;
      }
      vectorDataOffset = meta.readVLong();
      vectorDataLength = meta.readVLong();
      vectorIndexOffset = meta.readVLong();
      vectorIndexLength = meta.readVLong();
      dimension = meta.readVInt();

      pqDataOffset = meta.readLong();
      pqDataLength = meta.readLong();
      pqFactor = meta.readVInt();
      if (pqFactor > 0 && hasGraph) {
        int numCodebooks = meta.readVInt();
        int codebookSize = meta.readVInt();
        int centroidDimensions = meta.readVInt();

        Codebook[] codebooks = new Codebook[numCodebooks];
        for (int i = 0; i < numCodebooks; i++) {
          float[][] centroids = new float[codebookSize][];
          for (int j = 0; j < codebookSize; j++) {
            centroids[j] = new float[centroidDimensions];
            meta.readFloats(centroids[j], 0, centroidDimensions);
          }

          codebooks[i] = new Codebook(centroids);
        }

        int M = dimension / pqFactor;
        pq = ProductQuantization.fromCodebooks(codebooks, dimension, M, similarityFunction);
      } else {
        pq = null;
      }

      size = meta.readInt();
      if (isQuantized) {
        quantizedOrdToDoc = OrdToDocDISIReaderConfiguration.fromStoredMeta(meta, size);
      } else {
        quantizedOrdToDoc = null;
      }
      ordToDoc = OrdToDocDISIReaderConfiguration.fromStoredMeta(meta, size);

      // read node offsets
      inGraphVectors = meta.readVInt() == 1;
      M = meta.readVInt();
      if (size > 0 && hasGraph) {
        entryNode = meta.readVInt();
        offsetsOffset = meta.readLong();
        offsetsBlockShift = meta.readVInt();
        offsetsMeta = DirectMonotonicReader.loadMeta(meta, size, offsetsBlockShift);
        offsetsLength = meta.readLong();
      } else {
        if (!hasGraph) {
          meta.readVInt();
        }
        entryNode = -1;
        offsetsOffset = 0;
        offsetsBlockShift = 0;
        offsetsMeta = null;
        offsetsLength = 0;
      }
    }

    int size() {
      return size;
    }

    boolean hasQuantizedVectors() {
      return isQuantized;
    }

    @Override
    public long ramBytesUsed() {
      return SHALLOW_SIZE
          + RamUsageEstimator.sizeOf(ordToDoc)
          + (quantizedOrdToDoc == null ? 0 : RamUsageEstimator.sizeOf(quantizedOrdToDoc))
          + RamUsageEstimator.sizeOf(offsetsMeta);
    }
  }

  /** Read the nearest-neighbors graph from the index input */
  private static final class OffHeapVamanaGraph extends VamanaGraph {

    final IndexInput dataIn;
    final int entryNode;
    final int size;
    final int dimensions;
    final VectorEncoding encoding;
    int arcCount;
    int arcUpTo;
    int arc;
    NodesIterator neighborsIter;
    private final DirectMonotonicReader graphNodeOffsets;
    // Allocated to be M to track the current neighbors being explored
    private final int[] currentNeighborsBuffer;
    private final int vectorSize;
    private final boolean inGraphVectors;

    OffHeapVamanaGraph(FieldEntry entry, IndexInput vectorIndex) throws IOException {
      this.dataIn =
          vectorIndex.slice("graph-data", entry.vectorIndexOffset, entry.vectorIndexLength);
      this.entryNode = entry.entryNode;
      this.size = entry.size();
      this.dimensions = entry.dimension;
      this.encoding = entry.vectorEncoding;
      final RandomAccessInput addressesData =
          vectorIndex.randomAccessSlice(entry.offsetsOffset, entry.offsetsLength);
      this.graphNodeOffsets = DirectMonotonicReader.getInstance(entry.offsetsMeta, addressesData);
      this.currentNeighborsBuffer = new int[entry.M];
      this.vectorSize =
          entry.isQuantized
              ? this.dimensions + Float.BYTES
              : this.dimensions * this.encoding.byteSize;
      this.inGraphVectors = entry.inGraphVectors;
    }

    @Override
    public void seek(int targetOrd) throws IOException {
      assert targetOrd >= 0;
      // unsafe; no bounds checking

      // seek to the [vector | adjacency list] for this ordinal, then seek past the vector.
      var targetOffset = graphNodeOffsets.get(targetOrd);
      var vectorOffset = inGraphVectors ? this.vectorSize : 0;
      dataIn.seek(targetOffset + vectorOffset);

      arcCount = dataIn.readVInt();
      if (arcCount > 0) {
        currentNeighborsBuffer[0] = dataIn.readVInt();
        for (int i = 1; i < arcCount; i++) {
          currentNeighborsBuffer[i] = currentNeighborsBuffer[i - 1] + dataIn.readVInt();
        }
      }

      neighborsIter = new ArrayNodesIterator(currentNeighborsBuffer, arcCount);

      arc = -1;
      arcUpTo = 0;
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public int nextNeighbor() throws IOException {
      if (arcUpTo >= arcCount) {
        return NO_MORE_DOCS;
      }
      arc = currentNeighborsBuffer[arcUpTo];
      ++arcUpTo;
      return arc;
    }

    @Override
    public int entryNode() throws IOException {
      return entryNode;
    }

    @Override
    public NodesIterator getNodes() {
      return new ArrayNodesIterator(size());
    }

    @Override
    public NodesIterator getNeighbors() {
      return neighborsIter;
    }
  }

  private static class InGraphOffHeapFloatVectorValues extends FloatVectorValues
      implements RandomAccessVectorValues<float[]> {

    final IndexInput dataIn;
    private final int size;
    private final int dimensions;
    private final DirectMonotonicReader graphNodeOffsets;
    private int lastOrd = -1;
    private int doc = -1;
    private final float[] value;

    static InGraphOffHeapFloatVectorValues load(FieldEntry entry, IndexInput vectorIndex)
        throws IOException {
      IndexInput slicedInput =
          vectorIndex.slice("graph-data", entry.vectorIndexOffset, entry.vectorIndexLength);
      RandomAccessInput addressesData =
          vectorIndex.randomAccessSlice(entry.offsetsOffset, entry.offsetsLength);
      DirectMonotonicReader graphNodeOffsets =
          DirectMonotonicReader.getInstance(entry.offsetsMeta, addressesData);

      return new InGraphOffHeapFloatVectorValues(
          slicedInput, entry.size, entry.dimension, graphNodeOffsets);
    }

    InGraphOffHeapFloatVectorValues(
        IndexInput vectorIndex, int size, int dimensions, DirectMonotonicReader graphNodeOffsets) {
      this.dataIn = vectorIndex;
      this.size = size;
      this.dimensions = dimensions;
      this.graphNodeOffsets = graphNodeOffsets;
      this.value = new float[dimensions];
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public int dimension() {
      return dimensions;
    }

    @Override
    public float[] vectorValue(int targetOrd) throws IOException {
      if (lastOrd == targetOrd) {
        return value;
      }

      // unsafe; no bounds checking
      long targetOffset = graphNodeOffsets.get(targetOrd);
      dataIn.seek(targetOffset);
      dataIn.readFloats(value, 0, dimensions);
      lastOrd = targetOrd;
      return value;
    }

    @Override
    public RandomAccessVectorValues<float[]> copy() throws IOException {
      return new InGraphOffHeapFloatVectorValues(
          this.dataIn.clone(), this.size, this.dimensions, this.graphNodeOffsets);
    }

    @Override
    public float[] vectorValue() throws IOException {
      return vectorValue(doc);
    }

    @Override
    public int docID() {
      return doc;
    }

    @Override
    public int nextDoc() throws IOException {
      return advance(doc + 1);
    }

    @Override
    public int advance(int target) throws IOException {
      assert docID() < target;
      if (target >= size) {
        return doc = NO_MORE_DOCS;
      }
      return doc = target;
    }
  }

  private static class InGraphOffHeapQuantizedByteVectorValues extends QuantizedByteVectorValues
      implements RandomAccessQuantizedByteVectorValues {

    final IndexInput dataIn;
    private final int size;
    private final int dimensions;
    private final DirectMonotonicReader graphNodeOffsets;
    protected final byte[] binaryValue;
    protected final ByteBuffer byteBuffer;
    private int lastOrd = -1;
    private int doc = -1;
    protected final float[] scoreCorrectionConstant = new float[1];

    static InGraphOffHeapQuantizedByteVectorValues load(FieldEntry entry, IndexInput vectorIndex)
        throws IOException {
      IndexInput slicedInput =
          vectorIndex.slice("graph-data", entry.vectorIndexOffset, entry.vectorIndexLength);
      RandomAccessInput addressesData =
          vectorIndex.randomAccessSlice(entry.offsetsOffset, entry.offsetsLength);
      DirectMonotonicReader graphNodeOffsets =
          DirectMonotonicReader.getInstance(entry.offsetsMeta, addressesData);

      return new InGraphOffHeapQuantizedByteVectorValues(
          slicedInput, entry.size, entry.dimension, graphNodeOffsets);
    }

    InGraphOffHeapQuantizedByteVectorValues(
        IndexInput vectorIndex, int size, int dimensions, DirectMonotonicReader graphNodeOffsets) {
      this.dataIn = vectorIndex;
      this.size = size;
      this.dimensions = dimensions;
      this.graphNodeOffsets = graphNodeOffsets;
      this.byteBuffer = ByteBuffer.allocate(dimensions);
      this.binaryValue = byteBuffer.array();
    }

    @Override
    public int dimension() {
      return dimensions;
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public byte[] vectorValue(int targetOrd) throws IOException {
      if (lastOrd == targetOrd) {
        return binaryValue;
      }

      // unsafe; no bounds checking
      long targetOffset = graphNodeOffsets.get(targetOrd);
      dataIn.seek(targetOffset);
      dataIn.readBytes(byteBuffer.array(), byteBuffer.arrayOffset(), dimensions);
      dataIn.readFloats(scoreCorrectionConstant, 0, 1);
      lastOrd = targetOrd;
      return binaryValue;
    }

    @Override
    public float getScoreCorrectionConstant() {
      return scoreCorrectionConstant[0];
    }

    @Override
    public RandomAccessQuantizedByteVectorValues copy() throws IOException {
      return new InGraphOffHeapQuantizedByteVectorValues(
          this.dataIn.clone(), this.size, this.dimensions, this.graphNodeOffsets);
    }

    @Override
    public byte[] vectorValue() throws IOException {
      return vectorValue(doc);
    }

    @Override
    public int docID() {
      return doc;
    }

    @Override
    public int nextDoc() throws IOException {
      return advance(doc + 1);
    }

    @Override
    public int advance(int target) throws IOException {
      assert docID() < target;
      if (target >= size) {
        return doc = NO_MORE_DOCS;
      }
      return doc = target;
    }
  }

  private static class InGraphOffHeapByteVectorValues extends ByteVectorValues
      implements RandomAccessVectorValues<byte[]> {

    final IndexInput dataIn;
    private final int size;
    private final int dimensions;
    private final DirectMonotonicReader graphNodeOffsets;
    private int lastOrd = -1;
    private int doc = -1;
    private final byte[] value;

    static InGraphOffHeapByteVectorValues load(FieldEntry entry, IndexInput vectorIndex)
        throws IOException {
      IndexInput slicedInput =
          vectorIndex.slice("graph-data", entry.vectorIndexOffset, entry.vectorIndexLength);
      RandomAccessInput addressesData =
          vectorIndex.randomAccessSlice(entry.offsetsOffset, entry.offsetsLength);
      DirectMonotonicReader graphNodeOffsets =
          DirectMonotonicReader.getInstance(entry.offsetsMeta, addressesData);

      return new InGraphOffHeapByteVectorValues(
          slicedInput, entry.size, entry.dimension, graphNodeOffsets);
    }

    InGraphOffHeapByteVectorValues(
        IndexInput vectorIndex, int size, int dimensions, DirectMonotonicReader graphNodeOffsets) {
      this.dataIn = vectorIndex;
      this.size = size;
      this.dimensions = dimensions;
      this.graphNodeOffsets = graphNodeOffsets;
      this.value = new byte[dimensions];
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public int dimension() {
      return dimensions;
    }

    @Override
    public byte[] vectorValue(int targetOrd) throws IOException {
      if (lastOrd == targetOrd) {
        return value;
      }

      // unsafe; no bounds checking
      long targetOffset = graphNodeOffsets.get(targetOrd);
      dataIn.seek(targetOffset);
      dataIn.readBytes(value, 0, dimensions);
      lastOrd = targetOrd;
      return value;
    }

    @Override
    public RandomAccessVectorValues<byte[]> copy() throws IOException {
      return new InGraphOffHeapByteVectorValues(
          this.dataIn.clone(), this.size, this.dimensions, this.graphNodeOffsets);
    }

    @Override
    public byte[] vectorValue() throws IOException {
      return vectorValue(doc);
    }

    @Override
    public int docID() {
      return doc;
    }

    @Override
    public int nextDoc() throws IOException {
      return advance(doc + 1);
    }

    @Override
    public int advance(int target) throws IOException {
      assert docID() < target;
      if (target >= size) {
        return doc = NO_MORE_DOCS;
      }
      return doc = target;
    }
  }

  private static class NoneReranker implements Reranker {

    @Override
    public TopDocs rerank(TopDocs initial) {
      return initial;
    }
  }

  private record SequentialReranker(RandomVectorScorer exactScorer) implements Reranker {

    @Override
    public TopDocs rerank(TopDocs initial) {
      var totalHits = initial.totalHits;
      var wrappedScoreDocs = initial.scoreDocs;

      ScoreDoc[] scoreDocs = new ScoreDoc[wrappedScoreDocs.length];
      for (int i = 0; i < scoreDocs.length; i++) {
        try {
          int doc = wrappedScoreDocs[i].doc;
          float score = exactScorer.score(doc);
          scoreDocs[i] = new ScoreDoc(doc, score, wrappedScoreDocs[i].shardIndex);
        } catch (Exception e) {
          throw new RuntimeException(e);
        }
      }

      Arrays.sort(scoreDocs, Comparator.comparing(scoreDoc -> -scoreDoc.score));
      return new TopDocs(totalHits, scoreDocs);
    }
  }

  private record CachedReranker(
      VectorSimilarityFunction similarityFunction, float[] query, Map<Integer, float[]> cached)
      implements Reranker {

    @Override
    public TopDocs rerank(TopDocs initial) {
      var totalHits = initial.totalHits;
      var wrappedScoreDocs = initial.scoreDocs;

      ScoreDoc[] scoreDocs = new ScoreDoc[wrappedScoreDocs.length];
      for (int i = 0; i < scoreDocs.length; i++) {
        try {
          int doc = wrappedScoreDocs[i].doc;
          float[] vector = cached.get(doc);
          float score = similarityFunction.compare(query, vector);
          scoreDocs[i] = new ScoreDoc(doc, score, wrappedScoreDocs[i].shardIndex);
        } catch (Exception e) {
          throw new RuntimeException(e);
        }
      }

      Arrays.sort(scoreDocs, Comparator.comparing(scoreDoc -> -scoreDoc.score));
      return new TopDocs(totalHits, scoreDocs);
    }
  }

  private record ParallelMmapReranker(
      VectorSimilarityFunction similarityFunction,
      RandomAccessVectorValues<float[]> vectorValues,
      float[] query)
      implements Reranker {

    @Override
    public TopDocs rerank(TopDocs initial) {
      var totalHits = initial.totalHits;
      var wrappedScoreDocs = initial.scoreDocs;

      ScoreDoc[] scoreDocs = new ScoreDoc[wrappedScoreDocs.length];
      var futures =
          IntStream.range(0, scoreDocs.length)
              .mapToObj(
                  i -> {
                    int doc = wrappedScoreDocs[i].doc;
                    return CompletableFuture.supplyAsync(
                            () -> {
                              try {
                                return vectorValues().copy().vectorValue(doc);
                              } catch (Exception e) {
                                throw new RuntimeException(e);
                              }
                            },
                            PARALLEL_READ_EXECUTOR)
                        .thenAcceptAsync(
                            (float[] vector) -> {
                              float score = similarityFunction.compare(query, vector);
                              scoreDocs[i] =
                                  new ScoreDoc(doc, score, wrappedScoreDocs[i].shardIndex);
                            },
                            Runnable::run);
                  })
              .toList();

      CompletableFuture.allOf(futures.toArray(CompletableFuture<?>[]::new)).join();

      Arrays.sort(scoreDocs, Comparator.comparing(scoreDoc -> -scoreDoc.score));
      return new TopDocs(totalHits, scoreDocs);
    }
  }

  private record ParallelFileIoReranker(
      VectorSimilarityFunction similarityFunction,
      float[] query,
      FileChannel channel,
      long fieldVectorsOffset)
      implements Reranker {

    @Override
    public TopDocs rerank(TopDocs initial) {
      var totalHits = initial.totalHits;
      var wrappedScoreDocs = initial.scoreDocs;
      int vectorSize = query.length * Float.BYTES;

      ScoreDoc[] scoreDocs = new ScoreDoc[wrappedScoreDocs.length];
      var futures =
          IntStream.range(0, scoreDocs.length)
              .mapToObj(
                  i -> {
                    ByteBuffer buffer =
                        ByteBuffer.allocate(vectorSize).order(ByteOrder.LITTLE_ENDIAN);
                    int doc = wrappedScoreDocs[i].doc;
                    return CompletableFuture.supplyAsync(
                            () -> {
                              try {
                                buffer.clear();
                                int bytesRead =
                                    channel.read(
                                        buffer, (long) doc * vectorSize + fieldVectorsOffset);
                                if (bytesRead != vectorSize) {
                                  // Handle the case where not enough data was read
                                  throw new IOException("Not enough data read from the channel");
                                }

                                buffer.flip();
                                FloatBuffer floatBuffer = buffer.asFloatBuffer();

                                float[] vector = new float[query.length];
                                floatBuffer.get(vector);
                                return vector;
                              } catch (Exception e) {
                                throw new RuntimeException(e);
                              }
                            },
                            PARALLEL_READ_EXECUTOR)
                        .thenAcceptAsync(
                            (float[] vector) -> {
                              float score = similarityFunction.compare(query, vector);
                              scoreDocs[i] =
                                  new ScoreDoc(doc, score, wrappedScoreDocs[i].shardIndex);
                            },
                            Runnable::run);
                  })
              .toList();

      CompletableFuture.allOf(futures.toArray(CompletableFuture<?>[]::new)).join();

      Arrays.sort(scoreDocs, Comparator.comparing(scoreDoc -> -scoreDoc.score));
      return new TopDocs(totalHits, scoreDocs);
    }
  }

  private record IoUringReranker(
      VectorSimilarityFunction similarityFunction,
      IoUring ring,
      float[] query,
      long fieldVectorsOffset)
      implements Reranker {

    @Override
    public TopDocs rerank(TopDocs initial) {
      var totalHits = initial.totalHits;
      var wrappedScoreDocs = initial.scoreDocs;
      int vectorSize = query.length * Float.BYTES;

      try (Arena arena = Arena.ofConfined()) {
        ScoreDoc[] scoreDocs = new ScoreDoc[wrappedScoreDocs.length];
        MemorySegment querySegment = arena.allocate(vectorSize, 64);
        for (int i = 0; i < query.length; i++) {
          querySegment.setAtIndex(ValueLayout.JAVA_FLOAT, i, query[i]);
        }

        var futures =
            IntStream.range(0, scoreDocs.length)
                .mapToObj(
                    i -> {
                      int doc = wrappedScoreDocs[i].doc;
                      // align to 64 bytes for ideal AVX512 perf
                      MemorySegment buffer = arena.allocate(vectorSize, 64);

                      CompletableFuture<Void> future =
                          ring.prepare(
                              buffer, vectorSize, (long) doc * vectorSize + fieldVectorsOffset);

                      future.thenRun(
                          () -> {
                            float score =
                                similarityFunction.compare(querySegment, buffer, query.length);
                            scoreDocs[i] = new ScoreDoc(doc, score, wrappedScoreDocs[i].shardIndex);
                          });

                      return future;
                    })
                .toList();

        ring.submit();
        ring.awaitAll();
        CompletableFuture.allOf(futures.toArray(CompletableFuture<?>[]::new)).join();

        Arrays.sort(scoreDocs, Comparator.comparing(scoreDoc -> -scoreDoc.score));
        return new TopDocs(totalHits, scoreDocs);
      }
    }
  }
}
