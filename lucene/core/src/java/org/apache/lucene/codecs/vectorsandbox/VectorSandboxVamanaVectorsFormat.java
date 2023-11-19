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
import java.util.concurrent.ExecutorService;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.lucene90.IndexedDISI;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.vamana.VamanaGraph;

/**
 * Lucene 9.9 vector format, which encodes numeric vector values and an optional associated graph
 * connecting the documents having values. The graph is used to power Vamana search. The format
 * consists of three files, with an optional fourth file:
 *
 * <h2>.vec (vector data) file</h2>
 *
 * <p>For each field:
 *
 * <ul>
 *   <li>Vector data ordered by field, document ordinal, and vector dimension. When the
 *       vectorEncoding is BYTE, each sample is stored as a single byte. When it is FLOAT32, each
 *       sample is stored as an IEEE float in little-endian byte order.
 *   <li>DocIds encoded by {@link IndexedDISI#writeBitSet(DocIdSetIterator, IndexOutput, byte)},
 *       note that only in sparse case
 *   <li>OrdToDoc was encoded by {@link org.apache.lucene.util.packed.DirectMonotonicWriter}, note
 *       that only in sparse case
 * </ul>
 *
 * <h2>.vex (vector index)</h2>
 *
 * <p>Stores graphs connecting the documents for each field organized as a list of nodes' neighbours
 * as following:
 *
 * <ul>
 *   <li>For each level:
 *       <ul>
 *         <li>For each node:
 *             <ul>
 *               <li><b>[vint]</b> the number of neighbor nodes
 *               <li><b>array[vint]</b> the delta encoded neighbor ordinals
 *             </ul>
 *       </ul>
 *   <li>After all levels are encoded memory offsets for each node's neighbor nodes encoded by
 *       {@link org.apache.lucene.util.packed.DirectMonotonicWriter} are appened to the end of the
 *       file.
 * </ul>
 *
 * <h2>.vem (vector metadata) file</h2>
 *
 * <p>For each field:
 *
 * <ul>
 *   <li><b>[int32]</b> field number
 *   <li><b>[int32]</b> vector similarity function ordinal
 *   <li><b>[byte]</b> if equals to 1 indicates if the field is for quantized vectors
 *   <li><b>[int32]</b> if quantized: the configured quantile float int bits.
 *   <li><b>[int32]</b> if quantized: the calculated lower quantile float int32 bits.
 *   <li><b>[int32]</b> if quantized: the calculated upper quantile float int32 bits.
 *   <li><b>[vlong]</b> if quantized: offset to this field's vectors in the .veq file
 *   <li><b>[vlong]</b> if quantized: length of this field's vectors, in bytes in the .veq file
 *   <li><b>[vlong]</b> offset to this field's vectors in the .vec file
 *   <li><b>[vlong]</b> length of this field's vectors, in bytes
 *   <li><b>[vlong]</b> offset to this field's index in the .vex file
 *   <li><b>[vlong]</b> length of this field's index data, in bytes
 *   <li><b>[vint]</b> dimension of this field's vectors
 *   <li><b>[int]</b> the number of documents having values for this field
 *   <li><b>[int8]</b> if equals to -1, dense – all documents have values for a field. If equals to
 *       0, sparse – some documents missing values.
 *   <li>DocIds were encoded by {@link IndexedDISI#writeBitSet(DocIdSetIterator, IndexOutput, byte)}
 *   <li>OrdToDoc was encoded by {@link org.apache.lucene.util.packed.DirectMonotonicWriter}, note
 *       that only in sparse case
 *   <li><b>[vint]</b> the maximum number of connections (neigbours) that each node can have
 *   <li><b>[vint]</b> number of levels in the graph
 *   <li>Graph nodes by level. For each level
 *       <ul>
 *         <li><b>[vint]</b> the number of nodes on this level
 *         <li><b>array[vint]</b> for levels greater than 0 list of nodes on this level, stored as
 *             the level 0th delta encoded nodes' ordinals.
 *       </ul>
 * </ul>
 *
 * <h2>.veq (quantized vector data) file</h2>
 *
 * <p>For each field:
 *
 * <ul>
 *   <li>Vector data ordered by field, document ordinal, and vector dimension. Each vector dimension
 *       is stored as a single byte and every vector has a single float32 value for scoring
 *       corrections.
 *   <li>DocIds encoded by {@link IndexedDISI#writeBitSet(DocIdSetIterator, IndexOutput, byte)},
 *       note that only in sparse case
 *   <li>OrdToDoc was encoded by {@link org.apache.lucene.util.packed.DirectMonotonicWriter}, note
 *       that only in sparse case
 * </ul>
 *
 * @lucene.experimental
 */
public final class VectorSandboxVamanaVectorsFormat extends KnnVectorsFormat {

  static final String META_CODEC_NAME = "VectorSandboxVamanaFormatMeta";
  static final String VECTOR_DATA_CODEC_NAME = "VectorSandboxVamanaVectorsFormatData";
  static final String VECTOR_INDEX_CODEC_NAME = "VectorSandboxVamanaVectorsFormatIndex";
  static final String PQ_DATA_CODEC_NAME = "VectorSandboxVamanaPQData";
  static final String META_EXTENSION = "vem";
  static final String VECTOR_DATA_EXTENSION = "vec";
  static final String VECTOR_INDEX_EXTENSION = "vex";
  static final String PQ_DATA_EXTENSION = "vpq";

  public enum PQRerank {
    NONE,
    CACHED,
    SEQUENTIAL,
    PARALLEL,
  }

  public static final int VERSION_START = 0;
  public static final int VERSION_CURRENT = VERSION_START;

  /**
   * A maximum configurable maximum max conn.
   *
   * <p>NOTE: We eagerly populate `float[MAX_CONN*2]` and `int[MAX_CONN*2]`, so exceptionally large
   * numbers here will use an inordinate amount of heap
   */
  private static final int MAXIMUM_MAX_CONN = 512;

  /**
   * Default number of maximum connections per node
   */
  public static final int DEFAULT_MAX_CONN = 16;

  /**
   * The maximum size of the queue to maintain while searching during graph construction This
   * maximum value preserves the ratio of the DEFAULT_BEAM_WIDTH/DEFAULT_MAX_CONN i.e. `6.25 * 16 =
   * 3200`
   */
  private static final int MAXIMUM_BEAM_WIDTH = 3200;

  /**
   * Default number of the size of the queue maintained while searching during a graph
   * construction.
   */
  public static final int DEFAULT_BEAM_WIDTH = 100;

  public static final float DEFAULT_ALPHA = 1.2f;

  public static final int DEFAULT_PQ_FACTOR = 0;
  public static final boolean DEFAULT_IN_GRAPH_VECTORS = true;
  public static final PQRerank DEFAULT_PQ_RERANK = PQRerank.CACHED;

  /**
   * Default to use single thread merge
   */
  public static final int DEFAULT_NUM_MERGE_WORKER = 1;

  static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;

  /**
   * Controls how many of the nearest neighbor candidates are connected to the new node. Defaults to
   * {@link VectorSandboxVamanaVectorsFormat#DEFAULT_MAX_CONN}. See {@link VamanaGraph} for more
   * details.
   */
  private final int maxConn;

  /**
   * The number of candidate neighbors to track while searching the graph for each newly inserted
   * node. Defaults to to {@link VectorSandboxVamanaVectorsFormat#DEFAULT_BEAM_WIDTH}. See
   * {@link VamanaGraph} for details.
   */
  private final int beamWidth;

  private final float alpha;
  private final int pqFactor;
  private final boolean inGraphVectors;
  private final PQRerank pqRerank;

  /**
   * Should this codec scalar quantize float32 vectors and use this format
   */
  private final VectorSandboxScalarQuantizedVectorsFormat scalarQuantizedVectorsFormat;

  private final int numMergeWorkers;
  private final ExecutorService mergeExec;

  /**
   * Constructs a format using default graph construction parameters
   */
  public VectorSandboxVamanaVectorsFormat() {
    this(
        DEFAULT_MAX_CONN,
        DEFAULT_BEAM_WIDTH,
        DEFAULT_ALPHA,
        DEFAULT_PQ_FACTOR,
        DEFAULT_IN_GRAPH_VECTORS,
        null);
  }

  public VectorSandboxVamanaVectorsFormat(
      int maxConn,
      int beamWidth,
      float alpha,
      int pqFactor,
      boolean inGraphVectors,
      VectorSandboxScalarQuantizedVectorsFormat scalarQuantize) {
    this(
        maxConn,
        beamWidth,
        alpha,
        pqFactor,
        inGraphVectors,
        scalarQuantize,
        DEFAULT_NUM_MERGE_WORKER,
        null);
  }

  /**
   * Constructs a format using the given graph construction parameters.
   *
   * @param maxConn   the maximum number of connections to a node in the Vamana graph
   * @param beamWidth the size of the queue maintained during graph construction.
   */
  public VectorSandboxVamanaVectorsFormat(
      int maxConn,
      int beamWidth,
      float alpha,
      int pqFactor,
      boolean inGraphVectors) {
    this(maxConn, beamWidth, alpha, pqFactor, inGraphVectors, null);
  }

  /**
   * Constructs a format using the given graph construction parameters and scalar quantization.
   *
   * @param maxConn         the maximum number of connections to a node in the Vamana graph
   * @param beamWidth       the size of the queue maintained during graph construction.
   * @param scalarQuantize  the scalar quantization format
   * @param numMergeWorkers number of workers (threads) that will be used when doing merge. If
   *                        larger than 1, a non-null {@link ExecutorService} must be passed as
   *                        mergeExec
   * @param mergeExec       the {@link ExecutorService} that will be used by ALL vector writers that
   *                        are generated by this format to do the merge
   */
  public VectorSandboxVamanaVectorsFormat(
      int maxConn,
      int beamWidth,
      float alpha,
      int pqFactor,
      boolean inGraphVectors,
      VectorSandboxScalarQuantizedVectorsFormat scalarQuantize,
      int numMergeWorkers,
      ExecutorService mergeExec) {
    super("VectorSandboxVamanaVectorsFormat");
    if (maxConn <= 0 || maxConn > MAXIMUM_MAX_CONN) {
      throw new IllegalArgumentException(
          "maxConn must be positive and less than or equal to"
              + MAXIMUM_MAX_CONN
              + "; maxConn="
              + maxConn);
    }
    if (beamWidth <= 0 || beamWidth > MAXIMUM_BEAM_WIDTH) {
      throw new IllegalArgumentException(
          "beamWidth must be positive and less than or equal to"
              + MAXIMUM_BEAM_WIDTH
              + "; beamWidth="
              + beamWidth);
    }
    if (pqFactor < 0) {
      throw new IllegalArgumentException(
          "pqFactor must be greater or equal than 0; pqFactor=" + pqFactor);
    }
    if (numMergeWorkers > 1 && mergeExec == null) {
      throw new IllegalArgumentException(
          "No executor service passed in when " + numMergeWorkers + " merge workers are requested");
    }
    if (numMergeWorkers == 1 && mergeExec != null) {
      throw new IllegalArgumentException(
          "No executor service is needed as we'll use single thread to merge");
    }
    this.maxConn = maxConn;
    this.beamWidth = beamWidth;
    this.alpha = alpha;
    this.pqFactor = pqFactor;
    this.inGraphVectors = inGraphVectors;
    this.pqRerank = System.getenv("VAMANA_PQ_RERANK") == null ? DEFAULT_PQ_RERANK
        : PQRerank.valueOf(System.getenv("VAMANA_PQ_RERANK").toUpperCase());
    this.scalarQuantizedVectorsFormat = scalarQuantize;
    this.numMergeWorkers = numMergeWorkers;
    this.mergeExec = mergeExec;
  }

  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new VectorSandboxVamanaVectorsWriter(
        state,
        maxConn,
        beamWidth,
        alpha,
        pqFactor,
        inGraphVectors,
        scalarQuantizedVectorsFormat,
        numMergeWorkers,
        mergeExec);
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new VectorSandboxVamanaVectorsReader(state, pqRerank);
  }

  @Override
  public int getMaxDimensions(String fieldName) {
    return 1024;
  }

  @Override
  public String toString() {
    return "VectorSandboxVamanaVectorsFormat(name=VectorSandboxVamanaVectorsFormat, maxConn="
        + maxConn
        + ", beamWidth="
        + beamWidth
        + ", quantizer="
        + (scalarQuantizedVectorsFormat == null ? "none" : scalarQuantizedVectorsFormat.toString())
        + ")";
  }
}
