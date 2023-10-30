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

package org.apache.lucene.util.vamana;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.InfoStream;

/**
 * Builder for Vamana graph. See {@link VamanaGraph} for a gloss on the algorithm and the meaning of
 * the hyper-parameters.
 */
public class VamanaGraphBuilder {

  /** Default number of maximum connections per node */
  public static final int DEFAULT_MAX_CONN = 16;

  /**
   * Default number of the size of the queue maintained while searching during a graph construction.
   */
  public static final int DEFAULT_BEAM_WIDTH = 100;

  public static final float DEFAULT_ALPHA = 1.2f;

  /** Default random seed for level generation * */
  private static final long DEFAULT_RAND_SEED = 42;

  /** A name for the HNSW component for the info-stream * */
  public static final String VAMANA_COMPONENT = "VAMANA";

  /** Random seed for level generation; public to expose for testing * */
  public static long randSeed = DEFAULT_RAND_SEED;

  private final int M; // max number of connections on upper layers
  private final float alpha;
  private final NeighborArray scratch;

  private final RandomVectorScorerSupplier scorerSupplier;
  private final VamanaGraphSearcher graphSearcher;
  private final GraphBuilderKnnCollector beamCandidates;

  protected final OnHeapVamanaGraph vamana;

  private InfoStream infoStream = InfoStream.getDefault();

  public static VamanaGraphBuilder create(
      RandomVectorScorerSupplier scorerSupplier, int M, int beamWidth, float alpha, long seed)
      throws IOException {
    return new VamanaGraphBuilder(scorerSupplier, M, beamWidth, alpha, seed, -1);
  }

  public static VamanaGraphBuilder create(
      RandomVectorScorerSupplier scorerSupplier,
      int M,
      int beamWidth,
      float alpha,
      long seed,
      int graphSize)
      throws IOException {
    return new VamanaGraphBuilder(scorerSupplier, M, beamWidth, alpha, seed, graphSize);
  }

  /**
   * Reads all the vectors from vector values, builds a graph connecting them by their dense
   * ordinals, using the given hyperparameter settings, and returns the resulting graph.
   *
   * @param scorerSupplier a supplier to create vector scorer from ordinals.
   * @param M – graph fanout parameter used to calculate the maximum number of connections a node
   *     can have – M on upper layers, and M * 2 on the lowest level.
   * @param beamWidth the size of the beam search to use when finding nearest neighbors.
   * @param seed the seed for a random number generator used during graph construction. Provide this
   *     to ensure repeatable construction.
   * @param graphSize size of graph, if unknown, pass in -1
   */
  protected VamanaGraphBuilder(
      RandomVectorScorerSupplier scorerSupplier,
      int M,
      int beamWidth,
      float alpha,
      long seed,
      int graphSize)
      throws IOException {
    this(scorerSupplier, M, beamWidth, alpha, seed, new OnHeapVamanaGraph(M, graphSize));
  }

  /**
   * Reads all the vectors from vector values, builds a graph connecting them by their dense
   * ordinals, using the given hyperparameter settings, and returns the resulting graph.
   *
   * @param scorerSupplier a supplier to create vector scorer from ordinals.
   * @param M – graph fanout parameter used to calculate the maximum number of connections a node
   *     can have – M on upper layers, and M * 2 on the lowest level.
   * @param beamWidth the size of the beam search to use when finding nearest neighbors.
   * @param seed the seed for a random number generator used during graph construction. Provide this
   *     to ensure repeatable construction.
   * @param vamana the graph to build, can be previously initialized
   */
  protected VamanaGraphBuilder(
      RandomVectorScorerSupplier scorerSupplier,
      int M,
      int beamWidth,
      float alpha,
      long seed,
      OnHeapVamanaGraph vamana)
      throws IOException {
    if (M <= 0) {
      throw new IllegalArgumentException("maxConn must be positive");
    }
    if (beamWidth <= 0) {
      throw new IllegalArgumentException("beamWidth must be positive");
    }
    this.M = M;
    this.scorerSupplier =
        Objects.requireNonNull(scorerSupplier, "scorer supplier must not be null");
    // normalization factor for level generation; currently not configurable
    this.alpha = alpha;
    this.vamana = vamana;
    this.graphSearcher =
        new VamanaGraphSearcher(
            new NeighborQueue(beamWidth, true), new FixedBitSet(this.getGraph().size()));
    // in scratch we store candidates in reverse order: worse candidates are first
    scratch = new NeighborArray(Math.max(beamWidth, M + 1), false);
    beamCandidates = new GraphBuilderKnnCollector(beamWidth);
  }

  /**
   * Adds all nodes to the graph up to the provided {@code maxOrd}.
   *
   * @param maxOrd The maximum ordinal of the nodes to be added.
   */
  public OnHeapVamanaGraph build(int maxOrd) throws IOException {
    if (infoStream.isEnabled(VAMANA_COMPONENT)) {
      infoStream.message(VAMANA_COMPONENT, "build graph from " + maxOrd + " vectors");
    }
    addVectors(maxOrd);
    return vamana;
  }

  /** Set info-stream to output debugging information * */
  public void setInfoStream(InfoStream infoStream) {
    this.infoStream = infoStream;
  }

  public OnHeapVamanaGraph getGraph() {
    return vamana;
  }

  private void addVectors(int maxOrd) throws IOException {
    long start = System.nanoTime(), t = start;
    for (int node = 0; node < maxOrd; node++) {
      addGraphNode(node);
      if ((node % 10000 == 0) && infoStream.isEnabled(VAMANA_COMPONENT)) {
        t = printGraphBuildStatus(node, start, t);
      }
    }
  }

  /** Inserts a doc with vector value to the graph */
  public void addGraphNode(int node) throws IOException {
    RandomVectorScorer scorer = scorerSupplier.scorer(node);

    // If entrynode is -1, then this should finish without adding neighbors
    if (vamana.entryNode() == -1) {
      vamana.addNode(node);
      return;
    }
    int[] eps = new int[] {vamana.entryNode()};

    GraphBuilderKnnCollector candidates = beamCandidates;
    candidates.clear();
    graphSearcher.search(candidates, scorer, eps, vamana, null);
    vamana.addNode(node);
    addDiverseNeighbors(node, candidates);
  }

  public void finish(List<float[]> vectors, VectorSimilarityFunction similarityFunction)
      throws IOException {
    cleanup();
    setEntryPointToMedioid(vectors, similarityFunction);
  }

  private void cleanup() throws IOException {
    var graph = getGraph();
    VamanaGraph.NodesIterator it = graph.getNodes();
    while (it.hasNext()) {
      int node = it.nextInt();
      var neighbors = graph.getNeighbors(node);
      if (neighbors.size() <= M) {
        continue;
      }

      var selected = selectDiverse(neighbors, M);
      neighbors.clear();
      for (var candidate : selected) {
        neighbors.addInOrder(candidate.node, candidate.score);
      }
    }
  }

  private void setEntryPointToMedioid(
      List<float[]> vectors, VectorSimilarityFunction similarityFunction) {
    var ep = calculateEntryPoint(vectors, similarityFunction);
    getGraph().setEntryNode(ep);
  }

  private int calculateEntryPoint(
      List<float[]> vectors, VectorSimilarityFunction similarityFunction) {
    int dimensions = vectors.get(0).length;
    int numVectors = vectors.size();
    float[] centroid = new float[dimensions];

    // Initialize centroid
    for (var vector : vectors) {
      for (int j = 0; j < dimensions; j++) {
        centroid[j] += vector[j];
      }
    }

    for (int i = 0; i < dimensions; i++) {
      centroid[i] /= numVectors;
    }

    var maxIdx = 0;
    var maxScore = 0.0f;
    for (int i = 0; i < vectors.size(); i++) {
      var score = similarityFunction.compare(centroid, vectors.get(i));
      if (Float.compare(maxScore, score) >= 0) {
        continue;
      }
      maxIdx = i;
      maxScore = score;
    }

    return maxIdx;
  }

  private long printGraphBuildStatus(int node, long start, long t) {
    long now = System.nanoTime();
    infoStream.message(
        VAMANA_COMPONENT,
        String.format(
            Locale.ROOT,
            "built %d in %d/%d ms",
            node,
            TimeUnit.NANOSECONDS.toMillis(now - t),
            TimeUnit.NANOSECONDS.toMillis(now - start)));
    return now;
  }

  private void addDiverseNeighbors(int node, GraphBuilderKnnCollector candidates)
      throws IOException {
    /* For each of the beamWidth nearest candidates (going from best to worst), select it only if it
     * is closer to target than it is to any of the already-selected neighbors (ie selected in this method,
     * since the node is new and has no prior neighbors).
     */
    NeighborArray neighbors = vamana.getNeighbors(node);
    assert neighbors.size() == 0; // new node
    popToScratch(candidates);
    // FIXME: why M * 2?
    int maxConn = M * 2;
    var selected = selectDiverse(scratch, maxConn);
    for (var candidate : selected) {
      neighbors.addInOrder(candidate.node, candidate.score);
    }

    // Link the selected nodes to the new node, and the new node to the selected nodes (again
    // applying diversity heuristic)
    int size = neighbors.size();
    for (int i = 0; i < size; i++) {
      int nbr = neighbors.node[i];
      NeighborArray nbrsOfNbr = vamana.getNeighbors(nbr);
      nbrsOfNbr.addOutOfOrder(node, neighbors.score[i]);
      if (nbrsOfNbr.size() > maxConn) {
        int indexToRemove = findWorstNonDiverse(nbrsOfNbr, nbr);
        nbrsOfNbr.removeIndex(indexToRemove);
      }
    }
  }

  // FIXME: write second version that uses occlude_factor like DiskANN, or prove this is equivalent
  private List<Candidate> selectDiverse(NeighborArray candidates, int maxConn) throws IOException {
    var selected = new FixedBitSet(candidates.size());
    List<Candidate> selectedCandidates = new ArrayList<>(maxConn);

    for (float a = 1.0f; a < alpha + 1E-6 && selectedCandidates.size() < maxConn; a += 0.2f) {
      for (int i = candidates.size() - 1; selectedCandidates.size() < maxConn && i >= 0; i--) {
        if (selected.get(i)) {
          continue;
        }

        // compare each neighbor (in distance order) against the closer neighbors selected so far,
        // only adding it if it is closer to the target than to any of the other selected neighbors
        int cNode = candidates.node[i];
        float cScore = candidates.score[i];
        assert cNode <= vamana.maxNodeId();
        if (diversityCheck(cNode, cScore, candidates, selected, a)) {
          selected.set(i);
          selectedCandidates.add(new Candidate(cNode, cScore));
        }
      }
    }

    selectedCandidates.sort(Comparator.reverseOrder());
    return selectedCandidates;
  }

  record Candidate(int node, float score) implements Comparable<Candidate> {

    @Override
    public int compareTo(Candidate o) {
      var score = Float.compare(this.score, o.score);
      if (score != 0) {
        return score;
      }

      return Integer.compare(this.node, o.node);
    }
  }

  private void popToScratch(GraphBuilderKnnCollector candidates) {
    scratch.clear();
    int candidateCount = candidates.size();
    // extract all the Neighbors from the queue into an array; these will now be
    // sorted from worst to best
    for (int i = 0; i < candidateCount; i++) {
      float maxSimilarity = candidates.minimumScore();
      scratch.addInOrder(candidates.popNode(), maxSimilarity);
    }
  }

  /**
   * @param candidate the vector of a new candidate neighbor of a node n
   * @param score the score of the new candidate and node n, to be compared with scores of the
   *     candidate and n's neighbors
   * @param selected the neighbors selected so far
   * @return whether the candidate is diverse given the existing neighbors
   */
  private boolean diversityCheck(
      int candidate, float score, NeighborArray candidates, BitSet selected, float a)
      throws IOException {
    RandomVectorScorer scorer = scorerSupplier.scorer(candidate);
    for (int i = selected.nextSetBit(0);
        i != DocIdSetIterator.NO_MORE_DOCS;
        i = selected.nextSetBit(i + 1)) {
      int other = candidates.node[i];
      if (other == candidate) {
        // FIXME: explain this break
        return false;
      }

      float neighborSimilarity = scorer.score(candidates.node[i]);
      if (neighborSimilarity > score * a) {
        return false;
      }

      // nextSetBit will assert if you're past the end, so check ourselves
      if (i + 1 >= selected.length()) {
        break;
      }
    }

    return true;
  }

  /**
   * Find first non-diverse neighbour among the list of neighbors starting from the most distant
   * neighbours
   */
  private int findWorstNonDiverse(NeighborArray neighbors, int nodeOrd) throws IOException {
    RandomVectorScorer scorer = scorerSupplier.scorer(nodeOrd);
    int[] uncheckedIndexes = neighbors.sort(scorer);
    if (uncheckedIndexes == null) {
      // all nodes are checked, we will directly return the most distant one
      return neighbors.size() - 1;
    }
    int uncheckedCursor = uncheckedIndexes.length - 1;
    for (int i = neighbors.size() - 1; i > 0; i--) {
      if (uncheckedCursor < 0) {
        // no unchecked node left
        break;
      }
      if (isWorstNonDiverse(i, neighbors, uncheckedIndexes, uncheckedCursor)) {
        return i;
      }
      if (i == uncheckedIndexes[uncheckedCursor]) {
        uncheckedCursor--;
      }
    }
    return neighbors.size() - 1;
  }

  private boolean isWorstNonDiverse(
      int candidateIndex, NeighborArray neighbors, int[] uncheckedIndexes, int uncheckedCursor)
      throws IOException {
    float minAcceptedSimilarity = neighbors.score[candidateIndex];
    RandomVectorScorer scorer = scorerSupplier.scorer(neighbors.node[candidateIndex]);
    if (candidateIndex == uncheckedIndexes[uncheckedCursor]) {
      // the candidate itself is unchecked
      for (int i = candidateIndex - 1; i >= 0; i--) {
        float neighborSimilarity = scorer.score(neighbors.node[i]);
        // candidate node is too similar to node i given its score relative to the base node
        if (neighborSimilarity >= minAcceptedSimilarity) {
          return true;
        }
      }
    } else {
      // else we just need to make sure candidate does not violate diversity with the (newly
      // inserted) unchecked nodes
      assert candidateIndex > uncheckedIndexes[uncheckedCursor];
      for (int i = uncheckedCursor; i >= 0; i--) {
        float neighborSimilarity = scorer.score(neighbors.node[uncheckedIndexes[i]]);
        // candidate node is too similar to node i given its score relative to the base node
        if (neighborSimilarity >= minAcceptedSimilarity) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * A restricted, specialized knnCollector that can be used when building a graph.
   *
   * <p>Does not support TopDocs
   */
  public static final class GraphBuilderKnnCollector implements KnnCollector {

    private final NeighborQueue queue;
    private final int k;
    private long visitedCount;

    /**
     * @param k the number of neighbors to collect
     */
    public GraphBuilderKnnCollector(int k) {
      this.queue = new NeighborQueue(k, false);
      this.k = k;
    }

    public int size() {
      return queue.size();
    }

    public int popNode() {
      return queue.pop();
    }

    public int[] popUntilNearestKNodes() {
      while (size() > k()) {
        queue.pop();
      }
      return queue.nodes();
    }

    float minimumScore() {
      return queue.topScore();
    }

    public void clear() {
      this.queue.clear();
      this.visitedCount = 0;
    }

    @Override
    public boolean earlyTerminated() {
      return false;
    }

    @Override
    public void incVisitedCount(int count) {
      this.visitedCount += count;
    }

    @Override
    public long visitedCount() {
      return visitedCount;
    }

    @Override
    public long visitLimit() {
      return Long.MAX_VALUE;
    }

    @Override
    public int k() {
      return k;
    }

    @Override
    public boolean collect(int docId, float similarity) {
      return queue.insertWithOverflow(docId, similarity);
    }

    @Override
    public float minCompetitiveSimilarity() {
      return queue.size() >= k() ? queue.topScore() : Float.NEGATIVE_INFINITY;
    }

    @Override
    public TopDocs topDocs() {
      throw new IllegalArgumentException();
    }
  }
}
