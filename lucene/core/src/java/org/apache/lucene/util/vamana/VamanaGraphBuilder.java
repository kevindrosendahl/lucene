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
 * Builder for HNSW graph. See {@link VamanaGraph} for a gloss on the algorithm and the meaning of
 * the hyper-parameters.
 */
public class VamanaGraphBuilder implements VamanaBuilder {

  /** Default number of maximum connections per node */
  public static final int DEFAULT_MAX_CONN = 16;

  /**
   * Default number of the size of the queue maintained while searching during a graph construction.
   */
  public static final int DEFAULT_BEAM_WIDTH = 100;

  public static final float DEFAULT_ALPHA = 1.2f;

  /** A name for the HNSW component for the info-stream * */
  public static final String VAMANA_COMPONENT = "VAMANA";

  private final int M;
  private final float alpha;

  private final RandomVectorScorerSupplier scorerSupplier;
  private final VectorSimilarityFunction similarityFunction;
  private final VamanaGraphSearcher graphSearcher;
  private final GraphBuilderKnnCollector
      beamCandidates; // for levels of graph where we add the node

  protected final OnHeapVamanaGraph vamana;

  private InfoStream infoStream = InfoStream.getDefault();

  public static VamanaGraphBuilder create(
      RandomVectorScorerSupplier scorerSupplier,
      VectorSimilarityFunction similarityFunction,
      int M,
      int beamWidth,
      float alpha)
      throws IOException {
    return new VamanaGraphBuilder(scorerSupplier, similarityFunction, M, beamWidth, alpha, -1);
  }

  public static VamanaGraphBuilder create(
      RandomVectorScorerSupplier scorerSupplier,
      VectorSimilarityFunction similarityFunction,
      int M,
      int beamWidth,
      float alpha,
      int graphSize)
      throws IOException {
    return new VamanaGraphBuilder(
        scorerSupplier, similarityFunction, M, beamWidth, alpha, graphSize);
  }

  /**
   * Reads all the vectors from vector values, builds a graph connecting them by their dense
   * ordinals, using the given hyperparameter settings, and returns the resulting graph.
   *
   * @param scorerSupplier a supplier to create vector scorer from ordinals.
   * @param M – graph fanout parameter used to calculate the maximum number of connections a node
   *     can have – M on upper layers, and M * 2 on the lowest level.
   * @param beamWidth the size of the beam search to use when finding nearest neighbors.
   * @param graphSize size of graph, if unknown, pass in -1
   */
  protected VamanaGraphBuilder(
      RandomVectorScorerSupplier scorerSupplier,
      VectorSimilarityFunction similarityFunction,
      int M,
      int beamWidth,
      float alpha,
      int graphSize)
      throws IOException {
    this(
        scorerSupplier,
        similarityFunction,
        M,
        beamWidth,
        alpha,
        new OnHeapVamanaGraph(M, graphSize));
  }

  protected VamanaGraphBuilder(
      RandomVectorScorerSupplier scorerSupplier,
      VectorSimilarityFunction similarityFunction,
      int M,
      int beamWidth,
      float alpha,
      OnHeapVamanaGraph vamana)
      throws IOException {
    this(
        scorerSupplier,
        similarityFunction,
        M,
        beamWidth,
        alpha,
        vamana,
        new VamanaGraphSearcher(
            new NeighborQueue(beamWidth, true), new FixedBitSet(vamana.size())));
  }

  /**
   * Reads all the vectors from vector values, builds a graph connecting them by their dense
   * ordinals, using the given hyperparameter settings, and returns the resulting graph.
   *
   * @param scorerSupplier a supplier to create vector scorer from ordinals.
   * @param M – graph fanout parameter used to calculate the maximum number of connections a node
   *     can have – M on upper layers, and M * 2 on the lowest level.
   * @param beamWidth the size of the beam search to use when finding nearest neighbors.
   * @param vamana the graph to build, can be previously initialized
   */
  protected VamanaGraphBuilder(
      RandomVectorScorerSupplier scorerSupplier,
      VectorSimilarityFunction similarityFunction,
      int M,
      int beamWidth,
      float alpha,
      OnHeapVamanaGraph vamana,
      VamanaGraphSearcher graphSearcher)
      throws IOException {
    if (M <= 0) {
      throw new IllegalArgumentException("maxConn must be positive");
    }
    if (beamWidth <= 0) {
      throw new IllegalArgumentException("beamWidth must be positive");
    }
    this.M = M;
    this.alpha = alpha;
    this.scorerSupplier =
        Objects.requireNonNull(scorerSupplier, "scorer supplier must not be null");
    this.similarityFunction = similarityFunction;
    this.vamana = vamana;
    this.graphSearcher = graphSearcher;
    beamCandidates = new GraphBuilderKnnCollector(beamWidth);
  }

  @Override
  public OnHeapVamanaGraph build(int maxOrd) throws IOException {
    if (infoStream.isEnabled(VAMANA_COMPONENT)) {
      infoStream.message(VAMANA_COMPONENT, "build graph from " + maxOrd + " vectors");
    }
    addVectors(maxOrd);
    return vamana;
  }

  @Override
  public void setInfoStream(InfoStream infoStream) {
    this.infoStream = infoStream;
  }

  @Override
  public OnHeapVamanaGraph getGraph() {
    return vamana;
  }

  /** add vectors in range [minOrd, maxOrd) */
  protected void addVectors(int minOrd, int maxOrd) throws IOException {
    long start = System.nanoTime(), t = start;
    if (infoStream.isEnabled(VAMANA_COMPONENT)) {
      infoStream.message(VAMANA_COMPONENT, "addVectors [" + minOrd + " " + maxOrd + ")");
    }
    for (int node = minOrd; node < maxOrd; node++) {
      addGraphNode(node);
      if ((node % 10000 == 0) && infoStream.isEnabled(VAMANA_COMPONENT)) {
        t = printGraphBuildStatus(node, start, t);
      }
    }
  }

  private void addVectors(int maxOrd) throws IOException {
    addVectors(0, maxOrd);
  }

  @Override
  public void addGraphNode(int node) throws IOException {
    /*
    Note: this implementation is thread safe when graph size is fixed (e.g. when merging)
    The process of adding a node is roughly:
    1. Add the node to all level from top to the bottom, but do not connect it to any other node,
       nor try to promote itself to an entry node before the connection is done. (Unless the graph is empty
       and this is the first node, in that case we set the entry node and return)
    2. Do the search from top to bottom, remember all the possible neighbours on each level the node
       is on.
    3. Add the neighbor to the node from bottom to top level, when adding the neighbour,
       we always add all the outgoing links first before adding incoming link such that
       when a search visits this node, it can always find a way out
    4. If the node has level that is less or equal to graph level, then we're done here.
       If the node has level larger than graph level, then we need to promote the node
       as the entry node. If, while we add the node to the graph, the entry node has changed
       (which means the graph level has changed as well), we need to reinsert the node
       to the newly introduced levels (repeating step 2,3 for new levels) and again try to
       promote the node to entry node.
    */
    RandomVectorScorer scorer = scorerSupplier.scorer(node);
    // then promote itself as entry node if entry node is not set
    if (vamana.trySetNewEntryNode(node)) {
      return;
    }

    // if the entry node is already set, then we have to do all connections first before we can
    // promote ourselves as entry node

    // NOTE: the entry node and max level may not be paired, but because we get the level first
    // we ensure that the entry node we get later will always exist on the curMaxLevel
    int[] eps = new int[] {vamana.entryNode()};

    // for levels <= nodeLevel search with topk = beamWidth, and add connections
    GraphBuilderKnnCollector candidates = beamCandidates;
    candidates.clear();
    graphSearcher.search(candidates, scorer, eps, vamana, null);
    vamana.addNode(node);

    NeighborArray scratch = new NeighborArray(Math.max(beamCandidates.k(), M + 1), true);
    popToScratch(candidates, scratch);
    addDiverseNeighbors(node, scratch);
  }

  public void finish(List<float[]> vectors) throws IOException {
    cleanup();
    setEntryPointToMedioid(vectors);
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

      SelectedDiverse selected = selectDiverse(neighbors);
      neighbors.clear();
      for (Candidate candidate : selected.candidates) {
        neighbors.addInOrder(candidate.node, candidate.score);
      }
    }
  }

  private void setEntryPointToMedioid(List<float[]> vectors) {
    var ep = calculateEntryPoint(vectors);
    getGraph().setEntryNode(ep);
  }

  private int calculateEntryPoint(List<float[]> vectors) {
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

  private void addDiverseNeighbors(int node, NeighborArray candidates) throws IOException {
    /* For each of the beamWidth nearest candidates (going from best to worst), select it only if it
     * is closer to target than it is to any of the already-selected neighbors (ie selected in this method,
     * since the node is new and has no prior neighbors).
     */
    NeighborArray neighbors = vamana.getNeighbors(node);
    assert neighbors.size() == 0; // new node
    SelectedDiverse selected = selectDiverse(candidates);
    // here we don't need to lock, because there's no incoming link so no others is able to
    // discover this node such that no others will modify this neighbor array as well
    for (Candidate candidate : selected.candidates) {
      neighbors.addInOrder(candidate.node, candidate.score);
    }

    // Link the selected nodes to the new node, and the new node to the selected nodes (again
    // applying diversity heuristic)
    // NOTE: here we're using candidates and mask but not the neighbour array because once we have
    // added incoming link there will be possibilities of this node being discovered and neighbour
    // array being modified. So using local candidates and mask is a safer option.
    for (int i = 0; i < candidates.size(); i++) {
      if (!selected.selected.get(i)) {
        continue;
      }

      int nbr = candidates.node[i];
      NeighborArray nbrsOfNbr = vamana.getNeighbors(nbr);
      nbrsOfNbr.rwlock.writeLock().lock();
      try {
        nbrsOfNbr.insertSorted(node, candidates.score[i]);
        // FIXME: this M * 2 is the overflow factor in jvector, and SLACK in diskann
        if (nbrsOfNbr.size() > M * 2) {
          SelectedDiverse stillDiverse = selectDiverse(nbrsOfNbr);

          nbrsOfNbr.clear();
          for (Candidate diverse : stillDiverse.candidates) {
            nbrsOfNbr.addInOrder(diverse.node, diverse.score);
          }
        }
      } finally {
        nbrsOfNbr.rwlock.writeLock().unlock();
      }
    }
  }

  /**
   * This method will select neighbors to add and return a mask telling the caller which candidates
   * are selected
   */
  private SelectedDiverse selectDiverse(NeighborArray candidates) throws IOException {
    BitSet selected = new FixedBitSet(candidates.size());
    List<Candidate> selectedCandidates = new ArrayList<>(M);

    for (float a = 1.0f; a < alpha + 1E-6 && selectedCandidates.size() < M; a += 0.2f) {
      for (int i = 0; selectedCandidates.size() < M && i < candidates.size(); i++) {
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
    return new SelectedDiverse(selected, selectedCandidates);
  }

  private static void popToScratch(GraphBuilderKnnCollector candidates, NeighborArray scratch) {
    int candidateCount = candidates.size();
    var reverseOrdered = new Candidate[candidateCount];
    // FIXME: fix the collector to be in the right order

    for (int i = 0; i < candidateCount; i++) {
      float maxSimilarity = candidates.minimumScore();
      reverseOrdered[i] = new Candidate(candidates.popNode(), maxSimilarity);
    }

    for (int i = reverseOrdered.length - 1; i >= 0; i--) {
      var candidate = reverseOrdered[i];
      scratch.addInOrder(candidate.node, candidate.score);
    }
  }

  /**
   * @param candidate the vector of a new candidate neighbor of a node n
   * @param score the score of the new candidate and node n, to be compared with scores of the
   *     candidate and n's neighbors
   * @param neighbors the neighbors selected so far
   * @return whether the candidate is diverse given the existing neighbors
   */
  private boolean diversityCheck(
      int candidate, float score, NeighborArray neighbors, BitSet selected, float a)
      throws IOException {
    RandomVectorScorer scorer = scorerSupplier.scorer(candidate);

    for (int i = selected.nextSetBit(0);
        i != DocIdSetIterator.NO_MORE_DOCS;
        i = selected.nextSetBit(i + 1)) {
      int otherNode = neighbors.node[i];
      if (otherNode == candidate) {
        return false;
      }

      float neighborSimilarity = scorer.score(otherNode);
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

  private record SelectedDiverse(BitSet selected, List<Candidate> candidates) {}

  private record Candidate(int node, float score) implements Comparable<Candidate> {

    @Override
    public int compareTo(Candidate o) {
      var score = Float.compare(this.score, o.score);
      if (score != 0) {
        return score;
      }

      return Integer.compare(this.node, o.node);
    }
  }
}
