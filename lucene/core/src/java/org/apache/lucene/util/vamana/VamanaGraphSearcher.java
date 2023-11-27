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

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.util.Map;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.SparseFixedBitSet;
import org.apache.lucene.util.vamana.VamanaGraph.ArrayNodesIterator;

/**
 * Searches an HNSW graph to find nearest neighbors to a query vector. For more background on the
 * search algorithm, see {@link VamanaGraph}.
 */
public class VamanaGraphSearcher {

  public record CachedNode(float[] vector, ArrayNodesIterator neighbors) {}

  /**
   * Scratch data structures that are used in each {@link #search} call. These can be expensive to
   * allocate, so they're cleared and reused across calls.
   */
  private final NeighborQueue candidates;

  private final Map<Integer, CachedNode> cache;

  private BitSet visited;

  /**
   * Creates a new graph searcher.
   *
   * @param candidates max heap that will track the candidate nodes to explore
   * @param visited bit set that will track nodes that have already been visited
   */
  public VamanaGraphSearcher(NeighborQueue candidates, BitSet visited) {
    this(candidates, visited, null);
  }

  public VamanaGraphSearcher(
      NeighborQueue candidates, BitSet visited, Map<Integer, CachedNode> cache) {
    this.candidates = candidates;
    this.visited = visited;
    this.cache = cache;
  }

  /**
   * Searches HNSW graph for the nearest neighbors of a query vector.
   *
   * @param scorer the scorer to compare the query with the nodes
   * @param knnCollector a collector of top knn results to be returned
   * @param graph the graph values. May represent the entire graph, or a level in a hierarchical
   *     graph.
   * @param acceptOrds {@link Bits} that represents the allowed document ordinals to match, or
   *     {@code null} if they are all allowed to match.
   */
  public static void search(
      RandomVectorScorer scorer,
      KnnCollector knnCollector,
      VamanaGraph graph,
      Bits acceptOrds,
      Map<Integer, CachedNode> cache)
      throws IOException {
    VamanaGraphSearcher graphSearcher =
        new VamanaGraphSearcher(
            new NeighborQueue(knnCollector.k(), true),
            new SparseFixedBitSet(getGraphSize(graph)),
            cache);
    search(scorer, knnCollector, graph, graphSearcher, acceptOrds);
  }

  /**
   * Search {@link OnHeapVamanaGraph}, this method is thread safe.
   *
   * @param scorer the scorer to compare the query with the nodes
   * @param topK the number of nodes to be returned
   * @param graph the graph values. May represent the entire graph, or a level in a hierarchical
   *     graph.
   * @param acceptOrds {@link Bits} that represents the allowed document ordinals to match, or
   *     {@code null} if they are all allowed to match.
   * @param visitedLimit the maximum number of nodes that the search is allowed to visit
   * @return a set of collected vectors holding the nearest neighbors found
   */
  public static KnnCollector search(
      RandomVectorScorer scorer,
      int topK,
      OnHeapVamanaGraph graph,
      Bits acceptOrds,
      int visitedLimit)
      throws IOException {
    KnnCollector knnCollector = new TopKnnCollector(topK, visitedLimit);
    OnHeapVamanaGraphSearcher graphSearcher =
        new OnHeapVamanaGraphSearcher(
            new NeighborQueue(topK, true), new SparseFixedBitSet(getGraphSize(graph)));
    search(scorer, knnCollector, graph, graphSearcher, acceptOrds);
    return knnCollector;
  }

  private static void search(
      RandomVectorScorer scorer,
      KnnCollector knnCollector,
      VamanaGraph graph,
      VamanaGraphSearcher graphSearcher,
      Bits acceptOrds)
      throws IOException {
    int initialEp = graph.entryNode();
    if (initialEp == -1) {
      return;
    }

    graphSearcher.search(knnCollector, scorer, new int[] {initialEp}, graph, acceptOrds);
  }

  /**
   * Searches for the nearest neighbors of a query vector in a given level.
   *
   * <p>If the search stops early because it reaches the visited nodes limit, then the results will
   * be marked incomplete through {@link NeighborQueue#incomplete()}.
   *
   * @param scorer the scorer to compare the query with the nodes
   * @param topK the number of nearest to query results to return
   * @param eps the entry points for search at this level expressed as level 0th ordinals
   * @param graph the graph values
   * @return a set of collected vectors holding the nearest neighbors found
   */
  public VamanaGraphBuilder.GraphBuilderKnnCollector search(
      // Note: this is only public because Lucene91HnswGraphBuilder needs it
      RandomVectorScorer scorer, int topK, final int[] eps, VamanaGraph graph) throws IOException {
    VamanaGraphBuilder.GraphBuilderKnnCollector results =
        new VamanaGraphBuilder.GraphBuilderKnnCollector(topK);
    search(results, scorer, eps, graph, null);
    return results;
  }

  /**
   * Function to find the best entry point from which to search the zeroth graph layer.
   *
   * @param scorer the scorer to compare the query with the nodes
   * @param graph the HNSWGraph
   * @param visitLimit How many vectors are allowed to be visited
   * @return An integer array whose first element is the best entry point, and second is the number
   *     of candidates visited. Entry point of `-1` indicates visitation limit exceed
   * @throws IOException When accessing the vector fails
   */
  private int[] findBestEntryPoint(RandomVectorScorer scorer, VamanaGraph graph, long visitLimit)
      throws IOException {
    int size = getGraphSize(graph);
    int visitedCount = 1;
    prepareScratchState(size);
    int currentEp = graph.entryNode();
    float currentScore = scorer.score(currentEp);
    boolean foundBetter = true;
    visited.set(currentEp);
    // Keep searching until we stop finding a better candidate entry point
    while (foundBetter) {
      foundBetter = false;
      graphSeek(graph, currentEp);
      int friendOrd;
      while ((friendOrd = graphNextNeighbor(graph)) != NO_MORE_DOCS) {
        assert friendOrd < size : "friendOrd=" + friendOrd + "; size=" + size;
        if (visited.getAndSet(friendOrd)) {
          continue;
        }
        if (visitedCount >= visitLimit) {
          return new int[] {-1, visitedCount};
        }
        float friendSimilarity = scorer.score(friendOrd);
        visitedCount++;
        if (friendSimilarity > currentScore
            || (friendSimilarity == currentScore && friendOrd < currentEp)) {
          currentScore = friendSimilarity;
          currentEp = friendOrd;
          foundBetter = true;
        }
      }
    }
    return new int[] {currentEp, visitedCount};
  }

  /**
   * Add the closest neighbors found to a priority queue (heap). These are returned in REVERSE
   * proximity order -- the most distant neighbor of the topK found, i.e. the one with the lowest
   * score/comparison value, will be at the top of the heap, while the closest neighbor will be the
   * last to be popped.
   */
  void search(
      KnnCollector results,
      RandomVectorScorer scorer,
      final int[] eps,
      VamanaGraph graph,
      Bits acceptOrds)
      throws IOException {

    int size = getGraphSize(graph);

    prepareScratchState(size);

    for (int ep : eps) {
      if (visited.getAndSet(ep) == false) {
        if (results.earlyTerminated()) {
          break;
        }
        float score = scorer.score(ep);
        results.incVisitedCount(1);
        candidates.add(ep, score);
        results.cacheNode(ep);
        if (acceptOrds == null || acceptOrds.get(ep)) {
          results.collect(ep, score);
        }
      }
    }

    // A bound that holds the minimum similarity to the query vector that a candidate vector must
    // have to be considered.
    float minAcceptedSimilarity = results.minCompetitiveSimilarity();
    while (candidates.size() > 0 && results.earlyTerminated() == false) {
      // get the best candidate (closest or best scoring)
      float topCandidateSimilarity = candidates.topScore();
      if (topCandidateSimilarity < minAcceptedSimilarity) {
        break;
      }

      int topCandidateNode = candidates.pop();
      graphSeek(graph, topCandidateNode);
      results.cacheNode(topCandidateNode);
      int friendOrd;
      var neighbors = graph.getNeighbors();
      while (neighbors.hasNext()) {
        //      while ((friendOrd = graphNextNeighbor(graph)) != NO_MORE_DOCS) {
        friendOrd = neighbors.nextInt();
        assert friendOrd < size : "friendOrd=" + friendOrd + "; size=" + size;
        if (visited.getAndSet(friendOrd)) {
          continue;
        }

        if (results.earlyTerminated()) {
          break;
        }
        float friendSimilarity = scorer.score(friendOrd);
        results.incVisitedCount(1);
        if (friendSimilarity >= minAcceptedSimilarity) {
          candidates.add(friendOrd, friendSimilarity);
          if (acceptOrds == null || acceptOrds.get(friendOrd)) {
            if (results.collect(friendOrd, friendSimilarity)) {
              minAcceptedSimilarity = results.minCompetitiveSimilarity();
            }
          }
        }
      }
    }
  }

  private void prepareScratchState(int capacity) {
    candidates.clear();
    if (visited.length() < capacity) {
      visited = FixedBitSet.ensureCapacity((FixedBitSet) visited, capacity);
    }
    visited.clear();
  }

  /**
   * Seek a specific node in the given graph. The default implementation will just call {@link
   * VamanaGraph#seek(int)}
   *
   * @throws IOException when seeking the graph
   */
  void graphSeek(VamanaGraph graph, int targetNode) throws IOException {
    graph.seek(targetNode);
  }

  /**
   * Get the next neighbor from the graph, you must call {@link #graphSeek(VamanaGraph, int)} before
   * calling this method. The default implementation will just call {@link
   * VamanaGraph#nextNeighbor()}
   *
   * @return see {@link VamanaGraph#nextNeighbor()}
   * @throws IOException when advance neighbors
   */
  int graphNextNeighbor(VamanaGraph graph) throws IOException {
    return graph.nextNeighbor();
  }

  private static int getGraphSize(VamanaGraph graph) {
    return graph.maxNodeId() + 1;
  }

  /**
   * This class allows {@link OnHeapVamanaGraph} to be searched in a thread-safe manner by avoiding
   * the unsafe methods (seek and nextNeighbor, which maintain state in the graph object) and
   * instead maintaining the state in the searcher object.
   *
   * <p>Note the class itself is NOT thread safe, but since each search will create a new Searcher,
   * the search methods using this class are thread safe.
   */
  private static class OnHeapVamanaGraphSearcher extends VamanaGraphSearcher {

    private NeighborArray cur;
    private int upto;

    private OnHeapVamanaGraphSearcher(NeighborQueue candidates, BitSet visited) {
      super(candidates, visited);
    }

    @Override
    void graphSeek(VamanaGraph graph, int targetNode) {
      cur = ((OnHeapVamanaGraph) graph).getNeighbors(targetNode);
      upto = -1;
    }

    @Override
    int graphNextNeighbor(VamanaGraph graph) {
      if (++upto < cur.size()) {
        return cur.node[upto];
      }
      return NO_MORE_DOCS;
    }
  }
}
