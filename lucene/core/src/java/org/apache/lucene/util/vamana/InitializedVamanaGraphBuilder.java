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
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.BitSet;

/**
 * This creates a graph builder that is initialized with the provided HnswGraph. This is useful for
 * merging HnswGraphs from multiple segments.
 *
 * @lucene.experimental
 */
public final class InitializedVamanaGraphBuilder extends VamanaGraphBuilder {

  /**
   * Create a new HnswGraphBuilder that is initialized with the provided HnswGraph.
   *
   * @param scorerSupplier the scorer to use for vectors
   * @param M the number of connections to keep per node
   * @param beamWidth the number of nodes to explore in the search
   * @param initializerGraph the graph to initialize the new graph builder
   * @param newOrdMap a mapping from the old node ordinal to the new node ordinal
   * @param initializedNodes a bitset of nodes that are already initialized in the initializerGraph
   * @param totalNumberOfVectors the total number of vectors in the new graph, this should include
   *     all vectors expected to be added to the graph in the future
   * @return a new HnswGraphBuilder that is initialized with the provided HnswGraph
   * @throws IOException when reading the graph fails
   */
  public static InitializedVamanaGraphBuilder fromGraph(
      RandomVectorScorerSupplier scorerSupplier,
      VectorSimilarityFunction similarityFunction,
      int M,
      int beamWidth,
      float alpha,
      VamanaGraph initializerGraph,
      int[] newOrdMap,
      BitSet initializedNodes,
      int totalNumberOfVectors)
      throws IOException {
    return new InitializedVamanaGraphBuilder(
        scorerSupplier,
        similarityFunction,
        M,
        beamWidth,
        alpha,
        initGraph(M, initializerGraph, newOrdMap, totalNumberOfVectors),
        initializedNodes);
  }

  public static OnHeapVamanaGraph initGraph(
      int M, VamanaGraph initializerGraph, int[] newOrdMap, int totalNumberOfVectors)
      throws IOException {
    OnHeapVamanaGraph vamana = new OnHeapVamanaGraph(M, totalNumberOfVectors);
    VamanaGraph.NodesIterator it = initializerGraph.getNodes();
    while (it.hasNext()) {
      int oldOrd = it.nextInt();
      int newOrd = newOrdMap[oldOrd];
      vamana.addNode(newOrd);
      vamana.trySetNewEntryNode(newOrd);
      NeighborArray newNeighbors = vamana.getNeighbors(newOrd);
      initializerGraph.seek(oldOrd);
      for (int oldNeighbor = initializerGraph.nextNeighbor();
          oldNeighbor != NO_MORE_DOCS;
          oldNeighbor = initializerGraph.nextNeighbor()) {
        int newNeighbor = newOrdMap[oldNeighbor];
        // we will compute these scores later when we need to pop out the non-diverse nodes
        newNeighbors.addOutOfOrder(newNeighbor, Float.NaN);
      }
    }
    return vamana;
  }

  private final BitSet initializedNodes;

  public InitializedVamanaGraphBuilder(
      RandomVectorScorerSupplier scorerSupplier,
      VectorSimilarityFunction similarityFunction,
      int M,
      int beamWidth,
      float alpha,
      OnHeapVamanaGraph initializedGraph,
      BitSet initializedNodes)
      throws IOException {
    super(scorerSupplier, similarityFunction, M, beamWidth, alpha, initializedGraph);
    this.initializedNodes = initializedNodes;
  }

  @Override
  public void addGraphNode(int node) throws IOException {
    if (initializedNodes.get(node)) {
      return;
    }
    super.addGraphNode(node);
  }
}
