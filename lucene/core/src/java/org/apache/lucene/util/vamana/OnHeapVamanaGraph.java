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

import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.RamUsageEstimator;

/**
 * An {@link VamanaGraph} where all nodes and connections are held in memory. This class is used to
 * construct the Vamana graph before it's written to the index.
 */
public final class OnHeapVamanaGraph extends VamanaGraph implements Accountable {

  private static final int INIT_SIZE = 128;

  private int entryNode; // the current graph entry node on the top level. -1 if not set

  // the internal graph representation
  // e.g. graph[1] is all the neighbours of node 1
  private NeighborArray[] graph;
  // levelToNodes
  private int size; // graph size, which is number of nodes in level 0
  private int maxNodeId;
  private final int nsize; // neighbour array size at zero level
  private final boolean
      noGrowth; // if an initial size is passed in, we don't expect the graph to grow itself

  // KnnGraphValues iterator members
  private int upto;
  private NeighborArray cur;

  /**
   * ctor
   *
   * @param numNodes number of nodes that will be added to this graph, passing in -1 means unbounded
   *     while passing in a non-negative value will lock the whole graph and disable the graph from
   *     growing itself (you cannot add a node with has id >= numNodes)
   */
  OnHeapVamanaGraph(int M, int numNodes) {
    this.entryNode = -1; // Entry node should be negative until a node is added
    // Neighbours' size on upper levels (nsize) and level 0 (nsize0)
    // We allocate extra space for neighbours, but then prune them to keep allowed maximum
    this.maxNodeId = -1;
    this.nsize = (M * 2 + 1);
    noGrowth = numNodes != -1;
    if (noGrowth == false) {
      numNodes = INIT_SIZE;
    }
    this.graph = new NeighborArray[numNodes];
  }

  /**
   * Returns the {@link NeighborQueue} connected to the given node.
   *
   * @param node the node whose neighbors are returned, represented as an ordinal on the level 0.
   */
  public NeighborArray getNeighbors(int node) {
    assert graph[node] != null;
    return graph[node];
  }

  @Override
  public int size() {
    return size;
  }

  /**
   * When we initialize from another graph, the max node id is different from {@link #size()},
   * because we will add nodes out of order, such that we need two method for each
   *
   * @return max node id (inclusive)
   */
  @Override
  public int maxNodeId() {
    return maxNodeId;
  }

  /**
   * Add node on the given level. Nodes can be inserted out of order, but it requires that the nodes
   * preceded by the node inserted out of order are eventually added.
   *
   * <p>NOTE: You must add a node starting from the node's top level
   *
   * @param node the node to add, represented as an ordinal on the level 0.
   */
  public void addNode(int node) {
    if (entryNode == -1) {
      entryNode = node;
    }

    if (node >= graph.length) {
      if (noGrowth) {
        throw new IllegalStateException(
            "The graph does not expect to grow when an initial size is given");
      }
      graph = ArrayUtil.grow(graph, node + 1);
    }

    graph[node] = new NeighborArray(nsize, true);
    size++;
    maxNodeId = Math.max(maxNodeId, node);
  }

  @Override
  public void seek(int targetNode) {
    cur = getNeighbors(targetNode);
    upto = -1;
  }

  @Override
  public int nextNeighbor() {
    if (++upto < cur.size()) {
      return cur.node[upto];
    }
    return NO_MORE_DOCS;
  }

  /**
   * Returns the graph's current entry node on the top level shown as ordinals of the nodes on 0th
   * level
   *
   * @return the graph's current entry node on the top level
   */
  @Override
  public int entryNode() {
    return entryNode;
  }

  /**
   * WARN: calling this method will essentially iterate through all nodes, we have built some
   * caching mechanism such that if graph is not changed only the first non-zero level call will pay
   * the cost. So it is highly NOT recommended to call this method while the graph is still
   * building.
   *
   * <p>NOTE: calling this method while the graph is still building is prohibited
   */
  @Override
  public NodesIterator getNodes() {
    if (size() != maxNodeId() + 1) {
      throw new IllegalStateException(
          "graph build not complete, size=" + size() + " maxNodeId=" + maxNodeId());
    }

    return new ArrayNodesIterator(size());
  }

  @Override
  public long ramBytesUsed() {
    long neighborArrayBytes0 =
        (long) nsize * (Integer.BYTES + Float.BYTES)
            + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER
            + RamUsageEstimator.NUM_BYTES_OBJECT_REF * 2L
            + Integer.BYTES * 3;
    long total = 0;
    total +=
        size * (neighborArrayBytes0 + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER)
            + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER; // for graph and level 0;
    total += 8 * Integer.BYTES; // all int fields
    total += RamUsageEstimator.NUM_BYTES_OBJECT_REF; // field: cur
    return total;
  }

  @Override
  public String toString() {
    return "OnHeapVamanaGraph(size=" + size() + ", entryNode=" + entryNode + ")";
  }
}
