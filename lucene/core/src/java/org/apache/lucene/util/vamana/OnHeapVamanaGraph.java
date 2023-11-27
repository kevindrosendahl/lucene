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

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.RamUsageEstimator;

/**
 * An {@link VamanaGraph} where all nodes and connections are held in memory. This class is used to
 * construct the HNSW graph before it's written to the index.
 */
public final class OnHeapVamanaGraph extends VamanaGraph implements Accountable {

  private static final int INIT_SIZE = 128;

  private final AtomicReference<EntryNode> entryNode;

  private NeighborArray[] graph;
  private final AtomicInteger size =
      new AtomicInteger(0); // graph size, which is number of nodes in level 0

  // is only used to account memory usage
  private final AtomicInteger maxNodeId = new AtomicInteger(-1);
  private final int nsize0; // neighbour array size at zero level
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
    this.entryNode = new AtomicReference<>(new EntryNode(-1));
    this.nsize0 = M;
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
    return graph[node];
  }

  @Override
  public int size() {
    return size.get();
  }

  /**
   * When we initialize from another graph, the max node id is different from {@link #size()},
   * because we will add nodes out of order, such that we need two method for each
   *
   * @return max node id (inclusive)
   */
  @Override
  public int maxNodeId() {
    if (noGrowth) {
      // we know the eventual graph size and the graph can possibly
      // being concurrently modified
      return graph.length - 1;
    } else {
      // The graph cannot be concurrently modified (and searched) if
      // we don't know the size beforehand, so it's safe to return the
      // actual maxNodeId
      return maxNodeId.get();
    }
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
    if (node >= graph.length) {
      if (noGrowth) {
        throw new IllegalStateException(
            "The graph does not expect to grow when an initial size is given");
      }
      graph = ArrayUtil.grow(graph, node + 1);
    }

    graph[node] = new NeighborArray(nsize0, true);
    size.incrementAndGet();
    maxNodeId.accumulateAndGet(node, Math::max);
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

  @Override
  public NodesIterator getNeighbors() {
    return new ArrayNodesIterator(cur.node, cur.node.length);
  }

  /**
   * Returns the graph's current entry node on the top level shown as ordinals of the nodes on 0th
   * level
   *
   * @return the graph's current entry node on the top level
   */
  @Override
  public int entryNode() {
    return entryNode.get().node;
  }

  /**
   * Try to set the entry node if the graph does not have one
   *
   * @return True if the entry node is set to the provided node. False if the entry node already
   *     exists
   */
  public boolean trySetNewEntryNode(int node) {
    EntryNode current = entryNode.get();
    if (current.node == -1) {
      return entryNode.compareAndSet(current, new EntryNode(node));
    }
    return false;
  }

  public void setEntryNode(int node) {
    entryNode.set(new EntryNode(node));
  }

  /**
   * WARN: calling this method will essentially iterate through all nodes at level 0 (even if you're
   * not getting node at level 0), we have built some caching mechanism such that if graph is not
   * changed only the first non-zero level call will pay the cost. So it is highly NOT recommended
   * to call this method while the graph is still building.
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
        (long) nsize0 * (Integer.BYTES + Float.BYTES)
            + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER * 2L
            + RamUsageEstimator.NUM_BYTES_OBJECT_REF * 2L
            + Integer.BYTES * 3;
    long total = 0;
    total +=
        size() * (neighborArrayBytes0 + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER)
            + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER; // for graph and level 0;
    total += 2 * Integer.BYTES; // all int fields
    total += 1; // field: noGrowth
    total +=
        RamUsageEstimator.NUM_BYTES_OBJECT_REF
            + RamUsageEstimator.NUM_BYTES_OBJECT_HEADER
            + 2 * Integer.BYTES; // field: entryNode
    total += 2L * (Integer.BYTES + RamUsageEstimator.NUM_BYTES_OBJECT_HEADER); // 2 AtomicInteger
    total += RamUsageEstimator.NUM_BYTES_OBJECT_REF; // field: cur

    return total;
  }

  @Override
  public String toString() {
    return "OnHeapHnswGraph(size=" + size() + ", entryNode=" + entryNode() + ")";
  }

  private record EntryNode(int node) {}
}
