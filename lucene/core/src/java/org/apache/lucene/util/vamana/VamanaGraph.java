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
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.PrimitiveIterator;
import org.apache.lucene.index.FloatVectorValues;

public abstract class VamanaGraph {

  /** Sole constructor */
  protected VamanaGraph() {}

  /**
   * Move the pointer to exactly the given {@code target}. After this method returns, call {@link
   * #nextNeighbor()} to return successive (ordered) connected node ordinals.
   *
   * @param target ordinal of a node in the graph, must be &ge; 0 and &lt; {@link
   *     FloatVectorValues#size()}.
   */
  public abstract void seek(int target) throws IOException;

  /** Returns the number of nodes in the graph */
  public abstract int size();

  /** Returns max node id, inclusive, normally this value will be size - 1 */
  public int maxNodeId() {
    return size() - 1;
  }

  /**
   * Iterates over the neighbor list. It is illegal to call this method after it returns
   * NO_MORE_DOCS without calling {@link #seek(int)}, which resets the iterator.
   *
   * @return a node ordinal in the graph, or NO_MORE_DOCS if the iteration is complete.
   */
  public abstract int nextNeighbor() throws IOException;

  /** Returns graph's entry point on the top level * */
  public abstract int entryNode() throws IOException;

  /**
   * Get all nodes on a given level as node 0th ordinals. The nodes are NOT guaranteed to be
   * presented in any particular order.
   *
   * @return an iterator over nodes where {@code nextInt} returns a next node on the level
   */
  public abstract NodesIterator getNodes() throws IOException;

  /** Empty graph value */
  public static VamanaGraph EMPTY =
      new VamanaGraph() {

        @Override
        public int nextNeighbor() {
          return NO_MORE_DOCS;
        }

        @Override
        public void seek(int target) {}

        @Override
        public int size() {
          return 0;
        }

        @Override
        public int entryNode() {
          return 0;
        }

        @Override
        public NodesIterator getNodes() {
          return ArrayNodesIterator.EMPTY;
        }
      };

  /**
   * Iterator over the graph nodes on a certain level, Iterator also provides the size – the total
   * number of nodes to be iterated over. The nodes are NOT guaranteed to be presented in any
   * particular order.
   */
  public abstract static class NodesIterator implements PrimitiveIterator.OfInt {

    protected final int size;

    /** Constructor for iterator based on the size */
    public NodesIterator(int size) {
      this.size = size;
    }

    /** The number of elements in this iterator * */
    public int size() {
      return size;
    }

    /**
     * Consume integers from the iterator and place them into the `dest` array.
     *
     * @param dest where to put the integers
     * @return The number of integers written to `dest`
     */
    public abstract int consume(int[] dest);

    public static int[] getSortedNodes(NodesIterator nodesOnLevel) {
      int[] sortedNodes = new int[nodesOnLevel.size()];
      for (int n = 0; nodesOnLevel.hasNext(); n++) {
        sortedNodes[n] = nodesOnLevel.nextInt();
      }
      Arrays.sort(sortedNodes);
      return sortedNodes;
    }
  }

  /** NodesIterator that accepts nodes as an integer array. */
  public static class ArrayNodesIterator extends NodesIterator {

    static NodesIterator EMPTY = new ArrayNodesIterator(0);

    private final int[] nodes;
    private int cur = 0;

    /** Constructor for iterator based on integer array representing nodes */
    public ArrayNodesIterator(int[] nodes, int size) {
      super(size);
      assert nodes != null;
      assert size <= nodes.length;
      this.nodes = nodes;
    }

    /** Constructor for iterator based on the size */
    public ArrayNodesIterator(int size) {
      super(size);
      this.nodes = null;
    }

    @Override
    public int consume(int[] dest) {
      if (hasNext() == false) {
        throw new NoSuchElementException();
      }
      int numToCopy = Math.min(size - cur, dest.length);
      if (nodes == null) {
        for (int i = 0; i < numToCopy; i++) {
          dest[i] = cur + i;
        }
        return numToCopy;
      }
      System.arraycopy(nodes, cur, dest, 0, numToCopy);
      cur += numToCopy;
      return numToCopy;
    }

    @Override
    public int nextInt() {
      if (hasNext() == false) {
        throw new NoSuchElementException();
      }
      if (nodes == null) {
        return cur++;
      } else {
        return nodes[cur++];
      }
    }

    @Override
    public boolean hasNext() {
      return cur < size;
    }
  }

  /** Nodes iterator based on set representation of nodes. */
  public static class CollectionNodesIterator extends NodesIterator {

    Iterator<Integer> nodes;

    /** Constructor for iterator based on collection representing nodes */
    public CollectionNodesIterator(Collection<Integer> nodes) {
      super(nodes.size());
      this.nodes = nodes.iterator();
    }

    @Override
    public int consume(int[] dest) {
      if (hasNext() == false) {
        throw new NoSuchElementException();
      }

      int destIndex = 0;
      while (hasNext() && destIndex < dest.length) {
        dest[destIndex++] = nextInt();
      }

      return destIndex;
    }

    @Override
    public int nextInt() {
      if (hasNext() == false) {
        throw new NoSuchElementException();
      }
      return nodes.next();
    }

    @Override
    public boolean hasNext() {
      return nodes.hasNext();
    }
  }
}
