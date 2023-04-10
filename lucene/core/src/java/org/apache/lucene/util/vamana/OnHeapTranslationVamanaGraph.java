package org.apache.lucene.util.vamana;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.IntStream;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

public class OnHeapTranslationVamanaGraph {
  private final int L;
  private final int R;
  private final int C;
  private final float alpha;
  private final boolean saturateGraph;

  private final VectorSimilarityFunction similarityFunction;
  private final VectorEncoding vectorEncoding;
  private final RandomAccessVectorValues<float[]> vectors;

  public OnHeapTranslationVamanaGraph(
      int L,
      int R,
      int C,
      float alpha,
      boolean saturateGraph,
      VectorSimilarityFunction similarityFunction,
      VectorEncoding vectorEncoding,
      RandomAccessVectorValues<float[]> vectors) {
    this.L = L;
    this.R = R;
    this.C = C;
    this.alpha = alpha;
    this.saturateGraph = saturateGraph;
    this.similarityFunction = similarityFunction;
    this.vectorEncoding = vectorEncoding;
    this.vectors = vectors;
  }

  private void build() {

  }

  // https://github.com/microsoft/DiskANN/blob/d23642271b1dd29740904ef97b81134f1ceb159c/src/index.cpp#L721
  private void link() throws IOException {
    int numVectors = vectors.size();

    List<Integer> visitOrder = IntStream.range(0, numVectors).boxed().toList();
    List<List<Integer>> finalGraph = new ArrayList<>(numVectors);
    for (int i = 0; i < numVectors; i++) {
      // magic numbers from DiskANN
      finalGraph.set(i, new ArrayList<>((int)Math.ceil(this.R * 1.3 * 1.05)));
    }

    // FIXME: can probably remove. not in current code
    int numSyncs = 40;

    int entryPoint = calculateEntryPoint();
    Set<Integer> uniqueStartPoints = new HashSet<>();
    uniqueStartPoints.add(entryPoint);

    List<Integer> initIds = new ArrayList<>();
    initIds.add(entryPoint);

    // first round
    int roundSize = (int) Math.ceil(numVectors / (double)numSyncs);
    List<Integer> needToSync = new ArrayList<>(numVectors);
    List<List<Integer>> prunedListVector = new ArrayList<>(roundSize);
    for (int i = 0; i < roundSize; i++) {
      prunedListVector.set(i, new ArrayList<>());
    }

    for (int syncNum = 0; syncNum < numSyncs; syncNum++) {
      int startId = syncNum * roundSize;
      int endId = Math.min(numVectors, (syncNum +1) * roundSize);

      for (int nodeCtr = startId; nodeCtr < endId; nodeCtr++) {
        int node = visitOrder.get(nodeCtr);
        int nodeOffset = nodeCtr - startId;

        Set<Integer> visited = new HashSet<>(this.L * 2);
        List<Integer> prunedList = prunedListVector.get(nodeOffset);

        // "get nearest neighbors of n in tmp" (???)
        // "pool contains all the points that were checked along with their distances
        //  from n. visited contains all the points visited, just the ids"
        // FIXME: immediately reserve this.L * 10 in iterateToFixedPoint
        ArrayList<Neighbor> pool = new ArrayList<>(this.L * 2);
        getExpandedNodes(node, this.L, initIds, pool, visited, entryPoint, finalGraph);

        // "check the neighbors of the query that are not part of visited, check their
        //  distances to the query, and add it to pool"
        if (!finalGraph.get(node).isEmpty()) {
          for (int id : finalGraph.get(node)) {
            if (!visited.contains(id) && id != node) {
              float dist = this.similarityFunction.compare(
                  this.vectors.vectorValue(node),
                  this.vectors.vectorValue(id)
              );
              pool.add(new Neighbor(id, dist, true));
            }
          }
        }

        pruneNeighbors(node, pool, prunedList);
      }

      // pruneNeighbors will check pool, and remove some of the points and creat a cut_graph,
      // which contains neighbors for point n
      for (int nodeCtr = startId; nodeCtr < endId; nodeCtr++) {
        int node = visitOrder.get(nodeCtr);
        int nodeOffset = nodeCtr - startId;
        List<Integer> prunedList = prunedListVector.get(nodeOffset);

        finalGraph.get(node).clear();
        for (int id : prunedList) {
          finalGraph.get(node).add(id);
        }
      }

      for (int nodeCtr = startId; nodeCtr < endId; nodeCtr++) {
        int node = visitOrder.get(nodeCtr);
        int nodeOffset = nodeCtr - startId;
        List<Integer> prunedList = prunedListVector.get(nodeOffset);
        batchInterInsert(node, prunedList, needToSync, finalGraph);
        prunedList.clear();
      }
    }

  }

  // https://github.com/microsoft/DiskANN/blob/d23642271b1dd29740904ef97b81134f1ceb159c/src/index.cpp#L361
  private int calculateEntryPoint() throws IOException  {
    int numVectors = vectors.size();
    int dimensions = vectors.dimension();
    float[] center = new float[dimensions];

    // initialize centroid
    for (int i = 0; i < numVectors; i++) {
      for (int j = 0; j < dimensions; j++) {
        center[j] += vectors.vectorValue(i)[j];
      }
    }

    for (int i = 0; i < dimensions; i++) {
      center[i] /= dimensions;
    }

    // compute distances from centroid
    float[] distances = new float[numVectors];
    for (int i = 0; i < numVectors; i++) {
      float[] vector = vectors.vectorValue(i);

      float distance = 0;
      for (int j = 0; j < dimensions; j++) {
        distance += (center[j] - vector[j]) * (center[j] - vector[j]);
      }

      distances[i] = distance;
    }

    // find vector closest to the centroid
    int minIdx = 0;
    float minDist = distances[0];
    for (int i = 1; i < numVectors; i++) {
      float distance = distances[i];
      if (distance < minDist) {
        minIdx = i;
        minDist = distance;
      }
    }

    return minIdx;
  }

  // https://github.com/microsoft/DiskANN/blob/d23642271b1dd29740904ef97b81134f1ceb159c/src/index.cpp#L499
  private void getExpandedNodes(
      int nodeId,
      // FIXME: is this ever not this.L?
      int lIndex,
      List<Integer> initIds,
      ArrayList<Neighbor> expandedNodesInfo,
      Set<Integer> expandedNodesIds,
      int ep,
      List<List<Integer>> finalGraph) throws IOException {
    float[] nodeCoords = vectors.vectorValue(nodeId);
    ArrayList<Neighbor> bestLNodes = new ArrayList<>();

    // FIXME: maybe unneeded, can initIds ever be empty?
    if (initIds.size() == 0) {
      initIds.add(ep);
    }

    iterateToFixedPoint(
        nodeCoords,
        lIndex,
        initIds,
        expandedNodesInfo,
        expandedNodesIds,
        bestLNodes,
        finalGraph
    );
  }

  // https://github.com/microsoft/DiskANN/blob/d23642271b1dd29740904ef97b81134f1ceb159c/src/index.cpp#L415
  private FixedPointResult iterateToFixedPoint(
      float[] nodeCoords /* the vector we're iterating against */,
      int lSize,
      List<Integer> initIds,
      ArrayList<Neighbor> expandedNodesInfo,
      Set<Integer> expandedNodesIds,
      ArrayList<Neighbor> bestLNodes,
      List<List<Integer>> finalGraph) throws IOException {
    bestLNodes.ensureCapacity(lSize + 1);
    expandedNodesInfo.ensureCapacity(10 * lSize);
    // FIXME: supposed to make expandedNodesIds capacity = 10 * lSize, just do when initialized?

    int l = 0;
    Neighbor nn;
    Set<Integer> insertedIntoPool = new HashSet<>(lSize * 20);

    // For each vector in the initial set, add them to the pool, up to when we've reached L.
    for (int id : initIds) {
      nn = new Neighbor(
          id,
          this.similarityFunction.compare(vectors.vectorValue(id), nodeCoords),
          true
      );

      if (!insertedIntoPool.contains(id)) {
        insertedIntoPool.add(id);
        bestLNodes.set(l++, nn);
      }

      if (l == lSize) {
        break;
      }
    }

    // Sort bestLNodes based on distance of each point to nodeCords
    bestLNodes.sort(Comparator.comparingDouble(Neighbor::distance));
    int k = 0;
    int hops = 0;
    int cmps = 0;

    // By here l is min(initIds, this.L)
    while (k < l) {
      int nk = l;

      if (bestLNodes.get(k).flag) {
        bestLNodes.get(k).flag = false;
        int n = bestLNodes.get(k).id;
        expandedNodesInfo.add(bestLNodes.get(k));
        expandedNodesIds.add(n);

        for (int m = 0; m < finalGraph.get(n).size(); m++) {
          int id = finalGraph.get(n).get(m);
          if (!insertedIntoPool.contains(id)) {
            insertedIntoPool.add(id);

            if ((m + 1) < finalGraph.get(n).size()) {
              @SuppressWarnings("unused")
              int nextn = finalGraph.get(n).get(m+1);
              // FIXME: does an _mm_prefetch here
            }

            cmps++;
            float dist = this.similarityFunction.compare(nodeCoords, this.vectors.vectorValue(id));

            if (dist >= bestLNodes.get(l - 1).distance && (l == lSize)) {
              continue;
            }

            nn = new Neighbor(id, dist, true);
            insertIntoPool(bestLNodes, l, nn);
            if (l < lSize) {
              l++;
            }

            if (this.R < nk) {
              nk = this.R;
            }
          }
        }

        if (nk <= k) {
          k = nk;
        } else {
          k++;
        }
      } else {
        k++;
      }
    }

    return new FixedPointResult(hops, cmps);
  }

  private void pruneNeighbors(int location, List<Neighbor> pool, List<Integer> prunedList) throws IOException {
    if (pool.size() == 0) {
      return;
    }

    pool.sort(Comparator.comparingDouble(Neighbor::distance));

    List<Neighbor> result = new ArrayList<>(this.R);
    List<Float> occludeFactor = new ArrayList<>(pool.size());
    for (int i = 0; i < pool.size(); i++) {
      occludeFactor.set(i, 0f);
    }

    occludeList(pool, result, occludeFactor);

    // Add all the nodes in the result into a variable called cutGraph so this contains
    // all the neighbors of id location
    prunedList.clear();
    for (Neighbor neighbor : result) {
      if (neighbor.id != location) {
        prunedList.add(neighbor.id);
      }
    }

    if (this.saturateGraph && this.alpha > 1) {
      for (int i = 0; i < pool.size() && prunedList.size() < this.R; i++) {
        final int currI = i;
        if (prunedList.stream().anyMatch(pruned -> pruned == pool.get(currI).id) && pool.get(i).id != location) {
          prunedList.add(pool.get(i).id);
        }
      }
    }
  }

  private void occludeList(List<Neighbor> pool, List<Neighbor> result, List<Float> occludeFactor) throws IOException {
    if (pool.isEmpty()) {
      return;
    }

    float curAlpha = 1;
    while (curAlpha < this.alpha && result.size() < this.R) {
      int start = 0;

      while (result.size() < this.R && start < pool.size() && start < this.C) {
        Neighbor p = pool.get(start);
        if (occludeFactor.get(start) > curAlpha) {
          start++;
          continue;
        }

        occludeFactor.set(start, Float.MAX_VALUE);
        result.add(p);
        for (int t = start + 1; t < pool.size() && t < this.C; t++) {
          // FIXME: pass in alpha since it changes in the second round
          if (occludeFactor.get(t) > this.alpha) {
            continue;
          }

          float djk = this.similarityFunction.compare(
              this.vectors.vectorValue(pool.get(t).id),
              this.vectors.vectorValue(p.id)
          );

          occludeFactor.set(t, Math.max(occludeFactor.get(t), pool.get(t).distance / djk));
        }
        start++;
      }
      curAlpha *= 1.2;
    }
  }

  // https://github.com/microsoft/DiskANN/blob/d23642271b1dd29740904ef97b81134f1ceb159c/include/neighbor.h#L112
  private static int insertIntoPool(ArrayList<Neighbor> bestLNodes, int k, Neighbor nn) {
    int left = 0;
    int right = k - 1;
    if (bestLNodes.get(left).distance > nn.distance) {
      // FIXME: this is what DiskANN does. is it right?
      // seems like we should maybe pop the last off the list?
      // FIXME: indeed feels like a bug, now uses a priority queue
      Neighbor currentFirst = bestLNodes.get(0);
      bestLNodes.set(1, currentFirst);
      bestLNodes.set(0, nn);
      return left;
    }

    if (bestLNodes.get(right).distance < nn.distance) {
      bestLNodes.set(k, nn);
      return k;
    }

    while (right > 1 && left < right - 1) {
      int mid = (left + right) / 2;
      if (bestLNodes.get(mid).distance > nn.distance) {
        right = mid;
      } else {
        left = mid;
      }
    }

    while (left > 0) {
      if (bestLNodes.get(left).distance < nn.distance) {
        break;
      }

      if (bestLNodes.get(left).id == nn.id) {
        return k + 1;
      }

      left--;
    }

    if (bestLNodes.get(left).id == nn.id || bestLNodes.get(right).id == nn.id) {
      return k+ 1;
    }

    Neighbor currentRight = bestLNodes.get(right);
    bestLNodes.set(right + 1, currentRight);
    bestLNodes.set(right, nn);
    return right;
  }

  private void batchInterInsert(
      int n,
      List<Integer> prunedList,
      List<Integer> needToSync,
      List<List<Integer>> finalGraph
  ) {
    int range = this.R;

    for (int des : prunedList) {
      if (des == n) {
        continue;
      }

      if (finalGraph.get(des).stream().noneMatch(i -> i == n)) {
        finalGraph.get(des).add(n);
        // magic number from DiskANN
        if (finalGraph.get(des).size() > range * 1.3) {
          needToSync.set(des, 1);
        }
      }
    }
  }

  private record FixedPointResult(int hops, int compares) {}

  private static class Neighbor {
    public final int id;
    public final float distance;
    // FIXME: what does flag do?
    public boolean flag;


    public Neighbor(int id, float distance, boolean flag) {
      this.id = id;
      this.distance = distance;
      this.flag = flag;
    }

    public int id() {
      return id;
    }

    public float distance() {
      return distance;
    }

    public boolean flag() {
      return flag;
    }
  }
}
