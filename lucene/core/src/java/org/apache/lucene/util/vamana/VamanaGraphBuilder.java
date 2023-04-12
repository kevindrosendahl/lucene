package org.apache.lucene.util.vamana;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.NavigableSet;
import java.util.Set;
import java.util.TreeSet;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

// FIXME: use NeighborQueue instead of NavigableSet<Candidate>
// FIXME: investigate NeighborArray for mutable graph
// FIXME: investigate BitSet for visited
public class VamanaGraphBuilder {
  private final int L;
  private final int R;
  private final int C;
  private final float alpha;

  private final VectorSimilarityFunction similarityFunction;
  private final RandomAccessVectorValues<float[]> vectors;

  public VamanaGraphBuilder(
      int L,
      int R,
      int C,
      float alpha,
      VectorSimilarityFunction similarityFunction,
      RandomAccessVectorValues<float[]> vectors) {
    this.L = L;
    this.R = R;
    this.C = C;
    this.alpha = alpha;
    this.similarityFunction = similarityFunction;
    this.vectors = vectors;
  }

  public MutableGraph build() throws IOException {
    int n = this.vectors.size();

    int s = calculateEntryPoint();
    MutableGraph graph = new MutableGraph(n, s);


    for (int iOrd = 0; iOrd < n; iOrd++) {
      var result = greedySearch(s, this.vectors.vectorValue(iOrd), 1, this.L, graph);
      // FIXME: alpha should maybe be this.alpha here?
      robustPrune(iOrd, result.visited, 1, this.R, graph);

      for (int jOrd : graph.getNode(iOrd).neighbors) {
        Node jNode = graph.getNode(jOrd);
        Set<Integer> jNeighbors = jNode.neighbors;
        List<Integer> jCandidates = new ArrayList<>(jNeighbors);
        if (!jNeighbors.contains(iOrd)) {
          jCandidates.add(iOrd);
        }

        if (jCandidates.size() > this.R) {
          robustPrune(jOrd, jCandidates, this.alpha, this.R, graph);
        } else {
          jNode.neighbors.add(iOrd);
        }
      }
    }

    // FIXME: add slack into the initial build and prune on second pass to R.

    return graph;
  }

  public List<Integer> search(MutableGraph graph, float[] query, int k) throws IOException {
    return greedySearch(graph.entryPoint, query, k, k, graph).topK;
  }

  // https://github.com/microsoft/DiskANN/blob/d23642271b1dd29740904ef97b81134f1ceb159c/src/index.cpp#L361
  private int calculateEntryPoint() throws IOException {
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

  private GreedySearchResult greedySearch(
      int startNode, float[] queryVector, int k, int numCandidates, MutableGraph graph)
      throws IOException {
    NavigableSet<Candidate> candidates = new TreeSet<>(Comparator.comparing(Candidate::distance));
    candidates.add(new Candidate(startNode, getDistance(queryVector, startNode)));

    Set<Integer> visited = new HashSet<>();

    while (true) {
      var unvisitedCandidates =
          candidates.stream().filter(candidate -> !visited.contains(candidate.ordinal)).toList();
      if (unvisitedCandidates.isEmpty()) {
        break;
      }

      int closestOrd = findClosest(queryVector, unvisitedCandidates);
      Node closest = graph.getNode(closestOrd);
      List<Candidate> neighborCandidates = getNeighborCandidates(queryVector, closest);

      visited.add(closestOrd);
      candidates.addAll(neighborCandidates);

      if (candidates.size() > numCandidates) {
        trimToNumCandidates(candidates, numCandidates);
      }
    }

    List<Integer> topK = candidates.stream().sequential().limit(k).map(Candidate::ordinal).toList();
    return new GreedySearchResult(topK, visited.stream().toList());
  }

  // FIXME: DiskANN does a few rounds here with increasing alpha, starting at 1 then
  //        bumping up by *= 1.2 until reaching the actual alpha
  private void robustPrune(
      int pOrd, List<Integer> candidateOrds, float alpha, int R, MutableGraph graph)
      throws IOException {
    float[] pVector = this.vectors.vectorValue(pOrd);

    // Begin by considering all candidates except p itself.
    NavigableSet<Candidate> candidates = new TreeSet<>(Comparator.comparing(Candidate::distance));
    for (int candidateOrd : candidateOrds) {
      if (candidateOrd == pOrd) {
        continue;
      }

      float distance = getDistance(pVector, candidateOrd);
      candidates.add(new Candidate(candidateOrd, distance));
    }

    // Also consider neighbors of p.
    Node pNode = graph.getNode(pOrd);
    for (int neighborOrd : pNode.neighbors) {
      assert neighborOrd != pOrd;

      float distance = getDistance(pVector, neighborOrd);
      candidates.add(new Candidate(neighborOrd, distance));
    }

    assert !candidates.isEmpty();

    // We will be resetting p's neighbors, so clear them for now.
    pNode.neighbors.clear();

    while (!candidates.isEmpty()) {
      // Add the closest candidate
      Candidate closest = candidates.pollFirst();
      pNode.neighbors.add(closest.ordinal);

      if (pNode.neighbors.size() >= R) {
        break;
      }

      float[] closestVector = this.vectors.vectorValue(closest.ordinal);
      Set<Candidate> removals = new HashSet<>();
      for (Candidate pPrime : candidates) {
        // If the distance between a candidate and this iteration's closest vector with a boost
        // of alpha is smaller than the distance between the candidate and the original vector,
        // don't consider it a candidate any more.
        if (pPrime.distance / getDistance(closestVector, pPrime.ordinal) > alpha) {
          removals.add(pPrime);
        }
      }

      candidates.removeAll(removals);
    }
  }

  private int findClosest(float[] queryVector, List<Candidate> candidates) throws IOException {
    assert !candidates.isEmpty();
    int closest = -1;
    float closestDistance = Float.MAX_VALUE;

    for (Candidate candidate : candidates) {
      float distance = getDistance(queryVector, candidate.ordinal);

      if (distance < closestDistance) {
        closest = candidate.ordinal;
        closestDistance = distance;
      }
    }

    return closest;
  }

  private float getDistance(float[] queryVector, int candidateOrd) throws IOException {
    float[] candidate = vectors.vectorValue(candidateOrd);
    return similarityFunction.compare(queryVector, candidate);
  }

  private List<Candidate> getNeighborCandidates(float[] queryVector, Node node) throws IOException {
    List<Candidate> neighborCandidates = new ArrayList<>(node.neighbors.size());
    for (int neighbor : node.neighbors) {
      neighborCandidates.add(new Candidate(neighbor, getDistance(queryVector, neighbor)));
    }

    return neighborCandidates;
  }

  private void trimToNumCandidates(NavigableSet<Candidate> candidates, int numCandidates) {
    assert candidates.size() > numCandidates;

    int numToRemove = candidates.size() - numCandidates;
    for (int i = 0; i < numToRemove; i++) {
      candidates.pollLast();
    }
  }

  public static class MutableGraph {
    private Node[] nodes;
    public int entryPoint;

    public MutableGraph(int n, int entryPoint) {
      this.nodes = new Node[n];
      for (int i = 0; i < n; i++) {
        this.nodes[i] = new Node(i, new HashSet<>());
      }

      this.entryPoint = entryPoint;
    }

    public Node getNode(int ordinal) {
      return nodes[ordinal];
    }
  }

  private record Candidate(int ordinal, float distance) {}

  private record Node(int ordinal, Set<Integer> neighbors) {}

  private record GreedySearchResult(List<Integer> topK, List<Integer> visited) {}
}
