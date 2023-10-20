package org.apache.lucene.util.clustering;

import java.util.Arrays;
import java.util.Random;

import org.apache.lucene.index.VectorSimilarityFunction;

public class KMeansPlusPlusClusterer implements Clusterer {

  private final VectorSimilarityFunction similarityFunction;
  private final int maxIterations;
  private final Random random;

  public KMeansPlusPlusClusterer(VectorSimilarityFunction similarityFunction, int maxIterations,
      Random random) {
    this.similarityFunction = similarityFunction;
    this.maxIterations = maxIterations;
    this.random = random;
  }

  public float[][] cluster(float[][] points, int k) {
    if (k <= 0 || k > points.length) {
      throw new IllegalArgumentException(String.format("invalid number of clusters: %d", k));
    }

    var runner = new ClusterRun(points, k, this.similarityFunction, random);
    return runner.run(this.maxIterations);
  }

  private static class ClusterRun {

    private final int k;
    private final VectorSimilarityFunction similarityFunction;
    private final Random random;
    private final float[][] points;
    private int[] assignments;
    private int[] clusterSizes;
    private float[][] clusterSums;
    private float[][] centroids;

    private ClusterRun(float[][] points, int k,
        VectorSimilarityFunction similarityFunction, Random random) {
      this.points = points;
      this.k = k;
      this.similarityFunction = similarityFunction;
      this.random = random;
      this.assignments = new int[points.length];
      this.clusterSizes = new int[k];
      this.clusterSums = new float[k][points[0].length];
      this.centroids = new float[k][];
    }

    private float[][] run(int maxIterations) {
      initialize();
      for (int i = 0; i < maxIterations; i++) {
        if (reassignPoints() <= 0.01 * points.length) {
          break;
        }
        recalculateCentroids();
      }
      return centroids;
    }

    private void initialize() {
      this.centroids = initializeCentroids();
      initialAssignments();
    }

    private float[][] initializeCentroids() {
      float[][] centroids = new float[k][];
      float[] distances = new float[points.length];
      Arrays.fill(distances, Float.MAX_VALUE);

      centroids[0] = points[random.nextInt(points.length)];
      updateDistances(distances, centroids[0]);

      for (int i = 1; i < k; i++) {
        float totalDistance = 0;
        for (float distance : distances) {
          totalDistance += distance;
        }

        float r = random.nextFloat() * totalDistance;

        int selectedIdx = selectIndexByDistance(distances, r);
        centroids[i] = points[selectedIdx];
        updateDistances(distances, centroids[i]);
      }

      return centroids;
    }

    private void updateDistances(float[] distances, float[] newCentroid) {
      for (int i = 0; i < points.length; i++) {
        float newDistance = distance(points[i], newCentroid);
        distances[i] = Math.min(distances[i], newDistance);
      }
    }

    private int selectIndexByDistance(float[] distances, float randomValue) {
      int selectedIdx = -1;
      for (int j = 0; j < distances.length; j++) {
        randomValue -= distances[j];
        if (randomValue < 1e-6) {
          selectedIdx = j;
          break;
        }
      }
      return selectedIdx != -1 ? selectedIdx : random.nextInt(points.length);
    }


    private void initialAssignments() {
      for (int i = 0; i < points.length; i++) {
        float[] point = points[i];
        int nearestCluster = findNearestCluster(point, centroids);
        assignments[i] = nearestCluster;

        clusterSizes[nearestCluster]++;
        addInPlace(clusterSums[nearestCluster], point);
      }
    }

    private int findNearestCluster(float[] point, float[][] centroids) {
      float minDistance = Float.MAX_VALUE;
      int nearestCluster = 0;
      for (int i = 0; i < centroids.length; i++) {
        float distance = distance(point, centroids[i]);
        if (distance < minDistance) {
          minDistance = distance;
          nearestCluster = i;
        }
      }
      return nearestCluster;
    }

    private int reassignPoints() {
      int changedCount = 0;

      for (int i = 0; i < points.length; i++) {
        float[] point = points[i];
        int oldCluster = assignments[i];
        int newCluster = findNearestCluster(point, centroids);

        if (newCluster != oldCluster) {
          clusterSizes[oldCluster]--;
          subInPlace(clusterSums[oldCluster], point);

          clusterSizes[newCluster]++;
          addInPlace(clusterSums[newCluster], point);

          assignments[i] = newCluster;
          changedCount++;
        }
      }
      return changedCount;
    }

    private void recalculateCentroids() {
      for (int i = 0; i < centroids.length; i++) {
        int size = clusterSizes[i];
        if (size == 0) {
          centroids[i] = points[random.nextInt(points.length)];
        } else {
          float[] sum = clusterSums[i];
          for (int j = 0; j < sum.length; j++) {
            centroids[i][j] = sum[j] / size;
          }
        }
      }
    }

    private void addInPlace(float[] target, float[] source) {
      for (int i = 0; i < target.length; i++) {
        target[i] += source[i];
      }
    }

    private void subInPlace(float[] target, float[] source) {
      for (int i = 0; i < target.length; i++) {
        target[i] -= source[i];
      }
    }

    private float distance(float[] v1, float[] v2) {
      return 1 - similarityFunction.compare(v1, v2);
    }
  }
}
