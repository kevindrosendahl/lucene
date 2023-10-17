package org.apache.lucene.util.quantization;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;

public class KMeansPlusPlus {
  private final int k;
  private final VectorSimilarityFunction similarityFunction;
  private final Random random;
  private final float[][] points;
  private final int[] assignments;
  private final float[][] centroids;
  private final int[] centroidDenoms;
  private final float[][] centroidNums;

  public KMeansPlusPlus(float[][] points, int k, VectorSimilarityFunction similarityFunction, Random random) {
    if (k <= 0) {
      throw new IllegalArgumentException("Number of clusters must be positive.");
    }
    if (k > points.length) {
      throw new IllegalArgumentException(String.format("Number of clusters %d cannot exceed number of points %d", k, points.length));
    }

    this.points = points;
    this.k = k;
    this.similarityFunction = similarityFunction;
    this.random = random;
    this.centroidDenoms = new int[k];
    this.centroidNums = new float[k][points[0].length];
    this.centroids = chooseInitialCentroids(points);
    this.assignments = new int[points.length];
    initializeAssignedPoints();
  }

  /**
   * Performs clustering on the provided set of points.
   *
   * @return an array of cluster centroids.
   */
  public float[][] cluster(int maxIterations) {
    for (int i = 0; i < maxIterations; i++) {
      int changedCount = clusterOnce();
      if (changedCount <= 0.01 * points.length) {
        break;
      }
    }
    return centroids;
  }

  // This is broken out as a separate public method to allow implementing OPQ efficiently
  public int clusterOnce() {
    updateCentroids();
    return updateAssignedPoints();
  }

  /**
   * Chooses the initial centroids for clustering.
   * The first centroid is chosen randomly from the data points. Subsequent centroids
   * are selected with a probability proportional to the square of their distance
   * to the nearest existing centroid. This ensures that the centroids are spread out
   * across the data and not initialized too closely to each other, leading to better
   * convergence and potentially improved final clusterings.
   *
   * @param points a list of points from which centroids are chosen.
   * @return an array of initial centroids.
   */
  private float[][] chooseInitialCentroids(float[][] points) {
    float[][] centroids = new float[k][];
    float[] distances = new float[points.length];
    Arrays.fill(distances, Float.MAX_VALUE);

    // Choose the first centroid randomly
    float[] firstCentroid = points[random.nextInt(points.length)];
    centroids[0] = firstCentroid;
    for (int i = 0; i < points.length; i++) {
      float distance1 = distanceFunction(points[i], firstCentroid);
      distances[i] = Math.min(distances[i], distance1);
    }

    // For each subsequent centroid
    for (int i = 1; i < k; i++) {
      float totalDistance = 0;
      for (float distance : distances) {
        totalDistance += distance;
      }

      float r = random.nextFloat() * totalDistance;
      int selectedIdx = -1;
      for (int j = 0; j < distances.length; j++) {
        r -= distances[j];
        if (r < 1e-6) {
          selectedIdx = j;
          break;
        }
      }

      if (selectedIdx == -1) {
        selectedIdx = random.nextInt(points.length);
      }

      float[] nextCentroid = points[selectedIdx];
      centroids[i] = nextCentroid;

      // Update distances, but only if the new centroid provides a closer distance
      for (int j = 0; j < points.length; j++) {
        float newDistance = distanceFunction(points[j], nextCentroid);
        distances[j] = Math.min(distances[j], newDistance);
      }
    }

    return centroids;
  }

  /**
   * Assigns points to the nearest cluster.  The results are stored as ordinals in `assignments`.
   * This method should only be called once after initial centroids are chosen.
   */
  private void initializeAssignedPoints() {
    for (int i = 0; i < points.length; i++) {
      float[] point = points[i];
      var newAssignment = getNearestCluster(point, centroids);
      centroidDenoms[newAssignment] = centroidDenoms[newAssignment] + 1;
      VectorUtil.addInPlace(centroidNums[newAssignment], point);
      assignments[i] = newAssignment;
    }
  }

  /**
   * Assigns points to the nearest cluster.  The results are stored as ordinals in `assignments`.
   * This method relies on valid assignments existing from either initializeAssignedPoints or
   * a previous invocation of this method.
   *
   * @return the number of points that changed clusters
   */
  private int updateAssignedPoints() {
    int changedCount = 0;

    for (int i = 0; i < points.length; i++) {
      float[] point = points[i];
      var oldAssignment = assignments[i];
      var newAssignment = getNearestCluster(point, centroids);

      if (newAssignment != oldAssignment) {
        centroidDenoms[oldAssignment] = centroidDenoms[oldAssignment] - 1;
        VectorUtil.subInPlace(centroidNums[oldAssignment], point);
        centroidDenoms[newAssignment] = centroidDenoms[newAssignment] + 1;
        VectorUtil.addInPlace(centroidNums[newAssignment], point);
        assignments[i] = newAssignment;
        changedCount++;
      }
    }

    return changedCount;
  }

  /**
   * Return the index of the closest centroid to the given point
   */
  private int getNearestCluster(float[] point, float[][] centroids) {
    float minDistance = Float.MAX_VALUE;
    int nearestCluster = 0;

    for (int i = 0; i < k; i++) {
      float distance = distanceFunction(point, centroids[i]);
      if (distance < minDistance) {
        minDistance = distance;
        nearestCluster = i;
      }
    }

    return nearestCluster;
  }

  /**
   * Calculates centroids from centroidNums/centroidDenoms updated during point assignment
   */
  private void updateCentroids() {
    for (int i = 0; i < centroids.length; i++) {
      var denom = centroidDenoms[i];
      if (denom == 0) {
        centroids[i] = points[random.nextInt(points.length)];
      } else {
        centroids[i] = Arrays.copyOf(centroidNums[i], centroidNums[i].length);
        VectorUtil.divInPlace(centroids[i], centroidDenoms[i]);
      }
    }
  }

   private float distanceFunction(float[] v1, float[] v2) {
    return 1 - this.similarityFunction.compare(v1, v2);
   }

  /**
   * Computes the centroid of a list of points.
   */
  public static float[] centroidOf(List<float[]> points) {
    if (points.isEmpty()) {
      throw new IllegalArgumentException("Can't compute centroid of empty points list");
    }

    float[] centroid = VectorUtil.sum(points);
    VectorUtil.divInPlace(centroid, points.size());

    return centroid;
  }
}
