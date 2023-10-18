package org.apache.lucene.util.pq;

import java.util.Random;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.quantization.KMeansPlusPlus;
import org.junit.Test;

public class TestKmeansPlusPlus extends LuceneTestCase {

  @Test
  public void testCluster() {
    float[][] points = {
        {0.1f, 0.2f},
        {0.15f, 0.25f},
        {0.9f, 0.95f},
        {0.85f, 0.9f},
        {0.5f, 0.5f},
        {0.55f, 0.55f}
    };

    int k = 3;
    VectorSimilarityFunction simFunc = VectorSimilarityFunction.EUCLIDEAN;
    Random random = new Random(123);
    int maxIterations = 100;

    float[][] centroids = KMeansPlusPlus.cluster(points, k, simFunc, random, maxIterations);
    assertNotNull("Centroids should not be null", centroids);

    // There should be k centroids and they should have the same dimensionality as the original points.
    assertEquals(k, centroids.length);
    assertEquals(points[0].length, centroids[0].length);

    // Sanity check that the centroids are at least within the bounds of the original data set.
    for (float[] centroid : centroids) {
      for (int dim = 0; dim < centroid.length; dim++) {
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        for (float[] point : points) {
          if (point[dim] < min) {
            min = point[dim];
          }
          if (point[dim] > max) {
            max = point[dim];
          }
        }
        assertTrue(centroid[dim] >= min && centroid[dim] <= max);
      }
    }
  }

  @Test
  public void testDataPointToCentroidAssignment() {
    float[][] points = {
        {0.1f, 0.2f},
        {0.15f, 0.25f},
        {0.9f, 0.95f},
        {0.85f, 0.9f},
        {0.5f, 0.5f},
        {0.55f, 0.55f}
    };
    int k = 3;
    Random random = new Random();
    VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.EUCLIDEAN;

    float[][] centroids = KMeansPlusPlus.cluster(points, k, similarityFunction, random, 100);

    for (float[] point : points) {
      int nearestCluster = findNearestCluster(point, centroids, similarityFunction);
      float distance = distance(point, centroids[nearestCluster], similarityFunction);

      for (int i = 0; i < centroids.length; i++) {
        if (i != nearestCluster) {
          float otherDistance = distance(point, centroids[i], similarityFunction);
          assertTrue("point should be nearest to assigned centroid", distance <= otherDistance);
        }
      }
    }
  }

  @Test(expected = IllegalArgumentException.class)
  public void testInvalidK() {
    float[][] points = {{0.1f, 0.2f}, {0.15f, 0.25f}};
    int k = 0;
    VectorSimilarityFunction simFunc = VectorSimilarityFunction.EUCLIDEAN;
    Random random = new Random();
    int maxIterations = 100;
    KMeansPlusPlus.cluster(points, k, simFunc, random, maxIterations);
  }

  @Test
  public void testSamePointMultipleTimes() {
    float[][] points = {{0.1f, 0.1f}, {0.1f, 0.1f}, {0.1f, 0.1f}, {0.9f, 0.9f}, {0.9f, 0.9f},
        {0.9f, 0.9f}};
    int k = 2;
    VectorSimilarityFunction simFunc = VectorSimilarityFunction.EUCLIDEAN;
    Random random = new Random(123);
    int maxIterations = 100;
    float[][] centroids = KMeansPlusPlus.cluster(points, k, simFunc, random, maxIterations);

    assertEquals("Should have two centroids", 2, centroids.length);
  }

  @Test
  public void testOneCluster() {
    float[][] points = {{0.1f, 0.2f}, {0.15f, 0.25f}, {0.9f, 0.95f}, {0.85f, 0.9f}, {0.5f, 0.5f},
        {0.55f, 0.55f}};
    int k = 1;
    VectorSimilarityFunction simFunc = VectorSimilarityFunction.EUCLIDEAN;
    Random random = new Random(123);
    int maxIterations = 100;
    float[][] centroids = KMeansPlusPlus.cluster(points, k, simFunc, random, maxIterations);

    assertEquals("Should have one centroid", 1, centroids.length);
  }

  private static int findNearestCluster(float[] point, float[][] centroids,
      VectorSimilarityFunction similarityFunction) {
    float minDistance = Float.MAX_VALUE;
    int nearestCluster = 0;
    for (int i = 0; i < centroids.length; i++) {
      float distance = distance(point, centroids[i], similarityFunction);
      if (distance < minDistance) {
        minDistance = distance;
        nearestCluster = i;
      }
    }
    return nearestCluster;
  }

  private static float distance(float[] v1, float[] v2,
      VectorSimilarityFunction similarityFunction) {
    return 1 - similarityFunction.compare(v1, v2);
  }
}
