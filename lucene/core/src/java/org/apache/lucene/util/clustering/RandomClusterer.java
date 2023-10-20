package org.apache.lucene.util.clustering;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class RandomClusterer implements Clusterer {

  private final Random random;

  public RandomClusterer(Random random) {
    this.random = random;
  }

  @Override
  public float[][] cluster(float[][] points, int k) {
    if (k > points.length) {
      throw new IllegalArgumentException("k cannot be greater than the number of points.");
    }

    float[][] centroids = new float[k][];
    Set<Integer> selectedIndices = new HashSet<>();

    for (int i = 0; i < k; i++) {
      int randomIndex = random.nextInt(points.length);

      while (selectedIndices.contains(randomIndex)) {
        randomIndex = random.nextInt(points.length);
      }

      selectedIndices.add(randomIndex);
      centroids[i] = points[randomIndex];
    }

    return centroids;
  }
}
