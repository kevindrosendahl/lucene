package org.apache.lucene.util.pq;

import java.util.Arrays;
import java.util.Random;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.clustering.KMeansPlusPlusClusterer;
import org.apache.lucene.util.vamana.ListRandomAccessVectorValues;
import org.junit.Test;

public class TestPQParity {

  private static final int NUM_VECTORS = 10000;
  private static final int VECTOR_DIMENSIONS = 50;
  private static final float[][] VECTORS = new float[NUM_VECTORS][];
  private static final Random RANDOM = new Random(0);

  static {
    for (var i = 0; i < NUM_VECTORS; i++) {
      VECTORS[i] = new float[VECTOR_DIMENSIONS];
      for (var j = 0; j < VECTOR_DIMENSIONS; j++) {
        VECTORS[i][j] = RANDOM.nextFloat();
      }
    }
  }

  @Test
  public void testKmeansParity() {
    var kmeans = new KMeansPlusPlusClusterer(VectorUtil::squareDistance, 6);
    float[][] clusters = kmeans.cluster(VECTORS, 64);
    System.out.println("clusters = " + Arrays.deepToString(clusters));
  }

  @Test
  public void testPQParity() throws Exception {
    var ravv = new ListRandomAccessVectorValues<>(Arrays.stream(VECTORS).toList(), VECTOR_DIMENSIONS);
    var pq = ProductQuantization.compute(ravv, 2);
    System.out.println("pq = " + pq);
  }
}
