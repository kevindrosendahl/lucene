package org.apache.lucene.util.qantization;

import java.util.Arrays;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.quantization.ProductQuantization;
import org.junit.Test;

public class TestProductQuantization extends LuceneTestCase {

  @Test
  public void testReconstruction() throws Exception {
    // If there are less vectors than the number of clusters they should reconstruct
    var random = random();

    var numVectors = 256;
    var dimensions = 128;
    float[][] vectors = new float[numVectors][dimensions];
    for (int i = 0; i < numVectors; i++) {
      for (int j = 0; j < dimensions; j++) {
        vectors[i][j] = random.nextFloat();
      }
    }

    var ravv = MockVectorValues.fromValues(vectors);
    var pq = ProductQuantization.compute(ravv, 2, VectorSimilarityFunction.EUCLIDEAN, random);
    var encoded = Arrays.stream(vectors).map(pq::encode).toList();

    for (int i = 0; i < numVectors; i++) {
      var original = vectors[i];
      var decoded = pq.decode(encoded.get(i));
      assertArrayEquals(Arrays.toString(original) + "!=" + Arrays.toString(decoded), original,
          decoded, 0);
    }
  }

}
