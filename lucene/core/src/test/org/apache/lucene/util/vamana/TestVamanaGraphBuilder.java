package org.apache.lucene.util.vamana;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestVamanaGraphBuilder extends LuceneTestCase {

  private static final int NUM_VECTORS = 10000;
  private static final int VECTOR_DIMENSIONS = 4;
  private static final List<float[]> VECTORS = new ArrayList<>(NUM_VECTORS);
  private static final Random RANDOM = new Random(0);

  static {
    for (var i = 0; i < NUM_VECTORS; i++) {
      VECTORS.add(new float[VECTOR_DIMENSIONS]);
      for (var j = 0; j < VECTOR_DIMENSIONS; j++) {
        VECTORS.get(i)[j] = RANDOM.nextFloat();
      }
    }
  }

  @Test
  public void createGraph() throws Exception {
    var values = new RAVectorValues<>(VECTORS, VECTOR_DIMENSIONS);

    var builder =
        VamanaGraphBuilder.create(
            RandomVectorScorerSupplier.createFloats(values, VectorSimilarityFunction.COSINE),
            32,
            100,
            1.2f);

    for (int i = 0; i < VECTORS.size(); i++) {
      builder.addGraphNode(i);
    }

    builder.finish();
    var graph = builder.getGraph();
    System.out.println("graph = " + graph);
  }

  private static class RAVectorValues<T> implements RandomAccessVectorValues<T> {

    private final List<T> vectors;
    private final int dim;

    RAVectorValues(List<T> vectors, int dim) {
      this.vectors = vectors;
      this.dim = dim;
    }

    @Override
    public int size() {
      return vectors.size();
    }

    @Override
    public int dimension() {
      return dim;
    }

    @Override
    public T vectorValue(int targetOrd) {
      return vectors.get(targetOrd);
    }

    @Override
    public RandomAccessVectorValues<T> copy() {
      return this;
    }
  }
}
