package org.apache.lucene.util.vamana;

import java.io.IOException;
import java.util.List;

public class ListRandomAccessVectorValues<T> implements RandomAccessVectorValues<T> {

  private final List<T> vectors;
  private final int dim;

  public ListRandomAccessVectorValues(List<T> vectors, int dim) {
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
  public T vectorValue(int targetOrd) throws IOException {
    return vectors.get(targetOrd);
  }

  @Override
  public RandomAccessVectorValues<T> copy() throws IOException {
    return this;
  }
}
