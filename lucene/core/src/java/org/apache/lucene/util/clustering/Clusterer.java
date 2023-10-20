package org.apache.lucene.util.clustering;

public interface Clusterer {
  float[][] cluster(float[][] points, int k);
}
