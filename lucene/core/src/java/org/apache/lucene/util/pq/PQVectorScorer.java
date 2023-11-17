package org.apache.lucene.util.pq;

import java.io.IOException;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.vamana.RandomVectorScorer;

public class PQVectorScorer implements RandomVectorScorer {

  private final VectorSimilarityFunction similarityFunction;
  private final byte[][] encoded;
  private final float[] partialSums;
  private final float[] partialMagnitudes;
  private final float queryMagnitude;

  public PQVectorScorer(ProductQuantization pq, VectorSimilarityFunction similarityFunction,
      byte[][] encoded, float[] query) {
    this.similarityFunction = similarityFunction;
    this.encoded = encoded;

    this.partialSums = new float[pq.M() * ProductQuantization.CLUSTERS];
    this.partialMagnitudes =
        similarityFunction == VectorSimilarityFunction.COSINE ? new float[pq.M()
            * ProductQuantization.CLUSTERS] : null;

    float queryMagnitude = 0.0f;

    for (int i = 0; i < pq.M(); i++) {
      int offset = pq.subvectorInfo(i).offset();
      int baseOffset = i * ProductQuantization.CLUSTERS;
      for (int j = 0; j < ProductQuantization.CLUSTERS; j++) {
        float[] centroid = pq.codebooks()[i].centroid(j);
        switch (similarityFunction) {
          case DOT_PRODUCT -> partialSums[baseOffset + j] = VectorUtil.dotProduct(centroid, 0, query, offset, centroid.length);
          case EUCLIDEAN -> partialSums[baseOffset + j] = VectorUtil.squareDistance(centroid, 0, query, offset, centroid.length);
          case COSINE -> {
            partialSums[baseOffset + j] = VectorUtil.dotProduct(centroid, 0, query, offset, centroid.length);
            partialMagnitudes[baseOffset + j] = VectorUtil.dotProduct(centroid, 0, centroid, 0, centroid.length);
          }
          default -> throw new UnsupportedOperationException("unsupported PQ similarity function " + similarityFunction);
        }
      }

      if (similarityFunction == VectorSimilarityFunction.COSINE) {
        queryMagnitude += VectorUtil.dotProduct(query, offset, query, offset, pq.subvectorInfo(i).size());
      }
    }

    this.queryMagnitude = queryMagnitude;
  }

  @Override
  public float score(int node) throws IOException {
    byte[] encoded = this.encoded[node];
    return switch (similarityFunction) {
      case COSINE, DOT_PRODUCT -> (1 + decodedSimilarity(encoded)) / 2;
      case EUCLIDEAN -> 1 / (1 + decodedSimilarity(encoded));
      default -> throw new UnsupportedOperationException("unsupported PQ similarity function " + similarityFunction);
    };
  }

  private float decodedSimilarity(byte[] encoded) {
    return switch (similarityFunction) {
      case DOT_PRODUCT, EUCLIDEAN -> VectorUtil.assembleAndSum(partialSums, ProductQuantization.CLUSTERS, encoded);
      case COSINE -> {
        float sum = 0.0f;
        float mag = 0.0f;

        for (int m = 0; m < encoded.length; ++m) {
          int centroidIndex = Byte.toUnsignedInt(encoded[m]);
          sum += partialSums[(m * ProductQuantization.CLUSTERS) + centroidIndex];
          mag += partialMagnitudes[(m * ProductQuantization.CLUSTERS) + centroidIndex];
        }

        yield (float) (sum / Math.sqrt(mag * queryMagnitude));
      }
      default -> throw new UnsupportedOperationException("unsupported PQ similarity function " + similarityFunction);
    };
  }
}
