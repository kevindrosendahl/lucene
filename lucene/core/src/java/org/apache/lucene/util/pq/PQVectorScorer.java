package org.apache.lucene.util.pq;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.CompletableFuture;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.iouring.IoUring;
import org.apache.lucene.util.vamana.RandomAccessVectorValues;
import org.apache.lucene.util.vamana.RandomVectorScorer;

public class PQVectorScorer implements RandomVectorScorer {

  private final VectorSimilarityFunction similarityFunction;
  private final RandomAccessVectorValues<byte[]> encodedRavv;
  private final IoUring encodedUring;
  private final long encodedOffset;
  private final float[] partialSums;
  private final float[] partialMagnitudes;
  private final float queryMagnitude;

  public PQVectorScorer(
      ProductQuantization pq,
      VectorSimilarityFunction similarityFunction,
      RandomAccessVectorValues<byte[]> encodedRavv,
      IoUring encodedUring,
      long encodedOffset,
      float[] query) {
    this.similarityFunction = similarityFunction;
    this.encodedRavv = encodedRavv;
    this.encodedUring = encodedUring;
    this.encodedOffset = encodedOffset;

    this.partialSums = new float[pq.M() * ProductQuantization.CLUSTERS];
    this.partialMagnitudes =
        similarityFunction == VectorSimilarityFunction.COSINE
            ? new float[pq.M() * ProductQuantization.CLUSTERS]
            : null;

    float queryMagnitude = 0.0f;

    for (int i = 0; i < pq.M(); i++) {
      int offset = pq.subvectorInfo(i).offset();
      int baseOffset = i * ProductQuantization.CLUSTERS;
      for (int j = 0; j < ProductQuantization.CLUSTERS; j++) {
        float[] centroid = pq.codebooks()[i].centroid(j);
        switch (similarityFunction) {
          case DOT_PRODUCT -> partialSums[baseOffset + j] =
              VectorUtil.dotProduct(centroid, 0, query, offset, centroid.length);
          case EUCLIDEAN -> partialSums[baseOffset + j] =
              VectorUtil.squareDistance(centroid, 0, query, offset, centroid.length);
          case COSINE -> {
            partialSums[baseOffset + j] =
                VectorUtil.dotProduct(centroid, 0, query, offset, centroid.length);
            partialMagnitudes[baseOffset + j] =
                VectorUtil.dotProduct(centroid, 0, centroid, 0, centroid.length);
          }
          default -> throw new UnsupportedOperationException(
              "unsupported PQ similarity function " + similarityFunction);
        }
      }

      if (similarityFunction == VectorSimilarityFunction.COSINE) {
        queryMagnitude +=
            VectorUtil.dotProduct(query, offset, query, offset, pq.subvectorInfo(i).size());
      }
    }

    this.queryMagnitude = queryMagnitude;
  }

  @Override
  public float score(int node) throws IOException {
    byte[] encoded = this.encodedRavv.vectorValue(node);
    return decodedSimilarity(encoded);
  }

  @Override
  public CompletableFuture<Float> prepareScoreAsync(int node) {
    System.out.println("preparing async score");
    var arena = Arena.ofConfined();
    int size = this.encodedRavv.dimension();
    var buffer = arena.allocate(size);
    var future = this.encodedUring.prepare(buffer, size, (long) node * size + encodedOffset);

    var scoredFuture =
        future.thenApply(
            nothing -> {
              var encoded = buffer.toArray(ValueLayout.JAVA_BYTE);
              return decodedSimilarity(encoded);
            });

    return scoredFuture.handle(
        (score, throwable) -> {
          arena.close();
          return score;
        });
  }

  @Override
  public void submitAndAwaitAsyncScores() {
    System.out.println("submitting async scores");
    this.encodedUring.submit();
    System.out.println("awaiting async scores");
    this.encodedUring.awaitAll();
    System.out.println("finished awaiting");
  }

  private float decodedSimilarity(byte[] encoded) {
    float similarity =
        switch (similarityFunction) {
          case DOT_PRODUCT, EUCLIDEAN -> VectorUtil.assembleAndSum(
              partialSums, ProductQuantization.CLUSTERS, encoded);
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
          default -> throw new UnsupportedOperationException(
              "unsupported PQ similarity function " + similarityFunction);
        };

    return switch (similarityFunction) {
      case COSINE, DOT_PRODUCT -> (1 + similarity) / 2;
      case EUCLIDEAN -> 1 / (1 + similarity);
      default -> throw new UnsupportedOperationException(
          "unsupported PQ similarity function " + similarityFunction);
    };
  }
}
