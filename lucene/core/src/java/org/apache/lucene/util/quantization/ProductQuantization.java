package org.apache.lucene.util.quantization;

import static java.lang.Math.min;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

public class ProductQuantization {

  // Cannot go above 256 since we're packing these values into a byte.
  private static final int CLUSTERS = 256;
  private static final int K_MEANS_ITERATIONS = 6;
  private static final int MAX_PQ_TRAINING_SET_SIZE = 128000;

  private final Codebook[] codebooks;
  private final int M;
  private final SubvectorInfo[] subvectorInfos;
  private final int decodedDimensionSize;
  private final VectorSimilarityFunction similarityFunction;

  private static class SubvectorInfo {

    final int size;
    final int offset;

    public SubvectorInfo(int size, int offset) {
      this.size = size;
      this.offset = offset;
    }
  }

  private static class Codebook {

    private float[][] centroids;

    public Codebook(float[][] centroids) {
      this.centroids = centroids;
    }

    public float[] getCentroid(int index) {
      return centroids[index];
    }

    public int size() {
      return centroids.length;
    }
  }


  public static ProductQuantization compute(RandomAccessVectorValues<float[]> ravv, int M,
      VectorSimilarityFunction similarityFunction,
      Random random) throws IOException {
    var trainingVectors = sampleTrainingVectors(ravv, MAX_PQ_TRAINING_SET_SIZE, random);
    var subvectorInfos = getSubvectorInfo(ravv.dimension(), M);
    var codebooks = createCodebooks(trainingVectors, M, subvectorInfos, similarityFunction, random);
    return new ProductQuantization(codebooks, subvectorInfos, similarityFunction);
  }

  private static List<float[]> sampleTrainingVectors(RandomAccessVectorValues<float[]> ravv,
      int maxTrainingSize, Random random)
      throws IOException {
    var sampleRate = min(1.0f, maxTrainingSize / (float) ravv.size());
    var vectors = new ArrayList<float[]>();
    for (int i = 0; i < ravv.size(); i++) {
      if (random.nextFloat() < sampleRate) {
        vectors.add(ravv.vectorValue(i));
      }
    }
    return vectors;
  }

  public ProductQuantization(Codebook[] codebooks, SubvectorInfo[] subvectorInfos,
      VectorSimilarityFunction similarityFunction) {
    this.codebooks = codebooks;
    this.M = codebooks.length;
    this.subvectorInfos = subvectorInfos;
    this.decodedDimensionSize = Arrays.stream(subvectorInfos).mapToInt(info -> info.size).sum();
    this.similarityFunction = similarityFunction;
  }

  public byte[] encode(float[] vector) {
    byte[] encoded = new byte[M];
    for (int m = 0; m < M; m++) {
      var subVector = getSubVector(vector, m, this.subvectorInfos);
      encoded[m] = (byte) closestCentroidIndex(subVector, codebooks[m], this.similarityFunction);
    }
    return encoded;
  }

  public float[] decode(byte[] encoded) {
    float[] target = new float[this.decodedDimensionSize];
    for (int m = 0; m < M; m++) {
      float[] centroid = codebooks[m].getCentroid(Byte.toUnsignedInt(encoded[m]));
      System.arraycopy(centroid, 0, target, this.subvectorInfos[m].offset,
          this.subvectorInfos[m].size);
    }
    return target;
  }


  private static float[] getSubVector(float[] vector, int m, SubvectorInfo[] subvectorInfos) {
    float[] subvector = new float[subvectorInfos[m].size];
    System.arraycopy(vector, subvectorInfos[m].offset, subvector, 0, subvectorInfos[m].size);
    return subvector;
  }

  private static int closestCentroidIndex(float[] subvector, Codebook codebook,
      VectorSimilarityFunction similarityFunction) {
    int closestIndex = 0;
    float closestDistance = Float.MAX_VALUE;

    for (int i = 0; i < codebook.size(); i++) {
      float distance = distance(subvector, codebook.getCentroid(i), similarityFunction);
      if (distance < closestDistance) {
        closestDistance = distance;
        closestIndex = i;
      }
    }

    return closestIndex;
  }


  private static Codebook[] createCodebooks(List<float[]> vectors, int M,
      SubvectorInfo[] subvectorInfos, VectorSimilarityFunction similarityFunction, Random random) {
    return IntStream.range(0, M)
        .mapToObj(m -> clusterSubvectors(vectors, m, subvectorInfos, similarityFunction, random))
        .map(Codebook::new)
        .toArray(Codebook[]::new);
  }

  private static float[][] clusterSubvectors(List<float[]> vectors, int m,
      SubvectorInfo[] subvectorInfos, VectorSimilarityFunction similarityFunction, Random random) {
    float[][] subvectors = vectors.stream()
        .map(v -> getSubVector(v, m, subvectorInfos))
        .toArray(float[][]::new);
    return KMeansPlusPlus.cluster(subvectors, CLUSTERS, similarityFunction,
        random, K_MEANS_ITERATIONS);
  }


  /**
   * Generates information about the division of a high-dimensional vector into subvectors. Each
   * subvector is described by its size and offset within the original vector.
   * <p>
   * - Size: The number of dimensions in the subvector. - Offset: The starting dimension in the
   * original vector for this subvector.
   * </p>
   * <p>
   * When `M` is a factor of `dimensions`: - Each subvector will have an equal size of `dimensions /
   * M`. - Offsets will be [0, size, 2*size, ..., (M-1)*size].
   * </p>
   * <p>
   * When `M` is not a factor of `dimensions`: - The base size for each subvector is `dimensions /
   * M`. - A remainder of `dimensions % M` is distributed among the first few subvectors. - Offsets
   * will be calculated based on these sizes.
   * </p>
   */
  private static SubvectorInfo[] getSubvectorInfo(int dimensions, int M) {
    SubvectorInfo[] subvectorInfos = new SubvectorInfo[M];
    int baseSize = dimensions / M;
    int remainder = dimensions % M;
    int offset = 0;
    for (int i = 0; i < M; i++) {
      int size = baseSize + (i < remainder ? 1 : 0);
      subvectorInfos[i] = new SubvectorInfo(size, offset);
      offset += size;
    }
    return subvectorInfos;
  }

  private static float distance(float[] v1, float[] v2,
      VectorSimilarityFunction similarityFunction) {
    return 1 - similarityFunction.compare(v1, v2);
  }
}