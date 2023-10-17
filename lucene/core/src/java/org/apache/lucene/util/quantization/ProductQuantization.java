package org.apache.lucene.util.quantization;

import static java.lang.Math.min;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

public class ProductQuantization {

  static final int CLUSTERS = 256; // number of clusters per subspace = one byte's worth
  private static final int K_MEANS_ITERATIONS = 6;
  private static final int MAX_PQ_TRAINING_SET_SIZE = 128000;

  final float[][][] codebooks;
  final int M;
  private final int originalDimension;
  private final float[] globalCentroid;
  final int[][] subvectorSizesAndOffsets;

  /**
   * Initializes the codebooks by clustering the input data using Product Quantization.
   *
   * @param ravv           the vectors to quantize
   * @param M              number of subspaces
   * @param globallyCenter whether to center the vectors globally before quantization (not
   *                       recommended when using the quantization for dot product)
   */
  public static ProductQuantization compute(RandomAccessVectorValues<float[]> ravv, int M,
      boolean globallyCenter) throws IOException {
    // limit the number of vectors we train on
    var P = min(1.0f, MAX_PQ_TRAINING_SET_SIZE / (float) ravv.size());
    var subvectorSizesAndOffsets = getSubvectorSizesAndOffsets(ravv.dimension(), M);

    var size = ravv.size();
    var vectors = new ArrayList<float[]>();
    for (int i = 0; i < size; i++) {
      if (ThreadLocalRandom.current().nextFloat() < P) {
        continue;
      }

      vectors.add(ravv.vectorValue(i));
    }

    // subtract the centroid from each training vector
    float[] globalCentroid = null;
    if (globallyCenter) {
      globalCentroid = KMeansPlusPlus.centroidOf(vectors);
      var localGlobalCentroid = globalCentroid;
      vectors.forEach(vector -> VectorUtil.subInPlace(vector, localGlobalCentroid));
    }

    // derive the codebooks
    var codebooks = createCodebooks(vectors, M, subvectorSizesAndOffsets);
    return new ProductQuantization(codebooks, globalCentroid);
  }

  ProductQuantization(float[][][] codebooks, float[] globalCentroid) {
    this.codebooks = codebooks;
    this.globalCentroid = globalCentroid;
    this.M = codebooks.length;
    this.subvectorSizesAndOffsets = new int[M][];
    int offset = 0;
    for (int i = 0; i < M; i++) {
      int size = codebooks[i][0].length;
      this.subvectorSizesAndOffsets[i] = new int[]{size, offset};
      offset += size;
    }
    this.originalDimension = Arrays.stream(subvectorSizesAndOffsets).mapToInt(m -> m[0]).sum();
  }

  /**
   * Encodes the given vectors in parallel using the PQ codebooks.
   */
  public byte[][] encodeAll(List<float[]> vectors) {
    return vectors.stream().map(this::encode).toArray(byte[][]::new);
  }

  /**
   * Encodes the input vector using the PQ codebooks.
   *
   * @return one byte per subspace
   */
  public byte[] encode(float[] vector) {
    if (globalCentroid != null) {
      vector = VectorUtil.sub(vector, globalCentroid);
    }

    float[] finalVector = vector;
    byte[] encoded = new byte[M];
    for (int m = 0; m < M; m++) {
      encoded[m] = (byte) closetCentroidIndex(
          getSubVector(finalVector, m, subvectorSizesAndOffsets), codebooks[m]);
    }
    return encoded;
  }

  /**
   * Computes the cosine of the (approximate) original decoded vector with another vector.
   * <p>
   * This method can compute the cosine without materializing the decoded vector as a new float[],
   * which will be roughly 1.5x as fast as decode() + dot().
   * <p>
   * It is the caller's responsibility to center the `other` vector by subtracting the global
   * centroid before calling this method.
   */
  public float decodedCosine(byte[] encoded, float[] other) {
    float sum = 0.0f;
    float aMagnitude = 0.0f;
    float bMagnitude = 0.0f;
    for (int m = 0; m < M; ++m) {
      int offset = subvectorSizesAndOffsets[m][1];
      int centroidIndex = Byte.toUnsignedInt(encoded[m]);
      float[] centroidSubvector = codebooks[m][centroidIndex];
      var length = centroidSubvector.length;
      sum += VectorUtil.dotProduct(centroidSubvector, 0, other, offset, length);
      aMagnitude += VectorUtil.dotProduct(centroidSubvector, 0, centroidSubvector, 0, length);
      bMagnitude += VectorUtil.dotProduct(other, offset, other, offset, length);
    }

    return (float) (sum / Math.sqrt(aMagnitude * bMagnitude));
  }

  /**
   * Decodes the quantized representation (byte array) to its approximate original vector.
   */
  public void decode(byte[] encoded, float[] target) {
    decodeCentered(encoded, target);

    if (globalCentroid != null) {
      // Add back the global centroid to get the approximate original vector.
      VectorUtil.addInPlace(target, globalCentroid);
    }
  }

  /**
   * Decodes the quantized representation (byte array) to its approximate original vector, relative
   * to the global centroid.
   */
  void decodeCentered(byte[] encoded, float[] target) {
    for (int m = 0; m < M; m++) {
      int centroidIndex = Byte.toUnsignedInt(encoded[m]);
      float[] centroidSubvector = codebooks[m][centroidIndex];
      System.arraycopy(centroidSubvector, 0, target, subvectorSizesAndOffsets[m][1],
          subvectorSizesAndOffsets[m][0]);
    }
  }

  /**
   * @return The dimension of the vectors being quantized.
   */
  public int getOriginalDimension() {
    return originalDimension;
  }

  /**
   * @return how many bytes we are compressing to
   */
  public int getSubspaceCount() {
    return M;
  }

  // for testing
  static void printCodebooks(List<List<float[]>> codebooks) {
    List<List<String>> strings = codebooks.stream()
        .map(L -> L.stream()
            .map(ProductQuantization::arraySummary)
            .collect(Collectors.toList()))
        .collect(Collectors.toList());
    System.out.printf("Codebooks: [%s]%n", String.join("\n ", strings.stream()
        .map(L -> "[" + String.join(", ", L) + "]")
        .collect(Collectors.toList())));
  }

  private static String arraySummary(float[] a) {
    List<String> b = new ArrayList<>();
    for (int i = 0; i < min(4, a.length); i++) {
      b.add(String.valueOf(a[i]));
    }
    if (a.length > 4) {
      b.set(3, "... (" + a.length + ")");
    }
    return "[" + String.join(", ", b) + "]";
  }

  static float[][][] createCodebooks(List<float[]> vectors, int M, int[][] subvectorSizeAndOffset) {
    return IntStream.range(0, M)
        .mapToObj(m -> {
          float[][] subvectors = vectors.stream().parallel()
              .map(vector -> getSubVector(vector, m, subvectorSizeAndOffset))
              .toArray(float[][]::new);
          var clusterer = new KMeansPlusPlus(subvectors, CLUSTERS,
              // FIXME: not always cosine
              VectorSimilarityFunction.COSINE, new Random());
          return clusterer.cluster(K_MEANS_ITERATIONS);
        })
        .toArray(float[][][]::new);
  }

  static int closetCentroidIndex(float[] subvector, float[][] codebook) {
    int index = 0;
    float minDist = Integer.MAX_VALUE;
    for (int i = 0; i < codebook.length; i++) {
      float dist = VectorUtil.squareDistance(subvector, codebook[i]);
      if (dist < minDist) {
        minDist = dist;
        index = i;
      }
    }
    return index;
  }

  /**
   * Extracts the m-th subvector from a single vector.
   */
  static float[] getSubVector(float[] vector, int m, int[][] subvectorSizeAndOffset) {
    float[] subvector = new float[subvectorSizeAndOffset[m][0]];
    System.arraycopy(vector, subvectorSizeAndOffset[m][1], subvector, 0,
        subvectorSizeAndOffset[m][0]);
    return subvector;
  }

  /**
   * Splits the vector dimension into M subvectors of roughly equal size.
   */
  static int[][] getSubvectorSizesAndOffsets(int dimensions, int M) {
    int[][] sizes = new int[M][];
    int baseSize = dimensions / M;
    int remainder = dimensions % M;
    // distribute the remainder among the subvectors
    int offset = 0;
    for (int i = 0; i < M; i++) {
      int size = baseSize + (i < remainder ? 1 : 0);
      sizes[i] = new int[]{size, offset};
      offset += size;
    }
    return sizes;
  }

}
