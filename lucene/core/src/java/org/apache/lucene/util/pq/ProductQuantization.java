/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.util.pq;

import static java.lang.Math.min;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.clustering.KMeansPlusPlusClusterer;
import org.apache.lucene.util.vamana.RandomAccessVectorValues;

/** ProductQuantization */
public class ProductQuantization {

  //  private static final Random RANDOM = new Random(0);
  private static final ForkJoinPool FORK_JOIN_POOL =
      new ForkJoinPool(Runtime.getRuntime().availableProcessors());

  // TODO: consider normalizing around the global centroid for euclidean
  public static ProductQuantization compute(RandomAccessVectorValues<float[]> ravv, int M)
      throws IOException {
    return compute(ravv, M, new Random(0));
  }

  public static ProductQuantization compute(
      RandomAccessVectorValues<float[]> ravv, int M, Random random) throws IOException {
    var trainingVectors = sampleTrainingVectors(ravv, MAX_PQ_TRAINING_SET_SIZE, random);
    var subvectorInfos = getSubvectorInfo(ravv.dimension(), M);
    var codebooks = createCodebooks(trainingVectors, M, subvectorInfos, random);
    return new ProductQuantization(codebooks, subvectorInfos);
  }

  public static ProductQuantization fromCodebooks(
      Codebook[] codebooks, int dimensions, int M, VectorSimilarityFunction similarityFunction) {
    var subvectorInfos = getSubvectorInfo(dimensions, M);
    return new ProductQuantization(codebooks, subvectorInfos);
  }

  // Cannot go above 256 since we're packing these values into a byte.
  public static final int CLUSTERS = 256;
  private static final int K_MEANS_ITERATIONS = 6;
  private static final int MAX_PQ_TRAINING_SET_SIZE = 128000;

  private final Codebook[] codebooks;
  private final int M;
  private final SubvectorInfo[] subvectorInfos;
  private final int decodedDimensionSize;

  public record SubvectorInfo(int size, int offset) {}

  public static class Codebook {

    public float[][] centroids;

    public Codebook(float[][] centroids) {
      this.centroids = centroids;
    }

    public float[] centroid(int index) {
      return centroids[index];
    }

    public int size() {
      return centroids.length;
    }

    @Override
    public int hashCode() {
      return Objects.hash(Arrays.deepHashCode(centroids));
    }
  }

  private static List<float[]> sampleTrainingVectors(
      RandomAccessVectorValues<float[]> ravv, int maxTrainingSize, Random random)
      throws IOException {
    var sampleRate = min(1.0f, maxTrainingSize / (float) ravv.size());
    var vectors = new ArrayList<float[]>();
    for (int i = 0; i < ravv.size(); i++) {
      if (random.nextFloat() < sampleRate) {
        float[] copy = new float[ravv.dimension()];
        float[] vector = ravv.vectorValue(i);
        System.arraycopy(vector, 0, copy, 0, vector.length);
        vectors.add(copy);
      }
    }
    return vectors;
  }

  private ProductQuantization(Codebook[] codebooks, SubvectorInfo[] subvectorInfos) {
    this.codebooks = codebooks;
    this.M = codebooks.length;
    this.subvectorInfos = subvectorInfos;
    this.decodedDimensionSize = Arrays.stream(subvectorInfos).mapToInt(info -> info.size).sum();
  }

  public Codebook[] codebooks() {
    return this.codebooks;
  }

  public int M() {
    return this.M;
  }

  public byte[] encode(float[] vector) {
    byte[] encoded = new byte[M];
    for (int m = 0; m < M; m++) {
      var subVector = getSubVector(vector, m, this.subvectorInfos);
      encoded[m] = (byte) closestCentroidIndex(subVector, codebooks[m]);
    }
    return encoded;
  }

  public float[] decode(byte[] encoded) {
    float[] target = new float[this.decodedDimensionSize];
    for (int m = 0; m < M; m++) {
      float[] centroid = codebooks[m].centroid(Byte.toUnsignedInt(encoded[m]));
      System.arraycopy(
          centroid, 0, target, this.subvectorInfos[m].offset, this.subvectorInfos[m].size);
    }
    return target;
  }

  public SubvectorInfo subvectorInfo(int subvector) {
    return this.subvectorInfos[subvector];
  }

  private static float[] getSubVector(float[] vector, int m, SubvectorInfo[] subvectorInfos) {
    float[] subvector = new float[subvectorInfos[m].size];
    System.arraycopy(vector, subvectorInfos[m].offset, subvector, 0, subvectorInfos[m].size);
    return subvector;
  }

  private static int closestCentroidIndex(float[] subvector, Codebook codebook) {
    int closestIndex = 0;
    float closestDistance = Float.MAX_VALUE;

    for (int i = 0; i < codebook.size(); i++) {
      float distance = distance(subvector, codebook.centroid(i));
      if (distance < closestDistance) {
        closestDistance = distance;
        closestIndex = i;
      }
    }

    return closestIndex;
  }

  private static Codebook[] createCodebooks(
      List<float[]> vectors, int M, SubvectorInfo[] subvectorInfos, Random random) {
    return FORK_JOIN_POOL
        .submit(
            () ->
                IntStream.range(0, M)
                    .parallel()
                    .mapToObj(m -> clusterSubvectors(vectors, m, subvectorInfos))
                    .map(Codebook::new)
                    .toArray(Codebook[]::new))
        .join();
  }

  private static float[][] clusterSubvectors(
      List<float[]> vectors, int m, SubvectorInfo[] subvectorInfos) {
    float[][] subvectors =
        vectors.stream().map(v -> getSubVector(v, m, subvectorInfos)).toArray(float[][]::new);
    var clusterer = new KMeansPlusPlusClusterer(VectorUtil::squareDistance, K_MEANS_ITERATIONS);
    return clusterer.cluster(subvectors, CLUSTERS);
  }

  /**
   * Generates information about the division of a high-dimensional vector into subvectors. Each
   * subvector is described by its size and offset within the original vector.
   *
   * <p>
   * <li>Size: The number of dimensions in the subvector.
   * <li>Offset: The starting dimension in the original vector for this subvector.
   *
   *     <p>When `M` is a factor of `dimensions`:
   * <li>Each subvector will have an equal size of `dimensions / M`.
   * <li>Offsets will be [0, size, 2*size, ..., (M-1)*size].
   *
   *     <p>When `M` is not a factor of `dimensions`:
   * <li>The base size for each subvector is `dimensions /M`.
   * <li>A remainder of `dimensions % M` is distributed among the first few subvectors.
   * <li>Offsets will be calculated based on these sizes.
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

  private static float distance(float[] v1, float[] v2) {
    return VectorUtil.squareDistance(v1, v2);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        Arrays.hashCode(codebooks), M, Arrays.hashCode(subvectorInfos), decodedDimensionSize);
  }
}
