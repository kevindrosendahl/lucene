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

package org.apache.lucene.util;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.Arrays;
import java.util.Base64;

/** Utilities for computations with numeric arrays */
@SuppressWarnings("unused")
public final class VectorUtil {

  // org.apache.lucene.util.VectorUtilSIMD#dotProduct(float[], float[])
  private static final String SIMD_BASE64 =
      "yv66vgAAADwAbQoAAgADBwAEDAAFAAYBABBqYXZhL2xhbmcvT2JqZWN0AQAGPGluaXQ+AQADKClW\n"
          + "BwAIAQAiamF2YS9sYW5nL0lsbGVnYWxBcmd1bWVudEV4Y2VwdGlvbhIAAAAKDAALAAwBABdtYWtl\n"
          + "Q29uY2F0V2l0aENvbnN0YW50cwEAFihJSSlMamF2YS9sYW5nL1N0cmluZzsKAAcADgwABQAPAQAV\n"
          + "KExqYXZhL2xhbmcvU3RyaW5nOylWCQARABIHABMMABQAFQEAJW9yZy9hcGFjaGUvbHVjZW5lL3V0\n"
          + "aWwvVmVjdG9yVXRpbFNJTUQBAAdTUEVDSUVTAQAkTGpkay9pbmN1YmF0b3IvdmVjdG9yL1ZlY3Rv\n"
          + "clNwZWNpZXM7CwAXABgHABkMABoAGwEAImpkay9pbmN1YmF0b3IvdmVjdG9yL1ZlY3RvclNwZWNp\n"
          + "ZXMBAAZsZW5ndGgBAAMoKUkKAB0AHgcAHwwAIAAhAQAgamRrL2luY3ViYXRvci92ZWN0b3IvRmxv\n"
          + "YXRWZWN0b3IBAAR6ZXJvAQBIKExqZGsvaW5jdWJhdG9yL3ZlY3Rvci9WZWN0b3JTcGVjaWVzOylM\n"
          + "amRrL2luY3ViYXRvci92ZWN0b3IvRmxvYXRWZWN0b3I7CwAXACMMACQAJQEACWxvb3BCb3VuZAEA\n"
          + "BChJKUkKAB0AJwwAKAApAQAJZnJvbUFycmF5AQBLKExqZGsvaW5jdWJhdG9yL3ZlY3Rvci9WZWN0\n"
          + "b3JTcGVjaWVzO1tGSSlMamRrL2luY3ViYXRvci92ZWN0b3IvRmxvYXRWZWN0b3I7CgAdACsMACwA\n"
          + "LQEAA211bAEAQShMamRrL2luY3ViYXRvci92ZWN0b3IvVmVjdG9yOylMamRrL2luY3ViYXRvci92\n"
          + "ZWN0b3IvRmxvYXRWZWN0b3I7CgAdAC8MADAALQEAA2FkZAkAMgAzBwA0DAA1ADYBACRqZGsvaW5j\n"
          + "dWJhdG9yL3ZlY3Rvci9WZWN0b3JPcGVyYXRvcnMBAANBREQBADJMamRrL2luY3ViYXRvci92ZWN0\n"
          + "b3IvVmVjdG9yT3BlcmF0b3JzJEFzc29jaWF0aXZlOwoAHQA4DAA5ADoBAAtyZWR1Y2VMYW5lcwEA\n"
          + "NShMamRrL2luY3ViYXRvci92ZWN0b3IvVmVjdG9yT3BlcmF0b3JzJEFzc29jaWF0aXZlOylGCQAd\n"
          + "ADwMAD0AFQEAEVNQRUNJRVNfUFJFRkVSUkVEAQAJU2lnbmF0dXJlAQA3TGpkay9pbmN1YmF0b3Iv\n"
          + "dmVjdG9yL1ZlY3RvclNwZWNpZXM8TGphdmEvbGFuZy9GbG9hdDs+OwEABENvZGUBAA9MaW5lTnVt\n"
          + "YmVyVGFibGUBABJMb2NhbFZhcmlhYmxlVGFibGUBAAR0aGlzAQAnTG9yZy9hcGFjaGUvbHVjZW5l\n"
          + "L3V0aWwvVmVjdG9yVXRpbFNJTUQ7AQAKZG90UHJvZHVjdAEAByhbRltGKUYBAAJ2YQEAIkxqZGsv\n"
          + "aW5jdWJhdG9yL3ZlY3Rvci9GbG9hdFZlY3RvcjsBAAJ2YgEAAnZjAQACdmQBAARhY2MxAQAEYWNj\n"
          + "MgEACnVwcGVyQm91bmQBAAFJAQABYQEAAltGAQABYgEAAWkBAANyZXMBAAFGAQANU3RhY2tNYXBU\n"
          + "YWJsZQcAUQEACDxjbGluaXQ+AQAKU291cmNlRmlsZQEAE1ZlY3RvclV0aWxTSU1ELmphdmEBABBC\n"
          + "b290c3RyYXBNZXRob2RzDwYAXQoAXgBfBwBgDAALAGEBACRqYXZhL2xhbmcvaW52b2tlL1N0cmlu\n"
          + "Z0NvbmNhdEZhY3RvcnkBAJgoTGphdmEvbGFuZy9pbnZva2UvTWV0aG9kSGFuZGxlcyRMb29rdXA7\n"
          + "TGphdmEvbGFuZy9TdHJpbmc7TGphdmEvbGFuZy9pbnZva2UvTWV0aG9kVHlwZTtMamF2YS9sYW5n\n"
          + "L1N0cmluZztbTGphdmEvbGFuZy9PYmplY3Q7KUxqYXZhL2xhbmcvaW52b2tlL0NhbGxTaXRlOwgA\n"
          + "YwEAHnZlY3RvciBkaW1lbnNpb25zIGRpZmZlcjogASE9AQEADElubmVyQ2xhc3NlcwcAZgEAMGpk\n"
          + "ay9pbmN1YmF0b3IvdmVjdG9yL1ZlY3Rvck9wZXJhdG9ycyRBc3NvY2lhdGl2ZQEAC0Fzc29jaWF0\n"
          + "aXZlBwBpAQAlamF2YS9sYW5nL2ludm9rZS9NZXRob2RIYW5kbGVzJExvb2t1cAcAawEAHmphdmEv\n"
          + "bGFuZy9pbnZva2UvTWV0aG9kSGFuZGxlcwEABkxvb2t1cAAhABEAAgAAAAEAGgAUABUAAQA+AAAA\n"
          + "AgA/AAMAAQAFAAYAAQBAAAAALwABAAEAAAAFKrcAAbEAAAACAEEAAAAGAAEAAAAZAEIAAAAMAAEA\n"
          + "AAAFAEMARAAAAAkARQBGAAEAQAAAAewABAALAAAA6Cq+K76fABS7AAdZKr4rvroACQAAtwANvwM9\n"
          + "C0YqvgWyABC5ABYBAGihAKiyABC4ABw6BLIAELgAHDoFsgAQKr6yABC5ABYBAGS5ACICADYGHBUG\n"
          + "ogBpsgAQKhy4ACY6B7IAECscuAAmOggZBBkHGQi2ACq2AC46BLIAECocsgAQuQAWAQBguAAmOgmy\n"
          + "ABArHLIAELkAFgEAYLgAJjoKGQUZCRkKtgAqtgAuOgUcBbIAELkAFgEAaGA9p/+XJRkEsgAxtgA3\n"
          + "GQWyADG2ADdiYkYcKr6iABMlKxwwKhwwamJGhAIBp//tJa4AAAADAEEAAABWABUAAAAhAAcAIgAY\n"
          + "ACQAGgAlABwAKAArACkAMwAqADsAKwBQACwAVgAtAGAALgBqAC8AeAAwAIsAMQCeADIArAAsALwA\n"
          + "NADQADYA1gA3AOAANgDmADkAQgAAAHAACwBgAEwARwBIAAcAagBCAEkASAAIAIsAIQBKAEgACQCe\n"
          + "AA4ASwBIAAoAMwCdAEwASAAEADsAlQBNAEgABQBQAIAATgBPAAYAAADoAFAAUQAAAAAA6ABSAFEA\n"
          + "AQAaAM4AUwBPAAIAHADMAFQAVQADAFYAAAAgAAUY/wA3AAcHAFcHAFcBAgcAHQcAHQEAAPsAa/gA\n"
          + "ExUACABYAAYAAQBAAAAAHwABAAAAAAAHsgA7swAQsQAAAAEAQQAAAAYAAQAAABoAAwBZAAAAAgBa\n"
          + "AFsAAAAIAAEAXAABAGIAZAAAABIAAgBlADIAZwYJAGgAagBsABk=";

  private static final MethodHandle DOTPRODUCT;
  private static final MethodType DOTPRODUCT_TYPE =
      MethodType.methodType(float.class, float[].class, float[].class);

  static final class Loader extends ClassLoader {
    Loader(ClassLoader parent) {
      super(parent);
    }

    public Class<?> define(byte[] code) {
      return defineClass("org.apache.lucene.util.VectorUtilSIMD", code, 0, code.length);
    }
  }

  /**
   * True if vectorized dot product is supported.
   *
   * <p>For this to work, you need java 16, and you need to opt-in by passing {@code --add-modules
   * jdk.incubator.vector} to the java command line.
   */
  public static final boolean DOTPRODUCT_VECTORIZATION_SUPPORTED;

  static {
    MethodHandle impl = null;
    boolean vectorSupported = false;

    try {
      impl =
          MethodHandles.lookup().findStatic(VectorUtil.class, "dotProductScalar", DOTPRODUCT_TYPE);
    } catch (NoSuchMethodException | IllegalAccessException e) {
      throw new RuntimeException(e);
    }

    try {
      Class<?> clazz =
          new Loader(VectorUtil.class.getClassLoader())
              .define(Base64.getMimeDecoder().decode(SIMD_BASE64));
      System.out.println("clazz.getMethods() = " + Arrays.toString(clazz.getMethods()));
      impl = MethodHandles.lookup().findStatic(clazz, "dotProduct", DOTPRODUCT_TYPE);
      vectorSupported = true;
    } catch (Throwable e) {
      System.out.println("e = " + e);
    }

    DOTPRODUCT = impl;
    DOTPRODUCT_VECTORIZATION_SUPPORTED = vectorSupported;
  }

  private VectorUtil() {}

  /**
   * Returns the vector dot product of the two vectors.
   *
   * @throws IllegalArgumentException if the vectors' dimensions differ.
   */
  public static float dotProduct(float[] a, float[] b) {
    try {
      return (float) DOTPRODUCT.invokeExact(a, b);
    } catch (RuntimeException e) {
      throw e;
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
  }

  public static float dotProductScalar(float[] a, float[] b) {
    if (a.length != b.length) {
      throw new IllegalArgumentException("vector dimensions differ: " + a.length + "!=" + b.length);
    }
    float res = 0f;
    /*
     * If length of vector is larger than 8, we use unrolled dot product to accelerate the
     * calculation.
     */
    int i;
    for (i = 0; i < a.length % 8; i++) {
      res += b[i] * a[i];
    }
    if (a.length < 8) {
      return res;
    }
    for (; i + 31 < a.length; i += 32) {
      res +=
          b[i + 0] * a[i + 0]
              + b[i + 1] * a[i + 1]
              + b[i + 2] * a[i + 2]
              + b[i + 3] * a[i + 3]
              + b[i + 4] * a[i + 4]
              + b[i + 5] * a[i + 5]
              + b[i + 6] * a[i + 6]
              + b[i + 7] * a[i + 7];
      res +=
          b[i + 8] * a[i + 8]
              + b[i + 9] * a[i + 9]
              + b[i + 10] * a[i + 10]
              + b[i + 11] * a[i + 11]
              + b[i + 12] * a[i + 12]
              + b[i + 13] * a[i + 13]
              + b[i + 14] * a[i + 14]
              + b[i + 15] * a[i + 15];
      res +=
          b[i + 16] * a[i + 16]
              + b[i + 17] * a[i + 17]
              + b[i + 18] * a[i + 18]
              + b[i + 19] * a[i + 19]
              + b[i + 20] * a[i + 20]
              + b[i + 21] * a[i + 21]
              + b[i + 22] * a[i + 22]
              + b[i + 23] * a[i + 23];
      res +=
          b[i + 24] * a[i + 24]
              + b[i + 25] * a[i + 25]
              + b[i + 26] * a[i + 26]
              + b[i + 27] * a[i + 27]
              + b[i + 28] * a[i + 28]
              + b[i + 29] * a[i + 29]
              + b[i + 30] * a[i + 30]
              + b[i + 31] * a[i + 31];
    }
    for (; i + 7 < a.length; i += 8) {
      res +=
          b[i + 0] * a[i + 0]
              + b[i + 1] * a[i + 1]
              + b[i + 2] * a[i + 2]
              + b[i + 3] * a[i + 3]
              + b[i + 4] * a[i + 4]
              + b[i + 5] * a[i + 5]
              + b[i + 6] * a[i + 6]
              + b[i + 7] * a[i + 7];
    }
    return res;
  }

  /**
   * Returns the cosine similarity between the two vectors.
   *
   * @throws IllegalArgumentException if the vectors' dimensions differ.
   */
  public static float cosine(float[] v1, float[] v2) {
    if (v1.length != v2.length) {
      throw new IllegalArgumentException(
          "vector dimensions differ: " + v1.length + "!=" + v2.length);
    }

    float sum = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    int dim = v1.length;

    for (int i = 0; i < dim; i++) {
      float elem1 = v1[i];
      float elem2 = v2[i];
      sum += elem1 * elem2;
      norm1 += elem1 * elem1;
      norm2 += elem2 * elem2;
    }
    return (float) (sum / Math.sqrt(norm1 * norm2));
  }

  /** Returns the cosine similarity between the two vectors. */
  public static float cosine(byte[] a, byte[] b) {
    // Note: this will not overflow if dim < 2^18, since max(byte * byte) = 2^14.
    int sum = 0;
    int norm1 = 0;
    int norm2 = 0;

    for (int i = 0; i < a.length; i++) {
      byte elem1 = a[i];
      byte elem2 = b[i];
      sum += elem1 * elem2;
      norm1 += elem1 * elem1;
      norm2 += elem2 * elem2;
    }
    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
  }

  /**
   * Returns the sum of squared differences of the two vectors.
   *
   * @throws IllegalArgumentException if the vectors' dimensions differ.
   */
  public static float squareDistance(float[] v1, float[] v2) {
    if (v1.length != v2.length) {
      throw new IllegalArgumentException(
          "vector dimensions differ: " + v1.length + "!=" + v2.length);
    }
    float squareSum = 0.0f;
    int dim = v1.length;
    int i;
    for (i = 0; i + 8 <= dim; i += 8) {
      squareSum += squareDistanceUnrolled(v1, v2, i);
    }
    for (; i < dim; i++) {
      float diff = v1[i] - v2[i];
      squareSum += diff * diff;
    }
    return squareSum;
  }

  private static float squareDistanceUnrolled(float[] v1, float[] v2, int index) {
    float diff0 = v1[index + 0] - v2[index + 0];
    float diff1 = v1[index + 1] - v2[index + 1];
    float diff2 = v1[index + 2] - v2[index + 2];
    float diff3 = v1[index + 3] - v2[index + 3];
    float diff4 = v1[index + 4] - v2[index + 4];
    float diff5 = v1[index + 5] - v2[index + 5];
    float diff6 = v1[index + 6] - v2[index + 6];
    float diff7 = v1[index + 7] - v2[index + 7];
    return diff0 * diff0
        + diff1 * diff1
        + diff2 * diff2
        + diff3 * diff3
        + diff4 * diff4
        + diff5 * diff5
        + diff6 * diff6
        + diff7 * diff7;
  }

  /** Returns the sum of squared differences of the two vectors. */
  public static float squareDistance(byte[] a, byte[] b) {
    // Note: this will not overflow if dim < 2^18, since max(byte * byte) = 2^14.
    int squareSum = 0;
    for (int i = 0; i < a.length; i++) {
      int diff = a[i] - b[i];
      squareSum += diff * diff;
    }
    return squareSum;
  }

  /**
   * Modifies the argument to be unit length, dividing by its l2-norm. IllegalArgumentException is
   * thrown for zero vectors.
   *
   * @return the input array after normalization
   */
  public static float[] l2normalize(float[] v) {
    l2normalize(v, true);
    return v;
  }

  /**
   * Modifies the argument to be unit length, dividing by its l2-norm.
   *
   * @param v the vector to normalize
   * @param throwOnZero whether to throw an exception when <code>v</code> has all zeros
   * @return the input array after normalization
   * @throws IllegalArgumentException when the vector is all zero and throwOnZero is true
   */
  public static float[] l2normalize(float[] v, boolean throwOnZero) {
    double squareSum = 0.0f;
    int dim = v.length;
    for (float x : v) {
      squareSum += x * x;
    }
    if (squareSum == 0) {
      if (throwOnZero) {
        throw new IllegalArgumentException("Cannot normalize a zero-length vector");
      } else {
        return v;
      }
    }
    double length = Math.sqrt(squareSum);
    for (int i = 0; i < dim; i++) {
      v[i] /= length;
    }
    return v;
  }

  /**
   * Adds the second argument to the first
   *
   * @param u the destination
   * @param v the vector to add to the destination
   */
  public static void add(float[] u, float[] v) {
    for (int i = 0; i < u.length; i++) {
      u[i] += v[i];
    }
  }

  /**
   * Dot product computed over signed bytes.
   *
   * @param a bytes containing a vector
   * @param b bytes containing another vector, of the same dimension
   * @return the value of the dot product of the two vectors
   */
  public static float dotProduct(byte[] a, byte[] b) {
    assert a.length == b.length;
    int total = 0;
    for (int i = 0; i < a.length; i++) {
      total += a[i] * b[i];
    }
    return total;
  }

  /**
   * Dot product score computed over signed bytes, scaled to be in [0, 1].
   *
   * @param a bytes containing a vector
   * @param b bytes containing another vector, of the same dimension
   * @return the value of the similarity function applied to the two vectors
   */
  public static float dotProductScore(byte[] a, byte[] b) {
    // divide by 2 * 2^14 (maximum absolute value of product of 2 signed bytes) * len
    float denom = (float) (a.length * (1 << 15));
    return 0.5f + dotProduct(a, b) / denom;
  }

  /**
   * Convert a floating point vector to an array of bytes using casting; the vector values should be
   * in [-128,127]
   *
   * @param vector a vector
   * @return a new BytesRef containing the vector's values cast to byte.
   */
  public static BytesRef toBytesRef(float[] vector) {
    BytesRef b = new BytesRef(new byte[vector.length]);
    for (int i = 0; i < vector.length; i++) {
      if (vector[i] < -128 || vector[i] > 127) {
        throw new IllegalArgumentException(
            "Vector value at " + i + " is out of range [-128.127]: " + vector[i]);
      }
      b.bytes[i] = (byte) vector[i];
    }
    return b;
  }
}
