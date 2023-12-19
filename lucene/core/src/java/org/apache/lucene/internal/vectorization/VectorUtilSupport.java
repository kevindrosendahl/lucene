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

package org.apache.lucene.internal.vectorization;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Interface for implementations of VectorUtil support.
 *
 * @lucene.internal
 */
public interface VectorUtilSupport {

  /** Calculates the dot product of the given float arrays. */
  float dotProduct(float[] a, float[] b);

  float dotProduct(float[] a, int aOffset, float[] b, int bOffset, int length);

  /** Returns the cosine similarity between the two vectors. */
  float cosine(float[] v1, float[] v2);

  /** Returns the sum of squared differences of the two vectors. */
  float squareDistance(float[] a, float[] b);

  float squareDistance(MemorySegment a, MemorySegment b, int length);

  float squareDistance(float[] a, int aOffset, float[] b, int bOffset, int length);

  /** Returns the dot product computed over signed bytes. */
  int dotProduct(byte[] a, byte[] b);

  /** Returns the cosine similarity between the two byte vectors. */
  float cosine(byte[] a, byte[] b);

  /** Returns the sum of squared differences of the two byte vectors. */
  int squareDistance(byte[] a, byte[] b);

  default float assembleAndSum(float[] data, int dataBase, byte[] dataOffsets) {
    float sum = 0f;
    for (int i = 0; i < dataOffsets.length; i++) {
      sum += data[dataBase * i + Byte.toUnsignedInt(dataOffsets[i])];
    }
    return sum;
  }

  default float assembleAndSum(
      float[] data, int dataBase, MemorySegment dataOffsets, int dataOffsetsLen) {
    float sum = 0f;
    for (int i = 0; i < dataOffsetsLen; i++) {
      sum +=
          data[dataBase * i + Byte.toUnsignedInt(dataOffsets.getAtIndex(ValueLayout.JAVA_BYTE, i))];
    }
    return sum;
  }
}
