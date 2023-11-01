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

package org.apache.lucene.codecs.vectorsandbox;

/**
 * Format supporting vector quantization, storage, and retrieval
 *
 * @lucene.experimental
 */
public final class VectorSandboxScalarQuantizedVectorsFormat {
  public static final String QUANTIZED_VECTOR_COMPONENT = "QVEC";

  static final String NAME = "VectorSandboxScalarQuantizedVectorsFormat";

  static final int VERSION_START = 0;
  static final int VERSION_CURRENT = VERSION_START;
  static final String QUANTIZED_VECTOR_DATA_CODEC_NAME = "VectorSandboxScalarQuantizedVectorsData";
  static final String QUANTIZED_VECTOR_DATA_EXTENSION = "veq";

  /** The minimum quantile */
  private static final float MINIMUM_QUANTILE = 0.9f;

  /** The maximum quantile */
  private static final float MAXIMUM_QUANTILE = 1f;

  /**
   * Controls the quantile used to scalar quantize the vectors the default quantile is calculated as
   * `1-1/(vector_dimensions + 1)`
   */
  final Float quantile;

  /** Constructs a format using default graph construction parameters */
  public VectorSandboxScalarQuantizedVectorsFormat() {
    this(null);
  }

  /**
   * Constructs a format using the given graph construction parameters.
   *
   * @param quantile the quantile for scalar quantizing the vectors, when `null` it is calculated
   *     based on the vector field dimensions.
   */
  public VectorSandboxScalarQuantizedVectorsFormat(Float quantile) {
    if (quantile != null && (quantile < MINIMUM_QUANTILE || quantile > MAXIMUM_QUANTILE)) {
      throw new IllegalArgumentException(
          "quantile must be between "
              + MINIMUM_QUANTILE
              + " and "
              + MAXIMUM_QUANTILE
              + "; quantile="
              + quantile);
    }
    this.quantile = quantile;
  }

  static float calculateDefaultQuantile(int vectorDimension) {
    return Math.max(MINIMUM_QUANTILE, 1f - (1f / (vectorDimension + 1)));
  }

  @Override
  public String toString() {
    return NAME + "(name=" + NAME + ", quantile=" + quantile + ")";
  }
}
