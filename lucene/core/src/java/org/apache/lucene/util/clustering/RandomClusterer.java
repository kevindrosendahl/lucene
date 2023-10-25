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
package org.apache.lucene.util.clustering;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/** RandomClusterer */
public class RandomClusterer implements Clusterer {

  private final Random random;

  public RandomClusterer(Random random) {
    this.random = random;
  }

  @Override
  public float[][] cluster(float[][] points, int k) {
    if (k > points.length) {
      throw new IllegalArgumentException("k cannot be greater than the number of points.");
    }

    float[][] centroids = new float[k][];
    Set<Integer> selectedIndices = new HashSet<>();

    for (int i = 0; i < k; i++) {
      int randomIndex = random.nextInt(points.length);

      while (selectedIndices.contains(randomIndex)) {
        randomIndex = random.nextInt(points.length);
      }

      selectedIndices.add(randomIndex);
      centroids[i] = points[randomIndex];
    }

    return centroids;
  }
}
