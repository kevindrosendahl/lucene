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
package org.apache.lucene.util.vamana;

import java.io.IOException;
import org.apache.lucene.codecs.HnswGraphProvider;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.InfoStream;

/**
 * This selects the biggest Hnsw graph from the provided merge state and initializes a new
 * HnswGraphBuilder with that graph as a starting point.
 *
 * @lucene.experimental
 */
public class IncrementalVamanaGraphMerger implements VamanaGraphMerger {

  protected final FieldInfo fieldInfo;
  protected final RandomVectorScorerSupplier scorerSupplier;
  protected final int M;
  protected final int beamWidth;
  protected final float alpha;

  /**
   * @param fieldInfo FieldInfo for the field being merged
   */
  public IncrementalVamanaGraphMerger(
      FieldInfo fieldInfo,
      RandomVectorScorerSupplier scorerSupplier,
      int M,
      int beamWidth,
      float alpha) {
    this.fieldInfo = fieldInfo;
    this.scorerSupplier = scorerSupplier;
    this.M = M;
    this.beamWidth = beamWidth;
    this.alpha = alpha;
  }

  /**
   * Adds a reader to the graph merger if it meets the following criteria: 1. Does not contain any
   * deleted docs 2. Is a HnswGraphProvider/PerFieldKnnVectorReader 3. Has the most docs of any
   * previous reader that met the above criteria
   */
  @Override
  public IncrementalVamanaGraphMerger addReader(
      KnnVectorsReader reader, MergeState.DocMap docMap, Bits liveDocs) throws IOException {
    KnnVectorsReader currKnnVectorsReader = reader;
    if (reader instanceof PerFieldKnnVectorsFormat.FieldsReader candidateReader) {
      currKnnVectorsReader = candidateReader.getFieldReader(fieldInfo.name);
    }

    if (!(currKnnVectorsReader instanceof HnswGraphProvider) || !noDeletes(liveDocs)) {
      return this;
    }

    int candidateVectorCount = 0;
    switch (fieldInfo.getVectorEncoding()) {
      case BYTE -> {
        ByteVectorValues byteVectorValues =
            currKnnVectorsReader.getByteVectorValues(fieldInfo.name);
        if (byteVectorValues == null) {
          return this;
        }
        candidateVectorCount = byteVectorValues.size();
      }
      case FLOAT32 -> {
        FloatVectorValues vectorValues = currKnnVectorsReader.getFloatVectorValues(fieldInfo.name);
        if (vectorValues == null) {
          return this;
        }
        candidateVectorCount = vectorValues.size();
      }
    }
    return this;
  }

  /**
   * Builds a new HnswGraphBuilder using the biggest graph from the merge state as a starting point.
   * If no valid readers were added to the merge state, a new graph is created.
   *
   * @param mergedVectorIterator iterator over the vectors in the merged segment
   * @param maxOrd max num of vectors that will be merged into the graph
   * @return HnswGraphBuilder
   * @throws IOException If an error occurs while reading from the merge state
   */
  protected VamanaBuilder createBuilder(DocIdSetIterator mergedVectorIterator, int maxOrd)
      throws IOException {
    return VamanaGraphBuilder.create(scorerSupplier, M, beamWidth, alpha, maxOrd);
  }

  @Override
  public OnHeapVamanaGraph merge(
      DocIdSetIterator mergedVectorIterator, InfoStream infoStream, int maxOrd) throws IOException {
    VamanaBuilder builder = createBuilder(mergedVectorIterator, maxOrd);
    builder.setInfoStream(infoStream);
    OnHeapVamanaGraph graph = builder.build(maxOrd);
    builder.finish();

    return graph;
  }

  private static boolean noDeletes(Bits liveDocs) {
    if (liveDocs == null) {
      return true;
    }

    for (int i = 0; i < liveDocs.length(); i++) {
      if (!liveDocs.get(i)) {
        return false;
      }
    }
    return true;
  }
}
