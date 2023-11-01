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

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99Codec;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.vamana.BuildLogger;
import org.apache.lucene.util.vamana.RandomVectorScorerSupplier;
import org.apache.lucene.util.vamana.VamanaGraphBuilder;
import org.apache.lucene.util.vectors.RandomAccessVectorValues;
import org.junit.Ignore;
import org.junit.Test;

public class TestVectorSandboxVamanaVectorsFormat extends LuceneTestCase {

  private static final int NUM_VECTORS = 10000;
  private static final int VECTOR_DIMENSIONS = 4;
  private static final List<float[]> VECTORS = new ArrayList<>(NUM_VECTORS);
  private static final Random RANDOM = new Random(0);

  static {
    for (var i = 0; i < NUM_VECTORS; i++) {
      VECTORS.add(new float[VECTOR_DIMENSIONS]);
      for (var j = 0; j < VECTOR_DIMENSIONS; j++) {
        VECTORS.get(i)[j] = RANDOM.nextFloat();
      }
    }
  }

  @Test
  public void createAndInspect() throws Exception {
    var sandboxCodec =
        new Lucene99Codec() {
          @Override
          public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
            return new VectorSandboxVamanaVectorsFormat();
          }
        };
    var lucene95Codec = new Lucene99Codec();

    try (var sandboxDirectory = new ByteBuffersDirectory()) {
      try (var luceneDirectory = new ByteBuffersDirectory()) {

        // Build indexes in both directories
        var directories =
            Map.ofEntries(
                Map.entry(sandboxDirectory, sandboxCodec),
                Map.entry(luceneDirectory, lucene95Codec));

        for (var entry : directories.entrySet()) {
          var config =
              new IndexWriterConfig()
                  .setCodec(entry.getValue())
                  .setCommitOnClose(true)
                  .setUseCompoundFile(false);
          try (var writer = new IndexWriter(entry.getKey(), config)) {
            for (var vector : VECTORS) {
              var doc = new Document();
              doc.add(new KnnFloatVectorField("vector", vector));
              writer.addDocument(doc);
            }
          }
        }

        var sandboxReader = DirectoryReader.open(sandboxDirectory);
        var sandboxSearcher = new IndexSearcher(sandboxReader);
        var luceneReader = DirectoryReader.open(luceneDirectory);
        var luceneSearcher = new IndexSearcher(luceneReader);

        var sandboxLeafReader = sandboxReader.leaves().get(0).reader();
        var sandboxPerFieldVectorReader =
            (PerFieldKnnVectorsFormat.FieldsReader)
                ((CodecReader) sandboxLeafReader).getVectorReader();
        var sandboxVectorReader =
            (VectorSandboxVamanaVectorsReader) sandboxPerFieldVectorReader.getFieldReader("vector");
        var sandboxGraph = sandboxVectorReader.getGraph("vector");

        var query = new KnnFloatVectorQuery("vector", VECTORS.get(0), 5);
        var sandbox =
            Arrays.stream(sandboxSearcher.search(query, 5).scoreDocs)
                .map(scoreDoc -> scoreDoc.doc)
                .toArray();
        var lucene =
            Arrays.stream(luceneSearcher.search(query, 5).scoreDocs)
                .map(scoreDoc -> scoreDoc.doc)
                .toArray();

        System.out.println("sandbox = " + sandbox);
        System.out.println("lucene = " + lucene);
      }
    }
  }

  @Test
  public void directGraphCreate() throws Exception {
    var values = new RAVectorValues<>(VECTORS, VECTOR_DIMENSIONS);

    var builder =
        VamanaGraphBuilder.create(
            RandomVectorScorerSupplier.createFloats(values, VectorSimilarityFunction.COSINE),
//            16,
            16 * 2,
            100,
            1.2f,
            13);

    for (int i = 0; i < VECTORS.size(); i++) {
      builder.addGraphNode(i);
    }

    builder.finish(VECTORS, VectorSimilarityFunction.COSINE);
    var graph = builder.getGraph();
    BuildLogger.flush();
    System.out.println("graph = " + graph);
  }

  private static class RAVectorValues<T> implements RandomAccessVectorValues<T> {

    private final List<T> vectors;
    private final int dim;

    RAVectorValues(List<T> vectors, int dim) {
      this.vectors = vectors;
      this.dim = dim;
    }

    @Override
    public int size() {
      return vectors.size();
    }

    @Override
    public int dimension() {
      return dim;
    }

    @Override
    public T vectorValue(int targetOrd) throws IOException {
      return vectors.get(targetOrd);
    }

    @Override
    public RandomAccessVectorValues<T> copy() throws IOException {
      return this;
    }
  }
}
