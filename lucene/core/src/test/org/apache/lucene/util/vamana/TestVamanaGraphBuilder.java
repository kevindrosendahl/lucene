package org.apache.lucene.util.vamana;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99Codec;
import org.apache.lucene.codecs.vectorsandbox.VectorSandboxScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.vectorsandbox.VectorSandboxVamanaVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestVamanaGraphBuilder extends LuceneTestCase {

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
  public void createOnHeapGraph() throws Exception {
    var graph = onHeapGraph();
    System.out.println("graph = " + graph);
  }

  @Test
  public void compareQuantizedGraphs() throws Exception {
    var codec =
        new Lucene99Codec() {
          @Override
          public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
            return new VectorSandboxVamanaVectorsFormat(32, 100, 1.2f, null);
          }
        };

    var quantizedCodec =
        new Lucene99Codec() {
          @Override
          public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
            return new VectorSandboxVamanaVectorsFormat(
                32, 100, 1.2f, new VectorSandboxScalarQuantizedVectorsFormat());
          }
        };

    try (var directory = new ByteBuffersDirectory()) {
      try (var quantizedDirectory = new ByteBuffersDirectory()) {
        var config =
            new IndexWriterConfig()
                .setCodec(codec)
                .setCommitOnClose(true)
                .setUseCompoundFile(false);
        try (var writer = new IndexWriter(directory, config)) {
          for (var vector : VECTORS) {
            var doc = new Document();
            doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
            writer.addDocument(doc);
          }
        }

        var quantizedConfig =
            new IndexWriterConfig()
                .setCodec(quantizedCodec)
                .setCommitOnClose(true)
                .setUseCompoundFile(false);
        try (var writer = new IndexWriter(quantizedDirectory, quantizedConfig)) {
          for (var vector : VECTORS) {
            var doc = new Document();
            doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
            writer.addDocument(doc);
          }
        }

        var reader = DirectoryReader.open(directory);
        var searcher = new IndexSearcher(reader);
        //        var leafReader = reader.leaves().get(0).reader();
        //        var perFieldVectorReader =
        //            (PerFieldKnnVectorsFormat.FieldsReader) ((CodecReader)
        // leafReader).getVectorReader();
        //        var vectorReader =
        //            (VectorSandboxVamanaVectorsReader)
        // perFieldVectorReader.getFieldReader("vector");

        var quantizedReader = DirectoryReader.open(quantizedDirectory);
        var quantizedSearcher = new IndexSearcher(quantizedReader);
        //        var quantizedLeafReader = quantizedReader.leaves().get(0).reader();
        //        var quantizedPerFieldVectorReader =
        //            (PerFieldKnnVectorsFormat.FieldsReader)
        //                ((CodecReader) quantizedLeafReader).getVectorReader();
        //        var quantizedVectorReader =
        //            (VectorSandboxVamanaVectorsReader)
        //                quantizedPerFieldVectorReader.getFieldReader("vector");

        //        var graph = vectorReader.getGraph("vector");
        //        var quantizedGraph = quantizedVectorReader.getGraph("vector");
        //        var onHeapGraph = onHeapGraph();

        //      sandboxGraph.seek(0);
        //      System.out.println("sandboxGraph.nextNeighbor() = " + sandboxGraph.nextNeighbor());

        var query = new KnnFloatVectorQuery("vector", VECTORS.get(0), 10);
        var results = searcher.search(query, 10).scoreDocs;
        var quantizedResults = quantizedSearcher.search(query, 10).scoreDocs;
        System.out.println("results = " + results);
        System.out.println("quantizedResults = " + quantizedResults);
      }
    }
  }

  @Test
  public void compareMergedGraphs() throws Exception {
    var codec =
        new Lucene99Codec() {
          @Override
          public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
            return new VectorSandboxVamanaVectorsFormat(32, 100, 1.2f, null);
          }
        };

    try (var directory = new ByteBuffersDirectory()) {
      try (var mergedDirectory = new ByteBuffersDirectory()) {
        var config =
            new IndexWriterConfig()
                .setCodec(codec)
                .setCommitOnClose(true)
                .setUseCompoundFile(false);
        try (var writer = new IndexWriter(directory, config)) {
          int i = 0;
          for (var vector : VECTORS) {
            var doc = new Document();
            doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
            doc.add(new StoredField("_id", i++));
            writer.addDocument(doc);
          }
        }

        var mergedConfig =
            new IndexWriterConfig()
                .setCodec(codec)
                .setCommitOnClose(true)
                .setUseCompoundFile(false);
        try (var writer = new IndexWriter(mergedDirectory, mergedConfig)) {
          int i = 0;
          for (var vector : VECTORS) {
            var doc = new Document();
            doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
            writer.addDocument(doc);
            doc.add(new StoredField("_id", i++));
            writer.flush();
          }

          writer.forceMerge(1);
        }

        var reader = DirectoryReader.open(directory);
        var searcher = new IndexSearcher(reader);
        //        var leafReader = reader.leaves().get(0).reader();
        //        var perFieldVectorReader =
        //            (PerFieldKnnVectorsFormat.FieldsReader) ((CodecReader)
        // leafReader).getVectorReader();
        //        var vectorReader =
        //            (VectorSandboxVamanaVectorsReader)
        // perFieldVectorReader.getFieldReader("vector");

        var mergedReader = DirectoryReader.open(mergedDirectory);
        var mergedSearcher = new IndexSearcher(mergedReader);
        //        var mergedLeafReader = mergedReader.leaves().get(0).reader();
        //        var mergedPerFieldVectorReader =
        //            (PerFieldKnnVectorsFormat.FieldsReader)
        //                ((CodecReader) mergedLeafReader).getVectorReader();
        //        var mergedVectorReader =
        //            (VectorSandboxVamanaVectorsReader)
        // mergedPerFieldVectorReader.getFieldReader("vector");

        //        var graph = vectorReader.getGraph("vector");
        //        var mergedGraph = mergedVectorReader.getGraph("vector");
        //        var onHeapGraph = onHeapGraph();

        //      sandboxGraph.seek(0);
        //      System.out.println("sandboxGraph.nextNeighbor() = " + sandboxGraph.nextNeighbor());

        var query = new KnnFloatVectorQuery("vector", VECTORS.get(0), 10);
        var results = searcher.search(query, 10).scoreDocs;
        var mergedResults = mergedSearcher.search(query, 10).scoreDocs;
        System.out.println("results = " + results);
        System.out.println("mergedResults = " + mergedResults);
      }
    }
  }

  private OnHeapVamanaGraph onHeapGraph() throws Exception {
    RAVectorValues<float[]> values = new RAVectorValues<>(VECTORS, VECTOR_DIMENSIONS);

    var builder =
        VamanaGraphBuilder.create(
            RandomVectorScorerSupplier.createFloats(values, VectorSimilarityFunction.COSINE),
            32,
            100,
            1.2f);

    for (int i = 0; i < VECTORS.size(); i++) {
      builder.addGraphNode(i);
    }

    builder.finish();
    return builder.getGraph();
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
    public T vectorValue(int targetOrd) {
      return vectors.get(targetOrd);
    }

    @Override
    public RandomAccessVectorValues<T> copy() {
      return this;
    }
  }
}
