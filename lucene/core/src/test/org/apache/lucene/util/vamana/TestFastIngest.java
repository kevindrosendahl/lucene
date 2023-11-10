package org.apache.lucene.util.vamana;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99Codec;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.codecs.vectorsandbox.VectorSandboxFastIngestVectorsFormat;
import org.apache.lucene.codecs.vectorsandbox.VectorSandboxFastIngestVectorsReader;
import org.apache.lucene.codecs.vectorsandbox.VectorSandboxVamanaVectorsFormat;
import org.apache.lucene.codecs.vectorsandbox.VectorSandboxVamanaVectorsReader;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Ignore;
import org.junit.Test;

public class TestFastIngest extends LuceneTestCase {

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
  @Ignore
  public void createOnHeapGraph() throws Exception {
    var graph = onHeapGraph();
    System.out.println("graph = " + graph);
  }

  @Test
  @Ignore
  public void compareQuantizedGraphs() throws Exception {
    var codec =
        new Lucene99Codec() {
          @Override
          public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
            return new VectorSandboxVamanaVectorsFormat(64, 100, 1.2f, null);
          }
        };

    var ingestCodec =
        new Lucene99Codec() {
          @Override
          public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
            return new VectorSandboxFastIngestVectorsFormat(
                //            new VectorSandboxVamanaVectorsFormat(32, 100, 1.2f, null));
                new Lucene99HnswVectorsFormat(32, 100));
          }
        };

    try (var directory = new ByteBuffersDirectory()) {
      try (var ingestDirectory = new ByteBuffersDirectory()) {
        var config =
            new IndexWriterConfig()
                .setCodec(codec)
                .setCommitOnClose(true)
                .setUseCompoundFile(false)
                .setMergeScheduler(new SerialMergeScheduler())
                .setMergePolicy(NoMergePolicy.INSTANCE);
        try (var writer = new IndexWriter(directory, config)) {
          int i = 0;
          for (var vector : VECTORS) {
            var doc = new Document();
            doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
            writer.addDocument(doc);

            if (i++ % 1000 == 0) {
              writer.flush();
            }
          }

          writer.getConfig().setMergePolicy(new TieredMergePolicy());
          writer.forceMerge(1);
        }

        var ingestConfig =
            new IndexWriterConfig()
                .setCodec(ingestCodec)
                .setCommitOnClose(true)
                .setUseCompoundFile(false)
                .setMergeScheduler(new SerialMergeScheduler())
                .setMergePolicy(NoMergePolicy.INSTANCE);
        try (var writer = new IndexWriter(ingestDirectory, ingestConfig)) {
          int i = 0;
          for (var vector : VECTORS) {
            var doc = new Document();
            doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
            doc.add(new StoredField("id", i++));
            writer.addDocument(doc);

            if (i % 1000 == 0) {
              writer.flush();
            }
          }

          writer.getConfig().setMergePolicy(new TieredMergePolicy());
          writer.forceMerge(1);
        }

        var reader = DirectoryReader.open(directory);
        var searcher = new IndexSearcher(reader);
        var leafReader = reader.leaves().get(0).reader();
        var perFieldVectorReader =
            (PerFieldKnnVectorsFormat.FieldsReader) ((CodecReader) leafReader).getVectorReader();
        var vectorReader =
            (VectorSandboxVamanaVectorsReader) perFieldVectorReader.getFieldReader("vector");

        var ingestReader = DirectoryReader.open(ingestDirectory);
        var ingestSearcher = new IndexSearcher(ingestReader);
        var ingestLeafReader = ingestReader.leaves().get(0).reader();
        var ingestPerFieldVectorReader =
            (PerFieldKnnVectorsFormat.FieldsReader)
                ((CodecReader) ingestLeafReader).getVectorReader();
        var ingestVectorReader =
            (VectorSandboxFastIngestVectorsReader)
                ingestPerFieldVectorReader.getFieldReader("vector");

        var graph = vectorReader.getGraph("vector");
        //        var quantizedGraph = ingestVectorReader.getGraph("vector");
        var onHeapGraph = onHeapGraph();

        //      sandboxGraph.seek(0);
        //      System.out.println("sandboxGraph.nextNeighbor() = " + sandboxGraph.nextNeighbor());

        var query = new KnnFloatVectorQuery("vector", VECTORS.get(0), 10);
        var results = searcher.search(query, 10).scoreDocs;
        var ingestResults = ingestSearcher.search(query, 10).scoreDocs;

        var ingestDocs = new int[ingestResults.length];
        for (int i = 0; i < ingestDocs.length; i++) {
          int id =
              ingestSearcher
                  .storedFields()
                  .document(ingestResults[i].doc)
                  .getField("id")
                  .numericValue()
                  .intValue();
          ingestDocs[i] = id;
        }

        System.out.println("results = " + results);
        System.out.println("ingestResults = " + ingestResults);
      }
    }
  }

  @Test
  @Ignore
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
        var leafReader = reader.leaves().get(0).reader();
        var perFieldVectorReader =
            (PerFieldKnnVectorsFormat.FieldsReader) ((CodecReader) leafReader).getVectorReader();
        var vectorReader =
            (VectorSandboxVamanaVectorsReader) perFieldVectorReader.getFieldReader("vector");

        var mergedReader = DirectoryReader.open(mergedDirectory);
        var mergedSearcher = new IndexSearcher(mergedReader);
        var mergedLeafReader = mergedReader.leaves().get(0).reader();
        var mergedPerFieldVectorReader =
            (PerFieldKnnVectorsFormat.FieldsReader)
                ((CodecReader) mergedLeafReader).getVectorReader();
        var mergedVectorReader =
            (VectorSandboxVamanaVectorsReader) mergedPerFieldVectorReader.getFieldReader("vector");

        var graph = vectorReader.getGraph("vector");
        var mergedGraph = mergedVectorReader.getGraph("vector");
        var onHeapGraph = onHeapGraph();

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
    var values = new RAVectorValues<>(VECTORS, VECTOR_DIMENSIONS);

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
