package org.apache.lucene.util.vamana;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99Codec;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.codecs.vectorsandbox.VectorSandboxVamanaVectorsFormat;
import org.apache.lucene.codecs.vectorsandbox.VectorSandboxVamanaVectorsReader;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
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
  public void compareGraphs() throws Exception {
    var sandboxCodec =
        new Lucene99Codec() {
          @Override
          public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
            return new VectorSandboxVamanaVectorsFormat(32, 100, 1.2f);
          }
        };

    try (var sandboxDirectory = new ByteBuffersDirectory()) {
      var config =
          new IndexWriterConfig()
              .setCodec(sandboxCodec)
              .setCommitOnClose(true)
              .setUseCompoundFile(false);
      try (var writer = new IndexWriter(sandboxDirectory, config)) {
        for (var vector : VECTORS) {
          var doc = new Document();
          doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
          writer.addDocument(doc);
        }
      }

      var sandboxReader = DirectoryReader.open(sandboxDirectory);
      var sandboxLeafReader = sandboxReader.leaves().get(0).reader();
      var sandboxPerFieldVectorReader =
          (PerFieldKnnVectorsFormat.FieldsReader)
              ((CodecReader) sandboxLeafReader).getVectorReader();
      var sandboxVectorReader =
          (VectorSandboxVamanaVectorsReader) sandboxPerFieldVectorReader.getFieldReader("vector");

      var sandboxGraph = sandboxVectorReader.getGraph("vector");
      var onHeapGraph = onHeapGraph();

      System.out.println("sandboxGraph = " + sandboxGraph);
      System.out.println("onHeapGraph = " + onHeapGraph);
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
