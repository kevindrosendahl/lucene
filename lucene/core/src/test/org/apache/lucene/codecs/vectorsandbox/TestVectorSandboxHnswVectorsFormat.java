package org.apache.lucene.codecs.vectorsandbox;

import java.nio.file.Path;
import java.util.Random;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene95.Lucene95Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestVectorSandboxHnswVectorsFormat extends LuceneTestCase {

  private static final float[][] VECTORS = new float[100][4];
  private static final Random RANDOM = new Random(0);

  static {
    for (var i = 0; i < VECTORS.length; i++) {
      for (var j = 0; j < VECTORS[0].length; j++) {
        VECTORS[i][j] = RANDOM.nextFloat();
      }
    }
  }

  @Test
  public void createIndex() throws Exception {
    var sandboxCodec = new Lucene95Codec() {
      @Override
      public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
        return new VectorSandboxHnswVectorsFormat();
      }
    };

    try (var directory = new MMapDirectory(
        Path.of("/Users/kevin.rosendahl/scratch/lucene-vector-sandbox/sandbox"))) {
      var config = new IndexWriterConfig().setCodec(sandboxCodec).setCommitOnClose(true)
          .setUseCompoundFile(false);
      try (var writer = new IndexWriter(directory, config)) {
        for (var vector : VECTORS) {
          var doc = new Document();
          doc.add(new KnnFloatVectorField("vector", vector));
          writer.addDocument(doc);
        }
      }
    }

    var lucene95Codec = new Lucene95Codec();

    try (var directory = new MMapDirectory(
        Path.of("/Users/kevin.rosendahl/scratch/lucene-vector-sandbox/lucene95"))) {
      var config = new IndexWriterConfig().setCodec(lucene95Codec).setCommitOnClose(true)
          .setUseCompoundFile(false);
      try (var writer = new IndexWriter(directory, config)) {
        for (var vector : VECTORS) {
          var doc = new Document();
          doc.add(new KnnFloatVectorField("vector", vector));
          writer.addDocument(doc);
        }
      }
    }
  }
}
