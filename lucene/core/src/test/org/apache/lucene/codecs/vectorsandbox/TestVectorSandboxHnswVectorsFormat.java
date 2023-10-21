package org.apache.lucene.codecs.vectorsandbox;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene95.Lucene95Codec;
import org.apache.lucene.codecs.lucene95.Lucene95HnswVectorsFormat;
import org.apache.lucene.codecs.lucene95.Lucene95HnswVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
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

  @Test
  public void loadGraph() throws Exception {
    try (var sandboxDirectory = new MMapDirectory(
        Path.of("/Users/kevin.rosendahl/scratch/lucene-vector-sandbox/sandbox"))) {
      try (var lucene95Directory = new MMapDirectory(
          Path.of("/Users/kevin.rosendahl/scratch/lucene-vector-sandbox/lucene95"))) {

        var sandboxReader = DirectoryReader.open(sandboxDirectory);
        var sandboxLeafReader = sandboxReader.leaves().get(0).reader();
        var sandboxPerFieldVectorReader = (PerFieldKnnVectorsFormat.FieldsReader) ((CodecReader) sandboxLeafReader).getVectorReader();
        var sandboxVectorReader = (VectorSandboxHnswVectorsReader) sandboxPerFieldVectorReader.getFieldReader(
            "vector");
        var sandboxGraph = sandboxVectorReader.getGraph("vector");

        var lucene95Reader = DirectoryReader.open(lucene95Directory);
        var lucene95LeafReader = lucene95Reader.leaves().get(0).reader();
        var lucene95PerFieldVectorReader = (PerFieldKnnVectorsFormat.FieldsReader) ((CodecReader) lucene95LeafReader).getVectorReader();
        var lucene95VectorReader = (Lucene95HnswVectorsReader) lucene95PerFieldVectorReader.getFieldReader(
            "vector");
        var lucene95Graph = lucene95VectorReader.getGraph("vector");

        lucene95Graph.seek(1, 4);
        var lucene95Neighbors = new ArrayList<Integer>();
        var lucene95Neighbor = lucene95Graph.nextNeighbor();
        while (lucene95Neighbor != DocIdSetIterator.NO_MORE_DOCS) {
          lucene95Neighbors.add(lucene95Neighbor);
          lucene95Neighbor = lucene95Graph.nextNeighbor();
        }

        sandboxGraph.seek(1, 4);
        var sandboxNeighbors = new ArrayList<Integer>();
        var sandboxNeighbor = sandboxGraph.nextNeighbor();
        while (sandboxNeighbor != DocIdSetIterator.NO_MORE_DOCS) {
          sandboxNeighbors.add(sandboxNeighbor);
          sandboxNeighbor = sandboxGraph.nextNeighbor();
        }

        System.out.println("lucene95Neighbors = " + lucene95Neighbors);
        System.out.println("sandboxNeighbors = " + sandboxNeighbors);
      }
    }
  }


  @Test
  public void createAndInspect() throws Exception {
    var sandboxCodec = new Lucene95Codec() {
      @Override
      public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
        return new VectorSandboxHnswVectorsFormat();
      }
    };
    var lucene95Codec = new Lucene95Codec();

    try (var sandboxDirectory = new ByteBuffersDirectory()) {
      try (var lucene95Directory = new ByteBuffersDirectory()) {

        // Build indexes in both directories
        var directories = Map.ofEntries(Map.entry(sandboxDirectory, sandboxCodec),
            Map.entry(lucene95Directory, lucene95Codec));

        for (var entry : directories.entrySet()) {
          var config = new IndexWriterConfig().setCodec(entry.getValue()).setCommitOnClose(true)
              .setUseCompoundFile(false);
          try (var writer = new IndexWriter(entry.getKey(), config)) {
            for (var vector : VECTORS) {
              var doc = new Document();
              doc.add(new KnnFloatVectorField("vector", vector));
              writer.addDocument(doc);
            }
          }
        }

//        var sandboxReader = DirectoryReader.open(sandboxDirectory);
//        var sandboxLeafReader = sandboxReader.leaves().get(0).reader();
//        var sandboxPerFieldVectorReader = (PerFieldKnnVectorsFormat.FieldsReader) ((CodecReader) sandboxLeafReader).getVectorReader();
//        var sandboxVectorReader = (VectorSandboxHnswVectorsReader) sandboxPerFieldVectorReader.getFieldReader(
//            "vector");
//        var sandboxGraph = sandboxVectorReader.getGraph("vector");
//
//        var lucene95Reader = DirectoryReader.open(lucene95Directory);
//        var lucene95LeafReader = lucene95Reader.leaves().get(0).reader();
//        var lucene95PerFieldVectorReader = (PerFieldKnnVectorsFormat.FieldsReader) ((CodecReader) lucene95LeafReader).getVectorReader();
//        var lucene95VectorReader = (Lucene95HnswVectorsReader) lucene95PerFieldVectorReader.getFieldReader(
//            "vector");
//        var lucene95Graph = lucene95VectorReader.getGraph("vector");
//
//        lucene95Graph.seek(1, 4);
//        var lucene95Neighbors = new ArrayList<Integer>();
//        var lucene95Neighbor = lucene95Graph.nextNeighbor();
//        while (lucene95Neighbor != DocIdSetIterator.NO_MORE_DOCS) {
//          lucene95Neighbors.add(lucene95Neighbor);
//          lucene95Neighbor = lucene95Graph.nextNeighbor();
//        }
//
//        sandboxGraph.seek(1, 4);
//        var sandboxNeighbors = new ArrayList<Integer>();
//        var sandboxNeighbor = sandboxGraph.nextNeighbor();
//        while (sandboxNeighbor != DocIdSetIterator.NO_MORE_DOCS) {
//          sandboxNeighbors.add(sandboxNeighbor);
//          sandboxNeighbor = sandboxGraph.nextNeighbor();
//        }
//
//        System.out.println("lucene95Neighbors = " + lucene95Neighbors);
//        System.out.println("sandboxNeighbors = " + sandboxNeighbors);
        var sandboxReader = DirectoryReader.open(sandboxDirectory);
        var sandboxSearcher = new IndexSearcher(sandboxReader);
        var lucene95Reader = DirectoryReader.open(lucene95Directory);
        var lucene95Searcher = new IndexSearcher(lucene95Reader);

        var query = new KnnFloatVectorQuery("vector", VECTORS[0], 5);
        var sandboxResults = sandboxSearcher.search(query, 5);
        var lucene95Results = lucene95Searcher.search(query, 5);

        System.out.println("lucene95Results = " + lucene95Results);
      }
    }
  }
}
