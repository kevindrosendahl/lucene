package org.apache.lucene.util.vamana;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99Codec;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.vectorsandbox.VectorSandboxFastIngestVectorsFormat;
import org.apache.lucene.codecs.vectorsandbox.VectorSandboxVamanaVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Ignore;
import org.junit.Test;

public class TestPQ extends LuceneTestCase {

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
  public void compareWithNoPq() throws Exception {
    var hnsw =
        new Lucene99Codec() {
          @Override
          public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
            return new Lucene99HnswVectorsFormat(32, 100);
          }
        };

    var vamana =
        new Lucene99Codec() {
          @Override
          public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
            return new VectorSandboxVamanaVectorsFormat(
                VectorSandboxVamanaVectorsFormat.DEFAULT_MAX_CONN,
                VectorSandboxVamanaVectorsFormat.DEFAULT_MAX_CONN,
                VectorSandboxVamanaVectorsFormat.DEFAULT_ALPHA, 2);
          }
        };

    try (var hnswDirectory = new ByteBuffersDirectory()) {
      try (var vamanaDirectory = new ByteBuffersDirectory()) {
        var config =
            new IndexWriterConfig()
                .setCodec(hnsw)
                .setCommitOnClose(true)
                .setUseCompoundFile(false)
                .setMergeScheduler(new SerialMergeScheduler())
                .setMergePolicy(NoMergePolicy.INSTANCE);
        try (var writer = new IndexWriter(hnswDirectory, config)) {
//          int i = 0;
          for (var vector : VECTORS) {
            var doc = new Document();
            doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
            writer.addDocument(doc);

//            if (i++ % 1000 == 0) {
//              writer.flush();
//            }
          }

//          writer.getConfig().setMergePolicy(new TieredMergePolicy());
//          writer.forceMerge(1);
        }

        var ingestConfig =
            new IndexWriterConfig()
                .setCodec(vamana)
                .setCommitOnClose(true)
                .setUseCompoundFile(false)
                .setMergeScheduler(new SerialMergeScheduler())
                .setMergePolicy(NoMergePolicy.INSTANCE);
        try (var writer = new IndexWriter(vamanaDirectory, ingestConfig)) {
//          int i = 0;
          for (var vector : VECTORS) {
            var doc = new Document();
            doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
//            doc.add(new StoredField("id", i++));
            writer.addDocument(doc);

//            if (i % 1000 == 0) {
//              writer.flush();
//            }
          }

//          writer.getConfig().setMergePolicy(new TieredMergePolicy());
//          writer.forceMerge(1);
        }

        var hnswReader = DirectoryReader.open(hnswDirectory);
        var hnswSearcher = new IndexSearcher(hnswReader);
        //        var leafReader = reader.leaves().get(0).reader();
        //        var perFieldVectorReader =
        //            (PerFieldKnnVectorsFormat.FieldsReader) ((CodecReader)
        // leafReader).getVectorReader();
        //        var vectorReader =
        //            (VectorSandboxVamanaVectorsReader)
        // perFieldVectorReader.getFieldReader("vector");

        var vamanaReader = DirectoryReader.open(vamanaDirectory);
        var vamanaSearcher = new IndexSearcher(vamanaReader);
        //        var ingestLeafReader = ingestReader.leaves().get(0).reader();
        //        var ingestPerFieldVectorReader =
        //            (PerFieldKnnVectorsFormat.FieldsReader)
        //                ((CodecReader) ingestLeafReader).getVectorReader();
        //        var ingestVectorReader =
        //            (VectorSandboxFastIngestVectorsReader)
        //                ingestPerFieldVectorReader.getFieldReader("vector");

        //        var graph = vectorReader.getGraph("vector");
        //        var quantizedGraph = ingestVectorReader.getGraph("vector");
        //        var onHeapGraph = onHeapGraph();

        //      sandboxGraph.seek(0);
        //      System.out.println("sandboxGraph.nextNeighbor() = " + sandboxGraph.nextNeighbor());

        var query = new KnnFloatVectorQuery("vector", VECTORS.get(0), 10);
        var hnswResults = hnswSearcher.search(query, 10).scoreDocs;
        var vamanaResults = vamanaSearcher.search(query, 10).scoreDocs;

//        var ingestDocs = new int[vamanaResults.length];
//        for (int i = 0; i < ingestDocs.length; i++) {
//          int id =
//              vamanaSearcher
//                  .storedFields()
//                  .document(vamanaResults[i].doc)
//                  .getField("id")
//                  .numericValue()
//                  .intValue();
//          ingestDocs[i] = id;
//        }

        System.out.println("hnswResults = " + hnswResults);
        System.out.println("vamanaResults = " + vamanaResults);

        ScoreDoc[] rerankedResults = new ScoreDoc[vamanaResults.length];
        for (int i = 0; i < rerankedResults.length; i++) {
          try {
            int doc = vamanaResults[i].doc;
            float score = VectorSimilarityFunction.COSINE.compare(VECTORS.get(0), VECTORS.get(doc));
//            float score = exactScorer.score(doc);
            rerankedResults[i] = new ScoreDoc(doc, score, vamanaResults[i].shardIndex);
          } catch (Exception e) {
            throw new RuntimeException(e);
          }
        }

        Arrays.sort(rerankedResults, Comparator.comparing(scoreDoc -> -scoreDoc.score));

        System.out.println("rerankedResults = " + rerankedResults);
      }
    }
  }
}
