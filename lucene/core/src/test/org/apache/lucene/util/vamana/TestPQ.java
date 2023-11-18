package org.apache.lucene.util.vamana;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Executors;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99Codec;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.codecs.vectorsandbox.VectorSandboxVamanaVectorsFormat;
import org.apache.lucene.codecs.vectorsandbox.VectorSandboxVamanaVectorsFormat.PQRerank;
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
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
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
                VectorSandboxVamanaVectorsFormat.DEFAULT_ALPHA,
                2,
                false);
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

            //            if (i % (10000 / 2) == 0) {
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

        System.out.println("hnswResults = " + Arrays.toString(hnswResults));
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

  @Test
  public void compareWithMerges() throws Exception {
    var noMergeCodec =
        new Lucene99Codec() {
          @Override
          public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
            return new VectorSandboxVamanaVectorsFormat(
                VectorSandboxVamanaVectorsFormat.DEFAULT_MAX_CONN,
                VectorSandboxVamanaVectorsFormat.DEFAULT_MAX_CONN,
                VectorSandboxVamanaVectorsFormat.DEFAULT_ALPHA,
                2,
                true);
          }
        };

    var mergeCodec =
        new Lucene99Codec() {
          @Override
          public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
            return new VectorSandboxVamanaVectorsFormat(
                VectorSandboxVamanaVectorsFormat.DEFAULT_MAX_CONN,
                VectorSandboxVamanaVectorsFormat.DEFAULT_MAX_CONN,
                VectorSandboxVamanaVectorsFormat.DEFAULT_ALPHA,
                2,
                true,
                null,
                2,
                Executors.newCachedThreadPool());
          }
        };

    try (var noMergeDirectory = new ByteBuffersDirectory()) {
      try (var mergeDirectory = new ByteBuffersDirectory()) {
        var noMergeConfig =
            new IndexWriterConfig()
                .setCodec(noMergeCodec)
                .setCommitOnClose(true)
                .setUseCompoundFile(false)
                .setMergeScheduler(new SerialMergeScheduler())
                .setMergePolicy(NoMergePolicy.INSTANCE);
        try (var writer = new IndexWriter(noMergeDirectory, noMergeConfig)) {
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

        var mergeConfig =
            new IndexWriterConfig()
                .setCodec(mergeCodec)
                .setCommitOnClose(true)
                .setUseCompoundFile(false)
                .setMergeScheduler(new SerialMergeScheduler())
                .setMergePolicy(NoMergePolicy.INSTANCE);
        try (var writer = new IndexWriter(mergeDirectory, mergeConfig)) {
          int i = 0;
          for (var vector : VECTORS) {
            var doc = new Document();
            doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
            doc.add(new StoredField("id", i++));
            writer.addDocument(doc);

            if (i % (10000 / 10) == 0) {
              writer.flush();
            }
          }

          writer.getConfig().setMergePolicy(new TieredMergePolicy());
          writer.forceMerge(1);
        }

        var noMergeReader = DirectoryReader.open(noMergeDirectory);
        var noMergeSearcher = new IndexSearcher(noMergeReader);
        var noMergeleafReader = noMergeReader.leaves().get(0).reader();
        var noMergePerFieldVectorReader =
            (PerFieldKnnVectorsFormat.FieldsReader)
                ((CodecReader) noMergeleafReader).getVectorReader();
        var noMergeVectorReader =
            (VectorSandboxVamanaVectorsReader) noMergePerFieldVectorReader.getFieldReader("vector");

        var mergeReader = DirectoryReader.open(mergeDirectory);
        var mergeSearcher = new IndexSearcher(mergeReader);
        var mergeLeafReader = mergeReader.leaves().get(0).reader();
        var mergePerFieldVectorReader =
            (PerFieldKnnVectorsFormat.FieldsReader)
                ((CodecReader) mergeLeafReader).getVectorReader();
        var mergeVectorReader =
            (VectorSandboxVamanaVectorsReader) mergePerFieldVectorReader.getFieldReader("vector");

        //        var graph = vectorReader.getGraph("vector");
        //        var quantizedGraph = ingestVectorReader.getGraph("vector");
        //        var onHeapGraph = onHeapGraph();

        //      sandboxGraph.seek(0);
        //      System.out.println("sandboxGraph.nextNeighbor() = " + sandboxGraph.nextNeighbor());

        var query = new KnnFloatVectorQuery("vector", VECTORS.get(0), 10);
        var noMergeResults = noMergeSearcher.search(query, 10).scoreDocs;
        var mergeResults = mergeSearcher.search(query, 10).scoreDocs;

        var mergeLookupResults = new ScoreDoc[mergeResults.length];
        for (int i = 0; i < mergeLookupResults.length; i++) {
          int id =
              mergeSearcher
                  .storedFields()
                  .document(mergeResults[i].doc)
                  .getField("id")
                  .numericValue()
                  .intValue();
          mergeLookupResults[i] =
              new ScoreDoc(id, mergeResults[i].score, mergeResults[i].shardIndex);
        }

        System.out.println("noMergeResults = " + Arrays.toString(noMergeResults));
        System.out.println("mergeResults = " + Arrays.toString(mergeResults));
        System.out.println("mergeLookupResults = " + Arrays.toString(mergeLookupResults));

        var mergeVectors = mergeVectorReader.getFloatVectorValues("vector");
        mergeVectors.advance(123);
        var mergeZeroVec = mergeVectors.vectorValue();
        var encodedMergeZero = mergeVectorReader.pqVectors.get("vector")[123];
        var mergePq = mergeVectorReader.fields.get("vector").pq;
        var decodedMergeZero = mergePq.decode(encodedMergeZero);
        var zeroOrd =
            mergeSearcher.storedFields().document(123).getField("id").numericValue().intValue();
        System.out.println("decodedMergeZero = " + Arrays.toString(decodedMergeZero));

        var noMergeVectors = noMergeVectorReader.getFloatVectorValues("vector");
        noMergeVectors.advance(zeroOrd);
        var noMergeVec = noMergeVectors.vectorValue();
        var encodedNoMerge = noMergeVectorReader.pqVectors.get("vector")[zeroOrd];
        var noMergePq = noMergeVectorReader.fields.get("vector").pq;
        var decodedNoMergeZero = noMergePq.decode(encodedNoMerge);
        System.out.println("decodedNoMergeZero = " + Arrays.toString(decodedNoMergeZero));

        //        ScoreDoc[] rerankedResults = new ScoreDoc[mergeResults.length];
        //        for (int i = 0; i < rerankedResults.length; i++) {
        //          try {
        //            int doc = mergeResults[i].doc;
        //            float score = VectorSimilarityFunction.COSINE.compare(VECTORS.get(0),
        // VECTORS.get(doc));
        //            //            float score = exactScorer.score(doc);
        //            rerankedResults[i] = new ScoreDoc(doc, score, mergeResults[i].shardIndex);
        //          } catch (Exception e) {
        //            throw new RuntimeException(e);
        //          }
        //        }
        //
        //        Arrays.sort(rerankedResults, Comparator.comparing(scoreDoc -> -scoreDoc.score));
        //
        //        System.out.println("rerankedResults = " + Arrays.toString(rerankedResults));
      }
    }
  }

  @Test
  public void compareGlove100() throws Exception {
    try (var noMergeDirectory =
        new MMapDirectory(
            Path.of(
                "/Users/kevin.rosendahl/src/github.com/kevindrosendahl/java-ann-bench/indexes/glove-100-angular/lucene_sandbox-vamana_maxConn:32-beamWidth:100-alpha:1.2-pqFactor:2-scalarQuantization:false-numThreads:1-forceMerge:false"))) {
      try (var mergeDirectory =
          new MMapDirectory(
              Path.of(
                  "/Users/kevin.rosendahl/src/github.com/kevindrosendahl/java-ann-bench/indexes/glove-100-angular/lucene_sandbox-vamana_maxConn:32-beamWidth:100-alpha:1.2-pqFactor:2-scalarQuantization:false-numThreads:10-forceMerge:true"))) {
        var noMergeReader = DirectoryReader.open(noMergeDirectory);
        var noMergeSearcher = new IndexSearcher(noMergeReader);
        var noMergeleafReader = noMergeReader.leaves().get(0).reader();
        var noMergePerFieldVectorReader =
            (PerFieldKnnVectorsFormat.FieldsReader)
                ((CodecReader) noMergeleafReader).getVectorReader();
        var noMergeVectorReader =
            (VectorSandboxVamanaVectorsReader) noMergePerFieldVectorReader.getFieldReader("vector");

        var mergeReader = DirectoryReader.open(mergeDirectory);
        var mergeSearcher = new IndexSearcher(mergeReader);
        var mergeLeafReader = mergeReader.leaves().get(0).reader();
        var mergePerFieldVectorReader =
            (PerFieldKnnVectorsFormat.FieldsReader)
                ((CodecReader) mergeLeafReader).getVectorReader();
        var mergeVectorReader =
            (VectorSandboxVamanaVectorsReader) mergePerFieldVectorReader.getFieldReader("vector");

        //        var query = new KnnFloatVectorQuery("vector", VECTORS.get(0), 10);
        //        var noMergeResults = noMergeSearcher.search(query, 10).scoreDocs;
        //        var mergeResults = mergeSearcher.search(query, 10).scoreDocs;
        //
        //        var mergeLookupResults = new ScoreDoc[mergeResults.length];
        //        for (int i = 0; i < mergeLookupResults.length; i++) {
        //          int id =
        //              mergeSearcher
        //                  .storedFields()
        //                  .document(mergeResults[i].doc)
        //                  .getField("id")
        //                  .numericValue()
        //                  .intValue();
        //          mergeLookupResults[i] = new ScoreDoc(id, mergeResults[i].score,
        //              mergeResults[i].shardIndex);
        //        }
        //
        //        System.out.println("noMergeResults = " + Arrays.toString(noMergeResults));
        //        System.out.println("mergeResults = " + Arrays.toString(mergeResults));
        //        System.out.println("mergeLookupResults = " + Arrays.toString(mergeLookupResults));

        var mergeVectors = mergeVectorReader.getFloatVectorValues("vector");
        mergeVectors.advance(123);
        var mergeZeroVec = mergeVectors.vectorValue();
        var encodedMergeZero = mergeVectorReader.pqVectors.get("vector")[123];
        var mergePq = mergeVectorReader.fields.get("vector").pq;
        var decodedMergeZero = mergePq.decode(encodedMergeZero);
        var mergeId =
            mergeSearcher.storedFields().document(123).getField("id").numericValue().intValue();
        System.out.println("decodedMergeZero = " + Arrays.toString(decodedMergeZero));

        int noMergeOrd = -1;
        for (int i = 0; i < 1183514; i++) {
          int id =
              noMergeSearcher.storedFields().document(i).getField("id").numericValue().intValue();
          if (id == mergeId) {
            noMergeOrd = i;
          }
        }

        var noMergeVectors = noMergeVectorReader.getFloatVectorValues("vector");
        noMergeVectors.advance(noMergeOrd);
        var noMergeVec = noMergeVectors.vectorValue();
        var encodedNoMerge = noMergeVectorReader.pqVectors.get("vector")[noMergeOrd];
        var noMergePq = noMergeVectorReader.fields.get("vector").pq;
        var decodedNoMergeZero = noMergePq.decode(encodedNoMerge);
        System.out.println("decodedNoMergeZero = " + Arrays.toString(decodedNoMergeZero));
      }
    }
  }
}
