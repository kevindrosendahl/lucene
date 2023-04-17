package org.apache.lucene;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;

@SuppressWarnings("unused")
public class HnswTest {

  private static final Path PATH = Path.of("/Users/kevin.rosendahl/scratch/lucene-simd/data/local");

  private static float[] VEC_128_1 = new float[]{0.3315621f, 0.33339924f, 0.44922465f, 0.57794636f, 0.7977346f, 0.93452895f, 0.7990957f, 0.3844967f, 0.8246689f, 0.47435558f, 0.24389273f, 0.17842662f, 0.7403989f, 0.04041493f, 0.19196004f, 0.698226f, 0.9387964f, 0.6908557f, 0.34833324f, 0.10131681f, 0.7522585f, 0.6261683f, 0.12730712f, 0.04943216f, 0.56726307f, 0.6354304f, 0.029253185f, 0.7241589f, 0.13535792f, 0.5243005f, 0.06347954f, 0.045285642f, 0.5555706f, 0.6116037f, 0.22936451f, 0.030493557f, 0.024153292f, 0.9978422f, 0.16182446f, 0.17920542f, 0.50481325f, 0.81174546f, 0.59053165f, 0.67454827f, 0.5110827f, 0.8200847f, 0.86943305f, 0.52947676f, 0.20931596f, 0.8220836f, 0.6964213f, 0.02589035f, 0.7461635f, 0.96414745f, 0.28298193f, 0.64439416f, 0.73158306f, 0.2152183f, 0.7489189f, 0.40818924f, 0.023951888f, 0.5903908f, 0.8346008f, 0.18402702f, 0.67721075f, 0.54364645f, 0.1474579f, 0.028454006f, 0.13389784f, 0.15348089f, 0.92369837f, 0.43655932f, 0.44823635f, 0.07221627f, 0.6015418f, 0.15659344f, 0.6690701f, 0.6159548f, 0.60793585f, 0.6562549f, 0.12546676f, 0.143098f, 0.47412145f, 0.95986795f, 0.5634253f, 0.95142114f, 0.36204612f, 0.28045177f, 0.15865028f, 0.15377003f, 0.027121723f, 0.022783458f, 0.8592373f, 0.34807092f, 0.21495342f, 0.47459155f, 0.22792566f, 0.74668294f, 0.12029731f, 0.99020016f, 0.5940003f, 0.78196424f, 0.28365684f, 0.12713903f, 0.48080224f, 0.9586801f, 0.7007266f, 0.9599088f, 0.74006283f, 0.9195939f, 0.8365983f, 0.37035185f, 0.8459954f, 0.7589151f, 0.11263877f, 0.22065097f, 0.33044797f, 0.6305198f, 0.5397446f, 0.93820894f, 0.2787634f, 0.61592245f, 0.8558991f, 0.6825394f, 0.26075542f, 0.74340504f, 0.46536922f, 0.1894095f};
  private static float[] VEC_128_2 = new float[]{0.6019583f, 0.47766036f, 0.9263632f, 0.8790571f, 0.7697425f, 0.062573254f, 0.77652836f, 0.42120582f, 0.07282913f, 0.8227598f, 0.28244108f, 0.7332947f, 0.27012122f, 0.07724786f, 0.739484f, 0.17893356f, 0.9276813f, 0.24687767f, 0.29959583f, 0.37580878f, 0.026307285f, 0.24192095f, 0.8618338f, 0.62645644f, 0.934818f, 0.38496482f, 0.29813617f, 0.1774624f, 0.1573003f, 0.18383932f, 0.50586474f, 0.68434244f, 0.42076176f, 0.8384261f, 0.35954112f, 0.817835f, 0.5805706f, 0.7023847f, 0.28465772f, 0.5654409f, 0.3255704f, 0.17556131f, 0.68269175f, 0.6482647f, 0.80677336f, 0.9651635f, 0.9127252f, 0.5416311f, 0.59013695f, 0.004341662f, 0.6737654f, 0.8283145f, 0.945306f, 0.87691104f, 0.95347404f, 0.26086503f, 0.5816781f, 0.57848954f, 0.8121813f, 0.7154277f, 0.88716936f, 0.19985366f, 0.08254188f, 0.70022786f, 0.32319123f, 0.390993f, 0.08986956f, 0.76976776f, 0.39429033f, 0.6379622f, 0.0690645f, 0.8885914f, 0.008476138f, 0.7336026f, 0.4822681f, 0.7997104f, 0.74340624f, 0.7625685f, 0.66618896f, 0.8615381f, 0.65639067f, 0.25651944f, 0.47683442f, 0.71818674f, 0.6478096f, 0.353105f, 0.997654f, 0.9215838f, 0.7863283f, 0.1493991f, 0.1584928f, 0.10604882f, 0.4116947f, 0.4542697f, 0.8290197f, 0.16832918f, 0.6814767f, 0.68296015f, 0.73131996f, 0.7858582f, 0.65954405f, 0.20970315f, 0.32013273f, 0.43602228f, 0.1472367f, 0.49989915f, 0.6419044f, 0.29797316f, 0.5269377f, 0.32230687f, 0.4696746f, 0.23011452f, 0.6692502f, 0.22447878f, 0.66219896f, 0.64053774f, 0.89936423f, 0.81262076f, 0.48314577f, 0.98974544f, 0.5288081f, 0.6720697f, 0.38831705f, 0.88595206f, 0.14205587f, 0.59921485f, 0.7223977f, 0.0196926f};


  static final int DIMENSIONS = 128;



  public static void main(String[] args) throws Exception {
    for (int i = 0; i < 10; i++) {
      runQuery();
    }
//    distanceParity();
  }

  private static void runQuery() throws Exception {
//    clearDirectory(PATH);
    try (var directory = new MMapDirectory(PATH)) {
//      var writer = new IndexWriter(directory, new IndexWriterConfig());
//
//      var vectors = generateRandomVectors(10000, 128);
//      for (var vector : vectors) {
//        var doc = new Document();
//        doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.DOT_PRODUCT));
//        writer.addDocument(doc);
//      }
//
//      writer.forceMerge(1);
//      writer.commit();

      try (IndexReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = new IndexSearcher(reader);

        VectorUtil.DOT_PRODUCT_IMPL = VectorUtil.DotProductImpl.JAVA_SIMD;
        HnswGraphSearcher.USE_SEGMENTS = false;
        HnswGraphSearcher.USE_DENSE_FIXED_BIT_SET = false;

        var queryVector = generateRandomVectors(1, 128).get(0);
        var query = new KnnFloatVectorQuery("$type:knnVector/vector", queryVector, 100);

        System.out.println("--- no segments ---");
        Instant start1 = Instant.now();
        TopDocs hits1 = searcher.search(query, 100);
        Instant end1 = Instant.now();

        System.out.println("hits.totalHits = " + hits1.totalHits);

        Duration total1 = Duration.between(start1, end1);
        System.out.println("total = " + total1);

        HnswGraphSearcher.USE_SEGMENTS = true;
        System.out.println();
        System.out.println("--- with segments ---");
        Instant start2 = Instant.now();
        TopDocs hits2 = searcher.search(query, 100);
        Instant end2 = Instant.now();

        System.out.println("hits.totalHits = " + hits2.totalHits);

        Duration total2 = Duration.between(start2, end2);
        System.out.println("total = " + total2);
      }
    }
  }

  private static void distanceParity() {
    System.out.println("FloatVector.SPECIES_PREFERRED.length() = " + FloatVector.SPECIES_PREFERRED.length());
    
    var vecs = generateRandomVectors(2, DIMENSIONS);
    var vec1 = vecs.get(0);
    System.out.println("vec1 = " + Arrays.toString(vec1));
    var vec2 = vecs.get(1);
    System.out.println("vec2 = " + Arrays.toString(vec2));

    var scalar = VectorUtil.dotProductScalar(vec1, vec2);
    System.out.println("scalar     = " + scalar);

    var simd = VectorUtil.dotProductSimd(vec1, vec2);
    System.out.println("simd       = " + simd);

    try (Arena arena = Arena.openConfined()) {
      var vec1Segment = arena.allocateArray(ValueLayout.JAVA_FLOAT, vec1);
      var vec2Segment = arena.allocateArray(ValueLayout.JAVA_FLOAT, vec2);

      var memorySimd = VectorUtil.dotProductSimdSegment(vec1Segment, vec2Segment, DIMENSIONS);
      System.out.println("memorySimd = " + memorySimd);
    }
  }

  private static float[] vec1() {
    if (DIMENSIONS == 128) {
      return VEC_128_1;
    }

    float[] vec = new float[DIMENSIONS];
    System.arraycopy(VEC_128_1, 0, vec, 0, DIMENSIONS);

    return vec;
  }

  private static float[] vec2() {
    if (DIMENSIONS == 128) {
      return VEC_128_2;
    }

    float[] vec = new float[DIMENSIONS];
    System.arraycopy(VEC_128_2, 0, vec, 0, DIMENSIONS);

    return vec;
  }

  private static void clearDirectory(Path path) throws IOException  {
    if (Files.exists(path) && Files.isDirectory(path)) {
      try (DirectoryStream<Path> directoryStream = Files.newDirectoryStream(path)) {
        // Iterate through each file in the directory and delete it
        for (Path file : directoryStream) {
          if (Files.isRegularFile(file)) {
            Files.delete(file);
          }
        }
      }
    }
  }

  private static List<float[]> generateRandomVectors(int numOfVectors, int dimensions) {
    List<float[]> vectors = new ArrayList<>(numOfVectors);
    Random random = new Random();

    for (int i = 0; i < numOfVectors; i++) {
      float[] vector = new float[dimensions];
      for (int j = 0; j < dimensions; j++) {
        vector[j] = random.nextFloat();
      }
      vectors.add(vector);
    }

    return vectors;
  }
}
