package org.apache.lucene.util.quantization;

public class ScalarQuantization {

  public static int[][] quantize(float[][] vectors) {
    // Gather global statistics.
    float minValue = Float.MAX_VALUE;
    float maxValue = Float.MIN_VALUE;
    for (float[] vector : vectors) {
      for (float val : vector) {
        if (val < minValue) minValue = val;
        if (val > maxValue) maxValue = val;
      }
    }
    float alpha = (maxValue - minValue) / 255;
    float offset = minValue;

    int[][] quantizedData = new int[vectors.length][];
    for (int i = 0; i < vectors.length; i++) {
      quantizedData[i] = new int[vectors[i].length];
      for (int j = 0; j < vectors[i].length; j++) {
        quantizedData[i][j] = (int) ((vectors[i][j] - offset) / alpha);
      }
    }

    return quantizedData;
  }

}
