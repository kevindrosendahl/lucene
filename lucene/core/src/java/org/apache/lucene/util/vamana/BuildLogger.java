package org.apache.lucene.util.vamana;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.vamana.VamanaGraphBuilder.Candidate;

public class BuildLogger {

  public static final Path PATH = Path.of("/Users/kevin.rosendahl/scratch/vamana/lucene.log");
  private static FileWriter FILE_WRITER;
  private static BufferedWriter WRITER;
  private static final AtomicInteger COUNTER = new AtomicInteger(0);


  private static class NoOpWriter extends Writer {

    static NoOpWriter INSTANCE = new NoOpWriter();

    @Override
    public void write(char[] cbuf, int off, int len) {
    }

    @Override
    public void flush() {
    }

    @Override
    public void close() {
    }
  }

  static {
    wrap(() -> {
      FILE_WRITER = new FileWriter(PATH.toString(), true);
//      WRITER = new BufferedWriter(FILE_WRITER);
      WRITER = new BufferedWriter(NoOpWriter.INSTANCE);
    });
  }

  public static void logAddNode(int node) {
    wrap(() -> {
      WRITER.write(counter() + " ADD " + node);
      WRITER.newLine();
    });
  }

  public static void nothing() {}

  public static void logSelectedDiverse(List<Candidate> selected) {
    if (selected.isEmpty()) {
      return;
    }

    var bitset = new FixedBitSet(10000);
    for (var candidate : selected) {
      bitset.set(candidate.node());
    }

    logSelectedDiverse(bitset);
  }

  public static void logSelectedDiverse(BitSet selected) {
    if (selected.cardinality() == 0) {
      return;
    }

    wrap(() -> {
      WRITER.write(counter() + " SELECTED DIVERSE");
      for (int i = selected.nextSetBit(0);
          i != DocIdSetIterator.NO_MORE_DOCS;
          i = selected.nextSetBit(i + 1)) {
        int node = i;
        WRITER.write(" " + node);

        // nextSetBit will assert if you're past the end, so check ourselves
        if (i + 1 >= selected.length()) {
          break;
        }
      }

      WRITER.newLine();
    });
  }

  public static void logBacklinked(int node) {
    wrap(() -> {
      WRITER.write(counter() + " BACKLINKED " + node);
      WRITER.newLine();
    });
  }

  public static void logLink(int node, int neighbor, float score) {
    wrap(() -> {
      WRITER.write(counter() + " LINK " + node + " NEIGHBOR " + neighbor + " SCORE " + score);
      WRITER.newLine();
    });
  }

  public static void logPruned(int node) {
    wrap(() -> {
      WRITER.write(counter() + " PRUNED " + node);
      WRITER.newLine();
    });
  }

  public static void flush() {
    wrap(() -> WRITER.flush());
  }

  private static int counter() {
    int val = COUNTER.getAndIncrement();
    if (val == 26994) {
      doNothing();
    }

    return val;
  }

  private static void doNothing() {}

  private interface ThrowableRunnable {

    void run() throws Exception;
  }

  private static void wrap(ThrowableRunnable runnable) {
    try {
      runnable.run();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
}
