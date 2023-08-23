package org.apache.lucene.search;

import java.io.IOException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;

public class ExactKnnFloatVectorQuery extends AbstractKnnVectorQuery {

  private final float[] target;

  public ExactKnnFloatVectorQuery(String field, float[] target, int k, Query filter) {
    super(field, k, filter);
    this.target = target;
  }

  public ExactKnnFloatVectorQuery(String field, float[] target, int k) {
    this(field, target, k, null);
  }

  @Override
  protected TopDocs approximateSearch(LeafReaderContext context, Bits acceptDocs, int visitedLimit)
      throws IOException {
    // FIXME: need to handle docs without vectors
    var iterator = acceptDocs != null ? new BitSetIterator((BitSet) acceptDocs, visitedLimit)
        : DocIdSetIterator.all(context.reader().maxDoc());
    return exactSearch(context, iterator);
  }

  @Override
  VectorScorer createVectorScorer(LeafReaderContext context, FieldInfo fi) throws IOException {
    if (fi.getVectorEncoding() != VectorEncoding.FLOAT32) {
      return null;
    }
    return VectorScorer.create(context, fi, this.target);
  }

  @Override
  public String toString(String field) {
    return getClass().getSimpleName() + ":" + this.field + "[" + target[0] + ",...][" + k + "]";
  }
}
