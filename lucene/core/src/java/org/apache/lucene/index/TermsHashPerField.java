/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.index;

import java.io.IOException;
import org.apache.lucene.util.BitUtil;
import org.apache.lucene.util.ByteBlockPool;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.BytesRefHash;
import org.apache.lucene.util.BytesRefHash.BytesStartArray;
import org.apache.lucene.util.Counter;
import org.apache.lucene.util.IntBlockPool;

/**
 * This class stores streams of information per term without knowing the size of the stream ahead of
 * time. Each stream typically encodes one level of information like term frequency per document or
 * term proximity. Internally this class allocates a linked list of slices that can be read by a
 * {@link ByteSliceReader} for each term. Terms are first deduplicated in a {@link BytesRefHash}
 * once this is done internal data-structures point to the current offset of each stream that can be
 * written to.
 */
abstract class TermsHashPerField implements Comparable<TermsHashPerField> {
  private static final int HASH_INIT_SIZE = 4;

  private final TermsHashPerField nextPerField;
  private final IntBlockPool intPool;
  final ByteBlockPool bytePool;
  // for each term we store an integer per stream that points into the bytePool above
  // the address is updated once data is written to the stream to point to the next free offset
  // in the terms stream. The start address for the stream is stored in
  // postingsArray.byteStarts[termId]
  // This is initialized in the #addTerm method, either to a brand new per term stream if the term
  // is new or
  // to the addresses where the term stream was written to when we saw it the last time.
  private int[] termStreamAddressBuffer;
  private int streamAddressOffset;
  private final int streamCount;
  private final String fieldName;
  final IndexOptions indexOptions;
  /* This stores the actual term bytes for postings and offsets into the parent hash in the case that this
   * TermsHashPerField is hashing term vectors.*/
  private final BytesRefHash bytesHash;

  ParallelPostingsArray postingsArray;
  private int lastDocID; // only with assert

  /**
   * streamCount: how many streams this field stores per term. E.g. doc(+freq) is 1 stream,
   * prox+offset is a second.
   */
  TermsHashPerField(
      int streamCount,
      IntBlockPool intPool,
      ByteBlockPool bytePool,
      ByteBlockPool termBytePool,
      Counter bytesUsed,
      TermsHashPerField nextPerField,
      String fieldName,
      IndexOptions indexOptions) {
    this.intPool = intPool;
    this.bytePool = bytePool;
    this.streamCount = streamCount;
    this.fieldName = fieldName;
    this.nextPerField = nextPerField;
    assert indexOptions != IndexOptions.NONE;
    this.indexOptions = indexOptions;
    PostingsBytesStartArray byteStarts = new PostingsBytesStartArray(this, bytesUsed);
    bytesHash = new BytesRefHash(termBytePool, HASH_INIT_SIZE, byteStarts);
  }

  void reset() {
    bytesHash.clear(false);
    sortedTermIDs = null;
    if (nextPerField != null) {
      nextPerField.reset();
    }
  }

  final void initReader(ByteSliceReader reader, int termID, int stream) {
    assert stream < streamCount;
    int streamStartOffset = postingsArray.addressOffset[termID];
    final int[] streamAddressBuffer =
        intPool.buffers[streamStartOffset >> IntBlockPool.INT_BLOCK_SHIFT];
    final int offsetInAddressBuffer = streamStartOffset & IntBlockPool.INT_BLOCK_MASK;
    reader.init(
        bytePool,
        postingsArray.byteStarts[termID] + stream * FIRST_LEVEL_SIZE,
        streamAddressBuffer[offsetInAddressBuffer + stream]);
  }

  private int[] sortedTermIDs;

  /**
   * Collapse the hash table and sort in-place; also sets this.sortedTermIDs to the results This
   * method must not be called twice unless {@link #reset()} or {@link #reinitHash()} was called.
   */
  final void sortTerms() {
    assert sortedTermIDs == null;
    sortedTermIDs = bytesHash.sort();
  }

  /** Returns the sorted term IDs. {@link #sortTerms()} must be called before */
  final int[] getSortedTermIDs() {
    assert sortedTermIDs != null;
    return sortedTermIDs;
  }

  final void reinitHash() {
    sortedTermIDs = null;
    bytesHash.reinit();
  }

  private boolean doNextCall;

  // Secondary entry point (for 2nd & subsequent TermsHash),
  // because token text has already been "interned" into
  // textStart, so we hash by textStart.  term vectors use
  // this API.
  private void add(int textStart, final int docID) throws IOException {
    int termID = bytesHash.addByPoolOffset(textStart);
    if (termID >= 0) { // New posting
      // First time we are seeing this token since we last
      // flushed the hash.
      initStreamSlices(termID, docID);
    } else {
      positionStreamSlice(termID, docID);
    }
  }

  /**
   * Called when we first encounter a new term. We must allocate slies to store the postings (vInt
   * compressed doc/freq/prox), and also the int pointers to where (in our ByteBlockPool storage)
   * the postings for this term begin.
   */
  private void initStreamSlices(int termID, int docID) throws IOException {
    // Init stream slices
    if (streamCount + intPool.intUpto > IntBlockPool.INT_BLOCK_SIZE) {
      // not enough space remaining in this buffer -- jump to next buffer and lose this remaining
      // piece
      intPool.nextBuffer();
    }

    if (ByteBlockPool.BYTE_BLOCK_SIZE - bytePool.byteUpto < (2 * streamCount) * FIRST_LEVEL_SIZE) {
      // can we fit at least one byte per stream in the current buffer, if not allocate a new one
      bytePool.nextBuffer();
    }

    termStreamAddressBuffer = intPool.buffer;
    streamAddressOffset = intPool.intUpto;
    intPool.intUpto += streamCount; // advance the pool to reserve the N streams for this term

    postingsArray.addressOffset[termID] = streamAddressOffset + intPool.intOffset;

    for (int i = 0; i < streamCount; i++) {
      // initialize each stream with a slice we start with ByteBlockPool.FIRST_LEVEL_SIZE)
      // and grow as we need more space. see ByteBlockPool.LEVEL_SIZE_ARRAY
      final int upto = newSlice(bytePool, FIRST_LEVEL_SIZE, 0);
      termStreamAddressBuffer[streamAddressOffset + i] = upto + bytePool.byteOffset;
    }
    postingsArray.byteStarts[termID] = termStreamAddressBuffer[streamAddressOffset];
    newTerm(termID, docID);
  }

  private boolean assertDocId(int docId) {
    assert docId >= lastDocID : "docID must be >= " + lastDocID + " but was: " + docId;
    lastDocID = docId;
    return true;
  }

  /**
   * Called once per inverted token. This is the primary entry point (for first TermsHash); postings
   * use this API.
   */
  void add(BytesRef termBytes, final int docID) throws IOException {
    assert assertDocId(docID);
    // We are first in the chain so we must "intern" the
    // term text into textStart address
    // Get the text & hash of this term.
    int termID = bytesHash.add(termBytes);
    // System.out.println("add term=" + termBytesRef.utf8ToString() + " doc=" + docState.docID + "
    // termID=" + termID);
    if (termID >= 0) { // New posting
      // Init stream slices
      initStreamSlices(termID, docID);
    } else {
      termID = positionStreamSlice(termID, docID);
    }
    if (doNextCall) {
      nextPerField.add(postingsArray.textStarts[termID], docID);
    }
  }

  private int positionStreamSlice(int termID, final int docID) throws IOException {
    termID = (-termID) - 1;
    int intStart = postingsArray.addressOffset[termID];
    termStreamAddressBuffer = intPool.buffers[intStart >> IntBlockPool.INT_BLOCK_SHIFT];
    streamAddressOffset = intStart & IntBlockPool.INT_BLOCK_MASK;
    addTerm(termID, docID);
    return termID;
  }

  final void writeByte(int stream, byte b) {
    int streamAddress = streamAddressOffset + stream;
    int upto = termStreamAddressBuffer[streamAddress];
    byte[] bytes = bytePool.buffers[upto >> ByteBlockPool.BYTE_BLOCK_SHIFT];
    assert bytes != null;
    int offset = upto & ByteBlockPool.BYTE_BLOCK_MASK;
    if (bytes[offset] != 0) {
      // End of slice; allocate a new one
      offset = allocSlice(bytePool, bytes, offset);
      bytes = bytePool.buffer;
      termStreamAddressBuffer[streamAddress] = offset + bytePool.byteOffset;
    }
    bytes[offset] = b;
    (termStreamAddressBuffer[streamAddress])++;
  }

  final void writeBytes(int stream, byte[] b, int offset, int len) {
    final int end = offset + len;
    int streamAddress = streamAddressOffset + stream;
    int upto = termStreamAddressBuffer[streamAddress];
    byte[] slice = bytePool.buffers[upto >> ByteBlockPool.BYTE_BLOCK_SHIFT];
    assert slice != null;
    int sliceOffset = upto & ByteBlockPool.BYTE_BLOCK_MASK;

    while (slice[sliceOffset] == 0 && offset < end) {
      slice[sliceOffset++] = b[offset++];
      (termStreamAddressBuffer[streamAddress])++;
    }

    while (offset < end) {
      int offsetAndLength = allocKnownSizeSlice(bytePool, slice, sliceOffset);
      sliceOffset = offsetAndLength >> 8;
      int sliceLength = offsetAndLength & 0xff;
      slice = bytePool.buffer;
      int writeLength = Math.min(sliceLength - 1, end - offset);
      System.arraycopy(b, offset, slice, sliceOffset, writeLength);
      sliceOffset += writeLength;
      offset += writeLength;
      termStreamAddressBuffer[streamAddress] = sliceOffset + bytePool.byteOffset;
    }
  }

  // Size of each slice.  These arrays should be at most 16
  // elements (index is encoded with 4 bits).  First array
  // is just a compact way to encode X+1 with a max.  Second
  // array is the length of each slice, ie first slice is 5
  // bytes, next slice is 14 bytes, etc.

  /**
   * An array holding the offset into the {@link #LEVEL_SIZE_ARRAY} to quickly navigate to the next
   * slice level.
   */
  static final int[] NEXT_LEVEL_ARRAY = {1, 2, 3, 4, 5, 6, 7, 8, 9, 9};

  /** An array holding the level sizes for byte slices. */
  static final int[] LEVEL_SIZE_ARRAY = {5, 14, 20, 30, 40, 40, 80, 80, 120, 200};

  /** The first level size for new slices */
  static final int FIRST_LEVEL_SIZE = LEVEL_SIZE_ARRAY[0];

  /**
   * Allocates a new slice with the given size. As each slice is filled with 0's initially, we mark
   * the end with a non-zero byte. This way we don't need to record its length and instead allocate
   * new slice once they hit a non-zero byte.
   */
  // pkg private for access by tests
  static int newSlice(ByteBlockPool bytePool, final int size, final int level) {
    assert LEVEL_SIZE_ARRAY[level] == size;
    // Maybe allocate another block
    if (bytePool.byteUpto > ByteBlockPool.BYTE_BLOCK_SIZE - size) {
      bytePool.nextBuffer();
    }
    final int upto = bytePool.byteUpto;
    bytePool.byteUpto += size;
    bytePool.buffer[bytePool.byteUpto - 1] = (byte) (16 | level);
    return upto;
  }

  /**
   * Creates a new byte slice with the given starting size and returns the slices offset in the
   * pool.
   */
  // pkg private for access by tests
  static int allocSlice(ByteBlockPool bytePool, final byte[] slice, final int upto) {
    return allocKnownSizeSlice(bytePool, slice, upto) >> 8;
  }

  /**
   * Create a new byte slice with the given starting size return the slice offset in the pool and
   * length. The lower 8 bits of the returned int represent the length of the slice, and the upper
   * 24 bits represent the offset.
   */
  // pkg private for access by tests
  static int allocKnownSizeSlice(ByteBlockPool bytePool, final byte[] slice, final int upto) {
    // The idea is to allocate the next slice and then write the address of the new slice
    // into the last 4 bytes of the previous slice (the "forwarding address").
    final int level = slice[upto] & 15;
    final int newLevel = NEXT_LEVEL_ARRAY[level];
    final int newSize = LEVEL_SIZE_ARRAY[newLevel];

    final int newUpto = newSlice(bytePool, newSize, newLevel);
    final int offset = newUpto + bytePool.byteOffset;

    // Copy forward the past 3 bytes (which we are about to overwrite with the forwarding address).
    // We actually copy 4 bytes at once since VarHandles make it cheap.
    final int past3Bytes = ((int) BitUtil.VH_LE_INT.get(slice, upto - 3)) & 0xFFFFFF;
    // Ensure we're not changing the content of `buffer` by setting 4 bytes instead of 3. This
    // should never happen since the next `newSize` bytes must be equal to 0.
    assert bytePool.buffer[newUpto + 3] == 0;
    BitUtil.VH_LE_INT.set(bytePool.buffer, newUpto, past3Bytes);

    // Write forwarding address at end of last slice:
    BitUtil.VH_LE_INT.set(slice, upto - 3, offset);

    return ((newUpto + 3) << 8) | (newSize - 3);
  }

  final void writeVInt(int stream, int i) {
    assert stream < streamCount;
    while ((i & ~0x7F) != 0) {
      writeByte(stream, (byte) ((i & 0x7f) | 0x80));
      i >>>= 7;
    }
    writeByte(stream, (byte) i);
  }

  final TermsHashPerField getNextPerField() {
    return nextPerField;
  }

  final String getFieldName() {
    return fieldName;
  }

  private static final class PostingsBytesStartArray extends BytesStartArray {

    private final TermsHashPerField perField;
    private final Counter bytesUsed;

    private PostingsBytesStartArray(TermsHashPerField perField, Counter bytesUsed) {
      this.perField = perField;
      this.bytesUsed = bytesUsed;
    }

    @Override
    public int[] init() {
      if (perField.postingsArray == null) {
        perField.postingsArray = perField.createPostingsArray(2);
        perField.newPostingsArray();
        bytesUsed.addAndGet(
            perField.postingsArray.size * (long) perField.postingsArray.bytesPerPosting());
      }
      return perField.postingsArray.textStarts;
    }

    @Override
    public int[] grow() {
      ParallelPostingsArray postingsArray = perField.postingsArray;
      final int oldSize = perField.postingsArray.size;
      postingsArray = perField.postingsArray = postingsArray.grow();
      perField.newPostingsArray();
      bytesUsed.addAndGet(postingsArray.bytesPerPosting() * (long) (postingsArray.size - oldSize));
      return postingsArray.textStarts;
    }

    @Override
    public int[] clear() {
      if (perField.postingsArray != null) {
        bytesUsed.addAndGet(
            -(perField.postingsArray.size * perField.postingsArray.bytesPerPosting()));
        perField.postingsArray = null;
        perField.newPostingsArray();
      }
      return null;
    }

    @Override
    public Counter bytesUsed() {
      return bytesUsed;
    }
  }

  @Override
  public final int compareTo(TermsHashPerField other) {
    return fieldName.compareTo(other.fieldName);
  }

  /** Finish adding all instances of this field to the current document. */
  void finish() throws IOException {
    if (nextPerField != null) {
      nextPerField.finish();
    }
  }

  final int getNumTerms() {
    return bytesHash.size();
  }

  /**
   * Start adding a new field instance; first is true if this is the first time this field name was
   * seen in the document.
   */
  boolean start(IndexableField field, boolean first) {
    if (nextPerField != null) {
      doNextCall = nextPerField.start(field, first);
    }
    return true;
  }

  /** Called when a term is seen for the first time. */
  abstract void newTerm(int termID, final int docID) throws IOException;

  /** Called when a previously seen term is seen again. */
  abstract void addTerm(int termID, final int docID) throws IOException;

  /** Called when the postings array is initialized or resized. */
  abstract void newPostingsArray();

  /** Creates a new postings array of the specified size. */
  abstract ParallelPostingsArray createPostingsArray(int size);
}
