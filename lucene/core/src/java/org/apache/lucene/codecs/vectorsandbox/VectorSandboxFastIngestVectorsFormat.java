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

package org.apache.lucene.codecs.vectorsandbox;

import java.io.IOException;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/** Ingests vectors without indexing them, and indexes them on merge */
public final class VectorSandboxFastIngestVectorsFormat extends KnnVectorsFormat {

  static final String META_CODEC_NAME = "VectorSandboxFastIngestVectorsFormatMeta";
  static final String VECTOR_DATA_CODEC_NAME = "VectorSandboxFastIngestVectorsFormatData";
  static final String META_EXTENSION = "fivem";
  static final String VECTOR_DATA_EXTENSION = "fivec";

  public static final int VERSION_START = 0;
  public static final int VERSION_CURRENT = VERSION_START;

  private final KnnVectorsFormat wrapped;

  public VectorSandboxFastIngestVectorsFormat() {
    // FIXME: this is resulting in issues when loading from SPI, it instantiates the vamana format
    // always even if you're trying to wrap something else
    this(new VectorSandboxVamanaVectorsFormat());
  }

  public VectorSandboxFastIngestVectorsFormat(KnnVectorsFormat wrapped) {
    super("VectorSandboxFastIngestVectorsFormat");
    this.wrapped = wrapped;
  }

  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new VectorSandboxFastIngestVectorsWriter(state, wrapped.fieldsWriter(state));
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new VectorSandboxFastIngestVectorsReader(state, wrapped.fieldsReader(state));
  }

  @Override
  public int getMaxDimensions(String fieldName) {
    return 1024;
  }

  @Override
  public String toString() {
    return "VectorSandboxFastIngestVectorsFormat";
  }
}
