package org.apache.lucene.util.vamana;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

public class MLock {
  private static final MethodHandle MLOCK;
  private static final MethodHandle MUNLOCK;
  private static final MethodHandle ERRNO;

  static {
    var linker = Linker.nativeLinker();
    var stdlib = linker.defaultLookup();

    MLOCK =
        linker.downcallHandle(
            stdlib.find("mlock").get(),
            FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

    MUNLOCK =
        linker.downcallHandle(
            stdlib.find("munlock").get(),
            FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

    ERRNO =
        linker.downcallHandle(
            stdlib.find("errno").get(), FunctionDescriptor.of(ValueLayout.JAVA_INT));
  }

  public static void lock(MemorySegment segment) {
    int result;
    try {
      result = (int) MLOCK.invokeExact(segment, segment.byteSize());
    } catch (Throwable t) {
      throw new RuntimeException("caught exception invoking mlock", t);
    }

    if (result == 0) {
      return;
    }

    var errno = getErrno();
    throw new RuntimeException("got error calling mlock, errno " + errno);
  }

  public static void unlock(MemorySegment segment) {
    int result;
    try {
      result = (int) MUNLOCK.invokeExact(segment, segment.byteSize());
    } catch (Throwable t) {
      throw new RuntimeException("caught exception invoking munlock", t);
    }

    if (result == 0) {
      return;
    }

    var errno = getErrno();
    throw new RuntimeException("got error calling munlock, errno " + errno);
  }

  private static int getErrno() {
    try {
      return (int) ERRNO.invokeExact();
    } catch (Throwable t) {
      throw new RuntimeException("caught exception invoking errno", t);
    }
  }
}
