package org.apache.lucene.util.iouring;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.nio.file.Path;

class WrappedLib {

  private static final MemoryLayout WRAPPED_IO_URING =
      MemoryLayout.structLayout(
          ADDRESS.withName("wrapped"), ADDRESS.withName("cqe"), JAVA_INT.withName("fd"));

  private static final MemoryLayout WRAPPED_RESULT =
      MemoryLayout.structLayout(JAVA_LONG.withName("user_data"), JAVA_INT.withName("res"));

  static final VarHandle WRAPPED_RESULT_RES_HANDLE =
      WRAPPED_RESULT.varHandle(MemoryLayout.PathElement.groupElement("res"));
  static final VarHandle WRAPPED_RESULT_USER_DATA_HANDLE =
      WRAPPED_RESULT.varHandle(MemoryLayout.PathElement.groupElement("user_data"));

  private static final String INIT_RING_FROM_PATH = "wrapped_io_uring_init_from_path";
  private static final MethodHandle INIT_RING_FROM_PATH_HANDLE;

  private static final String INIT_RING_FROM_FD = "wrapped_io_uring_init_from_fd";
  private static final MethodHandle INIT_RING_FROM_FD_HANDLE;

  private static final String PREP_READ = "wrapped_io_uring_prep_read";
  private static final MethodHandle PREP_READ_HANDLE;

  private static final String SUBMIT_REQUESTS = "wrapped_io_uring_submit_requests";
  private static final MethodHandle SUBMIT_REQUESTS_HANDLE;

  private static final String WAIT_FOR_REQUESTS = "wrapped_io_uring_wait_for_request";
  private static final MethodHandle WAIT_FOR_REQUESTS_HANDLE;

  private static final String COMPLETE_REQUEST = "wrapped_io_uring_complete_request";
  private static final MethodHandle COMPLETE_REQUEST_HANDLE;

  private static final String CLOSE_RING = "wrapped_io_uring_close_ring";
  private static final MethodHandle CLOSE_RING_HANDLE;

  static {
    boolean isLinux = System.getProperty("os.name").toLowerCase().contains("linux");
    if (isLinux) {
      Arena global = Arena.global();
      SymbolLookup uringLookup =
          SymbolLookup.libraryLookup(
              Path.of(System.getProperty("user.dir")).resolve("src/main/c/libwrappeduring.so"),
              global);
      var linker = Linker.nativeLinker();

      INIT_RING_FROM_PATH_HANDLE =
          linker.downcallHandle(
              uringLookup.find(INIT_RING_FROM_PATH).get(),
              FunctionDescriptor.of(
                  ADDRESS, /* returns *wrapped_io_uring */
                  ADDRESS, /* char *path */
                  JAVA_INT /* unsigned entries */));

      INIT_RING_FROM_FD_HANDLE =
          linker.downcallHandle(
              uringLookup.find(INIT_RING_FROM_FD).get(),
              FunctionDescriptor.of(
                  ADDRESS, /* returns *wrapped_io_uring */
                  JAVA_INT, /* int fd */
                  JAVA_INT /* unsigned entries */));

      PREP_READ_HANDLE =
          linker.downcallHandle(
              uringLookup.find(PREP_READ).get(),
              FunctionDescriptor.ofVoid(
                  ADDRESS, /* wrapped_io_uring *ring */
                  JAVA_LONG, /* uint64_t user_data */
                  ADDRESS, /* void *buf */
                  JAVA_INT, /* unsigned nbytes */
                  JAVA_LONG /* off_t offset */));

      SUBMIT_REQUESTS_HANDLE =
          linker.downcallHandle(
              uringLookup.find(SUBMIT_REQUESTS).get(),
              FunctionDescriptor.ofVoid(ADDRESS /* wrapped_io_uring *ring */));

      WAIT_FOR_REQUESTS_HANDLE =
          linker.downcallHandle(
              uringLookup.find(WAIT_FOR_REQUESTS).get(),
              FunctionDescriptor.of(
                  ADDRESS, /* returns wrapped_result* */ ADDRESS /* wrapped_io_uring *ring */));

      COMPLETE_REQUEST_HANDLE =
          linker.downcallHandle(
              uringLookup.find(COMPLETE_REQUEST).get(),
              FunctionDescriptor.ofVoid(
                  ADDRESS, /* wrapped_io_uring *ring */ ADDRESS /* wrapped_result *result */));

      CLOSE_RING_HANDLE =
          linker.downcallHandle(
              uringLookup.find(CLOSE_RING).get(),
              FunctionDescriptor.ofVoid(ADDRESS /* wrapped_io_uring *ring */));
    } else {
      INIT_RING_FROM_PATH_HANDLE = null;
      INIT_RING_FROM_FD_HANDLE = null;
      PREP_READ_HANDLE = null;
      SUBMIT_REQUESTS_HANDLE = null;
      WAIT_FOR_REQUESTS_HANDLE = null;
      COMPLETE_REQUEST_HANDLE = null;
      CLOSE_RING_HANDLE = null;
    }
  }

  static MemorySegment initRing(MemorySegment path, int entries) {
    MemorySegment uninterpreted;
    try {
      uninterpreted = (MemorySegment) INIT_RING_FROM_PATH_HANDLE.invokeExact(path, entries);
    } catch (Throwable t) {
      throw new RuntimeException(invokeErrorString(INIT_RING_FROM_PATH), t);
    }

    if (uninterpreted == null) {
      throw new RuntimeException("ring could not be created");
    }

    return uninterpreted.reinterpret(WRAPPED_IO_URING.byteSize());
  }

  static MemorySegment initRing(int fd, int entries) {
    MemorySegment uninterpreted;
    try {
      uninterpreted = (MemorySegment) INIT_RING_FROM_FD_HANDLE.invokeExact(fd, entries);
    } catch (Throwable t) {
      throw new RuntimeException(invokeErrorString(INIT_RING_FROM_FD), t);
    }

    if (uninterpreted == null || uninterpreted.address() == 0) {
      throw new RuntimeException("ring could not be created");
    }

    return uninterpreted.reinterpret(WRAPPED_IO_URING.byteSize());
  }

  static void prepRead(
      MemorySegment ring, long userData, MemorySegment buf, int nbytes, long offset) {
    try {
      PREP_READ_HANDLE.invokeExact(ring, userData, buf, nbytes, offset);
    } catch (Throwable t) {
      throw new RuntimeException(invokeErrorString(PREP_READ), t);
    }
  }

  static void submitRequests(MemorySegment ring) {
    try {
      SUBMIT_REQUESTS_HANDLE.invokeExact(ring);
    } catch (Throwable t) {
      throw new RuntimeException(invokeErrorString(SUBMIT_REQUESTS), t);
    }
  }

  static MemorySegment waitForRequest(MemorySegment ring) {
    MemorySegment uninterpreted;
    try {
      uninterpreted = (MemorySegment) WAIT_FOR_REQUESTS_HANDLE.invokeExact(ring);
    } catch (Throwable t) {
      throw new RuntimeException(invokeErrorString(WAIT_FOR_REQUESTS), t);
    }

    if (uninterpreted == null || uninterpreted.address() == 0) {
      throw new RuntimeException("error waiting for request");
    }

    return uninterpreted.reinterpret(WRAPPED_RESULT.byteSize());
  }

  static void completeRequest(MemorySegment ring, MemorySegment result) {
    try {
      COMPLETE_REQUEST_HANDLE.invokeExact(ring, result);
    } catch (Throwable t) {
      throw new RuntimeException(invokeErrorString(COMPLETE_REQUEST), t);
    }
  }

  static void closeRing(MemorySegment ring) {
    try {
      CLOSE_RING_HANDLE.invokeExact(ring);
    } catch (Throwable t) {
      throw new RuntimeException(invokeErrorString(CLOSE_RING), t);
    }
  }

  private static String invokeErrorString(String name) {
    return "caught excpetion invoking " + name;
  }
}
