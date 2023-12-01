package org.apache.lucene.util.iouring;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.io.Closeable;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

public class IoUring implements Closeable {

  public final MemorySegment ring;
  private final AtomicLong counter = new AtomicLong();
  private final Map<Long, CompletableFuture<Void>> futures = new ConcurrentHashMap<>();

  private IoUring(MemorySegment ring) {
    this.ring = ring;
  }

  public static FileFactory factory(Path file, boolean directIO) {
    return FileFactory.create(file, directIO);
  }

  public static IoUring create(Path file, int entries) {
    MemorySegment ring;
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment path = arena.allocateUtf8String(file.toString());
      ring = WrappedLib.initRing(path, entries);
    }
    return new IoUring(ring);
  }

  static IoUring create(int fd, int entries) {
    MemorySegment ring = WrappedLib.initRing(fd, entries);
    return new IoUring(ring);
  }

  public CompletableFuture<Void> prepare(MemorySegment buffer, int numbytes, long offset) {
    long id = counter.getAndIncrement();
    WrappedLib.prepRead(ring, id, buffer, numbytes, offset);
    CompletableFuture<Void> future = new CompletableFuture<>();
    futures.put(id, future);
    return future;
  }

  public void submit() {
    WrappedLib.submitRequests(ring);
  }

  public void awaitAll() {
    int i = 0;
    while (true) {
      boolean empty = await();
      if (empty) {
        return;
      }
    }
  }

  public boolean await() {
    MemorySegment result = WrappedLib.waitForRequest(ring);
    int res = (int) WrappedLib.WRAPPED_RESULT_RES_HANDLE.get(result);
    long id = (long) WrappedLib.WRAPPED_RESULT_USER_DATA_HANDLE.get(result);
    WrappedLib.completeRequest(ring, result);

    CompletableFuture<Void> future = futures.remove(id);
    if (future == null) {
      throw new RuntimeException("couldn't find future for id " + id);
    }

    if (res < 0) {
      future.completeExceptionally(new IOException("error reading file, errno: " + res));
    } else {
      future.complete(null);
    }

    return futures.isEmpty();
  }

  @Override
  public void close() {
    WrappedLib.closeRing(ring);
  }

  public static class FileFactory implements Closeable {

    private static final MethodHandle OPEN;
    private static final MethodHandle CLOSE;
    private static final MethodHandle ERRNO;

    static {
      var linker = Linker.nativeLinker();
      var stdlib = linker.defaultLookup();

      OPEN =
          linker.downcallHandle(
              stdlib.find("open").get(),
              FunctionDescriptor.of(
                  JAVA_INT, /* returns int */
                  ADDRESS, /* const char *pathname */
                  JAVA_INT /* int flags */));

      CLOSE =
          linker.downcallHandle(
              stdlib.find("close").get(),
              FunctionDescriptor.of(JAVA_INT, /* returns int */ JAVA_INT /* int fd */));

      ERRNO = linker.downcallHandle(stdlib.find("errno").get(), FunctionDescriptor.of(JAVA_INT));
    }

    private final int fd;

    private FileFactory(int fd) {
      this.fd = fd;
    }

    static FileFactory create(Path path, boolean direct) {
      int flags = direct ? 16384 : 0;
      int fd = open(path, 0);
      return new FileFactory(fd);
    }

    public IoUring create(int entries) {
      return IoUring.create(fd, entries);
    }

    @Override
    public void close() {
      close(fd);
    }

    private static int open(Path path, int flags) {
      int result;
      try (Arena arena = Arena.ofConfined()) {
        MemorySegment pathname = arena.allocateUtf8String(path.toString());
        result = (int) OPEN.invokeExact(pathname, flags);
      } catch (Throwable t) {
        throw new RuntimeException("caught error invoking open()", t);
      }

      if (result > 0) {
        return result;
      }

      processErrnoResult(result, "open");
      return -1;
    }

    private static void close(int fd) {
      int result;
      try {
        result = (int) CLOSE.invokeExact(fd);
      } catch (Throwable t) {
        throw new RuntimeException("caught exception invoking close()", t);
      }

      processErrnoResult(result, "open");
    }

    private static void processErrnoResult(int result, String name) {
      if (result == 0) {
        return;
      }

      int errno;
      try {
        errno = (int) ERRNO.invokeExact();
      } catch (Throwable t) {
        throw new RuntimeException("caught exception invoking errno", t);
      }

      throw new RuntimeException("got error calling " + name + ", errno " + errno);
    }
  }
}
