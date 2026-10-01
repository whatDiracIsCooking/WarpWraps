// main.cpp -- the ready-made pool-tier device handle (kit::DeviceHandle).
//
// example/warp_reduce and example/buffer_suite both back their device buffers on
// a hand-named ext::StreamWrapper -- the stream tier, which routes allocations
// through wwrMallocAsync. This shows the other end of the ladder: kit::DeviceHandle,
// the opt-in helper that owns a stream AND a memory pool on one device, so a buffer
// built on it draws from that pool (wwrMallocFromPoolAsync) instead. It is the
// answer to "I don't want to hand-roll a handle at all" -- one line, no wrapper
// struct, and the device's static properties come for free.
//
// Nothing here names a backend, and there is no kernel: it exercises only the
// host-callable RAII + copy surface, so it is a single ordinary host module TU.
// Running it needs a GPU; building it does not -- which is what
// devtools/cross-backend-check.sh compiles for the other backend.

// stderr is a FILE*, which `import std;` does not give you (the std module exports
// std:: names, not the C library's objects). An #include beside an import is fine
// in a plain translation unit like this one.
#include <cstdio>

import std;

import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.runtime;
import wwr.extension.memory_buffer;

using namespace wwr;
namespace ext = wwr::extension;
namespace kit = wwr::extension::kit; // AbortPolicy, DeviceHandle, device_buffers

// One policy across the handle's three slots -- create, destroy, device-access --
// the kit's ready abort-on-error policy. A consumer wanting recovery brings their
// own; the handle bakes none (that is the no-forced-policy invariant #93/#97
// established, which is exactly why this type is policy-templated).
using Abort = kit::AbortPolicy<wwrError_t>;

// The whole handle in one line. kit::DeviceHandle is the fullest rung of the
// device_handle ladder: it owns a StreamWrapper and a MemPoolWrapper created on
// the same device, and forwards dev_idx() + stream() + pool(). That is what makes
// a buffer backed by it a device_handle_pool allocation -- wwrMallocFromPoolAsync
// from pool() on stream() -- where the StreamWrapper the other examples use is only
// a device_handle_stream (wwrMallocAsync, no pool). Unlike StreamWrapper it also
// queries the device's static properties once at construction, exposed as props().
using DeviceHandle = kit::DeviceHandle<Abort, Abort, Abort>;

// The buffer prelude, single-policy convenience, backed by the pool-tier handle
// above. Buf::device<float> now allocates from the handle's owned pool; the host
// side is unchanged (stdHostMemoryError_t, chosen by the suite).
using Buf = kit::device_buffers<kit::AbortPolicy, DeviceHandle>;

namespace {
constexpr std::size_t kCount = 4096;
} // namespace

int main() {
  // wwrGetDevice, not a device COUNT: one reachable device is the whole question.
  // It must come BEFORE the handle is constructed -- DeviceHandle queries
  // wwrGetDeviceProperties in its constructor, which on a driverless box would run
  // the AbortPolicy and abort instead of letting us skip cleanly.
  int device = 0;
  if (wwrGetDevice(&device) != wwrSuccess) {
    std::println(stderr, "no GPU available -- this example needs a device to RUN, "
                         "though building it is what proves it compiles");
    return 77; // ctest's conventional "skipped"
  }

  // Device 0, with its stream and pool. props() carries the properties queried at
  // construction -- the one thing the stream-tier handle cannot hand you, and the
  // reason you might reach for this tier even when a pool is incidental.
  auto handle = std::make_shared<DeviceHandle>();
  const wwrDeviceProp &props = handle->props(); // stream()/pool() convert to the raw
  const wwrStream_t stream = handle->stream();  // handles via operator T()
  // props.name is a fixed char[]; decay it to a const char* so std::format stops
  // at the NUL rather than printing the whole padded array.
  std::println("device_handle: device {} is \"{}\" -- {} SMs/CUs, {} MiB global", device,
               static_cast<const char *>(props.name), props.multiProcessorCount,
               props.totalGlobalMem / (1024 * 1024));

  // A host buffer filled 0, 1, 2, ...; and a second host buffer for the result.
  Buf::host<float> host(kCount);
  for (std::size_t i = 0; i < kCount; ++i)
    host.data()[i] = static_cast<float>(i);
  Buf::host<float> result(kCount);

  // The device buffer lives in its own scope so it frees -- returning its memory
  // to the pool -- before the second allocation below, which is what makes that a
  // reuse rather than a concurrent second draw.
  {
    Buf::device<float> device_buf(kCount, handle); // wwrMallocFromPoolAsync on the pool

    if (ext::copy(device_buf, host, stream) != wwrSuccess) {
      std::println(stderr, "host -> device copy failed");
      return 1;
    }
    if (ext::copy(result, device_buf, stream) != wwrSuccess) {
      std::println(stderr, "device -> host copy failed");
      return 1;
    }
    if (ext::sync(stream) != wwrSuccess) {
      std::println(stderr, "stream synchronize failed -- the copies did not complete");
      return 1;
    }
  }

  bool ok = true;
  for (std::size_t i = 0; i < kCount && ok; ++i)
    ok = result.data()[i] == static_cast<float>(i);
  if (!ok) {
    std::println(stderr, "round-trip mismatch");
    return 1;
  }

  // A second device allocation of the same size, now that the first has freed its
  // memory back to the pool. Because the pool holds freed memory (its release
  // threshold is not crossed) rather than handing it back to the OS, this draw is
  // served from the pool without a fresh driver allocation -- the whole point of
  // the pool tier over the stream tier. It is semantics, not something this example
  // can measure: no pool-attribute getter is exported, so there is nothing honest
  // to assert here beyond that the allocation succeeds.
  {
    Buf::device<float> reused(kCount, handle);
    if (ext::copy(reused, host, stream) != wwrSuccess || ext::sync(stream) != wwrSuccess) {
      std::println(stderr, "pool re-allocation round-trip failed");
      return 1;
    }
  }

  std::println("device_handle: round-tripped {} floats through a pool-backed buffer, "
               "then reused the pool for a second allocation",
               kCount);
  return 0;
}
