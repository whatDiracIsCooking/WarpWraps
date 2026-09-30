// main.cpp -- the host half of the warp-reduction example.
//
// Allocates, fills, launches warp_reduce_sum and checks the number that comes
// back. Nothing here names a backend: the buffers and the stream are the
// extension layer's, the kernel behind warp_reduce_sum is written once against
// cooperative_groups.cuh, and the same source builds and runs on CUDA and HIP.
//
// Running it needs a GPU; building it does not, which is what
// devtools/cross-backend-check.sh compiles for the other backend.

// stderr is a FILE*, which `import std;` does not give you -- the std module
// exports the std:: names, not the C library's macros and objects. An ordinary
// #include beside an import is fine in a plain translation unit like this one;
// it is only inside a MODULE unit's global module fragment that this project's
// headers and `import std;` collide.
#include <cstdio>

#include "warp_reduce_bridge.h"

import std;

import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.runtime;
import wwr.extension.memory_buffer;

using namespace wwr;
namespace ext = wwr::extension;

// The extension layer ships the RAII wrappers themselves, but no error policy --
// policy choice belongs to the consumer. This is this example's own: print to
// stderr and abort on any failure. A consumer wanting recovery or logging writes
// a different policy and passes it as the wrappers' error-policy arguments.
template<typename T>
struct AbortPolicy {
  using error_type = T;
  void handle_error(const T error, std::source_location loc) noexcept {
    if (error != ext::success_code<T>()) {
      std::println(stderr, "GPU error at {}:{} in {}: {} ({})", loc.file_name(), loc.line(),
                   loc.function_name(), ext::error_name(error), ext::error_string(error));
      std::abort();
    }
  }
};

// StreamWrapper names its error policies explicitly -- it is not a buffer, so the
// buffer suite below does not bind it. The device-bound wrappers name a wwrError_t
// device-access policy for their device set/get calls.
using Stream =
    ext::StreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>>;

// A DeviceBuffer is backed by whatever handle type the consumer provides: the
// library ships no concrete one, only the device_handle capability ladder the
// handle must satisfy (see src/extension/handle/device_handle.cppm). This is
// this example's own -- the stream tier (a device index plus an owned stream),
// which routes the buffer's allocations through wwrMallocAsync on that stream.
struct DeviceHandle {
  explicit DeviceHandle(int dev = 0) : dev_(dev), stream_(dev) {}
  int dev_idx() const noexcept { return dev_; }
  Stream &stream() noexcept { return stream_; }
  const Stream &stream() const noexcept { return stream_; }
  int dev_;
  Stream stream_;
};

// The whole buffer prelude in one line. device_buffers is the single-policy
// convenience over the suite: it binds host/device (and pinned/unified/views,
// unused here) to this example's AbortPolicy and DeviceHandle, choosing the error
// family each kind speaks -- stdHostMemoryError_t for host, wwrError_t for device,
// including the device-access policy -- so no per-kind alias is written by hand.
using Buf = ext::device_buffers<AbortPolicy, DeviceHandle>;

namespace {

// One million ones: the exact sum is representable in float (integers are, up
// to 2^24), so the check needs no tolerance for the reduction itself.
constexpr std::size_t kCount = 1'000'000;

} // namespace

int main() {
  // wwrGetDevice, not a device COUNT: the wwr* layer re-exports the former
  // and not the latter, and one reachable device is the whole question here.
  int device = 0;
  if (wwrGetDevice(&device) != wwrSuccess) {
    std::println(stderr, "no GPU available -- this example needs a device to RUN, "
                         "though building it is what proves it compiles");
    return 77; // ctest's conventional "skipped"
  }

  // A DeviceBuffer is drawn from a shared handle; this one's owned stream is
  // what this example submits its copies and kernel on.
  auto device_handle = std::make_shared<DeviceHandle>();
  Stream &stream = device_handle->stream();

  Buf::host<float> host(kCount);
  std::fill_n(host.data(), kCount, 1.0F);

  Buf::device<float> input(kCount, device_handle);
  Buf::device<float> output(1, device_handle);
  Buf::host<float> result(1);

  if (ext::copy(input, host, stream.get()) != wwrSuccess) {
    std::println(stderr, "host -> device copy failed");
    return 1;
  }

  example::warp_reduce_sum(stream.get(), kCount, input.data(), output.data());

  if (ext::copy(result, output, stream.get()) != wwrSuccess) {
    std::println(stderr, "device -> host copy failed");
    return 1;
  }
  if (ext::sync(stream) != wwrSuccess) {
    std::println(stderr, "stream synchronize failed -- the kernel did not run");
    return 1;
  }

  const float expected = static_cast<float>(kCount);
  if (result.data()[0] != expected) {
    std::println(stderr, "reduction mismatch: got {}, want {}", result.data()[0], expected);
    return 1;
  }

  std::println("warp_reduce: summed {} elements to {:.0f}", kCount, result.data()[0]);
  return 0;
}
