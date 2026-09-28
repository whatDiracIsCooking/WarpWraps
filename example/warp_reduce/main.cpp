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

// The extension layer ships the RAII wrappers themselves, not default-policy
// aliases for them. Each wrapper names its create/alloc and destroy/free error
// policies explicitly -- neither has a default -- so binding them is a one-line
// `using` a consumer writes once, for exactly the names it uses; here they are,
// this example's own, bound to abort-on-failure. Swap in a custom policy pair
// (e.g. DeviceBufferWrapper<T, MyAlloc, MyFree>, GpuStreamWrapper<MyCreate,
// MyDestroy>) to route errors somewhere other than aborting.
template<typename T>
using DeviceBuffer = ext::DeviceBufferWrapper<T, ext::AbortPolicy<wwrError_t>, ext::AbortPolicy<wwrError_t>>;
template<typename T>
using HostBuffer =
    ext::HostBufferWrapper<T, ext::AbortPolicy<ext::stdHostMemoryError_t>,
                           ext::AbortPolicy<ext::stdHostMemoryError_t>>;
using GpuStream = ext::GpuStreamWrapper<ext::AbortPolicy<wwrError_t>, ext::AbortPolicy<wwrError_t>>;

namespace {

// One million ones: the exact sum is representable in float (integers are, up
// to 2^24), so the check needs no tolerance for the reduction itself.
constexpr std::size_t kCount = 1'000'000;

} // namespace

int main() {
  // wwrGetDevice, not a device COUNT: the gpu* layer re-exports the former
  // and not the latter, and one reachable device is the whole question here.
  int device = 0;
  if (wwrGetDevice(&device) != wwrSuccess) {
    std::println(stderr, "no GPU available -- this example needs a device to RUN, "
                         "though building it is what proves it compiles");
    return 77; // ctest's conventional "skipped"
  }

  // A DeviceBuffer is drawn from a shared DeviceHandle now; its default
  // allocation stream is what this example submits its copies and kernel on.
  auto device_handle = std::make_shared<ext::DeviceHandle>();
  GpuStream &stream = device_handle->alloc_stream();

  HostBuffer<float> host(kCount);
  std::fill_n(host.data(), kCount, 1.0F);

  DeviceBuffer<float> input(kCount, device_handle);
  DeviceBuffer<float> output(1, device_handle);
  HostBuffer<float> result(1);

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
