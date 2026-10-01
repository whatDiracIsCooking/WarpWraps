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
namespace kit = wwr::extension::kit; // the opt-in ready-made helpers: AbortPolicy, device_buffers

// The kit ships an opt-in abort-on-error policy, kit::AbortPolicy, so this example
// need not hand-roll one. A consumer wanting recovery or logging passes their own
// policy in the wrappers' error-policy slots instead.

// A DeviceBuffer is backed by whatever handle type the consumer provides. For the
// stream tier the library ships no bespoke type: a StreamWrapper already exposes
// dev_idx() and stream(), so it *is* a device_handle_stream (see
// src/extension/handle/device_handle.cppm) and backs the buffer directly, routing
// allocations through wwrMallocAsync on its stream -- no wrapping struct. (The
// fullest tier, with an owned pool, is the shipped DeviceHandle in
// wwr.extension.runtime.) The wrapper names its three error policies, the kit's
// AbortPolicy.
using DeviceHandle = ext::StreamWrapper<kit::AbortPolicy<wwrError_t>, kit::AbortPolicy<wwrError_t>,
                                        kit::AbortPolicy<wwrError_t>>;

// The whole buffer prelude in one line. kit::device_buffers is the single-policy
// convenience over the suite: it binds host/device (and pinned/unified/views,
// unused here) to kit::AbortPolicy and DeviceHandle, choosing the error family each
// kind speaks -- stdHostMemoryError_t for host, wwrError_t for device, including the
// device-access policy -- so no per-kind alias is written by hand.
using Buf = kit::device_buffers<kit::AbortPolicy, DeviceHandle>;

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

  // A DeviceBuffer is drawn from a shared handle; the handle's own stream is what
  // this example submits its copies and kernel on.
  auto device_handle = std::make_shared<DeviceHandle>();
  const wwrStream_t stream = device_handle->stream();

  Buf::host<float> host(kCount);
  std::fill_n(host.data(), kCount, 1.0F);

  Buf::device<float> input(kCount, device_handle);
  Buf::device<float> output(1, device_handle);
  Buf::host<float> result(1);

  if (ext::copy(input, host, stream) != wwrSuccess) {
    std::println(stderr, "host -> device copy failed");
    return 1;
  }

  example::warp_reduce_sum(stream, kCount, input.data(), output.data());

  if (ext::copy(result, output, stream) != wwrSuccess) {
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
