// main.cpp -- a tour of the memory-buffer suite (#177).
//
// The buffer suite binds a whole alias prelude from one place. example/warp_reduce
// shows the terse single-policy convenience (device_buffers); this shows the other
// entry point -- the policy MAP -- which is what you reach for when the alloc and
// free slots want different policies, together with the views the suite pairs to
// their buffers so a view's policies can never drift from its buffer's.
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

// Two policies -- the whole point of the map is that the two slots need not agree.
// alloc aborts, because a failed allocation is unrecoverable here; free only
// warns, because it runs from a destructor and must never throw or abort a
// teardown (the free slot requires a nothrow_error_policy -- both of these are).
// Each is a template on the error type, so one policy serves every error family.
template<typename E>
struct AbortPolicy {
  using error_type = E;
  void handle_error(const E error, std::source_location loc) noexcept {
    if (error != ext::success_code<E>()) {
      std::println(stderr, "alloc error at {}:{}: {} ({})", loc.file_name(), loc.line(),
                   ext::error_name(error), ext::error_string(error));
      std::abort();
    }
  }
};
template<typename E>
struct WarnPolicy {
  using error_type = E;
  void handle_error(const E error, std::source_location loc) noexcept {
    if (error != ext::success_code<E>())
      std::println(stderr, "free warning at {}:{}: {} ({})", loc.file_name(), loc.line(),
                   ext::error_name(error), ext::error_string(error));
  }
};

// The policy map: abort on allocation, warn on free -- for every error family the
// suite touches (stdHostMemoryError_t for host, wwrError_t for the GPU kinds). The
// single-policy convenience device_buffers<P, H> uses one policy for both slots
// and so cannot express this split; the map is the escape hatch that can.
struct AllocAbortFreeWarn {
  template<typename E>
  using alloc = AbortPolicy<E>;
  template<typename E>
  using free = WarnPolicy<E>;
};

// The consumer's own handle -- the library ships none, only the device_handle
// ladder it must satisfy (see src/extension/handle/device_handle.cppm). The stream
// tier (a device index plus an owned stream) routes device allocations and copies
// through wwrMallocAsync on that stream. Stream is not a buffer, so it names its
// own policies directly rather than through the suite.
using Stream =
    ext::StreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>>;
struct DeviceHandle {
  explicit DeviceHandle(int dev = 0) : dev_(dev), stream_(dev) {}
  int dev_idx() const noexcept { return dev_; }
  Stream &stream() noexcept { return stream_; }
  const Stream &stream() const noexcept { return stream_; }
  int dev_;
  Stream stream_;
};

// The whole buffer prelude in one line: host/pinned/unified/device and each of
// their views, all bound to the map above and this handle. From here the names are
// Buf::host<T>, Buf::device<T>, Buf::host_view<T>, and so on.
using Buf = ext::device_buffer_suite<AllocAbortFreeWarn, DeviceHandle>;

namespace {
constexpr std::size_t kCount = 4096;
} // namespace

int main() {
  // wwrGetDevice, not a device COUNT: one reachable device is the whole question.
  int device = 0;
  if (wwrGetDevice(&device) != wwrSuccess) {
    std::println(stderr, "no GPU available -- this example needs a device to RUN, "
                         "though building it is what proves it compiles");
    return 77; // ctest's conventional "skipped"
  }

  auto handle = std::make_shared<DeviceHandle>();
  Stream &stream = handle->stream();

  // A host buffer filled 0, 1, 2, ...; a device buffer to round-trip through; and
  // a second host buffer for the result. Each is one alias off the suite -- no
  // per-kind prelude, and host speaks stdHostMemoryError_t while device speaks
  // wwrError_t, chosen by the suite rather than spelled here.
  Buf::host<float> host(kCount);
  for (std::size_t i = 0; i < kCount; ++i)
    host.data()[i] = static_cast<float>(i);

  Buf::device<float> device_buf(kCount, handle);
  Buf::host<float> result(kCount);

  if (ext::copy(device_buf, host, stream.get()) != wwrSuccess) {
    std::println(stderr, "host -> device copy failed");
    return 1;
  }
  if (ext::copy(result, device_buf, stream.get()) != wwrSuccess) {
    std::println(stderr, "device -> host copy failed");
    return 1;
  }
  if (ext::sync(stream) != wwrSuccess) {
    std::println(stderr, "stream synchronize failed -- the copies did not complete");
    return 1;
  }

  // A non-owning view over the back half of the result. Its type is
  // Buf::host_view<float>, which the suite derives from Buf::host<float> -- a view
  // whose policies drifted from its buffer would not name the same type, so the
  // mismatch that a hand-written prelude risks cannot arise here.
  Buf::host_view<float> tail(result, kCount / 2, kCount / 2);
  bool ok = tail.num_elements() == kCount / 2;
  for (std::size_t i = 0; i < tail.num_elements() && ok; ++i)
    ok = tail.data()[i] == static_cast<float>(kCount / 2 + i);
  if (!ok) {
    std::println(stderr, "round-trip or view mismatch");
    return 1;
  }

  std::println("buffer_suite: round-tripped {} floats through the device; "
               "a {}-element host_view over the back half checks out",
               kCount, tail.num_elements());
  return 0;
}
