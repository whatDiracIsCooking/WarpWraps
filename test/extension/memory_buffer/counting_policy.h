#pragma once

// counting_policy.h - shared error-recording policy and counted-buffer aliases
//
// The failure-path buffer tests need an error policy that records rather than
// aborts, so the counts can be read back through alloc_policy(). Both the
// size-overflow rejection suite (buffer_tests.cpp) and the allocation-failure
// suite (allocation_failure_tests.cpp) use it, so it lives here rather than in
// either TU.
//
// This header references types from wwr.extension.memory_buffer and the std
// module, so it must be included AFTER those imports -- the failure-path TUs
// are plain .cpp files, not module units, so there is no global module fragment
// to include it into. It pulls in no headers of its own, so nothing here
// conflicts with `import std;`.

namespace wwr::extension::test {

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test error policy
//
// AbortPolicy aborts, so it cannot be used to observe the failure
// paths. This one records instead, which is all the error_policy concept
// requires, and is stored by value in the buffer so the counts can be read
// back through alloc_policy().
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

template<typename T>
class CountingPolicy {
public:
  using error_type = T;

  void handle_error(const T error, std::source_location) noexcept {
    ++count_;
    last_ = error;
  }
  std::size_t count() const noexcept { return count_; }
  T last() const noexcept { return last_; }

private:
  std::size_t count_ = 0;
  T last_{};
};

using HostPolicy = CountingPolicy<stdHostMemoryError_t>;
using GpuPolicy = CountingPolicy<wwrError_t>;

template<typename T>
using CountedHostBuffer = HostBufferWrapper<T, HostPolicy>;
template<typename T>
using CountedDeviceBuffer = DeviceBufferWrapper<T, GpuPolicy>;
template<typename T>
using CountedHostView = BufferViewWrapper<T, MemoryKind::Host, HostPolicy>;

} // namespace wwr::extension::test
