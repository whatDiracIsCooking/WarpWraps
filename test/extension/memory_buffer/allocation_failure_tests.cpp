// allocation_failure_tests.cpp - the deliberately-failing allocation paths
//
// Split out from buffer_tests.cpp because these two cases force a real
// allocation failure, which every sanitizer treats as an error of its own:
// ASAN's allocator aborts on the oversized host malloc (allocation-size-too-big)
// and compute-sanitizer's memcheck counts the intentional
// cudaErrorMemoryAllocation. The AllocationFailureTests ctest entry therefore
// carries the no_sanitizer label and is excluded from the sanitized runs -- see
// this directory's CMakeLists.txt. Keeping them in their own TU (and suite) is
// what lets the size-overflow rejection cases, which allocate nothing and are
// safe under every sanitizer, stay in buffer_tests.cpp and keep running there.
//
// A plain .cpp, not a .cppm, for the reasons buffer_tests.cpp gives.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.runtime;
import wwr.extension.memory_buffer;
import wwr.test.shared.device_handle;

#include "counting_policy.h"

namespace wwr::extension::test {

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Allocation failure: the alloc passes the size check but cannot succeed
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(AllocationFailureTests, FailedAllocationLeavesCoherentEmptyBuffer) {
  // Largest count that passes the size check; the malloc itself cannot
  // succeed. A non-aborting policy must not then memset a null pointer.
  HostPolicy policy;
  CountedHostBuffer<float> buf(CountedHostBuffer<float>::max_num_elements, policy);

  EXPECT_EQ(buf.alloc_policy().count(), std::size_t{1});
  EXPECT_EQ(buf.alloc_policy().last(), stdHostMemAllocFailure);
  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_EQ(buf.num_elements(), std::size_t{0});
  EXPECT_EQ(buf.size_bytes(), std::size_t{0});
}

TEST(AllocationFailureTests, FailedDeviceAllocationLeavesCoherentEmptyBuffer) {
  GpuPolicy policy;
  auto dev = std::make_shared<DeviceHandle>(0);
  CountedDeviceBuffer<float> buf(CountedDeviceBuffer<float>::max_num_elements, dev, policy);

  EXPECT_GE(buf.alloc_policy().count(), std::size_t{1});
  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_EQ(buf.num_elements(), std::size_t{0});

  // Clear the sticky error so later tests see a clean context.
  static_cast<void>(wwrGetLastError());
}

TEST(AllocationFailureTests, FailedPinnedAllocationLeavesCoherentEmptyBuffer) {
  // Largest count that passes the size check; wwrHostAlloc itself cannot
  // satisfy it. As with the host kind, a non-aborting policy must record the
  // failure and leave an empty, null buffer rather than memset a null pointer.
  GpuPolicy policy;
  CountedPinnedBuffer<float> buf(CountedPinnedBuffer<float>::max_num_elements, policy);

  EXPECT_GE(buf.alloc_policy().count(), std::size_t{1});
  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_EQ(buf.num_elements(), std::size_t{0});
  EXPECT_EQ(buf.size_bytes(), std::size_t{0});

  static_cast<void>(wwrGetLastError());
}

TEST(AllocationFailureTests, FailedUnifiedAllocationLeavesCoherentEmptyBuffer) {
  // Same contract for managed memory: wwrMallocManaged fails the oversized
  // request and the buffer stays coherent and empty.
  GpuPolicy policy;
  CountedUnifiedBuffer<float> buf(CountedUnifiedBuffer<float>::max_num_elements, policy);

  EXPECT_GE(buf.alloc_policy().count(), std::size_t{1});
  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_EQ(buf.num_elements(), std::size_t{0});
  EXPECT_EQ(buf.size_bytes(), std::size_t{0});

  static_cast<void>(wwrGetLastError());
}

} // namespace wwr::extension::test
