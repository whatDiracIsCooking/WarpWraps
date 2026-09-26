// host_memory_tests.cpp - Tests for gpumod.extension.memory_buffer:host_memory
// and the common error-handling infrastructure it instantiates.
//
// Every suite here is host-only: stdHostMemoryError_t and std_malloc/std_free
// compute on the CPU, and the error-policy / gpu_check paths are driven through
// that host error type, so none of this needs a device. The file compiles into
// the same executable as buffer_tests.cpp.
//
// The failure paths of DefaultErrorPolicy and the single-argument gpu_check end
// in std::abort(), so they are covered with death tests -- named *DeathTest so
// GoogleTest runs them before any suite in the binary that touches CUDA.
// gpu_check's custom-policy overload returns false instead of aborting, so that
// branch is covered directly with a non-aborting policy.

#include <gtest/gtest.h>

import std;
import gpumod.extension.common;
import gpumod.extension.memory_buffer;

namespace gpumod::extension::test {

namespace {

// A non-aborting error policy, so gpu_check's failure path returns rather than
// terminating the process. Named differently from buffer_tests.cpp's
// CountingPolicy and kept in an anonymous namespace to avoid any cross-TU
// clash within the shared executable.
template<typename T>
class RecordingPolicy : public BaseErrorPolicy<T> {
public:
  void handle_error(const T error, std::source_location) override {
    ++count_;
    last_ = error;
  }
  int count() const noexcept { return count_; }
  T last() const noexcept { return last_; }

private:
  int count_ = 0;
  T last_{};
};

// A value with no enumerator, to drive the default: arm of the switches.
constexpr auto kBogusCode = static_cast<stdHostMemoryError_t>(999);

} // namespace

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// host_memory.cppm: error-code surface
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(HostMemoryErrorTests, SuccessCodeIsSuccess) {
  EXPECT_EQ(success_code<stdHostMemoryError_t>(), stdHostMemSuccess);
}

TEST(HostMemoryErrorTests, ErrorNameCoversEveryCode) {
  EXPECT_STREQ(error_name(stdHostMemSuccess), "stdHostMemSuccess");
  EXPECT_STREQ(error_name(stdHostMemAllocFailure), "stdHostMemAllocFailure");
  EXPECT_STREQ(error_name(stdHostMemDeallocFailure), "stdHostMemDeallocFailure");
  EXPECT_STREQ(error_name(stdHostMemInvalidValue), "stdHostMemInvalidValue");
  EXPECT_STREQ(error_name(kBogusCode), "UnknownError");
}

TEST(HostMemoryErrorTests, ErrorStringCoversEveryCode) {
  EXPECT_STREQ(error_string(stdHostMemSuccess), "operation completed successfully");
  EXPECT_STREQ(error_string(stdHostMemAllocFailure), "std::malloc failed");
  EXPECT_STREQ(error_string(stdHostMemDeallocFailure), "std::free failed");
  EXPECT_STREQ(error_string(stdHostMemInvalidValue), "invalid parameter value");
  EXPECT_STREQ(error_string(kBogusCode), "unknown error");
}

TEST(HostMemoryErrorTests, StreamInsertionMatchesErrorName) {
  // operator<< is found by ADL and delegates to error_name.
  std::ostringstream os;
  os << stdHostMemAllocFailure;
  EXPECT_EQ(os.str(), "stdHostMemAllocFailure");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// host_memory.cppm: std_malloc / std_free
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(HostMemoryAllocTests, MallocThenFreeRoundTrips) {
  void *p = nullptr;
  EXPECT_EQ(std_malloc(&p, 64), stdHostMemSuccess);
  EXPECT_NE(p, nullptr);
  EXPECT_EQ(std_free(p), stdHostMemSuccess);
}

// Its own suite, not HostMemoryAllocTests, so the no_sanitizer label can be
// confined to it: the SIZE_MAX request is an allocation-size-too-big that ASAN's
// allocator aborts on, while the sibling round-trip and free-nullptr cases stay
// safe under every sanitizer. See this directory's CMakeLists.txt.
TEST(HostMemoryAllocFailureTests, MallocReportsAllocationFailure) {
  // A SIZE_MAX request cannot be satisfied, so std::malloc returns null and
  // std_malloc must report the failure and leave the out-pointer null.
  void *p = std::addressof(p); // non-null sentinel; must be overwritten
  EXPECT_EQ(std_malloc(&p, std::numeric_limits<std::size_t>::max()), stdHostMemAllocFailure);
  EXPECT_EQ(p, nullptr);
}

TEST(HostMemoryAllocTests, FreeNullptrIsError) {
  EXPECT_EQ(std_free(nullptr), stdHostMemDeallocFailure);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// default_error_policy.cppm
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(DefaultErrorPolicyTests, SuccessCodeIsNoOp) {
  DefaultErrorPolicy<stdHostMemoryError_t> policy;
  policy.handle_error(stdHostMemSuccess, std::source_location::current());
  SUCCEED(); // returned without printing or aborting
}

TEST(DefaultErrorPolicyDeathTest, FailureAborts) {
  DefaultErrorPolicy<stdHostMemoryError_t> policy;
  EXPECT_DEATH(policy.handle_error(stdHostMemAllocFailure, std::source_location::current()),
               "GPU error at");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// gpu_check.cppm
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(GpuCheckTests, DefaultPolicySuccessReturnsTrue) {
  EXPECT_TRUE(gpu_check(stdHostMemSuccess));
}

TEST(GpuCheckTests, CustomPolicySuccessReturnsTrue) {
  RecordingPolicy<stdHostMemoryError_t> policy;
  EXPECT_TRUE(gpu_check(stdHostMemSuccess, policy));
  EXPECT_EQ(policy.count(), 0);
}

TEST(GpuCheckTests, CustomPolicyFailureReturnsFalse) {
  RecordingPolicy<stdHostMemoryError_t> policy;
  EXPECT_FALSE(gpu_check(stdHostMemAllocFailure, policy));
  EXPECT_EQ(policy.count(), 1);
  EXPECT_EQ(policy.last(), stdHostMemAllocFailure);
}

TEST(GpuCheckDeathTest, DefaultPolicyFailureAborts) {
  EXPECT_DEATH((void)gpu_check(stdHostMemAllocFailure), "GPU error at");
}

} // namespace gpumod::extension::test
