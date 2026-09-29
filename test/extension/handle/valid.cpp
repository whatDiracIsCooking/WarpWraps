// valid.cpp - BaseHandle::valid() liveness query
//
// valid() reports whether the wrapper owns a handle its destructor will destroy.
// The point of the query is the *recoverable* construction path: a non-throwing
// policy lets a failed create() return instead of aborting, leaving a
// constructed-but-un-owned wrapper the caller must be able to detect. get()
// cannot express that for a value handle -- a failed cufftHandle create leaves a
// garbage int with no reserved invalid value -- so valid() reads the private
// liveness (null sentinel for pointers, the owns_ flag otherwise) instead.
//
// Pure query, no runtime calls: unlike basic.cpp (REQUIRES_GPU, it drives
// wwrSetDevice), this needs no card, so CI runs it. create() here is a test
// double that records through RecordingPolicy and never touches a device.
//
// A plain TU: self-registering tests compiled straight in, GTest::gtest_main
// supplies main().

#include <gtest/gtest.h>

import std;
import wwr.runtime_api; // wwrError_t, wwrErrorInvalidValue
import wwr.extension.common;
import wwr.extension.handle;
import wwr.test.shared.recording_policy;

namespace wwr::extension::test {
namespace {

using Recording = RecordingPolicy<wwrError_t>;

// Whether the next wrapper's create() should succeed. Namespace scope, not a
// member: create() runs from inside BaseHandle's constructor, before the derived
// object is alive, so it cannot read a member to decide its outcome.
bool g_create_succeeds = true;

// A pointer handle (null sentinel path): liveness is "handle_ != nullptr", no
// owns_ flag. A failed create leaves it null; a successful one stores a non-null
// stand-in that destroy() never dereferences.
struct fake_handle_tag;
using FakeHandle = fake_handle_tag *;

class PointerWrapper : public BaseHandle<FakeHandle, PointerWrapper, Recording, Recording> {
public:
  using BaseHandle::BaseHandle;
  void create(FakeHandle *h, std::source_location loc) {
    if (g_create_succeeds) {
      *h = reinterpret_cast<FakeHandle>(0x1);
      return;
    }
    // Failed create: record through the policy (record-and-continue, no abort)
    // and leave the handle null so liveness reads false.
    policy_create_.handle_error(wwrErrorInvalidValue, loc);
    *h = nullptr;
  }
  void destroy(FakeHandle) {}
  const Recording &create_policy() const { return policy_create_; }
};

// A value handle (owns_ flag path): the cufftHandle shape -- a plain int with no
// reserved invalid value. create() returns its success bool, which BaseHandle
// stores into owns_. A failed create deliberately leaves a garbage int behind to
// prove valid() ignores the handle value and reads owns_.
class ValueWrapper : public BaseHandle<int, ValueWrapper, Recording, Recording> {
public:
  using BaseHandle::BaseHandle;
  bool create(int *h, std::source_location loc) {
    if (g_create_succeeds) {
      *h = 42;
      return true;
    }
    policy_create_.handle_error(wwrErrorInvalidValue, loc);
    *h = 999; // garbage: get() would return this, valid() must still be false
    return false;
  }
  void destroy(int) {}
  const Recording &create_policy() const { return policy_create_; }
};

} // namespace

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// BaseHandle valid() Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Pointer handle, happy path: create stored a non-null handle, so valid() and
// the policy was never told of an error.
TEST(BaseHandleValidTests, PointerLiveAfterSuccessfulCreate) {
  g_create_succeeds = true;
  PointerWrapper w;
  EXPECT_TRUE(w.valid());
  EXPECT_EQ(w.create_policy().calls, 0);
}

// Pointer handle, the recoverable path: a non-throwing failed create returns
// (no abort), records the error, and leaves the wrapper un-owned. valid() is the
// only clean way to see that -- and here it agrees with get() == nullptr.
TEST(BaseHandleValidTests, PointerNotValidAfterFailedCreate) {
  g_create_succeeds = false;
  PointerWrapper w;
  EXPECT_FALSE(w.valid());
  EXPECT_EQ(w.get(), nullptr);
  EXPECT_EQ(w.create_policy().calls, 1);
  EXPECT_EQ(w.create_policy().last_error, wwrErrorInvalidValue);
}

// Value handle, happy path.
TEST(BaseHandleValidTests, ValueLiveAfterSuccessfulCreate) {
  g_create_succeeds = true;
  ValueWrapper w;
  EXPECT_TRUE(w.valid());
  EXPECT_EQ(w.get(), 42);
}

// The case get() cannot express: a failed value-handle create leaves a garbage
// int, so get() is meaningless, but owns_ is false and valid() reports it.
TEST(BaseHandleValidTests, ValueNotValidAfterFailedCreate) {
  g_create_succeeds = false;
  ValueWrapper w;
  EXPECT_FALSE(w.valid());
  EXPECT_EQ(w.get(), 999); // garbage handle is still readable -- that is the point
  EXPECT_EQ(w.create_policy().calls, 1);
  EXPECT_EQ(w.create_policy().last_error, wwrErrorInvalidValue);
}

// Moved-from is also !valid(): the move ctor release()s the source. One query
// covers both "construction failed" and "moved-from".
TEST(BaseHandleValidTests, PointerNotValidAfterMove) {
  g_create_succeeds = true;
  PointerWrapper src;
  ASSERT_TRUE(src.valid());
  PointerWrapper dst(std::move(src));
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());
}

// Same for the value handle: move assignment release()s the source's owns_.
TEST(BaseHandleValidTests, ValueNotValidAfterMove) {
  g_create_succeeds = true;
  ValueWrapper src;
  ValueWrapper dst;
  ASSERT_TRUE(src.valid());
  dst = std::move(src);
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());
}

} // namespace wwr::extension::test
