// object_tests.cpp - RAII contract of wwr.extension.surface's SurfaceObject
//
// SurfaceObject is the sibling of TextureObject and shares its base and its
// hazard: cudaSurfaceObject_t is an integer with no invalid value on CUDA (a
// pointer on HIP), so liveness rides BaseHandle's explicit ownership flag, set
// through adopt() from the skip-default-create constructor. The cases below pin
// the destroy-exactly-once contract through a counting error policy -- a broken
// move that left the source owning its handle would double-free, and the second
// wwrDestroySurfaceObject would bump the counter.
//
// A surface object binds a CUDA array that MUST be allocated with
// wwrArraySurfaceLoadStore; the test's backing array sets it.
//
// Runtime, device-requiring: wwrCreateSurfaceObject needs a live GPU context.
// Backend-neutral -- built and run for either WWR_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api; // wwrError_t, wwrArray_t, the descriptor + array API
import wwr.extension.common; // success_code, for wwrSuccess comparisons
import wwr.extension.handle; // BaseHandle
import wwr.extension.surface; // SurfaceObject
import wwr.test.shared.abort_policy; // AbortPolicy

namespace wwr::extension::test {

using Abort = AbortPolicy<wwrError_t>;

struct CountingErrorPolicy {
  using error_type = wwrError_t;
  int *errors = nullptr;

  CountingErrorPolicy() = default;
  explicit CountingErrorPolicy(int *counter) : errors(counter) {}

  void handle_error(wwrError_t, std::source_location) noexcept {
    if (errors != nullptr) {
      ++*errors;
    }
  }
};

using AbortSurf = SurfaceObject<Abort, Abort>;
using CountingSurf = SurfaceObject<CountingErrorPolicy, CountingErrorPolicy>;

// A surface object is created from a resource descriptor, never default-built,
// and like every handle it is move-only.
static_assert(std::is_constructible_v<AbortSurf, const wwrResourceDesc &>);
static_assert(!std::is_default_constructible_v<AbortSurf>);
static_assert(!std::is_copy_constructible_v<AbortSurf>);
static_assert(!std::is_copy_assignable_v<AbortSurf>);
static_assert(std::is_nothrow_move_constructible_v<AbortSurf>);
static_assert(std::is_nothrow_move_assignable_v<AbortSurf>);

// A 4x4 float CUDA array allocated for surface load/store -- the backing store a
// surface object binds and must outlive it.
class TestSurfaceArray {
public:
  TestSurfaceArray() {
    wwrChannelFormatDesc channel{};
    channel.x = 32; // one 32-bit float channel
    channel.f = wwrChannelFormatKindFloat;
    if (wwrMallocArray(&array_, &channel, 4, 4, wwrArraySurfaceLoadStore) !=
        success_code<wwrError_t>()) {
      array_ = nullptr;
    }
  }
  ~TestSurfaceArray() {
    if (array_ != nullptr) {
      wwrFreeArray(array_);
    }
  }
  TestSurfaceArray(const TestSurfaceArray &) = delete;
  TestSurfaceArray &operator=(const TestSurfaceArray &) = delete;

  wwrArray_t get() const noexcept { return array_; }

private:
  wwrArray_t array_ = nullptr;
};

wwrResourceDesc array_resource(wwrArray_t array) {
  wwrResourceDesc desc{};
  desc.resType = wwrResourceTypeArray;
  desc.res.array.array = array;
  return desc;
}

TEST(SurfaceObjectTests, ConstructAndDestroyReportNoError) {
  TestSurfaceArray backing;
  ASSERT_NE(backing.get(), nullptr);
  int errors = 0;
  {
    CountingSurf object{array_resource(backing.get()), CountingErrorPolicy{&errors},
                       CountingErrorPolicy{&errors}};
    EXPECT_TRUE(object.valid());
  }
  EXPECT_EQ(errors, 0);
}

TEST(SurfaceObjectTests, ImplicitConversionMatchesGet) {
  TestSurfaceArray backing;
  ASSERT_NE(backing.get(), nullptr);
  AbortSurf object{array_resource(backing.get())};
  wwrSurfaceObject_t raw = object; // operator wwrSurfaceObject_t()
  EXPECT_EQ(raw, object.get());
}

TEST(SurfaceObjectTests, MoveConstructorTransfersOwnershipAndFreesOnce) {
  TestSurfaceArray backing;
  ASSERT_NE(backing.get(), nullptr);
  int errors = 0;
  {
    CountingSurf source{array_resource(backing.get()), CountingErrorPolicy{&errors},
                       CountingErrorPolicy{&errors}};
    const wwrSurfaceObject_t raw = source.get();

    CountingSurf dest(std::move(source));
    EXPECT_EQ(dest.get(), raw);
    EXPECT_TRUE(dest.valid());
    EXPECT_FALSE(source.valid()); // NOLINT(bugprone-use-after-move): moved-from state is the contract
  }
  EXPECT_EQ(errors, 0);
}

TEST(SurfaceObjectTests, MoveAssignmentTransfersOwnershipAndFreesOnce) {
  TestSurfaceArray a;
  TestSurfaceArray b;
  ASSERT_NE(a.get(), nullptr);
  ASSERT_NE(b.get(), nullptr);
  int errors = 0;
  {
    CountingSurf source{array_resource(a.get()), CountingErrorPolicy{&errors},
                       CountingErrorPolicy{&errors}};
    CountingSurf dest{array_resource(b.get()), CountingErrorPolicy{&errors},
                     CountingErrorPolicy{&errors}};
    const wwrSurfaceObject_t raw = source.get();

    dest = std::move(source);
    EXPECT_EQ(dest.get(), raw);
  }
  EXPECT_EQ(errors, 0);
}

TEST(SurfaceObjectTests, SelfMoveAssignmentIsSafe) {
  TestSurfaceArray backing;
  ASSERT_NE(backing.get(), nullptr);
  int errors = 0;
  {
    CountingSurf object{array_resource(backing.get()), CountingErrorPolicy{&errors},
                       CountingErrorPolicy{&errors}};
    const wwrSurfaceObject_t raw = object.get();

    object = std::move(object); // NOLINT(clang-diagnostic-self-move): guarded self-assign must not free
    EXPECT_EQ(object.get(), raw);
  }
  EXPECT_EQ(errors, 0);
}

} // namespace wwr::extension::test
