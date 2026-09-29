// object_tests.cpp - RAII contract of wwr.extension.texture's TextureObject
//
// TextureObject is a BaseHandle subclass like FftPlanWrapper, and like the FFT
// plan it is one of the handles whose liveness cannot ride the base's null
// sentinel: cudaTextureObject_t is an integer with no reserved invalid value on
// CUDA (a pointer on HIP). BaseHandle tracks that with an explicit ownership
// flag, set here through adopt() from the skip-default-create constructor. A
// moved-from object is left owning nothing while its integer handle is
// unchanged -- indistinguishable from a live one through get() -- so a broken
// move would double-free undetected by the public API.
//
// The cases below assert the destroy-exactly-once contract through a counting
// error policy substituted for the default: gpu_check routes a failing
// wwrDestroyTextureObject to the policy instead of a return value, so a
// double-free surfaces as a bumped counter. errors staying 0 across a
// move-then-destroy is the proof the move cleared ownership.
//
// Runtime, device-requiring: wwrCreateTextureObject needs a live GPU context
// and a backing CUDA array. Backend-neutral -- built and run for either
// WWR_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api; // wwrError_t, wwrArray_t, the descriptor + array API
import wwr.extension.common; // success_code, for wwrSuccess comparisons
import wwr.extension.handle; // BaseHandle
import wwr.extension.texture; // TextureObject
import wwr.test.shared.abort_policy; // AbortPolicy

namespace wwr::extension::test {

using Abort = AbortPolicy<wwrError_t>;

// Tally destroy/create failures into an external counter instead of aborting,
// so a botched destroy is observable after the object is gone.
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

using AbortTex = TextureObject<Abort, Abort>;
using CountingTex = TextureObject<CountingErrorPolicy, CountingErrorPolicy>;

// A texture object is created from descriptors, never default-constructed, and
// like every handle it is move-only.
static_assert(std::is_constructible_v<AbortTex, const wwrResourceDesc &, const wwrTextureDesc &>);
static_assert(!std::is_default_constructible_v<AbortTex>);
static_assert(!std::is_copy_constructible_v<AbortTex>);
static_assert(!std::is_copy_assignable_v<AbortTex>);
static_assert(std::is_nothrow_move_constructible_v<AbortTex>);
static_assert(std::is_nothrow_move_assignable_v<AbortTex>);

// A 4x4 float CUDA array, owned for the test's duration -- the backing store a
// texture object references and must outlive it.
class TestArray {
public:
  explicit TestArray(unsigned int flags = wwrArrayDefault) {
    wwrChannelFormatDesc channel{};
    channel.x = 32; // one 32-bit float channel
    channel.f = wwrChannelFormatKindFloat;
    if (wwrMallocArray(&array_, &channel, 4, 4, flags) != success_code<wwrError_t>()) {
      array_ = nullptr;
    }
  }
  ~TestArray() {
    if (array_ != nullptr) {
      wwrFreeArray(array_);
    }
  }
  TestArray(const TestArray &) = delete;
  TestArray &operator=(const TestArray &) = delete;

  wwrArray_t get() const noexcept { return array_; }

private:
  wwrArray_t array_ = nullptr;
};

// An array-backed resource descriptor.
wwrResourceDesc array_resource(wwrArray_t array) {
  wwrResourceDesc desc{};
  desc.resType = wwrResourceTypeArray;
  desc.res.array.array = array;
  return desc;
}

// A texture descriptor valid for an array of element-typed floats. The zero
// default addressing is wrap, which the runtime accepts only with normalized
// coordinates, so opt into those.
wwrTextureDesc element_texture_desc() {
  wwrTextureDesc desc{};
  desc.normalizedCoords = 1;
  return desc;
}

TEST(TextureObjectTests, ConstructAndDestroyReportNoError) {
  TestArray backing;
  ASSERT_NE(backing.get(), nullptr);
  int errors = 0;
  {
    const auto res = array_resource(backing.get());
    const auto tex = element_texture_desc();
    CountingTex object{res, tex, CountingErrorPolicy{&errors}, CountingErrorPolicy{&errors}};
    EXPECT_TRUE(object.valid());
  }
  EXPECT_EQ(errors, 0);
}

TEST(TextureObjectTests, ImplicitConversionMatchesGet) {
  TestArray backing;
  ASSERT_NE(backing.get(), nullptr);
  AbortTex object{array_resource(backing.get()), element_texture_desc()};
  wwrTextureObject_t raw = object; // operator wwrTextureObject_t()
  EXPECT_EQ(raw, object.get());
}

TEST(TextureObjectTests, MoveConstructorTransfersOwnershipAndFreesOnce) {
  TestArray backing;
  ASSERT_NE(backing.get(), nullptr);
  int errors = 0;
  {
    CountingTex source{array_resource(backing.get()), element_texture_desc(),
                       CountingErrorPolicy{&errors}, CountingErrorPolicy{&errors}};
    const wwrTextureObject_t raw = source.get();

    CountingTex dest(std::move(source));
    EXPECT_EQ(dest.get(), raw); // the handle moved across
    EXPECT_TRUE(dest.valid());
    EXPECT_FALSE(source.valid()); // NOLINT(bugprone-use-after-move): moved-from state is the contract
  }
  // A move that failed to clear the source would destroy the same object twice;
  // the second wwrDestroyTextureObject fails and the policy counts it.
  EXPECT_EQ(errors, 0);
}

TEST(TextureObjectTests, MoveAssignmentTransfersOwnershipAndFreesOnce) {
  TestArray a;
  TestArray b;
  ASSERT_NE(a.get(), nullptr);
  ASSERT_NE(b.get(), nullptr);
  int errors = 0;
  {
    CountingTex source{array_resource(a.get()), element_texture_desc(),
                       CountingErrorPolicy{&errors}, CountingErrorPolicy{&errors}};
    CountingTex dest{array_resource(b.get()), element_texture_desc(),
                    CountingErrorPolicy{&errors}, CountingErrorPolicy{&errors}};
    const wwrTextureObject_t raw = source.get();

    // dest's original object is freed here (exactly once), then dest adopts
    // source's object and source is left owning nothing.
    dest = std::move(source);
    EXPECT_EQ(dest.get(), raw);
  }
  EXPECT_EQ(errors, 0);
}

TEST(TextureObjectTests, SelfMoveAssignmentIsSafe) {
  TestArray backing;
  ASSERT_NE(backing.get(), nullptr);
  int errors = 0;
  {
    CountingTex object{array_resource(backing.get()), element_texture_desc(),
                      CountingErrorPolicy{&errors}, CountingErrorPolicy{&errors}};
    const wwrTextureObject_t raw = object.get();

    object = std::move(object); // NOLINT(clang-diagnostic-self-move): guarded self-assign must not free
    EXPECT_EQ(object.get(), raw);
  }
  EXPECT_EQ(errors, 0);
}

} // namespace wwr::extension::test
