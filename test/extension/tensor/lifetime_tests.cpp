// lifetime_tests.cpp - RAII contract of wwr.extension.tensor's handle + descriptor
//
// TensorHandleWrapper is a plain BaseHandle subclass over wwrtensorHandle_t (a
// pointer on CUDA and HIP), created on the default path -- wwrtensorCreate takes
// no arguments. TensorDescriptorWrapper takes the skip_default_create + adopt
// path, since wwrtensorCreateTensorDescriptor needs a handle + shape + datatype.
// Both are pointer handles, so liveness rides BaseHandle's null sentinel and a
// moved-from wrapper is left holding nullptr.
//
// These cases pin destroy-exactly-once directly, through a counting error policy
// substituted for the default one: a double-free (the moved-from source
// re-destroying a handle/descriptor already freed by the destination) routes a
// failing wwrtensorDestroy* through the policy and bumps the counter. errors
// staying 0 across a move-then-destroy is the proof the move cleared ownership.
//
// Runtime, device-requiring: wwrtensorCreate needs a live GPU context.
// Backend-neutral -- built and run for either WWR_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import wwr.extension.common; // the error_policy concept, for the counting policy
import wwr.extension.handle; // BaseHandle
import wwr.extension.tensor; // the wrappers; re-exports wwr.tensor's types/enums

namespace wwr::extension::test {

// Tallies failures into an external counter instead of aborting, so a botched
// destroy is observable after the objects are gone rather than terminating.
struct CountingErrorPolicy {
  using error_type = wwrtensorStatus_t;
  int *errors = nullptr;

  CountingErrorPolicy() = default;
  explicit CountingErrorPolicy(int *counter) : errors(counter) {}

  void handle_error(wwrtensorStatus_t, std::source_location) noexcept {
    if (errors != nullptr) {
      ++*errors;
    }
  }
};

using CountingHandle = TensorHandleWrapper<CountingErrorPolicy, CountingErrorPolicy>;
using CountingDesc = TensorDescriptorWrapper<CountingErrorPolicy, CountingErrorPolicy>;

// Copy is deleted at the base; a copyable RAII handle would double-free.
static_assert(!std::is_copy_constructible_v<CountingHandle>);
static_assert(!std::is_copy_assignable_v<CountingHandle>);
static_assert(std::is_nothrow_move_constructible_v<CountingHandle>);
static_assert(std::is_nothrow_move_assignable_v<CountingHandle>);
static_assert(!std::is_copy_constructible_v<CountingDesc>);
static_assert(std::is_nothrow_move_constructible_v<CountingDesc>);
// A descriptor is built from a handle + shape, never default-constructed.
static_assert(!std::is_default_constructible_v<CountingDesc>);

// --- TensorHandleWrapper -----------------------------------------------------

TEST(TensorHandleTests, ConstructsLiveHandle) {
  CountingHandle handle;
  EXPECT_NE(handle.get(), nullptr);
  EXPECT_TRUE(handle.valid());
}

TEST(TensorHandleTests, ImplicitConversionMatchesGet) {
  CountingHandle handle;
  wwrtensorHandle_t raw = handle; // operator wwrtensorHandle_t()
  EXPECT_EQ(raw, handle.get());
}

TEST(TensorHandleTests, ConstructAndDestroyReportNoError) {
  int errors = 0;
  {
    CountingHandle handle{CountingErrorPolicy{&errors}, CountingErrorPolicy{&errors}};
  }
  EXPECT_EQ(errors, 0);
}

TEST(TensorHandleTests, MoveConstructorTransfersOwnershipAndFreesOnce) {
  int errors = 0;
  {
    CountingHandle source{CountingErrorPolicy{&errors}, CountingErrorPolicy{&errors}};
    const wwrtensorHandle_t raw = source.get();

    CountingHandle dest(std::move(source));
    EXPECT_EQ(dest.get(), raw);
    EXPECT_EQ(source.get(), nullptr); // NOLINT(bugprone-use-after-move): moved-from is the contract
  }
  EXPECT_EQ(errors, 0);
}

TEST(TensorHandleTests, MoveAssignmentTransfersOwnershipAndFreesOnce) {
  int errors = 0;
  {
    CountingHandle source{CountingErrorPolicy{&errors}, CountingErrorPolicy{&errors}};
    CountingHandle dest{CountingErrorPolicy{&errors}, CountingErrorPolicy{&errors}};
    const wwrtensorHandle_t raw = source.get();

    dest = std::move(source);
    EXPECT_EQ(dest.get(), raw);
    EXPECT_EQ(source.get(), nullptr); // NOLINT(bugprone-use-after-move)
  }
  EXPECT_EQ(errors, 0);
}

TEST(TensorHandleTests, SelfMoveAssignmentIsSafe) {
  int errors = 0;
  {
    CountingHandle handle{CountingErrorPolicy{&errors}, CountingErrorPolicy{&errors}};
    const wwrtensorHandle_t raw = handle.get();

    handle = std::move(handle); // NOLINT(clang-diagnostic-self-move): guarded self-assign must not free
    EXPECT_EQ(handle.get(), raw);
  }
  EXPECT_EQ(errors, 0);
}

// --- TensorDescriptorWrapper -------------------------------------------------

TEST(TensorDescriptorTests, ConstructAndDestroyReportNoError) {
  CountingHandle handle;
  const std::array<std::int64_t, 2> extent{64, 64};
  int errors = 0;
  {
    CountingDesc desc{handle, extent, WWRTENSOR_R_32F, 256, CountingErrorPolicy{&errors},
                      CountingErrorPolicy{&errors}};
    EXPECT_TRUE(desc.valid());
    EXPECT_NE(desc.get(), nullptr);
  }
  EXPECT_EQ(errors, 0);
}

TEST(TensorDescriptorTests, ImplicitConversionMatchesGet) {
  CountingHandle handle;
  const std::array<std::int64_t, 3> extent{8, 16, 32};
  CountingDesc desc{handle, extent, WWRTENSOR_R_64F};
  wwrtensorTensorDescriptor_t raw = desc; // operator wwrtensorTensorDescriptor_t()
  EXPECT_EQ(raw, desc.get());
}

TEST(TensorDescriptorTests, MoveConstructorTransfersOwnershipAndFreesOnce) {
  CountingHandle handle;
  const std::array<std::int64_t, 2> extent{32, 32};
  int errors = 0;
  {
    CountingDesc source{handle, extent, WWRTENSOR_R_32F, 256, CountingErrorPolicy{&errors},
                        CountingErrorPolicy{&errors}};
    const wwrtensorTensorDescriptor_t raw = source.get();

    CountingDesc dest(std::move(source));
    EXPECT_EQ(dest.get(), raw);
    EXPECT_FALSE(source.valid()); // NOLINT(bugprone-use-after-move): moved-from is the contract
  }
  EXPECT_EQ(errors, 0);
}

TEST(TensorDescriptorTests, MoveAssignmentTransfersOwnershipAndFreesOnce) {
  CountingHandle handle;
  const std::array<std::int64_t, 2> extent_a{16, 16};
  const std::array<std::int64_t, 2> extent_b{8, 8};
  int errors = 0;
  {
    CountingDesc source{handle, extent_a, WWRTENSOR_R_32F, 256, CountingErrorPolicy{&errors},
                        CountingErrorPolicy{&errors}};
    CountingDesc dest{handle, extent_b, WWRTENSOR_R_32F, 256, CountingErrorPolicy{&errors},
                      CountingErrorPolicy{&errors}};
    const wwrtensorTensorDescriptor_t raw = source.get();

    dest = std::move(source);
    EXPECT_EQ(dest.get(), raw);
  }
  EXPECT_EQ(errors, 0);
}

} // namespace wwr::extension::test
