// plan_tests.cpp - RAII contract of gpumod.extension.fft's FftPlan
//
// FftPlan derives from GpuBoundHandle like the other library handles, but it is
// the one handle whose liveness cannot ride the base's null sentinel:
// gpufftHandle is an integer on CUDA (cufftHandle is `int`) with no reserved
// invalid value. BaseGpuHandle handles that by tracking ownership with an
// explicit flag for handle types with no in-band null, so a moved-from plan is
// left owning nothing while its integer handle is unchanged -- indistinguishable
// from a live one through get(), which is exactly why a broken move would
// double-free undetected by the public API.
//
// A cuFFT/hipFFT plan is device-bound, so GpuBoundHandle's select-device /
// record-device contract applies: dev_idx() reports the creation device and the
// move clears it to -1. The cases below pin that alongside the double-free
// contract.
//
// So these cases assert the destroy-exactly-once contract directly, through a
// counting error policy substituted for the default one. gpu_check routes a
// failing gpufftDestroy to the policy instead of a return value, so a
// double-free (the source re-destroying a plan already freed by the
// destination) surfaces as GPUFFT_INVALID_PLAN and bumps the counter. errors
// staying 0 across a move-then-destroy is the proof the move cleared ownership.
//
// Runtime, device-requiring: gpufftCreate needs a live GPU context.
// Backend-neutral -- built and run for either GPUMOD_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import gpumod.extension.common; // the error_policy concept, for the counting policy
import gpumod.extension.common.handle; // BaseGpuHandle, GpuBoundHandle
import gpumod.extension.fft; // re-exports gpumod.fft: gpufftHandle, gpufftResult_t, GPUFFT_SUCCESS

namespace gpumod::extension::test {

// An error policy that tallies failures into an external counter instead of
// aborting, so a botched destroy is observable after the objects are gone
// rather than terminating the process (which DefaultErrorPolicy would).
struct CountingErrorPolicy {
  using error_type = gpufftResult_t;
  int *errors = nullptr;

  CountingErrorPolicy() = default;
  explicit CountingErrorPolicy(int *counter) : errors(counter) {}

  void handle_error(gpufftResult_t, std::source_location) noexcept {
    if (errors != nullptr) {
      ++*errors;
    }
  }
};

using CountingPlan = FftPlanWrapper<CountingErrorPolicy>;

static_assert(!std::is_copy_constructible_v<FftPlan>);
static_assert(!std::is_copy_assignable_v<FftPlan>);
static_assert(std::is_nothrow_move_constructible_v<FftPlan>);
static_assert(std::is_nothrow_move_assignable_v<FftPlan>);

TEST(FftPlanTests, ConstructAndDestroyReportNoError) {
  int errors = 0;
  {
    CountingPlan plan{CountingErrorPolicy{&errors}};
  }
  EXPECT_EQ(errors, 0);
}

TEST(FftPlanTests, ImplicitConversionMatchesGet) {
  // The raw gpufftMakePlan/gpufftExec* calls documented in fft_plan.cppm rely on
  // operator gpufftHandle(); the blas/solver/sparse handles all pin this and fft
  // did not. Only get() was exercised here before.
  FftPlan plan;
  gpufftHandle raw = plan; // operator gpufftHandle()
  EXPECT_EQ(raw, plan.get());
}

TEST(FftPlanTests, MoveConstructorTransfersOwnershipAndFreesOnce) {
  int errors = 0;
  {
    CountingPlan source{CountingErrorPolicy{&errors}};
    gpufftHandle raw = source.get();

    CountingPlan dest(std::move(source));
    EXPECT_EQ(dest.get(), raw); // the plan handle moved across
  }
  // A move that failed to clear the source's ownership would destroy the same
  // plan twice; the second gpufftDestroy fails and the policy counts it.
  EXPECT_EQ(errors, 0);
}

TEST(FftPlanTests, MoveAssignmentTransfersOwnershipAndFreesOnce) {
  int errors = 0;
  {
    CountingPlan source{CountingErrorPolicy{&errors}};
    CountingPlan dest{CountingErrorPolicy{&errors}};
    gpufftHandle raw = source.get();

    // dest's original plan is freed here (exactly once), then dest adopts
    // source's plan and source is left owning nothing.
    dest = std::move(source);
    EXPECT_EQ(dest.get(), raw);
  }
  EXPECT_EQ(errors, 0);
}

// The three-argument constructor threads SEPARATE create and destroy policies;
// the cases above all use the single-policy ctor, which copies one into both. A
// full construct -> move-assign -> destroy lifecycle with two distinct counters
// proves both are stored and moved independently and that the clean path touches
// neither. (On the success path the two cannot be told apart -- distinguishing
// them would need a forced failure, which a live gpufft has no cheap way to
// produce.)
TEST(FftPlanTests, TwoPolicyConstructorThreadsBothPolicies) {
  using TwoPolicyPlan = FftPlanWrapper<CountingErrorPolicy, CountingErrorPolicy>;
  int create_errors = 0;
  int destroy_errors = 0;
  {
    TwoPolicyPlan source{CountingErrorPolicy{&create_errors}, CountingErrorPolicy{&destroy_errors}};
    TwoPolicyPlan dest{CountingErrorPolicy{&create_errors}, CountingErrorPolicy{&destroy_errors}};
    const gpufftHandle raw = source.get();

    dest = std::move(source); // frees dest's original plan through policy_destroy_
    EXPECT_EQ(dest.get(), raw);
  }
  EXPECT_EQ(create_errors, 0);
  EXPECT_EQ(destroy_errors, 0);
}

TEST(FftPlanTests, SelfMoveAssignmentIsSafe) {
  int errors = 0;
  {
    CountingPlan plan{CountingErrorPolicy{&errors}};
    gpufftHandle raw = plan.get();

    plan = std::move(plan); // guarded self-assign: must not free itself
    EXPECT_EQ(plan.get(), raw);
  }
  EXPECT_EQ(errors, 0);
}

TEST(FftPlanTests, RecordsCreationDevice) {
  // The default constructor creates on device 0; dev_idx is the (defaulted)
  // first constructor argument, and device 0 always exists.
  FftPlan plan;
  EXPECT_EQ(plan.dev_idx(), 0);

  FftPlan on0(0);
  EXPECT_EQ(on0.dev_idx(), 0);
}

TEST(FftPlanTests, MovePreservesDevice) {
  FftPlan plan1;
  const int dev = plan1.dev_idx();

  FftPlan plan2(std::move(plan1));
  EXPECT_EQ(plan2.dev_idx(), dev);
  EXPECT_EQ(plan1.dev_idx(), -1);
}

} // namespace gpumod::extension::test
