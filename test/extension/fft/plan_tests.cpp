// plan_tests.cpp - RAII contract of gpumod.extension.fft's FftPlan
//
// FftPlan is the one extension handle that does NOT derive from BaseGpuHandle:
// gpufftHandle is an integer on CUDA (cufftHandle is `int`) with no reserved
// invalid value, so the wrapper tracks liveness with an explicit `created_`
// flag and hand-rolls its own move ctor / move assign / destructor rather than
// inheriting the base's null-sentinel logic. That bespoke code is exactly what
// can double-free, and get() cannot see it: the move leaves the source's
// integer handle unchanged and only flips its private `created_`, so a
// moved-from plan is indistinguishable from a live one through the public API.
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
import gpumod.extension.common; // BaseErrorPolicy, for the counting policy
import gpumod.extension.fft; // re-exports gpumod.fft: gpufftHandle, gpufftResult_t, GPUFFT_SUCCESS

namespace gpumod::extension::test {

// An error policy that tallies failures into an external counter instead of
// aborting, so a botched destroy is observable after the objects are gone
// rather than terminating the process (which DefaultErrorPolicy would).
struct CountingErrorPolicy : BaseErrorPolicy<gpufftResult_t> {
  int *errors = nullptr;

  CountingErrorPolicy() = default;
  explicit CountingErrorPolicy(int *counter) : errors(counter) {}

  void handle_error(gpufftResult_t, std::source_location) override {
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

} // namespace gpumod::extension::test
