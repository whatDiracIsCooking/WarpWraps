// plan_tests.cpp - RAII contract of wwr.extension.fft's FftPlanWrapper
//
// FftPlanWrapper derives from DeviceBoundHandle like the other library handles, but it is
// the one handle whose liveness cannot ride the base's null sentinel:
// wwrfftHandle is an integer on CUDA (cufftHandle is `int`) with no reserved
// invalid value. BaseHandle handles that by tracking ownership with an
// explicit flag for handle types with no in-band null, so a moved-from plan is
// left owning nothing while its integer handle is unchanged -- indistinguishable
// from a live one through get(), which is exactly why a broken move would
// double-free undetected by the public API.
//
// A cuFFT/hipFFT plan is device-bound, so DeviceBoundHandle's select-device /
// record-device contract applies: dev_idx() reports the creation device and the
// move clears it to -1. The cases below pin that alongside the double-free
// contract.
//
// So these cases assert the destroy-exactly-once contract directly, through a
// counting error policy substituted for the default one. gpu_check routes a
// failing wwrfftDestroy to the policy instead of a return value, so a
// double-free (the source re-destroying a plan already freed by the
// destination) surfaces as WWRFFT_INVALID_PLAN and bumps the counter. errors
// staying 0 across a move-then-destroy is the proof the move cleared ownership.
//
// Runtime, device-requiring: wwrfftCreate needs a live GPU context.
// Backend-neutral -- built and run for either WWR_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import wwr.extension.common; // the error_policy concept, for the counting policy
import wwr.extension.handle; // BaseHandle, DeviceBoundHandle
import wwr.extension.fft; // re-exports wwr.fft: wwrfftHandle, wwrfftResult_t, WWRFFT_SUCCESS

namespace wwr::extension::test {
// Bind abort-on-failure once, for this file's wrapper instantiations.
using Abort = AbortPolicy<wwrfftResult_t>;

// An error policy that tallies failures into an external counter instead of
// aborting, so a botched destroy is observable after the objects are gone
// rather than terminating the process (which AbortPolicy would).
struct CountingErrorPolicy {
  using error_type = wwrfftResult_t;
  int *errors = nullptr;

  CountingErrorPolicy() = default;
  explicit CountingErrorPolicy(int *counter) : errors(counter) {}

  void handle_error(wwrfftResult_t, std::source_location) noexcept {
    if (errors != nullptr) {
      ++*errors;
    }
  }
};

using CountingPlan = FftPlanWrapper<CountingErrorPolicy, CountingErrorPolicy>;

static_assert(!std::is_copy_constructible_v<FftPlanWrapper<Abort, Abort>>);
static_assert(!std::is_copy_assignable_v<FftPlanWrapper<Abort, Abort>>);
static_assert(std::is_nothrow_move_constructible_v<FftPlanWrapper<Abort, Abort>>);
static_assert(std::is_nothrow_move_assignable_v<FftPlanWrapper<Abort, Abort>>);

TEST(FftPlanTests, ConstructAndDestroyReportNoError) {
  int errors = 0;
  {
    CountingPlan plan{CountingErrorPolicy{&errors}};
  }
  EXPECT_EQ(errors, 0);
}

TEST(FftPlanTests, ImplicitConversionMatchesGet) {
  // The raw wwrfftMakePlan/wwrfftExec* calls documented in fft_plan.cppm rely on
  // operator wwrfftHandle(); the blas/solver/sparse handles all pin this and fft
  // did not. Only get() was exercised here before.
  FftPlanWrapper<Abort, Abort> plan;
  wwrfftHandle raw = plan; // operator wwrfftHandle()
  EXPECT_EQ(raw, plan.get());
}

TEST(FftPlanTests, MoveConstructorTransfersOwnershipAndFreesOnce) {
  int errors = 0;
  {
    CountingPlan source{CountingErrorPolicy{&errors}};
    wwrfftHandle raw = source.get();

    CountingPlan dest(std::move(source));
    EXPECT_EQ(dest.get(), raw); // the plan handle moved across
  }
  // A move that failed to clear the source's ownership would destroy the same
  // plan twice; the second wwrfftDestroy fails and the policy counts it.
  EXPECT_EQ(errors, 0);
}

TEST(FftPlanTests, MoveAssignmentTransfersOwnershipAndFreesOnce) {
  int errors = 0;
  {
    CountingPlan source{CountingErrorPolicy{&errors}};
    CountingPlan dest{CountingErrorPolicy{&errors}};
    wwrfftHandle raw = source.get();

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
// them would need a forced failure, which a live wwrfft has no cheap way to
// produce.)
TEST(FftPlanTests, TwoPolicyConstructorThreadsBothPolicies) {
  using TwoPolicyPlan = FftPlanWrapper<CountingErrorPolicy, CountingErrorPolicy>;
  int create_errors = 0;
  int destroy_errors = 0;
  {
    TwoPolicyPlan source{CountingErrorPolicy{&create_errors}, CountingErrorPolicy{&destroy_errors}};
    TwoPolicyPlan dest{CountingErrorPolicy{&create_errors}, CountingErrorPolicy{&destroy_errors}};
    const wwrfftHandle raw = source.get();

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
    wwrfftHandle raw = plan.get();

    plan = std::move(plan); // guarded self-assign: must not free itself
    EXPECT_EQ(plan.get(), raw);
  }
  EXPECT_EQ(errors, 0);
}

TEST(FftPlanTests, RecordsCreationDevice) {
  // The default constructor creates on device 0; dev_idx is the (defaulted)
  // first constructor argument, and device 0 always exists.
  FftPlanWrapper<Abort, Abort> plan;
  EXPECT_EQ(plan.dev_idx(), 0);

  FftPlanWrapper<Abort, Abort> on0(0);
  EXPECT_EQ(on0.dev_idx(), 0);
}

TEST(FftPlanTests, MovePreservesDevice) {
  FftPlanWrapper<Abort, Abort> plan1;
  const int dev = plan1.dev_idx();

  FftPlanWrapper<Abort, Abort> plan2(std::move(plan1));
  EXPECT_EQ(plan2.dev_idx(), dev);
  EXPECT_EQ(plan1.dev_idx(), -1);
}

} // namespace wwr::extension::test
