// comm_tests.cpp - error-policy mapping and RAII contract of wwr.extension.ccl
//
// Two halves, split by what each needs:
//
//  - CclErrorTests is HOST-ONLY (no live device): it pins the wwrcclResult_t
//    error-code specializations -- success_code maps to WWRCCL_SUCCESS, and both
//    error_name and error_string return a usable string for a known failure enum
//    (NCCL/RCCL expose a single string function, so the two route to the same
//    place). This suite is NOT labeled gpu, so it runs under `ctest -LE gpu` / in
//    CI, which is the link check that the module and its specializations build.
//
//  - CclCommTests is REQUIRES_GPU: wwrcclCommInitAll needs a live GPU context. It
//    uses the single-process single-GPU path (wwrcclCommInitAll(&comm, 1,
//    &device)), which runs on one card, and a counting error policy substituted
//    for the default to prove destroy-exactly-once. wwrcclComm_t is a pointer, so
//    a moved-from communicator is nulled (BaseHandle's null-sentinel liveness)
//    and its destructor becomes a no-op; a double-free would route a failing
//    wwrcclCommDestroy through the policy and bump the counter.
//
// Backend-neutral -- built and run for either WWR_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import wwr.extension.common; // success_code, error_name, error_string, error_policy
import wwr.extension.handle; // BaseHandle
import wwr.extension.ccl; // re-exports wwr.ccl: wwrcclComm_t, wwrcclResult_t, WWRCCL_*
import wwr.test.shared.abort_policy; // AbortPolicy for this file's instantiations

namespace wwr::extension::test {

using Abort = AbortPolicy<wwrcclResult_t>;

// Tally destroy/create failures into an external counter instead of aborting, so
// a botched destroy is observable after the communicator is gone.
struct CountingErrorPolicy {
  using error_type = wwrcclResult_t;
  int *errors = nullptr;

  CountingErrorPolicy() = default;
  explicit CountingErrorPolicy(int *counter) : errors(counter) {}

  void handle_error(wwrcclResult_t, std::source_location) noexcept {
    if (errors != nullptr) {
      ++*errors;
    }
  }
};

using AbortComm = CommWrapper<Abort, Abort>;
using CountingComm = CommWrapper<CountingErrorPolicy, CountingErrorPolicy>;

// A communicator is created from init arguments, never default-constructed, and
// like every handle it is move-only.
static_assert(std::is_constructible_v<AbortComm, single_device_t, int>);
static_assert(std::is_constructible_v<AbortComm, int, wwrcclUniqueId, int>);
static_assert(!std::is_default_constructible_v<AbortComm>);
static_assert(!std::is_copy_constructible_v<AbortComm>);
static_assert(!std::is_copy_assignable_v<AbortComm>);
static_assert(std::is_nothrow_move_constructible_v<AbortComm>);
static_assert(std::is_nothrow_move_assignable_v<AbortComm>);

// ============================================================================
// Host-only: error-code specialization mapping (runs under ctest -LE gpu)
// ============================================================================

TEST(CclErrorTests, SuccessCodeMapsToWwrcclSuccess) {
  EXPECT_EQ(success_code<wwrcclResult_t>(), WWRCCL_SUCCESS);
}

TEST(CclErrorTests, ErrorStringIsNonNullForKnownError) {
  // A known failure enum must map to a usable (non-null) description. NCCL/RCCL
  // expose only wwrcclGetErrorString, so error_name routes to it too.
  EXPECT_NE(error_string<wwrcclResult_t>(WWRCCL_INVALID_ARGUMENT), nullptr);
  EXPECT_NE(error_name(WWRCCL_INVALID_ARGUMENT), nullptr);
}

TEST(CclErrorTests, IsRegisteredErrorType) {
  // All three specializations present -> the type satisfies the error_type
  // concept, so a policy over it is a typed_error_policy (what CommWrapper needs).
  static_assert(error_type<wwrcclResult_t>);
  static_assert(typed_error_policy<Abort>);
  SUCCEED();
}

// ============================================================================
// Device-requiring: communicator construct / destroy (REQUIRES_GPU)
// ============================================================================

TEST(CclCommTests, ConstructAndDestroyReportNoError) {
  int errors = 0;
  {
    CountingComm comm{single_device, 0, CountingErrorPolicy{&errors}, CountingErrorPolicy{&errors}};
    EXPECT_TRUE(comm.valid());
    EXPECT_NE(comm.get(), nullptr);
  }
  EXPECT_EQ(errors, 0);
}

TEST(CclCommTests, ImplicitConversionMatchesGet) {
  AbortComm comm{single_device, 0};
  wwrcclComm_t raw = comm; // operator wwrcclComm_t()
  EXPECT_EQ(raw, comm.get());
}

TEST(CclCommTests, MoveConstructorTransfersOwnershipAndFreesOnce) {
  int errors = 0;
  {
    CountingComm source{single_device, 0, CountingErrorPolicy{&errors}, CountingErrorPolicy{&errors}};
    const wwrcclComm_t raw = source.get();

    CountingComm dest(std::move(source));
    EXPECT_EQ(dest.get(), raw); // the communicator moved across
    EXPECT_TRUE(dest.valid());
    EXPECT_FALSE(source.valid()); // NOLINT(bugprone-use-after-move): moved-from state is the contract
  }
  // A move that failed to clear the source would destroy the same communicator
  // twice; the second wwrcclCommDestroy fails and the policy counts it.
  EXPECT_EQ(errors, 0);
}

TEST(CclCommTests, MoveAssignmentTransfersOwnershipAndFreesOnce) {
  int errors = 0;
  {
    CountingComm source{single_device, 0, CountingErrorPolicy{&errors}, CountingErrorPolicy{&errors}};
    CountingComm dest{single_device, 0, CountingErrorPolicy{&errors}, CountingErrorPolicy{&errors}};
    const wwrcclComm_t raw = source.get();

    // dest's original communicator is freed here (exactly once), then dest adopts
    // source's and source is left owning nothing.
    dest = std::move(source);
    EXPECT_EQ(dest.get(), raw);
  }
  EXPECT_EQ(errors, 0);
}

TEST(CclCommTests, SelfMoveAssignmentIsSafe) {
  int errors = 0;
  {
    CountingComm comm{single_device, 0, CountingErrorPolicy{&errors}, CountingErrorPolicy{&errors}};
    const wwrcclComm_t raw = comm.get();

    comm = std::move(comm); // NOLINT(clang-diagnostic-self-move): guarded self-assign must not free
    EXPECT_EQ(comm.get(), raw);
  }
  EXPECT_EQ(errors, 0);
}

} // namespace wwr::extension::test
