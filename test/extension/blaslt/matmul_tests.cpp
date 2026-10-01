// matmul_tests.cpp - RAII contract of wwr.extension.blaslt's handle + descriptors
//
// Two suites:
//
//   BlasLtErrorTests (host-only, runs in CI) pins the SHARED error policy.
//   wwrblasLtStatus_t IS cublasStatus_t / hipblasStatus_t -- the exact same
//   underlying type as wwrblasStatus_t -- so wwr.extension.blaslt deliberately
//   adds no success_code / error_name / error_string of its own: its
//   :blaslt_error partition re-exports wwr.extension.blas, whose specializations
//   already cover this type. This suite imports BOTH wwr.extension.blas and
//   wwr.extension.blaslt and resolves the templates for wwrblasLtStatus_t, which
//   would be an ambiguous / duplicate-specialization error if blaslt redefined
//   them. It needs no GPU -- a pure template/table check.
//
//   BlasLtHandleTests (REQUIRES_GPU) pins the RAII lifetime of the library handle
//   and the matmul descriptor / matrix layout / matmul preference wrappers. Each
//   is a BaseHandle subclass; the handle rides the default-create path (no stream,
//   unlike BlasHandleWrapper), the three descriptors ride skip-default-create +
//   adopt (their creates take arguments). A counting error policy substituted for
//   the default proves destroy-exactly-once across a move: gpu_check routes a
//   failing destroy to the policy, so a double-free bumps the counter.
//
// Backend-neutral -- built and run for either WWR_GPU_BACKEND.

// The compute-type enumerator is a vendor constant src/blaslt.cppm does not
// surface under a wwrblasLt* name (its numeric value differs per backend, which
// is why it stays unneutralised). Reach it by including the vendor header behind
// the backend define that wwr::backend supplies. The scale/element data type is
// wwrsolverDataType_t (cudaDataType / hipDataType), which wwr.solver already
// neutralises as WWRSOLVER_R_32F. A plain gtest consumer TU is not a module unit,
// so these #includes go at the top directly, with no global module fragment.
#if defined(WWR_GPU_BACKEND_CUDA)
#include <cublas_api.h> // CUBLAS_COMPUTE_32F
#else
#include <hipblas/hipblas.h> // HIPBLAS_COMPUTE_32F
#endif

#include <gtest/gtest.h>

import std;
import wwr.extension.common; // success_code, the error_policy concept
import wwr.extension.handle;  // BaseHandle
import wwr.extension.blas;    // the wwrblasStatus_t specializations, imported ALONGSIDE blaslt
import wwr.extension.blaslt;  // BlasLtHandleWrapper + the descriptor wrappers; re-exports wwr.blaslt
import wwr.solver;            // WWRSOLVER_R_32F for the scale / element data type
import wwr.test.shared.abort_policy; // AbortPolicy

namespace wwr::extension::test {

#if defined(WWR_GPU_BACKEND_CUDA)
inline constexpr auto kComputeType = CUBLAS_COMPUTE_32F;
#else
inline constexpr auto kComputeType = HIPBLAS_COMPUTE_32F;
#endif

// ============================================================================
// Host-only: the shared wwrblasLtStatus_t error policy (runs in CI)
// ============================================================================

// Proving the reuse compiles with both wwr.extension.blas and wwr.extension.blaslt
// visible: if blaslt redefined these for the identical type the instantiations
// below would be ambiguous / duplicate. success_code must be the success
// enumerator; error_name / error_string must return a usable string.
TEST(BlasLtErrorTests, SuccessCodeIsSuccess) {
  EXPECT_EQ(success_code<wwrblasLtStatus_t>(), WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(std::to_underlying(success_code<wwrblasLtStatus_t>()), 0);
}

TEST(BlasLtErrorTests, ErrorNameAndStringAreNonNull) {
  EXPECT_NE(error_name<wwrblasLtStatus_t>(success_code<wwrblasLtStatus_t>()), nullptr);
  EXPECT_NE(error_string<wwrblasLtStatus_t>(success_code<wwrblasLtStatus_t>()), nullptr);
}

// ============================================================================
// Device-requiring: RAII lifetime of the handle + descriptor wrappers
// ============================================================================

using Abort = AbortPolicy<wwrblasLtStatus_t>;

// Tally failures into an external counter instead of aborting, so a botched
// destroy is observable after the object is gone.
struct CountingBlasLtPolicy {
  using error_type = wwrblasLtStatus_t;
  int *errors = nullptr;

  CountingBlasLtPolicy() = default;
  explicit CountingBlasLtPolicy(int *counter) : errors(counter) {}

  void handle_error(wwrblasLtStatus_t, std::source_location) noexcept {
    if (errors != nullptr) {
      ++*errors;
    }
  }
};

using AbortHandle = BlasLtHandleWrapper<Abort, Abort>;
using CountingHandle = BlasLtHandleWrapper<CountingBlasLtPolicy, CountingBlasLtPolicy>;
using AbortDesc = MatmulDesc<Abort, Abort>;
using CountingDesc = MatmulDesc<CountingBlasLtPolicy, CountingBlasLtPolicy>;
using AbortLayout = MatrixLayout<Abort, Abort>;
using CountingLayout = MatrixLayout<CountingBlasLtPolicy, CountingBlasLtPolicy>;
using CountingPref = MatmulPreference<CountingBlasLtPolicy, CountingBlasLtPolicy>;

// Every wrapper is move-only; a copyable RAII handle would double-free.
static_assert(!std::is_copy_constructible_v<AbortHandle>);
static_assert(!std::is_copy_assignable_v<AbortHandle>);
static_assert(std::is_nothrow_move_constructible_v<AbortHandle>);
static_assert(std::is_nothrow_move_assignable_v<AbortHandle>);
static_assert(!std::is_copy_constructible_v<AbortDesc>);
static_assert(std::is_nothrow_move_constructible_v<AbortDesc>);
static_assert(!std::is_copy_constructible_v<AbortLayout>);
static_assert(std::is_nothrow_move_constructible_v<AbortLayout>);

TEST(BlasLtHandleTests, ConstructsLiveHandle) {
  AbortHandle handle;
  EXPECT_NE(handle.get(), nullptr);
}

TEST(BlasLtHandleTests, ImplicitConversionMatchesGet) {
  AbortHandle handle;
  wwrblasLtHandle_t raw = handle; // operator wwrblasLtHandle_t()
  EXPECT_EQ(raw, handle.get());
}

TEST(BlasLtHandleTests, MoveConstructorTransfersOwnershipAndFreesOnce) {
  int errors = 0;
  {
    CountingHandle source{CountingBlasLtPolicy{&errors}, CountingBlasLtPolicy{&errors}};
    const wwrblasLtHandle_t raw = source.get();

    CountingHandle dest(std::move(source));
    EXPECT_EQ(dest.get(), raw);
    EXPECT_EQ(source.get(), nullptr); // NOLINT(bugprone-use-after-move): moved-from is the contract
  }
  EXPECT_EQ(errors, 0);
}

TEST(BlasLtHandleTests, MoveAssignmentTransfersOwnershipAndFreesOnce) {
  int errors = 0;
  {
    CountingHandle source{CountingBlasLtPolicy{&errors}, CountingBlasLtPolicy{&errors}};
    CountingHandle dest{CountingBlasLtPolicy{&errors}, CountingBlasLtPolicy{&errors}};
    const wwrblasLtHandle_t raw = source.get();

    dest = std::move(source);
    EXPECT_EQ(dest.get(), raw);
    EXPECT_EQ(source.get(), nullptr); // NOLINT(bugprone-use-after-move)
  }
  EXPECT_EQ(errors, 0);
}

TEST(BlasLtHandleTests, SelfMoveAssignmentKeepsHandle) {
  AbortHandle handle;
  const wwrblasLtHandle_t raw = handle.get();
  handle = std::move(handle); // NOLINT(clang-diagnostic-self-move): guarded self-assign must not free
  EXPECT_EQ(handle.get(), raw);
}

TEST(BlasLtHandleTests, MatmulDescLifetime) {
  int errors = 0;
  {
    CountingDesc desc{kComputeType, WWRSOLVER_R_32F, CountingBlasLtPolicy{&errors},
                      CountingBlasLtPolicy{&errors}};
    EXPECT_NE(desc.get(), nullptr);

    CountingDesc moved(std::move(desc));
    EXPECT_NE(moved.get(), nullptr);
    EXPECT_EQ(desc.get(), nullptr); // NOLINT(bugprone-use-after-move)
  }
  EXPECT_EQ(errors, 0);
}

TEST(BlasLtHandleTests, MatrixLayoutLifetime) {
  int errors = 0;
  {
    // A 4x4 column-major float layout.
    CountingLayout layout{WWRSOLVER_R_32F, 4, 4, 4, CountingBlasLtPolicy{&errors},
                          CountingBlasLtPolicy{&errors}};
    EXPECT_NE(layout.get(), nullptr);

    CountingLayout moved(std::move(layout));
    EXPECT_NE(moved.get(), nullptr);
    EXPECT_EQ(layout.get(), nullptr); // NOLINT(bugprone-use-after-move)
  }
  EXPECT_EQ(errors, 0);
}

TEST(BlasLtHandleTests, MatmulPreferenceLifetime) {
  int errors = 0;
  {
    CountingPref pref{CountingBlasLtPolicy{&errors}, CountingBlasLtPolicy{&errors}};
    EXPECT_NE(pref.get(), nullptr);

    CountingPref moved(std::move(pref));
    EXPECT_NE(moved.get(), nullptr);
    EXPECT_EQ(pref.get(), nullptr); // NOLINT(bugprone-use-after-move)
  }
  EXPECT_EQ(errors, 0);
}

} // namespace wwr::extension::test
