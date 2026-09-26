/**
 * @file fft_plan.cppm
 * @brief RAII wrapper for a GPU FFT plan handle
 *
 * Provides FftPlan, which creates a bare plan handle with gpufftCreate on
 * construction and destroys it with gpufftDestroy on destruction. Configure
 * the transform afterwards with the raw gpufftMakePlan / gpufftPlan functions
 * (the plan converts implicitly to gpufftHandle), then run it with the typed
 * exec_c2c / exec_r2c / exec_c2r wrappers.
 */

export module gpumod.extension.fft:fft_plan;

import :fft_error;
import gpumod.fft;
import gpumod.extension.common;
import std;

export namespace gpumod::extension {

/**
 * @brief RAII wrapper for a GPU FFT plan handle
 *
 * Automatically creates a plan handle on construction and destroys it on
 * destruction. Supports move semantics for transferring ownership; copy is
 * deleted.
 *
 * Unlike GpublasHandle / GpusolverDnHandle this does NOT derive from
 * BaseGpuHandle: that base null-initialises its handle with nullptr and tests
 * it for null, but gpufftHandle is an integer on CUDA (cufftHandle is `int`;
 * hipfftHandle is a pointer), which nullptr neither initialises nor compares
 * against. cuFFT hands back an opaque index with no reserved invalid value, so
 * ownership is tracked with an explicit `created_` flag rather than a
 * handle-null sentinel.
 *
 * @tparam P_create Error policy type for creation (defaults to DefaultErrorPolicy<gpufftResult_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<gpufftResult_t> P_create = DefaultErrorPolicy<gpufftResult_t>,
         error_policy<gpufftResult_t> P_destroy = P_create>
class FftPlanWrapper : private NonCopyable {
private:
  gpufftHandle plan_{};
  bool created_ = false;
  P_create policy_create_{};
  P_destroy policy_destroy_{};

  void create(std::source_location location) {
    created_ = gpu_check(gpufftCreate(&plan_), policy_create_, location);
  }

  void reset() noexcept {
    if (created_) {
      gpu_check(gpufftDestroy(plan_), policy_destroy_);
      created_ = false;
    }
  }

public:
  FftPlanWrapper(std::source_location location = std::source_location::current()) {
    create(location);
  }

  FftPlanWrapper(P_create policy, std::source_location location = std::source_location::current())
      : policy_create_(policy), policy_destroy_(std::move(policy)) {
    create(location);
  }

  FftPlanWrapper(P_create policy_create, P_destroy policy_destroy,
                 std::source_location location = std::source_location::current())
      : policy_create_(std::move(policy_create)), policy_destroy_(std::move(policy_destroy)) {
    create(location);
  }

  ~FftPlanWrapper() { reset(); }

  FftPlanWrapper(FftPlanWrapper &&other) noexcept
      : plan_(other.plan_), created_(other.created_),
        policy_create_(std::move(other.policy_create_)),
        policy_destroy_(std::move(other.policy_destroy_)) {
    other.created_ = false;
  }

  FftPlanWrapper &operator=(FftPlanWrapper &&other) noexcept {
    if (this != &other) {
      reset();
      plan_ = other.plan_;
      created_ = other.created_;
      policy_create_ = std::move(other.policy_create_);
      policy_destroy_ = std::move(other.policy_destroy_);
      other.created_ = false;
    }
    return *this;
  }

  // Copy operations are implicitly deleted via the NonCopyable base.

  operator gpufftHandle() const noexcept { return plan_; }
  gpufftHandle get() const noexcept { return plan_; }
};

/**
 * @brief Convenient alias for FftPlanWrapper with default error policies
 *
 * Usage:
 *   FftPlan plan;  // Instead of FftPlanWrapper<>
 */
using FftPlan = FftPlanWrapper<>;

} // namespace gpumod::extension
