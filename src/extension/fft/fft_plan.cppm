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
import gpumod.runtime_api;
import gpumod.extension.common;
import std;

export namespace gpumod::extension {

/**
 * @brief RAII wrapper for a GPU FFT plan handle
 *
 * A cuFFT/hipFFT plan belongs to whatever device was current when it was
 * created, so like the other library handles it is device-bound: construction
 * selects dev_idx (the first constructor argument, default 0), creates the
 * plan there, and records the device -- read it back with dev_idx(). Destroys
 * the plan on destruction; supports move semantics, copy is deleted.
 *
 * It cannot reuse GpuBoundHandle to get that the way GpublasHandle /
 * GpusolverDnHandle do: GpuBoundHandle sits on BaseGpuHandle, which
 * null-initialises its handle with nullptr and tests it for null, but
 * gpufftHandle is an integer on CUDA (cufftHandle is `int`; hipfftHandle is a
 * pointer), which nullptr neither initialises nor compares against. cuFFT hands
 * back an opaque index with no reserved invalid value, so ownership is tracked
 * with an explicit `created_` flag rather than a handle-null sentinel, and the
 * select-device / record-device contract is hand-rolled here to match.
 *
 * @note The transform's work area is allocated later, by the raw gpufftMakePlan
 *       / gpufftPlan the caller invokes on the handle; dev_idx() reports the
 *       device selected at gpufftCreate, so keep that device current when
 *       configuring the plan.
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
  int dev_idx_ = -1; ///< Index of the device the plan was created on (-1 until recorded)
  P_create policy_create_{};
  P_destroy policy_destroy_{};

  // Select dev_idx, create the plan there, then record the device actually
  // current -- the hand-rolled equivalent of GpuBoundHandle's create path.
  // gpuSetDevice/gpuGetDevice return gpuError_t, not the plan's gpufftResult_t,
  // so they go through the default checker (abort on failure), not policy_create_.
  void create(int dev_idx, std::source_location location) {
    gpu_check(gpuSetDevice(dev_idx), location);
    created_ = gpu_check(gpufftCreate(&plan_), policy_create_, location);
    gpu_check(gpuGetDevice(&dev_idx_), location);
  }

  void reset() noexcept {
    if (created_) {
      gpu_check(gpufftDestroy(plan_), policy_destroy_);
      created_ = false;
    }
  }

public:
  explicit FftPlanWrapper(int dev_idx = 0,
                          std::source_location location = std::source_location::current()) {
    create(dev_idx, location);
  }

  FftPlanWrapper(P_create policy, int dev_idx = 0,
                 std::source_location location = std::source_location::current())
      : policy_create_(policy), policy_destroy_(std::move(policy)) {
    create(dev_idx, location);
  }

  FftPlanWrapper(P_create policy_create, P_destroy policy_destroy, int dev_idx = 0,
                 std::source_location location = std::source_location::current())
      : policy_create_(std::move(policy_create)), policy_destroy_(std::move(policy_destroy)) {
    create(dev_idx, location);
  }

  ~FftPlanWrapper() { reset(); }

  FftPlanWrapper(FftPlanWrapper &&other) noexcept
      : plan_(other.plan_), created_(other.created_), dev_idx_(other.dev_idx_),
        policy_create_(std::move(other.policy_create_)),
        policy_destroy_(std::move(other.policy_destroy_)) {
    other.created_ = false;
    other.dev_idx_ = -1;
  }

  FftPlanWrapper &operator=(FftPlanWrapper &&other) noexcept {
    if (this != &other) {
      reset();
      plan_ = other.plan_;
      created_ = other.created_;
      dev_idx_ = other.dev_idx_;
      policy_create_ = std::move(other.policy_create_);
      policy_destroy_ = std::move(other.policy_destroy_);
      other.created_ = false;
      other.dev_idx_ = -1;
    }
    return *this;
  }

  // Copy operations are implicitly deleted via the NonCopyable base.

  operator gpufftHandle() const noexcept { return plan_; }
  gpufftHandle get() const noexcept { return plan_; }

  /// @brief Index of the physical device this plan belongs to (-1 if not recorded)
  int dev_idx() const noexcept { return dev_idx_; }
};

/**
 * @brief Convenient alias for FftPlanWrapper with default error policies
 *
 * Usage:
 *   FftPlan plan;  // Instead of FftPlanWrapper<>
 */
using FftPlan = FftPlanWrapper<>;

} // namespace gpumod::extension
