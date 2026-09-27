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
import gpumod.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for a GPU FFT plan handle
 *
 * A cuFFT/hipFFT plan belongs to whatever device was current when it was
 * created, so this derives from DeviceBoundHandle exactly like GpublasHandle /
 * GpusolverDnHandle: construction selects dev_idx (the first constructor
 * argument, default 0), creates the plan there, and records it -- read it back
 * with dev_idx(). Destroys the plan on destruction; supports move semantics,
 * copy is deleted.
 *
 * The plan used to hand-roll all of that because gpufftHandle is an integer on
 * CUDA (cufftHandle is `int`; hipfftHandle is a pointer) with no reserved
 * invalid value, so it could not ride BaseHandle's null-sentinel liveness
 * test. BaseHandle now tracks liveness with an explicit flag for exactly the
 * handle types with no in-band null, so the plan needs nothing more than the
 * create()/destroy() hooks below.
 *
 * @note The transform's work area is allocated later, by the raw gpufftMakePlan
 *       / gpufftPlan the caller invokes on the handle (the plan converts
 *       implicitly to gpufftHandle); dev_idx() reports the device selected at
 *       gpufftCreate, so keep that device current when configuring the plan.
 *
 * @tparam P_create Error policy type for creation (defaults to DefaultErrorPolicy<gpufftResult_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<gpufftResult_t> P_create = DefaultErrorPolicy<gpufftResult_t>,
         nothrow_error_policy<gpufftResult_t> P_destroy = P_create>
class FftPlanWrapper
    : public DeviceBoundHandle<gpufftHandle, FftPlanWrapper<P_create, P_destroy>, P_create,
                            P_destroy> {
private:
  using Base =
      DeviceBoundHandle<gpufftHandle, FftPlanWrapper<P_create, P_destroy>, P_create, P_destroy>;

public:
  // The dev_idx / policy constructors, inherited from DeviceBoundHandle, which
  // selects and records the owning device.
  using DeviceBoundHandle<gpufftHandle, FftPlanWrapper<P_create, P_destroy>, P_create,
                       P_destroy>::DeviceBoundHandle;

  /// @brief Create a bare GPU FFT plan handle
  /// @param handle Output parameter for the created plan
  /// @param location Source location where creation was requested
  /// @return Whether the plan was created -- gpufftHandle has no null sentinel,
  ///         so BaseHandle tracks ownership from this bool.
  bool create(gpufftHandle *handle, std::source_location location) {
    return gpu_check(gpufftCreate(handle), this->policy_create_, location);
  }

  /// @brief Destroy a GPU FFT plan handle
  /// @param handle The plan to destroy
  void destroy(gpufftHandle handle) {
    gpu_check(gpufftDestroy(handle), this->policy_destroy_);
  }
};

} // namespace wwr::extension
