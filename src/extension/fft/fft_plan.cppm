/**
 * @file fft_plan.cppm
 * @brief RAII wrapper for a GPU FFT plan handle
 *
 * Provides FftPlanWrapper, which creates a bare plan handle with wwrfftCreate on
 * construction and destroys it with wwrfftDestroy on destruction. Configure
 * the transform afterwards with the raw wwrfftMakePlan / wwrfftPlan functions
 * (the plan converts implicitly to wwrfftHandle), then run it with the typed
 * exec_c2c / exec_r2c / exec_c2r wrappers.
 */

export module wwr.extension.fft:fft_plan;

import :fft_error;
import wwr.fft;
import wwr.runtime_api; // wwrError_t (the device-access policy's error type)
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for a GPU FFT plan handle bound to a work stream
 *
 * A cuFFT/hipFFT plan binds a work stream and enqueues asynchronous work on it,
 * so this derives from StreamBoundHandle exactly like BlasHandleWrapper /
 * SolverDnHandleWrapper: it is constructed from a shared stream owner (never a
 * bare device index), created on that owner's device, bound to that owner's
 * stream via wwrfftSetStream, and retains the owner so the stream outlives it.
 * Read the owning device back with dev_idx() and the bound stream with stream()
 * (it is a device_handle_stream). Destroys the plan on destruction; supports move
 * semantics, copy is deleted.
 *
 * The plan used to hand-roll device binding because wwrfftHandle is an integer on
 * CUDA (cufftHandle is `int`; hipfftHandle is a pointer) with no reserved
 * invalid value, so it could not ride BaseHandle's null-sentinel liveness
 * test. BaseHandle now tracks liveness with an explicit flag for exactly the
 * handle types with no in-band null, so the plan needs nothing more than the
 * create()/destroy()/set_stream() hooks below.
 *
 * @note The transform's work area is allocated later, by the raw wwrfftMakePlan
 *       / wwrfftPlan the caller invokes on the handle (the plan converts
 *       implicitly to wwrfftHandle); dev_idx() reports the device selected at
 *       wwrfftCreate, so keep that device current when configuring the plan.
 * @note wwrfftSetStream has no wwrfftGetStream counterpart, but stream() needs
 *       neither: StreamBoundHandle reads it from the retained owner, never back
 *       from the plan handle.
 *
 * @tparam P_create Error policy type for creation (and the wwrfftSetStream bind)
 * @tparam P_destroy Error policy type for destruction
 * @tparam S The stream owner's type (a device_handle_stream), retained so the
 *         bound stream outlives the plan -- as DeviceBufferWrapper retains H.
 * @tparam P_device_access Error policy for the device set/get calls
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<wwrfftResult_t> P_create,
         nothrow_error_policy<wwrfftResult_t> P_destroy, device_handle_stream S,
         error_policy<wwrError_t> P_device_access>
class FftPlanWrapper
    : public StreamBoundHandle<wwrfftHandle, FftPlanWrapper<P_create, P_destroy, S, P_device_access>,
                            P_create, P_destroy, S, P_device_access> {
public:
  // The `FftPlanWrapper(shared_ptr<S> owner)` stream-owner constructor, inherited
  // from StreamBoundHandle, which creates the plan on the owner's device and binds
  // its stream (see set_stream below).
  using StreamBoundHandle<wwrfftHandle, FftPlanWrapper<P_create, P_destroy, S, P_device_access>,
                       P_create, P_destroy, S, P_device_access>::StreamBoundHandle;

  /// @brief Create a bare GPU FFT plan handle
  /// @param handle Output parameter for the created plan
  /// @param location Source location where creation was requested
  /// @return Whether the plan was created -- wwrfftHandle has no null sentinel,
  ///         so BaseHandle tracks ownership from this bool.
  bool create(wwrfftHandle *handle, std::source_location location) {
    return gpu_check(wwrfftCreate(handle), this->policy_create_, location);
  }

  /// @brief Destroy a GPU FFT plan handle
  /// @param handle The plan to destroy
  void destroy(wwrfftHandle handle) {
    gpu_check(wwrfftDestroy(handle), this->destroy_policy());
  }

  /// @brief Bind the work stream onto the plan (StreamBoundHandle hook)
  /// @param handle The plan to bind the stream onto
  /// @param stream The work stream to bind
  /// @param location Source location where the bind was requested
  void set_stream(wwrfftHandle handle, wwrStream_t stream, std::source_location location) {
    gpu_check(wwrfftSetStream(handle, stream), this->policy_create_, location);
  }
};

} // namespace wwr::extension
