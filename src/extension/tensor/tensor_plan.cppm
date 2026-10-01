/**
 * @file tensor_plan.cppm
 * @brief RAII wrapper for a GPU tensor descriptor
 *
 * Provides TensorDescriptorWrapper, which creates a tensor descriptor with
 * wwrtensorCreateTensorDescriptor on construction and frees it with
 * wwrtensorDestroyTensorDescriptor on scope exit.
 *
 * WHY A DESCRIPTOR, NOT A CONTRACTION PLAN. The issue frames this as "the plan
 * RAII", mirroring fft_plan. But a cuTENSOR/hipTensor wwrtensorPlan_t is NOT the
 * free-standing handle a cuFFT plan is: wwrtensorCreatePlan takes an operation
 * descriptor AND a plan preference, and the operation descriptor in turn
 * (wwrtensorCreateContraction/Reduction/...) takes a fully specified
 * contraction -- several tensor descriptors, mode labels, a compute descriptor.
 * There is no way to construct a plan whose lifetime can be exercised without
 * standing up an entire contraction, so a "plan wrapper" tested in isolation
 * would be testing a contraction, not RAII.
 *
 * The tensor descriptor is the smallest piece of that chain that is both
 * genuinely useful (every operation is built from them) and independently
 * constructible: wwrtensorCreateTensorDescriptor needs only a live handle plus a
 * shape/stride/datatype, so its create/destroy pairing -- the leak the RAII
 * removes -- is testable on its own. The operation descriptor, plan preference
 * and plan itself all take the same skip_default_create + adopt shape when a
 * caller has the inputs to build them; this wrapper is the one worth having as a
 * standalone convenience.
 *
 * It takes BaseHandle's skip-default-create path exactly like TextureObject: the
 * create call needs arguments (handle, shape, stride, datatype) the create()
 * hook cannot carry, so the constructor builds the descriptor itself and records
 * ownership with adopt(). The descriptor is a pointer on CUDA and HIP, so
 * liveness rides the null sentinel; adopt()'s bool is ignored for a pointer
 * handle (BaseHandle reads liveness from the handle), but is passed for symmetry
 * with the flagged-handle wrappers.
 *
 * Usage:
 *   import wwr.extension.tensor;
 *
 *   using wwr::extension::kit::AbortPolicy;
 *   TensorHandleWrapper<AbortPolicy<wwrtensorStatus_t>, AbortPolicy<wwrtensorStatus_t>> h;
 *   std::array<std::int64_t, 2> extent{64, 64};
 *   TensorDescriptorWrapper<AbortPolicy<wwrtensorStatus_t>, AbortPolicy<wwrtensorStatus_t>>
 *       desc{h, extent, WWRTENSOR_R_32F};
 *   // wwrtensorDestroyTensorDescriptor on scope exit
 */

export module wwr.extension.tensor:tensor_plan;

import :tensor_error;
import :tensor_handle;
import wwr.tensor;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for a GPU tensor descriptor
 *
 * @tparam P_create Error policy type for creation, typed to wwrtensorStatus_t
 * @tparam P_destroy Error policy type for destruction, typed to wwrtensorStatus_t
 *
 * @note P_destroy MUST NOT THROW -- it is called from the destructor.
 */
template<error_policy<wwrtensorStatus_t> P_create,
         nothrow_error_policy<wwrtensorStatus_t> P_destroy>
class TensorDescriptorWrapper
    : public BaseHandle<wwrtensorTensorDescriptor_t,
                        TensorDescriptorWrapper<P_create, P_destroy>, P_create, P_destroy> {
  using Base = BaseHandle<wwrtensorTensorDescriptor_t,
                          TensorDescriptorWrapper<P_create, P_destroy>, P_create, P_destroy>;

public:
  /// @brief Create a tensor descriptor with a packed (compiler-chosen) stride
  /// @param handle A live tensor library handle (wwrtensorCreateTensorDescriptor's
  ///        first argument); its create call is read, not retained
  /// @param extent The tensor's per-mode extents; its size is the mode count
  /// @param data_type The element data type (e.g. WWRTENSOR_R_32F)
  /// @param alignment Alignment requirement in bytes of the data pointer the
  ///        descriptor will be used with (256 suits any cudaMalloc/hipMalloc
  ///        allocation)
  /// @param policy_create Error policy for the create call
  /// @param policy_destroy Error policy for the destroy call (must not throw)
  /// @param location Source location where creation was requested
  ///
  /// @note A null stride asks the vendor for the natural (generalized-packed)
  ///       layout, which is what the common dense case wants; a caller needing an
  ///       explicit stride uses the raw wwrtensorCreateTensorDescriptor directly.
  TensorDescriptorWrapper(wwrtensorHandle_t handle, std::span<const std::int64_t> extent,
                          wwrtensorDataType_t data_type, std::uint32_t alignment = 256,
                          P_create policy_create = {}, P_destroy policy_destroy = {},
                          std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_create_t{}, std::move(policy_create),
             std::move(policy_destroy)) {
    this->adopt(gpu_check(
        wwrtensorCreateTensorDescriptor(handle, &this->handle_,
                                        static_cast<std::uint32_t>(extent.size()), extent.data(),
                                        /*stride=*/nullptr, data_type, alignment),
        this->create_policy(), location));
  }

  /// @brief Destroy the tensor descriptor (BaseHandle destructor hook)
  /// @param handle The descriptor to destroy
  void destroy(wwrtensorTensorDescriptor_t handle) {
    gpu_check(wwrtensorDestroyTensorDescriptor(handle), this->destroy_policy());
  }
};

} // namespace wwr::extension
