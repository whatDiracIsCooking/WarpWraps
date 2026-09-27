/**
 * @file solver_params.cppm
 * @brief RAII wrapper for GPU solver params
 *
 * Provides GpusolverDnParams class for automatic GPU solver params management.
 */

export module gpumod.extension.solver:solver_params;

import :solver_error;
import gpumod.solver;
import gpumod.extension.common;
import gpumod.extension.handle;
import std;

export namespace gpumod::extension {

/**
 * @brief RAII wrapper for GPU solver params
 *
 * Automatically creates GPU solver params on construction and destroys them on destruction.
 * Supports move semantics for transferring ownership.
 *
 * @tparam P_create Error policy type for creation (defaults to DefaultErrorPolicy<gpusolverStatus_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<gpusolverStatus_t> P_create = DefaultErrorPolicy<gpusolverStatus_t>,
         nothrow_error_policy<gpusolverStatus_t> P_destroy = P_create>
class GpusolverDnParamsWrapper
    : public BaseHandle<gpusolverDnParams_t, GpusolverDnParamsWrapper<P_create, P_destroy>,
                           P_create, P_destroy> {
private:
  using Base = BaseHandle<gpusolverDnParams_t, GpusolverDnParamsWrapper<P_create, P_destroy>,
                             P_create, P_destroy>;

public:
  // Default constructors - inherited from base
  using BaseHandle<gpusolverDnParams_t, GpusolverDnParamsWrapper<P_create, P_destroy>, P_create,
                      P_destroy>::BaseHandle;

  /// @brief Create GPU solver params
  /// @param params Output parameter for the created params
  /// @param location Source location where creation was requested
  void create(gpusolverDnParams_t *params, std::source_location location) {
    gpu_check(gpusolverDnCreateParams(params), this->policy_create_, location);
  }

  /// @brief Destroy GPU solver params
  /// @param params The params to destroy
  void destroy(gpusolverDnParams_t params) {
    if (params != nullptr) {
      gpu_check(gpusolverDnDestroyParams(params), this->policy_destroy_);
    }
  }
};

} // namespace gpumod::extension
