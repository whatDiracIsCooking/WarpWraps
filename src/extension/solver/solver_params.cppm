/**
 * @file solver_params.cppm
 * @brief RAII wrapper for GPU solver params
 *
 * Provides SolverDnParamsWrapper class for automatic GPU solver params management.
 */

export module wwr.extension.solver:solver_params;

import :solver_error;
import wwr.solver;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for GPU solver params
 *
 * Automatically creates GPU solver params on construction and destroys them on destruction.
 * Supports move semantics for transferring ownership.
 *
 * @tparam P_create Error policy type for creation
 * @tparam P_destroy Error policy type for destruction
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<wwrsolverStatus_t> P_create,
         nothrow_error_policy<wwrsolverStatus_t> P_destroy>
class SolverDnParamsWrapper
    : public BaseHandle<wwrsolverDnParams_t, SolverDnParamsWrapper<P_create, P_destroy>,
                           P_create, P_destroy> {
private:
  using Base = BaseHandle<wwrsolverDnParams_t, SolverDnParamsWrapper<P_create, P_destroy>,
                             P_create, P_destroy>;

public:
  // Default constructors - inherited from base
  using BaseHandle<wwrsolverDnParams_t, SolverDnParamsWrapper<P_create, P_destroy>, P_create,
                      P_destroy>::BaseHandle;

  /// @brief Create GPU solver params
  /// @param params Output parameter for the created params
  /// @param location Source location where creation was requested
  void create(wwrsolverDnParams_t *params, std::source_location location) {
    gpu_check(wwrsolverDnCreateParams(params), this->policy_create_, location);
  }

  /// @brief Destroy GPU solver params
  /// @param params The params to destroy
  void destroy(wwrsolverDnParams_t params) {
    if (params != nullptr) {
      gpu_check(wwrsolverDnDestroyParams(params), this->policy_destroy_);
    }
  }
};

} // namespace wwr::extension
