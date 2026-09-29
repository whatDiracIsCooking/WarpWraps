/**
 * @file type_traits.cppm
 * @brief Type system for GPU solver operations
 *
 * Re-exports wwr.wrappers.common (usual_fp/real_fp/complex_fp
 * concepts and ComplexToRealType, already backend-neutral over src's
 * wwr* complex types) and provides the wwrsolverDataType_t mapping the
 * modern (X-prefixed) API needs.
 *
 * Usage:
 *   import wwr.wrappers.solver;
 */

export module wwr.wrappers.solver:type_traits;

export import wwr.wrappers.common;
import wwr.solver;
import wwr.complex;
import std;

export namespace wwr {

/**
 * @brief Get the wwrsolverDataType_t enum for a GPU-solver-supported type
 *
 * Maps C++ types to their corresponding cudaDataType/hipDataType
 * enumeration value, whichever backend this build is configured for:
 * - float -> WWRSOLVER_R_32F (32-bit real)
 * - double -> WWRSOLVER_R_64F (64-bit real)
 * - wwrFloatComplex -> WWRSOLVER_C_32F (32-bit complex)
 * - wwrDoubleComplex -> WWRSOLVER_C_64F (64-bit complex)
 *
 * @tparam T The type to map (must satisfy usual_fp concept)
 * @return The corresponding wwrsolverDataType_t enumeration value
 */
template<usual_fp T>
constexpr wwrsolverDataType_t get_wwrsolver_type() noexcept {
  if constexpr (std::is_same_v<T, float>) {
    return WWRSOLVER_R_32F;
  } else if constexpr (std::is_same_v<T, double>) {
    return WWRSOLVER_R_64F;
  } else if constexpr (std::is_same_v<T, wwrFloatComplex>) {
    return WWRSOLVER_C_32F;
  } else if constexpr (std::is_same_v<T, wwrDoubleComplex>) {
    return WWRSOLVER_C_64F;
  }
}

} // namespace wwr
