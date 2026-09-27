/**
 * @file type_traits.cppm
 * @brief Type system for GPU solver operations
 *
 * Re-exports gpumod.wrappers.common (usual_fp/real_fp/complex_fp
 * concepts and ComplexToRealType, already backend-neutral over src's
 * gpu* complex types) and provides the gpusolverDataType_t mapping the
 * modern (X-prefixed) API needs.
 *
 * Usage:
 *   import gpumod.wrappers.solver;
 */

export module gpumod.wrappers.solver:type_traits;

export import gpumod.wrappers.common;
import gpumod.solver;
import gpumod.complex;
import std;

export namespace wwr {

/**
 * @brief Get the gpusolverDataType_t enum for a GPU-solver-supported type
 *
 * Maps C++ types to their corresponding cudaDataType/hipDataType
 * enumeration value, whichever backend this build is configured for:
 * - float -> GPUSOLVER_R_32F (32-bit real)
 * - double -> GPUSOLVER_R_64F (64-bit real)
 * - gpuFloatComplex -> GPUSOLVER_C_32F (32-bit complex)
 * - gpuDoubleComplex -> GPUSOLVER_C_64F (64-bit complex)
 *
 * @tparam T The type to map (must satisfy usual_fp concept)
 * @return The corresponding gpusolverDataType_t enumeration value
 */
template<usual_fp T>
constexpr gpusolverDataType_t get_gpusolver_type() noexcept {
  if constexpr (std::is_same_v<T, float>) {
    return GPUSOLVER_R_32F;
  } else if constexpr (std::is_same_v<T, double>) {
    return GPUSOLVER_R_64F;
  } else if constexpr (std::is_same_v<T, gpuFloatComplex>) {
    return GPUSOLVER_C_32F;
  } else if constexpr (std::is_same_v<T, gpuDoubleComplex>) {
    return GPUSOLVER_C_64F;
  }
}

} // namespace wwr
