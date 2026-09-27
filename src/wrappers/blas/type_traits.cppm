/**
 * @file type_traits.cppm
 * @brief Type system for GPU BLAS operations
 *
 * Re-exports gpumod.wrappers.common (fp/int concepts and type mappings)
 * and provides half-precision type traits for GPU BLAS.
 *
 * Usage:
 *   import gpumod.wrappers.blas;
 */

module;

export module gpumod.wrappers.blas:type_traits;

export import gpumod.wrappers.common;
import gpumod.fp16;
import gpumod.bf16;

export namespace wwr {

// ========================================================================
// Type Traits
// ========================================================================

/**
 * @brief Helper to map half types to their single precision float types,
 * e.g. gemvStridedBatched
 */
template<half_fp T>
struct GetSinglePrecisionType {
  using type = T;
};

template<>
struct GetSinglePrecisionType<gpuHalf> {
  using type = float;
};

template<>
struct GetSinglePrecisionType<gpuBfloat16> {
  using type = float;
};

/**
 * @brief Type alias to get the single precision type for half-precision types
 */
template<half_fp T>
using SinglePrecisionType = typename GetSinglePrecisionType<T>::type;

} // namespace wwr
