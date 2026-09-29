/**
 * @file type_traits.cppm
 * @brief Type system for GPU FFT operations
 *
 * Re-exports wwr.wrappers.common (the real_fp/complex_fp concepts and
 * type mappings, already backend-neutral over src's wwr* types) and maps
 * an FFT real precision to the FFT library's own complex element type.
 *
 * Usage:
 *   import wwr.wrappers.fft;
 */

export module wwr.wrappers.fft:type_traits;

export import wwr.wrappers.common;
import wwr.fft;
import std;

export namespace wwr {

/**
 * @brief The FFT library's complex element type for a real precision T.
 *   float  -> wwrfftComplex
 *   double -> wwrfftDoubleComplex
 *
 * Named after the FFT library's own types (wwrfftComplex), not the shared
 * wwrFloatComplex/wwrDoubleComplex: on HIP hipfftComplex is a distinct type
 * from hipComplex, and the wwrfftExec* signatures name the former -- so the
 * wrappers over them must too, or the call fails to type-check on HIP.
 */
template<real_fp T>
using FftComplex = std::conditional_t<std::is_same_v<T, float>, wwrfftComplex, wwrfftDoubleComplex>;

} // namespace wwr
