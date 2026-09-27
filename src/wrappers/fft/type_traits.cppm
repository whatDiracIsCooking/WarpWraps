/**
 * @file type_traits.cppm
 * @brief Type system for GPU FFT operations
 *
 * Re-exports gpumod.wrappers.common (the real_fp/complex_fp concepts and
 * type mappings, already backend-neutral over src's gpu* types) and maps
 * an FFT real precision to the FFT library's own complex element type.
 *
 * Usage:
 *   import gpumod.wrappers.fft;
 */

export module gpumod.wrappers.fft:type_traits;

export import gpumod.wrappers.common;
import gpumod.fft;
import std;

export namespace wwr {

/**
 * @brief The FFT library's complex element type for a real precision T.
 *   float  -> gpufftComplex
 *   double -> gpufftDoubleComplex
 *
 * Named after the FFT library's own types (gpufftComplex), not the shared
 * gpuFloatComplex/gpuDoubleComplex: on HIP hipfftComplex is a distinct type
 * from hipComplex, and the gpufftExec* signatures name the former -- so the
 * wrappers over them must too, or the call fails to type-check on HIP.
 */
template<real_fp T>
using FftComplex = std::conditional_t<std::is_same_v<T, float>, gpufftComplex, gpufftDoubleComplex>;

} // namespace wwr
