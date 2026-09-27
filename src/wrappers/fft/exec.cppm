/**
 * @file exec.cppm
 * @brief Type-safe GPU FFT execution wrappers
 *
 * The six gpufftExec* entry points collapse into three functions templated on
 * the real precision T (float or double): exec_c2c, exec_r2c, exec_c2r. Each
 * token-pastes its precision-specific transform kind onto gpufftExec via
 * dispatch_macros.h. The complex element type is the FFT library's own
 * (FftComplex<T> = gpufftComplex / gpufftDoubleComplex), so the wrapper
 * signatures match the vendor functions on both backends.
 *
 * Usage:
 *   import gpumod.wrappers.fft;
 *   using namespace wwr;
 *
 *   FftPlan plan;
 *   gpufftMakePlan1d(plan, n, GPUFFT_C2C, 1, &work);
 *   exec_c2c<float>(plan, in, out, GPUFFT_FORWARD);
 */

module;

#include "dispatch_macros.h"

export module gpumod.wrappers.fft:exec;

import gpumod.fft;
import :type_traits;
import std;

export namespace wwr {

// ========================================================================
// Complex-to-complex: C2C (float) / Z2Z (double)
// ========================================================================

/**
 * @brief Execute a complex-to-complex transform in the given direction.
 *
 * @tparam T Real precision (float -> C2C, double -> Z2Z)
 * @param plan A configured FFT plan for the matching transform type
 * @param idata Input complex data (device memory)
 * @param odata Output complex data (device memory); may alias idata (in-place)
 * @param direction GPUFFT_FORWARD or GPUFFT_INVERSE
 * @return gpufftResult_t status code
 */
template<real_fp T>
gpufftResult_t exec_c2c(gpufftHandle plan, FftComplex<T> *idata, FftComplex<T> *odata,
                        int direction) {
  WWR_FFT_EXEC_DISPATCH(T, C2C, Z2Z, plan, idata, odata, direction);
}

// ========================================================================
// Real-to-complex (forward): R2C (float) / D2Z (double)
// ========================================================================

/**
 * @brief Execute a real-to-complex forward transform.
 *
 * @tparam T Real precision (float -> R2C, double -> D2Z)
 * @param plan A configured FFT plan for the matching transform type
 * @param idata Input real data (device memory)
 * @param odata Output complex data (device memory)
 * @return gpufftResult_t status code
 */
template<real_fp T>
gpufftResult_t exec_r2c(gpufftHandle plan, T *idata, FftComplex<T> *odata) {
  WWR_FFT_EXEC_DISPATCH(T, R2C, D2Z, plan, idata, odata);
}

// ========================================================================
// Complex-to-real (inverse): C2R (float) / Z2D (double)
// ========================================================================

/**
 * @brief Execute a complex-to-real inverse transform.
 *
 * @tparam T Real precision (float -> C2R, double -> Z2D)
 * @param plan A configured FFT plan for the matching transform type
 * @param idata Input complex data (device memory)
 * @param odata Output real data (device memory)
 * @return gpufftResult_t status code
 */
template<real_fp T>
gpufftResult_t exec_c2r(gpufftHandle plan, FftComplex<T> *idata, T *odata) {
  WWR_FFT_EXEC_DISPATCH(T, C2R, Z2D, plan, idata, odata);
}

// ==================== Explicit Template Instantiations ====================
// Hand-written, one per (function, precision). Matching `template`
// instantiations live in instantiations.cpp.

// Function: exec_c2c
extern template gpufftResult_t exec_c2c<float>(gpufftHandle, gpufftComplex *, gpufftComplex *, int);
extern template gpufftResult_t exec_c2c<double>(gpufftHandle, gpufftDoubleComplex *,
                                                gpufftDoubleComplex *, int);

// Function: exec_r2c
extern template gpufftResult_t exec_r2c<float>(gpufftHandle, float *, gpufftComplex *);
extern template gpufftResult_t exec_r2c<double>(gpufftHandle, double *, gpufftDoubleComplex *);

// Function: exec_c2r
extern template gpufftResult_t exec_c2r<float>(gpufftHandle, gpufftComplex *, float *);
extern template gpufftResult_t exec_c2r<double>(gpufftHandle, gpufftDoubleComplex *, double *);

} // namespace wwr
