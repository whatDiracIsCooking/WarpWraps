/**
 * @file instantiations.cpp
 * @brief Explicit template instantiations for the GPU FFT wrappers
 *
 * This file contains explicit template instantiations to avoid code bloat
 * from implicit instantiation at every call site. Hand-written -- the
 * `extern template` declarations live in exec.cppm, next to each function.
 */

module gpumod.wrappers.fft;

import gpumod.fft;

namespace gpumod {

// Function: exec_c2c
template gpufftResult_t exec_c2c<float>(gpufftHandle, gpufftComplex *, gpufftComplex *, int);
template gpufftResult_t exec_c2c<double>(gpufftHandle, gpufftDoubleComplex *, gpufftDoubleComplex *,
                                         int);

// Function: exec_r2c
template gpufftResult_t exec_r2c<float>(gpufftHandle, float *, gpufftComplex *);
template gpufftResult_t exec_r2c<double>(gpufftHandle, double *, gpufftDoubleComplex *);

// Function: exec_c2r
template gpufftResult_t exec_c2r<float>(gpufftHandle, gpufftComplex *, float *);
template gpufftResult_t exec_c2r<double>(gpufftHandle, gpufftDoubleComplex *, double *);

} // namespace gpumod
