/**
 * @file instantiations.cpp
 * @brief Explicit template instantiations for the GPU FFT wrappers
 *
 * This file contains explicit template instantiations to avoid code bloat
 * from implicit instantiation at every call site. Hand-written -- the
 * `extern template` declarations live in exec.cppm, next to each function.
 */

module wwr.wrappers.fft;

import wwr.fft;

namespace wwr {

// Function: exec_c2c
template wwrfftResult_t exec_c2c<float>(wwrfftHandle, wwrfftComplex *, wwrfftComplex *, int);
template wwrfftResult_t exec_c2c<double>(wwrfftHandle, wwrfftDoubleComplex *, wwrfftDoubleComplex *,
                                         int);

// Function: exec_r2c
template wwrfftResult_t exec_r2c<float>(wwrfftHandle, float *, wwrfftComplex *);
template wwrfftResult_t exec_r2c<double>(wwrfftHandle, double *, wwrfftDoubleComplex *);

// Function: exec_c2r
template wwrfftResult_t exec_c2r<float>(wwrfftHandle, wwrfftComplex *, float *);
template wwrfftResult_t exec_c2r<double>(wwrfftHandle, wwrfftDoubleComplex *, double *);

} // namespace wwr
