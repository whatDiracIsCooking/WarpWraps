/**
 * @file fft.h
 * @brief The single cuFFT / hipFFT vendor-include point for the fft layer
 *
 * A src/-root header that brings in the vendor FFT header for whichever backend
 * was selected, and nothing else: the wwrfft* surface itself is the shared
 * fragment detail/fft_names.h, bound to the `::cufft*` / `::hipfft*`
 * declarations this header supplies. It is the "rand.h shape" for a host-only
 * vendor library -- fft.cppm draws its whole surface from here (binding wwr*
 * references straight to the declarations with the _RAW macros, importing no raw
 * vendor module), and wwr/fft.h (the non-module #include path) does the same, so
 * this is the one place the vendor header enters the fft layer. The raw module
 * wwr.cuda.cufft / wwr.hip.hipfft stays as a single-vendor surface for the
 * base-API extras off this path (wwrfftGetProperty, the per-plan property
 * functions), and wwr.cuda.cufftXt / wwr.hip.hipfftXt for the multi-GPU Xt
 * surface. See src/fft.cppm and src/README.md.
 *
 * HOST only -- cuFFT/hipFFT's plan and exec API is host-side, so unlike rand.h /
 * complex.h this header carries no device-pass-gated section. It reaches
 * selected_backend.h directly, not device_guard.h, so it compiles in the host
 * TUs that include it.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from WWR_GPU_BACKEND_* in a host compile.
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

#include <cufft.h>

#else

// Load-bearing, and must stay before the HIP header: host_defines.h (pulled in
// transitively by the hipFFT header) poisons __noinline__ for libc++'s
// __config, so a HIP header reached before <array> makes __config fail to
// compile. Same pre-include src/hip/hipfft.cppm's global module fragment
// carries. docs/architecture.md, section 9.
#include <array>

#include <hipfft/hipfft.h>

#endif
