/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.fft
 *
 * The error-handling and RAII-plan layer for GPU FFT (cuFFT or hipFFT, per
 * WWR_GPU_BACKEND). The type-safe execution wrappers built on top of it
 * live separately in wwr.wrappers.fft. It aggregates:
 * - :fft_error - Error code specializations for gpufftResult_t
 * - :fft_plan - RAII wrapper for a GPU FFT plan handle
 * - :convenience_fft - Default-policy alias (FftPlan)
 *
 * Usage:
 *   import wwr.extension.fft;
 *   using namespace wwr::extension;
 */

export module wwr.extension.fft;

import std;

// Re-export the vendor FFT module: gpufftHandle is the return type of
// FftPlan::get() and its conversion operator (and gpufftResult_t is its error
// type), so a consumer can name them without importing wwr.fft separately.
export import wwr.fft;
export import :fft_error;
export import :fft_plan;
export import :convenience_fft;
