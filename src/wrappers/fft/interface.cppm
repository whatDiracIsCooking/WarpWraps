/**
 * @file interface.cppm
 * @brief Primary interface for gpumod.wrappers.fft
 *
 * This module provides type-safe C++ wrappers for GPU FFT operations (cuFFT or
 * hipFFT, per WWR_GPU_BACKEND). It aggregates:
 * - :type_traits - Type system and concepts
 * - :exec - Type-safe execution wrappers (exec_c2c / exec_r2c / exec_c2r)
 *
 * Usage:
 *   import gpumod.wrappers.fft;
 *   using namespace wwr;
 */

module;

export module gpumod.wrappers.fft;

import std;

export import :type_traits;
export import :exec;
