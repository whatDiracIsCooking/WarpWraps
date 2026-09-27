/**
 * @file interface.cppm
 * @brief Primary interface for gpumod.extension.solver
 *
 * The error-handling and RAII-handle layer for GPU dense solvers (cuSOLVER or
 * hipSOLVER, per GPUMOD_GPU_BACKEND). The type-safe dispatch wrappers built on
 * top of it live separately in gpumod.wrappers.solver. It aggregates:
 * - :solver_error - Error code specializations for gpusolverStatus_t
 * - :solver_handle - RAII wrapper for gpusolverDnHandle_t
 * - :solver_params - RAII wrapper for gpusolverDnParams_t
 * - :convenience_solver - Default-policy aliases (GpusolverDnHandle, GpusolverDnHandleView, GpusolverDnParams)
 *
 * Usage:
 *   import gpumod.extension.solver;
 *   using namespace gpumod::extension;
 */

export module gpumod.extension.solver;

import std;

// Re-export the vendor solver module: gpusolverDnHandle_t / gpusolverDnParams_t
// are the return types of the wrappers' get() and conversion operators, so a
// consumer can name them without importing gpumod.solver separately.
export import gpumod.solver;
export import :solver_error;
export import :solver_handle;
export import :solver_params;
export import :convenience_solver;
