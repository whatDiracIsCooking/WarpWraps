/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.solver
 *
 * The error-handling and RAII-handle layer for GPU dense solvers (cuSOLVER or
 * hipSOLVER, per WWR_GPU_BACKEND). The type-safe dispatch wrappers built on
 * top of it live separately in wwr.wrappers.solver. It aggregates:
 * - :solver_error - Error code specializations for wwrsolverStatus_t
 * - :solver_handle - RAII wrapper for wwrsolverDnHandle_t
 * - :solver_params - RAII wrapper for wwrsolverDnParams_t
 *
 * Usage:
 *   import wwr.extension.solver;
 *   using namespace wwr::extension;
 */

export module wwr.extension.solver;

import std;

// Re-export the vendor solver module: wwrsolverDnHandle_t / wwrsolverDnParams_t
// are the return types of the wrappers' get() and conversion operators, so a
// consumer can name them without importing wwr.solver separately.
export import wwr.solver;
export import :solver_error;
export import :solver_handle;
export import :solver_params;
