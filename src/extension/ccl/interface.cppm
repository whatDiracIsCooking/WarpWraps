/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.ccl
 *
 * The error-handling and RAII-communicator layer for GPU collectives (NCCL or
 * RCCL, per WWR_GPU_BACKEND). Unlike blas/fft, there is no wwr.wrappers.ccl
 * execution layer: a collective's element type is a runtime wwrcclDataType_t
 * enum, not a name-letter the dispatch-on-type wrappers key on, so NCCL/RCCL get
 * no layer-2 generic wrapper. The convenience here is RAII + typed errors only.
 * It aggregates:
 * - :ccl_error - Error code specializations for wwrcclResult_t
 * - :ccl_comm  - RAII wrapper for a wwrcclComm_t communicator
 *
 * Usage:
 *   import wwr.extension.ccl;
 *   using namespace wwr::extension;
 */

export module wwr.extension.ccl;

import std;

// Re-export the vendor collectives module: wwrcclComm_t is the return type of
// CommWrapper::get() and its conversion operator (and wwrcclResult_t is its
// error type), so a consumer can name them without importing wwr.ccl separately.
export import wwr.ccl;
export import :ccl_error;
export import :ccl_comm;
