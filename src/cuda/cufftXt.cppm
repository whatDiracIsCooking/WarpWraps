/**
 * @file cufftXt.cppm
 * @brief Primary interface for wwr.cuda.cufftXt
 *
 * This module wraps the native cuFFT eXtended (cufftXt) API and exports
 * types, enums, and functions for multi-GPU FFT operations and callbacks.
 * cufftXt extends cufft with multi-GPU support; cufft.h is included
 * transitively through cufftXt.h.
 *
 * Usage:
 *   import wwr.cuda.cufftXt;
 */

module;

#include <cufftXt.h>

export module wwr.cuda.cufftXt;

// ========================================================================
// Export all cufftXt types and functions in wwr namespace
// ========================================================================

export namespace wwr::cuda {

// ========================================================================
// Multi-GPU descriptor types (from cudalibxt.h, used in cufftXt signatures)
// ========================================================================

// Copy direction for descriptor-level transfers
using ::cudaLibXtCopyType;
using ::cudaXtCopyType_t;

// cudaXtCopyType_t enumerators
using ::LIB_XT_COPY_DEVICE_TO_DEVICE;
using ::LIB_XT_COPY_DEVICE_TO_HOST;
using ::LIB_XT_COPY_HOST_TO_DEVICE;

// Library format tag stored inside cudaLibXtDesc
using ::libFormat;
using ::libFormat_t;

// libFormat_t enumerators
using ::LIB_FORMAT_CUFFT;
using ::LIB_FORMAT_UNDEFINED;

// Per-GPU memory descriptor (array-of-pointers per device)
using ::cudaXtDesc;
using ::cudaXtDesc_t;

// Library-level descriptor wrapping cudaXtDesc
using ::cudaLibXtDesc;
using ::cudaLibXtDesc_t;

// ========================================================================
// cufftXt sub-format enum
// Describes the data layout of a cudaLibXtDesc owned by cuFFT.
// ========================================================================
using ::cufftXtSubFormat;
using ::cufftXtSubFormat_t;

// cufftXtSubFormat_t enumerators
using ::CUFFT_FORMAT_UNDEFINED;
using ::CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED;
using ::CUFFT_XT_FORMAT_DISTRIBUTED_INPUT;
using ::CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT;
using ::CUFFT_XT_FORMAT_INPLACE;
using ::CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
using ::CUFFT_XT_FORMAT_INPUT;
using ::CUFFT_XT_FORMAT_OUTPUT;

// ========================================================================
// cufftXt copy type enum
// Specifies the direction of copy for cufftXtMemcpy.
// ========================================================================
using ::cufftXtCopyType;
using ::cufftXtCopyType_t;

// cufftXtCopyType_t enumerators
using ::CUFFT_COPY_DEVICE_TO_DEVICE;
using ::CUFFT_COPY_DEVICE_TO_HOST;
using ::CUFFT_COPY_HOST_TO_DEVICE;
using ::CUFFT_COPY_UNDEFINED;

// ========================================================================
// cufftXt query type enum
// Specifies the kind of query for cufftXtQueryPlan.
// ========================================================================
using ::cufftXtQueryType;
using ::cufftXtQueryType_t;

// cufftXtQueryType_t enumerators
using ::CUFFT_QUERY_1D_FACTORS;
using ::CUFFT_QUERY_UNDEFINED;

// ========================================================================
// cufftXt1dFactors struct
// Returned by cufftXtQueryPlan when querying CUFFT_QUERY_1D_FACTORS.
// ========================================================================
using ::cufftXt1dFactors;
using ::cufftXt1dFactors_t;

// ========================================================================
// cufftXt work area policy enum
// Controls the workspace allocation strategy for cufftXtSetWorkAreaPolicy.
// ========================================================================
using ::cufftXtWorkAreaPolicy;
using ::cufftXtWorkAreaPolicy_t;

// cufftXtWorkAreaPolicy_t enumerators
using ::CUFFT_WORKAREA_MINIMAL;
using ::CUFFT_WORKAREA_PERFORMANCE;
using ::CUFFT_WORKAREA_USER;

// ========================================================================
// cufftXt callback type enum
// Identifies which callback slot (load/store, single/double, real/complex)
// is being registered via cufftXtSetCallback.
// ========================================================================
using ::cufftXtCallbackType;
using ::cufftXtCallbackType_t;

// cufftXtCallbackType_t enumerators
using ::CUFFT_CB_LD_COMPLEX;
using ::CUFFT_CB_LD_COMPLEX_DOUBLE;
using ::CUFFT_CB_LD_REAL;
using ::CUFFT_CB_LD_REAL_DOUBLE;
using ::CUFFT_CB_ST_COMPLEX;
using ::CUFFT_CB_ST_COMPLEX_DOUBLE;
using ::CUFFT_CB_ST_REAL;
using ::CUFFT_CB_ST_REAL_DOUBLE;
using ::CUFFT_CB_UNDEFINED;

// ========================================================================
// Legacy callback function pointer typedefs
// Load callbacks return the element; store callbacks return void.
// ========================================================================

// Load callbacks (size_t offset)
using ::cufftCallbackLoadC;
using ::cufftCallbackLoadD;
using ::cufftCallbackLoadR;
using ::cufftCallbackLoadZ;

// Store callbacks (size_t offset)
using ::cufftCallbackStoreC;
using ::cufftCallbackStoreD;
using ::cufftCallbackStoreR;
using ::cufftCallbackStoreZ;

// ========================================================================
// LTO/JIT callback function pointer typedefs
// Same semantics as legacy callbacks but with unsigned long long offset.
// ========================================================================

// Load callbacks (unsigned long long offset)
using ::cufftJITCallbackLoadC;
using ::cufftJITCallbackLoadD;
using ::cufftJITCallbackLoadR;
using ::cufftJITCallbackLoadZ;

// Store callbacks (unsigned long long offset)
using ::cufftJITCallbackStoreC;
using ::cufftJITCallbackStoreD;
using ::cufftJITCallbackStoreR;
using ::cufftJITCallbackStoreZ;

// ========================================================================
// Multi-GPU Setup
// ========================================================================
using ::cufftXtSetGPUs;

// ========================================================================
// Multi-GPU Memory Management
// ========================================================================
using ::cufftXtFree;
using ::cufftXtMalloc;
using ::cufftXtMemcpy;
using ::cufftXtSetWorkArea;

// ========================================================================
// Multi-GPU Execution (descriptor-based, typed)
// ========================================================================
using ::cufftXtExecDescriptorC2C;
using ::cufftXtExecDescriptorC2R;
using ::cufftXtExecDescriptorD2Z;
using ::cufftXtExecDescriptorR2C;
using ::cufftXtExecDescriptorZ2D;
using ::cufftXtExecDescriptorZ2Z;

// ========================================================================
// Generic Execution
// ========================================================================
using ::cufftXtExec;
using ::cufftXtExecDescriptor;

// ========================================================================
// Extended Plan Creation and Size Query
// ========================================================================
using ::cufftXtGetSizeMany;
using ::cufftXtMakePlanMany;

// ========================================================================
// Work Area Policy
// ========================================================================
using ::cufftXtSetWorkAreaPolicy;

// ========================================================================
// Query
// ========================================================================
using ::cufftXtQueryPlan;

// ========================================================================
// Callback Registration
// ========================================================================
using ::cufftXtClearCallback;
using ::cufftXtSetCallback;
using ::cufftXtSetCallbackSharedSize;
using ::cufftXtSetJITCallback;

} // namespace wwr::cuda
