/**
 * @file hipfftXt.cppm
 * @brief hipFFT eXtended (hipfftXt) API module wrapper for wwr project
 *
 * Wraps hipfft/hipfftXt.h, which itself includes hipfft/hipfft.h and
 * hipfft/hiplibxt.h. CUDA counterpart: wwr.cuda.cufftXt.
 *
 * Two modules, not one: unlike the hip_runtime_api and hipsolver collapses,
 * hipFFT keeps the exact two-header split CUDA does -- hipfft.h / hipfftXt.h
 * mirror cufft.h / cufftXt.h one for one. So wwr.hip.hipfft and
 * wwr.hip.hipfftXt stay separate.
 *
 * hiplibxt.h's descriptor types are exported here alongside hipfftXt.h's own,
 * the way cufftXt.cppm exports cudalibxt.h's.
 *
 * The legacy per-precision callback typedefs are declared in hipfftXt.h itself
 * in this ROCm release and are exported. hipFFT has no separate JIT-callback
 * (unsigned-long-long-offset) typedef family; only the size_t-offset one
 * exists.
 *
 * Usage:
 *   import wwr.hip.hipfftXt;
 */

module;

// Pre-include <array> before the HIP header -- see src/hip/hip_complex.cppm's
// file header for why (amd_hip_vector_types.h, pulled in transitively via
// hip/hip_complex.h, #includes host_defines.h immediately before <array>,
// poisoning __has_attribute(__noinline__) for any later first-inclusion of
// <array> in the TU). Confirmed necessary here by direct experiment too.
// Load-bearing, and must stay before the HIP header: host_defines.h poisons
// __noinline__ for libc++'s __config. docs/architecture.md, section 9.
#include <array>
#include <hipfft/hipfftXt.h>

export module wwr.hip.hipfftXt;

import std;

export namespace wwr::hip {

// ========================================================================
// Multi-GPU descriptor types (from hiplibxt.h, used in hipfftXt signatures)
// ========================================================================

// Per-GPU memory descriptor (array-of-pointers per device)
using ::hipXtDesc;
using ::hipXtDesc_t;

// Library format tag stored inside hipLibXtDesc
using ::hiplibFormat;
using ::hiplibFormat_t;

using ::HIPLIB_FORMAT_HIPFFT;
using ::HIPLIB_FORMAT_UNDEFINED;

// Library-level descriptor wrapping hipXtDesc
using ::hipLibXtDesc;
using ::hipLibXtDesc_t;

// ========================================================================
// hipfftXt copy type enum
// Specifies the direction of copy for hipfftXtMemcpy.
// ========================================================================
using ::hipfftXtCopyType;
using ::hipfftXtCopyType_t;

using ::HIPFFT_COPY_DEVICE_TO_DEVICE;
using ::HIPFFT_COPY_DEVICE_TO_HOST;
using ::HIPFFT_COPY_HOST_TO_DEVICE;
using ::HIPFFT_COPY_UNDEFINED;

// ========================================================================
// hipfftXt callback type enum
// Identifies which callback slot (load/store, single/double, real/complex)
// is being registered via hipfftXtSetCallback.
// ========================================================================
using ::hipfftXtCallbackType;
using ::hipfftXtCallbackType_t;

using ::HIPFFT_CB_LD_COMPLEX;
using ::HIPFFT_CB_LD_COMPLEX_DOUBLE;
using ::HIPFFT_CB_LD_REAL;
using ::HIPFFT_CB_LD_REAL_DOUBLE;
using ::HIPFFT_CB_ST_COMPLEX;
using ::HIPFFT_CB_ST_COMPLEX_DOUBLE;
using ::HIPFFT_CB_ST_REAL;
using ::HIPFFT_CB_ST_REAL_DOUBLE;
using ::HIPFFT_CB_UNDEFINED;

// ========================================================================
// hipfftXt sub-format enum
// Describes the data layout of a hipLibXtDesc owned by hipFFT.
// ========================================================================
using ::hipfftXtSubFormat;
using ::hipfftXtSubFormat_t;

using ::HIPFFT_FORMAT_UNDEFINED;
using ::HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED;
using ::HIPFFT_XT_FORMAT_INPLACE;
using ::HIPFFT_XT_FORMAT_INPLACE_SHUFFLED;
using ::HIPFFT_XT_FORMAT_INPUT;
using ::HIPFFT_XT_FORMAT_OUTPUT;

// ========================================================================
// Callback function pointer typedefs
// Load callbacks return the element; store callbacks return void.
// ========================================================================
using ::hipfftCallbackLoadC;
using ::hipfftCallbackLoadD;
using ::hipfftCallbackLoadR;
using ::hipfftCallbackLoadZ;

using ::hipfftCallbackStoreC;
using ::hipfftCallbackStoreD;
using ::hipfftCallbackStoreR;
using ::hipfftCallbackStoreZ;

// ========================================================================
// Callback Registration
// ========================================================================
using ::hipfftXtClearCallback;
using ::hipfftXtSetCallback;
using ::hipfftXtSetCallbackSharedSize;

// ========================================================================
// Extended Plan Creation and Size Query
// ========================================================================
using ::hipfftXtGetSizeMany;
using ::hipfftXtMakePlanMany;

// ========================================================================
// Generic Execution
// ========================================================================
using ::hipfftXtExec;

// ========================================================================
// Multi-GPU Setup
// ========================================================================
using ::hipfftXtSetGPUs;

// ========================================================================
// Multi-GPU Memory Management
// ========================================================================
using ::hipfftXtFree;
using ::hipfftXtMalloc;
using ::hipfftXtMemcpy;

// ========================================================================
// Multi-GPU Execution (descriptor-based, typed)
// ========================================================================
using ::hipfftXtExecDescriptor;
using ::hipfftXtExecDescriptorC2C;
using ::hipfftXtExecDescriptorC2R;
using ::hipfftXtExecDescriptorD2Z;
using ::hipfftXtExecDescriptorR2C;
using ::hipfftXtExecDescriptorZ2D;
using ::hipfftXtExecDescriptorZ2Z;

} // namespace wwr::hip
