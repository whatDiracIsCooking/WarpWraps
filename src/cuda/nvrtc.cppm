/**
 * @file nvrtc.cppm
 * @brief Primary interface for wwr.cuda.nvrtc
 *
 * This module wraps the NVRTC (NVIDIA Runtime Compilation) API and exports
 * types, constants, and functions for runtime compilation of CUDA source code
 * into PTX, CUBIN, or LTO IR at runtime.
 *
 * Usage:
 *   import wwr.cuda.nvrtc;
 */

module;

#include <nvrtc.h>

export module wwr.cuda.nvrtc;

import std;

export namespace wwr::cuda {

// ========================================================================
// Result / Status Type
// ========================================================================
using ::nvrtcResult;

// ========================================================================
// nvrtcResult Enum Values
// ========================================================================
using ::NVRTC_ERROR_BUILTIN_OPERATION_FAILURE;
using ::NVRTC_ERROR_CANCELLED;
using ::NVRTC_ERROR_COMPILATION;
using ::NVRTC_ERROR_INTERNAL_ERROR;
using ::NVRTC_ERROR_INVALID_INPUT;
using ::NVRTC_ERROR_INVALID_OPTION;
using ::NVRTC_ERROR_INVALID_PROGRAM;
using ::NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID;
using ::NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION;
using ::NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION;
using ::NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED;
using ::NVRTC_ERROR_OUT_OF_MEMORY;
using ::NVRTC_ERROR_PCH_CREATE;
using ::NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED;
using ::NVRTC_ERROR_PROGRAM_CREATION_FAILURE;
using ::NVRTC_ERROR_TIME_FILE_WRITE_FAILED;
using ::NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED;
using ::NVRTC_SUCCESS;

// ========================================================================
// Opaque Handle Type
// ========================================================================
using ::nvrtcProgram;

// ========================================================================
// Error Handling Functions
// ========================================================================
using ::nvrtcGetErrorString;

// ========================================================================
// General Information Query Functions
// ========================================================================
using ::nvrtcGetNumSupportedArchs;
using ::nvrtcGetSupportedArchs;
using ::nvrtcVersion;

// ========================================================================
// Program Lifecycle Functions
// ========================================================================
using ::nvrtcCreateProgram;
using ::nvrtcDestroyProgram;

// ========================================================================
// Compilation Functions
// ========================================================================
using ::nvrtcCompileProgram;

// ========================================================================
// PTX Retrieval Functions
// ========================================================================
using ::nvrtcGetPTX;
using ::nvrtcGetPTXSize;

// ========================================================================
// CUBIN Retrieval Functions
// ========================================================================
using ::nvrtcGetCUBIN;
using ::nvrtcGetCUBINSize;

// ========================================================================
// LTO IR Retrieval Functions
// ========================================================================
using ::nvrtcGetLTOIR;
using ::nvrtcGetLTOIRSize;

// ========================================================================
// OptiX IR Retrieval Functions
// ========================================================================
using ::nvrtcGetOptiXIR;
using ::nvrtcGetOptiXIRSize;

// ========================================================================
// Compilation Log Retrieval Functions
// ========================================================================
using ::nvrtcGetProgramLog;
using ::nvrtcGetProgramLogSize;

// ========================================================================
// Name Expression (Symbol Mangling) Functions
// ========================================================================
using ::nvrtcAddNameExpression;
using ::nvrtcGetLoweredName;

// ========================================================================
// Precompiled Header (PCH) Functions
// ========================================================================
using ::nvrtcGetPCHCreateStatus;
using ::nvrtcGetPCHHeapSize;
using ::nvrtcGetPCHHeapSizeRequired;
using ::nvrtcSetPCHHeapSize;

// ========================================================================
// Compilation Flow Control Functions
// ========================================================================
using ::nvrtcSetFlowCallback;

} // namespace wwr::cuda
