/**
 * @file nvFatbin.cppm
 * @brief Primary interface for gpumod.cuda.nvFatbin
 *
 * This module wraps the nvFatbin (NVIDIA Fatbinary Creation) API and exports
 * types, enumerators, and functions for constructing CUDA fatbinary images
 * from PTX, cubin, LTOIR, and relocatable object inputs at runtime.
 *
 * Usage:
 *   import gpumod.cuda.nvFatbin;
 */

module;

#include <nvFatbin.h>

export module gpumod.cuda.nvFatbin;

import std;

export namespace gpumod::cuda {

// ========================================================================
// Result / Status Type
// ========================================================================
using ::nvFatbinResult;

// ========================================================================
// nvFatbinResult Enum Values
// ========================================================================
using ::NVFATBIN_ERROR_COMPRESSED_SIZE_EXCEEDED;
using ::NVFATBIN_ERROR_COMPRESSION_FAILED;
using ::NVFATBIN_ERROR_ELF_ARCH_MISMATCH;
using ::NVFATBIN_ERROR_ELF_SIZE_MISMATCH;
using ::NVFATBIN_ERROR_EMPTY_INPUT;
using ::NVFATBIN_ERROR_IDENTIFIER_REUSE;
using ::NVFATBIN_ERROR_INTERNAL;
using ::NVFATBIN_ERROR_INTERNAL_PTX_OPTION;
using ::NVFATBIN_ERROR_INVALID_ARCH;
using ::NVFATBIN_ERROR_INVALID_INDEX;
using ::NVFATBIN_ERROR_INVALID_NVVM;
using ::NVFATBIN_ERROR_MISSING_FATBIN;
using ::NVFATBIN_ERROR_MISSING_PTX_ARCH;
using ::NVFATBIN_ERROR_MISSING_PTX_VERSION;
using ::NVFATBIN_ERROR_NULL_POINTER;
using ::NVFATBIN_ERROR_PTX_ARCH_MISMATCH;
using ::NVFATBIN_ERROR_UNRECOGNIZED_OPTION;
using ::NVFATBIN_SUCCESS;

// ========================================================================
// Opaque Handle Type
// ========================================================================
using ::nvFatbinHandle;

// ========================================================================
// Error Handling Functions
// ========================================================================
using ::nvFatbinGetErrorString;

// ========================================================================
// Handle Lifecycle Functions
// ========================================================================
using ::nvFatbinCreate;
using ::nvFatbinDestroy;

// ========================================================================
// Content Addition Functions
// ========================================================================
using ::nvFatbinAddCubin;
using ::nvFatbinAddIndex;
using ::nvFatbinAddLTOIR;
using ::nvFatbinAddPTX;
using ::nvFatbinAddReloc;

// ========================================================================
// Fatbinary Retrieval Functions
// ========================================================================
using ::nvFatbinGet;
using ::nvFatbinSize;

// ========================================================================
// Version Query Functions
// ========================================================================
using ::nvFatbinVersion;

} // namespace gpumod::cuda
