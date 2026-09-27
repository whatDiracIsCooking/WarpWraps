/**
 * @file nvJitLink.cppm
 * @brief Primary interface for wwr.cuda.nvJitLink
 *
 * This module wraps the nvJitLink (NVIDIA JIT Linking) API and exports
 * types, constants, and functions for JIT linking of CUDA device code
 * (CUBIN, PTX, LTO-IR, fatbin, and object files) at runtime.
 *
 * Usage:
 *   import wwr.cuda.nvJitLink;
 */

module;

#include <nvJitLink.h>

export module wwr.cuda.nvJitLink;

import std;

export namespace wwr::cuda {

// ========================================================================
// Result / Status Type
// ========================================================================
using ::nvJitLinkResult;

// ========================================================================
// nvJitLinkResult Enum Values
// ========================================================================
using ::NVJITLINK_ERROR_ARCH_MISMATCH;
using ::NVJITLINK_ERROR_FINALIZE;
using ::NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS;
using ::NVJITLINK_ERROR_INCORRECT_INPUT_TYPE;
using ::NVJITLINK_ERROR_INTERNAL;
using ::NVJITLINK_ERROR_INVALID_INPUT;
using ::NVJITLINK_ERROR_LTO_NOT_ENABLED;
using ::NVJITLINK_ERROR_MISSING_ARCH;
using ::NVJITLINK_ERROR_MISSING_FATBIN;
using ::NVJITLINK_ERROR_NULL_INPUT;
using ::NVJITLINK_ERROR_NVVM_COMPILE;
using ::NVJITLINK_ERROR_OUTDATED_LIBRARY;
using ::NVJITLINK_ERROR_PTX_COMPILE;
using ::NVJITLINK_ERROR_THREADPOOL;
using ::NVJITLINK_ERROR_UNRECOGNIZED_ARCH;
using ::NVJITLINK_ERROR_UNRECOGNIZED_INPUT;
using ::NVJITLINK_ERROR_UNRECOGNIZED_OPTION;
using ::NVJITLINK_ERROR_UNSUPPORTED_ARCH;
using ::NVJITLINK_SUCCESS;

// ========================================================================
// Input Type Enum
// ========================================================================
using ::nvJitLinkInputType;

// ========================================================================
// nvJitLinkInputType Enum Values
// ========================================================================
using ::NVJITLINK_INPUT_ANY;
using ::NVJITLINK_INPUT_CUBIN;
using ::NVJITLINK_INPUT_FATBIN;
using ::NVJITLINK_INPUT_INDEX;
using ::NVJITLINK_INPUT_LIBRARY;
using ::NVJITLINK_INPUT_LTOIR;
using ::NVJITLINK_INPUT_NONE;
using ::NVJITLINK_INPUT_OBJECT;
using ::NVJITLINK_INPUT_PTX;

// ========================================================================
// Opaque Handle Type
// ========================================================================
using ::nvJitLinkHandle;

// ========================================================================
// Handle Lifecycle Functions
// (nvJitLink API functions are static inline wrappers in the header;
//  they cannot be exported via using-declarations, so we provide
//  exported inline wrappers that forward to the header functions)
// ========================================================================
inline nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle *handle, uint32_t numOptions,
                                       const char **options) {
  return ::nvJitLinkCreate(handle, numOptions, options);
}
inline nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle *handle) {
  return ::nvJitLinkDestroy(handle);
}

// ========================================================================
// Input Addition Functions
// ========================================================================
inline nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle, nvJitLinkInputType inputType,
                                        const void *data, size_t size, const char *name) {
  return ::nvJitLinkAddData(handle, inputType, data, size, name);
}
inline nvJitLinkResult nvJitLinkAddFile(nvJitLinkHandle handle, nvJitLinkInputType inputType,
                                        const char *fileName) {
  return ::nvJitLinkAddFile(handle, inputType, fileName);
}

// ========================================================================
// Linking Functions
// ========================================================================
inline nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle handle) {
  return ::nvJitLinkComplete(handle);
}

// ========================================================================
// Linked Output Retrieval Functions (CUBIN)
// ========================================================================
inline nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, size_t *size) {
  return ::nvJitLinkGetLinkedCubinSize(handle, size);
}
inline nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void *cubin) {
  return ::nvJitLinkGetLinkedCubin(handle, cubin);
}

// ========================================================================
// Linked Output Retrieval Functions (PTX)
// ========================================================================
inline nvJitLinkResult nvJitLinkGetLinkedPtxSize(nvJitLinkHandle handle, size_t *size) {
  return ::nvJitLinkGetLinkedPtxSize(handle, size);
}
inline nvJitLinkResult nvJitLinkGetLinkedPtx(nvJitLinkHandle handle, char *ptx) {
  return ::nvJitLinkGetLinkedPtx(handle, ptx);
}

// ========================================================================
// Log Retrieval Functions (Error Log)
// ========================================================================
inline nvJitLinkResult nvJitLinkGetErrorLogSize(nvJitLinkHandle handle, size_t *size) {
  return ::nvJitLinkGetErrorLogSize(handle, size);
}
inline nvJitLinkResult nvJitLinkGetErrorLog(nvJitLinkHandle handle, char *log) {
  return ::nvJitLinkGetErrorLog(handle, log);
}

// ========================================================================
// Log Retrieval Functions (Info Log)
// ========================================================================
inline nvJitLinkResult nvJitLinkGetInfoLogSize(nvJitLinkHandle handle, size_t *size) {
  return ::nvJitLinkGetInfoLogSize(handle, size);
}
inline nvJitLinkResult nvJitLinkGetInfoLog(nvJitLinkHandle handle, char *log) {
  return ::nvJitLinkGetInfoLog(handle, log);
}

// ========================================================================
// Version Query Function
// ========================================================================
using ::nvJitLinkVersion;

} // namespace wwr::cuda
