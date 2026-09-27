/**
 * @file hiprtc.cppm
 * @brief HIP Runtime Compilation (hiprtc) API module wrapper
 *
 * Wraps hip/hiprtc.h for C++23 module-based code: types, constants and
 * functions for compiling HIP source into code objects at runtime. CUDA
 * counterpart: wwr.cuda.nvrtc. Like nvrtc.h it is declarations-only, so
 * this is a pure re-export via `using` declarations.
 *
 * This module links hiprtc::hiprtc, from ROCm's own hiprtc find_package
 * config, separate from the hip package hip::host comes from -- see
 * src/hip/CMakeLists.txt and src/hip/README.md.
 *
 * hiprtcJIT_option / hiprtcJITInputType are preprocessor aliases for
 * hipJitOption / hipJitInputType, which come from hip/linker_types.h rather
 * than hip_runtime_api.h. Both real enum types are exported here, because the
 * hiprtcLink* signatures take hipJitOption* after macro expansion and callers
 * need the real name.
 *
 * Usage:
 *   import wwr.hip.hiprtc;
 */

module;

#include <hip/hiprtc.h>

export module wwr.hip.hiprtc;

export namespace wwr::hip {

// ========================================================================
// Result / Status Type
// ========================================================================
using ::hiprtcResult;

// ========================================================================
// hiprtcResult Enum Values
// ========================================================================
using ::HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE;
using ::HIPRTC_ERROR_COMPILATION;
using ::HIPRTC_ERROR_INTERNAL_ERROR;
using ::HIPRTC_ERROR_INVALID_INPUT;
using ::HIPRTC_ERROR_INVALID_OPTION;
using ::HIPRTC_ERROR_INVALID_PROGRAM;
using ::HIPRTC_ERROR_LINKING;
using ::HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID;
using ::HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION;
using ::HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION;
using ::HIPRTC_ERROR_OUT_OF_MEMORY;
using ::HIPRTC_ERROR_PROGRAM_CREATION_FAILURE;
using ::HIPRTC_SUCCESS;

// ========================================================================
// Opaque Handle Types
// ========================================================================
using ::hiprtcLinkState;
using ::hiprtcProgram;

// ========================================================================
// JIT Option / Input Type (hip/linker_types.h, included by hiprtc.h;
// hiprtcJIT_option / hiprtcJITInputType are macro aliases for these -- see
// file header note above)
// ========================================================================
using ::hipJitInputType;
using ::hipJitOption;

// ========================================================================
// Error Handling Functions
// ========================================================================
using ::hiprtcGetErrorString;

// ========================================================================
// General Information Query Functions
// ========================================================================
using ::hiprtcVersion;

// ========================================================================
// Program Lifecycle Functions
// ========================================================================
using ::hiprtcCreateProgram;
using ::hiprtcDestroyProgram;

// ========================================================================
// Compilation Functions
// ========================================================================
using ::hiprtcCompileProgram;

// ========================================================================
// Code / Bitcode Retrieval Functions
// ========================================================================
using ::hiprtcGetBitcode;
using ::hiprtcGetBitcodeSize;
using ::hiprtcGetCode;
using ::hiprtcGetCodeSize;

// ========================================================================
// Compilation Log Retrieval Functions
// ========================================================================
using ::hiprtcGetProgramLog;
using ::hiprtcGetProgramLogSize;

// ========================================================================
// Name Expression (Symbol Mangling) Functions
// ========================================================================
using ::hiprtcAddNameExpression;
using ::hiprtcGetLoweredName;

// ========================================================================
// Linking Functions
// ========================================================================
using ::hiprtcLinkAddData;
using ::hiprtcLinkAddFile;
using ::hiprtcLinkComplete;
using ::hiprtcLinkCreate;
using ::hiprtcLinkDestroy;

} // namespace wwr::hip
