/**
 * @file rtc.cppm
 * @brief Backend-neutral runtime compilation: wwrrtc* names for NVRTC / hipRTC
 *
 * wwrrtc<name> stands for nvrtc<name> on a CUDA build and hiprtc<name> on a HIP
 * build. Only the surface the two libraries share by name is wrapped here -- the
 * measured intersection (devtools/header_intersection.py --cuda nvrtc.json --hip
 * hiprtc.json): the program lifecycle, compilation, the log and name-expression
 * queries, the result enum, and the version/error-string helpers. See backend.h.
 *
 * Backend difference resolved here, not above:
 *
 * - The compiled-output getter: NVRTC emits PTX (nvrtcGetPTX / nvrtcGetPTXSize),
 *   hipRTC emits a code object (hiprtcGetCode / hiprtcGetCodeSize). One neutral
 *   spelling, wwrrtcGetCode / wwrrtcGetCodeSize, maps to each backend's own.
 *
 * Single-backend names are reached through the raw module, not aliased here:
 * NVRTC's CUBIN / LTO IR / OptiX IR getters, its PCH and flow-callback surface
 * and its supported-arch query; hipRTC's linking surface (hiprtcLink*, with the
 * hipJitOption / hipJitInputType enums the hiprtcJIT_option / hiprtcJITInputType
 * macro aliases expand to) and its bitcode getters. The result codes likewise:
 * only the 12 both enums share get a WWRRTC_* alias -- an NVRTC-only code
 * (NVRTC_ERROR_CANCELLED, the PCH/time-trace codes, ...) or an hipRTC-only one
 * (HIPRTC_ERROR_LINKING) is reached through the raw module.
 *
 * Usage:
 *   import wwr.rtc;
 *
 *   wwrrtcProgram prog;
 *   wwrrtcCreateProgram(&prog, src, "k.cu", 0, nullptr, nullptr);
 */

module;

#include "backend.h"

export module wwr.rtc;

#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.nvrtc;
#else
import wwr.hip.hiprtc;
#endif

export namespace wwr {

// ========================================================================
// Types
// ========================================================================

WWR_TYPE(wwrrtcProgram, nvrtcProgram, hiprtcProgram)
WWR_TYPE(wwrrtcResult, nvrtcResult, hiprtcResult)

// ========================================================================
// Result codes -- the 12 shared by both enums. NVRTC-only codes (CANCELLED,
// the PCH and time-trace codes) and the hipRTC-only LINKING code are reached
// through the raw module.
// ========================================================================

WWR_VALUE(WWRRTC_SUCCESS, NVRTC_SUCCESS, HIPRTC_SUCCESS)
WWR_VALUE(WWRRTC_ERROR_OUT_OF_MEMORY, NVRTC_ERROR_OUT_OF_MEMORY, HIPRTC_ERROR_OUT_OF_MEMORY)
WWR_VALUE(WWRRTC_ERROR_PROGRAM_CREATION_FAILURE, NVRTC_ERROR_PROGRAM_CREATION_FAILURE,
          HIPRTC_ERROR_PROGRAM_CREATION_FAILURE)
WWR_VALUE(WWRRTC_ERROR_INVALID_INPUT, NVRTC_ERROR_INVALID_INPUT, HIPRTC_ERROR_INVALID_INPUT)
WWR_VALUE(WWRRTC_ERROR_INVALID_PROGRAM, NVRTC_ERROR_INVALID_PROGRAM, HIPRTC_ERROR_INVALID_PROGRAM)
WWR_VALUE(WWRRTC_ERROR_INVALID_OPTION, NVRTC_ERROR_INVALID_OPTION, HIPRTC_ERROR_INVALID_OPTION)
WWR_VALUE(WWRRTC_ERROR_COMPILATION, NVRTC_ERROR_COMPILATION, HIPRTC_ERROR_COMPILATION)
WWR_VALUE(WWRRTC_ERROR_BUILTIN_OPERATION_FAILURE, NVRTC_ERROR_BUILTIN_OPERATION_FAILURE,
          HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE)
WWR_VALUE(WWRRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
          NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
          HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION)
WWR_VALUE(WWRRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
          NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
          HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION)
WWR_VALUE(WWRRTC_ERROR_NAME_EXPRESSION_NOT_VALID, NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID,
          HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID)
WWR_VALUE(WWRRTC_ERROR_INTERNAL_ERROR, NVRTC_ERROR_INTERNAL_ERROR, HIPRTC_ERROR_INTERNAL_ERROR)

// ========================================================================
// Version / error string
// ========================================================================

WWR_FUNCTION(wwrrtcVersion, nvrtcVersion, hiprtcVersion)
WWR_FUNCTION(wwrrtcGetErrorString, nvrtcGetErrorString, hiprtcGetErrorString)

// ========================================================================
// Program lifecycle
// ========================================================================

WWR_FUNCTION(wwrrtcCreateProgram, nvrtcCreateProgram, hiprtcCreateProgram)
WWR_FUNCTION(wwrrtcDestroyProgram, nvrtcDestroyProgram, hiprtcDestroyProgram)

// ========================================================================
// Compilation
// ========================================================================

WWR_FUNCTION(wwrrtcCompileProgram, nvrtcCompileProgram, hiprtcCompileProgram)

// ========================================================================
// Compiled-output retrieval -- NVRTC emits PTX, hipRTC a code object; one
// neutral spelling maps to each backend's own getter (see file header).
// ========================================================================

WWR_FUNCTION(wwrrtcGetCode, nvrtcGetPTX, hiprtcGetCode)
WWR_FUNCTION(wwrrtcGetCodeSize, nvrtcGetPTXSize, hiprtcGetCodeSize)

// ========================================================================
// Compilation log retrieval
// ========================================================================

WWR_FUNCTION(wwrrtcGetProgramLog, nvrtcGetProgramLog, hiprtcGetProgramLog)
WWR_FUNCTION(wwrrtcGetProgramLogSize, nvrtcGetProgramLogSize, hiprtcGetProgramLogSize)

// ========================================================================
// Name expression (symbol mangling)
// ========================================================================

WWR_FUNCTION(wwrrtcAddNameExpression, nvrtcAddNameExpression, hiprtcAddNameExpression)
WWR_FUNCTION(wwrrtcGetLoweredName, nvrtcGetLoweredName, hiprtcGetLoweredName)

} // namespace wwr
