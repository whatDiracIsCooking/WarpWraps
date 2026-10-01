// rtc.cppm - Compile-time tests for wwr.rtc
//
// Every exported wwrrtc* name is checked against the backend's own entity: the
// same type, the same constant (type and value), the same function (see
// gpu_check_macros.h). The runtime-compilation surface is short enough to list
// in full, like test/gpu/fft.cppm.
//
// wwrrtcGetCode / wwrrtcGetCodeSize are the one spot that maps to a
// differently-named backend entity: NVRTC's nvrtcGetPTX / nvrtcGetPTXSize
// (PTX output) versus hipRTC's hiprtcGetCode / hiprtcGetCodeSize (code object).
// The per-backend restatement below pins each to the right one.

module;

#include "gpu_check_macros.h"

export module wwr.test.gpu.rtc;

import std;
import wwr.rtc;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.nvrtc;
#else
import wwr.hip.hiprtc;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(wwrrtcProgram, nvrtcProgram)
WWR_SAME_TYPE(wwrrtcResult, nvrtcResult)

WWR_SAME_VALUE(WWRRTC_SUCCESS, NVRTC_SUCCESS)
WWR_SAME_VALUE(WWRRTC_ERROR_OUT_OF_MEMORY, NVRTC_ERROR_OUT_OF_MEMORY)
WWR_SAME_VALUE(WWRRTC_ERROR_PROGRAM_CREATION_FAILURE, NVRTC_ERROR_PROGRAM_CREATION_FAILURE)
WWR_SAME_VALUE(WWRRTC_ERROR_INVALID_INPUT, NVRTC_ERROR_INVALID_INPUT)
WWR_SAME_VALUE(WWRRTC_ERROR_INVALID_PROGRAM, NVRTC_ERROR_INVALID_PROGRAM)
WWR_SAME_VALUE(WWRRTC_ERROR_INVALID_OPTION, NVRTC_ERROR_INVALID_OPTION)
WWR_SAME_VALUE(WWRRTC_ERROR_COMPILATION, NVRTC_ERROR_COMPILATION)
WWR_SAME_VALUE(WWRRTC_ERROR_BUILTIN_OPERATION_FAILURE, NVRTC_ERROR_BUILTIN_OPERATION_FAILURE)
WWR_SAME_VALUE(WWRRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
               NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION)
WWR_SAME_VALUE(WWRRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
               NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION)
WWR_SAME_VALUE(WWRRTC_ERROR_NAME_EXPRESSION_NOT_VALID, NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID)
WWR_SAME_VALUE(WWRRTC_ERROR_INTERNAL_ERROR, NVRTC_ERROR_INTERNAL_ERROR)

WWR_SAME_FUNCTION(wwrrtcVersion, nvrtcVersion)
WWR_SAME_FUNCTION(wwrrtcGetErrorString, nvrtcGetErrorString)

WWR_SAME_FUNCTION(wwrrtcCreateProgram, nvrtcCreateProgram)
WWR_SAME_FUNCTION(wwrrtcDestroyProgram, nvrtcDestroyProgram)

WWR_SAME_FUNCTION(wwrrtcCompileProgram, nvrtcCompileProgram)

// PTX output on CUDA.
WWR_SAME_FUNCTION(wwrrtcGetCode, nvrtcGetPTX)
WWR_SAME_FUNCTION(wwrrtcGetCodeSize, nvrtcGetPTXSize)

WWR_SAME_FUNCTION(wwrrtcGetProgramLog, nvrtcGetProgramLog)
WWR_SAME_FUNCTION(wwrrtcGetProgramLogSize, nvrtcGetProgramLogSize)

WWR_SAME_FUNCTION(wwrrtcAddNameExpression, nvrtcAddNameExpression)
WWR_SAME_FUNCTION(wwrrtcGetLoweredName, nvrtcGetLoweredName)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace wwr::hip;

WWR_SAME_TYPE(wwrrtcProgram, hiprtcProgram)
WWR_SAME_TYPE(wwrrtcResult, hiprtcResult)

WWR_SAME_VALUE(WWRRTC_SUCCESS, HIPRTC_SUCCESS)
WWR_SAME_VALUE(WWRRTC_ERROR_OUT_OF_MEMORY, HIPRTC_ERROR_OUT_OF_MEMORY)
WWR_SAME_VALUE(WWRRTC_ERROR_PROGRAM_CREATION_FAILURE, HIPRTC_ERROR_PROGRAM_CREATION_FAILURE)
WWR_SAME_VALUE(WWRRTC_ERROR_INVALID_INPUT, HIPRTC_ERROR_INVALID_INPUT)
WWR_SAME_VALUE(WWRRTC_ERROR_INVALID_PROGRAM, HIPRTC_ERROR_INVALID_PROGRAM)
WWR_SAME_VALUE(WWRRTC_ERROR_INVALID_OPTION, HIPRTC_ERROR_INVALID_OPTION)
WWR_SAME_VALUE(WWRRTC_ERROR_COMPILATION, HIPRTC_ERROR_COMPILATION)
WWR_SAME_VALUE(WWRRTC_ERROR_BUILTIN_OPERATION_FAILURE, HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE)
WWR_SAME_VALUE(WWRRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
               HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION)
WWR_SAME_VALUE(WWRRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
               HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION)
WWR_SAME_VALUE(WWRRTC_ERROR_NAME_EXPRESSION_NOT_VALID, HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID)
WWR_SAME_VALUE(WWRRTC_ERROR_INTERNAL_ERROR, HIPRTC_ERROR_INTERNAL_ERROR)

WWR_SAME_FUNCTION(wwrrtcVersion, hiprtcVersion)
WWR_SAME_FUNCTION(wwrrtcGetErrorString, hiprtcGetErrorString)

WWR_SAME_FUNCTION(wwrrtcCreateProgram, hiprtcCreateProgram)
WWR_SAME_FUNCTION(wwrrtcDestroyProgram, hiprtcDestroyProgram)

WWR_SAME_FUNCTION(wwrrtcCompileProgram, hiprtcCompileProgram)

// Code object on HIP.
WWR_SAME_FUNCTION(wwrrtcGetCode, hiprtcGetCode)
WWR_SAME_FUNCTION(wwrrtcGetCodeSize, hiprtcGetCodeSize)

WWR_SAME_FUNCTION(wwrrtcGetProgramLog, hiprtcGetProgramLog)
WWR_SAME_FUNCTION(wwrrtcGetProgramLogSize, hiprtcGetProgramLogSize)

WWR_SAME_FUNCTION(wwrrtcAddNameExpression, hiprtcAddNameExpression)
WWR_SAME_FUNCTION(wwrrtcGetLoweredName, hiprtcGetLoweredName)

#endif

} // namespace wwr::test
