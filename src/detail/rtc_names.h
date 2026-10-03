/**
 * @file detail/rtc_names.h
 * @brief The backend-neutral NVRTC / hipRTC surface, as a macro-driven include
 *        fragment shared by the module and the non-module #include path
 *
 * NOT a standalone header: it is the list of wwrrtc* / WWRRTC_* names (the two
 * types, the 12 shared result codes and the functions) with NO namespace of its
 * own and NO vendor #include. The includer supplies all of that and pastes this
 * inside its own `namespace wwr` -- so one list binds both ways the surface is
 * consumed: wwr.rtc (the module, `export namespace wwr`) and wwr/rtc.h (the
 * non-module #include path). Add a name here, once, and both paths gain it.
 *
 * HOST only -- NVRTC / hipRTC is a host API. The surface binds straight to the
 * vendor's external-linkage `::nvrtc*` / `::hiprtc*` declarations via the _RAW
 * macros (the "rand.h shape"): no raw vendor module is imported, so the same
 * `::`-prefixed names resolve in the module (vendor header in its GMF) and the
 * #include path alike. The single-backend surface (NVRTC's CUBIN / LTO IR /
 * OptiX IR getters and its PCH / flow-callback / supported-arch extras; hipRTC's
 * hiprtcLink* linking surface and its bitcode getters) is deliberately absent;
 * reach it through wwr.cuda.nvrtc / wwr.hip.hiprtc.
 *
 * Before including, the includer must have, in order:
 *   - the vendor header in scope (nvrtc.h / hip/hiprtc.h), which rtc.h pulls in;
 *   - WWR_SELECT_RAW(cuda, hip) plus WWR_TYPE_RAW / WWR_VALUE_RAW /
 *     WWR_FUNCTION_RAW on top of it -- keyed on WWR_GPU_BACKEND_* in the module
 *     (backend.h) or WWR_SELECTED_* in the #include path (wwr/rtc.h).
 *
 * Every line is a plain _RAW binding -- there is no hand-written body here. The
 * compiled-output getter's name divergence (nvrtcGetPTX vs hiprtcGetCode) is a
 * single WWR_FUNCTION_RAW line, exactly as a prefix swap would be. See
 * src/rtc.cppm, src/rtc.h, src/wwr/rtc.h and docs/architecture.md section 5.
 */

#pragma once

#ifndef WWR_FUNCTION_RAW
#error                                                                                             \
    "detail/rtc_names.h is an include fragment, not a standalone header: define WWR_TYPE_RAW/VALUE_RAW/FUNCTION_RAW and WWR_SELECT_RAW, ensure the vendor header (via rtc.h) is in scope, and #include it inside namespace wwr. See src/rtc.h, src/rtc.cppm and src/wwr/rtc.h."
#endif

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables): each wwrrtc*
// function below is a deliberate constexpr reference to the selected backend's
// entry point (via WWR_FUNCTION_RAW). A reference to a vendor function has no
// const form, so the check cannot be satisfied without abandoning the alias
// pattern -- see backend.h and detail/blas_names.h.

// ========================================================================
// Types
// ========================================================================

WWR_TYPE_RAW(wwrrtcProgram, nvrtcProgram, hiprtcProgram)
WWR_TYPE_RAW(wwrrtcResult, nvrtcResult, hiprtcResult)

// ========================================================================
// Result codes -- the 12 shared by both enums. NVRTC-only codes (CANCELLED,
// the PCH and time-trace codes) and the hipRTC-only LINKING code are reached
// through the raw module.
// ========================================================================

WWR_VALUE_RAW(WWRRTC_SUCCESS, NVRTC_SUCCESS, HIPRTC_SUCCESS)
WWR_VALUE_RAW(WWRRTC_ERROR_OUT_OF_MEMORY, NVRTC_ERROR_OUT_OF_MEMORY, HIPRTC_ERROR_OUT_OF_MEMORY)
WWR_VALUE_RAW(WWRRTC_ERROR_PROGRAM_CREATION_FAILURE, NVRTC_ERROR_PROGRAM_CREATION_FAILURE,
              HIPRTC_ERROR_PROGRAM_CREATION_FAILURE)
WWR_VALUE_RAW(WWRRTC_ERROR_INVALID_INPUT, NVRTC_ERROR_INVALID_INPUT, HIPRTC_ERROR_INVALID_INPUT)
WWR_VALUE_RAW(WWRRTC_ERROR_INVALID_PROGRAM, NVRTC_ERROR_INVALID_PROGRAM,
              HIPRTC_ERROR_INVALID_PROGRAM)
WWR_VALUE_RAW(WWRRTC_ERROR_INVALID_OPTION, NVRTC_ERROR_INVALID_OPTION, HIPRTC_ERROR_INVALID_OPTION)
WWR_VALUE_RAW(WWRRTC_ERROR_COMPILATION, NVRTC_ERROR_COMPILATION, HIPRTC_ERROR_COMPILATION)
WWR_VALUE_RAW(WWRRTC_ERROR_BUILTIN_OPERATION_FAILURE, NVRTC_ERROR_BUILTIN_OPERATION_FAILURE,
              HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE)
WWR_VALUE_RAW(WWRRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
              NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
              HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION)
WWR_VALUE_RAW(WWRRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
              NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
              HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION)
WWR_VALUE_RAW(WWRRTC_ERROR_NAME_EXPRESSION_NOT_VALID, NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID,
              HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID)
WWR_VALUE_RAW(WWRRTC_ERROR_INTERNAL_ERROR, NVRTC_ERROR_INTERNAL_ERROR, HIPRTC_ERROR_INTERNAL_ERROR)

// ========================================================================
// Version / error string
// ========================================================================

WWR_FUNCTION_RAW(wwrrtcVersion, nvrtcVersion, hiprtcVersion)
WWR_FUNCTION_RAW(wwrrtcGetErrorString, nvrtcGetErrorString, hiprtcGetErrorString)

// ========================================================================
// Program lifecycle
// ========================================================================

WWR_FUNCTION_RAW(wwrrtcCreateProgram, nvrtcCreateProgram, hiprtcCreateProgram)
WWR_FUNCTION_RAW(wwrrtcDestroyProgram, nvrtcDestroyProgram, hiprtcDestroyProgram)

// ========================================================================
// Compilation
// ========================================================================

WWR_FUNCTION_RAW(wwrrtcCompileProgram, nvrtcCompileProgram, hiprtcCompileProgram)

// ========================================================================
// Compiled-output retrieval -- NVRTC emits PTX, hipRTC a code object; one
// neutral spelling maps to each backend's own getter (see rtc.cppm's header).
// ========================================================================

WWR_FUNCTION_RAW(wwrrtcGetCode, nvrtcGetPTX, hiprtcGetCode)
WWR_FUNCTION_RAW(wwrrtcGetCodeSize, nvrtcGetPTXSize, hiprtcGetCodeSize)

// ========================================================================
// Compilation log retrieval
// ========================================================================

WWR_FUNCTION_RAW(wwrrtcGetProgramLog, nvrtcGetProgramLog, hiprtcGetProgramLog)
WWR_FUNCTION_RAW(wwrrtcGetProgramLogSize, nvrtcGetProgramLogSize, hiprtcGetProgramLogSize)

// ========================================================================
// Name expression (symbol mangling)
// ========================================================================

WWR_FUNCTION_RAW(wwrrtcAddNameExpression, nvrtcAddNameExpression, hiprtcAddNameExpression)
WWR_FUNCTION_RAW(wwrrtcGetLoweredName, nvrtcGetLoweredName, hiprtcGetLoweredName)

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
