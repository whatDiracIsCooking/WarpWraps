"""The `rtc` (NVRTC / hipRTC) surface as a FRAGMENT-SOURCED spec.

The runtime-compilation library, the smallest of the four milestone-#6 surfaces
(detail/rtc_names.h: two types, the 12 shared result codes, 11 functions). It is
HOST-only -- unlike blas/blaslt/solver/rand it executes NOTHING on the device, so
its thunks are fully exercisable on a no-GPU runner -- but it shares their SHAPE
exactly: a status enum returned by value, one opaque handle, no by-value enums or
PODs. So it registers through the same captypes machinery blas uses (see
spec/blas.py) and the generic fragment-sourced emit path classifies it.

  * HANDLE   wwrrtcProgram -> void* (wwr_rtc_program)
  * VALUE ENUM  wwrrtcResult (the 12 shared result codes) -- a stable capi enum
    the thunk maps capi<->neutral at the boundary; returned by every function
    except GetErrorString (which takes it BY VALUE and returns const char *)

No complex, no half, no stream, no by-value attribute enums. See spec/blas.py for
the ValueEnum / Handle rationale and docs/architecture.md section 25.
"""

from __future__ import annotations

from captypes import Handle, ValueEnum
from library import LibrarySpec, snake_capi_name

FRAGMENT = "detail/rtc_names.h"
PREFIX = "wwrrtc"
CAPI_PREFIX = "wwr_rtc_"

CUDA_MANIFEST = "cuda-13.0.x/nvrtc.json"
HIP_MANIFEST = "rocm-10.0.0/hiprtc.json"
MODULES = ["wwr.rtc"]

HANDLES = [
    Handle(vendor_types=("nvrtcProgram", "hiprtcProgram"),
           capi_type="wwr_rtc_program"),
]

POINTEE_PODS: list = []
PODS: list = []

# The result enum (wwrrtcResult): returned by every function, and taken by value
# by GetErrorString. Each pair is (capi_enumerator, neutral wwr constant from the
# fragment); the stable capi values are fixed §6 integers.
RESULT = ValueEnum(
    vendor_types=("nvrtcResult", "hiprtcResult"),
    capi_type="wwr_rtc_result",
    wwr_type="wwrrtcResult",
    fallback="WWR_RTC_RESULT_INTERNAL_ERROR",
    entries=(
        ("WWR_RTC_RESULT_SUCCESS", "WWRRTC_SUCCESS"),
        ("WWR_RTC_RESULT_OUT_OF_MEMORY", "WWRRTC_ERROR_OUT_OF_MEMORY"),
        ("WWR_RTC_RESULT_PROGRAM_CREATION_FAILURE",
         "WWRRTC_ERROR_PROGRAM_CREATION_FAILURE"),
        ("WWR_RTC_RESULT_INVALID_INPUT", "WWRRTC_ERROR_INVALID_INPUT"),
        ("WWR_RTC_RESULT_INVALID_PROGRAM", "WWRRTC_ERROR_INVALID_PROGRAM"),
        ("WWR_RTC_RESULT_INVALID_OPTION", "WWRRTC_ERROR_INVALID_OPTION"),
        ("WWR_RTC_RESULT_COMPILATION", "WWRRTC_ERROR_COMPILATION"),
        ("WWR_RTC_RESULT_BUILTIN_OPERATION_FAILURE",
         "WWRRTC_ERROR_BUILTIN_OPERATION_FAILURE"),
        ("WWR_RTC_RESULT_NO_NAME_EXPRESSIONS_AFTER_COMPILATION",
         "WWRRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION"),
        ("WWR_RTC_RESULT_NO_LOWERED_NAMES_BEFORE_COMPILATION",
         "WWRRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION"),
        ("WWR_RTC_RESULT_NAME_EXPRESSION_NOT_VALID",
         "WWRRTC_ERROR_NAME_EXPRESSION_NOT_VALID"),
        ("WWR_RTC_RESULT_INTERNAL_ERROR", "WWRRTC_ERROR_INTERNAL_ERROR"),
    ),
)

VALUE_ENUMS = [RESULT]

# Every result constant is a fragment VALUE entry present in both manifests, so
# none is exempt from the manifest value cross-check.
ENUM_CONSTANTS_NOT_IN_MANIFEST: set[str] = set()

# No vendor scalar typedefs to normalise.
SCALAR_ALIASES: dict[str, str] = {}

# No hand-written fragment forwarders are skipped: every rtc function is a plain
# WWR_FUNCTION_RAW macro entry, all 11 wrapped.
EXPLICIT_SKIPS: list[tuple[str, str]] = []


def capi_name(wwr_name: str) -> str:
    return snake_capi_name(wwr_name, PREFIX, CAPI_PREFIX)


SPEC = LibrarySpec(
    name="rtc",
    fragment=FRAGMENT,
    capi_prefix=CAPI_PREFIX,
    cuda_manifest=CUDA_MANIFEST,
    hip_manifest=HIP_MANIFEST,
    modules=MODULES,
    handles=HANDLES,
    pointee_pods=POINTEE_PODS,
    pods=PODS,
    value_enums=VALUE_ENUMS,
    enum_constants_not_in_manifest=ENUM_CONSTANTS_NOT_IN_MANIFEST,
    scalar_aliases=SCALAR_ALIASES,
    explicit_skips=EXPLICIT_SKIPS,
    capi_name=capi_name,
)
