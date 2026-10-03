// rtc_host.cpp - the non-module HOST #include path for the rtc surface
//
// The counterpart to rtc.cppm's module test, through the OTHER path a consumer
// has -- #include "wwr/rtc.h" from a plain, non-module TU -- proving that path
// yields the identical surface bound to the identical backend entities. Like the
// module test it samples rather than restating every entry point: the full
// surface is pinned name-by-name in rtc.cppm's module test, and this shares
// detail/rtc_names.h with the module, so what is under test here is the include
// PATH.
//
// A plain .cpp that imports nothing and links wwr::rtc::host, the header-only
// target, which carries WWR_GPU_BACKEND_* via wwr::backend.

#include <type_traits>

#include "wwr/rtc.h"

#include "gpu_check_macros.h"

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

WWR_SAME_TYPE(wwrrtcProgram, nvrtcProgram)
WWR_SAME_TYPE(wwrrtcResult, nvrtcResult)

WWR_SAME_VALUE(WWRRTC_SUCCESS, NVRTC_SUCCESS)

WWR_SAME_FUNCTION(wwrrtcCompileProgram, nvrtcCompileProgram)
// The compiled-output getter, the one name whose CUDA and HIP spellings diverge
// (PTX on CUDA) -- so it is sampled here, not just a prefix swap.
WWR_SAME_FUNCTION(wwrrtcGetCode, nvrtcGetPTX)

#else

WWR_SAME_TYPE(wwrrtcProgram, hiprtcProgram)
WWR_SAME_TYPE(wwrrtcResult, hiprtcResult)

WWR_SAME_VALUE(WWRRTC_SUCCESS, HIPRTC_SUCCESS)

WWR_SAME_FUNCTION(wwrrtcCompileProgram, hiprtcCompileProgram)
// Code object on HIP.
WWR_SAME_FUNCTION(wwrrtcGetCode, hiprtcGetCode)

#endif

int main() { return 0; }
