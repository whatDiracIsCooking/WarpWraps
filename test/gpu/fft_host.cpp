// fft_host.cpp - the non-module HOST #include path for the fft surface
//
// The counterpart to fft.cppm's module test, through the OTHER path a consumer
// has -- #include "wwr/fft.h" from a plain, non-module TU -- proving that path
// yields the identical surface bound to the identical backend entities. Like the
// module test it samples rather than restating every entry point: the full
// surface is pinned name-by-name by the dispatch check and the module test, and
// this shares detail/fft_names.h with the module, so what is under test here is
// the include PATH.
//
// A plain .cpp that imports nothing and links wwr::fft::host, the header-only
// target, which carries WWR_GPU_BACKEND_* via wwr::backend.

#include <type_traits>

#include "wwr/fft.h"

#include "gpu_check_macros.h"

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

WWR_SAME_TYPE(wwrfftHandle, cufftHandle)
WWR_SAME_TYPE(wwrfftResult_t, cufftResult_t)

WWR_SAME_VALUE(WWRFFT_SUCCESS, CUFFT_SUCCESS)
// WWRFFT_INVERSE maps to a differently-named constant per backend; pin it here.
WWR_SAME_VALUE(WWRFFT_INVERSE, CUFFT_INVERSE)

WWR_SAME_FUNCTION(wwrfftCreate, cufftCreate)
WWR_SAME_FUNCTION(wwrfftExecC2C, cufftExecC2C)

#else

WWR_SAME_TYPE(wwrfftHandle, hipfftHandle)
WWR_SAME_TYPE(wwrfftResult_t, hipfftResult_t)

WWR_SAME_VALUE(WWRFFT_SUCCESS, HIPFFT_SUCCESS)
// hipFFT names the inverse transform BACKWARD, not INVERSE.
WWR_SAME_VALUE(WWRFFT_INVERSE, HIPFFT_BACKWARD)

WWR_SAME_FUNCTION(wwrfftCreate, hipfftCreate)
WWR_SAME_FUNCTION(wwrfftExecC2C, hipfftExecC2C)

#endif

// wwrfftGetStatusName/String are hand-written per-backend switches (see
// detail/fft_names.h), not WWR_FUNCTION_RAW aliases, so they cannot be
// WWR_SAME_FUNCTION'd; taking their address exercises the switch through the
// #include path and forces it to link.
namespace {
[[maybe_unused]] auto *const status_name = &wwr::wwrfftGetStatusName;
[[maybe_unused]] auto *const status_string = &wwr::wwrfftGetStatusString;
} // namespace

int main() { return 0; }
