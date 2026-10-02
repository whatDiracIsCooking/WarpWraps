// rand_host.cpp - the non-module HOST #include path for the rand surface
//
// The counterpart to rand.cppm's module test, through the OTHER path a consumer
// has -- #include "wwr/rand.h" from a plain, non-module TU -- proving that path
// yields the identical host surface bound to the identical backend entities. A
// representative spread across types, constants, functions, a device state type
// (seen straight from rand.h) and the scramble-constant forwarder is enough: the
// full surface is pinned name-by-name by the module test, and this shares
// detail/rand_names.h with it, so what is under test here is the include PATH.
//
// The __device__ generator functions are NOT reachable here and must not be --
// they live in rand.h's device-pass-gated section, absent in this host TU.
//
// A plain .cpp that imports nothing and links wwr::rand::host, the header-only
// target, which carries WWR_GPU_BACKEND_* via wwr::backend.

#include <type_traits>

#include "wwr/rand.h"

#include "gpu_check_macros.h"

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

WWR_SAME_TYPE(wwrrandGenerator_t, curandGenerator_t)
WWR_SAME_TYPE(wwrrandStatus_t, curandStatus_t)

WWR_SAME_VALUE(WWRRAND_STATUS_SUCCESS, CURAND_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRRAND_RNG_PSEUDO_DEFAULT, CURAND_RNG_PSEUDO_DEFAULT)

WWR_SAME_FUNCTION(wwrrandCreateGenerator, curandCreateGenerator)
WWR_SAME_FUNCTION(wwrrandGenerateNormal, curandGenerateNormal)

// A device generator state type, reached straight from rand.h (which wwr/rand.h
// includes) -- proving the state-type list travels the #include path too.
WWR_SAME_TYPE(wwrrandState, curandState)

#else

WWR_SAME_TYPE(wwrrandGenerator_t, hiprandGenerator_t)
WWR_SAME_TYPE(wwrrandStatus_t, hiprandStatus_t)

WWR_SAME_VALUE(WWRRAND_STATUS_SUCCESS, HIPRAND_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRRAND_RNG_PSEUDO_DEFAULT, HIPRAND_RNG_PSEUDO_DEFAULT)

WWR_SAME_FUNCTION(wwrrandCreateGenerator, hiprandCreateGenerator)
WWR_SAME_FUNCTION(wwrrandGenerateNormal, hiprandGenerateNormal)

WWR_SAME_TYPE(wwrrandState, hiprandState)

#endif

// wwrrandGetScrambleConstants32 is a forwarding function on CUDA (cuRAND hands
// the table out non-const) and a reference on HIP; its signature is hipRAND's
// const-correct one on both. Checking the signature exercises the scramble shim
// through the #include path; taking its address forces the forwarder to link.
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrrandGetScrambleConstants32)>,
              wwrrandStatus_t(const unsigned int **))

namespace {
[[maybe_unused]] auto *const scramble32 = &wwr::wwrrandGetScrambleConstants32;
} // namespace

int main() { return 0; }
