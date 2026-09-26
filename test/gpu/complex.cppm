// complex.cppm - Compile-time tests for gpumod.complex
//
// Every exported gpu* name is the backend's own type. See gpu_check_macros.h.

module;

#include "gpu_check_macros.h"

export module gpumod.test.gpu.complex;

import std;
import gpumod.complex;
#if defined(GPUMOD_GPU_BACKEND_CUDA)
import gpumod.cuda.cuComplex;
#else
import gpumod.hip.hip_complex;
#endif

namespace gpumod::test {

using namespace gpumod;

#if defined(GPUMOD_GPU_BACKEND_CUDA)

using namespace gpumod::cuda;

GPUMOD_SAME_TYPE(gpuFloatComplex, cuFloatComplex)
GPUMOD_SAME_TYPE(gpuDoubleComplex, cuDoubleComplex)
GPUMOD_SAME_TYPE(gpuComplex, cuComplex)

#else

using namespace gpumod::hip;

GPUMOD_SAME_TYPE(gpuFloatComplex, hipFloatComplex)
GPUMOD_SAME_TYPE(gpuDoubleComplex, hipDoubleComplex)
GPUMOD_SAME_TYPE(gpuComplex, hipComplex)

#endif

// Backend-independent: gpuComplex is the single-precision complex type.
static_assert(std::is_same_v<gpuComplex, gpuFloatComplex>);

// The host construction and arithmetic wrappers are forwarding functions, not
// GPUMOD_FUNCTION reference bindings, so &gpu != &backend and GPUMOD_SAME_FUNCTION cannot
// apply. A bare GPUMOD_LINK_CHECK from this importing TU is the build-time claim: each
// exported inline wrapper is reachable by name across the import and links. The
// device-side counterparts in complex.cuh are proved separately by complex.cu.
GPUMOD_LINK_CHECK(make_gpuFloatComplex)
GPUMOD_LINK_CHECK(make_gpuDoubleComplex)
GPUMOD_LINK_CHECK(gpuCrealf)
GPUMOD_LINK_CHECK(gpuCimagf)
GPUMOD_LINK_CHECK(gpuCreal)
GPUMOD_LINK_CHECK(gpuCimag)
GPUMOD_LINK_CHECK(gpuCabsf)
GPUMOD_LINK_CHECK(gpuCabs)
GPUMOD_LINK_CHECK(gpuConjf)
GPUMOD_LINK_CHECK(gpuConj)
GPUMOD_LINK_CHECK(gpuCaddf)
GPUMOD_LINK_CHECK(gpuCsubf)
GPUMOD_LINK_CHECK(gpuCmulf)
GPUMOD_LINK_CHECK(gpuCdivf)
GPUMOD_LINK_CHECK(gpuCadd)
GPUMOD_LINK_CHECK(gpuCsub)
GPUMOD_LINK_CHECK(gpuCmul)
GPUMOD_LINK_CHECK(gpuCdiv)

} // namespace gpumod::test
