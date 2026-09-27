// complex.cppm - Compile-time tests for gpumod.complex
//
// Every exported gpu* name is the backend's own type. See gpu_check_macros.h.

module;

#include "gpu_check_macros.h"

export module gpumod.test.gpu.complex;

import std;
import gpumod.complex;
#if defined(WWR_GPU_BACKEND_CUDA)
import gpumod.cuda.cuComplex;
#else
import gpumod.hip.hip_complex;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

using namespace wwr::cuda;

WWR_SAME_TYPE(gpuFloatComplex, cuFloatComplex)
WWR_SAME_TYPE(gpuDoubleComplex, cuDoubleComplex)
WWR_SAME_TYPE(gpuComplex, cuComplex)

#else

using namespace wwr::hip;

WWR_SAME_TYPE(gpuFloatComplex, hipFloatComplex)
WWR_SAME_TYPE(gpuDoubleComplex, hipDoubleComplex)
WWR_SAME_TYPE(gpuComplex, hipComplex)

#endif

// Backend-independent: gpuComplex is the single-precision complex type.
static_assert(std::is_same_v<gpuComplex, gpuFloatComplex>);

// The host construction and arithmetic wrappers are forwarding functions, not
// WWR_FUNCTION reference bindings, so &gpu != &backend and WWR_SAME_FUNCTION cannot
// apply. A bare WWR_LINK_CHECK from this importing TU is the build-time claim: each
// exported inline wrapper is reachable by name across the import and links. The
// device-side counterparts in complex.cuh are proved separately by complex.cu.
WWR_LINK_CHECK(make_gpuFloatComplex)
WWR_LINK_CHECK(make_gpuDoubleComplex)
WWR_LINK_CHECK(gpuCrealf)
WWR_LINK_CHECK(gpuCimagf)
WWR_LINK_CHECK(gpuCreal)
WWR_LINK_CHECK(gpuCimag)
WWR_LINK_CHECK(gpuCabsf)
WWR_LINK_CHECK(gpuCabs)
WWR_LINK_CHECK(gpuConjf)
WWR_LINK_CHECK(gpuConj)
WWR_LINK_CHECK(gpuCaddf)
WWR_LINK_CHECK(gpuCsubf)
WWR_LINK_CHECK(gpuCmulf)
WWR_LINK_CHECK(gpuCdivf)
WWR_LINK_CHECK(gpuCadd)
WWR_LINK_CHECK(gpuCsub)
WWR_LINK_CHECK(gpuCmul)
WWR_LINK_CHECK(gpuCdiv)

} // namespace wwr::test
