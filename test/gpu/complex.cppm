// complex.cppm - Compile-time tests for wwr.complex
//
// Every exported wwr* name is the backend's own type. See gpu_check_macros.h.

module;

#include "gpu_check_macros.h"

export module wwr.test.gpu.complex;

import std;
import wwr.complex;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cuComplex;
#else
import wwr.hip.hip_complex;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

using namespace wwr::cuda;

WWR_SAME_TYPE(wwrFloatComplex, cuFloatComplex)
WWR_SAME_TYPE(wwrDoubleComplex, cuDoubleComplex)
WWR_SAME_TYPE(wwrComplex, cuComplex)

#else

using namespace wwr::hip;

WWR_SAME_TYPE(wwrFloatComplex, hipFloatComplex)
WWR_SAME_TYPE(wwrDoubleComplex, hipDoubleComplex)
WWR_SAME_TYPE(wwrComplex, hipComplex)

#endif

// Backend-independent: wwrComplex is the single-precision complex type.
static_assert(std::is_same_v<wwrComplex, wwrFloatComplex>);

// Construction and the real/imag accessors are usable in a constant expression
// on both backends (brace-init + `.x`/`.y` reads; see docs/architecture.md §3).
constexpr wwrFloatComplex kFloat = make_wwrFloatComplex(1.0f, 2.0f);
static_assert(wwrCrealf(kFloat) == 1.0f && wwrCimagf(kFloat) == 2.0f);

constexpr wwrDoubleComplex kDouble = make_wwrDoubleComplex(3.0, 4.0);
static_assert(wwrCreal(kDouble) == 3.0 && wwrCimag(kDouble) == 4.0);

constexpr wwrComplex kComplex = make_wwrComplex(5.0f, 6.0f);
static_assert(wwrCrealf(kComplex) == 5.0f && wwrCimagf(kComplex) == 6.0f);

// The host arithmetic wrappers are forwarding functions, not WWR_FUNCTION
// reference bindings, so &gpu != &backend and WWR_SAME_FUNCTION cannot apply. A
// bare WWR_LINK_CHECK from this importing TU is the build-time claim: each
// exported wrapper is reachable by name across the import and links. The
// device-side counterparts in complex.h's device section are proved separately by complex.cu.
WWR_LINK_CHECK(make_wwrFloatComplex)
WWR_LINK_CHECK(make_wwrDoubleComplex)
WWR_LINK_CHECK(wwrCrealf)
WWR_LINK_CHECK(wwrCimagf)
WWR_LINK_CHECK(wwrCreal)
WWR_LINK_CHECK(wwrCimag)
WWR_LINK_CHECK(wwrCabsf)
WWR_LINK_CHECK(wwrCabs)
WWR_LINK_CHECK(wwrConjf)
WWR_LINK_CHECK(wwrConj)
WWR_LINK_CHECK(wwrCaddf)
WWR_LINK_CHECK(wwrCsubf)
WWR_LINK_CHECK(wwrCmulf)
WWR_LINK_CHECK(wwrCdivf)
WWR_LINK_CHECK(wwrCadd)
WWR_LINK_CHECK(wwrCsub)
WWR_LINK_CHECK(wwrCmul)
WWR_LINK_CHECK(wwrCdiv)

} // namespace wwr::test
