/**
 * @file blas.cppm
 * @brief Backend-neutral BLAS: wwrblas* names for cuBLAS / hipBLAS
 *
 * wwrblas<X> stands for cublas<X>_v2 where cuBLAS has a _v2 name pair and
 * cublas<X> otherwise, and for hipblas<X>, which has no _v2 names; 64-bit
 * variants are wwrblas<X>_64. Each is written out in full, one line per name.
 * See backend.h.
 *
 * Only the names src/wrappers/blas uses are listed, plus the handle, stream
 * and pointer-mode calls a caller needs. cuBLAS-only functions (gemm3m,
 * gemmGroupedBatched, matinvBatched, tpttr, trttp) are absent; reach those
 * through wwr.cuda.cublas_v2.
 *
 * Two backend differences are resolved here, not above: hipBLAS's one
 * status-to-string function backs both wwrblasGetStatusName and
 * wwrblasGetStatusString, and getrsBatched / getriBatched keep cuBLAS's
 * const-correct signature, forwarding with a const_cast on HIP.
 * See docs/architecture.md, section 5.
 *
 * The wwrblas* surface itself is NOT restated here: it is the one fragment the
 * two host paths share, detail/blas_names.h, pasted below inside the exported
 * `namespace wwr` exactly as wwr/blas.h (the non-module #include path) pastes it.
 * Add a name there, once, and both paths gain it. The vendor header is drawn from
 * blas.h, the single vendor-include point (the "rand.h shape"): this module binds
 * wwr* references straight to the `::cublas*_v2` / `::hipblas*` declarations with
 * the _RAW macros and imports no raw vendor module -- the host API is a real
 * external-linkage library, so a reference needs only the declaration. import
 * wwr.complex stays, for the complex types the getri/getrs forwarders name.
 *
 * Usage:
 *   import wwr.blas;
 *
 *   wwrblasHandle_t handle;
 *   wwrblasCreate(&handle);
 */

module;

#include "backend.h"

// The single vendor-include point for the blas layer: cublas_v2.h / hipblas.h,
// whose `::cublas*_v2` / `::hipblas*` declarations the _RAW bindings in
// detail/blas_names.h resolve against. No raw vendor module is imported -- the
// host API is external-linkage, so a reference or type alias needs only these
// declarations. See blas.h and backend.h.
#include "blas.h"

export module wwr.blas;

// The complex types the getri/getrs batched forwarders name in their signatures.
// Not a vendor-include dependency -- the cuBLAS/hipBLAS declarations come from
// blas.h above; this is only wwrFloatComplex / wwrDoubleComplex.
import wwr.complex;

export namespace wwr {

// The whole neutral BLAS surface -- types, constants, functions and the
// const-correct getrs/getri forwarders -- lives in detail/blas_names.h, the one
// list both host paths share: this module and wwr/blas.h (the non-module #include
// path). The WWR_*_RAW macros from backend.h and WWR_SELECTED_* and the vendor
// header from blas.h's #include are exactly what that fragment's header documents
// it needs in scope.
#include "detail/blas_names.h"

} // namespace wwr
