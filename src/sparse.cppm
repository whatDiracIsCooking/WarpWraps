/**
 * @file sparse.cppm
 * @brief Backend-neutral sparse: wwrsparse* names for cuSPARSE / hipSPARSE
 *
 * wwrsparse<X><name> stands for cusparse<X><name> on a CUDA build and
 * hipsparse<X><name> on a HIP build. See backend.h.
 *
 * Only the names src/wrappers/sparse uses are listed, plus the handle, stream,
 * pointer-mode, error-string and matrix-descriptor helpers a caller needs. This
 * layer covers the legacy typed (S/D/C/Z) functions the two backends still share
 * and that cuSPARSE has not deprecated: the BSR matrix-vector multiply, the
 * tridiagonal/pentadiagonal batch solvers (gtsv2 / gpsvInterleavedBatch), CSR
 * matrix addition (csrgeam2) and a few format conversions (nnz, gebsr2gebsc,
 * csr2gebsr). The modern generic API (SpMV, SpMM, SpGEMM, ...) is not wrapped
 * here -- its element type is a runtime cudaDataType/hipDataType argument rather
 * than a name letter, so it needs no S/D/C/Z dispatch; reach it through
 * wwr.cuda.cusparse / wwr.hip.hipsparse.
 *
 * cuSPARSE-only functions (the Preview SpMMOp API, CreateSlicedEll) and the
 * legacy typed functions cuSPARSE removed but hipSPARSE keeps (csrmv, csrsv2,
 * csrmm, csrsm2, csric02, csrilu02, gemvi, the HYB path, ...) are absent -- this
 * module adapts what both vendors offer, same policy as wwr.blas / wwr.solver.
 *
 * One backend difference is resolved in the fragment, not above: cuSPARSE's
 * non-Ext gebsr2gebsc_bufferSize / csr2gebsr_bufferSize write the byte count as
 * int* where hipSPARSE writes std::size_t*, so these keep hipSPARSE's
 * std::size_t* signature, forwarding through an int on CUDA.
 * See docs/architecture.md, section 5.
 *
 * The wwrsparse* surface itself is NOT restated here: it is the one fragment the
 * two host paths share, detail/sparse_names.h, pasted below inside the exported
 * `namespace wwr` exactly as wwr/sparse.h (the non-module #include path) pastes
 * it. Add a name there, once, and both paths gain it. The vendor header is drawn
 * from sparse.h, the single vendor-include point (the "rand.h shape"): this
 * module binds wwr* references straight to the `::cusparse*` / `::hipsparse*`
 * declarations with the _RAW macros and imports no raw vendor module -- the host
 * API is a real external-linkage library, so a reference needs only the
 * declaration. import wwr.complex stays, for the complex types the *_bufferSize
 * shims name.
 *
 * Usage:
 *   import wwr.sparse;
 *
 *   wwrsparseHandle_t handle;
 *   wwrsparseCreate(&handle);
 */

module;

#include "backend.h"

// The single vendor-include point for the sparse layer: cusparse.h /
// hipsparse/hipsparse.h, whose `::cusparse*` / `::hipsparse*` declarations the
// _RAW bindings in detail/sparse_names.h resolve against, plus <cstddef> for the
// std::size_t* the *_bufferSize shims name. No raw vendor module is imported --
// the host API is external-linkage, so a reference or type alias needs only
// these declarations. See sparse.h and backend.h.
#include "sparse.h"

export module wwr.sparse;

// Load-bearing on HIP, and the one spot sparse diverges from blas / solver:
// hipsparse.h (unlike hipblas.h / hipsolver.h, which pull only the light
// <hip/hip_runtime_api.h>) drags in hipsparse-bfloat16.h, hence <ostream> /
// <iostream>, textually into this module's global module fragment via sparse.h.
// A std stream header textually in a GMF collides with a consumer's `import
// std` and breaks that consumer's std::format / std::println (the deleted
// formatter<basic_format_string> path) -- unless the module that pulled the
// textual stream header ALSO imports std, which reconciles the two std views
// and seals it. So wwr.sparse imports std even though its own purview names
// only std::size_t (which <cstddef> via sparse.h already supplies); the raw
// module wwr.hip.hipsparse imports std for exactly the same reason.
import std;

// The complex types the *_bufferSize shims name in their signatures. Not a
// vendor-include dependency -- the cuSPARSE/hipSPARSE declarations come from
// sparse.h above; this is only wwrFloatComplex / wwrDoubleComplex.
import wwr.complex;

export namespace wwr {

// The whole neutral sparse surface -- types, constants, functions and the
// std::size_t* buffer-size shims -- lives in detail/sparse_names.h, the one list
// both host paths share: this module and wwr/sparse.h (the non-module #include
// path). The WWR_*_RAW macros from backend.h and WWR_SELECTED_*, the vendor
// header and std::size_t from sparse.h's #include, and wwrFloatComplex /
// wwrDoubleComplex from import wwr.complex are exactly what that fragment's
// header documents it needs in scope.
#include "detail/sparse_names.h"

} // namespace wwr
