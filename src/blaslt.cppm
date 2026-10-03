/**
 * @file blaslt.cppm
 * @brief Backend-neutral cuBLASLt / hipBLASLt: wwrblasLt* names
 *
 * wwrblasLt<X> stands for cublasLt<X> on a CUDA build and hipblasLt<X> on a HIP
 * build -- the modern "Lt" GEMM surface (epilogue-fused / mixed-precision /
 * narrow-float matmul), the counterpart to wwr.blas's classic API. Each name is
 * written out in full, one line per name, as wwr.blas does. See backend.h.
 *
 * The surface is exactly the intersection both backends spell the same, as
 * reported by `devtools/header_intersection.py --cuda cublasLt.h --hip
 * hipblaslt.h` (pinned prefixes cublas/hipblas for the functions and types,
 * cublaslt/hipblaslt for the enum constants): the handle, the matmul / matrix
 * layout / preference / matrix-transform descriptors with their attribute
 * get/set, the algo-heuristic search, wwrblasLtMatmul and wwrblasLtMatrixTransform
 * themselves, and the epilogue / order / pointer-mode / matrix-scale enums.
 *
 * Two shared names inherited from classic cuBLAS/hipBLAS are renamed under this
 * module's prefix, as gpu.solver renames cudaDataType to wwrsolverDataType_t:
 * wwrblasLtStatus_t is cublasStatus_t / hipblasStatus_t (cuBLASLt has no Lt
 * status type of its own; the STATUS_SUCCESS constants live in wwr.blas), and
 * wwrblasLtComputeType_t is cublasComputeType_t / hipblasComputeType_t.
 *
 * Deliberately left out, reach through wwr.cuda.cublasLt / wwr.hip.hipblaslt:
 *
 * - wwrblasLtGetVersion: the name is shared but the signatures diverge
 *   irreconcilably -- cublasLtGetVersion() takes no argument and returns size_t,
 *   hipblasLtGetVersion(handle, int*) returns a status.
 * - Everything only one backend exposes: cuBLASLt's algo introspection
 *   (cublasLtMatmulAlgoInit/Check/CapGetAttribute/ConfigSetAttribute), its logger
 *   and heuristics-cache controls, and the large tile / stages / reduction-scheme
 *   enums; hipBLASLt's hipblasLtGetArchName / hipblasLtGetGitRevision and its
 *   *_EXT epilogue and descriptor attributes.
 *
 * The wwrblasLt* surface itself is NOT restated here: it is the one fragment the
 * two host paths share, detail/blaslt_names.h, pasted below inside the exported
 * `namespace wwr` exactly as wwr/blaslt.h (the non-module #include path) pastes
 * it. Add a name there, once, and both paths gain it. The vendor header is drawn
 * from blaslt.h, the single vendor-include point (the "rand.h shape"): this
 * module binds wwr* references straight to the `::cublasLt*` / `::hipblasLt*`
 * declarations with the _RAW macros and imports no raw vendor module -- the host
 * API is a real external-linkage library, so a reference needs only the
 * declaration. No import is needed for the surface: it has no hand-written body,
 * only type / constant / function aliases.
 *
 * Usage:
 *   import wwr.blaslt;
 *
 *   wwrblasLtHandle_t handle;
 *   wwrblasLtCreate(&handle);
 */

module;

#include "backend.h"

// The single vendor-include point for the blaslt layer: cublasLt.h /
// hipblaslt/hipblaslt.h, whose `::cublasLt*` / `::hipblasLt*` declarations (and
// the cublasStatus_t / cublasComputeType_t the two inherited-type renames name)
// the _RAW bindings in detail/blaslt_names.h resolve against. No raw vendor
// module is imported -- the host API is external-linkage, so a reference or type
// alias needs only these declarations. See blaslt.h and backend.h.
#include "blaslt.h"

export module wwr.blaslt;

export namespace wwr {

// The whole neutral "Lt" GEMM surface -- types, constants and functions -- lives
// in detail/blaslt_names.h, the one list both host paths share: this module and
// wwr/blaslt.h (the non-module #include path). The WWR_*_RAW macros from
// backend.h and the vendor header from blaslt.h's #include are exactly what that
// fragment's header documents it needs in scope.
#include "detail/blaslt_names.h"

} // namespace wwr
