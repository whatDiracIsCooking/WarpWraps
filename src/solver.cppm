/**
 * @file solver.cppm
 * @brief Backend-neutral dense solver: wwrsolver* names for cuSOLVER / hipSOLVER
 *
 * wwrsolverDn<X><basename> stands for cusolverDn<X><basename> /
 * hipsolverDn<X><basename> for the legacy (int-based) API, and
 * wwrsolverDnX<basename> for the modern (cusolverDnParams_t-based,
 * int64_t-dimensioned) one. Every function is written out in full, one line
 * each. See backend.h.
 *
 * Both backends' legacy linear-solver and eigen/SVD partitions match 1:1
 * (32 and 48 functions), so every legacy function is listed.
 * hipSOLVER's modern surface is far narrower: only Xpotrf, Xpotrs, Xgetrf,
 * Xgetrs, Xgeqrf and their _bufferSize functions exist, and those 8 are what is
 * listed. The rest -- Xlarft, Xsytrs, Xtrtri and the whole modern
 * eigenvalue/SVD API -- has no hipSOLVER counterpart; reach it through
 * wwr.cuda.cusolverDn on a CUDA build. wwrsolverGetStatusName/String are
 * hand-written switches per backend -- docs/architecture.md, section 5.
 *
 * The wwrsolver* surface itself is NOT restated here: it is the one fragment the
 * two host paths share, detail/solver_names.h, pasted below inside the exported
 * `namespace wwr` exactly as wwr/solver.h (the non-module #include path) pastes
 * it. Add a name there, once, and both paths gain it. The vendor header is drawn
 * from solver.h, the single vendor-include point (the "rand.h shape"): this
 * module binds wwr* references straight to the `::cusolverDn*` / `::hipsolverDn*`
 * declarations with the _RAW macros and imports no raw vendor module -- the host
 * API is a real external-linkage library, so a reference needs only the
 * declaration. No import is needed for the surface: its only hand-written
 * functions are the status-string switches, which name no complex or blas type.
 *
 * Usage:
 *   import wwr.solver;
 *
 *   wwrsolverDnHandle_t handle;
 *   wwrsolverDnCreate(&handle);
 */

module;

#include "backend.h"

// The single vendor-include point for the solver layer: cusolverDn.h /
// hipsolver.h plus library_types.h, whose `::cusolverDn*` / `::hipsolverDn*` and
// cudaDataType / hipDataType declarations the _RAW bindings in
// detail/solver_names.h resolve against. No raw vendor module is imported -- the
// host API is external-linkage, so a reference or type alias needs only these
// declarations. See solver.h and backend.h.
#include "solver.h"

export module wwr.solver;

export namespace wwr {

// The whole neutral dense-solver surface -- types, constants, the status-string
// switches and the legacy and modern functions -- lives in
// detail/solver_names.h, the one list both host paths share: this module and
// wwr/solver.h (the non-module #include path). The WWR_*_RAW macros from
// backend.h and WWR_SELECTED_* and the vendor header from solver.h's #include are
// exactly what that fragment's header documents it needs in scope.
#include "detail/solver_names.h"

} // namespace wwr
