/**
 * @file runtime_api.cppm
 * @brief Backend-neutral GPU runtime API: wwr* names for cuda* / hip*
 *
 * Exports wwr-prefixed aliases of the CUDA runtime API (cuda_runtime_api.h)
 * or the HIP runtime API (hip_runtime_api.h), whichever backend this build is
 * configured for. See backend.h for the switch.
 *
 * Only the names src/wrappers uses are listed. Add a name here, once, when
 * code above this layer needs it; a name that differs between the backends
 * beyond the cuda/hip prefix gets its own explicit #if block.
 *
 * Usage:
 *   import wwr.runtime_api;
 *
 *   wwrStream_t stream;
 *   if (wwrStreamCreate(&stream) != wwrSuccess) { ... }
 */

module;

#include "backend.h"
#include "runtime_api.h"

// runtime_api.h is the single vendor-include point: it #includes the vendor
// runtime header and binds this module's wwr* names to the external-linkage
// ::cuda* / ::hip* declarations, so no raw vendor module is imported (the rand.h
// shape -- see runtime_api.h and src/README.md). The vendor header's colliding
// ALLOCATION-FLAG macros (cudaStreamDefault, cudaEventDefault, cudaArrayDefault,
// ...) would break the `::`-prefixed WWR_RT_VALUE expansions below, so
// runtime_api.h #undef's each and replaces it with an equally-named global
// constexpr -- the one thing a GMF #include leaks that an import did not.
// wwrStream_t is bound via WWR_RT_TYPE(Stream_t) to ::cudaStream_t / ::hipStream_t,
// the SAME vendor handle runtime.h names for device and GMF code, so a stream
// still crosses the module boundary as one type.

// Runtime API: wwrX -> ::cudaX / ::hipX. The _RAW macros bind to the vendor's
// OWN global names (the declarations runtime_api.h's #include brings in), not to
// a ::wwr::cuda / ::wwr::hip raw-module re-export -- this module imports none.
// See backend.h and runtime_api.h.
#define WWR_RT_TYPE(x) WWR_TYPE_RAW(wwr##x, cuda##x, hip##x)
#define WWR_RT_VALUE(x) WWR_VALUE_RAW(wwr##x, cuda##x, hip##x)
#define WWR_RT_FUNCTION(x) WWR_FUNCTION_RAW(wwr##x, cuda##x, hip##x)

export module wwr.runtime_api;

import std;

export namespace wwr {

// The whole neutral runtime surface -- types, constants, functions and the
// hand-written forwarders -- lives in runtime_api_surface.h, the one list this
// module and runtime.h's device section share (a device .cu/.cuh cannot import
// this module, so it pastes the same fragment through a #include). The WWR_RT_*
// / _RAW macros above and WWR_SELECT_RAW from backend.h, plus runtime_api.h's
// vendor header and flag dance and `import std`'s std::size_t, are exactly what
// that fragment's header documents it needs in scope.
#include "runtime_api_surface.h"

} // namespace wwr
