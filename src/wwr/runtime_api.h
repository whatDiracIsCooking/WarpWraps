/**
 * @file wwr/runtime_api.h
 * @brief The backend-neutral runtime API as a single, self-contained HOST
 *        #include, for a TU that does not use C++ modules
 *
 * The non-module counterpart to `import wwr.runtime_api;`: a plain .cpp that
 * #includes this one header gets the same wwr* runtime surface the module
 * exports -- wwrError_t, wwrMalloc, wwrStreamCreate, wwrSuccess, ... -- bound to
 * the selected backend's ::cuda* / ::hip* entry points.
 *
 * The wwr.runtime_api MODULE is split (a vendor-include + flag-dance part in its
 * global module fragment, the name bindings in its exported purview) because
 * C++ modules force that: names declared in a GMF are not exported, and the
 * dance's internal-linkage flag constants cannot be exported at all. A plain
 * header has no such split to make -- it runs the vendor include and the
 * allocation-flag dance top to bottom, like the vendor headers it wraps
 * (cuda_runtime_api.h / hip_runtime_api.h are monolithic).
 *
 * The wwr* name list itself is NOT restated here: it is the one fragment every
 * path shares, detail/runtime_api_names.h, pasted below inside `namespace wwr`
 * exactly as runtime_api.cppm's purview and runtime.h's device section paste it.
 * Add a runtime name there, once, and this path gains it too -- no hand-sync.
 * What this file supplies before that #include is only what the fragment's
 * header documents it needs: the vendor header + flag dance and the reference-
 * alias binding macros (the module keeps the flag dance in runtime_api.h and the
 * macros in backend.h; both are validated by static_assert, so a duplicated
 * flag-dance drift is a compile error, not a silent one).
 *
 * Backend selection is shared too: selected_backend.h, the project-wide switch.
 * The include root, the WWR_GPU_BACKEND_* define it reads in a host compile, and
 * the runtime link library arrive by linking wwr::runtime_api::host. HOST only: a
 * device .cu/.cuh gets this surface from runtime.h's device section (which binds
 * the forwarding-template form the full vendor runtime header forces). See
 * src/runtime_api.cppm, src/runtime.h and docs/architecture.md, section 3.
 */

#pragma once

// HOST only -- see the file header. A device pass gets the surface from runtime.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#error                                                                                              \
    "wwr/runtime_api.h is the HOST #include path; a device .cu/.cuh gets the runtime surface from runtime.h. See src/README.md."
#endif

#include <cstddef> // std::size_t for the surface's hand-written forwarders

// ---------------------------------------------------------------------------
// Vendor include + allocation-flag dance (selected_backend.h + the vendor api
// header, with the flag macros #undef'd to equal-valued global constexprs so the
// ::-prefixed bindings below resolve). Same content the module's
// runtime_api.h fragment carries -- restated here, not shared, per the file header.
// ---------------------------------------------------------------------------
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

#include <cuda_runtime_api.h>

// Validate each allocation flag's macro value before #undef-ing it, so a vendor
// value drift is caught here rather than silently baked into the module's
// surface. See this file's header for why these -- and only these -- need it.
static_assert(cudaHostAllocDefault == 0x00, "cudaHostAllocDefault value mismatch");
static_assert(cudaHostAllocMapped == 0x02, "cudaHostAllocMapped value mismatch");
static_assert(cudaHostAllocWriteCombined == 0x04, "cudaHostAllocWriteCombined value mismatch");
static_assert(cudaEventDefault == 0x00, "cudaEventDefault value mismatch");
static_assert(cudaEventBlockingSync == 0x01, "cudaEventBlockingSync value mismatch");
static_assert(cudaEventDisableTiming == 0x02, "cudaEventDisableTiming value mismatch");
static_assert(cudaStreamDefault == 0x00, "cudaStreamDefault value mismatch");
static_assert(cudaStreamNonBlocking == 0x01, "cudaStreamNonBlocking value mismatch");
static_assert(cudaMemAttachGlobal == 0x01, "cudaMemAttachGlobal value mismatch");
static_assert(cudaMemAttachHost == 0x02, "cudaMemAttachHost value mismatch");
static_assert(cudaArrayDefault == 0x00, "cudaArrayDefault value mismatch");
static_assert(cudaArraySurfaceLoadStore == 0x02, "cudaArraySurfaceLoadStore value mismatch");

#undef cudaHostAllocDefault
#undef cudaHostAllocMapped
#undef cudaHostAllocWriteCombined
#undef cudaEventDefault
#undef cudaEventBlockingSync
#undef cudaEventDisableTiming
#undef cudaStreamDefault
#undef cudaStreamNonBlocking
#undef cudaMemAttachGlobal
#undef cudaMemAttachHost
#undef cudaArrayDefault
#undef cudaArraySurfaceLoadStore

// Global scope, not namespace wwr: runtime_api.cppm's WWR_VALUE_RAW bindings
// name ::cuda<Flag>, so the replacements must sit where that resolves. constexpr
// gives them internal linkage -- private to this one TU, never a module export.
constexpr unsigned int cudaHostAllocDefault = 0x00;
constexpr unsigned int cudaHostAllocMapped = 0x02;
constexpr unsigned int cudaHostAllocWriteCombined = 0x04;
constexpr unsigned int cudaEventDefault = 0x00;
constexpr unsigned int cudaEventBlockingSync = 0x01;
constexpr unsigned int cudaEventDisableTiming = 0x02;
constexpr unsigned int cudaStreamDefault = 0x00;
constexpr unsigned int cudaStreamNonBlocking = 0x01;
constexpr unsigned int cudaMemAttachGlobal = 0x01;
constexpr unsigned int cudaMemAttachHost = 0x02;
constexpr unsigned int cudaArrayDefault = 0x00;
constexpr unsigned int cudaArraySurfaceLoadStore = 0x02;

#else

// __HIP_DISABLE_CPP_FUNCTIONS__ before the vendor header, as the raw
// wwr.hip.hip_runtime_api module's global module fragment does: it keeps the
// HIP headers from pulling in the C++ helper overloads a module cannot export.
#define __HIP_DISABLE_CPP_FUNCTIONS__ 1
#include <hip/hip_runtime_api.h>

// The HIP counterparts of the CUDA block above. HIP spells the pinned-host
// family hipHostMalloc* (its hipHostAlloc* names are equal-valued aliases);
// wwrHostAlloc* in runtime_api.cppm bind to these. Values verified against
// hip/hip_runtime_api.h independently of CUDA's, though they happen to agree.
static_assert(hipHostMallocDefault == 0x0, "hipHostMallocDefault value mismatch");
static_assert(hipHostMallocMapped == 0x2, "hipHostMallocMapped value mismatch");
static_assert(hipHostMallocWriteCombined == 0x4, "hipHostMallocWriteCombined value mismatch");
static_assert(hipEventDefault == 0x0, "hipEventDefault value mismatch");
static_assert(hipEventBlockingSync == 0x1, "hipEventBlockingSync value mismatch");
static_assert(hipEventDisableTiming == 0x2, "hipEventDisableTiming value mismatch");
static_assert(hipStreamDefault == 0x00, "hipStreamDefault value mismatch");
static_assert(hipStreamNonBlocking == 0x01, "hipStreamNonBlocking value mismatch");
static_assert(hipMemAttachGlobal == 0x01, "hipMemAttachGlobal value mismatch");
static_assert(hipMemAttachHost == 0x02, "hipMemAttachHost value mismatch");
static_assert(hipArrayDefault == 0x00, "hipArrayDefault value mismatch");
static_assert(hipArraySurfaceLoadStore == 0x02, "hipArraySurfaceLoadStore value mismatch");

#undef hipHostMallocDefault
#undef hipHostMallocMapped
#undef hipHostMallocWriteCombined
#undef hipEventDefault
#undef hipEventBlockingSync
#undef hipEventDisableTiming
#undef hipStreamDefault
#undef hipStreamNonBlocking
#undef hipMemAttachGlobal
#undef hipMemAttachHost
#undef hipArrayDefault
#undef hipArraySurfaceLoadStore

constexpr unsigned int hipHostMallocDefault = 0x0;
constexpr unsigned int hipHostMallocMapped = 0x2;
constexpr unsigned int hipHostMallocWriteCombined = 0x4;
constexpr unsigned int hipEventDefault = 0x0;
constexpr unsigned int hipEventBlockingSync = 0x1;
constexpr unsigned int hipEventDisableTiming = 0x2;
constexpr unsigned int hipStreamDefault = 0x00;
constexpr unsigned int hipStreamNonBlocking = 0x01;
constexpr unsigned int hipMemAttachGlobal = 0x01;
constexpr unsigned int hipMemAttachHost = 0x02;
constexpr unsigned int hipArrayDefault = 0x00;
constexpr unsigned int hipArraySurfaceLoadStore = 0x02;

#endif

// ---------------------------------------------------------------------------
// Binding macros, reference-alias form -- valid because only the MINIMAL api
// header above is in scope, whose entry points are not overloaded. Keyed on
// WWR_SELECTED_*, set by selected_backend.h above. #undef'd at the end so none
// leaks into the including TU.
// ---------------------------------------------------------------------------
#if defined(WWR_SELECTED_CUDA)
#define WWR_SELECT_RAW(cuda_name, hip_name) ::cuda_name
#else
#define WWR_SELECT_RAW(cuda_name, hip_name) ::hip_name
#endif
#define WWR_TYPE_RAW(wwr_name, cuda_name, hip_name)                                                 \
  using wwr_name = WWR_SELECT_RAW(cuda_name, hip_name);
#define WWR_VALUE_RAW(wwr_name, cuda_name, hip_name)                                                \
  inline constexpr auto wwr_name = WWR_SELECT_RAW(cuda_name, hip_name);
#define WWR_FUNCTION_RAW(wwr_name, cuda_name, hip_name)                                             \
  inline constexpr auto &wwr_name = WWR_SELECT_RAW(cuda_name, hip_name);
#define WWR_RT_TYPE(x) WWR_TYPE_RAW(wwr##x, cuda##x, hip##x)
#define WWR_RT_VALUE(x) WWR_VALUE_RAW(wwr##x, cuda##x, hip##x)
#define WWR_RT_FUNCTION(x) WWR_FUNCTION_RAW(wwr##x, cuda##x, hip##x)

// ---------------------------------------------------------------------------
// The wwr* runtime surface -- types, constants, functions and the hand-written
// forwarders -- pasted from the one fragment every path shares. The macros
// above plus the vendor header + flag dance and <cstddef>'s std::size_t are
// exactly what detail/runtime_api_names.h documents it needs in scope.
// ---------------------------------------------------------------------------
namespace wwr {
#include "detail/runtime_api_names.h"
} // namespace wwr

#undef WWR_SELECT_RAW
#undef WWR_TYPE_RAW
#undef WWR_VALUE_RAW
#undef WWR_FUNCTION_RAW
#undef WWR_RT_TYPE
#undef WWR_RT_VALUE
#undef WWR_RT_FUNCTION
