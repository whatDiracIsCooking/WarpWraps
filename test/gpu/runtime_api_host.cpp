// runtime_api_host.cpp - the non-module HOST #include path for the runtime surface
//
// The counterpart to runtime_api.cppm's module test, one layer over: that test
// imports wwr.runtime_api and checks every exported name is the backend's own
// entity; this one does the same through the OTHER path a consumer has --
// #include "wwr/runtime_api.h" from a plain, non-module TU -- proving that path
// yields the identical surface bound to the identical backend entities. A
// representative spread across types, constants (including a flag that went
// through runtime_api.h's #undef-to-constexpr dance) and functions is enough:
// the full surface is already pinned name-by-name by the module test, and this
// shares runtime_api_surface.h with it, so what is under test here is the include
// PATH, not the list.
//
// A plain .cpp, not a .cppm: that is the whole point. It imports nothing and
// links wwr::runtime_api::host, the header-only target. The expected backend
// names are spelled out in full and switched on WWR_GPU_BACKEND_* (carried in by
// that target via wwr::backend), exactly as the module test does -- so a mistake
// in the binding macros fails here instead of being mirrored.

#include <type_traits>

#include "wwr/runtime_api.h"

// WWR_SAME_TYPE / WWR_SAME_VALUE / WWR_SAME_FUNCTION (identity + WWR_LINK_CHECK).
// Reached root-relative, so the target puts PROJECT_SOURCE_DIR on the include
// path; the vendor header the backend names resolve against is already in scope,
// pulled by runtime_api_host.h.
#include "gpu_check_macros.h"

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(wwrError_t, cudaError_t)
WWR_SAME_TYPE(wwrStream_t, cudaStream_t)
WWR_SAME_TYPE(wwrEvent_t, cudaEvent_t)
WWR_SAME_TYPE(wwrMemcpyKind, cudaMemcpyKind)

WWR_SAME_VALUE(wwrSuccess, cudaSuccess)
WWR_SAME_VALUE(wwrMemcpyHostToDevice, cudaMemcpyHostToDevice)
// StreamDefault is an allocation flag: a macro in the vendor header that
// runtime_api.h validates, #undef's and replaces with an equal-valued global
// constexpr. Binding against the constexpr on both sides proves the dance the
// #include path inherits from runtime_api.h actually ran.
WWR_SAME_VALUE(wwrStreamDefault, cudaStreamDefault)

WWR_SAME_FUNCTION(wwrGetDevice, cudaGetDevice)
WWR_SAME_FUNCTION(wwrStreamCreate, cudaStreamCreate)
WWR_SAME_FUNCTION(wwrStreamSynchronize, cudaStreamSynchronize)
WWR_SAME_FUNCTION(wwrMemcpy, cudaMemcpy)
WWR_SAME_FUNCTION(wwrFree, cudaFree)
// wwrMalloc is bound to the void** overload explicitly (hipMalloc has a
// template overload), so it is a reference to the one entry point -- identity
// still holds.
WWR_SAME_FUNCTION(wwrMalloc, cudaMalloc)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(wwrError_t, hipError_t)
WWR_SAME_TYPE(wwrStream_t, hipStream_t)
WWR_SAME_TYPE(wwrEvent_t, hipEvent_t)
WWR_SAME_TYPE(wwrMemcpyKind, hipMemcpyKind)

WWR_SAME_VALUE(wwrSuccess, hipSuccess)
WWR_SAME_VALUE(wwrMemcpyHostToDevice, hipMemcpyHostToDevice)
WWR_SAME_VALUE(wwrStreamDefault, hipStreamDefault)

WWR_SAME_FUNCTION(wwrGetDevice, hipGetDevice)
WWR_SAME_FUNCTION(wwrStreamCreate, hipStreamCreate)
WWR_SAME_FUNCTION(wwrStreamSynchronize, hipStreamSynchronize)
WWR_SAME_FUNCTION(wwrMemcpy, hipMemcpy)
WWR_SAME_FUNCTION(wwrFree, hipFree)
WWR_SAME_FUNCTION(wwrMalloc, hipMalloc)

#endif

// The hand-written forwarders in the surface (wwrGraphInstantiate,
// wwrMallocAsync, ...) are inline functions, not references to one backend entry
// point, so WWR_SAME_FUNCTION's address identity does not apply; taking their
// address proves they compiled and are callable through the #include path.
namespace {
[[maybe_unused]] auto *const graph_instantiate = &wwr::wwrGraphInstantiate;
[[maybe_unused]] auto *const malloc_async = &wwr::wwrMallocAsync;
} // namespace

int main() { return 0; }
