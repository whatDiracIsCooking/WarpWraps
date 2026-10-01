/**
 * @file runtime_api.h
 * @brief The single CUDA / HIP runtime-API vendor-include point for the
 *        backend-neutral wwr.runtime_api module
 *
 * A src/-root header wrapping just enough of the vendor runtime header for
 * runtime_api.cppm to bind its wwr* names straight to the vendor's own
 * ::cuda* / ::hip* declarations with the _RAW macros -- so the neutral module
 * imports no raw vendor module, the shape rand.h gives wwr.rand. The runtime
 * host API is a real external-linkage library (cudart / the HIP runtime), so a
 * type alias or a function reference needs only the declaration this header's
 * vendor #include supplies; the link library comes from CMake (CUDA::cudart /
 * hip::host, as fp16.h / bf16.h rely on). The raw modules
 * wwr.cuda.cuda_runtime_api / wwr.hip.hip_runtime_api stay as full
 * single-vendor surfaces, off this path.
 *
 * It carries one thing that cannot be bound to the vendor header directly: the
 * stream / event / host-alloc / managed-attach / array ALLOCATION FLAGS, which
 * the vendors spell as object-like macros (#define cudaStreamDefault 0x00). A
 * macro expanding to an integer literal cannot follow the `::` the _RAW macros
 * put in front of it, and -- unlike an import -- a global-module-fragment
 * #include leaks those macros into the module body below, the very collision
 * the neutral module's old import sidestepped. So each flag this module uses is
 * validated against its macro value, the macro is #undef'd, and a global-scope
 * constexpr of the same name replaces it; `::cudaStreamDefault` then names that
 * constant. Only the flags runtime_api.cppm actually re-exports are captured;
 * the rest of the runtime surface (the error / stream / memcpy / resource
 * ENUMS, the opaque HANDLE typedefs, every entry-point FUNCTION) has external
 * linkage and is reached straight from the vendor header, no macro dance.
 *
 * Host-only: there is no device-pass-gated section (a .cu reaches the runtime
 * API through runtime.h / runtime.cuh, not this header), so this file is
 * #included by exactly one TU -- runtime_api.cppm's global module fragment. It
 * reads selected_backend.h directly rather than device_guard.h, so it carries
 * no device-pass #error and compiles in that host TU. See src/rand.h,
 * src/backend.h, and docs/architecture.md, section 12.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from the compiler's device macro in a
// device pass or from WWR_GPU_BACKEND_* in a host compile. Directly, not via
// device_guard.h: this header is host-safe and must not #error.
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
