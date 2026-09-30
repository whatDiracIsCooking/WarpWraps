// nccl.cppm - Compile-time tests for wwr.cuda.nccl

module;

#include "test/shared/link_check.h"

export module wwr.test.cuda.nccl;

import std;
import wwr.cuda.nccl;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.cuda.nccl
//
// The module is a pure re-export. We verify at compile time that the handle
// typedefs and enums re-export with the shapes the wwrccl* layer relies on, and
// that every re-exported FUNCTION symbol resolves at link time -- the check the
// floor-conformance gate (#117) asserts each library carries in full.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::cuda::test {

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// Handle / type traits
// ncclComm_t and ncclWindow_t are opaque pointer handles; the status, datatype
// and reduction-op selectors are enums.
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<ncclComm_t>);
static_assert(std::is_pointer_v<ncclWindow_t>);
static_assert(std::is_enum_v<ncclResult_t>);
static_assert(std::is_enum_v<ncclDataType_t>);
static_assert(std::is_enum_v<ncclRedOp_t>);
static_assert(std::is_enum_v<ncclScalarResidence_t>);
static_assert(std::is_trivially_copyable_v<ncclUniqueId>);
static_assert(std::is_standard_layout_v<ncclUniqueId>);

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// Forces the linker to resolve every re-exported function symbol, catching a
// missing or unresolvable export that the type-only checks miss.
// ────────────────────────────────────────────────────────────────────────

// Version and device memory
WWR_LINK_CHECK(ncclGetVersion)
WWR_LINK_CHECK(ncclMemAlloc)
WWR_LINK_CHECK(ncclMemFree)

// Communicator lifecycle
WWR_LINK_CHECK(ncclCommAbort)
WWR_LINK_CHECK(ncclCommDestroy)
WWR_LINK_CHECK(ncclCommFinalize)
WWR_LINK_CHECK(ncclCommInitAll)
WWR_LINK_CHECK(ncclCommInitRank)
WWR_LINK_CHECK(ncclCommInitRankConfig)
WWR_LINK_CHECK(ncclCommInitRankScalable)
WWR_LINK_CHECK(ncclCommShrink)
WWR_LINK_CHECK(ncclCommSplit)
WWR_LINK_CHECK(ncclGetUniqueId)

// Communicator queries and error reporting
WWR_LINK_CHECK(ncclCommGetAsyncError)
WWR_LINK_CHECK(ncclGetErrorString)
WWR_LINK_CHECK(ncclGetLastError)
WWR_LINK_CHECK(ncclCommCount)
WWR_LINK_CHECK(ncclCommCuDevice)
WWR_LINK_CHECK(ncclCommUserRank)

// Buffer / window registration
WWR_LINK_CHECK(ncclCommDeregister)
WWR_LINK_CHECK(ncclCommRegister)
WWR_LINK_CHECK(ncclCommWindowDeregister)
WWR_LINK_CHECK(ncclCommWindowRegister)

// Custom reduction operators
WWR_LINK_CHECK(ncclRedOpCreatePreMulSum)
WWR_LINK_CHECK(ncclRedOpDestroy)

// Collectives
WWR_LINK_CHECK(ncclAllGather)
WWR_LINK_CHECK(ncclAllReduce)
WWR_LINK_CHECK(ncclBcast)
WWR_LINK_CHECK(ncclBroadcast)
WWR_LINK_CHECK(ncclRecv)
WWR_LINK_CHECK(ncclReduce)
WWR_LINK_CHECK(ncclReduceScatter)
WWR_LINK_CHECK(ncclSend)

// Group calls
WWR_LINK_CHECK(ncclGroupEnd)
WWR_LINK_CHECK(ncclGroupSimulateEnd)
WWR_LINK_CHECK(ncclGroupStart)

} // namespace wwr::cuda::test
