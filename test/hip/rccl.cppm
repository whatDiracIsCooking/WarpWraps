// rccl.cppm - Compile-time tests for wwr.hip.rccl

module;

#include "test/shared/link_check.h"

export module wwr.test.hip.rccl;

import std;
import wwr.hip.rccl;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.hip.rccl
//
// Link-check-focused (#117): forces the linker to resolve every re-exported
// function symbol, so a re-export the library does not actually export fails the
// build. A symbol the header declares but the library does not define uses
// WWR_DECLARED_CHECK instead, naming the version it was found missing from.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::hip::test {

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(ncclGetVersion)
WWR_LINK_CHECK(ncclMemAlloc)
WWR_LINK_CHECK(ncclMemFree)
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
WWR_LINK_CHECK(ncclCommGetAsyncError)
WWR_LINK_CHECK(ncclGetErrorString)
WWR_LINK_CHECK(ncclGetLastError)
WWR_LINK_CHECK(ncclCommCount)
WWR_LINK_CHECK(ncclCommCuDevice)
WWR_LINK_CHECK(ncclCommUserRank)
WWR_LINK_CHECK(ncclCommDeregister)
WWR_LINK_CHECK(ncclCommRegister)
WWR_LINK_CHECK(ncclCommWindowDeregister)
WWR_LINK_CHECK(ncclCommWindowRegister)
WWR_LINK_CHECK(ncclRedOpCreatePreMulSum)
WWR_LINK_CHECK(ncclRedOpDestroy)
WWR_LINK_CHECK(ncclAllGather)
WWR_LINK_CHECK(ncclAllReduce)
WWR_LINK_CHECK(ncclBcast)
WWR_LINK_CHECK(ncclBroadcast)
WWR_LINK_CHECK(ncclRecv)
WWR_LINK_CHECK(ncclReduce)
WWR_LINK_CHECK(ncclReduceScatter)
WWR_LINK_CHECK(ncclSend)
WWR_LINK_CHECK(ncclAllReduceWithBias)
WWR_LINK_CHECK(ncclAllToAll)
WWR_LINK_CHECK(ncclAllToAllv)
WWR_LINK_CHECK(ncclGather)
WWR_LINK_CHECK(ncclScatter)
WWR_LINK_CHECK(ncclResetDebugInit)
WWR_LINK_CHECK(ncclGroupEnd)
WWR_LINK_CHECK(ncclGroupSimulateEnd)
WWR_LINK_CHECK(ncclGroupStart)

} // namespace wwr::hip::test
