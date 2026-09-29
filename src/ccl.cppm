/**
 * @file ccl.cppm
 * @brief Backend-neutral multi-GPU collectives: wwrccl* names for NCCL / RCCL
 *
 * The one broadly-popular vendor collectives library where BOTH backends agree
 * on the API: RCCL is a source-compatible reimplementation of NCCL, so there is
 * no prefix divergence to bridge -- both spell the entire surface nccl* /
 * NCCL_*. wwrccl<X> stands for nccl<X> on either backend, WWRCCL_<X> for
 * NCCL_<X>, each written out in full. See backend.h.
 *
 * This layer carries the measured intersection of the two vendor headers (run
 * devtools/header_intersection.py --cuda nccl.h --hip rccl.h to reproduce it) --
 * measured against the ACTUALLY INSTALLED libraries, not an upstream tag. RCCL's
 * version leads the packaged NCCL here, so a handful of RCCL names have no NCCL
 * counterpart and are therefore ABSENT -- reachable only through the raw
 * wwr.hip.rccl module, never portably:
 *   - ncclGather / ncclScatter, ncclAllToAll / ncclAllToAllv, ncclAllReduceWithBias
 *   - ncclResetDebugInit (in NCCL's upstream header too, but the libnccl-dev the
 *     CUDA image ships predates it)
 *
 * The compile-time NCCL_MAJOR / NCCL_VERSION_CODE macros are not aliased: their
 * values are backend-specific, and the runtime wwrcclGetVersion query is the
 * portable equivalent. The two struct-initializer macros
 * (NCCL_CONFIG_INITIALIZER, NCCL_SIM_INFO_INITIALIZER) are likewise not
 * exposed -- they embed a backend-specific size/magic/version, and a module
 * cannot re-export a macro. A wwrcclConfig_t must therefore be populated by the
 * caller (the RAII communicator wrapper that would own its defaulting is the
 * "optionally later" extension layer noted in the proposal, not built here).
 *
 * Enumerator VALUES are pinned per backend in test/gpu/ccl.cppm, one line per
 * name: matching names do not guarantee matching values (this bit WWRRAND_RNG_*
 * before -- see rand.cppm and docs/architecture.md, sections 1 and 6).
 *
 * Usage:
 *   import wwr.ccl;
 *
 *   wwrcclComm_t comm;
 *   wwrcclCommInitRank(&comm, nranks, id, rank);
 *   wwrcclAllReduce(sendbuff, recvbuff, count, WWRCCL_FLOAT32, WWRCCL_SUM, comm, stream);
 */

module;

#include "backend.h"

export module wwr.ccl;

#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.nccl;
#else
import wwr.hip.rccl;
#endif

export namespace wwr {

// ========================================================================
// Types
// ========================================================================

WWR_TYPE(wwrcclComm_t, ncclComm_t, ncclComm_t)
WWR_TYPE(wwrcclUniqueId, ncclUniqueId, ncclUniqueId)
WWR_TYPE(wwrcclWindow_t, ncclWindow_t, ncclWindow_t)
WWR_TYPE(wwrcclResult_t, ncclResult_t, ncclResult_t)
WWR_TYPE(wwrcclConfig_t, ncclConfig_t, ncclConfig_t)
WWR_TYPE(wwrcclSimInfo_t, ncclSimInfo_t, ncclSimInfo_t)
WWR_TYPE(wwrcclRedOp_dummy_t, ncclRedOp_dummy_t, ncclRedOp_dummy_t)
WWR_TYPE(wwrcclRedOp_t, ncclRedOp_t, ncclRedOp_t)
WWR_TYPE(wwrcclDataType_t, ncclDataType_t, ncclDataType_t)
WWR_TYPE(wwrcclScalarResidence_t, ncclScalarResidence_t, ncclScalarResidence_t)

// ========================================================================
// Constants - result codes (values differ between backends, see file header)
// ========================================================================

WWR_VALUE(WWRCCL_SUCCESS, ncclSuccess, ncclSuccess)
WWR_VALUE(WWRCCL_UNHANDLED_CUDA_ERROR, ncclUnhandledCudaError, ncclUnhandledCudaError)
WWR_VALUE(WWRCCL_SYSTEM_ERROR, ncclSystemError, ncclSystemError)
WWR_VALUE(WWRCCL_INTERNAL_ERROR, ncclInternalError, ncclInternalError)
WWR_VALUE(WWRCCL_INVALID_ARGUMENT, ncclInvalidArgument, ncclInvalidArgument)
WWR_VALUE(WWRCCL_INVALID_USAGE, ncclInvalidUsage, ncclInvalidUsage)
WWR_VALUE(WWRCCL_REMOTE_ERROR, ncclRemoteError, ncclRemoteError)
WWR_VALUE(WWRCCL_IN_PROGRESS, ncclInProgress, ncclInProgress)
WWR_VALUE(WWRCCL_NUM_RESULTS, ncclNumResults, ncclNumResults)

// ========================================================================
// Constants - reduction operators
// ========================================================================

WWR_VALUE(WWRCCL_NUM_OPS_DUMMY, ncclNumOps_dummy, ncclNumOps_dummy)
WWR_VALUE(WWRCCL_SUM, ncclSum, ncclSum)
WWR_VALUE(WWRCCL_PROD, ncclProd, ncclProd)
WWR_VALUE(WWRCCL_MAX, ncclMax, ncclMax)
WWR_VALUE(WWRCCL_MIN, ncclMin, ncclMin)
WWR_VALUE(WWRCCL_AVG, ncclAvg, ncclAvg)
WWR_VALUE(WWRCCL_NUM_OPS, ncclNumOps, ncclNumOps)
WWR_VALUE(WWRCCL_MAX_RED_OP, ncclMaxRedOp, ncclMaxRedOp)

// ========================================================================
// Constants - data types (several share a value: WWRCCL_CHAR == WWRCCL_INT8,
// WWRCCL_HALF == WWRCCL_FLOAT16, WWRCCL_INT == WWRCCL_INT32, ...)
// ========================================================================

WWR_VALUE(WWRCCL_INT8, ncclInt8, ncclInt8)
WWR_VALUE(WWRCCL_CHAR, ncclChar, ncclChar)
WWR_VALUE(WWRCCL_UINT8, ncclUint8, ncclUint8)
WWR_VALUE(WWRCCL_INT32, ncclInt32, ncclInt32)
WWR_VALUE(WWRCCL_INT, ncclInt, ncclInt)
WWR_VALUE(WWRCCL_UINT32, ncclUint32, ncclUint32)
WWR_VALUE(WWRCCL_INT64, ncclInt64, ncclInt64)
WWR_VALUE(WWRCCL_UINT64, ncclUint64, ncclUint64)
WWR_VALUE(WWRCCL_FLOAT16, ncclFloat16, ncclFloat16)
WWR_VALUE(WWRCCL_HALF, ncclHalf, ncclHalf)
WWR_VALUE(WWRCCL_FLOAT32, ncclFloat32, ncclFloat32)
WWR_VALUE(WWRCCL_FLOAT, ncclFloat, ncclFloat)
WWR_VALUE(WWRCCL_FLOAT64, ncclFloat64, ncclFloat64)
WWR_VALUE(WWRCCL_DOUBLE, ncclDouble, ncclDouble)
WWR_VALUE(WWRCCL_BFLOAT16, ncclBfloat16, ncclBfloat16)
WWR_VALUE(WWRCCL_FLOAT8E4M3, ncclFloat8e4m3, ncclFloat8e4m3)
WWR_VALUE(WWRCCL_FLOAT8E5M2, ncclFloat8e5m2, ncclFloat8e5m2)
WWR_VALUE(WWRCCL_NUM_TYPES, ncclNumTypes, ncclNumTypes)

// ========================================================================
// Constants - custom reduction operator scalar residence
// ========================================================================

WWR_VALUE(WWRCCL_SCALAR_DEVICE, ncclScalarDevice, ncclScalarDevice)
WWR_VALUE(WWRCCL_SCALAR_HOST_IMMEDIATE, ncclScalarHostImmediate, ncclScalarHostImmediate)

// ========================================================================
// Constants - scalar flag/param values (constexpr replacements the raw modules
// declare for the NCCL_* macros; see cuda/nccl.cppm and hip/rccl.cppm)
// ========================================================================

WWR_VALUE(WWRCCL_UNIQUE_ID_BYTES, NCCL_UNIQUE_ID_BYTES, NCCL_UNIQUE_ID_BYTES)
WWR_VALUE(WWRCCL_COMM_NULL, NCCL_COMM_NULL, NCCL_COMM_NULL)
WWR_VALUE(WWRCCL_SPLIT_NOCOLOR, NCCL_SPLIT_NOCOLOR, NCCL_SPLIT_NOCOLOR)
WWR_VALUE(WWRCCL_UNDEF_FLOAT, NCCL_UNDEF_FLOAT, NCCL_UNDEF_FLOAT)
WWR_VALUE(WWRCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT)
WWR_VALUE(WWRCCL_WIN_DEFAULT, NCCL_WIN_DEFAULT, NCCL_WIN_DEFAULT)
WWR_VALUE(WWRCCL_WIN_COLL_SYMMETRIC, NCCL_WIN_COLL_SYMMETRIC, NCCL_WIN_COLL_SYMMETRIC)
WWR_VALUE(WWRCCL_CTA_POLICY_DEFAULT, NCCL_CTA_POLICY_DEFAULT, NCCL_CTA_POLICY_DEFAULT)
WWR_VALUE(WWRCCL_CTA_POLICY_EFFICIENCY, NCCL_CTA_POLICY_EFFICIENCY, NCCL_CTA_POLICY_EFFICIENCY)
WWR_VALUE(WWRCCL_SHRINK_DEFAULT, NCCL_SHRINK_DEFAULT, NCCL_SHRINK_DEFAULT)
WWR_VALUE(WWRCCL_SHRINK_ABORT, NCCL_SHRINK_ABORT, NCCL_SHRINK_ABORT)

// ========================================================================
// Version query
// ========================================================================

WWR_FUNCTION(wwrcclGetVersion, ncclGetVersion, ncclGetVersion)

// ========================================================================
// Buffer allocation
// ========================================================================

WWR_FUNCTION(wwrcclMemAlloc, ncclMemAlloc, ncclMemAlloc)
WWR_FUNCTION(wwrcclMemFree, ncclMemFree, ncclMemFree)

// ========================================================================
// Communicator lifecycle
// ========================================================================

WWR_FUNCTION(wwrcclGetUniqueId, ncclGetUniqueId, ncclGetUniqueId)
WWR_FUNCTION(wwrcclCommInitRank, ncclCommInitRank, ncclCommInitRank)
WWR_FUNCTION(wwrcclCommInitRankConfig, ncclCommInitRankConfig, ncclCommInitRankConfig)
WWR_FUNCTION(wwrcclCommInitRankScalable, ncclCommInitRankScalable, ncclCommInitRankScalable)
WWR_FUNCTION(wwrcclCommInitAll, ncclCommInitAll, ncclCommInitAll)
WWR_FUNCTION(wwrcclCommSplit, ncclCommSplit, ncclCommSplit)
WWR_FUNCTION(wwrcclCommShrink, ncclCommShrink, ncclCommShrink)
WWR_FUNCTION(wwrcclCommFinalize, ncclCommFinalize, ncclCommFinalize)
WWR_FUNCTION(wwrcclCommDestroy, ncclCommDestroy, ncclCommDestroy)
WWR_FUNCTION(wwrcclCommAbort, ncclCommAbort, ncclCommAbort)

// ========================================================================
// Error checking
// ========================================================================

WWR_FUNCTION(wwrcclGetErrorString, ncclGetErrorString, ncclGetErrorString)
WWR_FUNCTION(wwrcclGetLastError, ncclGetLastError, ncclGetLastError)
WWR_FUNCTION(wwrcclCommGetAsyncError, ncclCommGetAsyncError, ncclCommGetAsyncError)

// ========================================================================
// Communicator information
// ========================================================================

WWR_FUNCTION(wwrcclCommCount, ncclCommCount, ncclCommCount)
WWR_FUNCTION(wwrcclCommCuDevice, ncclCommCuDevice, ncclCommCuDevice)
WWR_FUNCTION(wwrcclCommUserRank, ncclCommUserRank, ncclCommUserRank)

// ========================================================================
// Buffer / window registration
// ========================================================================

WWR_FUNCTION(wwrcclCommRegister, ncclCommRegister, ncclCommRegister)
WWR_FUNCTION(wwrcclCommDeregister, ncclCommDeregister, ncclCommDeregister)
WWR_FUNCTION(wwrcclCommWindowRegister, ncclCommWindowRegister, ncclCommWindowRegister)
WWR_FUNCTION(wwrcclCommWindowDeregister, ncclCommWindowDeregister, ncclCommWindowDeregister)

// ========================================================================
// Custom reduction operators
// ========================================================================

WWR_FUNCTION(wwrcclRedOpCreatePreMulSum, ncclRedOpCreatePreMulSum, ncclRedOpCreatePreMulSum)
WWR_FUNCTION(wwrcclRedOpDestroy, ncclRedOpDestroy, ncclRedOpDestroy)

// ========================================================================
// Collective communication
// ========================================================================

WWR_FUNCTION(wwrcclReduce, ncclReduce, ncclReduce)
WWR_FUNCTION(wwrcclBcast, ncclBcast, ncclBcast)
WWR_FUNCTION(wwrcclBroadcast, ncclBroadcast, ncclBroadcast)
WWR_FUNCTION(wwrcclAllReduce, ncclAllReduce, ncclAllReduce)
WWR_FUNCTION(wwrcclReduceScatter, ncclReduceScatter, ncclReduceScatter)
WWR_FUNCTION(wwrcclAllGather, ncclAllGather, ncclAllGather)
WWR_FUNCTION(wwrcclSend, ncclSend, ncclSend)
WWR_FUNCTION(wwrcclRecv, ncclRecv, ncclRecv)

// ========================================================================
// Group semantics
// ========================================================================

WWR_FUNCTION(wwrcclGroupStart, ncclGroupStart, ncclGroupStart)
WWR_FUNCTION(wwrcclGroupEnd, ncclGroupEnd, ncclGroupEnd)
WWR_FUNCTION(wwrcclGroupSimulateEnd, ncclGroupSimulateEnd, ncclGroupSimulateEnd)

} // namespace wwr
