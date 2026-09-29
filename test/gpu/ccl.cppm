// ccl.cppm - Compile-time tests for wwr.ccl
//
// Every exported wwrccl* name is checked against the backend's own entity: the
// same type, the same constant (type and value), the same function (see
// gpu_check_macros.h). The whole host API is listed -- like test/gpu/rand.cppm
// and unlike blas/solver, which sample -- because there is no runtime or
// dispatch test downstream to catch a wrong-but-existing backend name (the
// collectives need a real multi-GPU job to run).
//
// Unlike every other test/gpu module, the two backends are checked by ONE block
// of assertions rather than two: NCCL and RCCL are the one case where the HIP
// backend is a source-compatible reimplementation of the CUDA one, so both
// spell the entire surface with the identical nccl* / NCCL_* names. The names
// below are therefore still the vendor's own, spelled in full (never a
// gpu_backend.h macro) -- a backend-selected `using namespace` just points them
// at wwr::cuda or wwr::hip. Enumerator VALUES are pinned here regardless, since
// matching names never guarantee matching values (see rand.cppm).
//
// The RCCL-only collectives (ncclGather/Scatter/AllToAll{,v}/AllReduceWithBias)
// are NOT checked here: they are absent from wwr.ccl by design, exercised only
// through the raw wwr.hip.rccl module.

module;

#include "gpu_check_macros.h"

export module wwr.test.gpu.ccl;

import std;
import wwr.ccl;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.nccl;
#else
import wwr.hip.rccl;
#endif

namespace wwr::test {

using namespace wwr;
#if defined(WWR_GPU_BACKEND_CUDA)
using namespace wwr::cuda;
#else
using namespace wwr::hip;
#endif

// ────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(wwrcclComm_t, ncclComm_t)
WWR_SAME_TYPE(wwrcclUniqueId, ncclUniqueId)
WWR_SAME_TYPE(wwrcclWindow_t, ncclWindow_t)
WWR_SAME_TYPE(wwrcclResult_t, ncclResult_t)
WWR_SAME_TYPE(wwrcclConfig_t, ncclConfig_t)
WWR_SAME_TYPE(wwrcclSimInfo_t, ncclSimInfo_t)
WWR_SAME_TYPE(wwrcclRedOp_dummy_t, ncclRedOp_dummy_t)
WWR_SAME_TYPE(wwrcclRedOp_t, ncclRedOp_t)
WWR_SAME_TYPE(wwrcclDataType_t, ncclDataType_t)
WWR_SAME_TYPE(wwrcclScalarResidence_t, ncclScalarResidence_t)

// ────────────────────────────────────────────────────────────────────────
// Constants - result codes
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_VALUE(WWRCCL_SUCCESS, ncclSuccess)
WWR_SAME_VALUE(WWRCCL_UNHANDLED_CUDA_ERROR, ncclUnhandledCudaError)
WWR_SAME_VALUE(WWRCCL_SYSTEM_ERROR, ncclSystemError)
WWR_SAME_VALUE(WWRCCL_INTERNAL_ERROR, ncclInternalError)
WWR_SAME_VALUE(WWRCCL_INVALID_ARGUMENT, ncclInvalidArgument)
WWR_SAME_VALUE(WWRCCL_INVALID_USAGE, ncclInvalidUsage)
WWR_SAME_VALUE(WWRCCL_REMOTE_ERROR, ncclRemoteError)
WWR_SAME_VALUE(WWRCCL_IN_PROGRESS, ncclInProgress)
WWR_SAME_VALUE(WWRCCL_NUM_RESULTS, ncclNumResults)

// ────────────────────────────────────────────────────────────────────────
// Constants - reduction operators
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_VALUE(WWRCCL_NUM_OPS_DUMMY, ncclNumOps_dummy)
WWR_SAME_VALUE(WWRCCL_SUM, ncclSum)
WWR_SAME_VALUE(WWRCCL_PROD, ncclProd)
WWR_SAME_VALUE(WWRCCL_MAX, ncclMax)
WWR_SAME_VALUE(WWRCCL_MIN, ncclMin)
WWR_SAME_VALUE(WWRCCL_AVG, ncclAvg)
WWR_SAME_VALUE(WWRCCL_NUM_OPS, ncclNumOps)
WWR_SAME_VALUE(WWRCCL_MAX_RED_OP, ncclMaxRedOp)

// ────────────────────────────────────────────────────────────────────────
// Constants - data types
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_VALUE(WWRCCL_INT8, ncclInt8)
WWR_SAME_VALUE(WWRCCL_CHAR, ncclChar)
WWR_SAME_VALUE(WWRCCL_UINT8, ncclUint8)
WWR_SAME_VALUE(WWRCCL_INT32, ncclInt32)
WWR_SAME_VALUE(WWRCCL_INT, ncclInt)
WWR_SAME_VALUE(WWRCCL_UINT32, ncclUint32)
WWR_SAME_VALUE(WWRCCL_INT64, ncclInt64)
WWR_SAME_VALUE(WWRCCL_UINT64, ncclUint64)
WWR_SAME_VALUE(WWRCCL_FLOAT16, ncclFloat16)
WWR_SAME_VALUE(WWRCCL_HALF, ncclHalf)
WWR_SAME_VALUE(WWRCCL_FLOAT32, ncclFloat32)
WWR_SAME_VALUE(WWRCCL_FLOAT, ncclFloat)
WWR_SAME_VALUE(WWRCCL_FLOAT64, ncclFloat64)
WWR_SAME_VALUE(WWRCCL_DOUBLE, ncclDouble)
WWR_SAME_VALUE(WWRCCL_BFLOAT16, ncclBfloat16)
WWR_SAME_VALUE(WWRCCL_FLOAT8E4M3, ncclFloat8e4m3)
WWR_SAME_VALUE(WWRCCL_FLOAT8E5M2, ncclFloat8e5m2)
WWR_SAME_VALUE(WWRCCL_NUM_TYPES, ncclNumTypes)

// The datatype enum deliberately aliases several names to one value; pin those
// so a future divergence in either backend is caught here, not at a call site.
static_assert(WWRCCL_INT8 == WWRCCL_CHAR);
static_assert(WWRCCL_INT32 == WWRCCL_INT);
static_assert(WWRCCL_FLOAT16 == WWRCCL_HALF);
static_assert(WWRCCL_FLOAT32 == WWRCCL_FLOAT);
static_assert(WWRCCL_FLOAT64 == WWRCCL_DOUBLE);

// ────────────────────────────────────────────────────────────────────────
// Constants - scalar residence
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_VALUE(WWRCCL_SCALAR_DEVICE, ncclScalarDevice)
WWR_SAME_VALUE(WWRCCL_SCALAR_HOST_IMMEDIATE, ncclScalarHostImmediate)

// ────────────────────────────────────────────────────────────────────────
// Constants - scalar flag/param values (constexpr replacements for the NCCL_*
// macros the raw modules declare -- see cuda/nccl.cppm and hip/rccl.cppm)
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_VALUE(WWRCCL_UNIQUE_ID_BYTES, NCCL_UNIQUE_ID_BYTES)
WWR_SAME_VALUE(WWRCCL_COMM_NULL, NCCL_COMM_NULL)
WWR_SAME_VALUE(WWRCCL_SPLIT_NOCOLOR, NCCL_SPLIT_NOCOLOR)
WWR_SAME_VALUE(WWRCCL_UNDEF_FLOAT, NCCL_UNDEF_FLOAT)
WWR_SAME_VALUE(WWRCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT)
WWR_SAME_VALUE(WWRCCL_WIN_DEFAULT, NCCL_WIN_DEFAULT)
WWR_SAME_VALUE(WWRCCL_WIN_COLL_SYMMETRIC, NCCL_WIN_COLL_SYMMETRIC)
WWR_SAME_VALUE(WWRCCL_CTA_POLICY_DEFAULT, NCCL_CTA_POLICY_DEFAULT)
WWR_SAME_VALUE(WWRCCL_CTA_POLICY_EFFICIENCY, NCCL_CTA_POLICY_EFFICIENCY)
WWR_SAME_VALUE(WWRCCL_SHRINK_DEFAULT, NCCL_SHRINK_DEFAULT)
WWR_SAME_VALUE(WWRCCL_SHRINK_ABORT, NCCL_SHRINK_ABORT)

// The unique-id byte count sizes the ncclUniqueId payload on both backends.
static_assert(sizeof(wwrcclUniqueId) == WWRCCL_UNIQUE_ID_BYTES);

// ────────────────────────────────────────────────────────────────────────
// Functions
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_FUNCTION(wwrcclGetVersion, ncclGetVersion)
WWR_SAME_FUNCTION(wwrcclMemAlloc, ncclMemAlloc)
WWR_SAME_FUNCTION(wwrcclMemFree, ncclMemFree)
WWR_SAME_FUNCTION(wwrcclGetUniqueId, ncclGetUniqueId)
WWR_SAME_FUNCTION(wwrcclCommInitRank, ncclCommInitRank)
WWR_SAME_FUNCTION(wwrcclCommInitRankConfig, ncclCommInitRankConfig)
WWR_SAME_FUNCTION(wwrcclCommInitRankScalable, ncclCommInitRankScalable)
WWR_SAME_FUNCTION(wwrcclCommInitAll, ncclCommInitAll)
WWR_SAME_FUNCTION(wwrcclCommSplit, ncclCommSplit)
WWR_SAME_FUNCTION(wwrcclCommShrink, ncclCommShrink)
WWR_SAME_FUNCTION(wwrcclCommFinalize, ncclCommFinalize)
WWR_SAME_FUNCTION(wwrcclCommDestroy, ncclCommDestroy)
WWR_SAME_FUNCTION(wwrcclCommAbort, ncclCommAbort)
WWR_SAME_FUNCTION(wwrcclGetErrorString, ncclGetErrorString)
WWR_SAME_FUNCTION(wwrcclGetLastError, ncclGetLastError)
WWR_SAME_FUNCTION(wwrcclResetDebugInit, ncclResetDebugInit)
WWR_SAME_FUNCTION(wwrcclCommGetAsyncError, ncclCommGetAsyncError)
WWR_SAME_FUNCTION(wwrcclCommCount, ncclCommCount)
WWR_SAME_FUNCTION(wwrcclCommCuDevice, ncclCommCuDevice)
WWR_SAME_FUNCTION(wwrcclCommUserRank, ncclCommUserRank)
WWR_SAME_FUNCTION(wwrcclCommRegister, ncclCommRegister)
WWR_SAME_FUNCTION(wwrcclCommDeregister, ncclCommDeregister)
WWR_SAME_FUNCTION(wwrcclCommWindowRegister, ncclCommWindowRegister)
WWR_SAME_FUNCTION(wwrcclCommWindowDeregister, ncclCommWindowDeregister)
WWR_SAME_FUNCTION(wwrcclRedOpCreatePreMulSum, ncclRedOpCreatePreMulSum)
WWR_SAME_FUNCTION(wwrcclRedOpDestroy, ncclRedOpDestroy)
WWR_SAME_FUNCTION(wwrcclReduce, ncclReduce)
WWR_SAME_FUNCTION(wwrcclBcast, ncclBcast)
WWR_SAME_FUNCTION(wwrcclBroadcast, ncclBroadcast)
WWR_SAME_FUNCTION(wwrcclAllReduce, ncclAllReduce)
WWR_SAME_FUNCTION(wwrcclReduceScatter, ncclReduceScatter)
WWR_SAME_FUNCTION(wwrcclAllGather, ncclAllGather)
WWR_SAME_FUNCTION(wwrcclSend, ncclSend)
WWR_SAME_FUNCTION(wwrcclRecv, ncclRecv)
WWR_SAME_FUNCTION(wwrcclGroupStart, ncclGroupStart)
WWR_SAME_FUNCTION(wwrcclGroupEnd, ncclGroupEnd)
WWR_SAME_FUNCTION(wwrcclGroupSimulateEnd, ncclGroupSimulateEnd)

} // namespace wwr::test
