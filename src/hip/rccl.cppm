/**
 * @file rccl.cppm
 * @brief RCCL API module wrapper for wwr project
 *
 * Wraps rccl/rccl.h -- the ROCm Communication Collectives Library, AMD's
 * source-compatible reimplementation of NVIDIA's NCCL. Every entity is spelled
 * with the same nccl* / NCCL_* names NCCL uses (RCCL's whole point), so this is
 * the HIP counterpart to wwr.cuda.nccl and the two agree name-for-name across
 * their shared surface. See src/ccl.cppm for the backend-neutral wwrccl* layer.
 *
 * RCCL carries a handful of names NCCL does not, exported here because a raw
 * module is a faithful 1:1 of its backend -- but ABSENT from the wwrccl* layer,
 * which spans only the measured intersection (src/ccl.cppm names them):
 *   - ncclGather / ncclScatter          (RCCL_GATHER_SCATTER)
 *   - ncclAllToAll / ncclAllToAllv      (RCCL_ALLTOALLV)
 *   - ncclAllReduceWithBias             (RCCL_ALLREDUCE_WITH_BIAS)
 *
 * Not wrapped: the msccl* algorithm entry points (mscclLoadAlgo/RunAlgo/
 * UnloadAlgo, mscclAlgoHandle_t) are marked @deprecated "removed from the public
 * API" in the header; the pnccl* profiling-interface duplicates are internal;
 * and the version / feature-test macros (NCCL_MAJOR/MINOR/PATCH, NCCL_SUFFIX,
 * NCCL_VERSION_CODE, NCCL_VERSION, RCCL_BFLOAT16, ...) are backend-specific
 * values -- ncclGetVersion is the queryable equivalent. The two struct
 * initializer macros (NCCL_CONFIG_INITIALIZER, NCCL_SIM_INFO_INITIALIZER) are
 * not exposed either: they embed a backend-specific size/magic/version and a
 * module cannot re-export a macro. See src/ccl.cppm.
 *
 * Usage:
 *   import wwr.hip.rccl;
 */

module;

#include <climits> // INT_MIN, for the NCCL_CONFIG_UNDEF_INT constexpr below
#include <rccl/rccl.h>

// ========================================================================
// Validate the scalar flag/param macros, then #undef so the export namespace
// can re-declare them as constexpr (a module cannot export a macro). These
// checks fire at parse time in the global module fragment. cf. cufft.cppm.
// ========================================================================

static_assert(NCCL_UNIQUE_ID_BYTES == 128, "NCCL_UNIQUE_ID_BYTES value mismatch");
static_assert(NCCL_SPLIT_NOCOLOR == -1, "NCCL_SPLIT_NOCOLOR value mismatch");
static_assert(NCCL_UNDEF_FLOAT == -1.0f, "NCCL_UNDEF_FLOAT value mismatch");
static_assert(NCCL_WIN_DEFAULT == 0x00, "NCCL_WIN_DEFAULT value mismatch");
static_assert(NCCL_WIN_COLL_SYMMETRIC == 0x01, "NCCL_WIN_COLL_SYMMETRIC value mismatch");
static_assert(NCCL_CTA_POLICY_DEFAULT == 0x00, "NCCL_CTA_POLICY_DEFAULT value mismatch");
static_assert(NCCL_CTA_POLICY_EFFICIENCY == 0x01, "NCCL_CTA_POLICY_EFFICIENCY value mismatch");
static_assert(NCCL_SHRINK_DEFAULT == 0x00, "NCCL_SHRINK_DEFAULT value mismatch");
static_assert(NCCL_SHRINK_ABORT == 0x01, "NCCL_SHRINK_ABORT value mismatch");

#undef NCCL_UNIQUE_ID_BYTES
#undef NCCL_COMM_NULL
#undef NCCL_SPLIT_NOCOLOR
#undef NCCL_UNDEF_FLOAT
#undef NCCL_CONFIG_UNDEF_INT
#undef NCCL_WIN_DEFAULT
#undef NCCL_WIN_COLL_SYMMETRIC
#undef NCCL_CTA_POLICY_DEFAULT
#undef NCCL_CTA_POLICY_EFFICIENCY
#undef NCCL_SHRINK_DEFAULT
#undef NCCL_SHRINK_ABORT

export module wwr.hip.rccl;

export namespace wwr::hip {

// ========================================================================
// Core Types
// ========================================================================
using ::ncclComm_t;
using ::ncclUniqueId;
using ::ncclWindow_t;

// Dependent type from <hip/hip_runtime.h> that appears in collective signatures
using ::hipStream_t;

// ========================================================================
// Result codes (ncclResult_t)
// ========================================================================
using ::ncclResult_t;
using ::ncclInProgress;
using ::ncclInternalError;
using ::ncclInvalidArgument;
using ::ncclInvalidUsage;
using ::ncclNumResults;
using ::ncclRemoteError;
using ::ncclSuccess;
using ::ncclSystemError;
using ::ncclUnhandledCudaError;

// ========================================================================
// Communicator configuration / simulation info
// ========================================================================
using ::ncclConfig_t;
using ::ncclSimInfo_t;

// ========================================================================
// Reduction operators (ncclRedOp_t) and the dummy enum sizing ncclMaxRedOp
// ========================================================================
using ::ncclRedOp_dummy_t;
using ::ncclNumOps_dummy;

using ::ncclRedOp_t;
using ::ncclAvg;
using ::ncclMax;
using ::ncclMaxRedOp;
using ::ncclMin;
using ::ncclNumOps;
using ::ncclProd;
using ::ncclSum;

// ========================================================================
// Data types (ncclDataType_t) -- several names share a value (ncclChar ==
// ncclInt8, ncclHalf == ncclFloat16, ...); all are exported.
// ========================================================================
using ::ncclDataType_t;
using ::ncclBfloat16;
using ::ncclChar;
using ::ncclDouble;
using ::ncclFloat;
using ::ncclFloat16;
using ::ncclFloat32;
using ::ncclFloat64;
using ::ncclFloat8e4m3;
using ::ncclFloat8e5m2;
using ::ncclHalf;
using ::ncclInt;
using ::ncclInt32;
using ::ncclInt64;
using ::ncclInt8;
using ::ncclNumTypes;
using ::ncclUint32;
using ::ncclUint64;
using ::ncclUint8;

// ========================================================================
// Custom reduction operator scalar residence (ncclScalarResidence_t)
// ========================================================================
using ::ncclScalarResidence_t;
using ::ncclScalarDevice;
using ::ncclScalarHostImmediate;

// ========================================================================
// Constexpr replacements for the scalar flag/param macros (see #undef above)
// ========================================================================
inline constexpr int NCCL_UNIQUE_ID_BYTES = 128;
inline constexpr ::ncclComm_t NCCL_COMM_NULL = nullptr;
inline constexpr int NCCL_SPLIT_NOCOLOR = -1;
inline constexpr float NCCL_UNDEF_FLOAT = -1.0f;
inline constexpr int NCCL_CONFIG_UNDEF_INT = INT_MIN;
inline constexpr int NCCL_WIN_DEFAULT = 0x00;
inline constexpr int NCCL_WIN_COLL_SYMMETRIC = 0x01;
inline constexpr int NCCL_CTA_POLICY_DEFAULT = 0x00;
inline constexpr int NCCL_CTA_POLICY_EFFICIENCY = 0x01;
inline constexpr int NCCL_SHRINK_DEFAULT = 0x00;
inline constexpr int NCCL_SHRINK_ABORT = 0x01;

// ========================================================================
// Version query
// ========================================================================
using ::ncclGetVersion;

// ========================================================================
// Buffer allocation
// ========================================================================
using ::ncclMemAlloc;
using ::ncclMemFree;

// ========================================================================
// Communicator lifecycle
// ========================================================================
using ::ncclCommAbort;
using ::ncclCommDestroy;
using ::ncclCommFinalize;
using ::ncclCommInitAll;
using ::ncclCommInitRank;
using ::ncclCommInitRankConfig;
using ::ncclCommInitRankScalable;
using ::ncclCommShrink;
using ::ncclCommSplit;
using ::ncclGetUniqueId;

// ========================================================================
// Error checking
// ========================================================================
using ::ncclCommGetAsyncError;
using ::ncclGetErrorString;
using ::ncclGetLastError;
using ::ncclResetDebugInit;

// ========================================================================
// Communicator information
// ========================================================================
using ::ncclCommCount;
using ::ncclCommCuDevice;
using ::ncclCommUserRank;

// ========================================================================
// Buffer / window registration
// ========================================================================
using ::ncclCommDeregister;
using ::ncclCommRegister;
using ::ncclCommWindowDeregister;
using ::ncclCommWindowRegister;

// ========================================================================
// Custom reduction operators
// ========================================================================
using ::ncclRedOpCreatePreMulSum;
using ::ncclRedOpDestroy;

// ========================================================================
// Collective communication
// ========================================================================
using ::ncclAllGather;
using ::ncclAllReduce;
using ::ncclBcast;
using ::ncclBroadcast;
using ::ncclRecv;
using ::ncclReduce;
using ::ncclReduceScatter;
using ::ncclSend;

// ========================================================================
// RCCL-only collectives (no NCCL counterpart -- not in the wwrccl* layer)
// ========================================================================
using ::ncclAllReduceWithBias;
using ::ncclAllToAll;
using ::ncclAllToAllv;
using ::ncclGather;
using ::ncclScatter;

// ========================================================================
// Group semantics
// ========================================================================
using ::ncclGroupEnd;
using ::ncclGroupSimulateEnd;
using ::ncclGroupStart;

} // namespace wwr::hip
