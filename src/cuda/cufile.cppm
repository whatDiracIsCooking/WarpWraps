/**
 * @file cufile.cppm
 * @brief Primary interface for wwr.cuda.cufile
 *
 * This module wraps the cuFile (GPUDirect Storage) C API and exports types,
 * constants, and functions for high-performance direct I/O between GPU memory
 * and NVMe storage without staging through host memory.
 *
 * Usage:
 *   import wwr.cuda.cufile;
 */

module;

#include <cufile.h>

// ========================================================================
// cuFileDriverClose is renamed by a macro in the header:
//   #define cuFileDriverClose cuFileDriverClose_v2
// Undefine it so we can refer to the real versioned symbol by name below.
// ========================================================================
#undef cuFileDriverClose

// ========================================================================
// Validate call-site flag macros before #undef — these fire at parse time
// in the global module fragment, before any module-unit code is processed.
// ========================================================================

// Buffer registration flags (passed as `flags` to cuFileBufRegister)
static_assert(CU_FILE_RDMA_REGISTER == 1, "CU_FILE_RDMA_REGISTER value mismatch");
static_assert(CU_FILE_RDMA_RELAXED_ORDERING == (1 << 1),
              "CU_FILE_RDMA_RELAXED_ORDERING value mismatch");

// Stream registration flags (passed as `flags` to cuFileStreamRegister)
static_assert(CU_FILE_STREAM_FIXED_BUF_OFFSET == 1,
              "CU_FILE_STREAM_FIXED_BUF_OFFSET value mismatch");
static_assert(CU_FILE_STREAM_FIXED_FILE_OFFSET == 2,
              "CU_FILE_STREAM_FIXED_FILE_OFFSET value mismatch");
static_assert(CU_FILE_STREAM_FIXED_FILE_SIZE == 4, "CU_FILE_STREAM_FIXED_FILE_SIZE value mismatch");
static_assert(CU_FILE_STREAM_PAGE_ALIGNED_INPUTS == 8,
              "CU_FILE_STREAM_PAGE_ALIGNED_INPUTS value mismatch");

// GPU UUID buffer length (used to size the uuid field in CUfilePerGpuStats_t)
static_assert(CUFILE_GPU_UUID_LEN == 16, "CUFILE_GPU_UUID_LEN value mismatch");

// Undefine call-site flag macros so we can declare constexpr replacements.
#undef CU_FILE_RDMA_REGISTER
#undef CU_FILE_RDMA_RELAXED_ORDERING
#undef CU_FILE_STREAM_FIXED_BUF_OFFSET
#undef CU_FILE_STREAM_FIXED_FILE_OFFSET
#undef CU_FILE_STREAM_FIXED_FILE_SIZE
#undef CU_FILE_STREAM_PAGE_ALIGNED_INPUTS
#undef CUFILE_GPU_UUID_LEN

export module wwr.cuda.cufile;

// ========================================================================
// Export all cuFile types and functions in wwr namespace
// ========================================================================

export namespace wwr::cuda {

// ========================================================================
// Constexpr wrappers for cuFile call-site flag macros
// ========================================================================

// Buffer registration flags (passed as `flags` to cuFileBufRegister)
constexpr int CU_FILE_RDMA_REGISTER = 1;
constexpr int CU_FILE_RDMA_RELAXED_ORDERING = (1 << 1);

// Stream registration flags (passed as `flags` to cuFileStreamRegister)
constexpr unsigned CU_FILE_STREAM_FIXED_BUF_OFFSET = 1;
constexpr unsigned CU_FILE_STREAM_FIXED_FILE_OFFSET = 2;
constexpr unsigned CU_FILE_STREAM_FIXED_FILE_SIZE = 4;
constexpr unsigned CU_FILE_STREAM_PAGE_ALIGNED_INPUTS = 8;

// GPU UUID buffer length
constexpr int CUFILE_GPU_UUID_LEN = 16;

// ========================================================================
// Status / result enum
// ========================================================================
using ::CUfileOpError;

// CUfileOpError enumerators
using ::CU_FILE_ASYNC_NOT_SUPPORTED;
using ::CU_FILE_BATCH_FULL;
using ::CU_FILE_BATCH_NOCOMPAT_ERROR;
using ::CU_FILE_BATCH_SUBMIT_FAILED;
using ::CU_FILE_CUDA_CONTEXT_MISMATCH;
using ::CU_FILE_CUDA_DRIVER_ERROR;
using ::CU_FILE_CUDA_MEMORY_TYPE_INVALID;
using ::CU_FILE_CUDA_POINTER_INVALID;
using ::CU_FILE_CUDA_POINTER_RANGE_ERROR;
using ::CU_FILE_DEVICE_NOT_FOUND;
using ::CU_FILE_DEVICE_NOT_SUPPORTED;
using ::CU_FILE_DIO_NOT_SET;
using ::CU_FILE_DRIVER_ALREADY_OPEN;
using ::CU_FILE_DRIVER_CLOSING;
using ::CU_FILE_DRIVER_INVALID_PROPS;
using ::CU_FILE_DRIVER_NOT_INITIALIZED;
using ::CU_FILE_DRIVER_UNSUPPORTED_LIMIT;
using ::CU_FILE_DRIVER_VERSION_MISMATCH;
using ::CU_FILE_DRIVER_VERSION_READ_ERROR;
using ::CU_FILE_GETNEWFD_FAILED;
using ::CU_FILE_GPU_MEMORY_PINNING_FAILED;
using ::CU_FILE_HANDLE_ALREADY_REGISTERED;
using ::CU_FILE_HANDLE_NOT_REGISTERED;
using ::CU_FILE_HASH_OPR_ERROR;
using ::CU_FILE_INTERNAL_BATCH_CANCEL_ERROR;
using ::CU_FILE_INTERNAL_BATCH_GETSTATUS_ERROR;
using ::CU_FILE_INTERNAL_BATCH_SETUP_ERROR;
using ::CU_FILE_INTERNAL_BATCH_SUBMIT_ERROR;
using ::CU_FILE_INTERNAL_BUF_REGISTER_ERROR;
using ::CU_FILE_INTERNAL_ERROR;
using ::CU_FILE_INVALID_CONTEXT_ERROR;
using ::CU_FILE_INVALID_FILE_OPEN_FLAG;
using ::CU_FILE_INVALID_FILE_TYPE;
using ::CU_FILE_INVALID_MAPPING_RANGE;
using ::CU_FILE_INVALID_MAPPING_SIZE;
using ::CU_FILE_INVALID_VALUE;
using ::CU_FILE_IO_DISABLED;
using ::CU_FILE_IO_ERROR;
using ::CU_FILE_IO_MAX_ERROR;
using ::CU_FILE_IO_NOT_SUPPORTED;
using ::CU_FILE_MEMORY_ALREADY_REGISTERED;
using ::CU_FILE_MEMORY_NOT_REGISTERED;
using ::CU_FILE_NOMEM_ERROR;
using ::CU_FILE_NVFS_DRIVER_ERROR;
using ::CU_FILE_NVFS_INTERNAL_DRIVER_ERROR;
using ::CU_FILE_NVFS_SETUP_ERROR;
using ::CU_FILE_PERMISSION_DENIED;
using ::CU_FILE_PLATFORM_NOT_SUPPORTED;
using ::CU_FILE_SUCCESS;

// ========================================================================
// Error struct (bundles cuFile error + CUDA driver error)
// ========================================================================
using ::CUfileError_t;

// Inline helper: translate CUfileOpError to a human-readable string
// (cufileop_status_error is static inline in cufile.h and cannot be exported
// via a using-declaration; provide an exported inline wrapper instead)
inline const char *cufileop_status_error(CUfileOpError status) noexcept {
  return ::cufileop_status_error(status);
}

// ========================================================================
// Driver status flags enum
// ========================================================================
using ::CUfileDriverStatusFlags;
using ::CUfileDriverStatusFlags_t;

// CUfileDriverStatusFlags enumerators
using ::CU_FILE_BEEGFS_SUPPORTED;
using ::CU_FILE_GPFS_SUPPORTED;
using ::CU_FILE_LUSTRE_SUPPORTED;
using ::CU_FILE_NFS_SUPPORTED;
using ::CU_FILE_NVME_P2P_SUPPORTED;
using ::CU_FILE_NVME_SUPPORTED;
using ::CU_FILE_NVMEOF_SUPPORTED;
using ::CU_FILE_NVMESH_SUPPORTED;
using ::CU_FILE_SCALEFLUX_CSD_SUPPORTED;
using ::CU_FILE_SCATEFS_SUPPORTED;
using ::CU_FILE_SCSI_SUPPORTED;
using ::CU_FILE_WEKAFS_SUPPORTED;

// ========================================================================
// Driver control flags enum
// ========================================================================
using ::CUfileDriverControlFlags;
using ::CUfileDriverControlFlags_t;

// CUfileDriverControlFlags enumerators
using ::CU_FILE_ALLOW_COMPAT_MODE;
using ::CU_FILE_USE_POLL_MODE;

// ========================================================================
// Feature flags enum
// ========================================================================
using ::CUfileFeatureFlags;
using ::CUfileFeatureFlags_t;

// CUfileFeatureFlags enumerators
using ::CU_FILE_BATCH_IO_SUPPORTED;
using ::CU_FILE_DYN_ROUTING_SUPPORTED;
using ::CU_FILE_PARALLEL_IO_SUPPORTED;
using ::CU_FILE_STREAMS_SUPPORTED;

// ========================================================================
// Driver properties struct
// ========================================================================
using ::CUfileDrvProps_t;

// ========================================================================
// RDMA info struct
// ========================================================================
using ::cufileRDMAInfo_t;

// ========================================================================
// Userspace filesystem operations table
// ========================================================================
using ::CUfileFSOps_t;

// ========================================================================
// File handle type enum and descriptor
// ========================================================================
using ::CUfileFileHandleType;

// CUfileFileHandleType enumerators
using ::CU_FILE_HANDLE_TYPE_OPAQUE_FD;
using ::CU_FILE_HANDLE_TYPE_OPAQUE_WIN32;
using ::CU_FILE_HANDLE_TYPE_USERSPACE_FS;

using ::CUfileDescr_t;

// ========================================================================
// Opaque handle types
// ========================================================================
using ::CUfileBatchHandle_t;
using ::CUfileHandle_t;

// ========================================================================
// Batch I/O enums
// ========================================================================
using ::CUfileOpcode;
using ::CUfileOpcode_t;

// CUfileOpcode enumerators
using ::CUFILE_READ;
using ::CUFILE_WRITE;

using ::CUFILEStatus_enum;
using ::CUfileStatus_t;

// CUfileStatus_t enumerators
using ::CUFILE_CANCELED;
using ::CUFILE_COMPLETE;
using ::CUFILE_FAILED;
using ::CUFILE_INVALID;
using ::CUFILE_PENDING;
using ::CUFILE_TIMEOUT;
using ::CUFILE_WAITING;

using ::cufileBatchMode;
using ::CUfileBatchMode_t;

// cufileBatchMode enumerators
using ::CUFILE_BATCH;

// ========================================================================
// Batch I/O parameter and event structs
// ========================================================================
using ::CUfileIOEvents_t;
using ::CUfileIOParams_t;

// ========================================================================
// Configuration parameter enums
// ========================================================================
using ::CUFileSizeTConfigParameter_t;

// CUFileSizeTConfigParameter_t enumerators
using ::CUFILE_PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH;
using ::CUFILE_PARAM_EXECUTION_MAX_IO_THREADS;
using ::CUFILE_PARAM_EXECUTION_MAX_REQUEST_PARALLELISM;
using ::CUFILE_PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB;
using ::CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB;
using ::CUFILE_PARAM_PROFILE_STATS;
using ::CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS;
using ::CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE;
using ::CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB;
using ::CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB;
using ::CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB;
using ::CUFILE_PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB;

using ::CUFileBoolConfigParameter_t;

// CUFileBoolConfigParameter_t enumerators
using ::CUFILE_PARAM_EXECUTION_PARALLEL_IO;
using ::CUFILE_PARAM_FORCE_COMPAT_MODE;
using ::CUFILE_PARAM_FORCE_ODIRECT_MODE;
using ::CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE;
using ::CUFILE_PARAM_PREFER_IO_URING;
using ::CUFILE_PARAM_PROFILE_NVTX;
using ::CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE;
using ::CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY;
using ::CUFILE_PARAM_PROPERTIES_USE_POLL_MODE;
using ::CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION;
using ::CUFILE_PARAM_STREAM_MEMOPS_BYPASS;
using ::CUFILE_PARAM_USE_PCIP2PDMA;

using ::CUFileStringConfigParameter_t;

// CUFileStringConfigParameter_t enumerators
using ::CUFILE_PARAM_ENV_LOGFILE_PATH;
using ::CUFILE_PARAM_LOG_DIR;
using ::CUFILE_PARAM_LOGGING_LEVEL;

using ::CUFileArrayConfigParameter_t;

// CUFileArrayConfigParameter_t enumerators
using ::CUFILE_PARAM_POSIX_POOL_SLAB_COUNT;
using ::CUFILE_PARAM_POSIX_POOL_SLAB_SIZE_KB;

// ========================================================================
// Statistics structs
// ========================================================================
using ::CUfileOpCounter_t;
using ::CUfilePerGpuStats_t;
using ::CUfileStatsLevel1_t;
using ::CUfileStatsLevel2_t;
using ::CUfileStatsLevel3_t;

// ========================================================================
// File Handle Registration / Deregistration
// ========================================================================
using ::cuFileHandleDeregister;
using ::cuFileHandleRegister;

// ========================================================================
// Buffer Registration / Deregistration
// ========================================================================
using ::cuFileBufDeregister;
using ::cuFileBufRegister;

// ========================================================================
// Synchronous I/O
// ========================================================================
using ::cuFileRead;
using ::cuFileWrite;

// ========================================================================
// Driver Lifecycle
// ========================================================================
using ::cuFileDriverOpen;
// cuFileDriverClose is the versioned symbol cuFileDriverClose_v2 (the header
// uses a macro to rename it; the macro is #undef'd above so we bind to the
// real versioned symbol directly).
using ::cuFileDriverClose_v2;
using ::cuFileUseCount;

// ========================================================================
// Driver Property Functions
// ========================================================================
using ::cuFileDriverGetProperties;
using ::cuFileDriverSetMaxCacheSize;
using ::cuFileDriverSetMaxDirectIOSize;
using ::cuFileDriverSetMaxPinnedMemSize;
using ::cuFileDriverSetPollMode;

// ========================================================================
// Batch I/O Functions
// ========================================================================
using ::cuFileBatchIOCancel;
using ::cuFileBatchIODestroy;
using ::cuFileBatchIOGetStatus;
using ::cuFileBatchIOSetUp;
using ::cuFileBatchIOSubmit;

// ========================================================================
// Async (Stream-based) I/O Functions
// ========================================================================
using ::cuFileReadAsync;
using ::cuFileStreamDeregister;
using ::cuFileStreamRegister;
using ::cuFileWriteAsync;

// ========================================================================
// Version Query
// ========================================================================
using ::cuFileGetVersion;

// ========================================================================
// Configuration Get / Set Functions
// ========================================================================
using ::cuFileGetParameterBool;
using ::cuFileGetParameterMinMaxValue;
using ::cuFileGetParameterPosixPoolSlabArray;
using ::cuFileGetParameterSizeT;
using ::cuFileGetParameterString;
using ::cuFileSetParameterBool;
using ::cuFileSetParameterPosixPoolSlabArray;
using ::cuFileSetParameterSizeT;
using ::cuFileSetParameterString;

// ========================================================================
// Statistics Functions
// ========================================================================
using ::cuFileGetStatsL1;
using ::cuFileGetStatsL2;
using ::cuFileGetStatsL3;
using ::cuFileGetStatsLevel;
using ::cuFileSetStatsLevel;
using ::cuFileStatsReset;
using ::cuFileStatsStart;
using ::cuFileStatsStop;

// ========================================================================
// Hardware Query Functions
// ========================================================================
using ::cuFileGetBARSizeInKB;

} // namespace wwr::cuda
