// cufile.cppm - Compile-time tests for gpumod.cuda.cufile

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.cufile;

import std;
import gpumod.cuda.cufile;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.cuda.cufile
//
// The module is a pure re-export (using declarations + constexpr flag values).
// We verify at compile-time that:
//   1. Constexpr call-site flags have the correct values
//   2. Key enum types satisfy std::is_enum_v
//   3. Key enumerator values match the cuFile-specified integer values
//   4. Struct/handle type traits (trivially copyable, standard layout, pointer)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::cuda::test {

using namespace gpumod::cuda;

// ────────────────────────────────────────────────────────────────────────
// Constexpr call-site flag values
// ────────────────────────────────────────────────────────────────────────

static_assert(CU_FILE_RDMA_REGISTER == 1);
static_assert(CU_FILE_RDMA_RELAXED_ORDERING == 2);

static_assert(CU_FILE_STREAM_FIXED_BUF_OFFSET == 1);
static_assert(CU_FILE_STREAM_FIXED_FILE_OFFSET == 2);
static_assert(CU_FILE_STREAM_FIXED_FILE_SIZE == 4);
static_assert(CU_FILE_STREAM_PAGE_ALIGNED_INPUTS == 8);

static_assert(CUFILE_GPU_UUID_LEN == 16);

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<CUfileOpError>);
static_assert(std::is_enum_v<CUfileDriverStatusFlags>);
static_assert(std::is_enum_v<CUfileDriverControlFlags>);
static_assert(std::is_enum_v<CUfileFeatureFlags>);
static_assert(std::is_enum_v<CUfileFileHandleType>);
static_assert(std::is_enum_v<CUfileOpcode>);
static_assert(std::is_enum_v<CUFILEStatus_enum>);
static_assert(std::is_enum_v<cufileBatchMode>);
static_assert(std::is_enum_v<CUFileSizeTConfigParameter_t>);
static_assert(std::is_enum_v<CUFileBoolConfigParameter_t>);
static_assert(std::is_enum_v<CUFileStringConfigParameter_t>);
static_assert(std::is_enum_v<CUFileArrayConfigParameter_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUfileOpError
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_FILE_SUCCESS) == 0);
static_assert(static_cast<int>(CU_FILE_DRIVER_NOT_INITIALIZED) == 5001);
static_assert(static_cast<int>(CU_FILE_DRIVER_INVALID_PROPS) == 5002);
static_assert(static_cast<int>(CU_FILE_DRIVER_UNSUPPORTED_LIMIT) == 5003);
static_assert(static_cast<int>(CU_FILE_DRIVER_VERSION_MISMATCH) == 5004);
static_assert(static_cast<int>(CU_FILE_DRIVER_VERSION_READ_ERROR) == 5005);
static_assert(static_cast<int>(CU_FILE_DRIVER_CLOSING) == 5006);
static_assert(static_cast<int>(CU_FILE_PLATFORM_NOT_SUPPORTED) == 5007);
static_assert(static_cast<int>(CU_FILE_IO_NOT_SUPPORTED) == 5008);
static_assert(static_cast<int>(CU_FILE_DEVICE_NOT_SUPPORTED) == 5009);
static_assert(static_cast<int>(CU_FILE_NVFS_DRIVER_ERROR) == 5010);
static_assert(static_cast<int>(CU_FILE_CUDA_DRIVER_ERROR) == 5011);
static_assert(static_cast<int>(CU_FILE_CUDA_POINTER_INVALID) == 5012);
static_assert(static_cast<int>(CU_FILE_CUDA_MEMORY_TYPE_INVALID) == 5013);
static_assert(static_cast<int>(CU_FILE_CUDA_POINTER_RANGE_ERROR) == 5014);
static_assert(static_cast<int>(CU_FILE_CUDA_CONTEXT_MISMATCH) == 5015);
static_assert(static_cast<int>(CU_FILE_INVALID_MAPPING_SIZE) == 5016);
static_assert(static_cast<int>(CU_FILE_INVALID_MAPPING_RANGE) == 5017);
static_assert(static_cast<int>(CU_FILE_INVALID_FILE_TYPE) == 5018);
static_assert(static_cast<int>(CU_FILE_INVALID_FILE_OPEN_FLAG) == 5019);
static_assert(static_cast<int>(CU_FILE_DIO_NOT_SET) == 5020);
static_assert(static_cast<int>(CU_FILE_INVALID_VALUE) == 5022);
static_assert(static_cast<int>(CU_FILE_MEMORY_ALREADY_REGISTERED) == 5023);
static_assert(static_cast<int>(CU_FILE_MEMORY_NOT_REGISTERED) == 5024);
static_assert(static_cast<int>(CU_FILE_PERMISSION_DENIED) == 5025);
static_assert(static_cast<int>(CU_FILE_DRIVER_ALREADY_OPEN) == 5026);
static_assert(static_cast<int>(CU_FILE_HANDLE_NOT_REGISTERED) == 5027);
static_assert(static_cast<int>(CU_FILE_HANDLE_ALREADY_REGISTERED) == 5028);
static_assert(static_cast<int>(CU_FILE_DEVICE_NOT_FOUND) == 5029);
static_assert(static_cast<int>(CU_FILE_INTERNAL_ERROR) == 5030);
static_assert(static_cast<int>(CU_FILE_GETNEWFD_FAILED) == 5031);
static_assert(static_cast<int>(CU_FILE_NVFS_SETUP_ERROR) == 5033);
static_assert(static_cast<int>(CU_FILE_IO_DISABLED) == 5034);
static_assert(static_cast<int>(CU_FILE_BATCH_SUBMIT_FAILED) == 5035);
static_assert(static_cast<int>(CU_FILE_GPU_MEMORY_PINNING_FAILED) == 5036);
static_assert(static_cast<int>(CU_FILE_BATCH_FULL) == 5037);
static_assert(static_cast<int>(CU_FILE_ASYNC_NOT_SUPPORTED) == 5038);
static_assert(static_cast<int>(CU_FILE_INTERNAL_BATCH_SETUP_ERROR) == 5039);
static_assert(static_cast<int>(CU_FILE_INTERNAL_BATCH_SUBMIT_ERROR) == 5040);
static_assert(static_cast<int>(CU_FILE_INTERNAL_BATCH_GETSTATUS_ERROR) == 5041);
static_assert(static_cast<int>(CU_FILE_INTERNAL_BATCH_CANCEL_ERROR) == 5042);
static_assert(static_cast<int>(CU_FILE_NOMEM_ERROR) == 5043);
static_assert(static_cast<int>(CU_FILE_IO_ERROR) == 5044);
static_assert(static_cast<int>(CU_FILE_INTERNAL_BUF_REGISTER_ERROR) == 5045);
static_assert(static_cast<int>(CU_FILE_HASH_OPR_ERROR) == 5046);
static_assert(static_cast<int>(CU_FILE_INVALID_CONTEXT_ERROR) == 5047);
static_assert(static_cast<int>(CU_FILE_NVFS_INTERNAL_DRIVER_ERROR) == 5048);
static_assert(static_cast<int>(CU_FILE_BATCH_NOCOMPAT_ERROR) == 5049);
static_assert(static_cast<int>(CU_FILE_IO_MAX_ERROR) == 5050);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUfileDriverStatusFlags
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_FILE_LUSTRE_SUPPORTED) == 0);
static_assert(static_cast<int>(CU_FILE_WEKAFS_SUPPORTED) == 1);
static_assert(static_cast<int>(CU_FILE_NFS_SUPPORTED) == 2);
static_assert(static_cast<int>(CU_FILE_GPFS_SUPPORTED) == 3);
static_assert(static_cast<int>(CU_FILE_NVME_SUPPORTED) == 4);
static_assert(static_cast<int>(CU_FILE_NVMEOF_SUPPORTED) == 5);
static_assert(static_cast<int>(CU_FILE_SCSI_SUPPORTED) == 6);
static_assert(static_cast<int>(CU_FILE_SCALEFLUX_CSD_SUPPORTED) == 7);
static_assert(static_cast<int>(CU_FILE_NVMESH_SUPPORTED) == 8);
static_assert(static_cast<int>(CU_FILE_BEEGFS_SUPPORTED) == 9);
static_assert(static_cast<int>(CU_FILE_NVME_P2P_SUPPORTED) == 11);
static_assert(static_cast<int>(CU_FILE_SCATEFS_SUPPORTED) == 12);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUfileDriverControlFlags
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_FILE_USE_POLL_MODE) == 0);
static_assert(static_cast<int>(CU_FILE_ALLOW_COMPAT_MODE) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUfileFeatureFlags
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_FILE_DYN_ROUTING_SUPPORTED) == 0);
static_assert(static_cast<int>(CU_FILE_BATCH_IO_SUPPORTED) == 1);
static_assert(static_cast<int>(CU_FILE_STREAMS_SUPPORTED) == 2);
static_assert(static_cast<int>(CU_FILE_PARALLEL_IO_SUPPORTED) == 3);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUfileFileHandleType
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_FILE_HANDLE_TYPE_OPAQUE_FD) == 1);
static_assert(static_cast<int>(CU_FILE_HANDLE_TYPE_OPAQUE_WIN32) == 2);
static_assert(static_cast<int>(CU_FILE_HANDLE_TYPE_USERSPACE_FS) == 3);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUfileOpcode
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUFILE_READ) == 0);
static_assert(static_cast<int>(CUFILE_WRITE) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUfileStatus_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUFILE_WAITING) == 0x000001);
static_assert(static_cast<int>(CUFILE_PENDING) == 0x000002);
static_assert(static_cast<int>(CUFILE_INVALID) == 0x000004);
static_assert(static_cast<int>(CUFILE_CANCELED) == 0x000008);
static_assert(static_cast<int>(CUFILE_COMPLETE) == 0x0000010);
static_assert(static_cast<int>(CUFILE_TIMEOUT) == 0x0000020);
static_assert(static_cast<int>(CUFILE_FAILED) == 0x0000040);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUfileBatchMode_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUFILE_BATCH) == 1);

// ────────────────────────────────────────────────────────────────────────
// Handle type checks
// CUfileHandle_t and CUfileBatchHandle_t are both typedef'd as void*
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<CUfileHandle_t>);
static_assert(std::is_same_v<CUfileHandle_t, void *>);

static_assert(std::is_pointer_v<CUfileBatchHandle_t>);
static_assert(std::is_same_v<CUfileBatchHandle_t, void *>);

// ────────────────────────────────────────────────────────────────────────
// Struct type traits
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_standard_layout_v<CUfileError_t>);
static_assert(std::is_trivially_copyable_v<CUfileError_t>);

static_assert(std::is_standard_layout_v<CUfileDrvProps_t>);
static_assert(std::is_trivially_copyable_v<CUfileDrvProps_t>);

static_assert(std::is_standard_layout_v<CUfileIOParams_t>);
static_assert(std::is_trivially_copyable_v<CUfileIOParams_t>);

static_assert(std::is_standard_layout_v<CUfileIOEvents_t>);
static_assert(std::is_trivially_copyable_v<CUfileIOEvents_t>);

static_assert(std::is_standard_layout_v<CUfileOpCounter_t>);
static_assert(std::is_trivially_copyable_v<CUfileOpCounter_t>);

static_assert(std::is_standard_layout_v<CUfileStatsLevel1_t>);
static_assert(std::is_trivially_copyable_v<CUfileStatsLevel1_t>);

static_assert(std::is_standard_layout_v<CUfileStatsLevel2_t>);
static_assert(std::is_trivially_copyable_v<CUfileStatsLevel2_t>);

static_assert(std::is_standard_layout_v<CUfilePerGpuStats_t>);
static_assert(std::is_trivially_copyable_v<CUfilePerGpuStats_t>);

static_assert(std::is_standard_layout_v<CUfileStatsLevel3_t>);
static_assert(std::is_trivially_copyable_v<CUfileStatsLevel3_t>);

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// Forces the linker to resolve every re-exported function symbol,
// catching missing or unresolvable exports that type-only checks miss.
// ────────────────────────────────────────────────────────────────────────

// File Handle Registration / Deregistration
GPUMOD_LINK_CHECK(cuFileHandleRegister)
GPUMOD_LINK_CHECK(cuFileHandleDeregister)

// Buffer Registration / Deregistration
GPUMOD_LINK_CHECK(cuFileBufRegister)
GPUMOD_LINK_CHECK(cuFileBufDeregister)

// Synchronous I/O
GPUMOD_LINK_CHECK(cuFileRead)
GPUMOD_LINK_CHECK(cuFileWrite)

// Driver Lifecycle
GPUMOD_LINK_CHECK(cuFileDriverOpen)
// cuFileDriverClose is exported as cuFileDriverClose_v2 (the header macro renames it)
GPUMOD_LINK_CHECK(cuFileDriverClose_v2)
GPUMOD_LINK_CHECK(cuFileUseCount)

// Driver Property Functions
GPUMOD_LINK_CHECK(cuFileDriverGetProperties)
GPUMOD_LINK_CHECK(cuFileDriverSetPollMode)
GPUMOD_LINK_CHECK(cuFileDriverSetMaxDirectIOSize)
GPUMOD_LINK_CHECK(cuFileDriverSetMaxCacheSize)
GPUMOD_LINK_CHECK(cuFileDriverSetMaxPinnedMemSize)

// Batch I/O Functions
GPUMOD_LINK_CHECK(cuFileBatchIOSetUp)
GPUMOD_LINK_CHECK(cuFileBatchIOSubmit)
GPUMOD_LINK_CHECK(cuFileBatchIOGetStatus)
GPUMOD_LINK_CHECK(cuFileBatchIOCancel)
GPUMOD_LINK_CHECK(cuFileBatchIODestroy)

// Async (Stream-based) I/O Functions
GPUMOD_LINK_CHECK(cuFileReadAsync)
GPUMOD_LINK_CHECK(cuFileWriteAsync)
GPUMOD_LINK_CHECK(cuFileStreamRegister)
GPUMOD_LINK_CHECK(cuFileStreamDeregister)

// Version Query
GPUMOD_LINK_CHECK(cuFileGetVersion)

// Configuration Get / Set Functions
GPUMOD_LINK_CHECK(cuFileGetParameterSizeT)
GPUMOD_LINK_CHECK(cuFileGetParameterBool)
GPUMOD_LINK_CHECK(cuFileGetParameterString)
GPUMOD_LINK_CHECK(cuFileGetParameterMinMaxValue)
GPUMOD_LINK_CHECK(cuFileSetParameterSizeT)
GPUMOD_LINK_CHECK(cuFileSetParameterBool)
GPUMOD_LINK_CHECK(cuFileSetParameterString)
GPUMOD_LINK_CHECK(cuFileSetParameterPosixPoolSlabArray)
GPUMOD_LINK_CHECK(cuFileGetParameterPosixPoolSlabArray)

// Statistics Functions
GPUMOD_LINK_CHECK(cuFileSetStatsLevel)
GPUMOD_LINK_CHECK(cuFileGetStatsLevel)
GPUMOD_LINK_CHECK(cuFileStatsStart)
GPUMOD_LINK_CHECK(cuFileStatsStop)
GPUMOD_LINK_CHECK(cuFileStatsReset)
GPUMOD_LINK_CHECK(cuFileGetStatsL1)
GPUMOD_LINK_CHECK(cuFileGetStatsL2)
GPUMOD_LINK_CHECK(cuFileGetStatsL3)

// Hardware Query Functions
GPUMOD_LINK_CHECK(cuFileGetBARSizeInKB)

} // namespace gpumod::cuda::test
