// nvml.cppm - Compile-time tests for gpumod.cuda.nvml

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.nvml;

import std;
import gpumod.cuda.nvml;

// ========================================================================
// Compile-time tests for gpumod.cuda.nvml
//
// The module is a pure re-export (using declarations).
// We verify at compile-time that:
//   1. The result enum type satisfies std::is_enum_v
//   2. Key enumerator values match the NVML-specified integer values
//   3. Handle types are pointer types
//   4. Link-time symbol resolution for key functions
// ========================================================================

namespace wwr::cuda::test {

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<nvmlReturn_t>);
static_assert(std::is_enum_v<nvmlEnableState_t>);
static_assert(std::is_enum_v<nvmlBrandType_t>);
static_assert(std::is_enum_v<nvmlTemperatureSensors_t>);
static_assert(std::is_enum_v<nvmlTemperatureThresholds_t>);
static_assert(std::is_enum_v<nvmlComputeMode_t>);
static_assert(std::is_enum_v<nvmlClockType_t>);
static_assert(std::is_enum_v<nvmlClockId_t>);
static_assert(std::is_enum_v<nvmlDriverModel_t>);
static_assert(std::is_enum_v<nvmlPstates_t>);
static_assert(std::is_enum_v<nvmlGpuOperationMode_t>);
static_assert(std::is_enum_v<nvmlInforomObject_t>);
static_assert(std::is_enum_v<nvmlMemoryErrorType_t>);
static_assert(std::is_enum_v<nvmlEccCounterType_t>);
static_assert(std::is_enum_v<nvmlMemoryLocation_t>);
static_assert(std::is_enum_v<nvmlPageRetirementCause_t>);
static_assert(std::is_enum_v<nvmlRestrictedAPI_t>);
static_assert(std::is_enum_v<nvmlGpuTopologyLevel_t>);
static_assert(std::is_enum_v<nvmlGpuP2PStatus_t>);
static_assert(std::is_enum_v<nvmlGpuP2PCapsIndex_t>);
static_assert(std::is_enum_v<nvmlSamplingType_t>);
static_assert(std::is_enum_v<nvmlPcieUtilCounter_t>);
static_assert(std::is_enum_v<nvmlValueType_t>);
static_assert(std::is_enum_v<nvmlPerfPolicyType_t>);
static_assert(std::is_enum_v<nvmlFanState_t>);
static_assert(std::is_enum_v<nvmlLedColor_t>);
static_assert(std::is_enum_v<nvmlEncoderType_t>);
static_assert(std::is_enum_v<nvmlFBCSessionType_t>);
static_assert(std::is_enum_v<nvmlDetachGpuState_t>);
static_assert(std::is_enum_v<nvmlPcieLinkState_t>);
static_assert(std::is_enum_v<nvmlGpuVirtualizationMode_t>);
static_assert(std::is_enum_v<nvmlHostVgpuMode_t>);
static_assert(std::is_enum_v<nvmlVgpuVmIdType_t>);
static_assert(std::is_enum_v<nvmlVgpuGuestInfoState_t>);
static_assert(std::is_enum_v<nvmlGridLicenseFeatureCode_t>);
static_assert(std::is_enum_v<nvmlVgpuCapability_t>);
static_assert(std::is_enum_v<nvmlVgpuDriverCapability_t>);
static_assert(std::is_enum_v<nvmlDeviceVgpuCapability_t>);
static_assert(std::is_enum_v<nvmlNvLinkCapability_t>);
static_assert(std::is_enum_v<nvmlNvLinkErrorCounter_t>);
static_assert(std::is_enum_v<nvmlIntNvLinkDeviceType_t>);
static_assert(std::is_enum_v<nvmlNvLinkUtilizationCountUnits_t>);
static_assert(std::is_enum_v<nvmlNvLinkUtilizationCountPktTypes_t>);
static_assert(std::is_enum_v<nvmlBridgeChipType_t>);
static_assert(std::is_enum_v<nvmlThermalTarget_t>);
static_assert(std::is_enum_v<nvmlThermalController_t>);
static_assert(std::is_enum_v<nvmlCoolerControl_t>);
static_assert(std::is_enum_v<nvmlDeviceGpuRecoveryAction_t>);
static_assert(std::is_enum_v<nvmlDeviceAddressingModeType_t>);
static_assert(std::is_enum_v<nvmlGpuUtilizationDomainId_t>);
static_assert(std::is_enum_v<nvmlUUIDType_t>);
static_assert(std::is_enum_v<nvmlVgpuPgpuCompatibilityLimitCode_t>);
static_assert(std::is_enum_v<nvmlNvlinkVersion_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvmlReturn_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVML_SUCCESS) == 0);
static_assert(static_cast<int>(NVML_ERROR_UNINITIALIZED) == 1);
static_assert(static_cast<int>(NVML_ERROR_INVALID_ARGUMENT) == 2);
static_assert(static_cast<int>(NVML_ERROR_NOT_SUPPORTED) == 3);
static_assert(static_cast<int>(NVML_ERROR_NO_PERMISSION) == 4);
static_assert(static_cast<int>(NVML_ERROR_ALREADY_INITIALIZED) == 5);
static_assert(static_cast<int>(NVML_ERROR_NOT_FOUND) == 6);
static_assert(static_cast<int>(NVML_ERROR_INSUFFICIENT_SIZE) == 7);
static_assert(static_cast<int>(NVML_ERROR_INSUFFICIENT_POWER) == 8);
static_assert(static_cast<int>(NVML_ERROR_DRIVER_NOT_LOADED) == 9);
static_assert(static_cast<int>(NVML_ERROR_TIMEOUT) == 10);
static_assert(static_cast<int>(NVML_ERROR_IRQ_ISSUE) == 11);
static_assert(static_cast<int>(NVML_ERROR_LIBRARY_NOT_FOUND) == 12);
static_assert(static_cast<int>(NVML_ERROR_FUNCTION_NOT_FOUND) == 13);
static_assert(static_cast<int>(NVML_ERROR_CORRUPTED_INFOROM) == 14);
static_assert(static_cast<int>(NVML_ERROR_GPU_IS_LOST) == 15);
static_assert(static_cast<int>(NVML_ERROR_RESET_REQUIRED) == 16);
static_assert(static_cast<int>(NVML_ERROR_OPERATING_SYSTEM) == 17);
static_assert(static_cast<int>(NVML_ERROR_LIB_RM_VERSION_MISMATCH) == 18);
static_assert(static_cast<int>(NVML_ERROR_IN_USE) == 19);
static_assert(static_cast<int>(NVML_ERROR_MEMORY) == 20);
static_assert(static_cast<int>(NVML_ERROR_NO_DATA) == 21);
static_assert(static_cast<int>(NVML_ERROR_VGPU_ECC_NOT_SUPPORTED) == 22);
static_assert(static_cast<int>(NVML_ERROR_INSUFFICIENT_RESOURCES) == 23);
static_assert(static_cast<int>(NVML_ERROR_FREQ_NOT_SUPPORTED) == 24);
static_assert(static_cast<int>(NVML_ERROR_ARGUMENT_VERSION_MISMATCH) == 25);
static_assert(static_cast<int>(NVML_ERROR_DEPRECATED) == 26);
static_assert(static_cast<int>(NVML_ERROR_NOT_READY) == 27);
static_assert(static_cast<int>(NVML_ERROR_GPU_NOT_FOUND) == 28);
static_assert(static_cast<int>(NVML_ERROR_INVALID_STATE) == 29);
static_assert(static_cast<int>(NVML_ERROR_RESET_TYPE_NOT_SUPPORTED) == 30);
static_assert(static_cast<int>(NVML_ERROR_UNKNOWN) == 999);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvmlEnableState_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVML_FEATURE_DISABLED) == 0);
static_assert(static_cast<int>(NVML_FEATURE_ENABLED) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvmlComputeMode_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVML_COMPUTEMODE_DEFAULT) == 0);
static_assert(static_cast<int>(NVML_COMPUTEMODE_EXCLUSIVE_THREAD) == 1);
static_assert(static_cast<int>(NVML_COMPUTEMODE_PROHIBITED) == 2);
static_assert(static_cast<int>(NVML_COMPUTEMODE_EXCLUSIVE_PROCESS) == 3);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvmlClockType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVML_CLOCK_GRAPHICS) == 0);
static_assert(static_cast<int>(NVML_CLOCK_SM) == 1);
static_assert(static_cast<int>(NVML_CLOCK_MEM) == 2);
static_assert(static_cast<int>(NVML_CLOCK_VIDEO) == 3);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvmlTemperatureSensors_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVML_TEMPERATURE_GPU) == 0);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvmlMemoryErrorType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVML_MEMORY_ERROR_TYPE_CORRECTED) == 0);
static_assert(static_cast<int>(NVML_MEMORY_ERROR_TYPE_UNCORRECTED) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvmlEccCounterType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVML_VOLATILE_ECC) == 0);
static_assert(static_cast<int>(NVML_AGGREGATE_ECC) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvmlGpuOperationMode_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVML_GOM_ALL_ON) == 0);
static_assert(static_cast<int>(NVML_GOM_COMPUTE) == 1);
static_assert(static_cast<int>(NVML_GOM_LOW_DP) == 2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvmlGpuVirtualizationMode_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVML_GPU_VIRTUALIZATION_MODE_NONE) == 0);
static_assert(static_cast<int>(NVML_GPU_VIRTUALIZATION_MODE_PASSTHROUGH) == 1);
static_assert(static_cast<int>(NVML_GPU_VIRTUALIZATION_MODE_VGPU) == 2);
static_assert(static_cast<int>(NVML_GPU_VIRTUALIZATION_MODE_HOST_VGPU) == 3);
static_assert(static_cast<int>(NVML_GPU_VIRTUALIZATION_MODE_HOST_VSGA) == 4);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvmlFanState_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVML_FAN_NORMAL) == 0);
static_assert(static_cast<int>(NVML_FAN_FAILED) == 1);

// ────────────────────────────────────────────────────────────────────────
// Handle type checks
// nvmlDevice_t and nvmlUnit_t etc. are typedef'd as pointers to opaque structs
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<nvmlDevice_t>);
static_assert(std::is_pointer_v<nvmlUnit_t>);
static_assert(std::is_pointer_v<nvmlEventSet_t>);
static_assert(std::is_pointer_v<nvmlGpuInstance_t>);
static_assert(std::is_pointer_v<nvmlComputeInstance_t>);
static_assert(std::is_pointer_v<nvmlSystemEventSet_t>);
static_assert(std::is_pointer_v<nvmlGpmSample_t>);

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// Forces the linker to resolve every re-exported function symbol,
// catching missing or unresolvable exports that type-only checks miss.
// ────────────────────────────────────────────────────────────────────────

// Initialization and shutdown
WWR_LINK_CHECK(nvmlInit_v2)
WWR_LINK_CHECK(nvmlInitWithFlags)
WWR_LINK_CHECK(nvmlShutdown)

// Error reporting
WWR_LINK_CHECK(nvmlErrorString)

// System queries
WWR_LINK_CHECK(nvmlSystemGetDriverVersion)
WWR_LINK_CHECK(nvmlSystemGetNVMLVersion)
WWR_LINK_CHECK(nvmlSystemGetCudaDriverVersion)
WWR_LINK_CHECK(nvmlSystemGetCudaDriverVersion_v2)
WWR_LINK_CHECK(nvmlSystemGetProcessName)
WWR_LINK_CHECK(nvmlSystemGetHicVersion)
WWR_LINK_CHECK(nvmlSystemGetTopologyGpuSet)
WWR_LINK_CHECK(nvmlSystemGetDriverBranch)

// Unit queries (S-class)
WWR_LINK_CHECK(nvmlUnitGetCount)
WWR_LINK_CHECK(nvmlUnitGetHandleByIndex)
WWR_LINK_CHECK(nvmlUnitGetUnitInfo)
WWR_LINK_CHECK(nvmlUnitGetLedState)
WWR_LINK_CHECK(nvmlUnitGetPsuInfo)
WWR_LINK_CHECK(nvmlUnitGetTemperature)
WWR_LINK_CHECK(nvmlUnitGetFanSpeedInfo)
WWR_LINK_CHECK(nvmlUnitGetDevices)

// Device enumeration
WWR_LINK_CHECK(nvmlDeviceGetCount_v2)
WWR_LINK_CHECK(nvmlDeviceGetHandleByIndex_v2)
WWR_LINK_CHECK(nvmlDeviceGetHandleByUUID)
WWR_LINK_CHECK(nvmlDeviceGetHandleByUUIDV)
WWR_LINK_CHECK(nvmlDeviceGetHandleByPciBusId_v2)

// Device info
WWR_LINK_CHECK(nvmlDeviceGetName)
WWR_LINK_CHECK(nvmlDeviceGetBrand)
WWR_LINK_CHECK(nvmlDeviceGetIndex)
WWR_LINK_CHECK(nvmlDeviceGetSerial)
WWR_LINK_CHECK(nvmlDeviceGetModuleId)
WWR_LINK_CHECK(nvmlDeviceGetUUID)
WWR_LINK_CHECK(nvmlDeviceGetMinorNumber)
WWR_LINK_CHECK(nvmlDeviceGetBoardPartNumber)
WWR_LINK_CHECK(nvmlDeviceGetInforomVersion)
WWR_LINK_CHECK(nvmlDeviceGetInforomImageVersion)
WWR_LINK_CHECK(nvmlDeviceGetInforomConfigurationChecksum)
WWR_LINK_CHECK(nvmlDeviceValidateInforom)
WWR_LINK_CHECK(nvmlDeviceGetBoardId)
WWR_LINK_CHECK(nvmlDeviceGetMultiGpuBoard)
WWR_LINK_CHECK(nvmlDeviceGetNumaNodeId)
WWR_LINK_CHECK(nvmlDeviceGetDisplayMode)
WWR_LINK_CHECK(nvmlDeviceGetDisplayActive)
WWR_LINK_CHECK(nvmlDeviceGetPersistenceMode)
WWR_LINK_CHECK(nvmlDeviceGetAttributes_v2)

// Device PCI info
WWR_LINK_CHECK(nvmlDeviceGetPciInfo_v3)
WWR_LINK_CHECK(nvmlDeviceGetPciInfoExt)
WWR_LINK_CHECK(nvmlDeviceGetMaxPcieLinkGeneration)
WWR_LINK_CHECK(nvmlDeviceGetMaxPcieLinkWidth)
WWR_LINK_CHECK(nvmlDeviceGetCurrPcieLinkGeneration)
WWR_LINK_CHECK(nvmlDeviceGetCurrPcieLinkWidth)
WWR_LINK_CHECK(nvmlDeviceGetPcieThroughput)
WWR_LINK_CHECK(nvmlDeviceGetPcieReplayCounter)

// Device affinity
WWR_LINK_CHECK(nvmlDeviceGetMemoryAffinity)
WWR_LINK_CHECK(nvmlDeviceGetCpuAffinityWithinScope)
WWR_LINK_CHECK(nvmlDeviceGetCpuAffinity)
WWR_LINK_CHECK(nvmlDeviceSetCpuAffinity)
WWR_LINK_CHECK(nvmlDeviceClearCpuAffinity)
WWR_LINK_CHECK(nvmlDeviceGetTopologyCommonAncestor)
WWR_LINK_CHECK(nvmlDeviceGetTopologyNearestGpus)
WWR_LINK_CHECK(nvmlDeviceGetP2PStatus)

// Device clock
WWR_LINK_CHECK(nvmlDeviceGetClockInfo)
WWR_LINK_CHECK(nvmlDeviceGetMaxClockInfo)
WWR_LINK_CHECK(nvmlDeviceGetClock)
WWR_LINK_CHECK(nvmlDeviceGetMaxCustomerBoostClock)
WWR_LINK_CHECK(nvmlDeviceGetSupportedMemoryClocks)
WWR_LINK_CHECK(nvmlDeviceGetSupportedGraphicsClocks)
WWR_LINK_CHECK(nvmlDeviceGetAutoBoostedClocksEnabled)

// Device temperature and fan
WWR_LINK_CHECK(nvmlDeviceGetTemperatureThreshold)
WWR_LINK_CHECK(nvmlDeviceGetMarginTemperature)
WWR_LINK_CHECK(nvmlDeviceGetThermalSettings)
WWR_LINK_CHECK(nvmlDeviceGetFanSpeed)
WWR_LINK_CHECK(nvmlDeviceGetFanSpeed_v2)
WWR_LINK_CHECK(nvmlDeviceGetFanSpeedRPM)
WWR_LINK_CHECK(nvmlDeviceGetTargetFanSpeed)
WWR_LINK_CHECK(nvmlDeviceGetNumFans)

// Device performance state
WWR_LINK_CHECK(nvmlDeviceGetPerformanceState)
WWR_LINK_CHECK(nvmlDeviceGetDynamicPstatesInfo)
WWR_LINK_CHECK(nvmlDeviceGetCurrentClocksEventReasons)
WWR_LINK_CHECK(nvmlDeviceGetSupportedClocksEventReasons)

// Device power
WWR_LINK_CHECK(nvmlDeviceGetPowerManagementLimit)
WWR_LINK_CHECK(nvmlDeviceGetPowerManagementLimitConstraints)
WWR_LINK_CHECK(nvmlDeviceGetPowerManagementDefaultLimit)
WWR_LINK_CHECK(nvmlDeviceGetPowerUsage)
WWR_LINK_CHECK(nvmlDeviceGetTotalEnergyConsumption)
WWR_LINK_CHECK(nvmlDeviceGetEnforcedPowerLimit)
WWR_LINK_CHECK(nvmlDeviceSetPowerManagementLimit)

// Device memory
WWR_LINK_CHECK(nvmlDeviceGetMemoryInfo)
WWR_LINK_CHECK(nvmlDeviceGetMemoryInfo_v2)
WWR_LINK_CHECK(nvmlDeviceGetBAR1MemoryInfo)

// Device utilization
WWR_LINK_CHECK(nvmlDeviceGetUtilizationRates)
WWR_LINK_CHECK(nvmlDeviceGetEncoderUtilization)
WWR_LINK_CHECK(nvmlDeviceGetDecoderUtilization)

// Device compute mode and ECC
WWR_LINK_CHECK(nvmlDeviceGetComputeMode)
WWR_LINK_CHECK(nvmlDeviceGetCudaComputeCapability)
WWR_LINK_CHECK(nvmlDeviceGetEccMode)
WWR_LINK_CHECK(nvmlDeviceGetDefaultEccMode)
WWR_LINK_CHECK(nvmlDeviceGetTotalEccErrors)
WWR_LINK_CHECK(nvmlDeviceGetMemoryErrorCounter)

// Device GPU operation
WWR_LINK_CHECK(nvmlDeviceGetGpuOperationMode)
WWR_LINK_CHECK(nvmlDeviceSetGpuOperationMode)
WWR_LINK_CHECK(nvmlDeviceGetAccountingMode)
WWR_LINK_CHECK(nvmlDeviceSetAccountingMode)
WWR_LINK_CHECK(nvmlDeviceClearAccountingPids)

// Device encoder/FBC
WWR_LINK_CHECK(nvmlDeviceGetEncoderCapacity)
WWR_LINK_CHECK(nvmlDeviceGetEncoderStats)
WWR_LINK_CHECK(nvmlDeviceGetEncoderSessions)
WWR_LINK_CHECK(nvmlDeviceGetFBCStats)
WWR_LINK_CHECK(nvmlDeviceGetFBCSessions)

// Device field values
WWR_LINK_CHECK(nvmlDeviceGetFieldValues)
WWR_LINK_CHECK(nvmlDeviceClearFieldValues)

// Device drain/reset
WWR_LINK_CHECK(nvmlDeviceModifyDrainState)
WWR_LINK_CHECK(nvmlDeviceQueryDrainState)
WWR_LINK_CHECK(nvmlDeviceRemoveGpu_v2)
WWR_LINK_CHECK(nvmlDeviceDiscoverGpus)
WWR_LINK_CHECK(nvmlDeviceSetGpuLockedClocks)
WWR_LINK_CHECK(nvmlDeviceResetGpuLockedClocks)
WWR_LINK_CHECK(nvmlDeviceSetMemoryLockedClocks)
WWR_LINK_CHECK(nvmlDeviceResetMemoryLockedClocks)
WWR_LINK_CHECK(nvmlDeviceSetAutoBoostedClocksEnabled)
WWR_LINK_CHECK(nvmlDeviceSetDefaultAutoBoostedClocksEnabled)

// Device NvLink
WWR_LINK_CHECK(nvmlDeviceGetNvLinkState)
WWR_LINK_CHECK(nvmlDeviceGetNvLinkVersion)
WWR_LINK_CHECK(nvmlDeviceGetNvLinkCapability)
WWR_LINK_CHECK(nvmlDeviceGetNvLinkRemotePciInfo_v2)
WWR_LINK_CHECK(nvmlDeviceGetNvLinkErrorCounter)
WWR_LINK_CHECK(nvmlDeviceResetNvLinkErrorCounters)
WWR_LINK_CHECK(nvmlDeviceGetNvLinkRemoteDeviceType)
WWR_LINK_CHECK(nvmlDeviceSetNvLinkDeviceLowPowerThreshold)
WWR_LINK_CHECK(nvmlDeviceGetNvLinkInfo)
WWR_LINK_CHECK(nvmlSystemSetNvlinkBwMode)
WWR_LINK_CHECK(nvmlSystemGetNvlinkBwMode)

// Events
WWR_LINK_CHECK(nvmlEventSetCreate)
WWR_LINK_CHECK(nvmlDeviceRegisterEvents)
WWR_LINK_CHECK(nvmlDeviceGetSupportedEventTypes)
WWR_LINK_CHECK(nvmlEventSetWait_v2)
WWR_LINK_CHECK(nvmlEventSetFree)
WWR_LINK_CHECK(nvmlSystemEventSetCreate)
WWR_LINK_CHECK(nvmlSystemEventSetFree)
WWR_LINK_CHECK(nvmlSystemRegisterEvents)
WWR_LINK_CHECK(nvmlSystemEventSetWait)

// Virtualization
WWR_LINK_CHECK(nvmlDeviceGetVirtualizationMode)
WWR_LINK_CHECK(nvmlDeviceGetHostVgpuMode)
WWR_LINK_CHECK(nvmlDeviceSetVirtualizationMode)
WWR_LINK_CHECK(nvmlDeviceGetGridLicensableFeatures_v4)
WWR_LINK_CHECK(nvmlGetVgpuDriverCapabilities)
WWR_LINK_CHECK(nvmlDeviceGetVgpuCapabilities)
WWR_LINK_CHECK(nvmlDeviceGetSupportedVgpus)
WWR_LINK_CHECK(nvmlDeviceGetCreatableVgpus)
WWR_LINK_CHECK(nvmlVgpuTypeGetClass)
WWR_LINK_CHECK(nvmlVgpuTypeGetName)
WWR_LINK_CHECK(nvmlVgpuTypeGetDeviceID)
WWR_LINK_CHECK(nvmlVgpuTypeGetFramebufferSize)
WWR_LINK_CHECK(nvmlVgpuTypeGetNumDisplayHeads)
WWR_LINK_CHECK(nvmlVgpuTypeGetResolution)
WWR_LINK_CHECK(nvmlVgpuTypeGetLicense)
WWR_LINK_CHECK(nvmlVgpuTypeGetFrameRateLimit)
WWR_LINK_CHECK(nvmlVgpuTypeGetMaxInstances)
WWR_LINK_CHECK(nvmlVgpuTypeGetMaxInstancesPerVm)
WWR_LINK_CHECK(nvmlDeviceGetActiveVgpus)
WWR_LINK_CHECK(nvmlVgpuInstanceGetVmID)
WWR_LINK_CHECK(nvmlVgpuInstanceGetUUID)
WWR_LINK_CHECK(nvmlVgpuInstanceGetVmDriverVersion)
WWR_LINK_CHECK(nvmlVgpuInstanceGetFbUsage)
WWR_LINK_CHECK(nvmlVgpuInstanceGetType)
WWR_LINK_CHECK(nvmlVgpuInstanceGetFrameRateLimit)
WWR_LINK_CHECK(nvmlVgpuInstanceGetEccMode)
WWR_LINK_CHECK(nvmlVgpuInstanceGetEncoderCapacity)
WWR_LINK_CHECK(nvmlVgpuInstanceSetEncoderCapacity)
WWR_LINK_CHECK(nvmlVgpuInstanceGetEncoderStats)
WWR_LINK_CHECK(nvmlVgpuInstanceGetEncoderSessions)
WWR_LINK_CHECK(nvmlVgpuInstanceGetFBCStats)
WWR_LINK_CHECK(nvmlVgpuInstanceGetFBCSessions)
WWR_LINK_CHECK(nvmlVgpuInstanceGetGpuInstanceId)
WWR_LINK_CHECK(nvmlVgpuInstanceGetGpuPciId)
WWR_LINK_CHECK(nvmlVgpuTypeGetCapabilities)
WWR_LINK_CHECK(nvmlVgpuInstanceGetMdevUUID)
WWR_LINK_CHECK(nvmlGetVgpuVersion)
WWR_LINK_CHECK(nvmlSetVgpuVersion)
WWR_LINK_CHECK(nvmlVgpuInstanceGetLicenseInfo_v2)

// Excluded devices
WWR_LINK_CHECK(nvmlGetExcludedDeviceCount)
WWR_LINK_CHECK(nvmlGetExcludedDeviceInfoByIndex)

// MIG
WWR_LINK_CHECK(nvmlDeviceSetMigMode)
WWR_LINK_CHECK(nvmlDeviceGetMigMode)
WWR_LINK_CHECK(nvmlDeviceGetGpuInstanceProfileInfo)
WWR_LINK_CHECK(nvmlDeviceGetGpuInstanceRemainingCapacity)
WWR_LINK_CHECK(nvmlDeviceCreateGpuInstance)
WWR_LINK_CHECK(nvmlGpuInstanceDestroy)
WWR_LINK_CHECK(nvmlDeviceGetGpuInstances)
WWR_LINK_CHECK(nvmlDeviceGetGpuInstanceById)
WWR_LINK_CHECK(nvmlGpuInstanceGetInfo)
WWR_LINK_CHECK(nvmlGpuInstanceGetComputeInstanceProfileInfo)
WWR_LINK_CHECK(nvmlGpuInstanceGetComputeInstanceRemainingCapacity)
WWR_LINK_CHECK(nvmlGpuInstanceCreateComputeInstance)
WWR_LINK_CHECK(nvmlComputeInstanceDestroy)
WWR_LINK_CHECK(nvmlGpuInstanceGetComputeInstances)
WWR_LINK_CHECK(nvmlGpuInstanceGetComputeInstanceById)
WWR_LINK_CHECK(nvmlComputeInstanceGetInfo_v2)
WWR_LINK_CHECK(nvmlDeviceIsMigDeviceHandle)
WWR_LINK_CHECK(nvmlDeviceGetGpuInstanceId)
WWR_LINK_CHECK(nvmlDeviceGetComputeInstanceId)
WWR_LINK_CHECK(nvmlDeviceGetMaxMigDeviceCount)
WWR_LINK_CHECK(nvmlDeviceGetMigDeviceHandleByIndex)
WWR_LINK_CHECK(nvmlDeviceGetDeviceHandleFromMigDeviceHandle)

// GPM
WWR_LINK_CHECK(nvmlGpmMetricsGet)
WWR_LINK_CHECK(nvmlGpmSampleFree)
WWR_LINK_CHECK(nvmlGpmSampleAlloc)
WWR_LINK_CHECK(nvmlGpmSampleGet)
WWR_LINK_CHECK(nvmlGpmMigSampleGet)
WWR_LINK_CHECK(nvmlGpmQueryDeviceSupport)
WWR_LINK_CHECK(nvmlGpmQueryIfStreamingEnabled)
WWR_LINK_CHECK(nvmlGpmSetStreamingEnabled)

} // namespace wwr::cuda::test
