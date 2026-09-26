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

namespace gpumod::cuda::test {

using namespace gpumod::cuda;

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
GPUMOD_LINK_CHECK(nvmlInit_v2)
GPUMOD_LINK_CHECK(nvmlInitWithFlags)
GPUMOD_LINK_CHECK(nvmlShutdown)

// Error reporting
GPUMOD_LINK_CHECK(nvmlErrorString)

// System queries
GPUMOD_LINK_CHECK(nvmlSystemGetDriverVersion)
GPUMOD_LINK_CHECK(nvmlSystemGetNVMLVersion)
GPUMOD_LINK_CHECK(nvmlSystemGetCudaDriverVersion)
GPUMOD_LINK_CHECK(nvmlSystemGetCudaDriverVersion_v2)
GPUMOD_LINK_CHECK(nvmlSystemGetProcessName)
GPUMOD_LINK_CHECK(nvmlSystemGetHicVersion)
GPUMOD_LINK_CHECK(nvmlSystemGetTopologyGpuSet)
GPUMOD_LINK_CHECK(nvmlSystemGetDriverBranch)

// Unit queries (S-class)
GPUMOD_LINK_CHECK(nvmlUnitGetCount)
GPUMOD_LINK_CHECK(nvmlUnitGetHandleByIndex)
GPUMOD_LINK_CHECK(nvmlUnitGetUnitInfo)
GPUMOD_LINK_CHECK(nvmlUnitGetLedState)
GPUMOD_LINK_CHECK(nvmlUnitGetPsuInfo)
GPUMOD_LINK_CHECK(nvmlUnitGetTemperature)
GPUMOD_LINK_CHECK(nvmlUnitGetFanSpeedInfo)
GPUMOD_LINK_CHECK(nvmlUnitGetDevices)

// Device enumeration
GPUMOD_LINK_CHECK(nvmlDeviceGetCount_v2)
GPUMOD_LINK_CHECK(nvmlDeviceGetHandleByIndex_v2)
GPUMOD_LINK_CHECK(nvmlDeviceGetHandleByUUID)
GPUMOD_LINK_CHECK(nvmlDeviceGetHandleByUUIDV)
GPUMOD_LINK_CHECK(nvmlDeviceGetHandleByPciBusId_v2)

// Device info
GPUMOD_LINK_CHECK(nvmlDeviceGetName)
GPUMOD_LINK_CHECK(nvmlDeviceGetBrand)
GPUMOD_LINK_CHECK(nvmlDeviceGetIndex)
GPUMOD_LINK_CHECK(nvmlDeviceGetSerial)
GPUMOD_LINK_CHECK(nvmlDeviceGetModuleId)
GPUMOD_LINK_CHECK(nvmlDeviceGetUUID)
GPUMOD_LINK_CHECK(nvmlDeviceGetMinorNumber)
GPUMOD_LINK_CHECK(nvmlDeviceGetBoardPartNumber)
GPUMOD_LINK_CHECK(nvmlDeviceGetInforomVersion)
GPUMOD_LINK_CHECK(nvmlDeviceGetInforomImageVersion)
GPUMOD_LINK_CHECK(nvmlDeviceGetInforomConfigurationChecksum)
GPUMOD_LINK_CHECK(nvmlDeviceValidateInforom)
GPUMOD_LINK_CHECK(nvmlDeviceGetBoardId)
GPUMOD_LINK_CHECK(nvmlDeviceGetMultiGpuBoard)
GPUMOD_LINK_CHECK(nvmlDeviceGetNumaNodeId)
GPUMOD_LINK_CHECK(nvmlDeviceGetDisplayMode)
GPUMOD_LINK_CHECK(nvmlDeviceGetDisplayActive)
GPUMOD_LINK_CHECK(nvmlDeviceGetPersistenceMode)
GPUMOD_LINK_CHECK(nvmlDeviceGetAttributes_v2)

// Device PCI info
GPUMOD_LINK_CHECK(nvmlDeviceGetPciInfo_v3)
GPUMOD_LINK_CHECK(nvmlDeviceGetPciInfoExt)
GPUMOD_LINK_CHECK(nvmlDeviceGetMaxPcieLinkGeneration)
GPUMOD_LINK_CHECK(nvmlDeviceGetMaxPcieLinkWidth)
GPUMOD_LINK_CHECK(nvmlDeviceGetCurrPcieLinkGeneration)
GPUMOD_LINK_CHECK(nvmlDeviceGetCurrPcieLinkWidth)
GPUMOD_LINK_CHECK(nvmlDeviceGetPcieThroughput)
GPUMOD_LINK_CHECK(nvmlDeviceGetPcieReplayCounter)

// Device affinity
GPUMOD_LINK_CHECK(nvmlDeviceGetMemoryAffinity)
GPUMOD_LINK_CHECK(nvmlDeviceGetCpuAffinityWithinScope)
GPUMOD_LINK_CHECK(nvmlDeviceGetCpuAffinity)
GPUMOD_LINK_CHECK(nvmlDeviceSetCpuAffinity)
GPUMOD_LINK_CHECK(nvmlDeviceClearCpuAffinity)
GPUMOD_LINK_CHECK(nvmlDeviceGetTopologyCommonAncestor)
GPUMOD_LINK_CHECK(nvmlDeviceGetTopologyNearestGpus)
GPUMOD_LINK_CHECK(nvmlDeviceGetP2PStatus)

// Device clock
GPUMOD_LINK_CHECK(nvmlDeviceGetClockInfo)
GPUMOD_LINK_CHECK(nvmlDeviceGetMaxClockInfo)
GPUMOD_LINK_CHECK(nvmlDeviceGetClock)
GPUMOD_LINK_CHECK(nvmlDeviceGetMaxCustomerBoostClock)
GPUMOD_LINK_CHECK(nvmlDeviceGetSupportedMemoryClocks)
GPUMOD_LINK_CHECK(nvmlDeviceGetSupportedGraphicsClocks)
GPUMOD_LINK_CHECK(nvmlDeviceGetAutoBoostedClocksEnabled)

// Device temperature and fan
GPUMOD_LINK_CHECK(nvmlDeviceGetTemperatureThreshold)
GPUMOD_LINK_CHECK(nvmlDeviceGetMarginTemperature)
GPUMOD_LINK_CHECK(nvmlDeviceGetThermalSettings)
GPUMOD_LINK_CHECK(nvmlDeviceGetFanSpeed)
GPUMOD_LINK_CHECK(nvmlDeviceGetFanSpeed_v2)
GPUMOD_LINK_CHECK(nvmlDeviceGetFanSpeedRPM)
GPUMOD_LINK_CHECK(nvmlDeviceGetTargetFanSpeed)
GPUMOD_LINK_CHECK(nvmlDeviceGetNumFans)

// Device performance state
GPUMOD_LINK_CHECK(nvmlDeviceGetPerformanceState)
GPUMOD_LINK_CHECK(nvmlDeviceGetDynamicPstatesInfo)
GPUMOD_LINK_CHECK(nvmlDeviceGetCurrentClocksEventReasons)
GPUMOD_LINK_CHECK(nvmlDeviceGetSupportedClocksEventReasons)

// Device power
GPUMOD_LINK_CHECK(nvmlDeviceGetPowerManagementLimit)
GPUMOD_LINK_CHECK(nvmlDeviceGetPowerManagementLimitConstraints)
GPUMOD_LINK_CHECK(nvmlDeviceGetPowerManagementDefaultLimit)
GPUMOD_LINK_CHECK(nvmlDeviceGetPowerUsage)
GPUMOD_LINK_CHECK(nvmlDeviceGetTotalEnergyConsumption)
GPUMOD_LINK_CHECK(nvmlDeviceGetEnforcedPowerLimit)
GPUMOD_LINK_CHECK(nvmlDeviceSetPowerManagementLimit)

// Device memory
GPUMOD_LINK_CHECK(nvmlDeviceGetMemoryInfo)
GPUMOD_LINK_CHECK(nvmlDeviceGetMemoryInfo_v2)
GPUMOD_LINK_CHECK(nvmlDeviceGetBAR1MemoryInfo)

// Device utilization
GPUMOD_LINK_CHECK(nvmlDeviceGetUtilizationRates)
GPUMOD_LINK_CHECK(nvmlDeviceGetEncoderUtilization)
GPUMOD_LINK_CHECK(nvmlDeviceGetDecoderUtilization)

// Device compute mode and ECC
GPUMOD_LINK_CHECK(nvmlDeviceGetComputeMode)
GPUMOD_LINK_CHECK(nvmlDeviceGetCudaComputeCapability)
GPUMOD_LINK_CHECK(nvmlDeviceGetEccMode)
GPUMOD_LINK_CHECK(nvmlDeviceGetDefaultEccMode)
GPUMOD_LINK_CHECK(nvmlDeviceGetTotalEccErrors)
GPUMOD_LINK_CHECK(nvmlDeviceGetMemoryErrorCounter)

// Device GPU operation
GPUMOD_LINK_CHECK(nvmlDeviceGetGpuOperationMode)
GPUMOD_LINK_CHECK(nvmlDeviceSetGpuOperationMode)
GPUMOD_LINK_CHECK(nvmlDeviceGetAccountingMode)
GPUMOD_LINK_CHECK(nvmlDeviceSetAccountingMode)
GPUMOD_LINK_CHECK(nvmlDeviceClearAccountingPids)

// Device encoder/FBC
GPUMOD_LINK_CHECK(nvmlDeviceGetEncoderCapacity)
GPUMOD_LINK_CHECK(nvmlDeviceGetEncoderStats)
GPUMOD_LINK_CHECK(nvmlDeviceGetEncoderSessions)
GPUMOD_LINK_CHECK(nvmlDeviceGetFBCStats)
GPUMOD_LINK_CHECK(nvmlDeviceGetFBCSessions)

// Device field values
GPUMOD_LINK_CHECK(nvmlDeviceGetFieldValues)
GPUMOD_LINK_CHECK(nvmlDeviceClearFieldValues)

// Device drain/reset
GPUMOD_LINK_CHECK(nvmlDeviceModifyDrainState)
GPUMOD_LINK_CHECK(nvmlDeviceQueryDrainState)
GPUMOD_LINK_CHECK(nvmlDeviceRemoveGpu_v2)
GPUMOD_LINK_CHECK(nvmlDeviceDiscoverGpus)
GPUMOD_LINK_CHECK(nvmlDeviceSetGpuLockedClocks)
GPUMOD_LINK_CHECK(nvmlDeviceResetGpuLockedClocks)
GPUMOD_LINK_CHECK(nvmlDeviceSetMemoryLockedClocks)
GPUMOD_LINK_CHECK(nvmlDeviceResetMemoryLockedClocks)
GPUMOD_LINK_CHECK(nvmlDeviceSetAutoBoostedClocksEnabled)
GPUMOD_LINK_CHECK(nvmlDeviceSetDefaultAutoBoostedClocksEnabled)

// Device NvLink
GPUMOD_LINK_CHECK(nvmlDeviceGetNvLinkState)
GPUMOD_LINK_CHECK(nvmlDeviceGetNvLinkVersion)
GPUMOD_LINK_CHECK(nvmlDeviceGetNvLinkCapability)
GPUMOD_LINK_CHECK(nvmlDeviceGetNvLinkRemotePciInfo_v2)
GPUMOD_LINK_CHECK(nvmlDeviceGetNvLinkErrorCounter)
GPUMOD_LINK_CHECK(nvmlDeviceResetNvLinkErrorCounters)
GPUMOD_LINK_CHECK(nvmlDeviceGetNvLinkRemoteDeviceType)
GPUMOD_LINK_CHECK(nvmlDeviceSetNvLinkDeviceLowPowerThreshold)
GPUMOD_LINK_CHECK(nvmlDeviceGetNvLinkInfo)
GPUMOD_LINK_CHECK(nvmlSystemSetNvlinkBwMode)
GPUMOD_LINK_CHECK(nvmlSystemGetNvlinkBwMode)

// Events
GPUMOD_LINK_CHECK(nvmlEventSetCreate)
GPUMOD_LINK_CHECK(nvmlDeviceRegisterEvents)
GPUMOD_LINK_CHECK(nvmlDeviceGetSupportedEventTypes)
GPUMOD_LINK_CHECK(nvmlEventSetWait_v2)
GPUMOD_LINK_CHECK(nvmlEventSetFree)
GPUMOD_LINK_CHECK(nvmlSystemEventSetCreate)
GPUMOD_LINK_CHECK(nvmlSystemEventSetFree)
GPUMOD_LINK_CHECK(nvmlSystemRegisterEvents)
GPUMOD_LINK_CHECK(nvmlSystemEventSetWait)

// Virtualization
GPUMOD_LINK_CHECK(nvmlDeviceGetVirtualizationMode)
GPUMOD_LINK_CHECK(nvmlDeviceGetHostVgpuMode)
GPUMOD_LINK_CHECK(nvmlDeviceSetVirtualizationMode)
GPUMOD_LINK_CHECK(nvmlDeviceGetGridLicensableFeatures_v4)
GPUMOD_LINK_CHECK(nvmlGetVgpuDriverCapabilities)
GPUMOD_LINK_CHECK(nvmlDeviceGetVgpuCapabilities)
GPUMOD_LINK_CHECK(nvmlDeviceGetSupportedVgpus)
GPUMOD_LINK_CHECK(nvmlDeviceGetCreatableVgpus)
GPUMOD_LINK_CHECK(nvmlVgpuTypeGetClass)
GPUMOD_LINK_CHECK(nvmlVgpuTypeGetName)
GPUMOD_LINK_CHECK(nvmlVgpuTypeGetDeviceID)
GPUMOD_LINK_CHECK(nvmlVgpuTypeGetFramebufferSize)
GPUMOD_LINK_CHECK(nvmlVgpuTypeGetNumDisplayHeads)
GPUMOD_LINK_CHECK(nvmlVgpuTypeGetResolution)
GPUMOD_LINK_CHECK(nvmlVgpuTypeGetLicense)
GPUMOD_LINK_CHECK(nvmlVgpuTypeGetFrameRateLimit)
GPUMOD_LINK_CHECK(nvmlVgpuTypeGetMaxInstances)
GPUMOD_LINK_CHECK(nvmlVgpuTypeGetMaxInstancesPerVm)
GPUMOD_LINK_CHECK(nvmlDeviceGetActiveVgpus)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceGetVmID)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceGetUUID)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceGetVmDriverVersion)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceGetFbUsage)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceGetType)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceGetFrameRateLimit)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceGetEccMode)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceGetEncoderCapacity)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceSetEncoderCapacity)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceGetEncoderStats)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceGetEncoderSessions)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceGetFBCStats)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceGetFBCSessions)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceGetGpuInstanceId)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceGetGpuPciId)
GPUMOD_LINK_CHECK(nvmlVgpuTypeGetCapabilities)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceGetMdevUUID)
GPUMOD_LINK_CHECK(nvmlGetVgpuVersion)
GPUMOD_LINK_CHECK(nvmlSetVgpuVersion)
GPUMOD_LINK_CHECK(nvmlVgpuInstanceGetLicenseInfo_v2)

// Excluded devices
GPUMOD_LINK_CHECK(nvmlGetExcludedDeviceCount)
GPUMOD_LINK_CHECK(nvmlGetExcludedDeviceInfoByIndex)

// MIG
GPUMOD_LINK_CHECK(nvmlDeviceSetMigMode)
GPUMOD_LINK_CHECK(nvmlDeviceGetMigMode)
GPUMOD_LINK_CHECK(nvmlDeviceGetGpuInstanceProfileInfo)
GPUMOD_LINK_CHECK(nvmlDeviceGetGpuInstanceRemainingCapacity)
GPUMOD_LINK_CHECK(nvmlDeviceCreateGpuInstance)
GPUMOD_LINK_CHECK(nvmlGpuInstanceDestroy)
GPUMOD_LINK_CHECK(nvmlDeviceGetGpuInstances)
GPUMOD_LINK_CHECK(nvmlDeviceGetGpuInstanceById)
GPUMOD_LINK_CHECK(nvmlGpuInstanceGetInfo)
GPUMOD_LINK_CHECK(nvmlGpuInstanceGetComputeInstanceProfileInfo)
GPUMOD_LINK_CHECK(nvmlGpuInstanceGetComputeInstanceRemainingCapacity)
GPUMOD_LINK_CHECK(nvmlGpuInstanceCreateComputeInstance)
GPUMOD_LINK_CHECK(nvmlComputeInstanceDestroy)
GPUMOD_LINK_CHECK(nvmlGpuInstanceGetComputeInstances)
GPUMOD_LINK_CHECK(nvmlGpuInstanceGetComputeInstanceById)
GPUMOD_LINK_CHECK(nvmlComputeInstanceGetInfo_v2)
GPUMOD_LINK_CHECK(nvmlDeviceIsMigDeviceHandle)
GPUMOD_LINK_CHECK(nvmlDeviceGetGpuInstanceId)
GPUMOD_LINK_CHECK(nvmlDeviceGetComputeInstanceId)
GPUMOD_LINK_CHECK(nvmlDeviceGetMaxMigDeviceCount)
GPUMOD_LINK_CHECK(nvmlDeviceGetMigDeviceHandleByIndex)
GPUMOD_LINK_CHECK(nvmlDeviceGetDeviceHandleFromMigDeviceHandle)

// GPM
GPUMOD_LINK_CHECK(nvmlGpmMetricsGet)
GPUMOD_LINK_CHECK(nvmlGpmSampleFree)
GPUMOD_LINK_CHECK(nvmlGpmSampleAlloc)
GPUMOD_LINK_CHECK(nvmlGpmSampleGet)
GPUMOD_LINK_CHECK(nvmlGpmMigSampleGet)
GPUMOD_LINK_CHECK(nvmlGpmQueryDeviceSupport)
GPUMOD_LINK_CHECK(nvmlGpmQueryIfStreamingEnabled)
GPUMOD_LINK_CHECK(nvmlGpmSetStreamingEnabled)

} // namespace gpumod::cuda::test
