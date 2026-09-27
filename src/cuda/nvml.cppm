/**
 * @file nvml.cppm
 * @brief Primary interface for wwr.cuda.nvml
 *
 * This module wraps the NVML (NVIDIA Management Library) API and exports types,
 * constants, and functions for GPU management and monitoring.
 *
 * Usage:
 *   import wwr.cuda.nvml;
 */

module;

#include <nvml.h>

export module wwr.cuda.nvml;

import std;

export namespace wwr::cuda {

// ========================================================================
// Core Handle Types
// ========================================================================
using ::nvmlDevice_t;
using ::nvmlEventSet_t;
using ::nvmlGpuInstance_t;
using ::nvmlSystemEventSet_t;
using ::nvmlUnit_t;

// ========================================================================
// Return Status Enum
// ========================================================================
using ::NVML_ERROR_ALREADY_INITIALIZED;
using ::NVML_ERROR_ARGUMENT_VERSION_MISMATCH;
using ::NVML_ERROR_CORRUPTED_INFOROM;
using ::NVML_ERROR_DEPRECATED;
using ::NVML_ERROR_DRIVER_NOT_LOADED;
using ::NVML_ERROR_FREQ_NOT_SUPPORTED;
using ::NVML_ERROR_FUNCTION_NOT_FOUND;
using ::NVML_ERROR_GPU_IS_LOST;
using ::NVML_ERROR_GPU_NOT_FOUND;
using ::NVML_ERROR_IN_USE;
using ::NVML_ERROR_INSUFFICIENT_POWER;
using ::NVML_ERROR_INSUFFICIENT_RESOURCES;
using ::NVML_ERROR_INSUFFICIENT_SIZE;
using ::NVML_ERROR_INVALID_ARGUMENT;
using ::NVML_ERROR_INVALID_STATE;
using ::NVML_ERROR_IRQ_ISSUE;
using ::NVML_ERROR_LIB_RM_VERSION_MISMATCH;
using ::NVML_ERROR_LIBRARY_NOT_FOUND;
using ::NVML_ERROR_MEMORY;
using ::NVML_ERROR_NO_DATA;
using ::NVML_ERROR_NO_PERMISSION;
using ::NVML_ERROR_NOT_FOUND;
using ::NVML_ERROR_NOT_READY;
using ::NVML_ERROR_NOT_SUPPORTED;
using ::NVML_ERROR_OPERATING_SYSTEM;
using ::NVML_ERROR_RESET_REQUIRED;
using ::NVML_ERROR_RESET_TYPE_NOT_SUPPORTED;
using ::NVML_ERROR_TIMEOUT;
using ::NVML_ERROR_UNINITIALIZED;
using ::NVML_ERROR_UNKNOWN;
using ::NVML_ERROR_VGPU_ECC_NOT_SUPPORTED;
using ::NVML_SUCCESS;
using ::nvmlReturn_t;

// ========================================================================
// Enumeration: Enable State
// ========================================================================
using ::NVML_FEATURE_DISABLED;
using ::NVML_FEATURE_ENABLED;
using ::nvmlEnableState_t;

// ========================================================================
// Enumeration: Brand Type
// ========================================================================
using ::NVML_BRAND_COUNT;
using ::NVML_BRAND_GEFORCE;
using ::NVML_BRAND_GEFORCE_RTX;
using ::NVML_BRAND_GRID;
using ::NVML_BRAND_NVIDIA;
using ::NVML_BRAND_NVIDIA_CLOUD_GAMING;
using ::NVML_BRAND_NVIDIA_RTX;
using ::NVML_BRAND_NVIDIA_VAPPS;
using ::NVML_BRAND_NVIDIA_VCS;
using ::NVML_BRAND_NVIDIA_VGAMING;
using ::NVML_BRAND_NVIDIA_VPC;
using ::NVML_BRAND_NVIDIA_VWS;
using ::NVML_BRAND_NVS;
using ::NVML_BRAND_QUADRO;
using ::NVML_BRAND_QUADRO_RTX;
using ::NVML_BRAND_TESLA;
using ::NVML_BRAND_TITAN;
using ::NVML_BRAND_TITAN_RTX;
using ::NVML_BRAND_UNKNOWN;
using ::nvmlBrandType_t;

// ========================================================================
// Enumeration: Temperature Sensors
// ========================================================================
using ::NVML_TEMPERATURE_COUNT;
using ::NVML_TEMPERATURE_GPU;
using ::nvmlTemperatureSensors_t;

// ========================================================================
// Enumeration: Temperature Thresholds
// ========================================================================
using ::NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_CURR;
using ::NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MAX;
using ::NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MIN;
using ::NVML_TEMPERATURE_THRESHOLD_COUNT;
using ::NVML_TEMPERATURE_THRESHOLD_GPS_CURR;
using ::NVML_TEMPERATURE_THRESHOLD_GPU_MAX;
using ::NVML_TEMPERATURE_THRESHOLD_MEM_MAX;
using ::NVML_TEMPERATURE_THRESHOLD_SHUTDOWN;
using ::NVML_TEMPERATURE_THRESHOLD_SLOWDOWN;
using ::nvmlTemperatureThresholds_t;

// ========================================================================
// Enumeration: Compute Mode
// ========================================================================
using ::NVML_COMPUTEMODE_COUNT;
using ::NVML_COMPUTEMODE_DEFAULT;
using ::NVML_COMPUTEMODE_EXCLUSIVE_PROCESS;
using ::NVML_COMPUTEMODE_EXCLUSIVE_THREAD;
using ::NVML_COMPUTEMODE_PROHIBITED;
using ::nvmlComputeMode_t;

// ========================================================================
// Enumeration: Clock Types
// ========================================================================
using ::NVML_CLOCK_COUNT;
using ::NVML_CLOCK_GRAPHICS;
using ::NVML_CLOCK_MEM;
using ::NVML_CLOCK_SM;
using ::NVML_CLOCK_VIDEO;
using ::nvmlClockType_t;

// ========================================================================
// Enumeration: Clock IDs
// ========================================================================
using ::NVML_CLOCK_ID_APP_CLOCK_DEFAULT;
using ::NVML_CLOCK_ID_APP_CLOCK_TARGET;
using ::NVML_CLOCK_ID_COUNT;
using ::NVML_CLOCK_ID_CURRENT;
using ::NVML_CLOCK_ID_CUSTOMER_BOOST_MAX;
using ::nvmlClockId_t;

// ========================================================================
// Enumeration: Driver Model
// ========================================================================
using ::NVML_DRIVER_MCDM;
using ::NVML_DRIVER_WDDM;
using ::NVML_DRIVER_WDM;
using ::nvmlDriverModel_t;

// ========================================================================
// Enumeration: Performance States
// ========================================================================
using ::NVML_PSTATE_0;
using ::NVML_PSTATE_1;
using ::NVML_PSTATE_10;
using ::NVML_PSTATE_11;
using ::NVML_PSTATE_12;
using ::NVML_PSTATE_13;
using ::NVML_PSTATE_14;
using ::NVML_PSTATE_15;
using ::NVML_PSTATE_2;
using ::NVML_PSTATE_3;
using ::NVML_PSTATE_4;
using ::NVML_PSTATE_5;
using ::NVML_PSTATE_6;
using ::NVML_PSTATE_7;
using ::NVML_PSTATE_8;
using ::NVML_PSTATE_9;
using ::NVML_PSTATE_UNKNOWN;
using ::nvmlPstates_t;

// ========================================================================
// Enumeration: GPU Operation Mode
// ========================================================================
using ::NVML_GOM_ALL_ON;
using ::NVML_GOM_COMPUTE;
using ::NVML_GOM_LOW_DP;
using ::nvmlGpuOperationMode_t;

// ========================================================================
// Enumeration: InfoROM Objects
// ========================================================================
using ::NVML_INFOROM_COUNT;
using ::NVML_INFOROM_DEN;
using ::NVML_INFOROM_ECC;
using ::NVML_INFOROM_OEM;
using ::NVML_INFOROM_POWER;
using ::nvmlInforomObject_t;

// ========================================================================
// Enumeration: Memory Error Types
// ========================================================================
using ::NVML_MEMORY_ERROR_TYPE_CORRECTED;
using ::NVML_MEMORY_ERROR_TYPE_COUNT;
using ::NVML_MEMORY_ERROR_TYPE_UNCORRECTED;
using ::nvmlMemoryErrorType_t;

// ========================================================================
// Enumeration: ECC Counter Types
// ========================================================================
using ::NVML_AGGREGATE_ECC;
using ::NVML_ECC_COUNTER_TYPE_COUNT;
using ::NVML_VOLATILE_ECC;
using ::nvmlEccCounterType_t;

// ========================================================================
// Enumeration: Memory Location
// ========================================================================
using ::NVML_MEMORY_LOCATION_CBU;
using ::NVML_MEMORY_LOCATION_COUNT;
using ::NVML_MEMORY_LOCATION_DEVICE_MEMORY;
using ::NVML_MEMORY_LOCATION_DRAM;
using ::NVML_MEMORY_LOCATION_L1_CACHE;
using ::NVML_MEMORY_LOCATION_L2_CACHE;
using ::NVML_MEMORY_LOCATION_REGISTER_FILE;
using ::NVML_MEMORY_LOCATION_SRAM;
using ::NVML_MEMORY_LOCATION_TEXTURE_MEMORY;
using ::NVML_MEMORY_LOCATION_TEXTURE_SHM;
using ::nvmlMemoryLocation_t;

// ========================================================================
// Enumeration: Page Retirement Cause
// ========================================================================
using ::NVML_PAGE_RETIREMENT_CAUSE_COUNT;
using ::NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR;
using ::NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS;
using ::nvmlPageRetirementCause_t;

// ========================================================================
// Enumeration: Restricted API
// ========================================================================
using ::NVML_RESTRICTED_API_COUNT;
using ::NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS;
using ::NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS;
using ::nvmlRestrictedAPI_t;

// ========================================================================
// Enumeration: GPU Topology Level
// ========================================================================
using ::NVML_TOPOLOGY_HOSTBRIDGE;
using ::NVML_TOPOLOGY_INTERNAL;
using ::NVML_TOPOLOGY_MULTIPLE;
using ::NVML_TOPOLOGY_NODE;
using ::NVML_TOPOLOGY_SINGLE;
using ::NVML_TOPOLOGY_SYSTEM;
using ::nvmlGpuTopologyLevel_t;

// ========================================================================
// Enumeration: GPU P2P Status
// ========================================================================
using ::NVML_P2P_STATUS_CHIPSET_NOT_SUPPORED;
using ::NVML_P2P_STATUS_CHIPSET_NOT_SUPPORTED;
using ::NVML_P2P_STATUS_DISABLED_BY_REGKEY;
using ::NVML_P2P_STATUS_GPU_NOT_SUPPORTED;
using ::NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED;
using ::NVML_P2P_STATUS_NOT_SUPPORTED;
using ::NVML_P2P_STATUS_OK;
using ::NVML_P2P_STATUS_UNKNOWN;
using ::nvmlGpuP2PStatus_t;

// ========================================================================
// Enumeration: GPU P2P Caps Index
// ========================================================================
using ::NVML_P2P_CAPS_INDEX_ATOMICS;
using ::NVML_P2P_CAPS_INDEX_NVLINK;
using ::NVML_P2P_CAPS_INDEX_PCI;
using ::NVML_P2P_CAPS_INDEX_PROP;
using ::NVML_P2P_CAPS_INDEX_READ;
using ::NVML_P2P_CAPS_INDEX_UNKNOWN;
using ::NVML_P2P_CAPS_INDEX_WRITE;
using ::nvmlGpuP2PCapsIndex_t;

// ========================================================================
// Enumeration: NvLink Version
// ========================================================================
using ::NVML_NVLINK_VERSION_1_0;
using ::NVML_NVLINK_VERSION_2_0;
using ::NVML_NVLINK_VERSION_2_2;
using ::NVML_NVLINK_VERSION_3_0;
using ::NVML_NVLINK_VERSION_3_1;
using ::NVML_NVLINK_VERSION_4_0;
using ::NVML_NVLINK_VERSION_5_0;
using ::NVML_NVLINK_VERSION_INVALID;
using ::nvmlNvlinkVersion_t;

// ========================================================================
// Enumeration: NvLink Capability
// ========================================================================
using ::NVML_NVLINK_CAP_COUNT;
using ::NVML_NVLINK_CAP_P2P_ATOMICS;
using ::NVML_NVLINK_CAP_P2P_SUPPORTED;
using ::NVML_NVLINK_CAP_SLI_BRIDGE;
using ::NVML_NVLINK_CAP_SYSMEM_ACCESS;
using ::NVML_NVLINK_CAP_SYSMEM_ATOMICS;
using ::NVML_NVLINK_CAP_VALID;
using ::nvmlNvLinkCapability_t;

// ========================================================================
// Enumeration: NvLink Error Counter
// ========================================================================
using ::NVML_NVLINK_ERROR_COUNT;
using ::NVML_NVLINK_ERROR_DL_CRC_DATA;
using ::NVML_NVLINK_ERROR_DL_CRC_FLIT;
using ::NVML_NVLINK_ERROR_DL_ECC_DATA;
using ::NVML_NVLINK_ERROR_DL_RECOVERY;
using ::NVML_NVLINK_ERROR_DL_REPLAY;
using ::nvmlNvLinkErrorCounter_t;

// ========================================================================
// Enumeration: NvLink Device Type
// ========================================================================
using ::NVML_NVLINK_DEVICE_TYPE_GPU;
using ::NVML_NVLINK_DEVICE_TYPE_IBMNPU;
using ::NVML_NVLINK_DEVICE_TYPE_SWITCH;
using ::NVML_NVLINK_DEVICE_TYPE_UNKNOWN;
using ::nvmlIntNvLinkDeviceType_t;

// ========================================================================
// Enumeration: NvLink Utilization Counter Units
// ========================================================================
using ::NVML_NVLINK_COUNTER_UNIT_BYTES;
using ::NVML_NVLINK_COUNTER_UNIT_COUNT;
using ::NVML_NVLINK_COUNTER_UNIT_CYCLES;
using ::NVML_NVLINK_COUNTER_UNIT_PACKETS;
using ::NVML_NVLINK_COUNTER_UNIT_RESERVED;
using ::nvmlNvLinkUtilizationCountUnits_t;

// ========================================================================
// Enumeration: NvLink Utilization Counter Packet Types
// ========================================================================
using ::NVML_NVLINK_COUNTER_PKTFILTER_ALL;
using ::NVML_NVLINK_COUNTER_PKTFILTER_FLUSH;
using ::NVML_NVLINK_COUNTER_PKTFILTER_NOP;
using ::NVML_NVLINK_COUNTER_PKTFILTER_NRATOM;
using ::NVML_NVLINK_COUNTER_PKTFILTER_RATOM;
using ::NVML_NVLINK_COUNTER_PKTFILTER_READ;
using ::NVML_NVLINK_COUNTER_PKTFILTER_RESPDATA;
using ::NVML_NVLINK_COUNTER_PKTFILTER_RESPNODATA;
using ::NVML_NVLINK_COUNTER_PKTFILTER_WRITE;
using ::nvmlNvLinkUtilizationCountPktTypes_t;

// ========================================================================
// Enumeration: Sampling Type
// ========================================================================
using ::NVML_DEC_UTILIZATION_SAMPLES;
using ::NVML_ENC_UTILIZATION_SAMPLES;
using ::NVML_GPU_UTILIZATION_SAMPLES;
using ::NVML_JPG_UTILIZATION_SAMPLES;
using ::NVML_MEMORY_CLK_SAMPLES;
using ::NVML_MEMORY_UTILIZATION_SAMPLES;
using ::NVML_MODULE_POWER_SAMPLES;
using ::NVML_OFA_UTILIZATION_SAMPLES;
using ::NVML_PROCESSOR_CLK_SAMPLES;
using ::NVML_SAMPLINGTYPE_COUNT;
using ::NVML_TOTAL_POWER_SAMPLES;
using ::nvmlSamplingType_t;

// ========================================================================
// Enumeration: PCIe Utilization Counter
// ========================================================================
using ::NVML_PCIE_UTIL_COUNT;
using ::NVML_PCIE_UTIL_RX_BYTES;
using ::NVML_PCIE_UTIL_TX_BYTES;
using ::nvmlPcieUtilCounter_t;

// ========================================================================
// Enumeration: Value Type
// ========================================================================
using ::NVML_VALUE_TYPE_COUNT;
using ::NVML_VALUE_TYPE_DOUBLE;
using ::NVML_VALUE_TYPE_SIGNED_INT;
using ::NVML_VALUE_TYPE_SIGNED_LONG_LONG;
using ::NVML_VALUE_TYPE_UNSIGNED_INT;
using ::NVML_VALUE_TYPE_UNSIGNED_LONG;
using ::NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
using ::NVML_VALUE_TYPE_UNSIGNED_SHORT;
using ::nvmlValueType_t;

// ========================================================================
// Enumeration: Perf Policy Type
// ========================================================================
using ::NVML_PERF_POLICY_BOARD_LIMIT;
using ::NVML_PERF_POLICY_COUNT;
using ::NVML_PERF_POLICY_LOW_UTILIZATION;
using ::NVML_PERF_POLICY_POWER;
using ::NVML_PERF_POLICY_RELIABILITY;
using ::NVML_PERF_POLICY_SYNC_BOOST;
using ::NVML_PERF_POLICY_THERMAL;
using ::NVML_PERF_POLICY_TOTAL_APP_CLOCKS;
using ::NVML_PERF_POLICY_TOTAL_BASE_CLOCKS;
using ::nvmlPerfPolicyType_t;

// ========================================================================
// Enumeration: Thermal Target
// ========================================================================
using ::NVML_THERMAL_TARGET_ALL;
using ::NVML_THERMAL_TARGET_BOARD;
using ::NVML_THERMAL_TARGET_GPU;
using ::NVML_THERMAL_TARGET_MEMORY;
using ::NVML_THERMAL_TARGET_NONE;
using ::NVML_THERMAL_TARGET_POWER_SUPPLY;
using ::NVML_THERMAL_TARGET_UNKNOWN;
using ::NVML_THERMAL_TARGET_VCD_BOARD;
using ::NVML_THERMAL_TARGET_VCD_INLET;
using ::NVML_THERMAL_TARGET_VCD_OUTLET;
using ::nvmlThermalTarget_t;

// ========================================================================
// Enumeration: Thermal Controller
// ========================================================================
using ::NVML_THERMAL_CONTROLLER_ADM1032;
using ::NVML_THERMAL_CONTROLLER_ADT7461;
using ::NVML_THERMAL_CONTROLLER_ADT7473;
using ::NVML_THERMAL_CONTROLLER_ADT7473S;
using ::NVML_THERMAL_CONTROLLER_G781;
using ::NVML_THERMAL_CONTROLLER_GPU_INTERNAL;
using ::NVML_THERMAL_CONTROLLER_LM64;
using ::NVML_THERMAL_CONTROLLER_LM89;
using ::NVML_THERMAL_CONTROLLER_LM99;
using ::NVML_THERMAL_CONTROLLER_MAX1617;
using ::NVML_THERMAL_CONTROLLER_MAX6649;
using ::NVML_THERMAL_CONTROLLER_MAX6649R;
using ::NVML_THERMAL_CONTROLLER_NONE;
using ::NVML_THERMAL_CONTROLLER_NVSYSCON_CANOAS;
using ::NVML_THERMAL_CONTROLLER_NVSYSCON_E551;
using ::NVML_THERMAL_CONTROLLER_OS;
using ::NVML_THERMAL_CONTROLLER_SBMAX6649;
using ::NVML_THERMAL_CONTROLLER_UNKNOWN;
using ::NVML_THERMAL_CONTROLLER_VBIOSEVT;
using ::nvmlThermalController_t;

// ========================================================================
// Enumeration: Cooler Control
// ========================================================================
using ::NVML_THERMAL_COOLER_SIGNAL_COUNT;
using ::NVML_THERMAL_COOLER_SIGNAL_NONE;
using ::NVML_THERMAL_COOLER_SIGNAL_TOGGLE;
using ::NVML_THERMAL_COOLER_SIGNAL_VARIABLE;
using ::nvmlCoolerControl_t;

// ========================================================================
// Enumeration: Cooler Target
// ========================================================================
using ::NVML_THERMAL_COOLER_TARGET_GPU;
using ::NVML_THERMAL_COOLER_TARGET_GPU_RELATED;
using ::NVML_THERMAL_COOLER_TARGET_MEMORY;
using ::NVML_THERMAL_COOLER_TARGET_NONE;
using ::NVML_THERMAL_COOLER_TARGET_POWER_SUPPLY;
using ::nvmlCoolerTarget_t;

// ========================================================================
// Enumeration: Bridge Chip Type
// ========================================================================
using ::NVML_BRIDGE_CHIP_BRO4;
using ::NVML_BRIDGE_CHIP_PLX;
using ::nvmlBridgeChipType_t;

// ========================================================================
// Enumeration: Fan State
// ========================================================================
using ::NVML_FAN_FAILED;
using ::NVML_FAN_NORMAL;
using ::nvmlFanState_t;

// ========================================================================
// Enumeration: LED Color
// ========================================================================
using ::NVML_LED_COLOR_AMBER;
using ::NVML_LED_COLOR_GREEN;
using ::nvmlLedColor_t;

// ========================================================================
// Enumeration: Encoder Query Type
// ========================================================================
using ::NVML_ENCODER_QUERY_AV1;
using ::NVML_ENCODER_QUERY_H264;
using ::NVML_ENCODER_QUERY_HEVC;
using ::NVML_ENCODER_QUERY_UNKNOWN;
using ::nvmlEncoderType_t;

// ========================================================================
// Enumeration: FBC Session Type
// ========================================================================
using ::NVML_FBC_SESSION_TYPE_CUDA;
using ::NVML_FBC_SESSION_TYPE_HWENC;
using ::NVML_FBC_SESSION_TYPE_TOSYS;
using ::NVML_FBC_SESSION_TYPE_UNKNOWN;
using ::NVML_FBC_SESSION_TYPE_VID;
using ::nvmlFBCSessionType_t;

// ========================================================================
// Enumeration: Detach GPU State
// ========================================================================
using ::NVML_DETACH_GPU_KEEP;
using ::NVML_DETACH_GPU_REMOVE;
using ::nvmlDetachGpuState_t;

// ========================================================================
// Enumeration: PCIe Link State
// ========================================================================
using ::NVML_PCIE_LINK_KEEP;
using ::NVML_PCIE_LINK_SHUT_DOWN;
using ::nvmlPcieLinkState_t;

// ========================================================================
// Enumeration: GPU Virtualization Mode
// ========================================================================
using ::NVML_GPU_VIRTUALIZATION_MODE_HOST_VGPU;
using ::NVML_GPU_VIRTUALIZATION_MODE_HOST_VSGA;
using ::NVML_GPU_VIRTUALIZATION_MODE_NONE;
using ::NVML_GPU_VIRTUALIZATION_MODE_PASSTHROUGH;
using ::NVML_GPU_VIRTUALIZATION_MODE_VGPU;
using ::nvmlGpuVirtualizationMode_t;

// ========================================================================
// Enumeration: Host vGPU Mode
// ========================================================================
using ::NVML_HOST_VGPU_MODE_NON_SRIOV;
using ::NVML_HOST_VGPU_MODE_SRIOV;
using ::nvmlHostVgpuMode_t;

// ========================================================================
// Enumeration: vGPU VM ID Type
// ========================================================================
using ::NVML_VGPU_VM_ID_DOMAIN_ID;
using ::NVML_VGPU_VM_ID_UUID;
using ::nvmlVgpuVmIdType_t;

// ========================================================================
// Enumeration: vGPU Guest Info State
// ========================================================================
using ::NVML_VGPU_INSTANCE_GUEST_INFO_STATE_INITIALIZED;
using ::NVML_VGPU_INSTANCE_GUEST_INFO_STATE_UNINITIALIZED;
using ::nvmlVgpuGuestInfoState_t;

// ========================================================================
// Enumeration: GRID License Feature Code
// ========================================================================
using ::NVML_GRID_LICENSE_FEATURE_CODE_COMPUTE;
using ::NVML_GRID_LICENSE_FEATURE_CODE_GAMING;
using ::NVML_GRID_LICENSE_FEATURE_CODE_NVIDIA_RTX;
using ::NVML_GRID_LICENSE_FEATURE_CODE_UNKNOWN;
using ::NVML_GRID_LICENSE_FEATURE_CODE_VGPU;
using ::NVML_GRID_LICENSE_FEATURE_CODE_VWORKSTATION;
using ::nvmlGridLicenseFeatureCode_t;

// ========================================================================
// Enumeration: vGPU Capability
// ========================================================================
using ::NVML_VGPU_CAP_COUNT;
using ::NVML_VGPU_CAP_EXCLUSIVE_SIZE;
using ::NVML_VGPU_CAP_EXCLUSIVE_TYPE;
using ::NVML_VGPU_CAP_GPUDIRECT;
using ::NVML_VGPU_CAP_MULTI_VGPU_EXCLUSIVE;
using ::NVML_VGPU_CAP_NVLINK_P2P;
using ::nvmlVgpuCapability_t;

// ========================================================================
// Enumeration: vGPU Driver Capability
// ========================================================================
using ::NVML_VGPU_DRIVER_CAP_COUNT;
using ::NVML_VGPU_DRIVER_CAP_HETEROGENEOUS_MULTI_VGPU;
using ::NVML_VGPU_DRIVER_CAP_WARM_UPDATE;
using ::nvmlVgpuDriverCapability_t;

// ========================================================================
// Enumeration: Device vGPU Capability
// ========================================================================
using ::NVML_DEVICE_VGPU_CAP_COMPUTE_MEDIA_ENGINE_GPU;
using ::NVML_DEVICE_VGPU_CAP_COUNT;
using ::NVML_DEVICE_VGPU_CAP_DEVICE_STREAMING;
using ::NVML_DEVICE_VGPU_CAP_FRACTIONAL_MULTI_VGPU;
using ::NVML_DEVICE_VGPU_CAP_HETEROGENEOUS_TIMESLICE_PROFILES;
using ::NVML_DEVICE_VGPU_CAP_HETEROGENEOUS_TIMESLICE_SIZES;
using ::NVML_DEVICE_VGPU_CAP_HOMOGENEOUS_PLACEMENTS;
using ::NVML_DEVICE_VGPU_CAP_MIG_TIMESLICING_ENABLED;
using ::NVML_DEVICE_VGPU_CAP_MIG_TIMESLICING_SUPPORTED;
using ::NVML_DEVICE_VGPU_CAP_MINI_QUARTER_GPU;
using ::NVML_DEVICE_VGPU_CAP_READ_DEVICE_BUFFER_BW;
using ::NVML_DEVICE_VGPU_CAP_WARM_UPDATE;
using ::NVML_DEVICE_VGPU_CAP_WRITE_DEVICE_BUFFER_BW;
using ::nvmlDeviceVgpuCapability_t;

// ========================================================================
// Enumeration: GPU Recovery Action
// ========================================================================
using ::NVML_GPU_RECOVERY_ACTION_DRAIN_AND_RESET;
using ::NVML_GPU_RECOVERY_ACTION_DRAIN_P2P;
using ::NVML_GPU_RECOVERY_ACTION_GPU_RESET;
using ::NVML_GPU_RECOVERY_ACTION_NODE_REBOOT;
using ::NVML_GPU_RECOVERY_ACTION_NONE;
using ::nvmlDeviceGpuRecoveryAction_t;

// ========================================================================
// Enumeration: Device Addressing Mode Type
// ========================================================================
using ::NVML_DEVICE_ADDRESSING_MODE_ATS;
using ::NVML_DEVICE_ADDRESSING_MODE_HMM;
using ::NVML_DEVICE_ADDRESSING_MODE_NONE;
using ::nvmlDeviceAddressingModeType_t;

// ========================================================================
// Enumeration: GPU Utilization Domain
// ========================================================================
using ::NVML_GPU_UTILIZATION_DOMAIN_BUS;
using ::NVML_GPU_UTILIZATION_DOMAIN_FB;
using ::NVML_GPU_UTILIZATION_DOMAIN_GPU;
using ::NVML_GPU_UTILIZATION_DOMAIN_VID;
using ::nvmlGpuUtilizationDomainId_t;

// ========================================================================
// Enumeration: vGPU Pgpu Compatibility Limit Code
// ========================================================================
using ::nvmlVgpuPgpuCompatibilityLimitCode_t;

// ========================================================================
// Typedef Scalar Types
// ========================================================================
using ::nvmlAffinityScope_t;
using ::nvmlBusType_t;
using ::nvmlComputeInstance_t;
using ::nvmlDeviceArchitecture_t;
using ::nvmlFanControlPolicy_t;
using ::nvmlGpuFabricState_t;
using ::nvmlPowerScopeType_t;
using ::nvmlPowerSource_t;
using ::nvmlVgpuInstance_t;
using ::nvmlVgpuTypeId_t;

// ========================================================================
// Core Structs: Memory and Utilization
// ========================================================================
using ::nvmlBAR1Memory_t;
using ::nvmlEccErrorCounts_t;
using ::nvmlMemory_t;
using ::nvmlMemory_v2_st;
using ::nvmlMemory_v2_t;
using ::nvmlUtilization_t;

// ========================================================================
// Structs: PCI Information
// ========================================================================
using ::nvmlPciInfo_st;
using ::nvmlPciInfo_t;
using ::nvmlPciInfoExt_t;
using ::nvmlPciInfoExt_v1_t;

// ========================================================================
// Structs: Process Information
// ========================================================================
using ::nvmlProcessDetail_v1_t;
using ::nvmlProcessDetailList_t;
using ::nvmlProcessDetailList_v1_t;
using ::nvmlProcessesUtilizationInfo_t;
using ::nvmlProcessesUtilizationInfo_v1_t;
using ::nvmlProcessInfo_t;
using ::nvmlProcessInfo_v1_t;
using ::nvmlProcessInfo_v2_t;
using ::nvmlProcessUtilizationInfo_v1_t;
using ::nvmlProcessUtilizationSample_t;

// ========================================================================
// Structs: Device Attributes and Properties
// ========================================================================
using ::nvmlC2cModeInfo_v1_t;
using ::nvmlDeviceAddressingMode_t;
using ::nvmlDeviceAddressingMode_v1_t;
using ::nvmlDeviceAttributes_st;
using ::nvmlDeviceAttributes_t;
using ::nvmlGpuDynamicPstatesInfo_st;
using ::nvmlGpuDynamicPstatesInfo_t;
using ::nvmlRepairStatus_t;
using ::nvmlRepairStatus_v1_t;
using ::nvmlRowRemapperHistogramValues_t;

// ========================================================================
// Structs: Clock and Performance
// ========================================================================
using ::nvmlClkMonFaultInfo_t;
using ::nvmlClkMonStatus_t;
using ::nvmlClockOffset_t;
using ::nvmlClockOffset_v1_t;
using ::nvmlDeviceCurrentClockFreqs_t;
using ::nvmlDeviceCurrentClockFreqs_v1_t;
using ::nvmlDevicePerfModes_t;
using ::nvmlDevicePerfModes_v1_t;
using ::nvmlViolationTime_t;

// ========================================================================
// Structs: Power
// ========================================================================
using ::nvmlPowerValue_v2_t;

// ========================================================================
// Structs: Fan and Cooling
// ========================================================================
using ::nvmlCoolerInfo_t;
using ::nvmlCoolerInfo_v1_t;
using ::nvmlFanSpeedInfo_t;
using ::nvmlFanSpeedInfo_v1_t;
using ::nvmlGpuThermalSettings_t;
using ::nvmlMarginTemperature_t;
using ::nvmlMarginTemperature_v1_t;

// ========================================================================
// Structs: UUID
// ========================================================================
using ::NVML_UUID_TYPE_ASCII;
using ::NVML_UUID_TYPE_BINARY;
using ::NVML_UUID_TYPE_NONE;
using ::nvmlPdi_t;
using ::nvmlPdi_v1_t;
using ::nvmlUUID_t;
using ::nvmlUUID_v1_t;
using ::nvmlUUIDType_t;
using ::nvmlUUIDValue_t;

// ========================================================================
// Structs: DRAM Encryption
// ========================================================================
using ::nvmlDramEncryptionInfo_t;
using ::nvmlDramEncryptionInfo_v1_t;

// ========================================================================
// Structs: Sampling
// ========================================================================
using ::nvmlFieldValue_t;
using ::nvmlSample_t;
using ::nvmlValue_t;

// ========================================================================
// Structs: NvLink
// ========================================================================
using ::nvmlNvLinkInfo_t;
using ::nvmlNvLinkInfo_v2_t;
using ::nvmlNvLinkPowerThres_t;
using ::nvmlNvLinkUtilizationControl_t;

// ========================================================================
// Structs: Bridge Chip
// ========================================================================
using ::nvmlBridgeChipHierarchy_t;
using ::nvmlBridgeChipInfo_t;

// ========================================================================
// Structs: Unit (S-class)
// ========================================================================
using ::nvmlHwbcEntry_t;
using ::nvmlLedState_t;
using ::nvmlPSUInfo_t;
using ::nvmlUnitFanInfo_t;
using ::nvmlUnitFanSpeeds_t;
using ::nvmlUnitInfo_t;

// ========================================================================
// Structs: Events
// ========================================================================
using ::nvmlEventData_t;
using ::nvmlSystemEventData_v1_t;
using ::nvmlSystemEventSetCreateRequest_t;
using ::nvmlSystemEventSetCreateRequest_v1_t;
using ::nvmlSystemEventSetFreeRequest_t;
using ::nvmlSystemEventSetFreeRequest_v1_t;
using ::nvmlSystemEventSetWaitRequest_t;
using ::nvmlSystemEventSetWaitRequest_v1_t;
using ::nvmlSystemRegisterEventRequest_t;
using ::nvmlSystemRegisterEventRequest_v1_t;

// ========================================================================
// Structs: Accounting Stats
// ========================================================================
using ::nvmlAccountingStats_t;

// ========================================================================
// Structs: Encoder/FBC
// ========================================================================
using ::nvmlEncoderSessionInfo_t;
using ::nvmlFBCSessionInfo_t;
using ::nvmlFBCStats_t;

// ========================================================================
// Structs: Confidential Compute
// ========================================================================
using ::nvmlConfComputeGetKeyRotationThresholdInfo_t;
using ::nvmlConfComputeGetKeyRotationThresholdInfo_v1_t;
using ::nvmlConfComputeGpuAttestationReport_t;
using ::nvmlConfComputeGpuCertificate_t;
using ::nvmlConfComputeMemSizeInfo_t;
using ::nvmlConfComputeSetKeyRotationThresholdInfo_t;
using ::nvmlConfComputeSetKeyRotationThresholdInfo_v1_t;
using ::nvmlConfComputeSystemCaps_t;
using ::nvmlConfComputeSystemState_t;
using ::nvmlSystemConfComputeSettings_t;
using ::nvmlSystemConfComputeSettings_v1_t;

// ========================================================================
// Structs: Fabric
// ========================================================================
using ::nvmlGpuFabricInfo_t;
using ::nvmlGpuFabricInfo_v2_t;
using ::nvmlGpuFabricInfo_v3_t;
using ::nvmlGpuFabricInfoV_t;

// ========================================================================
// Structs: ECC SRAM
// ========================================================================
using ::nvmlEccSramErrorStatus_t;
using ::nvmlEccSramErrorStatus_v1_t;
using ::nvmlEccSramUniqueUncorrectedErrorCounts_t;
using ::nvmlEccSramUniqueUncorrectedErrorCounts_v1_t;
using ::nvmlEccSramUniqueUncorrectedErrorEntry_v1_t;

// ========================================================================
// Structs: Platform Info
// ========================================================================
using ::nvmlPlatformInfo_t;
using ::nvmlPlatformInfo_v1_t;
using ::nvmlPlatformInfo_v2_t;

// ========================================================================
// Structs: PRM
// ========================================================================
using ::nvmlPRMTLV_v1_t;

// ========================================================================
// Structs: Driver Branch Info
// ========================================================================
using ::nvmlSystemDriverBranchInfo_t;
using ::nvmlSystemDriverBranchInfo_v1_t;

// ========================================================================
// vGPU Types
// ========================================================================
using ::nvmlActiveVgpuInstanceInfo_t;
using ::nvmlActiveVgpuInstanceInfo_v1_t;
using ::nvmlVgpuCreatablePlacementInfo_t;
using ::nvmlVgpuCreatablePlacementInfo_v1_t;
using ::nvmlVgpuHeterogeneousMode_t;
using ::nvmlVgpuHeterogeneousMode_v1_t;
using ::nvmlVgpuInstancesUtilizationInfo_t;
using ::nvmlVgpuInstancesUtilizationInfo_v1_t;
using ::nvmlVgpuInstanceUtilizationInfo_v1_t;
using ::nvmlVgpuInstanceUtilizationSample_t;
using ::nvmlVgpuLicenseExpiry_t;
using ::nvmlVgpuLicenseInfo_t;
using ::nvmlVgpuMetadata_t;
using ::nvmlVgpuPgpuCompatibility_t;
using ::nvmlVgpuPgpuMetadata_t;
using ::nvmlVgpuPlacementId_t;
using ::nvmlVgpuPlacementId_v1_t;
using ::nvmlVgpuPlacementList_t;
using ::nvmlVgpuPlacementList_v2_t;
using ::nvmlVgpuProcessesUtilizationInfo_t;
using ::nvmlVgpuProcessesUtilizationInfo_v1_t;
using ::nvmlVgpuProcessUtilizationInfo_v1_t;
using ::nvmlVgpuProcessUtilizationSample_t;
using ::nvmlVgpuRuntimeState_t;
using ::nvmlVgpuRuntimeState_v1_t;
using ::nvmlVgpuSchedulerCapabilities_t;
using ::nvmlVgpuSchedulerGetState_t;
using ::nvmlVgpuSchedulerLog_t;
using ::nvmlVgpuSchedulerLogEntry_t;
using ::nvmlVgpuSchedulerLogInfo_t;
using ::nvmlVgpuSchedulerLogInfo_v1_t;
using ::nvmlVgpuSchedulerParams_t;
using ::nvmlVgpuSchedulerSetParams_t;
using ::nvmlVgpuSchedulerSetState_t;
using ::nvmlVgpuSchedulerState_t;
using ::nvmlVgpuSchedulerState_v1_t;
using ::nvmlVgpuSchedulerStateInfo_t;
using ::nvmlVgpuSchedulerStateInfo_v1_t;
using ::nvmlVgpuTypeBar1Info_t;
using ::nvmlVgpuTypeBar1Info_v1_t;
using ::nvmlVgpuTypeIdInfo_t;
using ::nvmlVgpuTypeIdInfo_v1_t;
using ::nvmlVgpuTypeMaxInstance_t;
using ::nvmlVgpuTypeMaxInstance_v1_t;
using ::nvmlVgpuVersion_t;

// ========================================================================
// Structs: GRID Licensing
// ========================================================================
using ::nvmlGridLicensableFeature_t;
using ::nvmlGridLicensableFeatures_t;
using ::nvmlGridLicenseExpiry_t;

// ========================================================================
// Structs: Excluded Devices
// ========================================================================
using ::nvmlExcludedDeviceInfo_t;

// ========================================================================
// Structs: MIG
// ========================================================================
using ::nvmlGpuInstancePlacement_t;
using ::nvmlGpuInstanceProfileInfo_t;
using ::nvmlGpuInstanceProfileInfo_v2_t;
using ::nvmlGpuInstanceProfileInfo_v3_t;

// ========================================================================
// Structs: GPM (GPU Performance Monitoring)
// ========================================================================
using ::nvmlGpmMetric_t;
using ::nvmlGpmMetricId_t;
using ::nvmlGpmMetricsGet_t;
using ::nvmlGpmSample_t;
using ::nvmlGpmSupport_t;

// ========================================================================
// Initialization and Cleanup Functions
// ========================================================================
using ::nvmlInit_v2;
using ::nvmlInitWithFlags;
using ::nvmlShutdown;

// ========================================================================
// Error Reporting Functions
// ========================================================================
using ::nvmlErrorString;

// ========================================================================
// System Query Functions
// ========================================================================
using ::nvmlSystemGetCudaDriverVersion;
using ::nvmlSystemGetCudaDriverVersion_v2;
using ::nvmlSystemGetDriverBranch;
using ::nvmlSystemGetDriverVersion;
using ::nvmlSystemGetHicVersion;
using ::nvmlSystemGetNVMLVersion;
using ::nvmlSystemGetProcessName;
using ::nvmlSystemGetTopologyGpuSet;

// ========================================================================
// Unit Query Functions (S-class)
// ========================================================================
using ::nvmlUnitGetCount;
using ::nvmlUnitGetDevices;
using ::nvmlUnitGetFanSpeedInfo;
using ::nvmlUnitGetHandleByIndex;
using ::nvmlUnitGetLedState;
using ::nvmlUnitGetPsuInfo;
using ::nvmlUnitGetTemperature;
using ::nvmlUnitGetUnitInfo;

// ========================================================================
// Device Enumeration Functions
// ========================================================================
using ::nvmlDeviceGetCount_v2;
using ::nvmlDeviceGetHandleByIndex_v2;
using ::nvmlDeviceGetHandleByPciBusId_v2;
using ::nvmlDeviceGetHandleBySerial;
using ::nvmlDeviceGetHandleByUUID;
using ::nvmlDeviceGetHandleByUUIDV;

// Unversioned aliases (defined via macros at top of nvml.h)
using ::nvmlDeviceGetAttributes;
using ::nvmlDeviceGetAttributes_v2;
using ::nvmlDeviceGetCount;
using ::nvmlDeviceGetHandleByIndex;
using ::nvmlDeviceGetHandleByPciBusId;

// ========================================================================
// Device Info Query Functions
// ========================================================================
using ::nvmlDeviceGetAddressingMode;
using ::nvmlDeviceGetArchitecture;
using ::nvmlDeviceGetBoardId;
using ::nvmlDeviceGetBoardPartNumber;
using ::nvmlDeviceGetBrand;
using ::nvmlDeviceGetC2cModeInfoV;
using ::nvmlDeviceGetDisplayActive;
using ::nvmlDeviceGetDisplayMode;
using ::nvmlDeviceGetIndex;
using ::nvmlDeviceGetInforomConfigurationChecksum;
using ::nvmlDeviceGetInforomImageVersion;
using ::nvmlDeviceGetInforomVersion;
using ::nvmlDeviceGetLastBBXFlushTime;
using ::nvmlDeviceGetMinorNumber;
using ::nvmlDeviceGetModuleId;
using ::nvmlDeviceGetMultiGpuBoard;
using ::nvmlDeviceGetName;
using ::nvmlDeviceGetNumaNodeId;
using ::nvmlDeviceGetPersistenceMode;
using ::nvmlDeviceGetRepairStatus;
using ::nvmlDeviceGetSerial;
using ::nvmlDeviceGetUUID;
using ::nvmlDeviceValidateInforom;

// ========================================================================
// Device PCI Info Functions
// ========================================================================
using ::nvmlDeviceGetBusType;
using ::nvmlDeviceGetCurrPcieLinkGeneration;
using ::nvmlDeviceGetCurrPcieLinkWidth;
using ::nvmlDeviceGetGpuMaxPcieLinkGeneration;
using ::nvmlDeviceGetMaxPcieLinkGeneration;
using ::nvmlDeviceGetMaxPcieLinkWidth;
using ::nvmlDeviceGetPcieReplayCounter;
using ::nvmlDeviceGetPcieThroughput;
using ::nvmlDeviceGetPciInfo;
using ::nvmlDeviceGetPciInfo_v3;
using ::nvmlDeviceGetPciInfoExt;

// ========================================================================
// Device Affinity Functions
// ========================================================================
using ::nvmlDeviceClearCpuAffinity;
using ::nvmlDeviceGetCpuAffinity;
using ::nvmlDeviceGetCpuAffinityWithinScope;
using ::nvmlDeviceGetMemoryAffinity;
using ::nvmlDeviceGetP2PStatus;
using ::nvmlDeviceGetTopologyCommonAncestor;
using ::nvmlDeviceGetTopologyNearestGpus;
using ::nvmlDeviceSetCpuAffinity;

// ========================================================================
// Device Clock Functions
// ========================================================================
using ::nvmlDeviceGetApplicationsClock;
using ::nvmlDeviceGetAutoBoostedClocksEnabled;
using ::nvmlDeviceGetClock;
using ::nvmlDeviceGetClockInfo;
using ::nvmlDeviceGetClockOffsets;
using ::nvmlDeviceGetCurrentClockFreqs;
using ::nvmlDeviceGetDefaultApplicationsClock;
using ::nvmlDeviceGetGpcClkMinMaxVfOffset;
using ::nvmlDeviceGetGpcClkVfOffset;
using ::nvmlDeviceGetMaxClockInfo;
using ::nvmlDeviceGetMaxCustomerBoostClock;
using ::nvmlDeviceGetMemClkMinMaxVfOffset;
using ::nvmlDeviceGetMemClkVfOffset;
using ::nvmlDeviceGetMinMaxClockOfPState;
using ::nvmlDeviceGetPerformanceModes;
using ::nvmlDeviceGetSupportedGraphicsClocks;
using ::nvmlDeviceGetSupportedMemoryClocks;
using ::nvmlDeviceGetSupportedPerformanceStates;
using ::nvmlDeviceSetClockOffsets;

// ========================================================================
// Device Temperature and Fan Functions
// ========================================================================
using ::nvmlDeviceGetCoolerInfo;
using ::nvmlDeviceGetFanControlPolicy_v2;
using ::nvmlDeviceGetFanSpeed;
using ::nvmlDeviceGetFanSpeed_v2;
using ::nvmlDeviceGetFanSpeedRPM;
using ::nvmlDeviceGetMarginTemperature;
using ::nvmlDeviceGetMinMaxFanSpeed;
using ::nvmlDeviceGetNumFans;
using ::nvmlDeviceGetTargetFanSpeed;
using ::nvmlDeviceGetTemperature;
using ::nvmlDeviceGetTemperatureThreshold;
using ::nvmlDeviceGetTemperatureV;
using ::nvmlDeviceGetThermalSettings;
using ::nvmlDeviceSetDefaultFanSpeed_v2;
using ::nvmlDeviceSetFanControlPolicy;
using ::nvmlDeviceSetFanSpeed_v2;
using ::nvmlDeviceSetTemperatureThreshold;

// ========================================================================
// Device Performance State Functions
// ========================================================================
using ::nvmlDeviceGetCurrentClocksEventReasons;
using ::nvmlDeviceGetCurrentClocksThrottleReasons;
using ::nvmlDeviceGetDynamicPstatesInfo;
using ::nvmlDeviceGetPerformanceState;
using ::nvmlDeviceGetPowerState;
using ::nvmlDeviceGetSupportedClocksEventReasons;
using ::nvmlDeviceGetSupportedClocksThrottleReasons;

// ========================================================================
// Device Power Functions
// ========================================================================
using ::nvmlDeviceGetEnforcedPowerLimit;
using ::nvmlDeviceGetPowerManagementDefaultLimit;
using ::nvmlDeviceGetPowerManagementLimit;
using ::nvmlDeviceGetPowerManagementLimitConstraints;
using ::nvmlDeviceGetPowerManagementMode;
using ::nvmlDeviceGetPowerMizerMode_v1;
using ::nvmlDeviceGetPowerSource;
using ::nvmlDeviceGetPowerUsage;
using ::nvmlDeviceGetTotalEnergyConsumption;
using ::nvmlDeviceSetPowerManagementLimit;
using ::nvmlDeviceSetPowerMizerMode_v1;

// ========================================================================
// Device Memory Functions
// ========================================================================
using ::nvmlDeviceGetBAR1MemoryInfo;
using ::nvmlDeviceGetMemoryInfo;
using ::nvmlDeviceGetMemoryInfo_v2;

// ========================================================================
// Device Utilization Functions
// ========================================================================
using ::nvmlDeviceGetDecoderUtilization;
using ::nvmlDeviceGetEncoderUtilization;
using ::nvmlDeviceGetJpgUtilization;
using ::nvmlDeviceGetOfaUtilization;
using ::nvmlDeviceGetProcessesUtilizationInfo;
using ::nvmlDeviceGetProcessUtilization;
using ::nvmlDeviceGetUtilizationRates;

// ========================================================================
// Device Compute Mode and ECC Functions
// ========================================================================
using ::nvmlDeviceGetComputeMode;
using ::nvmlDeviceGetCudaComputeCapability;
using ::nvmlDeviceGetDefaultEccMode;
using ::nvmlDeviceGetDetailedEccErrors;
using ::nvmlDeviceGetDramEncryptionMode;
using ::nvmlDeviceGetEccMode;
using ::nvmlDeviceGetMemoryErrorCounter;
using ::nvmlDeviceGetRowRemapperHistogram;
using ::nvmlDeviceGetSramEccErrorStatus;
using ::nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts;
using ::nvmlDeviceGetTotalEccErrors;
using ::nvmlDeviceSetDramEncryptionMode;

// ========================================================================
// Device GPU Operation and Accounting Functions
// ========================================================================
using ::nvmlDeviceClearAccountingPids;
using ::nvmlDeviceGetAccountingBufferSize;
using ::nvmlDeviceGetAccountingMode;
using ::nvmlDeviceGetAccountingPids;
using ::nvmlDeviceGetAccountingStats;
using ::nvmlDeviceGetAPIRestriction;
using ::nvmlDeviceGetGpuOperationMode;
using ::nvmlDeviceSetAccountingMode;
using ::nvmlDeviceSetAPIRestriction;
using ::nvmlDeviceSetGpuOperationMode;

// ========================================================================
// Device Retired Pages Functions
// ========================================================================
using ::nvmlDeviceGetRetiredPages;
using ::nvmlDeviceGetRetiredPages_v2;
using ::nvmlDeviceGetRetiredPagesPendingStatus;

// ========================================================================
// Device Encoder/Decoder Functions
// ========================================================================
using ::nvmlDeviceGetEncoderCapacity;
using ::nvmlDeviceGetEncoderSessions;
using ::nvmlDeviceGetEncoderStats;
using ::nvmlDeviceGetFBCSessions;
using ::nvmlDeviceGetFBCStats;

// ========================================================================
// Device Running Process Functions
// ========================================================================
using ::nvmlDeviceGetComputeRunningProcesses;
using ::nvmlDeviceGetComputeRunningProcesses_v3;
using ::nvmlDeviceGetGraphicsRunningProcesses;
using ::nvmlDeviceGetGraphicsRunningProcesses_v3;
using ::nvmlDeviceGetMPSComputeRunningProcesses;
using ::nvmlDeviceGetMPSComputeRunningProcesses_v3;
using ::nvmlDeviceGetRunningProcessDetailList;

// ========================================================================
// Device Field Value Functions
// ========================================================================
using ::nvmlDeviceClearFieldValues;
using ::nvmlDeviceGetFieldValues;

// ========================================================================
// Device Sampling Functions
// ========================================================================
using ::nvmlDeviceGetSamples;
using ::nvmlDeviceGetViolationStatus;

// ========================================================================
// Device Reset and Drain Functions
// ========================================================================
using ::nvmlDeviceDiscoverGpus;
using ::nvmlDeviceModifyDrainState;
using ::nvmlDeviceQueryDrainState;
using ::nvmlDeviceRemoveGpu;
using ::nvmlDeviceRemoveGpu_v2;
using ::nvmlDeviceResetApplicationsClocks;
using ::nvmlDeviceResetGpuLockedClocks;
using ::nvmlDeviceResetMemoryLockedClocks;
using ::nvmlDeviceSetApplicationsClocks;
using ::nvmlDeviceSetAutoBoostedClocksEnabled;
using ::nvmlDeviceSetDefaultAutoBoostedClocksEnabled;
using ::nvmlDeviceSetGpcClkVfOffset;
using ::nvmlDeviceSetGpuLockedClocks;
using ::nvmlDeviceSetMemClkVfOffset;
using ::nvmlDeviceSetMemoryLockedClocks;

// ========================================================================
// Device NvLink Functions
// ========================================================================
using ::nvmlDeviceFreezeNvLinkUtilizationCounter;
using ::nvmlDeviceGetNvlinkBwMode;
using ::nvmlDeviceGetNvLinkCapability;
using ::nvmlDeviceGetNvLinkErrorCounter;
using ::nvmlDeviceGetNvLinkInfo;
using ::nvmlDeviceGetNvLinkRemoteDeviceType;
using ::nvmlDeviceGetNvLinkRemotePciInfo;
using ::nvmlDeviceGetNvLinkRemotePciInfo_v2;
using ::nvmlDeviceGetNvLinkState;
using ::nvmlDeviceGetNvlinkSupportedBwModes;
using ::nvmlDeviceGetNvLinkUtilizationControl;
using ::nvmlDeviceGetNvLinkUtilizationCounter;
using ::nvmlDeviceGetNvLinkVersion;
using ::nvmlDeviceResetNvLinkErrorCounters;
using ::nvmlDeviceResetNvLinkUtilizationCounter;
using ::nvmlDeviceSetNvlinkBwMode;
using ::nvmlDeviceSetNvLinkDeviceLowPowerThreshold;
using ::nvmlDeviceSetNvLinkUtilizationControl;
using ::nvmlSystemGetNvlinkBwMode;
using ::nvmlSystemSetNvlinkBwMode;

// ========================================================================
// Event Functions
// ========================================================================
using ::nvmlDeviceGetSupportedEventTypes;
using ::nvmlDeviceRegisterEvents;
using ::nvmlEventSetCreate;
using ::nvmlEventSetFree;
using ::nvmlEventSetWait;
using ::nvmlEventSetWait_v2;
using ::nvmlSystemEventSetCreate;
using ::nvmlSystemEventSetFree;
using ::nvmlSystemEventSetWait;
using ::nvmlSystemRegisterEvents;

// ========================================================================
// Virtualization Functions
// ========================================================================
using ::nvmlDeviceGetActiveVgpus;
using ::nvmlDeviceGetCreatableVgpus;
using ::nvmlDeviceGetGridLicensableFeatures;
using ::nvmlDeviceGetGridLicensableFeatures_v4;
using ::nvmlDeviceGetHostVgpuMode;
using ::nvmlDeviceGetPgpuMetadataString;
using ::nvmlDeviceGetSupportedVgpus;
using ::nvmlDeviceGetVgpuCapabilities;
using ::nvmlDeviceGetVgpuHeterogeneousMode;
using ::nvmlDeviceGetVgpuInstancesUtilizationInfo;
using ::nvmlDeviceGetVgpuMetadata;
using ::nvmlDeviceGetVgpuProcessesUtilizationInfo;
using ::nvmlDeviceGetVgpuProcessUtilization;
using ::nvmlDeviceGetVgpuSchedulerCapabilities;
using ::nvmlDeviceGetVgpuSchedulerLog;
using ::nvmlDeviceGetVgpuSchedulerState;
using ::nvmlDeviceGetVgpuTypeCreatablePlacements;
using ::nvmlDeviceGetVgpuTypeSupportedPlacements;
using ::nvmlDeviceGetVgpuUtilization;
using ::nvmlDeviceGetVirtualizationMode;
using ::nvmlDeviceSetVgpuCapabilities;
using ::nvmlDeviceSetVgpuHeterogeneousMode;
using ::nvmlDeviceSetVgpuSchedulerState;
using ::nvmlDeviceSetVirtualizationMode;
using ::nvmlGetVgpuCompatibility;
using ::nvmlGetVgpuDriverCapabilities;
using ::nvmlGetVgpuVersion;
using ::nvmlGpuInstanceGetActiveVgpus;
using ::nvmlGpuInstanceGetCreatableVgpus;
using ::nvmlGpuInstanceGetVgpuHeterogeneousMode;
using ::nvmlGpuInstanceGetVgpuSchedulerLog;
using ::nvmlGpuInstanceGetVgpuSchedulerState;
using ::nvmlGpuInstanceGetVgpuTypeCreatablePlacements;
using ::nvmlGpuInstanceSetVgpuHeterogeneousMode;
using ::nvmlGpuInstanceSetVgpuSchedulerState;
using ::nvmlSetVgpuVersion;
using ::nvmlVgpuInstanceClearAccountingPids;
using ::nvmlVgpuInstanceGetAccountingMode;
using ::nvmlVgpuInstanceGetAccountingPids;
using ::nvmlVgpuInstanceGetAccountingStats;
using ::nvmlVgpuInstanceGetEccMode;
using ::nvmlVgpuInstanceGetEncoderCapacity;
using ::nvmlVgpuInstanceGetEncoderSessions;
using ::nvmlVgpuInstanceGetEncoderStats;
using ::nvmlVgpuInstanceGetFBCSessions;
using ::nvmlVgpuInstanceGetFBCStats;
using ::nvmlVgpuInstanceGetFbUsage;
using ::nvmlVgpuInstanceGetFrameRateLimit;
using ::nvmlVgpuInstanceGetGpuInstanceId;
using ::nvmlVgpuInstanceGetGpuPciId;
using ::nvmlVgpuInstanceGetLicenseInfo_v2;
using ::nvmlVgpuInstanceGetLicenseStatus;
using ::nvmlVgpuInstanceGetMdevUUID;
using ::nvmlVgpuInstanceGetMetadata;
using ::nvmlVgpuInstanceGetPlacementId;
using ::nvmlVgpuInstanceGetRuntimeStateSize;
using ::nvmlVgpuInstanceGetType;
using ::nvmlVgpuInstanceGetUUID;
using ::nvmlVgpuInstanceGetVmDriverVersion;
using ::nvmlVgpuInstanceGetVmID;
using ::nvmlVgpuInstanceSetEncoderCapacity;
using ::nvmlVgpuTypeGetBAR1Info;
using ::nvmlVgpuTypeGetCapabilities;
using ::nvmlVgpuTypeGetClass;
using ::nvmlVgpuTypeGetDeviceID;
using ::nvmlVgpuTypeGetFbReservation;
using ::nvmlVgpuTypeGetFramebufferSize;
using ::nvmlVgpuTypeGetFrameRateLimit;
using ::nvmlVgpuTypeGetGpuInstanceProfileId;
using ::nvmlVgpuTypeGetGspHeapSize;
using ::nvmlVgpuTypeGetLicense;
using ::nvmlVgpuTypeGetMaxInstances;
using ::nvmlVgpuTypeGetMaxInstancesPerGpuInstance;
using ::nvmlVgpuTypeGetMaxInstancesPerVm;
using ::nvmlVgpuTypeGetName;
using ::nvmlVgpuTypeGetNumDisplayHeads;
using ::nvmlVgpuTypeGetResolution;

// ========================================================================
// Excluded (Blacklisted) Device Functions
// ========================================================================
using ::nvmlGetExcludedDeviceCount;
using ::nvmlGetExcludedDeviceInfoByIndex;

// ========================================================================
// PRM Functions
// ========================================================================
using ::nvmlDeviceReadWritePRM_v1;

// ========================================================================
// MIG (Multi-Instance GPU) Functions
// ========================================================================
using ::nvmlComputeInstanceDestroy;
using ::nvmlComputeInstanceGetInfo;
using ::nvmlComputeInstanceGetInfo_v2;
using ::nvmlDeviceCreateGpuInstance;
using ::nvmlDeviceCreateGpuInstanceWithPlacement;
using ::nvmlDeviceGetComputeInstanceId;
using ::nvmlDeviceGetDeviceHandleFromMigDeviceHandle;
using ::nvmlDeviceGetGpuInstanceById;
using ::nvmlDeviceGetGpuInstanceId;
using ::nvmlDeviceGetGpuInstancePossiblePlacements;
using ::nvmlDeviceGetGpuInstancePossiblePlacements_v2;
using ::nvmlDeviceGetGpuInstanceProfileInfo;
using ::nvmlDeviceGetGpuInstanceProfileInfoByIdV;
using ::nvmlDeviceGetGpuInstanceProfileInfoV;
using ::nvmlDeviceGetGpuInstanceRemainingCapacity;
using ::nvmlDeviceGetGpuInstances;
using ::nvmlDeviceGetMaxMigDeviceCount;
using ::nvmlDeviceGetMigDeviceHandleByIndex;
using ::nvmlDeviceGetMigMode;
using ::nvmlDeviceIsMigDeviceHandle;
using ::nvmlDeviceSetMigMode;
using ::nvmlGpuInstanceCreateComputeInstance;
using ::nvmlGpuInstanceCreateComputeInstanceWithPlacement;
using ::nvmlGpuInstanceDestroy;
using ::nvmlGpuInstanceGetComputeInstanceById;
using ::nvmlGpuInstanceGetComputeInstancePossiblePlacements;
using ::nvmlGpuInstanceGetComputeInstanceProfileInfo;
using ::nvmlGpuInstanceGetComputeInstanceProfileInfoV;
using ::nvmlGpuInstanceGetComputeInstanceRemainingCapacity;
using ::nvmlGpuInstanceGetComputeInstances;
using ::nvmlGpuInstanceGetInfo;

// ========================================================================
// GPM (GPU Performance Monitoring) Functions
// ========================================================================
using ::nvmlGpmMetricsGet;
using ::nvmlGpmMigSampleGet;
using ::nvmlGpmQueryDeviceSupport;
using ::nvmlGpmQueryIfStreamingEnabled;
using ::nvmlGpmSampleAlloc;
using ::nvmlGpmSampleFree;
using ::nvmlGpmSampleGet;
using ::nvmlGpmSetStreamingEnabled;

// ========================================================================
// Capabilities and Misc Device Functions
// ========================================================================
using ::nvmlDeviceGetCapabilities;
using ::nvmlDeviceGetConfComputeGpuAttestationReport;
using ::nvmlDeviceGetConfComputeGpuCertificate;
using ::nvmlDeviceGetConfComputeMemSizeInfo;
using ::nvmlDeviceGetConfComputeProtectedMemoryUsage;
using ::nvmlDeviceGetDriverModel;
using ::nvmlDeviceGetDriverModel_v2;
using ::nvmlDeviceGetGpuFabricInfo;
using ::nvmlDeviceGetGpuFabricInfoV;
using ::nvmlDeviceGetPlatformInfo;
using ::nvmlDeviceGetRowRemapperHistogram;
using ::nvmlDevicePowerSmoothingActivatePresetProfile;
using ::nvmlDevicePowerSmoothingUpdatePresetProfileParam;
using ::nvmlDeviceWorkloadPowerProfileClearRequestedProfiles;
using ::nvmlDeviceWorkloadPowerProfileGetCurrentProfiles;
using ::nvmlDeviceWorkloadPowerProfileGetProfilesInfo;
using ::nvmlDeviceWorkloadPowerProfileSetRequestedProfiles;
using ::nvmlSystemGetConfComputeCapabilities;
using ::nvmlSystemGetConfComputeKeyRotationThresholdInfo;
using ::nvmlSystemGetConfComputeSettings;
using ::nvmlSystemGetConfComputeState;
using ::nvmlSystemSetConfComputeKeyRotationThresholdInfo;

} // namespace wwr::cuda
