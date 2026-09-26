/**
 * @file amd_smi.cppm
 * @brief AMD SMI (System Management Interface) module wrapper for gpumod project
 *
 * Wraps amd_smi/amdsmi.h -- the newer AMD device management library, meant to
 * eventually supersede gpumod.hip.rocm_smi. Both are real, independently
 * usable libraries with overlapping but not identical surfaces, so each gets
 * its own module; see src/hip/README.md "Why nvml became two modules".
 *
 * Like rocm_smi.h it is a pure C API, and every type, enumerator and function
 * it declares is exported by name below.
 *
 * The CPU/ESMI extension surface (RAPL MSR energy counters, HSMP statistics,
 * boost-limit control, DDR bandwidth, dimm statistics, xGMI/GMI3 link width,
 * P-state selection) sits behind #ifdef ENABLE_ESMI_LIB, which this project
 * does not define -- consistent with the rest of src/hip targeting the GPU
 * surface only. Those declarations do not exist in this TU and so cannot be
 * exported.
 *
 * Usage:
 *   import gpumod.hip.amd_smi;
 */

module;

#include <amd_smi/amdsmi.h>

export module gpumod.hip.amd_smi;

export namespace gpumod::hip {

// ========================================================================
// Types and Enumerations
// ========================================================================

// amdsmi_init_flags_t
using ::AMDSMI_INIT_ALL_PROCESSORS;
using ::AMDSMI_INIT_AMD_APUS;
using ::AMDSMI_INIT_AMD_CPUS;
using ::AMDSMI_INIT_AMD_GPUS;
using ::amdsmi_init_flags_t;
using ::AMDSMI_INIT_NON_AMD_CPUS;
using ::AMDSMI_INIT_NON_AMD_GPUS;

// amdsmi_mm_ip_t
using ::AMDSMI_MM__MAX;
using ::amdsmi_mm_ip_t;
using ::AMDSMI_MM_UVD;
using ::AMDSMI_MM_VCE;
using ::AMDSMI_MM_VCN;

// amdsmi_container_types_t
using ::AMDSMI_CONTAINER_DOCKER;
using ::AMDSMI_CONTAINER_LXC;
using ::amdsmi_container_types_t;

using ::amdsmi_node_handle;
using ::amdsmi_processor_handle;
using ::amdsmi_socket_handle;
// processor_type_t
using ::AMDSMI_PROCESSOR_TYPE_AMD_APU;
using ::AMDSMI_PROCESSOR_TYPE_AMD_CPU;
using ::AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE;
using ::AMDSMI_PROCESSOR_TYPE_AMD_GPU;
using ::AMDSMI_PROCESSOR_TYPE_NON_AMD_CPU;
using ::AMDSMI_PROCESSOR_TYPE_NON_AMD_GPU;
using ::AMDSMI_PROCESSOR_TYPE_UNKNOWN;
using ::processor_type_t;

// amdsmi_status_t
using ::AMDSMI_STATUS_ADDRESS_FAULT;
using ::AMDSMI_STATUS_AMDGPU_RESTART_ERR;
using ::AMDSMI_STATUS_API_FAILED;
using ::AMDSMI_STATUS_ARG_PTR_NULL;
using ::AMDSMI_STATUS_BUSY;
using ::AMDSMI_STATUS_CORRUPTED_EEPROM;
using ::AMDSMI_STATUS_DIRECTORY_NOT_FOUND;
using ::AMDSMI_STATUS_DRIVER_NOT_LOADED;
using ::AMDSMI_STATUS_DRM_ERROR;
using ::AMDSMI_STATUS_FAIL_LOAD_MODULE;
using ::AMDSMI_STATUS_FAIL_LOAD_SYMBOL;
using ::AMDSMI_STATUS_FILE_ERROR;
using ::AMDSMI_STATUS_FILE_NOT_FOUND;
using ::AMDSMI_STATUS_HSMP_TIMEOUT;
using ::AMDSMI_STATUS_INIT_ERROR;
using ::AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
using ::AMDSMI_STATUS_INSUFFICIENT_SIZE;
using ::AMDSMI_STATUS_INTERNAL_EXCEPTION;
using ::AMDSMI_STATUS_INTERRUPT;
using ::AMDSMI_STATUS_INVAL;
using ::AMDSMI_STATUS_IO;
using ::AMDSMI_STATUS_MAP_ERROR;
using ::AMDSMI_STATUS_MORE_DATA;
using ::AMDSMI_STATUS_NO_DATA;
using ::AMDSMI_STATUS_NO_DRV;
using ::AMDSMI_STATUS_NO_ENERGY_DRV;
using ::AMDSMI_STATUS_NO_HSMP_DRV;
using ::AMDSMI_STATUS_NO_HSMP_MSG_SUP;
using ::AMDSMI_STATUS_NO_HSMP_SUP;
using ::AMDSMI_STATUS_NO_MSR_DRV;
using ::AMDSMI_STATUS_NO_PERM;
using ::AMDSMI_STATUS_NO_SLOT;
using ::AMDSMI_STATUS_NON_AMD_CPU;
using ::AMDSMI_STATUS_NOT_FOUND;
using ::AMDSMI_STATUS_NOT_INIT;
using ::AMDSMI_STATUS_NOT_SUPPORTED;
using ::AMDSMI_STATUS_NOT_YET_IMPLEMENTED;
using ::AMDSMI_STATUS_OUT_OF_RESOURCES;
using ::AMDSMI_STATUS_REFCOUNT_OVERFLOW;
using ::AMDSMI_STATUS_RETRY;
using ::AMDSMI_STATUS_SETTING_UNAVAILABLE;
using ::AMDSMI_STATUS_SUCCESS;
using ::amdsmi_status_t;
using ::AMDSMI_STATUS_TIMEOUT;
using ::AMDSMI_STATUS_UNEXPECTED_DATA;
using ::AMDSMI_STATUS_UNEXPECTED_SIZE;
using ::AMDSMI_STATUS_UNKNOWN_ERROR;

// amdsmi_clk_type_t
using ::AMDSMI_CLK_TYPE__MAX;
using ::AMDSMI_CLK_TYPE_DCEF;
using ::AMDSMI_CLK_TYPE_DCLK0;
using ::AMDSMI_CLK_TYPE_DCLK1;
using ::AMDSMI_CLK_TYPE_DF;
using ::AMDSMI_CLK_TYPE_FIRST;
using ::AMDSMI_CLK_TYPE_GFX;
using ::AMDSMI_CLK_TYPE_MEM;
using ::AMDSMI_CLK_TYPE_PCIE;
using ::AMDSMI_CLK_TYPE_SOC;
using ::AMDSMI_CLK_TYPE_SYS;
using ::amdsmi_clk_type_t;
using ::AMDSMI_CLK_TYPE_VCLK0;
using ::AMDSMI_CLK_TYPE_VCLK1;

// amdsmi_accelerator_partition_type_t
using ::AMDSMI_ACCELERATOR_PARTITION_CPX;
using ::AMDSMI_ACCELERATOR_PARTITION_DPX;
using ::AMDSMI_ACCELERATOR_PARTITION_INVALID;
using ::AMDSMI_ACCELERATOR_PARTITION_MAX;
using ::AMDSMI_ACCELERATOR_PARTITION_QPX;
using ::AMDSMI_ACCELERATOR_PARTITION_SPX;
using ::AMDSMI_ACCELERATOR_PARTITION_TPX;
using ::amdsmi_accelerator_partition_type_t;

// amdsmi_accelerator_partition_resource_type_t
using ::AMDSMI_ACCELERATOR_DECODER;
using ::AMDSMI_ACCELERATOR_DMA;
using ::AMDSMI_ACCELERATOR_ENCODER;
using ::AMDSMI_ACCELERATOR_JPEG;
using ::AMDSMI_ACCELERATOR_MAX;
using ::amdsmi_accelerator_partition_resource_type_t;
using ::AMDSMI_ACCELERATOR_XCC;

// amdsmi_compute_partition_type_t
using ::AMDSMI_COMPUTE_PARTITION_CPX;
using ::AMDSMI_COMPUTE_PARTITION_DPX;
using ::AMDSMI_COMPUTE_PARTITION_INVALID;
using ::AMDSMI_COMPUTE_PARTITION_QPX;
using ::AMDSMI_COMPUTE_PARTITION_SPX;
using ::AMDSMI_COMPUTE_PARTITION_TPX;
using ::amdsmi_compute_partition_type_t;

// amdsmi_memory_partition_type_t
using ::AMDSMI_MEMORY_PARTITION_NPS1;
using ::AMDSMI_MEMORY_PARTITION_NPS2;
using ::AMDSMI_MEMORY_PARTITION_NPS4;
using ::AMDSMI_MEMORY_PARTITION_NPS8;
using ::amdsmi_memory_partition_type_t;
using ::AMDSMI_MEMORY_PARTITION_UNKNOWN;

// amdsmi_temperature_type_t
using ::AMDSMI_TEMPERATURE_TYPE__MAX;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_FIRST;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_IBC;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_IBC_HSC;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_LAST;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_0_1_2_3_3V3_VR;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_0_1_HSC;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_2_3_HSC;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_4_5_6_7_3V3_VR;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_4_5_HSC;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_6_7_HSC;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_0_1_0V9_VR;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_0_1_2_3_1V2_VR;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_2_3_0V9_VR;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_4_5_0V9_VR;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_4_5_6_7_1V2_VR;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_6_7_0V9_VR;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_BACK;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA_0V72_VR;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA_3V3_VR;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FRONT;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_IBC;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_OAM1;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_OAM7;
using ::AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_UFPGA;
using ::AMDSMI_TEMPERATURE_TYPE_EDGE;
using ::AMDSMI_TEMPERATURE_TYPE_FIRST;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_FIRST;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_LAST;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_B_VR;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_D_VR;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_IBC;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_IBC_2;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_VDD18_VR;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_RETIMER_X;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDD_085_HBM;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDD_USR;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_11_HBM_B;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_11_HBM_D;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOC_A;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOC_C;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOCIO_A;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOCIO_C;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD0;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD1;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD2;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD3;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_11_E32;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VR_FIRST;
using ::AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VR_LAST;
using ::AMDSMI_TEMPERATURE_TYPE_HBM_0;
using ::AMDSMI_TEMPERATURE_TYPE_HBM_1;
using ::AMDSMI_TEMPERATURE_TYPE_HBM_2;
using ::AMDSMI_TEMPERATURE_TYPE_HBM_3;
using ::AMDSMI_TEMPERATURE_TYPE_HOTSPOT;
using ::AMDSMI_TEMPERATURE_TYPE_JUNCTION;
using ::AMDSMI_TEMPERATURE_TYPE_PLX;
using ::amdsmi_temperature_type_t;
using ::AMDSMI_TEMPERATURE_TYPE_VRAM;

// amdsmi_fw_block_t
using ::amdsmi_fw_block_t;
using ::AMDSMI_FW_ID__MAX;
using ::AMDSMI_FW_ID_ASD;
using ::AMDSMI_FW_ID_CP_CE;
using ::AMDSMI_FW_ID_CP_ME;
using ::AMDSMI_FW_ID_CP_MEC1;
using ::AMDSMI_FW_ID_CP_MEC2;
using ::AMDSMI_FW_ID_CP_MEC_JT1;
using ::AMDSMI_FW_ID_CP_MEC_JT2;
using ::AMDSMI_FW_ID_CP_MES;
using ::AMDSMI_FW_ID_CP_PFP;
using ::AMDSMI_FW_ID_CP_PM4;
using ::AMDSMI_FW_ID_DFC;
using ::AMDSMI_FW_ID_DMCU;
using ::AMDSMI_FW_ID_DMCU_ERAM;
using ::AMDSMI_FW_ID_DMCU_ISR;
using ::AMDSMI_FW_ID_DRV_CAP;
using ::AMDSMI_FW_ID_FIRST;
using ::AMDSMI_FW_ID_IMU_DRAM;
using ::AMDSMI_FW_ID_IMU_IRAM;
using ::AMDSMI_FW_ID_ISP;
using ::AMDSMI_FW_ID_MC;
using ::AMDSMI_FW_ID_MES_KIQ;
using ::AMDSMI_FW_ID_MES_STACK;
using ::AMDSMI_FW_ID_MES_THREAD1;
using ::AMDSMI_FW_ID_MES_THREAD1_STACK;
using ::AMDSMI_FW_ID_MMSCH;
using ::AMDSMI_FW_ID_PLDM_BUNDLE;
using ::AMDSMI_FW_ID_PM;
using ::AMDSMI_FW_ID_PPTABLE;
using ::AMDSMI_FW_ID_PSP_BL;
using ::AMDSMI_FW_ID_PSP_DBG;
using ::AMDSMI_FW_ID_PSP_INTF;
using ::AMDSMI_FW_ID_PSP_KEYDB;
using ::AMDSMI_FW_ID_PSP_SOC;
using ::AMDSMI_FW_ID_PSP_SOSDRV;
using ::AMDSMI_FW_ID_PSP_SPL;
using ::AMDSMI_FW_ID_PSP_SYSDRV;
using ::AMDSMI_FW_ID_PSP_TOC;
using ::AMDSMI_FW_ID_REG_ACCESS_WHITELIST;
using ::AMDSMI_FW_ID_RLC;
using ::AMDSMI_FW_ID_RLC_P;
using ::AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL;
using ::AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM;
using ::AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM;
using ::AMDSMI_FW_ID_RLC_SAVE_RESTORE_LIST;
using ::AMDSMI_FW_ID_RLC_SRLG;
using ::AMDSMI_FW_ID_RLC_SRLS;
using ::AMDSMI_FW_ID_RLC_V;
using ::AMDSMI_FW_ID_RLCV_LX7;
using ::AMDSMI_FW_ID_RLX6;
using ::AMDSMI_FW_ID_RLX6_CORE1;
using ::AMDSMI_FW_ID_RLX6_DRAM_BOOT;
using ::AMDSMI_FW_ID_RLX6_DRAM_BOOT_CORE1;
using ::AMDSMI_FW_ID_RS64_ME;
using ::AMDSMI_FW_ID_RS64_ME_P0_DATA;
using ::AMDSMI_FW_ID_RS64_ME_P1_DATA;
using ::AMDSMI_FW_ID_RS64_MEC;
using ::AMDSMI_FW_ID_RS64_MEC_P0_DATA;
using ::AMDSMI_FW_ID_RS64_MEC_P1_DATA;
using ::AMDSMI_FW_ID_RS64_MEC_P2_DATA;
using ::AMDSMI_FW_ID_RS64_MEC_P3_DATA;
using ::AMDSMI_FW_ID_RS64_PFP;
using ::AMDSMI_FW_ID_RS64_PFP_P0_DATA;
using ::AMDSMI_FW_ID_RS64_PFP_P1_DATA;
using ::AMDSMI_FW_ID_SDMA0;
using ::AMDSMI_FW_ID_SDMA1;
using ::AMDSMI_FW_ID_SDMA2;
using ::AMDSMI_FW_ID_SDMA3;
using ::AMDSMI_FW_ID_SDMA4;
using ::AMDSMI_FW_ID_SDMA5;
using ::AMDSMI_FW_ID_SDMA6;
using ::AMDSMI_FW_ID_SDMA7;
using ::AMDSMI_FW_ID_SDMA_TH0;
using ::AMDSMI_FW_ID_SDMA_TH1;
using ::AMDSMI_FW_ID_SEC_POLICY_STAGE2;
using ::AMDSMI_FW_ID_SMU;
using ::AMDSMI_FW_ID_TA_RAS;
using ::AMDSMI_FW_ID_TA_XGMI;
using ::AMDSMI_FW_ID_UVD;
using ::AMDSMI_FW_ID_VCE;
using ::AMDSMI_FW_ID_VCN;

// amdsmi_vram_type_t
using ::AMDSMI_VRAM_TYPE__MAX;
using ::AMDSMI_VRAM_TYPE_DDR2;
using ::AMDSMI_VRAM_TYPE_DDR3;
using ::AMDSMI_VRAM_TYPE_DDR4;
using ::AMDSMI_VRAM_TYPE_DDR5;
using ::AMDSMI_VRAM_TYPE_GDDR1;
using ::AMDSMI_VRAM_TYPE_GDDR2;
using ::AMDSMI_VRAM_TYPE_GDDR3;
using ::AMDSMI_VRAM_TYPE_GDDR4;
using ::AMDSMI_VRAM_TYPE_GDDR5;
using ::AMDSMI_VRAM_TYPE_GDDR6;
using ::AMDSMI_VRAM_TYPE_GDDR7;
using ::AMDSMI_VRAM_TYPE_HBM;
using ::AMDSMI_VRAM_TYPE_HBM2;
using ::AMDSMI_VRAM_TYPE_HBM2E;
using ::AMDSMI_VRAM_TYPE_HBM3;
using ::AMDSMI_VRAM_TYPE_HBM3E;
using ::AMDSMI_VRAM_TYPE_LPDDR4;
using ::AMDSMI_VRAM_TYPE_LPDDR5;
using ::amdsmi_vram_type_t;
using ::AMDSMI_VRAM_TYPE_UNKNOWN;

using ::amdsmi_bdf_t;
using ::amdsmi_enumeration_info_t;
using ::amdsmi_frequency_range_t;
using ::amdsmi_range_t;
using ::amdsmi_violation_status_t;
using ::amdsmi_vram_usage_t;
using ::amdsmi_xgmi_info_t;
// amdsmi_card_form_factor_t
using ::AMDSMI_CARD_FORM_FACTOR_CEM;
using ::AMDSMI_CARD_FORM_FACTOR_OAM;
using ::AMDSMI_CARD_FORM_FACTOR_PCIE;
using ::amdsmi_card_form_factor_t;
using ::AMDSMI_CARD_FORM_FACTOR_UNKNOWN;

using ::amdsmi_pcie_info_t;
using ::amdsmi_power_cap_info_t;
// amdsmi_power_cap_type_t
using ::AMDSMI_POWER_CAP_TYPE_PPT0;
using ::AMDSMI_POWER_CAP_TYPE_PPT1;
using ::amdsmi_power_cap_type_t;

using ::amdsmi_vbios_info_t;
// amdsmi_cache_property_type_t
using ::AMDSMI_CACHE_PROPERTY_CPU_CACHE;
using ::AMDSMI_CACHE_PROPERTY_DATA_CACHE;
using ::AMDSMI_CACHE_PROPERTY_ENABLED;
using ::AMDSMI_CACHE_PROPERTY_INST_CACHE;
using ::AMDSMI_CACHE_PROPERTY_SIMD_CACHE;
using ::amdsmi_cache_property_type_t;

using ::amdsmi_accelerator_partition_profile_config_t;
using ::amdsmi_accelerator_partition_profile_t;
using ::amdsmi_accelerator_partition_resource_profile_t;
using ::amdsmi_asic_info_t;
using ::amdsmi_fw_info_t;
using ::amdsmi_gpu_cache_info_t;
using ::amdsmi_kfd_info_t;
using ::amdsmi_memory_partition_config_t;
using ::amdsmi_nps_caps_t;
// amdsmi_link_type_t
using ::AMDSMI_LINK_TYPE_INTERNAL;
using ::AMDSMI_LINK_TYPE_NOT_APPLICABLE;
using ::AMDSMI_LINK_TYPE_PCIE;
using ::amdsmi_link_type_t;
using ::AMDSMI_LINK_TYPE_UNKNOWN;
using ::AMDSMI_LINK_TYPE_XGMI;

using ::amdsmi_board_info_t;
using ::amdsmi_clk_info_t;
using ::amdsmi_cpu_util_t;
using ::amdsmi_driver_info_t;
using ::amdsmi_engine_usage_t;
using ::amdsmi_link_metrics_t;
using ::amdsmi_p2p_capability_t;
using ::amdsmi_power_info_t;
using ::amdsmi_proc_info_t;
using ::amdsmi_process_handle_t;
using ::amdsmi_vram_info_t;
// amdsmi_dev_perf_level_t
using ::AMDSMI_DEV_PERF_LEVEL_AUTO;
using ::AMDSMI_DEV_PERF_LEVEL_DETERMINISM;
using ::AMDSMI_DEV_PERF_LEVEL_FIRST;
using ::AMDSMI_DEV_PERF_LEVEL_HIGH;
using ::AMDSMI_DEV_PERF_LEVEL_LAST;
using ::AMDSMI_DEV_PERF_LEVEL_LOW;
using ::AMDSMI_DEV_PERF_LEVEL_MANUAL;
using ::AMDSMI_DEV_PERF_LEVEL_STABLE_MIN_MCLK;
using ::AMDSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK;
using ::AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK;
using ::AMDSMI_DEV_PERF_LEVEL_STABLE_STD;
using ::amdsmi_dev_perf_level_t;
using ::AMDSMI_DEV_PERF_LEVEL_UNKNOWN;

using ::amdsmi_event_handle_t;
// amdsmi_event_group_t
using ::amdsmi_event_group_t;
using ::AMDSMI_EVNT_GRP_INVALID;
using ::AMDSMI_EVNT_GRP_XGMI;
using ::AMDSMI_EVNT_GRP_XGMI_DATA_OUT;

// amdsmi_event_type_t
using ::amdsmi_event_type_t;
using ::AMDSMI_EVNT_FIRST;
using ::AMDSMI_EVNT_LAST;
using ::AMDSMI_EVNT_XGMI_0_BEATS_TX;
using ::AMDSMI_EVNT_XGMI_0_NOP_TX;
using ::AMDSMI_EVNT_XGMI_0_REQUEST_TX;
using ::AMDSMI_EVNT_XGMI_0_RESPONSE_TX;
using ::AMDSMI_EVNT_XGMI_1_BEATS_TX;
using ::AMDSMI_EVNT_XGMI_1_NOP_TX;
using ::AMDSMI_EVNT_XGMI_1_REQUEST_TX;
using ::AMDSMI_EVNT_XGMI_1_RESPONSE_TX;
using ::AMDSMI_EVNT_XGMI_DATA_OUT_0;
using ::AMDSMI_EVNT_XGMI_DATA_OUT_1;
using ::AMDSMI_EVNT_XGMI_DATA_OUT_2;
using ::AMDSMI_EVNT_XGMI_DATA_OUT_3;
using ::AMDSMI_EVNT_XGMI_DATA_OUT_4;
using ::AMDSMI_EVNT_XGMI_DATA_OUT_5;
using ::AMDSMI_EVNT_XGMI_DATA_OUT_FIRST;
using ::AMDSMI_EVNT_XGMI_DATA_OUT_LAST;
using ::AMDSMI_EVNT_XGMI_FIRST;
using ::AMDSMI_EVNT_XGMI_LAST;

// amdsmi_counter_command_t
using ::AMDSMI_CNTR_CMD_START;
using ::AMDSMI_CNTR_CMD_STOP;
using ::amdsmi_counter_command_t;

using ::amdsmi_counter_value_t;
// amdsmi_evt_notification_type_t
using ::AMDSMI_EVT_NOTIF_FIRST;
using ::AMDSMI_EVT_NOTIF_GPU_POST_RESET;
using ::AMDSMI_EVT_NOTIF_GPU_PRE_RESET;
using ::AMDSMI_EVT_NOTIF_LAST;
using ::AMDSMI_EVT_NOTIF_MIGRATE_END;
using ::AMDSMI_EVT_NOTIF_MIGRATE_START;
using ::AMDSMI_EVT_NOTIF_NONE;
using ::AMDSMI_EVT_NOTIF_PAGE_FAULT_END;
using ::AMDSMI_EVT_NOTIF_PAGE_FAULT_START;
using ::AMDSMI_EVT_NOTIF_PROCESS_END;
using ::AMDSMI_EVT_NOTIF_PROCESS_START;
using ::AMDSMI_EVT_NOTIF_QUEUE_EVICTION;
using ::AMDSMI_EVT_NOTIF_QUEUE_RESTORE;
using ::AMDSMI_EVT_NOTIF_THERMAL_THROTTLE;
using ::AMDSMI_EVT_NOTIF_UNMAP_FROM_GPU;
using ::AMDSMI_EVT_NOTIF_VMFAULT;
using ::amdsmi_evt_notification_type_t;

using ::amdsmi_evt_notification_data_t;
// amdsmi_temperature_metric_t
using ::AMDSMI_TEMP_CRIT_MIN;
using ::AMDSMI_TEMP_CRIT_MIN_HYST;
using ::AMDSMI_TEMP_CRITICAL;
using ::AMDSMI_TEMP_CRITICAL_HYST;
using ::AMDSMI_TEMP_CURRENT;
using ::AMDSMI_TEMP_EMERGENCY;
using ::AMDSMI_TEMP_EMERGENCY_HYST;
using ::AMDSMI_TEMP_FIRST;
using ::AMDSMI_TEMP_HIGHEST;
using ::AMDSMI_TEMP_LAST;
using ::AMDSMI_TEMP_LOWEST;
using ::AMDSMI_TEMP_MAX;
using ::AMDSMI_TEMP_MAX_HYST;
using ::AMDSMI_TEMP_MIN;
using ::AMDSMI_TEMP_MIN_HYST;
using ::AMDSMI_TEMP_OFFSET;
using ::AMDSMI_TEMP_SHUTDOWN;
using ::amdsmi_temperature_metric_t;

// amdsmi_voltage_metric_t
using ::AMDSMI_VOLT_AVERAGE;
using ::AMDSMI_VOLT_CURRENT;
using ::AMDSMI_VOLT_FIRST;
using ::AMDSMI_VOLT_HIGHEST;
using ::AMDSMI_VOLT_LAST;
using ::AMDSMI_VOLT_LOWEST;
using ::AMDSMI_VOLT_MAX;
using ::AMDSMI_VOLT_MAX_CRIT;
using ::AMDSMI_VOLT_MIN;
using ::AMDSMI_VOLT_MIN_CRIT;
using ::amdsmi_voltage_metric_t;

// amdsmi_voltage_type_t
using ::AMDSMI_VOLT_TYPE_FIRST;
using ::AMDSMI_VOLT_TYPE_INVALID;
using ::AMDSMI_VOLT_TYPE_LAST;
using ::AMDSMI_VOLT_TYPE_VDDBOARD;
using ::AMDSMI_VOLT_TYPE_VDDGFX;
using ::amdsmi_voltage_type_t;

// amdsmi_power_profile_preset_masks_t
using ::amdsmi_power_profile_preset_masks_t;
using ::AMDSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK;
using ::AMDSMI_PWR_PROF_PRST_BOOTUP_DEFAULT;
using ::AMDSMI_PWR_PROF_PRST_COMPUTE_MASK;
using ::AMDSMI_PWR_PROF_PRST_CUSTOM_MASK;
using ::AMDSMI_PWR_PROF_PRST_INVALID;
using ::AMDSMI_PWR_PROF_PRST_LAST;
using ::AMDSMI_PWR_PROF_PRST_POWER_SAVING_MASK;
using ::AMDSMI_PWR_PROF_PRST_VIDEO_MASK;
using ::AMDSMI_PWR_PROF_PRST_VR_MASK;

// amdsmi_gpu_block_t
using ::AMDSMI_GPU_BLOCK_ATHUB;
using ::AMDSMI_GPU_BLOCK_DF;
using ::AMDSMI_GPU_BLOCK_FIRST;
using ::AMDSMI_GPU_BLOCK_FUSE;
using ::AMDSMI_GPU_BLOCK_GFX;
using ::AMDSMI_GPU_BLOCK_HDP;
using ::AMDSMI_GPU_BLOCK_IH;
using ::AMDSMI_GPU_BLOCK_INVALID;
using ::AMDSMI_GPU_BLOCK_JPEG;
using ::AMDSMI_GPU_BLOCK_LAST;
using ::AMDSMI_GPU_BLOCK_MCA;
using ::AMDSMI_GPU_BLOCK_MMHUB;
using ::AMDSMI_GPU_BLOCK_MP0;
using ::AMDSMI_GPU_BLOCK_MP1;
using ::AMDSMI_GPU_BLOCK_MPIO;
using ::AMDSMI_GPU_BLOCK_PCIE_BIF;
using ::AMDSMI_GPU_BLOCK_RESERVED;
using ::AMDSMI_GPU_BLOCK_SDMA;
using ::AMDSMI_GPU_BLOCK_SEM;
using ::AMDSMI_GPU_BLOCK_SMN;
using ::amdsmi_gpu_block_t;
using ::AMDSMI_GPU_BLOCK_UMC;
using ::AMDSMI_GPU_BLOCK_VCN;
using ::AMDSMI_GPU_BLOCK_XGMI_WAFL;

// amdsmi_clk_limit_type_t
using ::amdsmi_clk_limit_type_t;
using ::CLK_LIMIT_MAX;
using ::CLK_LIMIT_MIN;

// amdsmi_cper_sev_t
using ::AMDSMI_CPER_SEV_FATAL;
using ::AMDSMI_CPER_SEV_NON_FATAL_CORRECTED;
using ::AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED;
using ::AMDSMI_CPER_SEV_NUM;
using ::amdsmi_cper_sev_t;
using ::AMDSMI_CPER_SEV_UNUSED;

// amdsmi_cper_notify_type_t
using ::AMDSMI_CPER_NOTIFY_TYPE_BOOT;
using ::AMDSMI_CPER_NOTIFY_TYPE_CMC;
using ::AMDSMI_CPER_NOTIFY_TYPE_CPE;
using ::AMDSMI_CPER_NOTIFY_TYPE_CXL_COMPONENT;
using ::AMDSMI_CPER_NOTIFY_TYPE_DMAR;
using ::AMDSMI_CPER_NOTIFY_TYPE_INIT;
using ::AMDSMI_CPER_NOTIFY_TYPE_MCE;
using ::AMDSMI_CPER_NOTIFY_TYPE_NMI;
using ::AMDSMI_CPER_NOTIFY_TYPE_PCIE;
using ::AMDSMI_CPER_NOTIFY_TYPE_PEI;
using ::AMDSMI_CPER_NOTIFY_TYPE_SEA;
using ::AMDSMI_CPER_NOTIFY_TYPE_SEI;
using ::amdsmi_cper_notify_type_t;

// amdsmi_ras_err_state_t
using ::AMDSMI_RAS_ERR_STATE_DISABLED;
using ::AMDSMI_RAS_ERR_STATE_ENABLED;
using ::AMDSMI_RAS_ERR_STATE_INVALID;
using ::AMDSMI_RAS_ERR_STATE_LAST;
using ::AMDSMI_RAS_ERR_STATE_MULT_UC;
using ::AMDSMI_RAS_ERR_STATE_NONE;
using ::AMDSMI_RAS_ERR_STATE_PARITY;
using ::AMDSMI_RAS_ERR_STATE_POISON;
using ::AMDSMI_RAS_ERR_STATE_SING_C;
using ::amdsmi_ras_err_state_t;

// amdsmi_memory_type_t
using ::AMDSMI_MEM_TYPE_FIRST;
using ::AMDSMI_MEM_TYPE_GTT;
using ::AMDSMI_MEM_TYPE_LAST;
using ::AMDSMI_MEM_TYPE_VIS_VRAM;
using ::AMDSMI_MEM_TYPE_VRAM;
using ::amdsmi_memory_type_t;

// amdsmi_freq_ind_t
using ::AMDSMI_FREQ_IND_INVALID;
using ::AMDSMI_FREQ_IND_MAX;
using ::AMDSMI_FREQ_IND_MIN;
using ::amdsmi_freq_ind_t;

// amdsmi_xgmi_status_t
using ::AMDSMI_XGMI_STATUS_ERROR;
using ::AMDSMI_XGMI_STATUS_MULTIPLE_ERRORS;
using ::AMDSMI_XGMI_STATUS_NO_ERRORS;
using ::amdsmi_xgmi_status_t;

using ::amdsmi_bit_field_t;
// amdsmi_memory_page_status_t
using ::AMDSMI_MEM_PAGE_STATUS_PENDING;
using ::AMDSMI_MEM_PAGE_STATUS_RESERVED;
using ::AMDSMI_MEM_PAGE_STATUS_UNRESERVABLE;
using ::amdsmi_memory_page_status_t;

// amdsmi_utilization_counter_type_t
using ::AMDSMI_COARSE_DECODER_ACTIVITY;
using ::AMDSMI_COARSE_GRAIN_GFX_ACTIVITY;
using ::AMDSMI_COARSE_GRAIN_MEM_ACTIVITY;
using ::AMDSMI_FINE_DECODER_ACTIVITY;
using ::AMDSMI_FINE_GRAIN_GFX_ACTIVITY;
using ::AMDSMI_FINE_GRAIN_MEM_ACTIVITY;
using ::AMDSMI_UTILIZATION_COUNTER_FIRST;
using ::AMDSMI_UTILIZATION_COUNTER_LAST;
using ::amdsmi_utilization_counter_type_t;

using ::amd_metrics_table_header_t;
using ::amdsmi_dpm_policy_entry_t;
using ::amdsmi_dpm_policy_t;
using ::amdsmi_freq_volt_region_t;
using ::amdsmi_frequencies_t;
using ::amdsmi_gpu_metrics_t;
using ::amdsmi_gpu_xcp_metrics_t;
using ::amdsmi_od_vddc_point_t;
using ::amdsmi_od_volt_curve_t;
using ::amdsmi_od_volt_freq_data_t;
using ::amdsmi_pcie_bandwidth_t;
using ::amdsmi_power_profile_status_t;
using ::amdsmi_retired_page_record_t;
using ::amdsmi_utilization_counter_t;
using ::amdsmi_version_t;
// amdsmi_xgmi_link_status_type_t
using ::AMDSMI_XGMI_LINK_DISABLE;
using ::AMDSMI_XGMI_LINK_DOWN;
using ::amdsmi_xgmi_link_status_type_t;
using ::AMDSMI_XGMI_LINK_UP;

using ::amdsmi_name_value_t;
using ::amdsmi_xgmi_link_status_t;
// amdsmi_reg_type_t
using ::AMDSMI_REG_PCIE;
using ::amdsmi_reg_type_t;
using ::AMDSMI_REG_USR;
using ::AMDSMI_REG_USR1;
using ::AMDSMI_REG_WAFL;
using ::AMDSMI_REG_XGMI;

using ::amdsmi_error_count_t;
using ::amdsmi_process_info_t;
using ::amdsmi_ras_feature_t;
using ::amdsmi_topology_nearest_t;
// amdsmi_virtualization_mode_t
using ::AMDSMI_VIRTUALIZATION_MODE_BAREMETAL;
using ::AMDSMI_VIRTUALIZATION_MODE_GUEST;
using ::AMDSMI_VIRTUALIZATION_MODE_HOST;
using ::AMDSMI_VIRTUALIZATION_MODE_PASSTHROUGH;
using ::amdsmi_virtualization_mode_t;
using ::AMDSMI_VIRTUALIZATION_MODE_UNKNOWN;

// amdsmi_affinity_scope_t
using ::AMDSMI_AFFINITY_SCOPE_NODE;
using ::AMDSMI_AFFINITY_SCOPE_SOCKET;
using ::amdsmi_affinity_scope_t;

// amdsmi_npm_status_t
using ::AMDSMI_NPM_STATUS_DISABLED;
using ::AMDSMI_NPM_STATUS_ENABLED;
using ::amdsmi_npm_status_t;

using ::amdsmi_npm_info_t;
// amdsmi_ptl_data_format_t
using ::AMDSMI_PTL_DATA_FORMAT_BF16;
using ::AMDSMI_PTL_DATA_FORMAT_F16;
using ::AMDSMI_PTL_DATA_FORMAT_F32;
using ::AMDSMI_PTL_DATA_FORMAT_F64;
using ::AMDSMI_PTL_DATA_FORMAT_F8;
using ::AMDSMI_PTL_DATA_FORMAT_I8;
using ::AMDSMI_PTL_DATA_FORMAT_INVALID;
using ::amdsmi_ptl_data_format_t;
using ::AMDSMI_PTL_DATA_FORMAT_VECTOR;

using ::amdsmi_cper_guid_t;
using ::amdsmi_cper_hdr_t;
using ::amdsmi_cper_timestamp_t;
using ::amdsmi_cper_valid_bits_t;
using ::amdsmi_sock_info_t;

// ========================================================================
// Functions
// ========================================================================

// Initialization and Shutdown
using ::amdsmi_init;
using ::amdsmi_shut_down;

// Discovery Queries
using ::amdsmi_get_cpu_affinity_with_scope;
using ::amdsmi_get_gpu_device_bdf;
using ::amdsmi_get_gpu_device_uuid;
using ::amdsmi_get_gpu_enumeration_info;
using ::amdsmi_get_gpu_virtualization_mode;
using ::amdsmi_get_node_handle;
using ::amdsmi_get_processor_handle_from_bdf;
using ::amdsmi_get_processor_handles;
using ::amdsmi_get_processor_type;
using ::amdsmi_get_socket_handles;
using ::amdsmi_get_socket_info;

// Identifier Queries
using ::amdsmi_get_gpu_id;
using ::amdsmi_get_gpu_revision;
using ::amdsmi_get_gpu_subsystem_id;
using ::amdsmi_get_gpu_subsystem_name;
using ::amdsmi_get_gpu_vendor_name;
using ::amdsmi_get_gpu_vram_vendor;

// PCIe Queries
using ::amdsmi_get_gpu_bdf_id;
using ::amdsmi_get_gpu_pci_bandwidth;
using ::amdsmi_get_gpu_pci_replay_counter;
using ::amdsmi_get_gpu_pci_throughput;
using ::amdsmi_get_gpu_topo_numa_affinity;

// PCIe Control
using ::amdsmi_set_gpu_pci_bandwidth;

// Power Queries
using ::amdsmi_get_energy_count;

// Power Control
using ::amdsmi_get_cpu_pwr_svi_telemetry_all_rails;
using ::amdsmi_get_cpu_socket_power;
using ::amdsmi_get_cpu_socket_power_cap;
using ::amdsmi_get_cpu_socket_power_cap_max;
using ::amdsmi_get_supported_power_cap;
using ::amdsmi_set_cpu_pwr_efficiency_mode;
using ::amdsmi_set_cpu_socket_power_cap;
using ::amdsmi_set_gpu_power_profile;
using ::amdsmi_set_power_cap;

// Memory Queries
using ::amdsmi_get_gpu_bad_page_info;
using ::amdsmi_get_gpu_bad_page_threshold;
using ::amdsmi_get_gpu_memory_reserved_pages;
using ::amdsmi_get_gpu_memory_total;
using ::amdsmi_get_gpu_memory_usage;
using ::amdsmi_get_gpu_ras_block_features_enabled;
using ::amdsmi_gpu_validate_ras_eeprom;

// Physical State Queries
using ::amdsmi_get_gpu_cache_info;
using ::amdsmi_get_gpu_fan_rpms;
using ::amdsmi_get_gpu_fan_speed;
using ::amdsmi_get_gpu_fan_speed_max;
using ::amdsmi_get_gpu_volt_metric;

// Physical State Control
using ::amdsmi_reset_gpu_fan;
using ::amdsmi_set_gpu_fan_speed;

// Clock, Power and Performance Queries
using ::amdsmi_get_clk_freq;
using ::amdsmi_get_gpu_busy_percent;
using ::amdsmi_get_gpu_mem_overdrive_level;
using ::amdsmi_get_gpu_metrics_header_info;
using ::amdsmi_get_gpu_metrics_info;
using ::amdsmi_get_gpu_od_volt_curve_regions;
using ::amdsmi_get_gpu_od_volt_info;
using ::amdsmi_get_gpu_overdrive_level;
using ::amdsmi_get_gpu_partition_metrics_info;
using ::amdsmi_get_gpu_perf_level;
using ::amdsmi_get_gpu_pm_metrics_info;
using ::amdsmi_get_gpu_power_profile_presets;
using ::amdsmi_get_gpu_reg_table_info;
using ::amdsmi_get_utilization_count;
using ::amdsmi_reset_gpu;
using ::amdsmi_set_gpu_clk_limit;
using ::amdsmi_set_gpu_clk_range;
using ::amdsmi_set_gpu_od_clk_info;
using ::amdsmi_set_gpu_od_volt_info;
using ::amdsmi_set_gpu_perf_determinism_mode;

// Clock, Power and Performance Control
using ::amdsmi_clean_gpu_local_data;
using ::amdsmi_get_gpu_process_isolation;
using ::amdsmi_get_soc_pstate;
using ::amdsmi_get_xgmi_plpd;
using ::amdsmi_set_clk_freq;
using ::amdsmi_set_gpu_overdrive_level;
using ::amdsmi_set_gpu_perf_level;
using ::amdsmi_set_gpu_process_isolation;
using ::amdsmi_set_soc_pstate;
using ::amdsmi_set_xgmi_plpd;

// Version Queries
using ::amdsmi_get_lib_version;

// ECC Information
using ::amdsmi_get_gpu_cper_entries;
using ::amdsmi_get_gpu_ecc_count;
using ::amdsmi_get_gpu_ecc_enabled;
using ::amdsmi_get_gpu_total_ecc_count;

// RAS information
using ::amdsmi_get_afids_from_cper;
using ::amdsmi_get_gpu_ras_feature_info;

// Error Queries
using ::amdsmi_get_gpu_ecc_status;
using ::amdsmi_status_code_to_string;

// Performance Counter Functions
using ::amdsmi_get_gpu_available_counters;
using ::amdsmi_gpu_control_counter;
using ::amdsmi_gpu_counter_group_supported;
using ::amdsmi_gpu_create_counter;
using ::amdsmi_gpu_destroy_counter;
using ::amdsmi_gpu_read_counter;

// System Information Functions
using ::amdsmi_get_gpu_compute_process_gpus;
using ::amdsmi_get_gpu_compute_process_info;
using ::amdsmi_get_gpu_compute_process_info_by_pid;

// XGMI Functions
using ::amdsmi_get_gpu_xgmi_link_status;
using ::amdsmi_get_xgmi_info;
using ::amdsmi_gpu_xgmi_error_status;
using ::amdsmi_reset_gpu_xgmi_error;

// Hardware Topology Functions
using ::amdsmi_get_link_metrics;
using ::amdsmi_get_link_topology_nearest;
using ::amdsmi_get_minmax_bandwidth_between_processors;
using ::amdsmi_is_P2P_accessible;
using ::amdsmi_topo_get_link_type;
using ::amdsmi_topo_get_link_weight;
using ::amdsmi_topo_get_numa_node_number;
using ::amdsmi_topo_get_p2p_status;

// Compute Partition Functions
using ::amdsmi_get_gpu_compute_partition;
using ::amdsmi_set_gpu_compute_partition;

// Memory Partition Functions
using ::amdsmi_get_gpu_memory_partition;
using ::amdsmi_get_gpu_memory_partition_config;
using ::amdsmi_set_gpu_memory_partition;
using ::amdsmi_set_gpu_memory_partition_mode;

// Accelerator Partition Profile Functions
using ::amdsmi_get_gpu_accelerator_partition_profile;
using ::amdsmi_get_gpu_accelerator_partition_profile_config;
using ::amdsmi_set_gpu_accelerator_partition_profile;

// Event Notification Functions
using ::amdsmi_get_gpu_event_notification;
using ::amdsmi_init_gpu_event_notification;
using ::amdsmi_set_gpu_event_notification_mask;
using ::amdsmi_stop_gpu_event_notification;

// Software Version Information
using ::amdsmi_get_gpu_driver_info;

// ASIC & Board Static Information
using ::amdsmi_get_gpu_asic_info;
using ::amdsmi_get_gpu_board_info;
using ::amdsmi_get_gpu_kfd_info;
using ::amdsmi_get_gpu_vram_info;
using ::amdsmi_get_gpu_xcd_counter;
using ::amdsmi_get_npm_info;
using ::amdsmi_get_pcie_info;
using ::amdsmi_get_power_cap_info;

// Firmware & VBIOS queries
using ::amdsmi_get_fw_info;
using ::amdsmi_get_gpu_vbios_info;

// GPU Monitoring
using ::amdsmi_get_clock_info;
using ::amdsmi_get_gpu_activity;
using ::amdsmi_get_gpu_vram_usage;
using ::amdsmi_get_power_info;
using ::amdsmi_get_temp_metric;
using ::amdsmi_get_violation_status;
using ::amdsmi_is_gpu_power_management_enabled;

// Process information
using ::amdsmi_get_gpu_process_list;

// Driver control mechanisms
using ::amdsmi_gpu_driver_reload;

// Peak Tops Limiter
using ::amdsmi_get_gpu_ptl_formats;
using ::amdsmi_get_gpu_ptl_state;
using ::amdsmi_set_gpu_ptl_formats;
using ::amdsmi_set_gpu_ptl_state;

} // namespace gpumod::hip
