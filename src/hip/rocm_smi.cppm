/**
 * @file rocm_smi.cppm
 * @brief ROCm SMI (System Management Interface) module wrapper for gpumod project
 *
 * This module wraps rocm_smi/rocm_smi.h -- the legacy/stable AMD GPU device
 * management and monitoring library. See src/hip/README.md "Why nvml became two
 * modules" for how this relates to gpumod.hip.amd_smi, the newer counterpart.
 *
 * Usage:
 *   import gpumod.hip.rocm_smi;
 *
 * rocm_smi.h is a pure C API (its whole body is wrapped in extern "C", including
 * the transitively-included <cstdint>). Every type, enumerator, and function it
 * declares is exported by name below via `using` declarations -- no convenience
 * templates or macros need special handling here (unlike hip_runtime_api.h).
 */

module;

#include <rocm_smi/rocm_smi.h>

export module gpumod.hip.rocm_smi;

export namespace wwr::hip {

// ========================================================================
// Types and Enumerations
// ========================================================================

// rsmi_status_t
using ::RSMI_INITIALIZATION_ERROR;
using ::RSMI_STATUS_AMDGPU_RESTART_ERR;
using ::RSMI_STATUS_BUSY;
using ::RSMI_STATUS_DRM_ERROR;
using ::RSMI_STATUS_FAIL_LOAD_MODULE;
using ::RSMI_STATUS_FAIL_LOAD_SYMBOL;
using ::RSMI_STATUS_FILE_ERROR;
using ::RSMI_STATUS_INIT_ERROR;
using ::RSMI_STATUS_INPUT_OUT_OF_BOUNDS;
using ::RSMI_STATUS_INSUFFICIENT_SIZE;
using ::RSMI_STATUS_INTERNAL_EXCEPTION;
using ::RSMI_STATUS_INTERRUPT;
using ::RSMI_STATUS_INVALID_ARGS;
using ::RSMI_STATUS_NO_DATA;
using ::RSMI_STATUS_NOT_FOUND;
using ::RSMI_STATUS_NOT_SUPPORTED;
using ::RSMI_STATUS_NOT_YET_IMPLEMENTED;
using ::RSMI_STATUS_OUT_OF_RESOURCES;
using ::RSMI_STATUS_PERMISSION;
using ::RSMI_STATUS_REFCOUNT_OVERFLOW;
using ::RSMI_STATUS_SETTING_UNAVAILABLE;
using ::RSMI_STATUS_SUCCESS;
using ::rsmi_status_t;
using ::RSMI_STATUS_UNEXPECTED_DATA;
using ::RSMI_STATUS_UNEXPECTED_SIZE;
using ::RSMI_STATUS_UNKNOWN_ERROR;

// rsmi_init_flags_t
using ::RSMI_INIT_FLAG_ALL_GPUS;
using ::RSMI_INIT_FLAG_RESRV_TEST1;
using ::RSMI_INIT_FLAG_THRAD_ONLY_MUTEX;
using ::rsmi_init_flags_t;

// rsmi_dev_perf_level_t
using ::RSMI_DEV_PERF_LEVEL_AUTO;
using ::RSMI_DEV_PERF_LEVEL_DETERMINISM;
using ::RSMI_DEV_PERF_LEVEL_FIRST;
using ::RSMI_DEV_PERF_LEVEL_HIGH;
using ::RSMI_DEV_PERF_LEVEL_LAST;
using ::RSMI_DEV_PERF_LEVEL_LOW;
using ::RSMI_DEV_PERF_LEVEL_MANUAL;
using ::RSMI_DEV_PERF_LEVEL_STABLE_MIN_MCLK;
using ::RSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK;
using ::RSMI_DEV_PERF_LEVEL_STABLE_PEAK;
using ::RSMI_DEV_PERF_LEVEL_STABLE_STD;
using ::rsmi_dev_perf_level_t;
using ::RSMI_DEV_PERF_LEVEL_UNKNOWN;

using ::rsmi_dev_perf_level;
// rsmi_sw_component_t
using ::RSMI_SW_COMP_DRIVER;
using ::RSMI_SW_COMP_FIRST;
using ::RSMI_SW_COMP_LAST;
using ::rsmi_sw_component_t;

using ::rsmi_event_handle_t;
// rsmi_event_group_t
using ::rsmi_event_group_t;
using ::RSMI_EVNT_GRP_INVALID;
using ::RSMI_EVNT_GRP_XGMI;
using ::RSMI_EVNT_GRP_XGMI_DATA_OUT;

// rsmi_event_type_t
using ::rsmi_event_type_t;
using ::RSMI_EVNT_FIRST;
using ::RSMI_EVNT_LAST;
using ::RSMI_EVNT_XGMI_0_BEATS_TX;
using ::RSMI_EVNT_XGMI_0_NOP_TX;
using ::RSMI_EVNT_XGMI_0_REQUEST_TX;
using ::RSMI_EVNT_XGMI_0_RESPONSE_TX;
using ::RSMI_EVNT_XGMI_1_BEATS_TX;
using ::RSMI_EVNT_XGMI_1_NOP_TX;
using ::RSMI_EVNT_XGMI_1_REQUEST_TX;
using ::RSMI_EVNT_XGMI_1_RESPONSE_TX;
using ::RSMI_EVNT_XGMI_DATA_OUT_0;
using ::RSMI_EVNT_XGMI_DATA_OUT_1;
using ::RSMI_EVNT_XGMI_DATA_OUT_2;
using ::RSMI_EVNT_XGMI_DATA_OUT_3;
using ::RSMI_EVNT_XGMI_DATA_OUT_4;
using ::RSMI_EVNT_XGMI_DATA_OUT_5;
using ::RSMI_EVNT_XGMI_DATA_OUT_FIRST;
using ::RSMI_EVNT_XGMI_DATA_OUT_LAST;
using ::RSMI_EVNT_XGMI_FIRST;
using ::RSMI_EVNT_XGMI_LAST;

// rsmi_counter_command_t
using ::RSMI_CNTR_CMD_START;
using ::RSMI_CNTR_CMD_STOP;
using ::rsmi_counter_command_t;

using ::rsmi_counter_value_t;
// rsmi_evt_notification_type_t
using ::RSMI_EVT_NOTIF_EVENT_ALL_PROCESS;
using ::RSMI_EVT_NOTIF_EVENT_MIGRATE_END;
using ::RSMI_EVT_NOTIF_EVENT_MIGRATE_START;
using ::RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_END;
using ::RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_START;
using ::RSMI_EVT_NOTIF_EVENT_QUEUE_EVICTION;
using ::RSMI_EVT_NOTIF_EVENT_QUEUE_RESTORE;
using ::RSMI_EVT_NOTIF_EVENT_UNMAP_FROM_GPU;
using ::RSMI_EVT_NOTIF_FIRST;
using ::RSMI_EVT_NOTIF_GPU_POST_RESET;
using ::RSMI_EVT_NOTIF_GPU_PRE_RESET;
using ::RSMI_EVT_NOTIF_LAST;
using ::RSMI_EVT_NOTIF_NONE;
using ::RSMI_EVT_NOTIF_THERMAL_THROTTLE;
using ::RSMI_EVT_NOTIF_VMFAULT;
using ::rsmi_evt_notification_type_t;

using ::rsmi_evt_notification_data_t;
// rsmi_clk_type_t
using ::RSMI_CLK_INVALID;
using ::RSMI_CLK_TYPE_DCEF;
using ::RSMI_CLK_TYPE_DF;
using ::RSMI_CLK_TYPE_FIRST;
using ::RSMI_CLK_TYPE_LAST;
using ::RSMI_CLK_TYPE_MEM;
using ::RSMI_CLK_TYPE_PCIE;
using ::RSMI_CLK_TYPE_SOC;
using ::RSMI_CLK_TYPE_SYS;
using ::rsmi_clk_type_t;

using ::rsmi_clk_type;
// rsmi_compute_partition_type_t
using ::RSMI_COMPUTE_PARTITION_DPX;
using ::RSMI_COMPUTE_PARTITION_INVALID;
using ::RSMI_COMPUTE_PARTITION_QPX;
using ::RSMI_COMPUTE_PARTITION_SPX;
using ::RSMI_COMPUTE_PARTITION_TPX;
using ::rsmi_compute_partition_type_t;

using ::rsmi_compute_partition_type;
// rsmi_memory_partition_type_t
using ::RSMI_MEMORY_PARTITION_NPS1;
using ::RSMI_MEMORY_PARTITION_NPS2;
using ::RSMI_MEMORY_PARTITION_NPS4;
using ::RSMI_MEMORY_PARTITION_NPS8;
using ::rsmi_memory_partition_type_t;
using ::RSMI_MEMORY_PARTITION_UNKNOWN;

using ::rsmi_memory_partition_type;
// rsmi_temperature_metric_t
using ::RSMI_TEMP_CRIT_MIN;
using ::RSMI_TEMP_CRIT_MIN_HYST;
using ::RSMI_TEMP_CRITICAL;
using ::RSMI_TEMP_CRITICAL_HYST;
using ::RSMI_TEMP_CURRENT;
using ::RSMI_TEMP_EMERGENCY;
using ::RSMI_TEMP_EMERGENCY_HYST;
using ::RSMI_TEMP_FIRST;
using ::RSMI_TEMP_HIGHEST;
using ::RSMI_TEMP_LAST;
using ::RSMI_TEMP_LOWEST;
using ::RSMI_TEMP_MAX;
using ::RSMI_TEMP_MAX_HYST;
using ::RSMI_TEMP_MIN;
using ::RSMI_TEMP_MIN_HYST;
using ::RSMI_TEMP_OFFSET;
using ::rsmi_temperature_metric_t;

using ::rsmi_temperature_metric;
// rsmi_temperature_type_t
using ::RSMI_TEMP_TYPE_EDGE;
using ::RSMI_TEMP_TYPE_FIRST;
using ::RSMI_TEMP_TYPE_HBM_0;
using ::RSMI_TEMP_TYPE_HBM_1;
using ::RSMI_TEMP_TYPE_HBM_2;
using ::RSMI_TEMP_TYPE_HBM_3;
using ::RSMI_TEMP_TYPE_INVALID;
using ::RSMI_TEMP_TYPE_JUNCTION;
using ::RSMI_TEMP_TYPE_LAST;
using ::RSMI_TEMP_TYPE_MEMORY;
using ::rsmi_temperature_type_t;

// rsmi_activity_metric_t
using ::RSMI_ACTIVITY_GFX;
using ::rsmi_activity_metric_t;
using ::RSMI_ACTIVITY_MM;
using ::RSMI_ACTIVITY_UMC;

// rsmi_voltage_metric_t
using ::RSMI_VOLT_AVERAGE;
using ::RSMI_VOLT_CURRENT;
using ::RSMI_VOLT_FIRST;
using ::RSMI_VOLT_HIGHEST;
using ::RSMI_VOLT_LAST;
using ::RSMI_VOLT_LOWEST;
using ::RSMI_VOLT_MAX;
using ::RSMI_VOLT_MAX_CRIT;
using ::RSMI_VOLT_MIN;
using ::RSMI_VOLT_MIN_CRIT;
using ::rsmi_voltage_metric_t;

// rsmi_voltage_type_t
using ::RSMI_VOLT_TYPE_FIRST;
using ::RSMI_VOLT_TYPE_INVALID;
using ::RSMI_VOLT_TYPE_LAST;
using ::RSMI_VOLT_TYPE_VDDBOARD;
using ::RSMI_VOLT_TYPE_VDDGFX;
using ::rsmi_voltage_type_t;

// rsmi_power_profile_preset_masks_t
using ::rsmi_power_profile_preset_masks_t;
using ::RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK;
using ::RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT;
using ::RSMI_PWR_PROF_PRST_COMPUTE_MASK;
using ::RSMI_PWR_PROF_PRST_CUSTOM_MASK;
using ::RSMI_PWR_PROF_PRST_INVALID;
using ::RSMI_PWR_PROF_PRST_LAST;
using ::RSMI_PWR_PROF_PRST_POWER_SAVING_MASK;
using ::RSMI_PWR_PROF_PRST_VIDEO_MASK;
using ::RSMI_PWR_PROF_PRST_VR_MASK;

using ::rsmi_power_profile_preset_masks;
// rsmi_gpu_block_t
using ::RSMI_GPU_BLOCK_ATHUB;
using ::RSMI_GPU_BLOCK_DF;
using ::RSMI_GPU_BLOCK_FIRST;
using ::RSMI_GPU_BLOCK_FUSE;
using ::RSMI_GPU_BLOCK_GFX;
using ::RSMI_GPU_BLOCK_HDP;
using ::RSMI_GPU_BLOCK_INVALID;
using ::RSMI_GPU_BLOCK_LAST;
using ::RSMI_GPU_BLOCK_MMHUB;
using ::RSMI_GPU_BLOCK_MP0;
using ::RSMI_GPU_BLOCK_MP1;
using ::RSMI_GPU_BLOCK_PCIE_BIF;
using ::RSMI_GPU_BLOCK_RESERVED;
using ::RSMI_GPU_BLOCK_SDMA;
using ::RSMI_GPU_BLOCK_SEM;
using ::RSMI_GPU_BLOCK_SMN;
using ::rsmi_gpu_block_t;
using ::RSMI_GPU_BLOCK_UMC;
using ::RSMI_GPU_BLOCK_XGMI_WAFL;

using ::rsmi_gpu_block;
// rsmi_ras_err_state_t
using ::RSMI_RAS_ERR_STATE_DISABLED;
using ::RSMI_RAS_ERR_STATE_ENABLED;
using ::RSMI_RAS_ERR_STATE_INVALID;
using ::RSMI_RAS_ERR_STATE_LAST;
using ::RSMI_RAS_ERR_STATE_MULT_UC;
using ::RSMI_RAS_ERR_STATE_NONE;
using ::RSMI_RAS_ERR_STATE_PARITY;
using ::RSMI_RAS_ERR_STATE_POISON;
using ::RSMI_RAS_ERR_STATE_SING_C;
using ::rsmi_ras_err_state_t;

// rsmi_memory_type_t
using ::RSMI_MEM_TYPE_FIRST;
using ::RSMI_MEM_TYPE_GTT;
using ::RSMI_MEM_TYPE_LAST;
using ::RSMI_MEM_TYPE_VIS_VRAM;
using ::RSMI_MEM_TYPE_VRAM;
using ::rsmi_memory_type_t;

// rsmi_freq_ind_t
using ::RSMI_FREQ_IND_INVALID;
using ::RSMI_FREQ_IND_MAX;
using ::RSMI_FREQ_IND_MIN;
using ::rsmi_freq_ind_t;

using ::rsmi_freq_ind;
// rsmi_fw_block_t
using ::RSMI_FW_BLOCK_ASD;
using ::RSMI_FW_BLOCK_CE;
using ::RSMI_FW_BLOCK_DMCU;
using ::RSMI_FW_BLOCK_FIRST;
using ::RSMI_FW_BLOCK_LAST;
using ::RSMI_FW_BLOCK_MC;
using ::RSMI_FW_BLOCK_ME;
using ::RSMI_FW_BLOCK_MEC;
using ::RSMI_FW_BLOCK_MEC2;
using ::RSMI_FW_BLOCK_MES;
using ::RSMI_FW_BLOCK_MES_KIQ;
using ::RSMI_FW_BLOCK_PFP;
using ::RSMI_FW_BLOCK_RLC;
using ::RSMI_FW_BLOCK_RLC_SRLC;
using ::RSMI_FW_BLOCK_RLC_SRLG;
using ::RSMI_FW_BLOCK_RLC_SRLS;
using ::RSMI_FW_BLOCK_SDMA;
using ::RSMI_FW_BLOCK_SDMA2;
using ::RSMI_FW_BLOCK_SMC;
using ::RSMI_FW_BLOCK_SOS;
using ::rsmi_fw_block_t;
using ::RSMI_FW_BLOCK_TA_RAS;
using ::RSMI_FW_BLOCK_TA_XGMI;
using ::RSMI_FW_BLOCK_UVD;
using ::RSMI_FW_BLOCK_VCE;
using ::RSMI_FW_BLOCK_VCN;

// rsmi_xgmi_status_t
using ::RSMI_XGMI_STATUS_ERROR;
using ::RSMI_XGMI_STATUS_MULTIPLE_ERRORS;
using ::RSMI_XGMI_STATUS_NO_ERRORS;
using ::rsmi_xgmi_status_t;

using ::rsmi_bit_field;
using ::rsmi_bit_field_t;
// rsmi_memory_page_status_t
using ::RSMI_MEM_PAGE_STATUS_PENDING;
using ::RSMI_MEM_PAGE_STATUS_RESERVED;
using ::rsmi_memory_page_status_t;

// RSMI_IO_LINK_TYPE
using ::RSMI_IO_LINK_TYPE;
using ::RSMI_IOLINK_TYPE_NUMIOLINKTYPES;
using ::RSMI_IOLINK_TYPE_PCIEXPRESS;
using ::RSMI_IOLINK_TYPE_SIZE;
using ::RSMI_IOLINK_TYPE_UNDEFINED;
using ::RSMI_IOLINK_TYPE_XGMI;

// RSMI_UTILIZATION_COUNTER_TYPE
using ::RSMI_COARSE_GRAIN_GFX_ACTIVITY;
using ::RSMI_COARSE_GRAIN_MEM_ACTIVITY;
using ::RSMI_UTILIZATION_COUNTER_FIRST;
using ::RSMI_UTILIZATION_COUNTER_LAST;
using ::RSMI_UTILIZATION_COUNTER_TYPE;

// RSMI_POWER_TYPE
using ::RSMI_AVERAGE_POWER;
using ::RSMI_CURRENT_POWER;
using ::RSMI_INVALID_POWER;
using ::RSMI_POWER_TYPE;

using ::metrics_table_header_t;
using ::rsmi_activity_metric_counter_t;
using ::rsmi_device_identifiers_t;
using ::rsmi_error_count_t;
using ::rsmi_freq_volt_region;
using ::rsmi_freq_volt_region_t;
using ::rsmi_frequencies;
using ::rsmi_frequencies_t;
using ::rsmi_func_id_iter_handle_t;
using ::rsmi_func_id_value_t;
using ::rsmi_gpu_metrics_t;
using ::rsmi_od_vddc_point;
using ::rsmi_od_vddc_point_t;
using ::rsmi_od_volt_curve;
using ::rsmi_od_volt_curve_t;
using ::rsmi_od_volt_freq_data;
using ::rsmi_od_volt_freq_data_t;
using ::rsmi_pcie_bandwidth;
using ::rsmi_pcie_bandwidth_t;
using ::rsmi_power_profile_status;
using ::rsmi_power_profile_status_t;
using ::rsmi_process_info_t;
using ::rsmi_range;
using ::rsmi_range_t;
using ::rsmi_retired_page_record_t;
using ::rsmi_utilization_counter_t;
using ::rsmi_version;
using ::rsmi_version_t;

// ========================================================================
// Functions
// ========================================================================

// Initialization and Shutdown
using ::rsmi_init;
using ::rsmi_shut_down;

// Identifier Queries
using ::rsmi_dev_brand_get;
using ::rsmi_dev_device_identifiers_get;
using ::rsmi_dev_drm_render_minor_get;
using ::rsmi_dev_guid_get;
using ::rsmi_dev_id_get;
using ::rsmi_dev_market_name_get;
using ::rsmi_dev_name_get;
using ::rsmi_dev_node_id_get;
using ::rsmi_dev_revision_get;
using ::rsmi_dev_serial_number_get;
using ::rsmi_dev_sku_get;
using ::rsmi_dev_subsystem_id_get;
using ::rsmi_dev_subsystem_name_get;
using ::rsmi_dev_subsystem_vendor_id_get;
using ::rsmi_dev_unique_id_get;
using ::rsmi_dev_vendor_id_get;
using ::rsmi_dev_vendor_name_get;
using ::rsmi_dev_vram_vendor_get;
using ::rsmi_dev_xgmi_physical_id_get;
using ::rsmi_num_monitor_devices;

// PCIe Queries
using ::rsmi_dev_pci_bandwidth_get;
using ::rsmi_dev_pci_id_get;
using ::rsmi_dev_pci_replay_counter_get;
using ::rsmi_dev_pci_throughput_get;
using ::rsmi_topo_numa_affinity_get;

// PCIe Control
using ::rsmi_dev_pci_bandwidth_set;

// Power Queries
using ::rsmi_dev_current_socket_power_get;
using ::rsmi_dev_energy_count_get;
using ::rsmi_dev_power_ave_get;
using ::rsmi_dev_power_cap_default_get;
using ::rsmi_dev_power_cap_get;
using ::rsmi_dev_power_cap_range_get;
using ::rsmi_dev_power_get;

// Power Control
using ::rsmi_dev_power_cap_set;
using ::rsmi_dev_power_profile_set;

// Memory Queries
using ::rsmi_dev_memory_busy_percent_get;
using ::rsmi_dev_memory_reserved_pages_get;
using ::rsmi_dev_memory_total_get;
using ::rsmi_dev_memory_usage_get;

// Physical State Queries
using ::rsmi_dev_fan_rpms_get;
using ::rsmi_dev_fan_speed_get;
using ::rsmi_dev_fan_speed_max_get;
using ::rsmi_dev_temp_metric_get;
using ::rsmi_dev_volt_metric_get;

// Physical State Control
using ::rsmi_dev_fan_reset;
using ::rsmi_dev_fan_speed_set;

// Clock, Power and Performance Queries
using ::rsmi_dev_activity_avg_mm_get;
using ::rsmi_dev_activity_metric_get;
using ::rsmi_dev_busy_percent_get;
using ::rsmi_dev_clk_extremum_set;
using ::rsmi_dev_clk_range_set;
using ::rsmi_dev_gpu_clk_freq_get;
using ::rsmi_dev_gpu_metrics_info_get;
using ::rsmi_dev_gpu_reset;
using ::rsmi_dev_mem_overdrive_level_get;
using ::rsmi_dev_od_clk_info_set;
using ::rsmi_dev_od_volt_curve_regions_get;
using ::rsmi_dev_od_volt_info_get;
using ::rsmi_dev_od_volt_info_set;
using ::rsmi_dev_overdrive_level_get;
using ::rsmi_dev_perf_level_get;
using ::rsmi_dev_power_profile_presets_get;
using ::rsmi_perf_determinism_mode_set;
using ::rsmi_utilization_count_get;

// Clock, Power and Performance Control
using ::rsmi_dev_gpu_clk_freq_set;
using ::rsmi_dev_overdrive_level_set;
using ::rsmi_dev_overdrive_level_set_v1;
using ::rsmi_dev_perf_level_set;
using ::rsmi_dev_perf_level_set_v1;

// Version Queries
using ::rsmi_dev_firmware_version_get;
using ::rsmi_dev_target_graphics_version_get;
using ::rsmi_dev_vbios_version_get;
using ::rsmi_version_get;
using ::rsmi_version_str_get;

// Error Queries
using ::rsmi_dev_ecc_count_get;
using ::rsmi_dev_ecc_enabled_get;
using ::rsmi_dev_ecc_status_get;
using ::rsmi_status_string;

// Performance Counter Functions
using ::rsmi_counter_available_counters_get;
using ::rsmi_counter_control;
using ::rsmi_counter_read;
using ::rsmi_dev_counter_create;
using ::rsmi_dev_counter_destroy;
using ::rsmi_dev_counter_group_supported;

// System Information Functions
using ::rsmi_compute_process_gpus_get;
using ::rsmi_compute_process_info_by_device_get;
using ::rsmi_compute_process_info_by_pid_get;
using ::rsmi_compute_process_info_get;

// XGMI Functions
using ::rsmi_dev_xgmi_error_reset;
using ::rsmi_dev_xgmi_error_status;
using ::rsmi_dev_xgmi_hive_id_get;

// Hardware Topology Functions
using ::rsmi_is_P2P_accessible;
using ::rsmi_minmax_bandwidth_get;
using ::rsmi_topo_get_link_type;
using ::rsmi_topo_get_link_weight;
using ::rsmi_topo_get_numa_node_number;

// Compute Partition Functions
using ::rsmi_dev_compute_partition_get;
using ::rsmi_dev_compute_partition_set;
using ::rsmi_dev_partition_id_get;

// The Memory Partition Functions
using ::rsmi_dev_memory_partition_capabilities_get;
using ::rsmi_dev_memory_partition_get;
using ::rsmi_dev_memory_partition_set;

// Supported Functions
using ::rsmi_dev_supported_func_iterator_close;
using ::rsmi_dev_supported_func_iterator_open;
using ::rsmi_dev_supported_variant_iterator_open;
using ::rsmi_func_iter_next;
using ::rsmi_func_iter_value_get;

// Event Notification Functions
using ::rsmi_event_notification_get;
using ::rsmi_event_notification_init;
using ::rsmi_event_notification_mask_set;
using ::rsmi_event_notification_stop;

// Metric Functions
using ::rsmi_dev_metrics_header_info_get;
using ::rsmi_dev_metrics_log_get;
using ::rsmi_dev_metrics_xcd_counter_get;

} // namespace wwr::hip
