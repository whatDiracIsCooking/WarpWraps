// rocm_smi.cppm - Compile-time tests for gpumod.hip.rocm_smi

module;

#include "test/shared/link_check.h"

export module gpumod.test.hip.rocm_smi;

import std;
import gpumod.hip.rocm_smi;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.hip.rocm_smi
//
// The module is pure re-export (using declarations only -- rocm_smi.h is a
// plain extern "C" API, no convenience-template collisions like hip_runtime_api.h).
// Runtime tests would just test ROCm SMI itself, and it requires a live device;
// we verify at compile/link time that:
//   1. Enum types satisfy std::is_enum_v
//   2. Every enumerator value matches the compiled rocm_smi.h value (generated
//      from the actual compiled values, not hand-copied)
//   3. Struct/union types are trivially copyable (C-interop guarantee)
//   4. Opaque handle typedefs have the expected pointer shape
//   5. Link-time symbol resolution for every exported function
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::hip::test {

using namespace wwr::hip;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<rsmi_status_t>);
static_assert(std::is_enum_v<rsmi_init_flags_t>);
static_assert(std::is_enum_v<rsmi_dev_perf_level_t>);
static_assert(std::is_enum_v<rsmi_sw_component_t>);
static_assert(std::is_enum_v<rsmi_event_group_t>);
static_assert(std::is_enum_v<rsmi_event_type_t>);
static_assert(std::is_enum_v<rsmi_counter_command_t>);
static_assert(std::is_enum_v<rsmi_evt_notification_type_t>);
static_assert(std::is_enum_v<rsmi_clk_type_t>);
static_assert(std::is_enum_v<rsmi_compute_partition_type_t>);
static_assert(std::is_enum_v<rsmi_memory_partition_type_t>);
static_assert(std::is_enum_v<rsmi_temperature_metric_t>);
static_assert(std::is_enum_v<rsmi_temperature_type_t>);
static_assert(std::is_enum_v<rsmi_activity_metric_t>);
static_assert(std::is_enum_v<rsmi_voltage_metric_t>);
static_assert(std::is_enum_v<rsmi_voltage_type_t>);
static_assert(std::is_enum_v<rsmi_power_profile_preset_masks_t>);
static_assert(std::is_enum_v<rsmi_gpu_block_t>);
static_assert(std::is_enum_v<rsmi_ras_err_state_t>);
static_assert(std::is_enum_v<rsmi_memory_type_t>);
static_assert(std::is_enum_v<rsmi_freq_ind_t>);
static_assert(std::is_enum_v<rsmi_fw_block_t>);
static_assert(std::is_enum_v<rsmi_xgmi_status_t>);
static_assert(std::is_enum_v<rsmi_memory_page_status_t>);
static_assert(std::is_enum_v<RSMI_IO_LINK_TYPE>);
static_assert(std::is_enum_v<RSMI_UTILIZATION_COUNTER_TYPE>);
static_assert(std::is_enum_v<RSMI_POWER_TYPE>);

// ────────────────────────────────────────────────────────────────────────
// Enum values (generated from the compiled rocm_smi.h values)
// ────────────────────────────────────────────────────────────────────────

// rsmi_status_t
static_assert(static_cast<long long>(RSMI_STATUS_SUCCESS) == 0);
static_assert(static_cast<long long>(RSMI_STATUS_INVALID_ARGS) == 1);
static_assert(static_cast<long long>(RSMI_STATUS_NOT_SUPPORTED) == 2);
static_assert(static_cast<long long>(RSMI_STATUS_FILE_ERROR) == 3);
static_assert(static_cast<long long>(RSMI_STATUS_PERMISSION) == 4);
static_assert(static_cast<long long>(RSMI_STATUS_OUT_OF_RESOURCES) == 5);
static_assert(static_cast<long long>(RSMI_STATUS_INTERNAL_EXCEPTION) == 6);
static_assert(static_cast<long long>(RSMI_STATUS_INPUT_OUT_OF_BOUNDS) == 7);
static_assert(static_cast<long long>(RSMI_STATUS_INIT_ERROR) == 8);
static_assert(static_cast<long long>(RSMI_INITIALIZATION_ERROR) == 8);
static_assert(static_cast<long long>(RSMI_STATUS_NOT_YET_IMPLEMENTED) == 9);
static_assert(static_cast<long long>(RSMI_STATUS_NOT_FOUND) == 10);
static_assert(static_cast<long long>(RSMI_STATUS_INSUFFICIENT_SIZE) == 11);
static_assert(static_cast<long long>(RSMI_STATUS_INTERRUPT) == 12);
static_assert(static_cast<long long>(RSMI_STATUS_UNEXPECTED_SIZE) == 13);
static_assert(static_cast<long long>(RSMI_STATUS_NO_DATA) == 14);
static_assert(static_cast<long long>(RSMI_STATUS_UNEXPECTED_DATA) == 15);
static_assert(static_cast<long long>(RSMI_STATUS_BUSY) == 16);
static_assert(static_cast<long long>(RSMI_STATUS_REFCOUNT_OVERFLOW) == 17);
static_assert(static_cast<long long>(RSMI_STATUS_SETTING_UNAVAILABLE) == 18);
static_assert(static_cast<long long>(RSMI_STATUS_AMDGPU_RESTART_ERR) == 19);
static_assert(static_cast<long long>(RSMI_STATUS_DRM_ERROR) == 20);
static_assert(static_cast<long long>(RSMI_STATUS_FAIL_LOAD_MODULE) == 21);
static_assert(static_cast<long long>(RSMI_STATUS_FAIL_LOAD_SYMBOL) == 22);
static_assert(static_cast<long long>(RSMI_STATUS_UNKNOWN_ERROR) == 4294967295);

// rsmi_init_flags_t
static_assert(static_cast<long long>(RSMI_INIT_FLAG_ALL_GPUS) == 1);
static_assert(static_cast<long long>(RSMI_INIT_FLAG_THRAD_ONLY_MUTEX) == 288230376151711744);
static_assert(static_cast<long long>(RSMI_INIT_FLAG_RESRV_TEST1) == 576460752303423488);

// rsmi_dev_perf_level_t
static_assert(static_cast<long long>(RSMI_DEV_PERF_LEVEL_AUTO) == 0);
static_assert(static_cast<long long>(RSMI_DEV_PERF_LEVEL_FIRST) == 0);
static_assert(static_cast<long long>(RSMI_DEV_PERF_LEVEL_LOW) == 1);
static_assert(static_cast<long long>(RSMI_DEV_PERF_LEVEL_HIGH) == 2);
static_assert(static_cast<long long>(RSMI_DEV_PERF_LEVEL_MANUAL) == 3);
static_assert(static_cast<long long>(RSMI_DEV_PERF_LEVEL_STABLE_STD) == 4);
static_assert(static_cast<long long>(RSMI_DEV_PERF_LEVEL_STABLE_PEAK) == 5);
static_assert(static_cast<long long>(RSMI_DEV_PERF_LEVEL_STABLE_MIN_MCLK) == 6);
static_assert(static_cast<long long>(RSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK) == 7);
static_assert(static_cast<long long>(RSMI_DEV_PERF_LEVEL_DETERMINISM) == 8);
static_assert(static_cast<long long>(RSMI_DEV_PERF_LEVEL_LAST) == 8);
static_assert(static_cast<long long>(RSMI_DEV_PERF_LEVEL_UNKNOWN) == 256);

// rsmi_sw_component_t
static_assert(static_cast<long long>(RSMI_SW_COMP_FIRST) == 0);
static_assert(static_cast<long long>(RSMI_SW_COMP_DRIVER) == 0);
static_assert(static_cast<long long>(RSMI_SW_COMP_LAST) == 0);

// rsmi_event_group_t
static_assert(static_cast<long long>(RSMI_EVNT_GRP_XGMI) == 0);
static_assert(static_cast<long long>(RSMI_EVNT_GRP_XGMI_DATA_OUT) == 10);
static_assert(static_cast<long long>(RSMI_EVNT_GRP_INVALID) == 4294967295);

// rsmi_event_type_t
static_assert(static_cast<long long>(RSMI_EVNT_FIRST) == 0);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_FIRST) == 0);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_0_NOP_TX) == 0);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_0_REQUEST_TX) == 1);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_0_RESPONSE_TX) == 2);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_0_BEATS_TX) == 3);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_1_NOP_TX) == 4);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_1_REQUEST_TX) == 5);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_1_RESPONSE_TX) == 6);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_1_BEATS_TX) == 7);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_LAST) == 7);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_DATA_OUT_FIRST) == 10);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_DATA_OUT_0) == 10);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_DATA_OUT_1) == 11);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_DATA_OUT_2) == 12);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_DATA_OUT_3) == 13);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_DATA_OUT_4) == 14);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_DATA_OUT_5) == 15);
static_assert(static_cast<long long>(RSMI_EVNT_XGMI_DATA_OUT_LAST) == 15);
static_assert(static_cast<long long>(RSMI_EVNT_LAST) == 15);

// rsmi_counter_command_t
static_assert(static_cast<long long>(RSMI_CNTR_CMD_START) == 0);
static_assert(static_cast<long long>(RSMI_CNTR_CMD_STOP) == 1);

// rsmi_evt_notification_type_t
static_assert(static_cast<long long>(RSMI_EVT_NOTIF_NONE) == 0);
static_assert(static_cast<long long>(RSMI_EVT_NOTIF_VMFAULT) == 1);
static_assert(static_cast<long long>(RSMI_EVT_NOTIF_FIRST) == 1);
static_assert(static_cast<long long>(RSMI_EVT_NOTIF_THERMAL_THROTTLE) == 2);
static_assert(static_cast<long long>(RSMI_EVT_NOTIF_GPU_PRE_RESET) == 3);
static_assert(static_cast<long long>(RSMI_EVT_NOTIF_GPU_POST_RESET) == 4);
static_assert(static_cast<long long>(RSMI_EVT_NOTIF_EVENT_MIGRATE_START) == 5);
static_assert(static_cast<long long>(RSMI_EVT_NOTIF_EVENT_MIGRATE_END) == 6);
static_assert(static_cast<long long>(RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_START) == 7);
static_assert(static_cast<long long>(RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_END) == 8);
static_assert(static_cast<long long>(RSMI_EVT_NOTIF_EVENT_QUEUE_EVICTION) == 9);
static_assert(static_cast<long long>(RSMI_EVT_NOTIF_EVENT_QUEUE_RESTORE) == 10);
static_assert(static_cast<long long>(RSMI_EVT_NOTIF_EVENT_UNMAP_FROM_GPU) == 11);
static_assert(static_cast<long long>(RSMI_EVT_NOTIF_EVENT_ALL_PROCESS) == 64);
static_assert(static_cast<long long>(RSMI_EVT_NOTIF_LAST) == 64);

// rsmi_clk_type_t
static_assert(static_cast<long long>(RSMI_CLK_TYPE_SYS) == 0);
static_assert(static_cast<long long>(RSMI_CLK_TYPE_FIRST) == 0);
static_assert(static_cast<long long>(RSMI_CLK_TYPE_DF) == 1);
static_assert(static_cast<long long>(RSMI_CLK_TYPE_DCEF) == 2);
static_assert(static_cast<long long>(RSMI_CLK_TYPE_SOC) == 3);
static_assert(static_cast<long long>(RSMI_CLK_TYPE_MEM) == 4);
static_assert(static_cast<long long>(RSMI_CLK_TYPE_PCIE) == 5);
static_assert(static_cast<long long>(RSMI_CLK_TYPE_LAST) == 4);
static_assert(static_cast<long long>(RSMI_CLK_INVALID) == 4294967295);

// rsmi_compute_partition_type_t
static_assert(static_cast<long long>(RSMI_COMPUTE_PARTITION_INVALID) == 0);
static_assert(static_cast<long long>(RSMI_COMPUTE_PARTITION_SPX) == 1);
static_assert(static_cast<long long>(RSMI_COMPUTE_PARTITION_DPX) == 2);
static_assert(static_cast<long long>(RSMI_COMPUTE_PARTITION_TPX) == 3);
static_assert(static_cast<long long>(RSMI_COMPUTE_PARTITION_QPX) == 4);

// rsmi_memory_partition_type_t
static_assert(static_cast<long long>(RSMI_MEMORY_PARTITION_UNKNOWN) == 0);
static_assert(static_cast<long long>(RSMI_MEMORY_PARTITION_NPS1) == 1);
static_assert(static_cast<long long>(RSMI_MEMORY_PARTITION_NPS2) == 2);
static_assert(static_cast<long long>(RSMI_MEMORY_PARTITION_NPS4) == 3);
static_assert(static_cast<long long>(RSMI_MEMORY_PARTITION_NPS8) == 4);

// rsmi_temperature_metric_t
static_assert(static_cast<long long>(RSMI_TEMP_CURRENT) == 0);
static_assert(static_cast<long long>(RSMI_TEMP_FIRST) == 0);
static_assert(static_cast<long long>(RSMI_TEMP_MAX) == 1);
static_assert(static_cast<long long>(RSMI_TEMP_MIN) == 2);
static_assert(static_cast<long long>(RSMI_TEMP_MAX_HYST) == 3);
static_assert(static_cast<long long>(RSMI_TEMP_MIN_HYST) == 4);
static_assert(static_cast<long long>(RSMI_TEMP_CRITICAL) == 5);
static_assert(static_cast<long long>(RSMI_TEMP_CRITICAL_HYST) == 6);
static_assert(static_cast<long long>(RSMI_TEMP_EMERGENCY) == 7);
static_assert(static_cast<long long>(RSMI_TEMP_EMERGENCY_HYST) == 8);
static_assert(static_cast<long long>(RSMI_TEMP_CRIT_MIN) == 9);
static_assert(static_cast<long long>(RSMI_TEMP_CRIT_MIN_HYST) == 10);
static_assert(static_cast<long long>(RSMI_TEMP_OFFSET) == 11);
static_assert(static_cast<long long>(RSMI_TEMP_LOWEST) == 12);
static_assert(static_cast<long long>(RSMI_TEMP_HIGHEST) == 13);
static_assert(static_cast<long long>(RSMI_TEMP_LAST) == 13);

// rsmi_temperature_type_t
static_assert(static_cast<long long>(RSMI_TEMP_TYPE_FIRST) == 0);
static_assert(static_cast<long long>(RSMI_TEMP_TYPE_EDGE) == 0);
static_assert(static_cast<long long>(RSMI_TEMP_TYPE_JUNCTION) == 1);
static_assert(static_cast<long long>(RSMI_TEMP_TYPE_MEMORY) == 2);
static_assert(static_cast<long long>(RSMI_TEMP_TYPE_HBM_0) == 3);
static_assert(static_cast<long long>(RSMI_TEMP_TYPE_HBM_1) == 4);
static_assert(static_cast<long long>(RSMI_TEMP_TYPE_HBM_2) == 5);
static_assert(static_cast<long long>(RSMI_TEMP_TYPE_HBM_3) == 6);
static_assert(static_cast<long long>(RSMI_TEMP_TYPE_LAST) == 6);
static_assert(static_cast<long long>(RSMI_TEMP_TYPE_INVALID) == 4294967295);

// rsmi_activity_metric_t
static_assert(static_cast<long long>(RSMI_ACTIVITY_GFX) == 1);
static_assert(static_cast<long long>(RSMI_ACTIVITY_UMC) == 2);
static_assert(static_cast<long long>(RSMI_ACTIVITY_MM) == 4);

// rsmi_voltage_metric_t
static_assert(static_cast<long long>(RSMI_VOLT_CURRENT) == 0);
static_assert(static_cast<long long>(RSMI_VOLT_FIRST) == 0);
static_assert(static_cast<long long>(RSMI_VOLT_MAX) == 1);
static_assert(static_cast<long long>(RSMI_VOLT_MIN_CRIT) == 2);
static_assert(static_cast<long long>(RSMI_VOLT_MIN) == 3);
static_assert(static_cast<long long>(RSMI_VOLT_MAX_CRIT) == 4);
static_assert(static_cast<long long>(RSMI_VOLT_AVERAGE) == 5);
static_assert(static_cast<long long>(RSMI_VOLT_LOWEST) == 6);
static_assert(static_cast<long long>(RSMI_VOLT_HIGHEST) == 7);
static_assert(static_cast<long long>(RSMI_VOLT_LAST) == 7);

// rsmi_voltage_type_t
static_assert(static_cast<long long>(RSMI_VOLT_TYPE_FIRST) == 0);
static_assert(static_cast<long long>(RSMI_VOLT_TYPE_VDDGFX) == 0);
static_assert(static_cast<long long>(RSMI_VOLT_TYPE_VDDBOARD) == 1);
static_assert(static_cast<long long>(RSMI_VOLT_TYPE_LAST) == 1);
static_assert(static_cast<long long>(RSMI_VOLT_TYPE_INVALID) == 4294967295);

// rsmi_power_profile_preset_masks_t
static_assert(static_cast<long long>(RSMI_PWR_PROF_PRST_CUSTOM_MASK) == 1);
static_assert(static_cast<long long>(RSMI_PWR_PROF_PRST_VIDEO_MASK) == 2);
static_assert(static_cast<long long>(RSMI_PWR_PROF_PRST_POWER_SAVING_MASK) == 4);
static_assert(static_cast<long long>(RSMI_PWR_PROF_PRST_COMPUTE_MASK) == 8);
static_assert(static_cast<long long>(RSMI_PWR_PROF_PRST_VR_MASK) == 16);
static_assert(static_cast<long long>(RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK) == 32);
static_assert(static_cast<long long>(RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT) == 64);
static_assert(static_cast<long long>(RSMI_PWR_PROF_PRST_LAST) == 64);
static_assert(static_cast<long long>(RSMI_PWR_PROF_PRST_INVALID) == -1);

// rsmi_gpu_block_t
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_INVALID) == 0);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_FIRST) == 1);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_UMC) == 1);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_SDMA) == 2);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_GFX) == 4);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_MMHUB) == 8);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_ATHUB) == 16);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_PCIE_BIF) == 32);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_HDP) == 64);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_XGMI_WAFL) == 128);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_DF) == 256);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_SMN) == 512);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_SEM) == 1024);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_MP0) == 2048);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_MP1) == 4096);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_FUSE) == 8192);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_LAST) == 8192);
static_assert(static_cast<long long>(RSMI_GPU_BLOCK_RESERVED) == -9223372036854775808);

// rsmi_ras_err_state_t
static_assert(static_cast<long long>(RSMI_RAS_ERR_STATE_NONE) == 0);
static_assert(static_cast<long long>(RSMI_RAS_ERR_STATE_DISABLED) == 1);
static_assert(static_cast<long long>(RSMI_RAS_ERR_STATE_PARITY) == 2);
static_assert(static_cast<long long>(RSMI_RAS_ERR_STATE_SING_C) == 3);
static_assert(static_cast<long long>(RSMI_RAS_ERR_STATE_MULT_UC) == 4);
static_assert(static_cast<long long>(RSMI_RAS_ERR_STATE_POISON) == 5);
static_assert(static_cast<long long>(RSMI_RAS_ERR_STATE_ENABLED) == 6);
static_assert(static_cast<long long>(RSMI_RAS_ERR_STATE_LAST) == 6);
static_assert(static_cast<long long>(RSMI_RAS_ERR_STATE_INVALID) == 4294967295);

// rsmi_memory_type_t
static_assert(static_cast<long long>(RSMI_MEM_TYPE_FIRST) == 0);
static_assert(static_cast<long long>(RSMI_MEM_TYPE_VRAM) == 0);
static_assert(static_cast<long long>(RSMI_MEM_TYPE_VIS_VRAM) == 1);
static_assert(static_cast<long long>(RSMI_MEM_TYPE_GTT) == 2);
static_assert(static_cast<long long>(RSMI_MEM_TYPE_LAST) == 2);

// rsmi_freq_ind_t
static_assert(static_cast<long long>(RSMI_FREQ_IND_MIN) == 0);
static_assert(static_cast<long long>(RSMI_FREQ_IND_MAX) == 1);
static_assert(static_cast<long long>(RSMI_FREQ_IND_INVALID) == 4294967295);

// rsmi_fw_block_t
static_assert(static_cast<long long>(RSMI_FW_BLOCK_FIRST) == 0);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_ASD) == 0);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_CE) == 1);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_DMCU) == 2);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_MC) == 3);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_ME) == 4);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_MEC) == 5);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_MEC2) == 6);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_MES) == 7);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_MES_KIQ) == 8);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_PFP) == 9);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_RLC) == 10);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_RLC_SRLC) == 11);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_RLC_SRLG) == 12);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_RLC_SRLS) == 13);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_SDMA) == 14);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_SDMA2) == 15);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_SMC) == 16);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_SOS) == 17);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_TA_RAS) == 18);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_TA_XGMI) == 19);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_UVD) == 20);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_VCE) == 21);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_VCN) == 22);
static_assert(static_cast<long long>(RSMI_FW_BLOCK_LAST) == 22);

// rsmi_xgmi_status_t
static_assert(static_cast<long long>(RSMI_XGMI_STATUS_NO_ERRORS) == 0);
static_assert(static_cast<long long>(RSMI_XGMI_STATUS_ERROR) == 1);
static_assert(static_cast<long long>(RSMI_XGMI_STATUS_MULTIPLE_ERRORS) == 2);

// rsmi_memory_page_status_t
static_assert(static_cast<long long>(RSMI_MEM_PAGE_STATUS_RESERVED) == 0);
static_assert(static_cast<long long>(RSMI_MEM_PAGE_STATUS_PENDING) == 1);

// RSMI_IO_LINK_TYPE
static_assert(static_cast<long long>(RSMI_IOLINK_TYPE_UNDEFINED) == 0);
static_assert(static_cast<long long>(RSMI_IOLINK_TYPE_PCIEXPRESS) == 1);
static_assert(static_cast<long long>(RSMI_IOLINK_TYPE_XGMI) == 2);
static_assert(static_cast<long long>(RSMI_IOLINK_TYPE_NUMIOLINKTYPES) == 3);
static_assert(static_cast<long long>(RSMI_IOLINK_TYPE_SIZE) == 4294967295);

// RSMI_UTILIZATION_COUNTER_TYPE
static_assert(static_cast<long long>(RSMI_UTILIZATION_COUNTER_FIRST) == 0);
static_assert(static_cast<long long>(RSMI_COARSE_GRAIN_GFX_ACTIVITY) == 0);
static_assert(static_cast<long long>(RSMI_COARSE_GRAIN_MEM_ACTIVITY) == 1);
static_assert(static_cast<long long>(RSMI_UTILIZATION_COUNTER_LAST) == 1);

// RSMI_POWER_TYPE
static_assert(static_cast<long long>(RSMI_AVERAGE_POWER) == 0);
static_assert(static_cast<long long>(RSMI_CURRENT_POWER) == 1);
static_assert(static_cast<long long>(RSMI_INVALID_POWER) == 4294967295);

// ────────────────────────────────────────────────────────────────────────
// Struct / union traits: trivial copyability (C-interop guarantee)
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_trivially_copyable_v<rsmi_counter_value_t>);
static_assert(std::is_trivially_copyable_v<rsmi_evt_notification_data_t>);
static_assert(std::is_trivially_copyable_v<rsmi_utilization_counter_t>);
static_assert(std::is_trivially_copyable_v<rsmi_retired_page_record_t>);
static_assert(std::is_trivially_copyable_v<rsmi_power_profile_status_t>);
static_assert(std::is_trivially_copyable_v<rsmi_frequencies_t>);
static_assert(std::is_trivially_copyable_v<rsmi_pcie_bandwidth_t>);
static_assert(std::is_trivially_copyable_v<rsmi_activity_metric_counter_t>);
static_assert(std::is_trivially_copyable_v<rsmi_version_t>);
static_assert(std::is_trivially_copyable_v<rsmi_range_t>);
static_assert(std::is_trivially_copyable_v<rsmi_od_vddc_point_t>);
static_assert(std::is_trivially_copyable_v<rsmi_freq_volt_region_t>);
static_assert(std::is_trivially_copyable_v<rsmi_od_volt_curve_t>);
static_assert(std::is_trivially_copyable_v<rsmi_od_volt_freq_data_t>);
static_assert(std::is_trivially_copyable_v<rsmi_gpu_metrics_t>);
static_assert(std::is_trivially_copyable_v<rsmi_error_count_t>);
static_assert(std::is_trivially_copyable_v<rsmi_process_info_t>);
static_assert(std::is_trivially_copyable_v<rsmi_func_id_value_t>);
static_assert(std::is_trivially_copyable_v<rsmi_device_identifiers_t>);

// ────────────────────────────────────────────────────────────────────────
// Opaque handle typedefs
// ────────────────────────────────────────────────────────────────────────

// rsmi_func_id_iter_handle_t is a pointer to a forward-declared, never-defined
// struct (opaque handle).
static_assert(std::is_pointer_v<rsmi_func_id_iter_handle_t>);
// rsmi_event_handle_t (uintptr_t) and rsmi_bit_field_t (uint64_t) are integral
// typedefs, not pointers -- sized to match their underlying type.
static_assert(sizeof(rsmi_event_handle_t) == sizeof(std::uintptr_t));
static_assert(sizeof(rsmi_bit_field_t) == sizeof(std::uint64_t));

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// ────────────────────────────────────────────────────────────────────────

// Initialization and Shutdown
WWR_LINK_CHECK(rsmi_init)
WWR_LINK_CHECK(rsmi_shut_down)

// Identifier Queries
WWR_LINK_CHECK(rsmi_num_monitor_devices)
WWR_LINK_CHECK(rsmi_dev_id_get)
WWR_LINK_CHECK(rsmi_dev_revision_get)
WWR_LINK_CHECK(rsmi_dev_sku_get)
WWR_LINK_CHECK(rsmi_dev_vendor_id_get)
WWR_LINK_CHECK(rsmi_dev_name_get)
WWR_LINK_CHECK(rsmi_dev_brand_get)
WWR_LINK_CHECK(rsmi_dev_vendor_name_get)
WWR_LINK_CHECK(rsmi_dev_market_name_get)
WWR_LINK_CHECK(rsmi_dev_vram_vendor_get)
WWR_LINK_CHECK(rsmi_dev_serial_number_get)
WWR_LINK_CHECK(rsmi_dev_subsystem_id_get)
WWR_LINK_CHECK(rsmi_dev_subsystem_name_get)
WWR_LINK_CHECK(rsmi_dev_drm_render_minor_get)
WWR_LINK_CHECK(rsmi_dev_subsystem_vendor_id_get)
WWR_LINK_CHECK(rsmi_dev_unique_id_get)
WWR_LINK_CHECK(rsmi_dev_xgmi_physical_id_get)
WWR_LINK_CHECK(rsmi_dev_guid_get)
WWR_LINK_CHECK(rsmi_dev_node_id_get)
WWR_LINK_CHECK(rsmi_dev_device_identifiers_get)

// PCIe Queries
WWR_LINK_CHECK(rsmi_dev_pci_bandwidth_get)
WWR_LINK_CHECK(rsmi_dev_pci_id_get)
WWR_LINK_CHECK(rsmi_topo_numa_affinity_get)
WWR_LINK_CHECK(rsmi_dev_pci_throughput_get)
WWR_LINK_CHECK(rsmi_dev_pci_replay_counter_get)

// PCIe Control
WWR_LINK_CHECK(rsmi_dev_pci_bandwidth_set)

// Power Queries
WWR_LINK_CHECK(rsmi_dev_power_ave_get)
WWR_LINK_CHECK(rsmi_dev_current_socket_power_get)
WWR_LINK_CHECK(rsmi_dev_power_get)
WWR_LINK_CHECK(rsmi_dev_energy_count_get)
WWR_LINK_CHECK(rsmi_dev_power_cap_get)
WWR_LINK_CHECK(rsmi_dev_power_cap_default_get)
WWR_LINK_CHECK(rsmi_dev_power_cap_range_get)

// Power Control
WWR_LINK_CHECK(rsmi_dev_power_cap_set)
WWR_LINK_CHECK(rsmi_dev_power_profile_set)

// Memory Queries
WWR_LINK_CHECK(rsmi_dev_memory_total_get)
WWR_LINK_CHECK(rsmi_dev_memory_usage_get)
WWR_LINK_CHECK(rsmi_dev_memory_busy_percent_get)
WWR_LINK_CHECK(rsmi_dev_memory_reserved_pages_get)

// Physical State Queries
WWR_LINK_CHECK(rsmi_dev_fan_rpms_get)
WWR_LINK_CHECK(rsmi_dev_fan_speed_get)
WWR_LINK_CHECK(rsmi_dev_fan_speed_max_get)
WWR_LINK_CHECK(rsmi_dev_temp_metric_get)
WWR_LINK_CHECK(rsmi_dev_volt_metric_get)

// Physical State Control
WWR_LINK_CHECK(rsmi_dev_fan_reset)
WWR_LINK_CHECK(rsmi_dev_fan_speed_set)

// Clock, Power and Performance Queries
WWR_LINK_CHECK(rsmi_dev_busy_percent_get)
WWR_LINK_CHECK(rsmi_utilization_count_get)
WWR_LINK_CHECK(rsmi_dev_activity_metric_get)
WWR_LINK_CHECK(rsmi_dev_activity_avg_mm_get)
WWR_LINK_CHECK(rsmi_dev_perf_level_get)
WWR_LINK_CHECK(rsmi_perf_determinism_mode_set)
WWR_LINK_CHECK(rsmi_dev_overdrive_level_get)
WWR_LINK_CHECK(rsmi_dev_mem_overdrive_level_get)
WWR_LINK_CHECK(rsmi_dev_gpu_clk_freq_get)
WWR_LINK_CHECK(rsmi_dev_gpu_reset)
WWR_LINK_CHECK(rsmi_dev_od_volt_info_get)
WWR_LINK_CHECK(rsmi_dev_gpu_metrics_info_get)
WWR_LINK_CHECK(rsmi_dev_clk_range_set)
WWR_LINK_CHECK(rsmi_dev_clk_extremum_set)
WWR_LINK_CHECK(rsmi_dev_od_clk_info_set)
WWR_LINK_CHECK(rsmi_dev_od_volt_info_set)
WWR_LINK_CHECK(rsmi_dev_od_volt_curve_regions_get)
WWR_LINK_CHECK(rsmi_dev_power_profile_presets_get)

// Clock, Power and Performance Control
WWR_LINK_CHECK(rsmi_dev_perf_level_set)
WWR_LINK_CHECK(rsmi_dev_perf_level_set_v1)
WWR_LINK_CHECK(rsmi_dev_overdrive_level_set)
WWR_LINK_CHECK(rsmi_dev_overdrive_level_set_v1)
WWR_LINK_CHECK(rsmi_dev_gpu_clk_freq_set)

// Version Queries
WWR_LINK_CHECK(rsmi_version_get)
WWR_LINK_CHECK(rsmi_version_str_get)
WWR_LINK_CHECK(rsmi_dev_vbios_version_get)
WWR_LINK_CHECK(rsmi_dev_firmware_version_get)
WWR_LINK_CHECK(rsmi_dev_target_graphics_version_get)

// Error Queries
WWR_LINK_CHECK(rsmi_dev_ecc_count_get)
WWR_LINK_CHECK(rsmi_dev_ecc_enabled_get)
WWR_LINK_CHECK(rsmi_dev_ecc_status_get)
WWR_LINK_CHECK(rsmi_status_string)

// Performance Counter Functions
WWR_LINK_CHECK(rsmi_dev_counter_group_supported)
WWR_LINK_CHECK(rsmi_dev_counter_create)
WWR_LINK_CHECK(rsmi_dev_counter_destroy)
WWR_LINK_CHECK(rsmi_counter_control)
WWR_LINK_CHECK(rsmi_counter_read)
WWR_LINK_CHECK(rsmi_counter_available_counters_get)

// System Information Functions
WWR_LINK_CHECK(rsmi_compute_process_info_get)
WWR_LINK_CHECK(rsmi_compute_process_info_by_pid_get)
WWR_LINK_CHECK(rsmi_compute_process_gpus_get)
WWR_LINK_CHECK(rsmi_compute_process_info_by_device_get)

// XGMI Functions
WWR_LINK_CHECK(rsmi_dev_xgmi_error_status)
WWR_LINK_CHECK(rsmi_dev_xgmi_error_reset)
WWR_LINK_CHECK(rsmi_dev_xgmi_hive_id_get)

// Hardware Topology Functions
WWR_LINK_CHECK(rsmi_topo_get_numa_node_number)
WWR_LINK_CHECK(rsmi_topo_get_link_weight)
WWR_LINK_CHECK(rsmi_minmax_bandwidth_get)
WWR_LINK_CHECK(rsmi_topo_get_link_type)
WWR_LINK_CHECK(rsmi_is_P2P_accessible)

// Compute Partition Functions
WWR_LINK_CHECK(rsmi_dev_compute_partition_get)
WWR_LINK_CHECK(rsmi_dev_compute_partition_set)
WWR_LINK_CHECK(rsmi_dev_partition_id_get)

// The Memory Partition Functions
WWR_LINK_CHECK(rsmi_dev_memory_partition_get)
WWR_LINK_CHECK(rsmi_dev_memory_partition_capabilities_get)
WWR_LINK_CHECK(rsmi_dev_memory_partition_set)

// Supported Functions
WWR_LINK_CHECK(rsmi_dev_supported_func_iterator_open)
WWR_LINK_CHECK(rsmi_dev_supported_variant_iterator_open)
WWR_LINK_CHECK(rsmi_func_iter_next)
WWR_LINK_CHECK(rsmi_dev_supported_func_iterator_close)
WWR_LINK_CHECK(rsmi_func_iter_value_get)

// Event Notification Functions
WWR_LINK_CHECK(rsmi_event_notification_init)
WWR_LINK_CHECK(rsmi_event_notification_mask_set)
WWR_LINK_CHECK(rsmi_event_notification_get)
WWR_LINK_CHECK(rsmi_event_notification_stop)

// Metric Functions
WWR_LINK_CHECK(rsmi_dev_metrics_header_info_get)
WWR_LINK_CHECK(rsmi_dev_metrics_xcd_counter_get)
WWR_LINK_CHECK(rsmi_dev_metrics_log_get)

} // namespace wwr::hip::test
