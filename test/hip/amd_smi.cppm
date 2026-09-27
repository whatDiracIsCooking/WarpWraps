// amd_smi.cppm - Compile-time tests for wwr.hip.amd_smi

module;

#include "test/shared/link_check.h"

export module wwr.test.hip.amd_smi;

import std;
import wwr.hip.amd_smi;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.hip.amd_smi
//
// The module is pure re-export (using declarations only -- amdsmi.h is a plain
// extern "C" API, no convenience-template collisions like hip_runtime_api.h).
// Runtime tests would just test AMD SMI itself, and it requires a live device;
// we verify at compile/link time that:
//   1. Enum types satisfy std::is_enum_v
//   2. Every enumerator value matches the compiled amdsmi.h value (generated
//      from the actual compiled values, not hand-copied)
//   3. Struct/union types are trivially copyable (C-interop guarantee)
//   4. Opaque handle typedefs have the expected pointer/integral shape
//   5. Link-time symbol resolution for every exported function
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::hip::test {

using namespace wwr::hip;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<amdsmi_init_flags_t>);
static_assert(std::is_enum_v<amdsmi_mm_ip_t>);
static_assert(std::is_enum_v<amdsmi_container_types_t>);
static_assert(std::is_enum_v<processor_type_t>);
static_assert(std::is_enum_v<amdsmi_status_t>);
static_assert(std::is_enum_v<amdsmi_clk_type_t>);
static_assert(std::is_enum_v<amdsmi_accelerator_partition_type_t>);
static_assert(std::is_enum_v<amdsmi_accelerator_partition_resource_type_t>);
static_assert(std::is_enum_v<amdsmi_compute_partition_type_t>);
static_assert(std::is_enum_v<amdsmi_memory_partition_type_t>);
static_assert(std::is_enum_v<amdsmi_temperature_type_t>);
static_assert(std::is_enum_v<amdsmi_fw_block_t>);
static_assert(std::is_enum_v<amdsmi_vram_type_t>);
static_assert(std::is_enum_v<amdsmi_card_form_factor_t>);
static_assert(std::is_enum_v<amdsmi_power_cap_type_t>);
static_assert(std::is_enum_v<amdsmi_cache_property_type_t>);
static_assert(std::is_enum_v<amdsmi_link_type_t>);
static_assert(std::is_enum_v<amdsmi_dev_perf_level_t>);
static_assert(std::is_enum_v<amdsmi_event_group_t>);
static_assert(std::is_enum_v<amdsmi_event_type_t>);
static_assert(std::is_enum_v<amdsmi_counter_command_t>);
static_assert(std::is_enum_v<amdsmi_evt_notification_type_t>);
static_assert(std::is_enum_v<amdsmi_temperature_metric_t>);
static_assert(std::is_enum_v<amdsmi_voltage_metric_t>);
static_assert(std::is_enum_v<amdsmi_voltage_type_t>);
static_assert(std::is_enum_v<amdsmi_power_profile_preset_masks_t>);
static_assert(std::is_enum_v<amdsmi_gpu_block_t>);
static_assert(std::is_enum_v<amdsmi_clk_limit_type_t>);
static_assert(std::is_enum_v<amdsmi_cper_sev_t>);
static_assert(std::is_enum_v<amdsmi_cper_notify_type_t>);
static_assert(std::is_enum_v<amdsmi_ras_err_state_t>);
static_assert(std::is_enum_v<amdsmi_memory_type_t>);
static_assert(std::is_enum_v<amdsmi_freq_ind_t>);
static_assert(std::is_enum_v<amdsmi_xgmi_status_t>);
static_assert(std::is_enum_v<amdsmi_memory_page_status_t>);
static_assert(std::is_enum_v<amdsmi_utilization_counter_type_t>);
static_assert(std::is_enum_v<amdsmi_xgmi_link_status_type_t>);
static_assert(std::is_enum_v<amdsmi_reg_type_t>);
static_assert(std::is_enum_v<amdsmi_virtualization_mode_t>);
static_assert(std::is_enum_v<amdsmi_affinity_scope_t>);
static_assert(std::is_enum_v<amdsmi_npm_status_t>);
static_assert(std::is_enum_v<amdsmi_ptl_data_format_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values (generated from the compiled amdsmi.h values)
// ────────────────────────────────────────────────────────────────────────

// amdsmi_init_flags_t
static_assert(static_cast<long long>(AMDSMI_INIT_ALL_PROCESSORS) == 4294967295);
static_assert(static_cast<long long>(AMDSMI_INIT_AMD_CPUS) == 1);
static_assert(static_cast<long long>(AMDSMI_INIT_AMD_GPUS) == 2);
static_assert(static_cast<long long>(AMDSMI_INIT_NON_AMD_CPUS) == 4);
static_assert(static_cast<long long>(AMDSMI_INIT_NON_AMD_GPUS) == 8);
static_assert(static_cast<long long>(AMDSMI_INIT_AMD_APUS) == 3);

// amdsmi_mm_ip_t
static_assert(static_cast<long long>(AMDSMI_MM_UVD) == 0);
static_assert(static_cast<long long>(AMDSMI_MM_VCE) == 1);
static_assert(static_cast<long long>(AMDSMI_MM_VCN) == 2);
static_assert(static_cast<long long>(AMDSMI_MM__MAX) == 3);

// amdsmi_container_types_t
static_assert(static_cast<long long>(AMDSMI_CONTAINER_LXC) == 0);
static_assert(static_cast<long long>(AMDSMI_CONTAINER_DOCKER) == 1);

// processor_type_t
static_assert(static_cast<long long>(AMDSMI_PROCESSOR_TYPE_UNKNOWN) == 0);
static_assert(static_cast<long long>(AMDSMI_PROCESSOR_TYPE_AMD_GPU) == 1);
static_assert(static_cast<long long>(AMDSMI_PROCESSOR_TYPE_AMD_CPU) == 2);
static_assert(static_cast<long long>(AMDSMI_PROCESSOR_TYPE_NON_AMD_GPU) == 3);
static_assert(static_cast<long long>(AMDSMI_PROCESSOR_TYPE_NON_AMD_CPU) == 4);
static_assert(static_cast<long long>(AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE) == 5);
static_assert(static_cast<long long>(AMDSMI_PROCESSOR_TYPE_AMD_APU) == 6);

// amdsmi_status_t
static_assert(static_cast<long long>(AMDSMI_STATUS_SUCCESS) == 0);
static_assert(static_cast<long long>(AMDSMI_STATUS_INVAL) == 1);
static_assert(static_cast<long long>(AMDSMI_STATUS_NOT_SUPPORTED) == 2);
static_assert(static_cast<long long>(AMDSMI_STATUS_NOT_YET_IMPLEMENTED) == 3);
static_assert(static_cast<long long>(AMDSMI_STATUS_FAIL_LOAD_MODULE) == 4);
static_assert(static_cast<long long>(AMDSMI_STATUS_FAIL_LOAD_SYMBOL) == 5);
static_assert(static_cast<long long>(AMDSMI_STATUS_DRM_ERROR) == 6);
static_assert(static_cast<long long>(AMDSMI_STATUS_API_FAILED) == 7);
static_assert(static_cast<long long>(AMDSMI_STATUS_TIMEOUT) == 8);
static_assert(static_cast<long long>(AMDSMI_STATUS_RETRY) == 9);
static_assert(static_cast<long long>(AMDSMI_STATUS_NO_PERM) == 10);
static_assert(static_cast<long long>(AMDSMI_STATUS_INTERRUPT) == 11);
static_assert(static_cast<long long>(AMDSMI_STATUS_IO) == 12);
static_assert(static_cast<long long>(AMDSMI_STATUS_ADDRESS_FAULT) == 13);
static_assert(static_cast<long long>(AMDSMI_STATUS_FILE_ERROR) == 14);
static_assert(static_cast<long long>(AMDSMI_STATUS_OUT_OF_RESOURCES) == 15);
static_assert(static_cast<long long>(AMDSMI_STATUS_INTERNAL_EXCEPTION) == 16);
static_assert(static_cast<long long>(AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS) == 17);
static_assert(static_cast<long long>(AMDSMI_STATUS_INIT_ERROR) == 18);
static_assert(static_cast<long long>(AMDSMI_STATUS_REFCOUNT_OVERFLOW) == 19);
static_assert(static_cast<long long>(AMDSMI_STATUS_DIRECTORY_NOT_FOUND) == 20);
static_assert(static_cast<long long>(AMDSMI_STATUS_BUSY) == 30);
static_assert(static_cast<long long>(AMDSMI_STATUS_NOT_FOUND) == 31);
static_assert(static_cast<long long>(AMDSMI_STATUS_NOT_INIT) == 32);
static_assert(static_cast<long long>(AMDSMI_STATUS_NO_SLOT) == 33);
static_assert(static_cast<long long>(AMDSMI_STATUS_DRIVER_NOT_LOADED) == 34);
static_assert(static_cast<long long>(AMDSMI_STATUS_MORE_DATA) == 39);
static_assert(static_cast<long long>(AMDSMI_STATUS_NO_DATA) == 40);
static_assert(static_cast<long long>(AMDSMI_STATUS_INSUFFICIENT_SIZE) == 41);
static_assert(static_cast<long long>(AMDSMI_STATUS_UNEXPECTED_SIZE) == 42);
static_assert(static_cast<long long>(AMDSMI_STATUS_UNEXPECTED_DATA) == 43);
static_assert(static_cast<long long>(AMDSMI_STATUS_NON_AMD_CPU) == 44);
static_assert(static_cast<long long>(AMDSMI_STATUS_NO_ENERGY_DRV) == 45);
static_assert(static_cast<long long>(AMDSMI_STATUS_NO_MSR_DRV) == 46);
static_assert(static_cast<long long>(AMDSMI_STATUS_NO_HSMP_DRV) == 47);
static_assert(static_cast<long long>(AMDSMI_STATUS_NO_HSMP_SUP) == 48);
static_assert(static_cast<long long>(AMDSMI_STATUS_NO_HSMP_MSG_SUP) == 49);
static_assert(static_cast<long long>(AMDSMI_STATUS_HSMP_TIMEOUT) == 50);
static_assert(static_cast<long long>(AMDSMI_STATUS_NO_DRV) == 51);
static_assert(static_cast<long long>(AMDSMI_STATUS_FILE_NOT_FOUND) == 52);
static_assert(static_cast<long long>(AMDSMI_STATUS_ARG_PTR_NULL) == 53);
static_assert(static_cast<long long>(AMDSMI_STATUS_AMDGPU_RESTART_ERR) == 54);
static_assert(static_cast<long long>(AMDSMI_STATUS_SETTING_UNAVAILABLE) == 55);
static_assert(static_cast<long long>(AMDSMI_STATUS_CORRUPTED_EEPROM) == 56);
static_assert(static_cast<long long>(AMDSMI_STATUS_MAP_ERROR) == 4294967294);
static_assert(static_cast<long long>(AMDSMI_STATUS_UNKNOWN_ERROR) == 4294967295);

// amdsmi_clk_type_t
static_assert(static_cast<long long>(AMDSMI_CLK_TYPE_SYS) == 0);
static_assert(static_cast<long long>(AMDSMI_CLK_TYPE_FIRST) == 0);
static_assert(static_cast<long long>(AMDSMI_CLK_TYPE_GFX) == 0);
static_assert(static_cast<long long>(AMDSMI_CLK_TYPE_DF) == 1);
static_assert(static_cast<long long>(AMDSMI_CLK_TYPE_DCEF) == 2);
static_assert(static_cast<long long>(AMDSMI_CLK_TYPE_SOC) == 3);
static_assert(static_cast<long long>(AMDSMI_CLK_TYPE_MEM) == 4);
static_assert(static_cast<long long>(AMDSMI_CLK_TYPE_PCIE) == 5);
static_assert(static_cast<long long>(AMDSMI_CLK_TYPE_VCLK0) == 6);
static_assert(static_cast<long long>(AMDSMI_CLK_TYPE_VCLK1) == 7);
static_assert(static_cast<long long>(AMDSMI_CLK_TYPE_DCLK0) == 8);
static_assert(static_cast<long long>(AMDSMI_CLK_TYPE_DCLK1) == 9);
static_assert(static_cast<long long>(AMDSMI_CLK_TYPE__MAX) == 9);

// amdsmi_accelerator_partition_type_t
static_assert(static_cast<long long>(AMDSMI_ACCELERATOR_PARTITION_INVALID) == 0);
static_assert(static_cast<long long>(AMDSMI_ACCELERATOR_PARTITION_SPX) == 1);
static_assert(static_cast<long long>(AMDSMI_ACCELERATOR_PARTITION_DPX) == 2);
static_assert(static_cast<long long>(AMDSMI_ACCELERATOR_PARTITION_TPX) == 3);
static_assert(static_cast<long long>(AMDSMI_ACCELERATOR_PARTITION_QPX) == 4);
static_assert(static_cast<long long>(AMDSMI_ACCELERATOR_PARTITION_CPX) == 5);
static_assert(static_cast<long long>(AMDSMI_ACCELERATOR_PARTITION_MAX) == 6);

// amdsmi_accelerator_partition_resource_type_t
static_assert(static_cast<long long>(AMDSMI_ACCELERATOR_XCC) == 0);
static_assert(static_cast<long long>(AMDSMI_ACCELERATOR_ENCODER) == 1);
static_assert(static_cast<long long>(AMDSMI_ACCELERATOR_DECODER) == 2);
static_assert(static_cast<long long>(AMDSMI_ACCELERATOR_DMA) == 3);
static_assert(static_cast<long long>(AMDSMI_ACCELERATOR_JPEG) == 4);
static_assert(static_cast<long long>(AMDSMI_ACCELERATOR_MAX) == 5);

// amdsmi_compute_partition_type_t
static_assert(static_cast<long long>(AMDSMI_COMPUTE_PARTITION_INVALID) == 0);
static_assert(static_cast<long long>(AMDSMI_COMPUTE_PARTITION_SPX) == 1);
static_assert(static_cast<long long>(AMDSMI_COMPUTE_PARTITION_DPX) == 2);
static_assert(static_cast<long long>(AMDSMI_COMPUTE_PARTITION_TPX) == 3);
static_assert(static_cast<long long>(AMDSMI_COMPUTE_PARTITION_QPX) == 4);
static_assert(static_cast<long long>(AMDSMI_COMPUTE_PARTITION_CPX) == 5);

// amdsmi_memory_partition_type_t
static_assert(static_cast<long long>(AMDSMI_MEMORY_PARTITION_UNKNOWN) == 0);
static_assert(static_cast<long long>(AMDSMI_MEMORY_PARTITION_NPS1) == 1);
static_assert(static_cast<long long>(AMDSMI_MEMORY_PARTITION_NPS2) == 2);
static_assert(static_cast<long long>(AMDSMI_MEMORY_PARTITION_NPS4) == 4);
static_assert(static_cast<long long>(AMDSMI_MEMORY_PARTITION_NPS8) == 8);

// amdsmi_temperature_type_t
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_EDGE) == 0);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_FIRST) == 0);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_HOTSPOT) == 1);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_JUNCTION) == 1);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_VRAM) == 2);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_HBM_0) == 3);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_HBM_1) == 4);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_HBM_2) == 5);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_HBM_3) == 6);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_PLX) == 7);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_FIRST) == 100);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_RETIMER_X) == 100);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_IBC) == 101);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_IBC_2) == 102);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_VDD18_VR) == 103);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_B_VR) ==
              104);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_D_VR) ==
              105);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_LAST) == 149);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VR_FIRST) == 150);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD0) == 150);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD1) == 151);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD2) == 152);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD3) == 153);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOC_A) == 154);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOC_C) == 155);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOCIO_A) == 156);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOCIO_C) == 157);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDD_085_HBM) == 158);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_11_HBM_B) == 159);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_11_HBM_D) == 160);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDD_USR) == 161);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_11_E32) == 162);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VR_LAST) == 199);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_FIRST) == 200);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA) == 200);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FRONT) == 201);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_BACK) == 202);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_OAM7) == 203);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_IBC) == 204);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_UFPGA) == 205);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_OAM1) == 206);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_0_1_HSC) == 207);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_2_3_HSC) == 208);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_4_5_HSC) == 209);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_6_7_HSC) == 210);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA_0V72_VR) == 211);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA_3V3_VR) == 212);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_0_1_2_3_1V2_VR) ==
              213);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_4_5_6_7_1V2_VR) ==
              214);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_0_1_0V9_VR) == 215);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_4_5_0V9_VR) == 216);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_2_3_0V9_VR) == 217);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_6_7_0V9_VR) == 218);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_0_1_2_3_3V3_VR) == 219);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_4_5_6_7_3V3_VR) == 220);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_IBC_HSC) == 221);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_IBC) == 222);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_LAST) == 249);
static_assert(static_cast<long long>(AMDSMI_TEMPERATURE_TYPE__MAX) == 249);

// amdsmi_fw_block_t
static_assert(static_cast<long long>(AMDSMI_FW_ID_SMU) == 1);
static_assert(static_cast<long long>(AMDSMI_FW_ID_FIRST) == 1);
static_assert(static_cast<long long>(AMDSMI_FW_ID_CP_CE) == 2);
static_assert(static_cast<long long>(AMDSMI_FW_ID_CP_PFP) == 3);
static_assert(static_cast<long long>(AMDSMI_FW_ID_CP_ME) == 4);
static_assert(static_cast<long long>(AMDSMI_FW_ID_CP_MEC_JT1) == 5);
static_assert(static_cast<long long>(AMDSMI_FW_ID_CP_MEC_JT2) == 6);
static_assert(static_cast<long long>(AMDSMI_FW_ID_CP_MEC1) == 7);
static_assert(static_cast<long long>(AMDSMI_FW_ID_CP_MEC2) == 8);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RLC) == 9);
static_assert(static_cast<long long>(AMDSMI_FW_ID_SDMA0) == 10);
static_assert(static_cast<long long>(AMDSMI_FW_ID_SDMA1) == 11);
static_assert(static_cast<long long>(AMDSMI_FW_ID_SDMA2) == 12);
static_assert(static_cast<long long>(AMDSMI_FW_ID_SDMA3) == 13);
static_assert(static_cast<long long>(AMDSMI_FW_ID_SDMA4) == 14);
static_assert(static_cast<long long>(AMDSMI_FW_ID_SDMA5) == 15);
static_assert(static_cast<long long>(AMDSMI_FW_ID_SDMA6) == 16);
static_assert(static_cast<long long>(AMDSMI_FW_ID_SDMA7) == 17);
static_assert(static_cast<long long>(AMDSMI_FW_ID_VCN) == 18);
static_assert(static_cast<long long>(AMDSMI_FW_ID_UVD) == 19);
static_assert(static_cast<long long>(AMDSMI_FW_ID_VCE) == 20);
static_assert(static_cast<long long>(AMDSMI_FW_ID_ISP) == 21);
static_assert(static_cast<long long>(AMDSMI_FW_ID_DMCU_ERAM) == 22);
static_assert(static_cast<long long>(AMDSMI_FW_ID_DMCU_ISR) == 23);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM) == 24);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM) == 25);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL) == 26);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RLC_V) == 27);
static_assert(static_cast<long long>(AMDSMI_FW_ID_MMSCH) == 28);
static_assert(static_cast<long long>(AMDSMI_FW_ID_PSP_SYSDRV) == 29);
static_assert(static_cast<long long>(AMDSMI_FW_ID_PSP_SOSDRV) == 30);
static_assert(static_cast<long long>(AMDSMI_FW_ID_PSP_TOC) == 31);
static_assert(static_cast<long long>(AMDSMI_FW_ID_PSP_KEYDB) == 32);
static_assert(static_cast<long long>(AMDSMI_FW_ID_DFC) == 33);
static_assert(static_cast<long long>(AMDSMI_FW_ID_PSP_SPL) == 34);
static_assert(static_cast<long long>(AMDSMI_FW_ID_DRV_CAP) == 35);
static_assert(static_cast<long long>(AMDSMI_FW_ID_MC) == 36);
static_assert(static_cast<long long>(AMDSMI_FW_ID_PSP_BL) == 37);
static_assert(static_cast<long long>(AMDSMI_FW_ID_CP_PM4) == 38);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RLC_P) == 39);
static_assert(static_cast<long long>(AMDSMI_FW_ID_SEC_POLICY_STAGE2) == 40);
static_assert(static_cast<long long>(AMDSMI_FW_ID_REG_ACCESS_WHITELIST) == 41);
static_assert(static_cast<long long>(AMDSMI_FW_ID_IMU_DRAM) == 42);
static_assert(static_cast<long long>(AMDSMI_FW_ID_IMU_IRAM) == 43);
static_assert(static_cast<long long>(AMDSMI_FW_ID_SDMA_TH0) == 44);
static_assert(static_cast<long long>(AMDSMI_FW_ID_SDMA_TH1) == 45);
static_assert(static_cast<long long>(AMDSMI_FW_ID_CP_MES) == 46);
static_assert(static_cast<long long>(AMDSMI_FW_ID_MES_KIQ) == 47);
static_assert(static_cast<long long>(AMDSMI_FW_ID_MES_STACK) == 48);
static_assert(static_cast<long long>(AMDSMI_FW_ID_MES_THREAD1) == 49);
static_assert(static_cast<long long>(AMDSMI_FW_ID_MES_THREAD1_STACK) == 50);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RLX6) == 51);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RLX6_DRAM_BOOT) == 52);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RS64_ME) == 53);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RS64_ME_P0_DATA) == 54);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RS64_ME_P1_DATA) == 55);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RS64_PFP) == 56);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RS64_PFP_P0_DATA) == 57);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RS64_PFP_P1_DATA) == 58);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RS64_MEC) == 59);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RS64_MEC_P0_DATA) == 60);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RS64_MEC_P1_DATA) == 61);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RS64_MEC_P2_DATA) == 62);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RS64_MEC_P3_DATA) == 63);
static_assert(static_cast<long long>(AMDSMI_FW_ID_PPTABLE) == 64);
static_assert(static_cast<long long>(AMDSMI_FW_ID_PSP_SOC) == 65);
static_assert(static_cast<long long>(AMDSMI_FW_ID_PSP_DBG) == 66);
static_assert(static_cast<long long>(AMDSMI_FW_ID_PSP_INTF) == 67);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RLX6_CORE1) == 68);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RLX6_DRAM_BOOT_CORE1) == 69);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RLCV_LX7) == 70);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RLC_SAVE_RESTORE_LIST) == 71);
static_assert(static_cast<long long>(AMDSMI_FW_ID_ASD) == 72);
static_assert(static_cast<long long>(AMDSMI_FW_ID_TA_RAS) == 73);
static_assert(static_cast<long long>(AMDSMI_FW_ID_TA_XGMI) == 74);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RLC_SRLG) == 75);
static_assert(static_cast<long long>(AMDSMI_FW_ID_RLC_SRLS) == 76);
static_assert(static_cast<long long>(AMDSMI_FW_ID_PM) == 77);
static_assert(static_cast<long long>(AMDSMI_FW_ID_DMCU) == 78);
static_assert(static_cast<long long>(AMDSMI_FW_ID_PLDM_BUNDLE) == 79);
static_assert(static_cast<long long>(AMDSMI_FW_ID__MAX) == 80);

// amdsmi_vram_type_t
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_UNKNOWN) == 0);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_HBM) == 1);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_HBM2) == 2);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_HBM2E) == 3);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_HBM3) == 4);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_HBM3E) == 5);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_DDR2) == 10);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_DDR3) == 11);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_DDR4) == 12);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_DDR5) == 13);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_GDDR1) == 17);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_GDDR2) == 18);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_GDDR3) == 19);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_GDDR4) == 20);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_GDDR5) == 21);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_GDDR6) == 22);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_GDDR7) == 23);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_LPDDR4) == 30);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE_LPDDR5) == 31);
static_assert(static_cast<long long>(AMDSMI_VRAM_TYPE__MAX) == 31);

// amdsmi_card_form_factor_t
static_assert(static_cast<long long>(AMDSMI_CARD_FORM_FACTOR_PCIE) == 0);
static_assert(static_cast<long long>(AMDSMI_CARD_FORM_FACTOR_OAM) == 1);
static_assert(static_cast<long long>(AMDSMI_CARD_FORM_FACTOR_CEM) == 2);
static_assert(static_cast<long long>(AMDSMI_CARD_FORM_FACTOR_UNKNOWN) == 3);

// amdsmi_power_cap_type_t
static_assert(static_cast<long long>(AMDSMI_POWER_CAP_TYPE_PPT0) == 0);
static_assert(static_cast<long long>(AMDSMI_POWER_CAP_TYPE_PPT1) == 1);

// amdsmi_cache_property_type_t
static_assert(static_cast<long long>(AMDSMI_CACHE_PROPERTY_ENABLED) == 1);
static_assert(static_cast<long long>(AMDSMI_CACHE_PROPERTY_DATA_CACHE) == 2);
static_assert(static_cast<long long>(AMDSMI_CACHE_PROPERTY_INST_CACHE) == 4);
static_assert(static_cast<long long>(AMDSMI_CACHE_PROPERTY_CPU_CACHE) == 8);
static_assert(static_cast<long long>(AMDSMI_CACHE_PROPERTY_SIMD_CACHE) == 16);

// amdsmi_link_type_t
static_assert(static_cast<long long>(AMDSMI_LINK_TYPE_INTERNAL) == 0);
static_assert(static_cast<long long>(AMDSMI_LINK_TYPE_PCIE) == 1);
static_assert(static_cast<long long>(AMDSMI_LINK_TYPE_XGMI) == 2);
static_assert(static_cast<long long>(AMDSMI_LINK_TYPE_NOT_APPLICABLE) == 3);
static_assert(static_cast<long long>(AMDSMI_LINK_TYPE_UNKNOWN) == 4);

// amdsmi_dev_perf_level_t
static_assert(static_cast<long long>(AMDSMI_DEV_PERF_LEVEL_AUTO) == 0);
static_assert(static_cast<long long>(AMDSMI_DEV_PERF_LEVEL_FIRST) == 0);
static_assert(static_cast<long long>(AMDSMI_DEV_PERF_LEVEL_LOW) == 1);
static_assert(static_cast<long long>(AMDSMI_DEV_PERF_LEVEL_HIGH) == 2);
static_assert(static_cast<long long>(AMDSMI_DEV_PERF_LEVEL_MANUAL) == 3);
static_assert(static_cast<long long>(AMDSMI_DEV_PERF_LEVEL_STABLE_STD) == 4);
static_assert(static_cast<long long>(AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK) == 5);
static_assert(static_cast<long long>(AMDSMI_DEV_PERF_LEVEL_STABLE_MIN_MCLK) == 6);
static_assert(static_cast<long long>(AMDSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK) == 7);
static_assert(static_cast<long long>(AMDSMI_DEV_PERF_LEVEL_DETERMINISM) == 8);
static_assert(static_cast<long long>(AMDSMI_DEV_PERF_LEVEL_LAST) == 8);
static_assert(static_cast<long long>(AMDSMI_DEV_PERF_LEVEL_UNKNOWN) == 256);

// amdsmi_event_group_t
static_assert(static_cast<long long>(AMDSMI_EVNT_GRP_XGMI) == 0);
static_assert(static_cast<long long>(AMDSMI_EVNT_GRP_XGMI_DATA_OUT) == 10);
static_assert(static_cast<long long>(AMDSMI_EVNT_GRP_INVALID) == 4294967295);

// amdsmi_event_type_t
static_assert(static_cast<long long>(AMDSMI_EVNT_FIRST) == 0);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_FIRST) == 0);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_0_NOP_TX) == 0);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_0_REQUEST_TX) == 1);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_0_RESPONSE_TX) == 2);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_0_BEATS_TX) == 3);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_1_NOP_TX) == 4);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_1_REQUEST_TX) == 5);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_1_RESPONSE_TX) == 6);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_1_BEATS_TX) == 7);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_LAST) == 7);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_DATA_OUT_FIRST) == 10);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_DATA_OUT_0) == 10);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_DATA_OUT_1) == 11);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_DATA_OUT_2) == 12);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_DATA_OUT_3) == 13);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_DATA_OUT_4) == 14);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_DATA_OUT_5) == 15);
static_assert(static_cast<long long>(AMDSMI_EVNT_XGMI_DATA_OUT_LAST) == 15);
static_assert(static_cast<long long>(AMDSMI_EVNT_LAST) == 15);

// amdsmi_counter_command_t
static_assert(static_cast<long long>(AMDSMI_CNTR_CMD_START) == 0);
static_assert(static_cast<long long>(AMDSMI_CNTR_CMD_STOP) == 1);

// amdsmi_evt_notification_type_t
static_assert(static_cast<long long>(AMDSMI_EVT_NOTIF_NONE) == 0);
static_assert(static_cast<long long>(AMDSMI_EVT_NOTIF_VMFAULT) == 1);
static_assert(static_cast<long long>(AMDSMI_EVT_NOTIF_FIRST) == 1);
static_assert(static_cast<long long>(AMDSMI_EVT_NOTIF_THERMAL_THROTTLE) == 2);
static_assert(static_cast<long long>(AMDSMI_EVT_NOTIF_GPU_PRE_RESET) == 3);
static_assert(static_cast<long long>(AMDSMI_EVT_NOTIF_GPU_POST_RESET) == 4);
static_assert(static_cast<long long>(AMDSMI_EVT_NOTIF_MIGRATE_START) == 5);
static_assert(static_cast<long long>(AMDSMI_EVT_NOTIF_MIGRATE_END) == 6);
static_assert(static_cast<long long>(AMDSMI_EVT_NOTIF_PAGE_FAULT_START) == 7);
static_assert(static_cast<long long>(AMDSMI_EVT_NOTIF_PAGE_FAULT_END) == 8);
static_assert(static_cast<long long>(AMDSMI_EVT_NOTIF_QUEUE_EVICTION) == 9);
static_assert(static_cast<long long>(AMDSMI_EVT_NOTIF_QUEUE_RESTORE) == 10);
static_assert(static_cast<long long>(AMDSMI_EVT_NOTIF_UNMAP_FROM_GPU) == 11);
static_assert(static_cast<long long>(AMDSMI_EVT_NOTIF_PROCESS_START) == 12);
static_assert(static_cast<long long>(AMDSMI_EVT_NOTIF_PROCESS_END) == 13);
static_assert(static_cast<long long>(AMDSMI_EVT_NOTIF_LAST) == 13);

// amdsmi_temperature_metric_t
static_assert(static_cast<long long>(AMDSMI_TEMP_CURRENT) == 0);
static_assert(static_cast<long long>(AMDSMI_TEMP_FIRST) == 0);
static_assert(static_cast<long long>(AMDSMI_TEMP_MAX) == 1);
static_assert(static_cast<long long>(AMDSMI_TEMP_MIN) == 2);
static_assert(static_cast<long long>(AMDSMI_TEMP_MAX_HYST) == 3);
static_assert(static_cast<long long>(AMDSMI_TEMP_MIN_HYST) == 4);
static_assert(static_cast<long long>(AMDSMI_TEMP_CRITICAL) == 5);
static_assert(static_cast<long long>(AMDSMI_TEMP_CRITICAL_HYST) == 6);
static_assert(static_cast<long long>(AMDSMI_TEMP_EMERGENCY) == 7);
static_assert(static_cast<long long>(AMDSMI_TEMP_EMERGENCY_HYST) == 8);
static_assert(static_cast<long long>(AMDSMI_TEMP_CRIT_MIN) == 9);
static_assert(static_cast<long long>(AMDSMI_TEMP_CRIT_MIN_HYST) == 10);
static_assert(static_cast<long long>(AMDSMI_TEMP_OFFSET) == 11);
static_assert(static_cast<long long>(AMDSMI_TEMP_LOWEST) == 12);
static_assert(static_cast<long long>(AMDSMI_TEMP_HIGHEST) == 13);
static_assert(static_cast<long long>(AMDSMI_TEMP_SHUTDOWN) == 14);
static_assert(static_cast<long long>(AMDSMI_TEMP_LAST) == 14);

// amdsmi_voltage_metric_t
static_assert(static_cast<long long>(AMDSMI_VOLT_CURRENT) == 0);
static_assert(static_cast<long long>(AMDSMI_VOLT_FIRST) == 0);
static_assert(static_cast<long long>(AMDSMI_VOLT_MAX) == 1);
static_assert(static_cast<long long>(AMDSMI_VOLT_MIN_CRIT) == 2);
static_assert(static_cast<long long>(AMDSMI_VOLT_MIN) == 3);
static_assert(static_cast<long long>(AMDSMI_VOLT_MAX_CRIT) == 4);
static_assert(static_cast<long long>(AMDSMI_VOLT_AVERAGE) == 5);
static_assert(static_cast<long long>(AMDSMI_VOLT_LOWEST) == 6);
static_assert(static_cast<long long>(AMDSMI_VOLT_HIGHEST) == 7);
static_assert(static_cast<long long>(AMDSMI_VOLT_LAST) == 7);

// amdsmi_voltage_type_t
static_assert(static_cast<long long>(AMDSMI_VOLT_TYPE_FIRST) == 0);
static_assert(static_cast<long long>(AMDSMI_VOLT_TYPE_VDDGFX) == 0);
static_assert(static_cast<long long>(AMDSMI_VOLT_TYPE_VDDBOARD) == 1);
static_assert(static_cast<long long>(AMDSMI_VOLT_TYPE_LAST) == 1);
static_assert(static_cast<long long>(AMDSMI_VOLT_TYPE_INVALID) == 4294967295);

// amdsmi_power_profile_preset_masks_t
static_assert(static_cast<long long>(AMDSMI_PWR_PROF_PRST_CUSTOM_MASK) == 1);
static_assert(static_cast<long long>(AMDSMI_PWR_PROF_PRST_VIDEO_MASK) == 2);
static_assert(static_cast<long long>(AMDSMI_PWR_PROF_PRST_POWER_SAVING_MASK) == 4);
static_assert(static_cast<long long>(AMDSMI_PWR_PROF_PRST_COMPUTE_MASK) == 8);
static_assert(static_cast<long long>(AMDSMI_PWR_PROF_PRST_VR_MASK) == 16);
static_assert(static_cast<long long>(AMDSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK) == 32);
static_assert(static_cast<long long>(AMDSMI_PWR_PROF_PRST_BOOTUP_DEFAULT) == 64);
static_assert(static_cast<long long>(AMDSMI_PWR_PROF_PRST_LAST) == 64);
static_assert(static_cast<long long>(AMDSMI_PWR_PROF_PRST_INVALID) == -1);

// amdsmi_gpu_block_t
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_INVALID) == 0);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_FIRST) == 1);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_UMC) == 1);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_SDMA) == 2);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_GFX) == 4);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_MMHUB) == 8);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_ATHUB) == 16);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_PCIE_BIF) == 32);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_HDP) == 64);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_XGMI_WAFL) == 128);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_DF) == 256);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_SMN) == 512);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_SEM) == 1024);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_MP0) == 2048);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_MP1) == 4096);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_FUSE) == 8192);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_MCA) == 16384);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_VCN) == 32768);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_JPEG) == 65536);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_IH) == 131072);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_MPIO) == 262144);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_LAST) == 262144);
static_assert(static_cast<long long>(AMDSMI_GPU_BLOCK_RESERVED) == -9223372036854775808);

// amdsmi_clk_limit_type_t
static_assert(static_cast<long long>(CLK_LIMIT_MIN) == 0);
static_assert(static_cast<long long>(CLK_LIMIT_MAX) == 1);

// amdsmi_cper_sev_t
static_assert(static_cast<long long>(AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED) == 0);
static_assert(static_cast<long long>(AMDSMI_CPER_SEV_FATAL) == 1);
static_assert(static_cast<long long>(AMDSMI_CPER_SEV_NON_FATAL_CORRECTED) == 2);
static_assert(static_cast<long long>(AMDSMI_CPER_SEV_NUM) == 3);
static_assert(static_cast<long long>(AMDSMI_CPER_SEV_UNUSED) == 10);

// amdsmi_cper_notify_type_t
static_assert(static_cast<long long>(AMDSMI_CPER_NOTIFY_TYPE_CMC) == 4976123370175105969);
static_assert(static_cast<long long>(AMDSMI_CPER_NOTIFY_TYPE_CPE) == 5356425115412803478);
static_assert(static_cast<long long>(AMDSMI_CPER_NOTIFY_TYPE_MCE) == 5531987820403847166);
static_assert(static_cast<long long>(AMDSMI_CPER_NOTIFY_TYPE_PCIE) == 5619395120325705759);
static_assert(static_cast<long long>(AMDSMI_CPER_NOTIFY_TYPE_INIT) == 4992964802890589160);
static_assert(static_cast<long long>(AMDSMI_CPER_NOTIFY_TYPE_NMI) == 4812579876830546431);
static_assert(static_cast<long long>(AMDSMI_CPER_NOTIFY_TYPE_BOOT) == 4655221457236894822);
static_assert(static_cast<long long>(AMDSMI_CPER_NOTIFY_TYPE_DMAR) == 5487573144795207569);
static_assert(static_cast<long long>(AMDSMI_CPER_NOTIFY_TYPE_SEA) == 1289362001033197706);
static_assert(static_cast<long long>(AMDSMI_CPER_NOTIFY_TYPE_SEI) == 5658685719731260545);
static_assert(static_cast<long long>(AMDSMI_CPER_NOTIFY_TYPE_PEI) == 4761520883332928940);
static_assert(static_cast<long long>(AMDSMI_CPER_NOTIFY_TYPE_CXL_COMPONENT) == 5306157213770398665);

// amdsmi_ras_err_state_t
static_assert(static_cast<long long>(AMDSMI_RAS_ERR_STATE_NONE) == 0);
static_assert(static_cast<long long>(AMDSMI_RAS_ERR_STATE_DISABLED) == 1);
static_assert(static_cast<long long>(AMDSMI_RAS_ERR_STATE_PARITY) == 2);
static_assert(static_cast<long long>(AMDSMI_RAS_ERR_STATE_SING_C) == 3);
static_assert(static_cast<long long>(AMDSMI_RAS_ERR_STATE_MULT_UC) == 4);
static_assert(static_cast<long long>(AMDSMI_RAS_ERR_STATE_POISON) == 5);
static_assert(static_cast<long long>(AMDSMI_RAS_ERR_STATE_ENABLED) == 6);
static_assert(static_cast<long long>(AMDSMI_RAS_ERR_STATE_LAST) == 6);
static_assert(static_cast<long long>(AMDSMI_RAS_ERR_STATE_INVALID) == 4294967295);

// amdsmi_memory_type_t
static_assert(static_cast<long long>(AMDSMI_MEM_TYPE_FIRST) == 0);
static_assert(static_cast<long long>(AMDSMI_MEM_TYPE_VRAM) == 0);
static_assert(static_cast<long long>(AMDSMI_MEM_TYPE_VIS_VRAM) == 1);
static_assert(static_cast<long long>(AMDSMI_MEM_TYPE_GTT) == 2);
static_assert(static_cast<long long>(AMDSMI_MEM_TYPE_LAST) == 2);

// amdsmi_freq_ind_t
static_assert(static_cast<long long>(AMDSMI_FREQ_IND_MIN) == 0);
static_assert(static_cast<long long>(AMDSMI_FREQ_IND_MAX) == 1);
static_assert(static_cast<long long>(AMDSMI_FREQ_IND_INVALID) == 4294967295);

// amdsmi_xgmi_status_t
static_assert(static_cast<long long>(AMDSMI_XGMI_STATUS_NO_ERRORS) == 0);
static_assert(static_cast<long long>(AMDSMI_XGMI_STATUS_ERROR) == 1);
static_assert(static_cast<long long>(AMDSMI_XGMI_STATUS_MULTIPLE_ERRORS) == 2);

// amdsmi_memory_page_status_t
static_assert(static_cast<long long>(AMDSMI_MEM_PAGE_STATUS_RESERVED) == 0);
static_assert(static_cast<long long>(AMDSMI_MEM_PAGE_STATUS_PENDING) == 1);
static_assert(static_cast<long long>(AMDSMI_MEM_PAGE_STATUS_UNRESERVABLE) == 2);

// amdsmi_utilization_counter_type_t
static_assert(static_cast<long long>(AMDSMI_UTILIZATION_COUNTER_FIRST) == 0);
static_assert(static_cast<long long>(AMDSMI_COARSE_GRAIN_GFX_ACTIVITY) == 0);
static_assert(static_cast<long long>(AMDSMI_COARSE_GRAIN_MEM_ACTIVITY) == 1);
static_assert(static_cast<long long>(AMDSMI_COARSE_DECODER_ACTIVITY) == 2);
static_assert(static_cast<long long>(AMDSMI_FINE_GRAIN_GFX_ACTIVITY) == 100);
static_assert(static_cast<long long>(AMDSMI_FINE_GRAIN_MEM_ACTIVITY) == 101);
static_assert(static_cast<long long>(AMDSMI_FINE_DECODER_ACTIVITY) == 102);
static_assert(static_cast<long long>(AMDSMI_UTILIZATION_COUNTER_LAST) == 102);

// amdsmi_xgmi_link_status_type_t
static_assert(static_cast<long long>(AMDSMI_XGMI_LINK_DOWN) == 0);
static_assert(static_cast<long long>(AMDSMI_XGMI_LINK_UP) == 1);
static_assert(static_cast<long long>(AMDSMI_XGMI_LINK_DISABLE) == 2);

// amdsmi_reg_type_t
static_assert(static_cast<long long>(AMDSMI_REG_XGMI) == 0);
static_assert(static_cast<long long>(AMDSMI_REG_WAFL) == 1);
static_assert(static_cast<long long>(AMDSMI_REG_PCIE) == 2);
static_assert(static_cast<long long>(AMDSMI_REG_USR) == 3);
static_assert(static_cast<long long>(AMDSMI_REG_USR1) == 4);

// amdsmi_virtualization_mode_t
static_assert(static_cast<long long>(AMDSMI_VIRTUALIZATION_MODE_UNKNOWN) == 0);
static_assert(static_cast<long long>(AMDSMI_VIRTUALIZATION_MODE_BAREMETAL) == 1);
static_assert(static_cast<long long>(AMDSMI_VIRTUALIZATION_MODE_HOST) == 2);
static_assert(static_cast<long long>(AMDSMI_VIRTUALIZATION_MODE_GUEST) == 3);
static_assert(static_cast<long long>(AMDSMI_VIRTUALIZATION_MODE_PASSTHROUGH) == 4);

// amdsmi_affinity_scope_t
static_assert(static_cast<long long>(AMDSMI_AFFINITY_SCOPE_NODE) == 0);
static_assert(static_cast<long long>(AMDSMI_AFFINITY_SCOPE_SOCKET) == 1);

// amdsmi_npm_status_t
static_assert(static_cast<long long>(AMDSMI_NPM_STATUS_DISABLED) == 0);
static_assert(static_cast<long long>(AMDSMI_NPM_STATUS_ENABLED) == 1);

// amdsmi_ptl_data_format_t
static_assert(static_cast<long long>(AMDSMI_PTL_DATA_FORMAT_I8) == 0);
static_assert(static_cast<long long>(AMDSMI_PTL_DATA_FORMAT_F16) == 1);
static_assert(static_cast<long long>(AMDSMI_PTL_DATA_FORMAT_BF16) == 2);
static_assert(static_cast<long long>(AMDSMI_PTL_DATA_FORMAT_F32) == 3);
static_assert(static_cast<long long>(AMDSMI_PTL_DATA_FORMAT_F64) == 4);
static_assert(static_cast<long long>(AMDSMI_PTL_DATA_FORMAT_F8) == 5);
static_assert(static_cast<long long>(AMDSMI_PTL_DATA_FORMAT_VECTOR) == 6);
static_assert(static_cast<long long>(AMDSMI_PTL_DATA_FORMAT_INVALID) == 4294967295);

// ────────────────────────────────────────────────────────────────────────
// Struct / union traits: trivial copyability (C-interop guarantee)
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_trivially_copyable_v<amdsmi_range_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_xgmi_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_vram_usage_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_violation_status_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_frequency_range_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_bdf_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_enumeration_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_pcie_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_power_cap_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_vbios_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_gpu_cache_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_fw_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_asic_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_kfd_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_nps_caps_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_memory_partition_config_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_accelerator_partition_profile_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_accelerator_partition_resource_profile_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_accelerator_partition_profile_config_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_cpu_util_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_link_metrics_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_vram_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_driver_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_board_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_power_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_clk_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_engine_usage_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_proc_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_p2p_capability_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_counter_value_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_evt_notification_data_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_utilization_counter_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_retired_page_record_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_power_profile_status_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_frequencies_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_dpm_policy_entry_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_dpm_policy_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_pcie_bandwidth_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_version_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_od_vddc_point_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_freq_volt_region_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_od_volt_curve_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_od_volt_freq_data_t>);
static_assert(std::is_trivially_copyable_v<amd_metrics_table_header_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_gpu_xcp_metrics_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_gpu_metrics_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_xgmi_link_status_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_name_value_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_ras_feature_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_error_count_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_process_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_topology_nearest_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_npm_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_sock_info_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_cper_guid_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_cper_timestamp_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_cper_valid_bits_t>);
static_assert(std::is_trivially_copyable_v<amdsmi_cper_hdr_t>);

// ────────────────────────────────────────────────────────────────────────
// Opaque handle typedefs
// ────────────────────────────────────────────────────────────────────────

// amdsmi_processor_handle / amdsmi_socket_handle / amdsmi_node_handle are
// `void*`-shaped opaque handles.
static_assert(std::is_pointer_v<amdsmi_processor_handle>);
static_assert(std::is_pointer_v<amdsmi_socket_handle>);
static_assert(std::is_pointer_v<amdsmi_node_handle>);
// amdsmi_process_handle_t (uint32_t), amdsmi_event_handle_t (uintptr_t), and
// amdsmi_bit_field_t (uint64_t) are integral typedefs, not pointers -- sized to
// match their underlying type.
static_assert(sizeof(amdsmi_process_handle_t) == sizeof(std::uint32_t));
static_assert(sizeof(amdsmi_event_handle_t) == sizeof(std::uintptr_t));
static_assert(sizeof(amdsmi_bit_field_t) == sizeof(std::uint64_t));

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// ────────────────────────────────────────────────────────────────────────

// Initialization and Shutdown
WWR_LINK_CHECK(amdsmi_init)
WWR_LINK_CHECK(amdsmi_shut_down)

// Discovery Queries
WWR_LINK_CHECK(amdsmi_get_socket_handles)
WWR_LINK_CHECK(amdsmi_get_socket_info)
WWR_LINK_CHECK(amdsmi_get_processor_handles)
WWR_LINK_CHECK(amdsmi_get_node_handle)
WWR_LINK_CHECK(amdsmi_get_processor_type)
WWR_LINK_CHECK(amdsmi_get_processor_handle_from_bdf)
WWR_LINK_CHECK(amdsmi_get_gpu_device_bdf)
WWR_LINK_CHECK(amdsmi_get_gpu_device_uuid)
WWR_LINK_CHECK(amdsmi_get_gpu_enumeration_info)
WWR_LINK_CHECK(amdsmi_get_cpu_affinity_with_scope)
WWR_LINK_CHECK(amdsmi_get_gpu_virtualization_mode)

// Identifier Queries
WWR_LINK_CHECK(amdsmi_get_gpu_id)
WWR_LINK_CHECK(amdsmi_get_gpu_revision)
WWR_LINK_CHECK(amdsmi_get_gpu_vendor_name)
WWR_LINK_CHECK(amdsmi_get_gpu_vram_vendor)
WWR_LINK_CHECK(amdsmi_get_gpu_subsystem_id)
WWR_LINK_CHECK(amdsmi_get_gpu_subsystem_name)

// PCIe Queries
WWR_LINK_CHECK(amdsmi_get_gpu_pci_bandwidth)
WWR_LINK_CHECK(amdsmi_get_gpu_bdf_id)
WWR_LINK_CHECK(amdsmi_get_gpu_topo_numa_affinity)
WWR_LINK_CHECK(amdsmi_get_gpu_pci_throughput)
WWR_LINK_CHECK(amdsmi_get_gpu_pci_replay_counter)

// PCIe Control
WWR_LINK_CHECK(amdsmi_set_gpu_pci_bandwidth)

// Power Queries
WWR_LINK_CHECK(amdsmi_get_energy_count)

// Power Control
WWR_LINK_CHECK(amdsmi_set_power_cap)
WWR_LINK_CHECK(amdsmi_set_gpu_power_profile)
WWR_LINK_CHECK(amdsmi_get_supported_power_cap)
WWR_LINK_CHECK(amdsmi_get_cpu_socket_power)
WWR_LINK_CHECK(amdsmi_get_cpu_socket_power_cap)
WWR_LINK_CHECK(amdsmi_get_cpu_socket_power_cap_max)
WWR_LINK_CHECK(amdsmi_get_cpu_pwr_svi_telemetry_all_rails)
WWR_LINK_CHECK(amdsmi_set_cpu_socket_power_cap)
WWR_LINK_CHECK(amdsmi_set_cpu_pwr_efficiency_mode)

// Memory Queries
WWR_LINK_CHECK(amdsmi_get_gpu_memory_total)
WWR_LINK_CHECK(amdsmi_get_gpu_memory_usage)
WWR_LINK_CHECK(amdsmi_get_gpu_bad_page_info)
WWR_LINK_CHECK(amdsmi_get_gpu_bad_page_threshold)
WWR_LINK_CHECK(amdsmi_gpu_validate_ras_eeprom)
WWR_LINK_CHECK(amdsmi_get_gpu_ras_block_features_enabled)
WWR_LINK_CHECK(amdsmi_get_gpu_memory_reserved_pages)

// Physical State Queries
WWR_LINK_CHECK(amdsmi_get_gpu_fan_rpms)
WWR_LINK_CHECK(amdsmi_get_gpu_fan_speed)
WWR_LINK_CHECK(amdsmi_get_gpu_fan_speed_max)
WWR_LINK_CHECK(amdsmi_get_gpu_cache_info)
WWR_LINK_CHECK(amdsmi_get_gpu_volt_metric)

// Physical State Control
WWR_LINK_CHECK(amdsmi_reset_gpu_fan)
WWR_LINK_CHECK(amdsmi_set_gpu_fan_speed)

// Clock, Power and Performance Queries
WWR_LINK_CHECK(amdsmi_get_gpu_busy_percent)
WWR_LINK_CHECK(amdsmi_get_utilization_count)
WWR_LINK_CHECK(amdsmi_get_gpu_perf_level)
WWR_LINK_CHECK(amdsmi_set_gpu_perf_determinism_mode)
WWR_LINK_CHECK(amdsmi_get_gpu_overdrive_level)
WWR_LINK_CHECK(amdsmi_get_gpu_mem_overdrive_level)
WWR_LINK_CHECK(amdsmi_get_clk_freq)
WWR_LINK_CHECK(amdsmi_reset_gpu)
WWR_LINK_CHECK(amdsmi_get_gpu_od_volt_info)
WWR_LINK_CHECK(amdsmi_get_gpu_metrics_header_info)
WWR_LINK_CHECK(amdsmi_get_gpu_metrics_info)
WWR_LINK_CHECK(amdsmi_get_gpu_partition_metrics_info)
WWR_LINK_CHECK(amdsmi_get_gpu_pm_metrics_info)
WWR_LINK_CHECK(amdsmi_get_gpu_reg_table_info)
WWR_LINK_CHECK(amdsmi_set_gpu_clk_range)
WWR_LINK_CHECK(amdsmi_set_gpu_clk_limit)
WWR_LINK_CHECK(amdsmi_set_gpu_od_clk_info)
WWR_LINK_CHECK(amdsmi_set_gpu_od_volt_info)
WWR_LINK_CHECK(amdsmi_get_gpu_od_volt_curve_regions)
WWR_LINK_CHECK(amdsmi_get_gpu_power_profile_presets)

// Clock, Power and Performance Control
WWR_LINK_CHECK(amdsmi_set_gpu_perf_level)
WWR_LINK_CHECK(amdsmi_set_gpu_overdrive_level)
WWR_LINK_CHECK(amdsmi_set_clk_freq)
WWR_LINK_CHECK(amdsmi_get_soc_pstate)
WWR_LINK_CHECK(amdsmi_set_soc_pstate)
WWR_LINK_CHECK(amdsmi_get_xgmi_plpd)
WWR_LINK_CHECK(amdsmi_set_xgmi_plpd)
WWR_LINK_CHECK(amdsmi_get_gpu_process_isolation)
WWR_LINK_CHECK(amdsmi_set_gpu_process_isolation)
WWR_LINK_CHECK(amdsmi_clean_gpu_local_data)

// Version Queries
WWR_LINK_CHECK(amdsmi_get_lib_version)

// ECC Information
WWR_LINK_CHECK(amdsmi_get_gpu_ecc_count)
WWR_LINK_CHECK(amdsmi_get_gpu_ecc_enabled)
WWR_LINK_CHECK(amdsmi_get_gpu_total_ecc_count)
WWR_LINK_CHECK(amdsmi_get_gpu_cper_entries)

// RAS information
WWR_LINK_CHECK(amdsmi_get_afids_from_cper)
WWR_LINK_CHECK(amdsmi_get_gpu_ras_feature_info)

// Error Queries
WWR_LINK_CHECK(amdsmi_get_gpu_ecc_status)
WWR_LINK_CHECK(amdsmi_status_code_to_string)

// Performance Counter Functions
WWR_LINK_CHECK(amdsmi_gpu_counter_group_supported)
WWR_LINK_CHECK(amdsmi_gpu_create_counter)
WWR_LINK_CHECK(amdsmi_gpu_destroy_counter)
WWR_LINK_CHECK(amdsmi_gpu_control_counter)
WWR_LINK_CHECK(amdsmi_gpu_read_counter)
WWR_LINK_CHECK(amdsmi_get_gpu_available_counters)

// System Information Functions
WWR_LINK_CHECK(amdsmi_get_gpu_compute_process_info)
WWR_LINK_CHECK(amdsmi_get_gpu_compute_process_info_by_pid)
WWR_LINK_CHECK(amdsmi_get_gpu_compute_process_gpus)

// XGMI Functions
WWR_LINK_CHECK(amdsmi_gpu_xgmi_error_status)
WWR_LINK_CHECK(amdsmi_reset_gpu_xgmi_error)
WWR_LINK_CHECK(amdsmi_get_xgmi_info)
WWR_LINK_CHECK(amdsmi_get_gpu_xgmi_link_status)

// Hardware Topology Functions
WWR_LINK_CHECK(amdsmi_get_link_metrics)
WWR_LINK_CHECK(amdsmi_topo_get_numa_node_number)
WWR_LINK_CHECK(amdsmi_topo_get_link_weight)
WWR_LINK_CHECK(amdsmi_get_minmax_bandwidth_between_processors)
WWR_LINK_CHECK(amdsmi_topo_get_link_type)
WWR_LINK_CHECK(amdsmi_get_link_topology_nearest)
WWR_LINK_CHECK(amdsmi_is_P2P_accessible)
WWR_LINK_CHECK(amdsmi_topo_get_p2p_status)

// Compute Partition Functions
WWR_LINK_CHECK(amdsmi_get_gpu_compute_partition)
WWR_LINK_CHECK(amdsmi_set_gpu_compute_partition)

// Memory Partition Functions
WWR_LINK_CHECK(amdsmi_get_gpu_memory_partition)
WWR_LINK_CHECK(amdsmi_set_gpu_memory_partition)
WWR_LINK_CHECK(amdsmi_get_gpu_memory_partition_config)
WWR_LINK_CHECK(amdsmi_set_gpu_memory_partition_mode)

// Accelerator Partition Profile Functions
WWR_LINK_CHECK(amdsmi_get_gpu_accelerator_partition_profile_config)
WWR_LINK_CHECK(amdsmi_get_gpu_accelerator_partition_profile)
WWR_LINK_CHECK(amdsmi_set_gpu_accelerator_partition_profile)

// Event Notification Functions
WWR_LINK_CHECK(amdsmi_init_gpu_event_notification)
WWR_LINK_CHECK(amdsmi_set_gpu_event_notification_mask)
WWR_LINK_CHECK(amdsmi_get_gpu_event_notification)
WWR_LINK_CHECK(amdsmi_stop_gpu_event_notification)

// Software Version Information
WWR_LINK_CHECK(amdsmi_get_gpu_driver_info)

// ASIC & Board Static Information
WWR_LINK_CHECK(amdsmi_get_gpu_asic_info)
WWR_LINK_CHECK(amdsmi_get_gpu_kfd_info)
WWR_LINK_CHECK(amdsmi_get_gpu_vram_info)
WWR_LINK_CHECK(amdsmi_get_gpu_board_info)
WWR_LINK_CHECK(amdsmi_get_power_cap_info)
WWR_LINK_CHECK(amdsmi_get_pcie_info)
WWR_LINK_CHECK(amdsmi_get_gpu_xcd_counter)
WWR_LINK_CHECK(amdsmi_get_npm_info)

// Firmware & VBIOS queries
WWR_LINK_CHECK(amdsmi_get_fw_info)
WWR_LINK_CHECK(amdsmi_get_gpu_vbios_info)

// GPU Monitoring
WWR_LINK_CHECK(amdsmi_get_temp_metric)
WWR_LINK_CHECK(amdsmi_get_gpu_activity)
WWR_LINK_CHECK(amdsmi_get_power_info)
WWR_LINK_CHECK(amdsmi_is_gpu_power_management_enabled)
WWR_LINK_CHECK(amdsmi_get_clock_info)
WWR_LINK_CHECK(amdsmi_get_gpu_vram_usage)
WWR_LINK_CHECK(amdsmi_get_violation_status)

// Process information
WWR_LINK_CHECK(amdsmi_get_gpu_process_list)

// Driver control mechanisms
WWR_LINK_CHECK(amdsmi_gpu_driver_reload)

// Peak Tops Limiter
WWR_LINK_CHECK(amdsmi_get_gpu_ptl_state)
WWR_LINK_CHECK(amdsmi_set_gpu_ptl_state)
WWR_LINK_CHECK(amdsmi_get_gpu_ptl_formats)
WWR_LINK_CHECK(amdsmi_set_gpu_ptl_formats)

} // namespace wwr::hip::test
