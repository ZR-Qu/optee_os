/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef __PTA_TNB_H
#define __PTA_TNB_H

#include <stdint.h>

#define PTA_TNB_UUID { 0x0ea88044, 0x846a, 0x47a2, \
	{ 0x87, 0xff, 0xc1, 0xb7, 0x20, 0xf0, 0x69, 0x4e } }

#define PTA_TNB_ABI_VERSION 40
#define PTA_TNB_BUILD_ID 0x49505431u

#define PTA_TNB_CMD_GET_CAPS 0
#define PTA_TNB_CMD_GEMM 1
#define PTA_TNB_CMD_ACTIVATE 2
#define PTA_TNB_CMD_RUN 3
#define PTA_TNB_CMD_CLOSE 4
#define PTA_TNB_CMD_GET_STATUS 5
#define PTA_TNB_CMD_LOCK 6
#define PTA_TNB_CMD_READ_HW 7
/* Fixed-command repeat diagnostic only. */
#define PTA_TNB_CMD_GEMM_REUSE 8
#define PTA_TNB_CMD_CLOSE_KEEP 9
/* Restore hardware while retaining this diagnostic's prepared work. */
#define PTA_TNB_CMD_SUSPEND 10
#define PTA_TNB_CMD_SUSPEND_MASKED 11
#define PTA_TNB_CMD_SUSPEND_SECURE 12
#define PTA_TNB_CMD_CYCLE_IRQ 13
#define PTA_TNB_CMD_CYCLE_PAGING 14
#define PTA_TNB_CMD_CYCLE_ZAP 15
#define PTA_TNB_CMD_CYCLE_ZAP_ON 16
#define PTA_TNB_CMD_CYCLE_ZAP0 17
#define PTA_TNB_CMD_CYCLE_ZAP1 18
#define PTA_TNB_CMD_CYCLE_ZAP_CMD 19
#define PTA_TNB_CMD_PREPARE_BATCH 20
#define PTA_TNB_CMD_UPDATE 21
#define PTA_TNB_CMD_SECURE_RAM_TEST_BEGIN 16
#define PTA_TNB_CMD_SECURE_RAM_TEST_CHECK 17
#define PTA_TNB_CMD_SECURE_RAM_TEST_END 18

#define PTA_TNB_DEBUG_WORDS 76
#define PTA_TNB_MAX_CORES 3
#define PTA_TNB_HW_NPU_WORDS 10
#define PTA_TNB_HW_IOMMU_WORDS 6
#define PTA_TNB_HW_IOMMUS 4
#define PTA_TNB_HW_STAGING_PAGES 8
#define PTA_TNB_WORK_COMMAND 1
#define PTA_TNB_WORK_DIRECTORY 2
#define PTA_TNB_WORK_PTE 3

#define PTA_TNB_GEMM_W8A8_SINGLE 1

struct pta_tnb_core_snapshot {
	uint32_t int_status;
	uint32_t int_raw_status;
	uint32_t task_status;
	uint32_t data_addr;
	uint32_t data_amount;
	uint32_t task_control;
	uint32_t cna_pointer;
	uint32_t core_pointer;
};

struct pta_tnb_trace {
	uint32_t valid;
	uint32_t sequence;
	uint32_t mode;
	uint32_t core_count;
	uint32_t complete_mask;
	struct pta_tnb_core_snapshot core[PTA_TNB_MAX_CORES];
};

enum pta_tnb_debug_index {
	PTA_TNB_DEBUG_PTE_CMD = 0,
	PTA_TNB_DEBUG_PTE_A = 1,
	PTA_TNB_DEBUG_PTE_B = 2,
	PTA_TNB_DEBUG_PTE_C = 3,
	PTA_TNB_DEBUG_DTE_WRITTEN = 4,
	PTA_TNB_DEBUG_DTE_READBACK0 = 5,
	PTA_TNB_DEBUG_DTE_READBACK1 = 6,
	PTA_TNB_DEBUG_DTE_READBACK2 = 7,
	PTA_TNB_DEBUG_DTE_READBACK3 = 8,
	PTA_TNB_DEBUG_IOMMU_STATUS0 = 9,
	PTA_TNB_DEBUG_IOMMU_STATUS1 = 10,
	PTA_TNB_DEBUG_IOMMU_STATUS2 = 11,
	PTA_TNB_DEBUG_IOMMU_STATUS3 = 12,
	PTA_TNB_DEBUG_NPU_MASK = 13,
	PTA_TNB_DEBUG_PRE_STATUS = 14,
	PTA_TNB_DEBUG_PRE_RAW = 15,
	PTA_TNB_DEBUG_PRE_TASK = 16,
	PTA_TNB_DEBUG_PRE_DATA = 17,
	PTA_TNB_DEBUG_PRE_AMOUNT = 18,
	PTA_TNB_DEBUG_PRE_CONTROL = 19,
	PTA_TNB_DEBUG_POST_STATUS = 20,
	PTA_TNB_DEBUG_POST_RAW = 21,
	PTA_TNB_DEBUG_POST_TASK = 22,
	PTA_TNB_DEBUG_POST_DATA = 23,
	PTA_TNB_DEBUG_POST_AMOUNT = 24,
	PTA_TNB_DEBUG_POST_CONTROL = 25,
	PTA_TNB_DEBUG_PTE_FLAGS = 26,
	PTA_TNB_DEBUG_CNA_POINTER = 27,
	PTA_TNB_DEBUG_CORE_POINTER = 28,
	PTA_TNB_DEBUG_BUSSGRF = 29,
	PTA_TNB_DEBUG_DDR_MST11 = 30,
	PTA_TNB_DEBUG_DDR_MST20 = 31,
	PTA_TNB_DEBUG_DDR_MST2 = 32,
	PTA_TNB_DEBUG_DDR_MST32 = 33,
	PTA_TNB_DEBUG_FLAGS = 34,
	PTA_TNB_DEBUG_CMD_ADDR = 35,
	PTA_TNB_DEBUG_A_ADDR = 36,
	PTA_TNB_DEBUG_B_ADDR = 37,
	PTA_TNB_DEBUG_C_ADDR = 38,
	PTA_TNB_DEBUG_DT_ADDR = 39,
	PTA_TNB_DEBUG_IOMMU_FAULT_ADDR0 = 40,
	PTA_TNB_DEBUG_IOMMU_RAW0 = 44,
	PTA_TNB_DEBUG_IOMMU_IRQ0 = 48,
	PTA_TNB_DEBUG_OUTPUT0 = 52,
	PTA_TNB_DEBUG_PATCH_A = 56,
	PTA_TNB_DEBUG_PATCH_B = 57,
	PTA_TNB_DEBUG_PATCH_C = 58,
	PTA_TNB_DEBUG_AFTER_CONTROL = 59,
	PTA_TNB_DEBUG_AFTER_CONTROL_TASK = 60,
	PTA_TNB_DEBUG_AFTER_OP_ENABLE = 61,
	PTA_TNB_DEBUG_AFTER_OP_ENABLE_TASK = 62,
	PTA_TNB_DEBUG_OP_ENABLE_READBACK = 63,
	PTA_TNB_DEBUG_PRE_CMD_INPUT = 64,
	PTA_TNB_DEBUG_PRE_CMD_WEIGHT = 65,
	PTA_TNB_DEBUG_PRE_CMD_CNA = 66,
	PTA_TNB_DEBUG_PRE_CMD_CORE = 67,
	PTA_TNB_DEBUG_PRE_CMD_DPU = 68,
	PTA_TNB_DEBUG_PRE_CMD_OUTPUT = 69,
	PTA_TNB_DEBUG_POST_CMD_INPUT = 70,
	PTA_TNB_DEBUG_POST_CMD_WEIGHT = 71,
	PTA_TNB_DEBUG_POST_CMD_CNA = 72,
	PTA_TNB_DEBUG_POST_CMD_CORE = 73,
	PTA_TNB_DEBUG_POST_CMD_DPU = 74,
	PTA_TNB_DEBUG_POST_CMD_OUTPUT = 75,
	PTA_TNB_DEBUG_FAILURE_INT_STATUS0 = 64,
	PTA_TNB_DEBUG_FAILURE_INT_RAW0 = 67,
	PTA_TNB_DEBUG_FAILURE_TASK_STATUS0 = 70,
	PTA_TNB_DEBUG_FAILURE_DATA_ADDR0 = 73,
};

#define PTA_TNB_CAP_CALLER_BOUND (1u << 0)
#define PTA_TNB_CAP_IOMMU_V2 (1u << 1)
#define PTA_TNB_CAP_SECURE_PTE (1u << 2)
#define PTA_TNB_CAP_FULL_SECURITY (1u << 3)
#define PTA_TNB_CAP_PROFILE (1u << 4)
#define PTA_TNB_CAP_GEMM (1u << 5)
#define PTA_TNB_CAP_SECURE_RAM_TEST (1u << 6)

#define PTA_TNB_SECURE_RAM_TEST_VERSION 1
#define PTA_TNB_SECURE_RAM_TEST_PAGE_SIZE 4096
#define PTA_TNB_SECURE_RAM_TEST_PATTERN_SIZE 32

/* This ABI is compiled into the test-only Secure RAM validation path. */
struct pta_tnb_secure_ram_test_info {
	uint32_t version;
	uint32_t page_size;
	uint64_t physical_address;
	uint64_t token;
	uint8_t pattern[PTA_TNB_SECURE_RAM_TEST_PATTERN_SIZE];
};

struct pta_tnb_secure_ram_test_check {
	uint32_t version;
	uint32_t active;
	uint32_t unchanged;
	uint32_t changed_bytes;
};

#define PTA_TNB_USER_TA_UUID { 0x96d83ee6, 0x5ca4, 0x427e, \
	{ 0xb3, 0xea, 0x90, 0xa7, 0x85, 0xd2, 0xe8, 0xa3 } }

#define PTA_TNB_LAYER_TA_UUID { 0x8aaaf200, 0x2450, 0x4d0e, \
	{ 0x8f, 0x0b, 0x2b, 0x55, 0x56, 0x6d, 0x11, 0x01 } }


#define PTA_TNB_STATE_READY 0
#define PTA_TNB_STATE_PREPARED 1
#define PTA_TNB_STATE_ACTIVE 2
#define PTA_TNB_STATE_RUNNING 3
#define PTA_TNB_STATE_LOCKED 4

#define PTA_TNB_STAGE_READY 0
#define PTA_TNB_STAGE_PREPARED 1
#define PTA_TNB_STAGE_IRQ 2
#define PTA_TNB_STAGE_SECURE 3
#define PTA_TNB_STAGE_IOMMU 4
#define PTA_TNB_STAGE_SUBMITTED 5
#define PTA_TNB_STAGE_COMPLETED 6
#define PTA_TNB_STAGE_RESTORED 7

#define PTA_TNB_FAULT_NONE 0
#define PTA_TNB_FAULT_CALLER 1
#define PTA_TNB_FAULT_MODE 2
#define PTA_TNB_FAULT_IRQ 3
#define PTA_TNB_FAULT_SECURE 4
#define PTA_TNB_FAULT_IOMMU 5
#define PTA_TNB_FAULT_NPU 6
#define PTA_TNB_FAULT_TIMEOUT 7
#define PTA_TNB_FAULT_RESTORE 8
#define PTA_TNB_FAULT_VERIFY 9

struct pta_tnb_caps {
	uint32_t abi_version;
	uint32_t features;
	uint32_t max_modes;
	uint32_t max_tensors;
	uint32_t mode;
	uint32_t n_min;
	uint32_t n_max;
	uint32_t n_align;
	uint32_t b_max;
	uint32_t reserved[1];
};

struct pta_tnb_gemm {
	uint32_t abi_version;
	uint32_t mode;
	uint32_t m;
	uint32_t k;
	uint32_t n;
};

#define PTA_TNB_MAX_JOBS 16
#define PTA_TNB_MAX_B_INPUTS 3

struct pta_tnb_job {
	/* Maximum execution rows at PREPARE; UPDATE supplies the current m. */
	struct pta_tnb_gemm gemm;
	uint32_t b_index;
	uint32_t b_offset;
};

struct pta_tnb_batch {
	/* BEGIN failure diagnostic: call setup or exact PTA allocation. */
	uint32_t oom_stage;
	uint32_t oom_size;
	uint32_t job_count;
	uint32_t b_count;
	struct pta_tnb_job job[PTA_TNB_MAX_JOBS];
};

/* Read-only register/work check for the fixed-GEMM repeat diagnostic. */
struct pta_tnb_hw_state {
	uint32_t npu[PTA_TNB_HW_NPU_WORDS];
	uint32_t iommu[PTA_TNB_HW_IOMMUS][PTA_TNB_HW_IOMMU_WORDS];
	uint32_t memory_area;
	uint32_t memory_index;
	uint32_t memory_expected;
	uint32_t memory_actual;
	uint32_t a_pte[PTA_TNB_HW_STAGING_PAGES];
	uint32_t c_pte[PTA_TNB_HW_STAGING_PAGES];
};

#define PTA_TNB_PROFILE_VERSION 5

struct pta_tnb_profile {
	uint32_t version;
	uint32_t counter_frequency;
	uint64_t prepare_calls;
	uint64_t prepare_ticks;
	uint64_t activate_calls;
	uint64_t activate_ticks;
	uint64_t run_calls;
	uint64_t run_ticks;
	uint64_t poll_calls;
	uint64_t poll_ticks;
	uint64_t close_calls;
	uint64_t close_ticks;
};

struct pta_tnb_status {
	uint32_t abi_version;
	uint32_t state;
	uint32_t stage;
	uint32_t fault;
	uint32_t npu_status;
	uint32_t npu_raw_status;
	uint32_t iommu_status[4];
	uint32_t reserved[4];
	uint32_t debug[PTA_TNB_DEBUG_WORDS];
	struct pta_tnb_trace last_success;
	struct pta_tnb_trace failure;
	struct pta_tnb_profile profile;
};

#endif
