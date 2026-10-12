/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef __PTA_TNB_H
#define __PTA_TNB_H

#include <stdint.h>

#define PTA_TNB_UUID { 0x0ea88044, 0x846a, 0x47a2, \
	{ 0x87, 0xff, 0xc1, 0xb7, 0x20, 0xf0, 0x69, 0x4e } }

#define PTA_TNB_ABI_VERSION 48
#define PTA_TNB_BUILD_ID 0x4e58526bu

#define PTA_TNB_CMD_GET_CAPS 0
#define PTA_TNB_CMD_PREPARE 1
#define PTA_TNB_CMD_ACTIVATE 2
#define PTA_TNB_CMD_RUN 3
#define PTA_TNB_CMD_CLOSE 4
#define PTA_TNB_CMD_GET_STATUS 5
#define PTA_TNB_CMD_LOCK 6
#define PTA_TNB_CMD_REPREPARE 7
#define PTA_TNB_CMD_PREPARE_MULTI 8
#define PTA_TNB_CMD_REPREPARE_MULTI 9
#define PTA_TNB_CMD_PREPARE_FFN_SLOT 10
#define PTA_TNB_CMD_PREPARE_MULTI_SLOT 11
#define PTA_TNB_CMD_REPREPARE_MULTI_SLOT 12
#define PTA_TNB_CMD_PREPARE_FFN_SLOT_SET 13
#define PTA_TNB_CMD_EXEC_MULTI_SLOT_CHAIN 14
#define PTA_TNB_CMD_SELECT_FFN_SLOT_SET 15
#define PTA_TNB_CMD_SECURE_RAM_TEST_BEGIN 16
#define PTA_TNB_CMD_SECURE_RAM_TEST_CHECK 17
#define PTA_TNB_CMD_SECURE_RAM_TEST_END 18
#define PTA_TNB_CMD_PREPARE_GEMM 19
#define PTA_TNB_CMD_PREPARE_GEMM_PAIR 20
#define PTA_TNB_CMD_SELECT_GEMM 21
#define PTA_TNB_CMD_PREPARE_GEMM_BATCH 22

/* W8A8, native B packing, INT32 C, RK3588 single-core command family.
 * No raw command, relocation index, interrupt mask, or security flag input.
 */
#define PTA_TNB_GEMM_MODE_W8A8 1u
#define PTA_TNB_GEMM_M_MAX 16u
#define PTA_TNB_GEMM_A_MAX 65536u
#define PTA_TNB_GEMM_K_MIN 2048u
#define PTA_TNB_GEMM_K_MAX 6208u
#define PTA_TNB_GEMM_B_ALIGN 4096u
#define PTA_TNB_GEMM_B_MAX (PTA_TNB_GEMM_K_MAX * PTA_TNB_GEMM_N_MAX)
#define PTA_TNB_GEMM_N_MIN 672u
#define PTA_TNB_GEMM_N_MAX 2048u
#define PTA_TNB_GEMM_N_ALIGN 32u
#define PTA_TNB_TEMPLATE_GENERIC 0x47454d4du
struct pta_tnb_gemm {
	uint32_t abi_version;
	uint32_t mode;
	uint32_t m;
	uint32_t k;
	uint32_t n;
	uint32_t reserved[3];
};

/* Variable compute plan; each EXEC_GROUP remains bounded to 32 jobs.
 * All B ranges stay mapped until CLOSE. */
#define PTA_TNB_GEMM_GROUP_MAX_JOBS 32u
#define PTA_TNB_GEMM_MAX_BUFFERS 3u
struct pta_tnb_gemm_job {
	struct pta_tnb_gemm gemm;
	uint32_t b_index;
	uint32_t b_offset;
};
struct pta_tnb_gemm_batch {
	uint32_t count;
	uint32_t reserved;
	struct pta_tnb_gemm_job job[];
};

#define PTA_TNB_PREPARE_FLAG_SECURE_PTE (1u << 0)
#define PTA_TNB_PREPARE_FLAG_FULL_SECURITY (1u << 1)
#define PTA_TNB_PREPARE_FLAG_DIRECT_PHYS (1u << 2)
#define PTA_TNB_PREPARE_FLAG_TRACE_STEPS (1u << 3)
#define PTA_TNB_PREPARE_FLAG_UNCACHED_COMMAND (1u << 4)
#define PTA_TNB_PREPARE_FLAG_NONSECURE_WEIGHT (1u << 5)
#define PTA_TNB_PREPARE_FLAGS_MASK \
	(PTA_TNB_PREPARE_FLAG_SECURE_PTE | \
	 PTA_TNB_PREPARE_FLAG_FULL_SECURITY | \
	 PTA_TNB_PREPARE_FLAG_DIRECT_PHYS | \
	 PTA_TNB_PREPARE_FLAG_TRACE_STEPS | \
	 PTA_TNB_PREPARE_FLAG_UNCACHED_COMMAND | \
	 PTA_TNB_PREPARE_FLAG_NONSECURE_WEIGHT)
#define PTA_TNB_DEBUG_WORDS 388
#define PTA_TNB_MAX_CORES 3

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
	uint32_t template_id;
	uint32_t prepare_flags;
	uint32_t core_count;
	uint32_t complete_mask;
	struct pta_tnb_core_snapshot core[PTA_TNB_MAX_CORES];
};

enum pta_tnb_debug_index {
	PTA_TNB_DEBUG_ENABLE_PRE = 129,
	PTA_TNB_DEBUG_POLL_HISTORY = 104,
	PTA_TNB_DEBUG_POLL_COUNT = 128,
	PTA_TNB_DEBUG_BANK_PRE = 88,
	PTA_TNB_DEBUG_BANK_PREPARED = 91,
	PTA_TNB_DEBUG_BANK_LAUNCH = 94,
	PTA_TNB_DEBUG_BANK_COMPLETION = 97,
	PTA_TNB_DEBUG_BANK_FAILURE = 100,
	PTA_TNB_DEBUG_BANK_VALID = 103,
	PTA_TNB_DEBUG_PTE_CMD = 0,
	PTA_TNB_DEBUG_PTE_INPUT = 1,
	PTA_TNB_DEBUG_PTE_WEIGHT = 2,
	PTA_TNB_DEBUG_PTE_OUTPUT = 3,
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
	PTA_TNB_DEBUG_INPUT_ADDR = 36,
	PTA_TNB_DEBUG_WEIGHT_ADDR = 37,
	PTA_TNB_DEBUG_OUTPUT_ADDR = 38,
	PTA_TNB_DEBUG_DT_ADDR = 39,
	PTA_TNB_DEBUG_IOMMU_FAULT_ADDR0 = 40,
	PTA_TNB_DEBUG_IOMMU_RAW0 = 44,
	PTA_TNB_DEBUG_IOMMU_IRQ0 = 48,
	PTA_TNB_DEBUG_OUTPUT0 = 52,
	PTA_TNB_DEBUG_PATCH_INPUT = 56,
	PTA_TNB_DEBUG_PATCH_WEIGHT = 57,
	PTA_TNB_DEBUG_PATCH_OUTPUT = 58,
	PTA_TNB_DEBUG_AFTER_CONTROL = 59,
	PTA_TNB_DEBUG_AFTER_CONTROL_TASK = 60,
	PTA_TNB_DEBUG_AFTER_OP_ENABLE = 61,
	PTA_TNB_DEBUG_AFTER_OP_ENABLE_TASK = 62,
	PTA_TNB_DEBUG_OP_ENABLE_READBACK = 63,
	PTA_TNB_DEBUG_PRE_CMD_INPUT = 76,
	PTA_TNB_DEBUG_PRE_CMD_WEIGHT = 77,
	PTA_TNB_DEBUG_PRE_CMD_CNA = 78,
	PTA_TNB_DEBUG_PRE_CMD_CORE = 79,
	PTA_TNB_DEBUG_PRE_CMD_DPU = 80,
	PTA_TNB_DEBUG_PRE_CMD_OUTPUT = 81,
	PTA_TNB_DEBUG_POST_CMD_INPUT = 82,
	PTA_TNB_DEBUG_POST_CMD_WEIGHT = 83,
	PTA_TNB_DEBUG_POST_CMD_CNA = 84,
	PTA_TNB_DEBUG_POST_CMD_CORE = 85,
	PTA_TNB_DEBUG_POST_CMD_DPU = 86,
	PTA_TNB_DEBUG_POST_CMD_OUTPUT = 87,
	PTA_TNB_DEBUG_FAILURE_INT_STATUS0 = 64,
	PTA_TNB_DEBUG_FAILURE_INT_RAW0 = 67,
	PTA_TNB_DEBUG_FAILURE_TASK_STATUS0 = 70,
	PTA_TNB_DEBUG_FAILURE_DATA_ADDR0 = 73,
};

#define PTA_TNB_CAP_CALLER_BOUND (1u << 0)
#define PTA_TNB_CAP_IOMMU_V2 (1u << 1)
#define PTA_TNB_CAP_SECURE_PTE (1u << 2)
#define PTA_TNB_CAP_FULL_SECURITY (1u << 3)
#define PTA_TNB_CAP_DIRECT_PHYS (1u << 4)
#define PTA_TNB_CAP_UNCACHED_COMMAND (1u << 5)
#define PTA_TNB_CAP_FIXED_TRAILER (1u << 6)
#define PTA_TNB_CAP_NONSECURE_WEIGHT (1u << 7)
#define PTA_TNB_CAP_MULTI_PAGE_IO (1u << 8)
#define PTA_TNB_CAP_REPREPARE (1u << 9)
#define PTA_TNB_CAP_MULTI_CORE (1u << 10)
#define PTA_TNB_CAP_STALLED_WEIGHT_REMAP (1u << 11)
#define PTA_TNB_CAP_FIXED_WEIGHT_PTES (1u << 12)
#define PTA_TNB_CAP_PROFILE (1u << 13)
#define PTA_TNB_CAP_FFN_WEIGHT_SLOTS (1u << 14)
#define PTA_TNB_CAP_MULTI_SLOT_CHAIN (1u << 15)
#define PTA_TNB_CAP_CACHED_FFN_SLOT_SET (1u << 16)
#define PTA_TNB_CAP_FUSED_SELECT_CHAIN (1u << 17)
#define PTA_TNB_CAP_SECURE_RAM_TEST (1u << 18)
#define PTA_TNB_CAP_GENERIC_GEMM (1u << 19)

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

#define PTA_TNB_MAX_COMMANDS 128
#define PTA_TNB_COMMAND_TRAILER_COUNT 4
#define PTA_TNB_MAX_INPUT_SIZE 8192
#define PTA_TNB_MAX_BATCH_INPUT_SIZE PTA_TNB_GEMM_A_MAX
#define PTA_TNB_MAX_SECURE_WEIGHT_SIZE 4096
#define PTA_TNB_MAX_WEIGHT_SIZE 12582912
#define PTA_TNB_MAX_OUTPUT_SIZE 24576
#define PTA_TNB_MAX_BATCH_OUTPUT_SIZE (PTA_TNB_MAX_CORES * PTA_TNB_GEMM_M_MAX * PTA_TNB_GEMM_N_MAX * sizeof(int32_t))
#define PTA_TNB_FFN_WEIGHT_SLOT_COUNT 3
#define PTA_TNB_FFN_WEIGHT_SLOT_GATE 0
#define PTA_TNB_FFN_WEIGHT_SLOT_UP 1
#define PTA_TNB_FFN_WEIGHT_SLOT_DOWN 2
#define PTA_TNB_CHAIN_MAX_JOBS 2
#define PTA_TNB_CHAIN_MAX_ROWS 10
#define PTA_TNB_MULTI_COMMAND_WORDS \
	(PTA_TNB_MAX_CORES * \
	 (PTA_TNB_MAX_COMMANDS + PTA_TNB_COMMAND_TRAILER_COUNT))

#define PTA_TNB_CHAIN_FLAG_BEGIN (1u << 0)
#define PTA_TNB_CHAIN_FLAG_END (1u << 1)
#define PTA_TNB_CHAIN_FLAG_SELECT_FFN_SLOT_SET (1u << 2)
#define PTA_TNB_CHAIN_FLAGS_MASK \
	(PTA_TNB_CHAIN_FLAG_BEGIN | PTA_TNB_CHAIN_FLAG_END | \
	 PTA_TNB_CHAIN_FLAG_SELECT_FFN_SLOT_SET)

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
#define PTA_TNB_FAULT_TEMPLATE 2
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
	uint32_t max_templates;
	uint32_t max_tensors;
	uint32_t reserved[4];
};

struct pta_tnb_prepare {
	uint32_t abi_version;
	uint32_t template_id;
	uint32_t command_count;
	uint32_t interrupt_mask;
	uint32_t input_size;
	uint32_t weight_size;
	uint32_t output_size;
	uint32_t input_patch;
	uint32_t weight_patch;
	uint32_t output_patch;
	uint32_t reserved[6];
};

struct pta_tnb_segment {
	uint32_t command_count;
	uint32_t interrupt_mask;
	uint32_t weight_offset;
	uint32_t weight_size;
	uint32_t output_offset;
	uint32_t output_size;
	uint32_t input_patch;
	uint32_t weight_patch;
	uint32_t output_patch;
	uint32_t reserved;
};

struct pta_tnb_prepare_multi {
	uint32_t abi_version;
	uint32_t template_id;
	uint32_t core_count;
	uint32_t input_size;
	uint32_t weight_size;
	uint32_t output_size;
	uint32_t flags;
	uint32_t weight_cookie;
	struct pta_tnb_segment segment[PTA_TNB_MAX_CORES];
};

struct pta_tnb_ffn_slot {
	uint32_t abi_version;
	uint32_t flags;
	uint32_t weight_size;
	uint32_t weight_cookie;
	uint32_t weight_slot;
	uint32_t reserved[3];
};

struct pta_tnb_prepare_multi_slot {
	struct pta_tnb_prepare_multi prepare;
	uint32_t weight_slot;
	uint32_t reserved[3];
};

struct pta_tnb_ffn_slot_set {
	uint32_t abi_version;
	uint32_t flags;
	uint32_t weight_size;
	uint32_t reserved;
	uint32_t weight_cookie[PTA_TNB_FFN_WEIGHT_SLOT_COUNT];
	uint32_t reserved2[5];
};

struct pta_tnb_chain_job {
	struct pta_tnb_prepare_multi_slot request;
	uint32_t command_words;
	uint32_t reserved[3];
	uint64_t commands[PTA_TNB_MULTI_COMMAND_WORDS];
};

#define PTA_TNB_PROFILE_VERSION 4

struct pta_tnb_profile {
	uint32_t version;
	uint32_t counter_frequency;
	uint64_t prepare_calls;
	uint64_t prepare_ticks;
	uint64_t reprepare_calls;
	uint64_t reprepare_ticks;
	uint64_t activate_calls;
	uint64_t activate_ticks;
	uint64_t run_calls;
	uint64_t run_ticks;
	uint64_t poll_calls;
	uint64_t poll_ticks;
	uint64_t close_calls;
	uint64_t close_ticks;
	uint64_t weight_remap_calls;
	uint64_t weight_remap_ticks;
	uint64_t slot_prepare_calls;
	uint64_t slot_prepare_ticks;
	uint64_t slot_prepare_cold_calls;
	uint64_t slot_prepare_cached_calls;
	uint64_t slot_repatch_calls;
	uint64_t slot_repatch_ticks;
	uint64_t slot_set_calls;
	uint64_t slot_set_ticks;
	uint64_t slot_select_calls;
	uint64_t slot_select_ticks;
	uint64_t fused_slot_select_calls;
	uint64_t fused_slot_select_ticks;
	uint64_t chain_calls;
	uint64_t chain_ticks;
	uint64_t chain_jobs;
	uint64_t chain_rows;
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

/* Same-A range of a previously authorized batch. No raw command input.
 * completed excludes the failed tile; attempted includes a failed SELECT.
 * Hardware diagnostics are retrieved with GET_STATUS only after failure. */
#define PTA_TNB_GROUP_SELECT 1u
#define PTA_TNB_GROUP_RUN 2u
struct pta_tnb_group_result {
    uint32_t completed;
    uint32_t attempted;
    uint32_t phase;
    uint32_t reserved;

};

struct pta_tnb_multi_slot_chain {
	uint32_t abi_version;
	uint32_t flags;
	uint32_t job_count;
	uint32_t rows;
	uint32_t failed_job;
	uint32_t failed_row;
	uint32_t result;
	uint32_t select_weight_cookie[PTA_TNB_FFN_WEIGHT_SLOT_COUNT];
	uint32_t reserved[2];
	struct pta_tnb_status status;
	struct pta_tnb_chain_job job[PTA_TNB_CHAIN_MAX_JOBS];
};


/* Isolated fixed-recover experiment only. */
#define PTA_TNB_CMD_TEST_RECOVER 0x7fff0020u

/* Batch preparation counters, reset with the existing session profile. */
#define PTA_TNB_CMD_REGISTER_WEIGHT 24
#define PTA_TNB_CMD_PREPARE_REGISTERED_BATCH 25
#define PTA_TNB_CMD_EXEC_GEMM_GROUP 26
#define PTA_TNB_CMD_GET_BATCH_PROFILE 23
#define PTA_TNB_BATCH_PROFILE_VERSION 5
struct pta_tnb_batch_profile {
    uint32_t version;
    uint32_t counter_frequency;
    uint64_t calls;
    uint64_t ticks;
    uint64_t setup_ticks;
    uint64_t map_ticks;
    uint64_t cache_ticks;
    uint64_t cache_bytes;
    uint64_t map_pages;
    uint64_t register_calls;
    uint64_t register_ticks;
    uint64_t register_pages;
    uint64_t register_cache_bytes;
    uint64_t group_calls;
    uint64_t group_jobs;
    uint64_t begin_clear_ticks;
    uint64_t begin_clean_ticks;
    uint64_t close_mmu_ticks;
    uint64_t close_clear_ticks;
    uint64_t close_restore_ticks;
    uint64_t config_ticks;
    uint64_t input_ticks;
    uint64_t output_ticks;


};

#endif
