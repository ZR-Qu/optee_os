/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef __PTA_TNB_H
#define __PTA_TNB_H

#include <stdint.h>

#define PTA_TNB_UUID { 0x0ea88044, 0x846a, 0x47a2, \
	{ 0x87, 0xff, 0xc1, 0xb7, 0x20, 0xf0, 0x69, 0x4e } }

#define PTA_TNB_ABI_VERSION 12
#define PTA_TNB_BUILD_ID 0x49505431u

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
#define PTA_TNB_DEBUG_WORDS 76
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
#define PTA_TNB_CAP_DIRECT_PHYS (1u << 4)
#define PTA_TNB_CAP_UNCACHED_COMMAND (1u << 5)
#define PTA_TNB_CAP_FIXED_TRAILER (1u << 6)
#define PTA_TNB_CAP_NONSECURE_WEIGHT (1u << 7)
#define PTA_TNB_CAP_MULTI_PAGE_IO (1u << 8)
#define PTA_TNB_CAP_REPREPARE (1u << 9)
#define PTA_TNB_CAP_MULTI_CORE (1u << 10)
#define PTA_TNB_CAP_STALLED_WEIGHT_REMAP (1u << 11)
#define PTA_TNB_CAP_FIXED_WEIGHT_PTES (1u << 12)

#define PTA_TNB_USER_TA_UUID { 0x96d83ee6, 0x5ca4, 0x427e, \
	{ 0xb3, 0xea, 0x90, 0xa7, 0x85, 0xd2, 0xe8, 0xa3 } }

#define PTA_TNB_MAX_COMMANDS 128
#define PTA_TNB_COMMAND_TRAILER_COUNT 4
#define PTA_TNB_MAX_INPUT_SIZE 8192
#define PTA_TNB_MAX_SECURE_WEIGHT_SIZE 4096
#define PTA_TNB_MAX_WEIGHT_SIZE 12582912
#define PTA_TNB_MAX_OUTPUT_SIZE 24576

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
};

#endif
