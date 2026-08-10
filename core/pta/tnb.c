// SPDX-License-Identifier: BSD-2-Clause
#include <arm.h>
#include <io.h>
#include <kernel/cache_helpers.h>
#include <kernel/delay.h>
#include <kernel/mutex.h>
#include <kernel/panic.h>
#include <kernel/pseudo_ta.h>
#include <kernel/tee_ta_manager.h>
#include <kernel/ts_manager.h>
#include <kernel/user_mode_ctx.h>
#include <mm/core_memprot.h>
#include <mm/mobj.h>
#include <mm/page_alloc.h>
#include <mm/vm.h>
#include <platform_config.h>
#include <pta_tnb.h>
#include <stdlib.h>
#include <string.h>
#include <trace.h>

#define NPU_INT_MASK 0x20
#define NPU_INT_CLEAR 0x24
#define NPU_INT_STATUS 0x28
#define NPU_INT_RAW_STATUS 0x2c
#define NPU_PC_OP_EN 0x8
#define NPU_PC_DATA_ADDR 0x10
#define NPU_PC_DATA_AMOUNT 0x14
#define NPU_PC_TASK_CONTROL 0x30
#define NPU_PC_DMA_BASE_ADDR 0x34
#define NPU_PC_TASK_STATUS 0x3c
#define NPU_CNA_POINTER 0x1004
#define NPU_CORE_POINTER 0x3004
#define NPU_INT_ALL 0x1ffff

#define RK3588_SOFTRST_CON(id) (0xa00 + ((id) / 16) * 4)
#define RK3588_SRST_A_RKNN1 432
#define RK3588_SRST_H_RKNN1 434
#define RK3588_SRST_A_RKNN2 448
#define RK3588_SRST_H_RKNN2 450
#define RK3588_SRST_A_RKNN0 486
#define RK3588_SRST_H_RKNN0 488

#define BUSSGRF_FIREWALL_CON8 0x260
#define BUSSGRF_NPU0_MASK 0x3

#define FIREWALL_DDR_MST(i) (0x40 + (i) * 4)
#define FIREWALL_DDR_NPU0_MST11 0xffff0000
#define FIREWALL_DDR_NPU0_MST20 0x0000ffff

#define IOMMU0_BASE (NPU_BASE + 0x9000)
#define IOMMU1_BASE (NPU_BASE + 0xa000)
#define IOMMU2_BASE (NPU1_BASE + 0xa000)
#define IOMMU3_BASE (NPU2_BASE + 0xa000)
#define IOMMU_DTE 0x0
#define IOMMU_STATUS 0x4
#define IOMMU_COMMAND 0x8
#define IOMMU_PAGE_FAULT_ADDR 0xc
#define IOMMU_INT_RAW_STATUS 0x14
#define IOMMU_INT_CLEAR 0x18
#define IOMMU_INT_MASK 0x1c
#define IOMMU_INT_STATUS 0x20
#define IOMMU_AUTO_GATING 0x24
#define IOMMU_STATUS_PAGING BIT(0)
#define IOMMU_STATUS_FAULT BIT(1)
#define IOMMU_STATUS_STALL BIT(2)
#define IOMMU_FAULT (BIT(0) | BIT(1))
#define IOMMU_CMD_ENABLE_PAGING 0
#define IOMMU_CMD_DISABLE_PAGING 1
#define IOMMU_CMD_ENABLE_STALL 2
#define IOMMU_CMD_DISABLE_STALL 3
#define IOMMU_CMD_ZAP_CACHE 4
#define IOMMU_DISABLE_FETCH_LIMIT BIT(31)

#define GICD_ISENABLER 0x100
#define GICD_ICENABLER 0x180
#define GICD_ICPENDR 0x280
#define NPU_IRQ_BASE (32 + 110)

#define TNB_PAGE_SIZE 4096
#define TNB_DT_ENTRIES 1024
#define TNB_PT_ENTRIES 1024
/* Match the IOVA window used by the Linux RKNPU allocator. */
#define TNB_IOVA_CMD 0x10000000
#define TNB_IOVA_CMD_CORE(core) (TNB_IOVA_CMD + (core) * TNB_PAGE_SIZE)
#define TNB_IOVA_INPUT 0x11000000
#define TNB_IOVA_WEIGHT 0x12000000
#define TNB_IOVA_WEIGHT_SLOT(slot) \
	((slot) == PTA_TNB_FFN_WEIGHT_SLOT_GATE ? 0x12000000 : \
	 (slot) == PTA_TNB_FFN_WEIGHT_SLOT_UP ? 0x14000000 : 0x15000000)
#define TNB_IOVA_OUTPUT 0x13000000
#define TNB_IOVA_DTE_SHIFT 22
#define TNB_IOVA_PTE_SHIFT 12
#define TNB_PTE_VALID BIT(0)
#define TNB_PTE_READ BIT(1)
#define TNB_PTE_WRITE BIT(2)
#define TNB_PTE_SECURE BIT(3)
#define TNB_DESC_LOW_MASK 0xfffff000
#define TNB_DESC_HI1_MASK GENMASK_64(35, 32)
#define TNB_DESC_HI2_MASK GENMASK_64(39, 36)
#define TNB_TIMEOUT_US 100000
#define TNB_PAGE_TABLE_COUNT 12
#define TNB_FFN_WEIGHT_CACHE_COUNT 84
#define TNB_FFN_WEIGHT_PTE_TABLE_COUNT 3
#define TNB_LEGACY_INPUT_STORAGE_SIZE 8192
#define TNB_LEGACY_OUTPUT_STORAGE_SIZE 24576
#define TNB_INPUT_STORAGE_SIZE 57344
#define TNB_WEIGHT_STORAGE_SIZE PTA_TNB_MAX_SECURE_WEIGHT_SIZE
#define TNB_OUTPUT_STORAGE_SIZE PTA_TNB_MAX_BATCH_OUTPUT_SIZE

#define TNB_COMMAND_TAG(entry) ((uint16_t)((entry) >> 48))
#define TNB_COMMAND_VALUE(entry) ((uint32_t)((entry) >> 16))
#define TNB_COMMAND_REG(entry) ((uint16_t)(entry))
#define TNB_COMMAND_WITH_VALUE(entry, value) \
	(((entry) & ~GENMASK_64(47, 16)) | ((uint64_t)(value) << 16))

#define TNB_INPUT_TAG 0x0201
#define TNB_INPUT_REG 0x1070
#define TNB_WEIGHT_TAG 0x0201
#define TNB_WEIGHT_REG 0x1110
#define TNB_OUTPUT_TAG 0x1001
#define TNB_OUTPUT_REG 0x4020

struct tnb_hw_snapshot {
	uint32_t bussgrf;
	uint32_t ddr_mst2;
	uint32_t ddr_mst11;
	uint32_t ddr_mst20;
	uint32_t ddr_mst32;
	uint32_t iommu_dte[4];
	uint32_t iommu_status[4];
};

struct tnb_mmu_save {
	uint32_t dte;
	uint32_t int_mask;
	uint32_t auto_gating;
	bool paging;
};

struct tnb_cached_weight {
	struct mobj *mobj;
	size_t mobj_offset;
	uint64_t signature;
	bool pte_valid;
	uint32_t page_table[TNB_FFN_WEIGHT_PTE_TABLE_COUNT][TNB_PT_ENTRIES];
};

struct tnb_session {
	struct tnb_cached_weight ffn_weight[TNB_FFN_WEIGHT_CACHE_COUNT];
};

struct tnb_dev {
	void *owner;
	uint32_t state;
	uint32_t stage;
	uint32_t fault;
	uint32_t npu_status;
	uint32_t npu_raw_status;
	uint32_t iommu_status[4];
	uint32_t pc_task_status;
	uint32_t pc_data_addr;
	uint32_t pc_data_amount;
	uint32_t pc_task_control;
	uint32_t prepare_flags;
	uint32_t pte_flags;
	uint32_t debug[PTA_TNB_DEBUG_WORDS];
	uint32_t command_count[PTA_TNB_MAX_CORES];
	uint32_t interrupt_mask[PTA_TNB_MAX_CORES];
	uint32_t completed_mask;
	uint32_t core_count;
	uint32_t input_size;
	uint32_t weight_size;
	uint32_t output_size;
	uint32_t input_mapping_size;
	uint32_t output_mapping_size;
	uint32_t template_id;
	uint32_t weight_cookie;
	void *weight_buffer;
	bool weight_mapping_valid;
	bool nonsecure_weight;
	uint32_t old_npu_mask[PTA_TNB_MAX_CORES];
	bool irq_enabled[PTA_TNB_MAX_CORES];
	bool multi_core;
	bool ffn_weight_slots;
	bool ffn_job_loaded;
	uint32_t ffn_weight_slot_mask;
	uint32_t ffn_weight_cookie[PTA_TNB_FFN_WEIGHT_SLOT_COUNT];
	bool security_touched;
	bool completed_trace_valid;
	struct pta_tnb_trace failure_trace;
	struct pta_tnb_trace completed_trace;
	struct tnb_hw_snapshot before;
	struct tnb_mmu_save mmu[4];
};

static uint64_t cached_command_page[PTA_TNB_MAX_CORES]
	[TNB_PAGE_SIZE / sizeof(uint64_t)]
	__aligned(TNB_PAGE_SIZE);
static uint64_t *uncached_command_page;
static paddr_t uncached_command_pa;
static bool command_page_is_mapped;
static bool command_page_is_uncached;
static struct pta_tnb_trace last_success_trace;
static uint32_t last_success_sequence;
static uint32_t last_failure_sequence;
static uint8_t input_pages[TNB_INPUT_STORAGE_SIZE] __aligned(TNB_PAGE_SIZE);
static uint8_t weight_page[TNB_WEIGHT_STORAGE_SIZE] __aligned(TNB_PAGE_SIZE);
static uint8_t output_pages[TNB_OUTPUT_STORAGE_SIZE] __aligned(TNB_PAGE_SIZE);
static uint32_t directory_table[TNB_DT_ENTRIES] __aligned(TNB_PAGE_SIZE);
static uint32_t page_tables[TNB_PAGE_TABLE_COUNT][TNB_PT_ENTRIES]
	__aligned(TNB_PAGE_SIZE);
static uint32_t alternate_weight_page_tables[3][TNB_PT_ENTRIES]
	__aligned(TNB_PAGE_SIZE);
static const uint64_t command_trailer[PTA_TNB_COMMAND_TRAILER_COUNT] = {
	0x0000000000000000,
	0x0101000000000014,
	0x0041000000000000,
	0x00810000000d0008,
};
static struct tnb_dev dev;
static struct pta_tnb_profile profile;
static struct mutex lock = MUTEX_INITIALIZER;

static TEE_Result close_device(void *session, uint32_t types);

static uint64_t profile_now(void)
{
	return barrier_read_counter_timer();
}

static void profile_add(uint64_t *calls, uint64_t *ticks, uint64_t start)
{
	(*calls)++;
	*ticks += profile_now() - start;
}

static vaddr_t io_map(paddr_t base, size_t size)
{
	return (vaddr_t)phys_to_virt(base, MEM_AREA_IO_SEC, size);
}

static vaddr_t iommu_map(size_t index)
{
	const paddr_t bases[] = {
		IOMMU0_BASE, IOMMU1_BASE, IOMMU2_BASE, IOMMU3_BASE,
	};

	return index < ARRAY_SIZE(bases) ? io_map(bases[index], SIZE_K(4)) : 0;
}

static vaddr_t npu_map(size_t index)
{
	const paddr_t bases[] = { NPU_BASE, NPU1_BASE, NPU2_BASE };

	return index < ARRAY_SIZE(bases) ? io_map(bases[index], NPU_SIZE) : 0;
}

static bool zero_words(const uint32_t *words, size_t count)
{
	size_t n = 0;

	for (n = 0; n < count; n++)
		if (words[n])
			return false;
	return true;
}

static uint32_t page_address(void *buffer)
{
	paddr_t pa = virt_to_phys(buffer);

	return pa <= UINT32_MAX ? (uint32_t)pa : 0;
}

static bool secure_page(void *buffer)
{
	paddr_t pa = virt_to_phys(buffer);

	return pa && !(pa & (TNB_PAGE_SIZE - 1)) && pa < BIT64(40) &&
	       core_pbuf_is(CORE_MEM_SEC, pa, TNB_PAGE_SIZE);
}

static bool page_in_area(void *buffer, uint32_t area)
{
	paddr_t pa = virt_to_phys(buffer);

	return pa && !(pa & (TNB_PAGE_SIZE - 1)) && pa < BIT64(40) &&
	       core_pbuf_is(area, pa, TNB_PAGE_SIZE);
}

static bool init_uncached_command_page(void)
{
	vaddr_t va = 0;
	paddr_t pa = 0;
	size_t n = 0;

	if (uncached_command_page)
		return true;
	va = virt_page_alloc(PTA_TNB_MAX_CORES, MAF_ZERO_INIT);
	if (!va)
		return false;
	pa = virt_to_phys((void *)va);
	if (!pa)
		return false;
	uncached_command_page = (uint64_t *)va;
	uncached_command_pa = pa;
	command_page_is_mapped = true;
	command_page_is_uncached = false;
	for (n = 0; n < PTA_TNB_MAX_CORES; n++)
		if (!secure_page(uncached_command_page +
				 n * TNB_PAGE_SIZE / sizeof(uint64_t)))
			return false;
	return true;
}

static bool set_command_page_uncached(bool uncached)
{
	enum teecore_memtypes old_type = MEM_AREA_TEE_DYN_VASPACE;
	enum teecore_memtypes new_type = MEM_AREA_TEE_DYN_VASPACE;
	TEE_Result res = TEE_SUCCESS;
	vaddr_t va = (vaddr_t)uncached_command_page;
	bool was_mapped = command_page_is_mapped;
	bool was_uncached = command_page_is_uncached;

	if (!uncached_command_page || !uncached_command_pa)
		return false;
	if (command_page_is_mapped && command_page_is_uncached == uncached)
		return true;
	if (command_page_is_uncached)
		old_type = MEM_AREA_IO_SEC;
	if (uncached) {
		if (command_page_is_mapped && !command_page_is_uncached)
			dcache_cleaninv_range(uncached_command_page,
					      PTA_TNB_MAX_CORES * TNB_PAGE_SIZE);
		dsb();
		new_type = MEM_AREA_IO_SEC;
	}
	if (command_page_is_mapped) {
		core_mmu_unmap_pages(va, PTA_TNB_MAX_CORES);
		command_page_is_mapped = false;
	}
	res = core_mmu_map_contiguous_pages(va, uncached_command_pa,
					    PTA_TNB_MAX_CORES,
					    new_type);
	if (res)
		goto restore;
	command_page_is_mapped = true;
	command_page_is_uncached = uncached;
	dsb();
	if (virt_to_phys(uncached_command_page) == uncached_command_pa)
		return true;
	core_mmu_unmap_pages(va, PTA_TNB_MAX_CORES);
	command_page_is_mapped = false;
restore:
	if (was_mapped &&
	    !core_mmu_map_contiguous_pages(va, uncached_command_pa,
					   PTA_TNB_MAX_CORES,
					   old_type)) {
		command_page_is_mapped = true;
		command_page_is_uncached = was_uncached;
		dsb();
	}
	return false;
}

static uint64_t *command_buffer(size_t core)
{
	if (core >= PTA_TNB_MAX_CORES)
		return NULL;
	if (dev.prepare_flags & PTA_TNB_PREPARE_FLAG_UNCACHED_COMMAND)
		return uncached_command_page +
		       core * TNB_PAGE_SIZE / sizeof(uint64_t);
	return cached_command_page[core];
}

static uint32_t make_desc_v2(paddr_t pa)
{
	return (uint32_t)(pa & TNB_DESC_LOW_MASK) |
	       (uint32_t)((pa & TNB_DESC_HI1_MASK) >> 24) |
	       (uint32_t)((pa & TNB_DESC_HI2_MASK) >> 32) |
	       TNB_PTE_VALID;
}

static paddr_t desc_address_v2(uint32_t desc)
{
	return (paddr_t)(desc & TNB_DESC_LOW_MASK) |
	       ((paddr_t)(desc & GENMASK_32(11, 8)) << 24) |
	       ((paddr_t)(desc & GENMASK_32(7, 4)) << 32);
}

static size_t dte_index(uint32_t iova)
{
	return iova >> TNB_IOVA_DTE_SHIFT;
}

static size_t pte_index(uint32_t iova)
{
	return (iova >> TNB_IOVA_PTE_SHIFT) & 0x3ff;
}

static uint32_t *page_table_for_iova(uint32_t iova)
{
	size_t dte = dte_index(iova);
	size_t weight_dte = dte_index(TNB_IOVA_WEIGHT);

	if (dte == dte_index(TNB_IOVA_CMD))
		return page_tables[0];
	if (dte == dte_index(TNB_IOVA_INPUT))
		return page_tables[1];
	if (dte >= weight_dte && dte < weight_dte + 3)
		return page_tables[2 + dte - weight_dte];
	if (dte == dte_index(TNB_IOVA_OUTPUT))
		return page_tables[5];
	if (dte >= dte_index(TNB_IOVA_WEIGHT_SLOT(1)) &&
	    dte < dte_index(TNB_IOVA_WEIGHT_SLOT(1)) + 3)
		return page_tables[6 + dte - dte_index(TNB_IOVA_WEIGHT_SLOT(1))];
	if (dte >= dte_index(TNB_IOVA_WEIGHT_SLOT(2)) &&
	    dte < dte_index(TNB_IOVA_WEIGHT_SLOT(2)) + 3)
		return page_tables[9 + dte - dte_index(TNB_IOVA_WEIGHT_SLOT(2))];
	return NULL;
}

static bool install_page_table(uint32_t iova)
{
	uint32_t *table = page_table_for_iova(iova);
	paddr_t pa = table ? virt_to_phys(table) : 0;

	if (!table || !secure_page(table))
		return false;
	directory_table[dte_index(iova)] = make_desc_v2(pa);
	return true;
}

static bool map_page(uint32_t iova, paddr_t pa, uint32_t flags,
		     uint32_t pte_flags)
{
	uint32_t *table = page_table_for_iova(iova);

	if (!pa || (pa & (TNB_PAGE_SIZE - 1)) || pa >= BIT64(40))
		return false;
	if (!table)
		return false;
	table[pte_index(iova)] = make_desc_v2(pa) | flags | pte_flags;
	return true;
}

static bool map_buffer(uint32_t iova, void *buffer, size_t size,
		       uint32_t flags, bool secure)
{
	size_t offset = 0;
	enum buf_is_attr area = secure ? CORE_MEM_SEC : CORE_MEM_NON_SEC;

	if (!buffer || !size || ((vaddr_t)buffer & (TNB_PAGE_SIZE - 1)) ||
	    (iova & (TNB_PAGE_SIZE - 1)))
		return false;
	for (offset = 0; offset < ROUNDUP(size, TNB_PAGE_SIZE);
	     offset += TNB_PAGE_SIZE) {
		void *page = (uint8_t *)buffer + offset;
		paddr_t pa = virt_to_phys(page);

		if (!page_in_area(page, area) ||
		    !map_page(iova + offset, pa, flags,
			      secure ? dev.pte_flags : 0))
			return false;
	}
	return true;
}

static uint32_t *weight_page_table(size_t slot)
{
	if (slot >= 3)
		return NULL;
	return alternate_weight_page_tables[slot];
}

static bool prepare_weight_page_tables(void *weight, size_t size, bool secure)
{
	enum buf_is_attr area = secure ? CORE_MEM_SEC : CORE_MEM_NON_SEC;
	uint32_t flags = TNB_PTE_READ | (secure ? dev.pte_flags : 0);
	size_t offset = 0;
	size_t n = 0;

	if (!weight || !size || size > PTA_TNB_MAX_WEIGHT_SIZE ||
	    ((vaddr_t)weight & (TNB_PAGE_SIZE - 1)))
		return false;
	for (n = 0; n < 3; n++)
		memset(weight_page_table(n), 0,
		       sizeof(alternate_weight_page_tables[n]));
	for (offset = 0; offset < ROUNDUP(size, TNB_PAGE_SIZE);
	     offset += TNB_PAGE_SIZE) {
		uint32_t iova = TNB_IOVA_WEIGHT + offset;
		size_t slot = dte_index(iova) - dte_index(TNB_IOVA_WEIGHT);
		uint32_t *table = weight_page_table(slot);
		void *page = (uint8_t *)weight + offset;
		paddr_t pa = virt_to_phys(page);

		if (!table || !page_in_area(page, area) || !pa ||
		    (pa & (TNB_PAGE_SIZE - 1)) || pa >= BIT64(40))
			return false;
		table[pte_index(iova)] = make_desc_v2(pa) | flags;
	}
	return true;
}

static bool prepare_page_tables(void *weight)
{
	size_t n = 0;

	if (!secure_page(directory_table))
		return false;

	memset(directory_table, 0, sizeof(directory_table));
	memset(page_tables, 0, sizeof(page_tables));
	if (!install_page_table(TNB_IOVA_CMD) ||
	    !install_page_table(TNB_IOVA_INPUT) ||
	    !install_page_table(TNB_IOVA_WEIGHT) ||
	    !install_page_table(TNB_IOVA_WEIGHT + SIZE_M(4)) ||
	    !install_page_table(TNB_IOVA_WEIGHT + SIZE_M(8)) ||
	    !install_page_table(TNB_IOVA_OUTPUT) ||
	    !map_buffer(TNB_IOVA_CMD, command_buffer(0),
			dev.core_count * TNB_PAGE_SIZE,
			TNB_PTE_READ, true) ||
	    !map_buffer(TNB_IOVA_INPUT, input_pages, sizeof(input_pages),
			TNB_PTE_READ, true) ||
	    !map_buffer(TNB_IOVA_WEIGHT, weight, dev.weight_size,
			TNB_PTE_READ, !dev.nonsecure_weight) ||
	    !map_buffer(TNB_IOVA_OUTPUT, output_pages, sizeof(output_pages),
			TNB_PTE_READ | TNB_PTE_WRITE, true))
		return false;

	dcache_clean_range(directory_table, sizeof(directory_table));
	for (n = 0; n < TNB_PAGE_TABLE_COUNT; ++n)
		dcache_clean_range(page_tables[n], sizeof(page_tables[n]));
	dev.debug[PTA_TNB_DEBUG_PTE_CMD] =
		page_table_for_iova(TNB_IOVA_CMD)[pte_index(TNB_IOVA_CMD)];
	dev.debug[PTA_TNB_DEBUG_PTE_INPUT] =
		page_table_for_iova(TNB_IOVA_INPUT)[pte_index(TNB_IOVA_INPUT)];
	dev.debug[PTA_TNB_DEBUG_PTE_WEIGHT] =
		page_table_for_iova(TNB_IOVA_WEIGHT)[pte_index(TNB_IOVA_WEIGHT)];
	dev.debug[PTA_TNB_DEBUG_PTE_OUTPUT] =
		page_table_for_iova(TNB_IOVA_OUTPUT)[pte_index(TNB_IOVA_OUTPUT)];
	return true;
}

static bool prepare_ffn_page_tables(void)
{
	size_t n = 0;

	if (!secure_page(directory_table))
		return false;
	memset(directory_table, 0, sizeof(directory_table));
	memset(page_tables, 0, sizeof(page_tables));
	if (!install_page_table(TNB_IOVA_CMD) ||
	    !install_page_table(TNB_IOVA_INPUT) ||
	    !install_page_table(TNB_IOVA_WEIGHT_SLOT(0)) ||
	    !install_page_table(TNB_IOVA_WEIGHT_SLOT(0) + SIZE_M(4)) ||
	    !install_page_table(TNB_IOVA_WEIGHT_SLOT(0) + SIZE_M(8)) ||
	    !install_page_table(TNB_IOVA_OUTPUT) ||
	    !install_page_table(TNB_IOVA_WEIGHT_SLOT(1)) ||
	    !install_page_table(TNB_IOVA_WEIGHT_SLOT(1) + SIZE_M(4)) ||
	    !install_page_table(TNB_IOVA_WEIGHT_SLOT(1) + SIZE_M(8)) ||
	    !install_page_table(TNB_IOVA_WEIGHT_SLOT(2)) ||
	    !install_page_table(TNB_IOVA_WEIGHT_SLOT(2) + SIZE_M(4)) ||
	    !install_page_table(TNB_IOVA_WEIGHT_SLOT(2) + SIZE_M(8)) ||
	    !map_buffer(TNB_IOVA_CMD, command_buffer(0),
			dev.core_count * TNB_PAGE_SIZE, TNB_PTE_READ, true) ||
	    !map_buffer(TNB_IOVA_INPUT, input_pages,
			TNB_LEGACY_INPUT_STORAGE_SIZE,
			TNB_PTE_READ, true) ||
	    !map_buffer(TNB_IOVA_OUTPUT, output_pages,
			TNB_LEGACY_OUTPUT_STORAGE_SIZE,
			TNB_PTE_READ | TNB_PTE_WRITE, true))
		return false;

	dcache_clean_range(directory_table, sizeof(directory_table));
	for (n = 0; n < TNB_PAGE_TABLE_COUNT; ++n)
		dcache_clean_range(page_tables[n], sizeof(page_tables[n]));
	dev.debug[PTA_TNB_DEBUG_PTE_CMD] =
		page_table_for_iova(TNB_IOVA_CMD)[pte_index(TNB_IOVA_CMD)];
	dev.debug[PTA_TNB_DEBUG_PTE_INPUT] =
		page_table_for_iova(TNB_IOVA_INPUT)[pte_index(TNB_IOVA_INPUT)];
	dev.debug[PTA_TNB_DEBUG_PTE_OUTPUT] =
		page_table_for_iova(TNB_IOVA_OUTPUT)[pte_index(TNB_IOVA_OUTPUT)];
	dev.input_mapping_size = TNB_LEGACY_INPUT_STORAGE_SIZE;
	dev.output_mapping_size = TNB_LEGACY_OUTPUT_STORAGE_SIZE;
	return true;
}

static bool ensure_ffn_io_page_tables(size_t input_size, size_t output_size)
{
	uint32_t *input_table = page_table_for_iova(TNB_IOVA_INPUT);
	uint32_t *output_table = page_table_for_iova(TNB_IOVA_OUTPUT);
	size_t input_mapping_size = input_size;
	size_t output_mapping_size = output_size;

	if (!input_size || input_size > sizeof(input_pages) || !output_size ||
	    output_size > sizeof(output_pages) || !input_table || !output_table)
		return false;
	/* The down job can require more input pages while the FFN chain is active. */
	if (input_size > TNB_LEGACY_INPUT_STORAGE_SIZE ||
	    output_size > TNB_LEGACY_OUTPUT_STORAGE_SIZE) {
		input_mapping_size = sizeof(input_pages);
		output_mapping_size = sizeof(output_pages);
	}
	if (input_mapping_size > dev.input_mapping_size) {
		if (!map_buffer(TNB_IOVA_INPUT, input_pages, input_mapping_size,
				TNB_PTE_READ, true))
			return false;
		dcache_clean_range(input_table, sizeof(page_tables[0]));
		dev.input_mapping_size = input_mapping_size;
	}
	if (output_mapping_size > dev.output_mapping_size) {
		if (!map_buffer(TNB_IOVA_OUTPUT, output_pages, output_mapping_size,
				TNB_PTE_READ | TNB_PTE_WRITE, true))
			return false;
		dcache_clean_range(output_table, sizeof(page_tables[0]));
		dev.output_mapping_size = output_mapping_size;
	}
	dsb();
	return true;
}

static bool caller_buffer_mobj(void *buffer, size_t size,
			       struct mobj **mobj, size_t *mobj_offset)
{
	struct ts_session *caller = ts_get_calling_session();
	struct user_mode_ctx *uctx = NULL;
	size_t physical_offset = 0;

	if (!caller || !is_user_ta_ctx(caller->ctx) || !buffer || !size ||
	    !mobj || !mobj_offset)
		return false;
	uctx = to_user_mode_ctx(caller->ctx);
	if (vm_buf_to_mboj_offs(uctx, buffer, size, mobj, mobj_offset) ||
	    !*mobj || !mobj_is_nonsec(*mobj))
		return false;
	physical_offset = mobj_get_phys_offs(*mobj, TNB_PAGE_SIZE);
	if (ADD_OVERFLOW(physical_offset, *mobj_offset, &physical_offset) ||
	    (physical_offset & (TNB_PAGE_SIZE - 1)) ||
	    !mobj_check_offset_and_len(*mobj, *mobj_offset, size))
		return false;
	return true;
}

static bool prepare_ffn_weight_slot(size_t slot, uint32_t weight_cookie,
				    struct mobj *weight_mobj,
				    size_t weight_mobj_offset, size_t size,
				    uint64_t *signature)
{
	uint32_t iova = TNB_IOVA_WEIGHT_SLOT(slot);
	uint64_t value = UINT64_C(1469598103934665603);
	size_t offset = 0;
	size_t n = 0;

	if (slot >= PTA_TNB_FFN_WEIGHT_SLOT_COUNT ||
	    size != PTA_TNB_MAX_WEIGHT_SIZE || !weight_mobj ||
	    !weight_cookie || !signature)
		return false;
	for (n = 0; n < 3; ++n) {
		uint32_t *table = page_table_for_iova(iova + n * SIZE_M(4));

		if (!table)
			return false;
		memset(table, 0, sizeof(page_tables[0]));
	}
	for (offset = 0; offset < size; offset += TNB_PAGE_SIZE) {
		paddr_t pa = 0;

		if (mobj_get_pa(weight_mobj, weight_mobj_offset + offset,
				TNB_PAGE_SIZE, &pa) ||
		    !map_page(iova + offset, pa, TNB_PTE_READ, 0))
			return false;
		value ^= pa;
		value *= UINT64_C(1099511628211);
	}
	for (n = 0; n < 3; ++n) {
		uint32_t *table = page_table_for_iova(iova + n * SIZE_M(4));

		dcache_clean_range(table, sizeof(page_tables[0]));
	}
	dev.debug[PTA_TNB_DEBUG_PTE_WEIGHT] =
		page_table_for_iova(iova)[pte_index(iova)];
	*signature = value;
	return true;
}

static bool cache_ffn_weight_slot(struct tnb_cached_weight *cached,
				  size_t slot)
{
	uint32_t iova = TNB_IOVA_WEIGHT_SLOT(slot);
	size_t n = 0;

	if (!cached || slot >= PTA_TNB_FFN_WEIGHT_SLOT_COUNT)
		return false;
	for (n = 0; n < TNB_FFN_WEIGHT_PTE_TABLE_COUNT; ++n) {
		uint32_t *table = page_table_for_iova(iova + n * SIZE_M(4));

		if (!table)
			return false;
		memcpy(cached->page_table[n], table, sizeof(cached->page_table[n]));
	}
	cached->pte_valid = true;
	return true;
}

static bool install_cached_ffn_weight_slot(
		const struct tnb_cached_weight *cached, size_t slot)
{
	uint32_t iova = TNB_IOVA_WEIGHT_SLOT(slot);
	size_t n = 0;

	if (!cached || !cached->pte_valid ||
	    slot >= PTA_TNB_FFN_WEIGHT_SLOT_COUNT)
		return false;
	for (n = 0; n < TNB_FFN_WEIGHT_PTE_TABLE_COUNT; ++n) {
		uint32_t *table = page_table_for_iova(iova + n * SIZE_M(4));

		if (!table)
			return false;
		memcpy(table, cached->page_table[n], sizeof(cached->page_table[n]));
		dcache_clean_range(table, sizeof(cached->page_table[n]));
	}
	dev.debug[PTA_TNB_DEBUG_PTE_WEIGHT] =
		page_table_for_iova(iova)[pte_index(iova)];
	return true;
}

static bool valid_command_register(uint16_t tag, uint16_t reg)
{
	if (reg & 3)
		return false;
	if (tag == 0x0201)
		return reg >= 0x1000 && reg <= 0x11ff;
	if (tag == 0x0801)
		return reg >= 0x3000 && reg <= 0x30ff;
	if (tag == 0x1001)
		return reg >= 0x4000 && reg <= 0x41ff;
	return false;
}

static bool validate_command_blob(uint32_t command_count,
				  uint32_t input_patch,
				  uint32_t weight_patch,
				  uint32_t output_patch,
				  const uint64_t *commands)
{
	size_t input_refs = 0;
	size_t weight_refs = 0;
	size_t output_refs = 0;
	size_t n = 0;

	if (!commands || !command_count ||
	    command_count > PTA_TNB_MAX_COMMANDS ||
	    input_patch >= command_count ||
	    weight_patch >= command_count ||
	    output_patch >= command_count ||
	    input_patch == weight_patch ||
	    input_patch == output_patch ||
	    weight_patch == output_patch)
		return false;

	for (n = 0; n < command_count; n++) {
		uint16_t tag = TNB_COMMAND_TAG(commands[n]);
		uint16_t reg = TNB_COMMAND_REG(commands[n]);

		if (!valid_command_register(tag, reg))
			return false;
		if (tag == TNB_INPUT_TAG && reg == TNB_INPUT_REG)
			input_refs++;
		if (tag == TNB_WEIGHT_TAG && reg == TNB_WEIGHT_REG)
			weight_refs++;
		if (tag == TNB_OUTPUT_TAG && reg == TNB_OUTPUT_REG)
			output_refs++;
	}

	return !memcmp(commands + command_count, command_trailer,
		       sizeof(command_trailer)) &&
	       input_refs == 1 && weight_refs == 1 && output_refs == 1 &&
	       TNB_COMMAND_TAG(commands[input_patch]) == TNB_INPUT_TAG &&
	       TNB_COMMAND_REG(commands[input_patch]) == TNB_INPUT_REG &&
	       TNB_COMMAND_TAG(commands[weight_patch]) == TNB_WEIGHT_TAG &&
	       TNB_COMMAND_REG(commands[weight_patch]) == TNB_WEIGHT_REG &&
	       TNB_COMMAND_TAG(commands[output_patch]) == TNB_OUTPUT_TAG &&
	       TNB_COMMAND_REG(commands[output_patch]) == TNB_OUTPUT_REG;
}

static bool validate_template(const struct pta_tnb_prepare *request,
			      const uint64_t *commands)
{
	if (request->abi_version != PTA_TNB_ABI_VERSION ||
	    !request->template_id ||
	    request->interrupt_mask != 0x300 ||
	    !request->input_size ||
	    request->input_size > PTA_TNB_MAX_INPUT_SIZE ||
	    !request->weight_size ||
	    request->weight_size > PTA_TNB_MAX_WEIGHT_SIZE ||
	    !request->output_size ||
	    request->output_size > PTA_TNB_MAX_OUTPUT_SIZE ||
	    (request->reserved[0] & ~PTA_TNB_PREPARE_FLAGS_MASK) ||
	    ((request->reserved[0] & PTA_TNB_PREPARE_FLAG_NONSECURE_WEIGHT) &&
	     (request->reserved[0] & PTA_TNB_PREPARE_FLAG_DIRECT_PHYS)) ||
	    (!(request->reserved[0] & PTA_TNB_PREPARE_FLAG_NONSECURE_WEIGHT) &&
	     request->weight_size > PTA_TNB_MAX_SECURE_WEIGHT_SIZE) ||
	    !zero_words(&request->reserved[1], ARRAY_SIZE(request->reserved) - 1))
		return false;

	return validate_command_blob(request->command_count,
				     request->input_patch,
				     request->weight_patch,
				     request->output_patch, commands);
}

static void clear_work(void)
{
	size_t input_clear_size = dev.input_size >
		TNB_LEGACY_INPUT_STORAGE_SIZE ? sizeof(input_pages) :
		TNB_LEGACY_INPUT_STORAGE_SIZE;
	size_t output_clear_size = dev.input_size >
		TNB_LEGACY_INPUT_STORAGE_SIZE || dev.output_size >
		TNB_LEGACY_OUTPUT_STORAGE_SIZE ? sizeof(output_pages) :
		TNB_LEGACY_OUTPUT_STORAGE_SIZE;

	dev.template_id = 0;
	dev.weight_cookie = 0;
	dev.weight_buffer = NULL;
	dev.weight_mapping_valid = false;
	dev.ffn_weight_slots = false;
	dev.ffn_job_loaded = false;
	memset(dev.ffn_weight_cookie, 0, sizeof(dev.ffn_weight_cookie));
	memset(cached_command_page, 0, sizeof(cached_command_page));
	if (uncached_command_page && command_page_is_mapped &&
	    !command_page_is_uncached)
		memset(uncached_command_page, 0,
		       PTA_TNB_MAX_CORES * TNB_PAGE_SIZE);
	memset(input_pages, 0, input_clear_size);
	memset(weight_page, 0, sizeof(weight_page));
	memset(output_pages, 0, output_clear_size);
	memset(directory_table, 0, sizeof(directory_table));
	memset(page_tables, 0, sizeof(page_tables));
	memset(alternate_weight_page_tables, 0,
	       sizeof(alternate_weight_page_tables));
	dcache_clean_range(cached_command_page, sizeof(cached_command_page));
	dcache_clean_range(input_pages, input_clear_size);
	dcache_clean_range(weight_page, sizeof(weight_page));
	dcache_clean_range(output_pages, output_clear_size);
	dcache_clean_range(directory_table, sizeof(directory_table));
	dcache_clean_range(page_tables, sizeof(page_tables));
	dcache_clean_range(alternate_weight_page_tables,
			   sizeof(alternate_weight_page_tables));
	dsb();
}

static TEE_Result snapshot(struct tnb_hw_snapshot *out)
{
	vaddr_t bussgrf = io_map(BUSSGRF_BASE, BUSSGRF_SIZE);
	vaddr_t firewall = io_map(FIREWALL_DDR_BASE, FIREWALL_DDR_SIZE);
	size_t n = 0;

	if (!out || !bussgrf || !firewall)
		return TEE_ERROR_GENERIC;
	memset(out, 0, sizeof(*out));
	out->bussgrf = io_read32(bussgrf + BUSSGRF_FIREWALL_CON8);
	out->ddr_mst2 = io_read32(firewall + FIREWALL_DDR_MST(2));
	out->ddr_mst11 = io_read32(firewall + FIREWALL_DDR_MST(11));
	out->ddr_mst20 = io_read32(firewall + FIREWALL_DDR_MST(20));
	out->ddr_mst32 = io_read32(firewall + FIREWALL_DDR_MST(32));
	for (n = 0; n < ARRAY_SIZE(out->iommu_dte); n++) {
		vaddr_t iommu = iommu_map(n);

		if (!iommu)
			return TEE_ERROR_GENERIC;
		out->iommu_dte[n] = io_read32(iommu + IOMMU_DTE);
		out->iommu_status[n] = io_read32(iommu + IOMMU_STATUS);
	}
	return TEE_SUCCESS;
}

static bool snapshot_equal(const struct tnb_hw_snapshot *left,
			   const struct tnb_hw_snapshot *right)
{
	return !memcmp(left, right, sizeof(*left));
}

static bool mmu_fault(void)
{
	size_t n = 0;

	for (n = 0; n < 4; n++) {
		vaddr_t iommu = iommu_map(n);

		if (!iommu ||
		    io_read32(iommu + IOMMU_STATUS) & IOMMU_STATUS_FAULT ||
		    io_read32(iommu + IOMMU_INT_STATUS) & IOMMU_FAULT ||
		    io_read32(iommu + IOMMU_INT_RAW_STATUS) & IOMMU_FAULT)
			return true;
	}
	return false;
}

static void capture_run_status(vaddr_t npu)
{
	size_t n = 0;

	dev.npu_status = io_read32(npu + NPU_INT_STATUS);
	dev.npu_raw_status = io_read32(npu + NPU_INT_RAW_STATUS);
	dev.pc_task_status = io_read32(npu + NPU_PC_TASK_STATUS);
	dev.pc_data_addr = io_read32(npu + NPU_PC_DATA_ADDR);
	dev.pc_data_amount = io_read32(npu + NPU_PC_DATA_AMOUNT);
	dev.pc_task_control = io_read32(npu + NPU_PC_TASK_CONTROL);
	for (n = 0; n < ARRAY_SIZE(dev.iommu_status); n++) {
		vaddr_t iommu = iommu_map(n);

		dev.iommu_status[n] = io_read32(iommu + IOMMU_STATUS);
		dev.debug[PTA_TNB_DEBUG_IOMMU_FAULT_ADDR0 + n] =
			io_read32(iommu + IOMMU_PAGE_FAULT_ADDR);
		dev.debug[PTA_TNB_DEBUG_IOMMU_RAW0 + n] =
			io_read32(iommu + IOMMU_INT_RAW_STATUS);
		dev.debug[PTA_TNB_DEBUG_IOMMU_IRQ0 + n] =
			io_read32(iommu + IOMMU_INT_STATUS);
	}
}

static bool wait_iommu_status(uint32_t bit, bool set)
{
	uint64_t deadline = timeout_init_us(1000);

	while (!timeout_elapsed(deadline)) {
		bool ready = true;
		size_t n = 0;

		for (n = 0; n < 4; n++) {
			vaddr_t iommu = iommu_map(n);
			bool value = iommu && !!(io_read32(iommu + IOMMU_STATUS) & bit);

			if (!iommu || value != set)
				ready = false;
		}
		if (ready)
			return true;
		udelay(1);
	}
	return false;
}

static bool disable_iommu(struct tnb_mmu_save saved[4])
{
	size_t n = 0;

	for (n = 0; n < 4; n++) {
		vaddr_t iommu = iommu_map(n);

		if (!iommu)
			return false;
		saved[n].dte = io_read32(iommu + IOMMU_DTE);
		saved[n].int_mask = io_read32(iommu + IOMMU_INT_MASK);
		saved[n].auto_gating = io_read32(iommu + IOMMU_AUTO_GATING);
		saved[n].paging = !!(io_read32(iommu + IOMMU_STATUS) &
					IOMMU_STATUS_PAGING);
		io_write32(iommu + IOMMU_INT_MASK, 0);
		if (saved[n].paging)
			io_write32(iommu + IOMMU_COMMAND,
				   IOMMU_CMD_ENABLE_STALL);
	}
	dsb();
	if (!wait_iommu_status(IOMMU_STATUS_STALL, true))
		return false;
	for (n = 0; n < 4; n++)
		if (saved[n].paging)
			io_write32(iommu_map(n) + IOMMU_COMMAND,
				   IOMMU_CMD_DISABLE_PAGING);
	dsb();
	if (!wait_iommu_status(IOMMU_STATUS_PAGING, false))
		return false;
	for (n = 0; n < 4; n++) {
		vaddr_t iommu = iommu_map(n);

		io_write32(iommu + IOMMU_DTE, 0);
		io_write32(iommu + IOMMU_INT_CLEAR, IOMMU_FAULT);
		io_write32(iommu + IOMMU_COMMAND, IOMMU_CMD_DISABLE_STALL);
	}
	dsb();
	return wait_iommu_status(IOMMU_STATUS_STALL, false) && !mmu_fault();
}

static bool install_iommu(void)
{
	paddr_t dt_pa = virt_to_phys(directory_table);
	uint32_t dte = make_desc_v2(dt_pa);
	size_t n = 0;

	for (n = 0; n < 4; n++) {
		vaddr_t iommu = iommu_map(n);

		if (!iommu)
			return false;
		io_write32(iommu + IOMMU_DTE, dte);
		io_write32(iommu + IOMMU_COMMAND, IOMMU_CMD_ZAP_CACHE);
		io_write32(iommu + IOMMU_INT_CLEAR, IOMMU_FAULT);
		io_write32(iommu + IOMMU_INT_MASK, IOMMU_FAULT);
		io_setbits32(iommu + IOMMU_AUTO_GATING,
			     IOMMU_DISABLE_FETCH_LIMIT);
		io_write32(iommu + IOMMU_COMMAND, IOMMU_CMD_ENABLE_PAGING);
	}
	dsb();
	if (!wait_iommu_status(IOMMU_STATUS_PAGING, true) || mmu_fault())
		return false;
	for (n = 0; n < 4; n++)
		if (desc_address_v2(io_read32(iommu_map(n) + IOMMU_DTE)) !=
		    dt_pa)
			return false;
	dev.debug[PTA_TNB_DEBUG_DTE_WRITTEN] = dte;
	for (n = 0; n < 4; n++) {
		dev.debug[PTA_TNB_DEBUG_DTE_READBACK0 + n] =
			io_read32(iommu_map(n) + IOMMU_DTE);
		dev.debug[PTA_TNB_DEBUG_IOMMU_STATUS0 + n] =
			io_read32(iommu_map(n) + IOMMU_STATUS);
	}
	return true;
}

static bool restore_iommu(const struct tnb_mmu_save saved[4])
{
	size_t n = 0;

	for (n = 0; n < 4; n++) {
		vaddr_t iommu = iommu_map(n);

		if (!iommu)
			return false;
		io_write32(iommu + IOMMU_INT_MASK, 0);
		io_write32(iommu + IOMMU_COMMAND, IOMMU_CMD_ENABLE_STALL);
	}
	dsb();
	if (!wait_iommu_status(IOMMU_STATUS_STALL, true))
		return false;
	for (n = 0; n < 4; n++) {
		vaddr_t iommu = iommu_map(n);

		io_write32(iommu + IOMMU_COMMAND, IOMMU_CMD_DISABLE_PAGING);
		io_write32(iommu + IOMMU_DTE, saved[n].dte);
		io_write32(iommu + IOMMU_COMMAND, IOMMU_CMD_ZAP_CACHE);
		io_write32(iommu + IOMMU_AUTO_GATING, saved[n].auto_gating);
		if (saved[n].paging)
			io_write32(iommu + IOMMU_COMMAND,
				   IOMMU_CMD_ENABLE_PAGING);
		io_write32(iommu + IOMMU_INT_MASK, saved[n].int_mask);
		io_write32(iommu + IOMMU_COMMAND, IOMMU_CMD_DISABLE_STALL);
	}
	dsb();
	for (n = 0; n < 4; n++) {
		vaddr_t iommu = iommu_map(n);
		bool paging = !!(io_read32(iommu + IOMMU_STATUS) &
					 IOMMU_STATUS_PAGING);

		if (io_read32(iommu + IOMMU_DTE) != saved[n].dte ||
		    paging != saved[n].paging)
			return false;
	}
	return true;
}

static void clear_irq_pending(size_t core)
{
	vaddr_t gic = (vaddr_t)phys_to_virt_io(GICD_BASE, GIC_SIZE);
	uint32_t irq = NPU_IRQ_BASE + core;

	if (gic) {
		io_write32(gic + GICD_ICPENDR + (irq / 32) * 4,
			   BIT(irq % 32));
		dsb();
	}
}

static bool irq_state(size_t core, bool *enabled)
{
	vaddr_t gic = (vaddr_t)phys_to_virt_io(GICD_BASE, GIC_SIZE);
	uint32_t irq = NPU_IRQ_BASE + core;

	if (!gic || core >= PTA_TNB_MAX_CORES)
		return false;
	*enabled = !!(io_read32(gic + GICD_ISENABLER +
				(irq / 32) * 4) & BIT(irq % 32));
	return true;
}

static bool disable_irq(size_t core, bool *enabled)
{
	vaddr_t gic = (vaddr_t)phys_to_virt_io(GICD_BASE, GIC_SIZE);
	bool current = false;
	uint32_t irq = NPU_IRQ_BASE + core;

	if (!gic || !irq_state(core, enabled))
		return false;
	io_write32(gic + GICD_ICENABLER + (irq / 32) * 4,
		   BIT(irq % 32));
	dsb();
	clear_irq_pending(core);
	return irq_state(core, &current) && !current;
}

static bool restore_irq(size_t core, bool enabled)
{
	vaddr_t gic = (vaddr_t)phys_to_virt_io(GICD_BASE, GIC_SIZE);
	bool current = false;
	uint32_t irq = NPU_IRQ_BASE + core;

	if (!gic)
		return false;
	clear_irq_pending(core);
	io_write32(gic + (enabled ? GICD_ISENABLER : GICD_ICENABLER) +
		   (irq / 32) * 4, BIT(irq % 32));
	dsb();
	return irq_state(core, &current) && current == enabled;
}

static bool set_secure(vaddr_t bussgrf, vaddr_t firewall, bool full)
{
	uint32_t mask = full ? 0xf : BUSSGRF_NPU0_MASK;

	io_write32(bussgrf + BUSSGRF_FIREWALL_CON8,
		   (mask << 16) | mask);
	if (full)
		io_clrbits32(firewall + FIREWALL_DDR_MST(2),
			       0xffff0000);
	io_clrbits32(firewall + FIREWALL_DDR_MST(11),
		       FIREWALL_DDR_NPU0_MST11);
	io_clrbits32(firewall + FIREWALL_DDR_MST(20),
		       FIREWALL_DDR_NPU0_MST20);
	if (full)
		io_clrbits32(firewall + FIREWALL_DDR_MST(32),
			       0xffff0000);
	dsb();
	return (io_read32(bussgrf + BUSSGRF_FIREWALL_CON8) &
		mask) == mask &&
	       (!full || !(io_read32(firewall + FIREWALL_DDR_MST(2)) &
			    0xffff0000)) &&
	       !(io_read32(firewall + FIREWALL_DDR_MST(11)) &
		 FIREWALL_DDR_NPU0_MST11) &&
	       !(io_read32(firewall + FIREWALL_DDR_MST(20)) &
		 FIREWALL_DDR_NPU0_MST20) &&
	       (!full || !(io_read32(firewall + FIREWALL_DDR_MST(32)) &
			    0xffff0000));
}

static bool restore_secure(vaddr_t bussgrf, vaddr_t firewall,
			   const struct tnb_hw_snapshot *before)
{
	io_write32(firewall + FIREWALL_DDR_MST(2), before->ddr_mst2);
	io_write32(firewall + FIREWALL_DDR_MST(11), before->ddr_mst11);
	io_write32(firewall + FIREWALL_DDR_MST(20), before->ddr_mst20);
	io_write32(firewall + FIREWALL_DDR_MST(32), before->ddr_mst32);
	io_write32(bussgrf + BUSSGRF_FIREWALL_CON8,
		   (0xf << 16) | (before->bussgrf & 0xf));
	dsb();
	return io_read32(firewall + FIREWALL_DDR_MST(2)) ==
		before->ddr_mst2 &&
	       io_read32(firewall + FIREWALL_DDR_MST(11)) ==
		before->ddr_mst11 &&
	       io_read32(firewall + FIREWALL_DDR_MST(20)) ==
		before->ddr_mst20 &&
	       io_read32(firewall + FIREWALL_DDR_MST(32)) ==
		before->ddr_mst32 &&
	       (io_read32(bussgrf + BUSSGRF_FIREWALL_CON8) &
		0xf) == (before->bussgrf & 0xf);
}

static void keep_secure(vaddr_t bussgrf, vaddr_t firewall)
{
	bool full = dev.prepare_flags & PTA_TNB_PREPARE_FLAG_FULL_SECURITY;
	uint32_t mask = full ? 0xf : BUSSGRF_NPU0_MASK;

	io_write32(bussgrf + BUSSGRF_FIREWALL_CON8,
		   (mask << 16) | mask);
	if (full)
		io_clrbits32(firewall + FIREWALL_DDR_MST(2), 0xffff0000);
	io_clrbits32(firewall + FIREWALL_DDR_MST(11),
		       FIREWALL_DDR_NPU0_MST11);
	io_clrbits32(firewall + FIREWALL_DDR_MST(20),
		       FIREWALL_DDR_NPU0_MST20);
	if (full)
		io_clrbits32(firewall + FIREWALL_DDR_MST(32), 0xffff0000);
	dsb();
}

static uint32_t normalize_status(uint32_t status)
{
	const uint32_t groups[] = { 0x3, 0xc, 0x30, 0xc0, 0x300, 0xc00 };
	uint32_t value = 0;
	size_t n = 0;

	for (n = 0; n < ARRAY_SIZE(groups); n++)
		if (status & groups[n])
			value |= groups[n];
	return value;
}

static void capture_core_snapshot(vaddr_t npu, uint32_t status,
					  uint32_t raw,
					  struct pta_tnb_core_snapshot *snapshot)
{
	snapshot->int_status = status;
	snapshot->int_raw_status = raw;
	snapshot->task_status = io_read32(npu + NPU_PC_TASK_STATUS);
	snapshot->data_addr = io_read32(npu + NPU_PC_DATA_ADDR);
	snapshot->data_amount = io_read32(npu + NPU_PC_DATA_AMOUNT);
	snapshot->task_control = io_read32(npu + NPU_PC_TASK_CONTROL);
	snapshot->cna_pointer = io_read32(npu + NPU_CNA_POINTER);
	snapshot->core_pointer = io_read32(npu + NPU_CORE_POINTER);
}

static void capture_core_failure_snapshot(
				vaddr_t npu, uint32_t status, uint32_t raw,
				struct pta_tnb_core_snapshot *snapshot)
{
	snapshot->int_status = status;
	snapshot->int_raw_status = raw;
	snapshot->task_status = io_read32(npu + NPU_PC_TASK_STATUS);
	snapshot->data_addr = io_read32(npu + NPU_PC_DATA_ADDR);
}

static void capture_failure_npu_debug(void)
{
	size_t n = 0;

	memset(&dev.failure_trace, 0, sizeof(dev.failure_trace));
	dev.failure_trace.valid = 1;
	dev.failure_trace.sequence = ++last_failure_sequence;
	dev.failure_trace.template_id = dev.template_id;
	dev.failure_trace.prepare_flags = dev.prepare_flags;
	dev.failure_trace.core_count = dev.core_count;
	dev.failure_trace.complete_mask = dev.completed_mask;

	for (n = 0; n < PTA_TNB_MAX_CORES; n++) {
		vaddr_t npu = npu_map(n);
		struct pta_tnb_core_snapshot *snapshot =
			&dev.failure_trace.core[n];

		if (!npu)
			continue;
		capture_core_failure_snapshot(
			npu, io_read32(npu + NPU_INT_STATUS),
			io_read32(npu + NPU_INT_RAW_STATUS), snapshot);
		dev.debug[PTA_TNB_DEBUG_FAILURE_INT_STATUS0 + n] =
			snapshot->int_status;
		dev.debug[PTA_TNB_DEBUG_FAILURE_INT_RAW0 + n] =
			snapshot->int_raw_status;
		dev.debug[PTA_TNB_DEBUG_FAILURE_TASK_STATUS0 + n] =
			snapshot->task_status;
		dev.debug[PTA_TNB_DEBUG_FAILURE_DATA_ADDR0 + n] =
			snapshot->data_addr;
	}
}

static uint32_t poll_npus(void)
{
	uint64_t deadline = timeout_init_us(TNB_TIMEOUT_US);
	bool done[PTA_TNB_MAX_CORES] = { false };
	struct pta_tnb_trace success_trace = {
		.template_id = dev.template_id,
		.prepare_flags = dev.prepare_flags,
		.core_count = dev.core_count,
	};
	size_t complete = 0;

	while (!timeout_elapsed(deadline)) {
		size_t n = 0;

		if (mmu_fault()) {
			capture_failure_npu_debug();
			capture_run_status(npu_map(0));
			return PTA_TNB_FAULT_IOMMU;
		}
		for (n = 0; n < dev.core_count; n++) {
			vaddr_t npu = npu_map(n);
			uint32_t status = 0;
			uint32_t raw = 0;

			if (!npu) {
				capture_failure_npu_debug();
				return PTA_TNB_FAULT_NPU;
			}
			status = io_read32(npu + NPU_INT_STATUS);
			raw = io_read32(npu + NPU_INT_RAW_STATUS);
			dev.npu_status |= status;
			dev.npu_raw_status |= raw;
			if (!status || done[n])
				continue;
			if (normalize_status(status) != dev.interrupt_mask[n]) {
				capture_failure_npu_debug();
				capture_run_status(npu);
				return PTA_TNB_FAULT_NPU;
			}
			capture_core_snapshot(npu, status, raw,
					      &success_trace.core[n]);
			io_write32(npu + NPU_INT_CLEAR, NPU_INT_ALL);
			done[n] = true;
			dev.completed_mask |= BIT(n);
			complete++;
		}
		if (complete == dev.core_count) {
			if (dev.multi_core) {
				success_trace.valid = 1;
				success_trace.complete_mask = dev.completed_mask;
				dev.completed_trace = success_trace;
				dev.completed_trace_valid = true;
			}
			return PTA_TNB_FAULT_NONE;
		}
		udelay(1);
	}
	capture_failure_npu_debug();
	capture_run_status(npu_map(0));
	return PTA_TNB_FAULT_TIMEOUT;
}

static void capture_npu_debug(vaddr_t npu, uint32_t base)
{
	dev.debug[base + 0] = io_read32(npu + NPU_INT_STATUS);
	dev.debug[base + 1] = io_read32(npu + NPU_INT_RAW_STATUS);
	dev.debug[base + 2] = io_read32(npu + NPU_PC_TASK_STATUS);
	dev.debug[base + 3] = io_read32(npu + NPU_PC_DATA_ADDR);
	dev.debug[base + 4] = io_read32(npu + NPU_PC_DATA_AMOUNT);
	dev.debug[base + 5] = io_read32(npu + NPU_PC_TASK_CONTROL);
}

static void capture_output_debug(void)
{
	uint32_t words[4] = { };
	size_t n = 0;

	dcache_inv_range(output_pages, sizeof(output_pages));
	memcpy(words, output_pages, sizeof(words));
	for (n = 0; n < 4; n++)
		dev.debug[PTA_TNB_DEBUG_OUTPUT0 + n] = words[n];
}

static void capture_command_registers(vaddr_t npu, uint32_t base)
{
	dev.debug[base + 0] = io_read32(npu + TNB_INPUT_REG);
	dev.debug[base + 1] = io_read32(npu + TNB_WEIGHT_REG);
	dev.debug[base + 2] = io_read32(npu + 0x1040);
	dev.debug[base + 3] = io_read32(npu + 0x3010);
	dev.debug[base + 4] = io_read32(npu + 0x4004);
	dev.debug[base + 5] = io_read32(npu + TNB_OUTPUT_REG);
}

static void fill_status(struct pta_tnb_status *status)
{
	memset(status, 0, sizeof(*status));
	status->abi_version = PTA_TNB_ABI_VERSION;
	status->state = dev.state;
	status->stage = dev.stage;
	status->fault = dev.fault;
	status->npu_status = dev.npu_status;
	status->npu_raw_status = dev.npu_raw_status;
	memcpy(status->iommu_status, dev.iommu_status,
	       sizeof(status->iommu_status));
	status->reserved[0] = dev.pc_task_status;
	status->reserved[1] = dev.pc_data_addr;
	status->reserved[2] = dev.pc_data_amount;
	status->reserved[3] = dev.pc_task_control;
	memcpy(status->debug, dev.debug, sizeof(status->debug));
	memcpy(&status->last_success, &last_success_trace,
	       sizeof(status->last_success));
	memcpy(&status->failure, &dev.failure_trace,
	       sizeof(status->failure));
	status->profile = profile;
}

static void pulse_npu_resets(void)
{
	const uint32_t resets[] = {
		RK3588_SRST_A_RKNN0,
		RK3588_SRST_A_RKNN1,
		RK3588_SRST_A_RKNN2,
		RK3588_SRST_H_RKNN0,
		RK3588_SRST_H_RKNN1,
		RK3588_SRST_H_RKNN2,
	};
	vaddr_t cru = io_map(CRU_BASE, CRU_SIZE);
	size_t n = 0;

	if (!cru)
		panic();
	for (n = 0; n < ARRAY_SIZE(resets); n++) {
		uint32_t bit = resets[n] % 16;

		io_write32(cru + RK3588_SOFTRST_CON(resets[n]),
			   BIT(bit + 16) | BIT(bit));
	}
	udelay(10);
	for (n = 0; n < ARRAY_SIZE(resets); n++) {
		uint32_t bit = resets[n] % 16;

		io_write32(cru + RK3588_SOFTRST_CON(resets[n]),
			   BIT(bit + 16));
	}
	udelay(10);
	dsb();
}

static void enter_locked(uint32_t fault)
{
	vaddr_t bussgrf = io_map(BUSSGRF_BASE, BUSSGRF_SIZE);
	vaddr_t firewall = io_map(FIREWALL_DDR_BASE, FIREWALL_DDR_SIZE);
	size_t n = 0;

	dev.state = PTA_TNB_STATE_LOCKED;
	dev.owner = NULL;
	dev.fault = fault;
	dev.template_id = 0;
	dev.weight_buffer = NULL;
	dev.weight_mapping_valid = false;
	for (n = 0; n < PTA_TNB_MAX_CORES; n++) {
		vaddr_t npu = npu_map(n);

		if (npu) {
			io_write32(npu + NPU_INT_MASK, 0);
			io_write32(npu + NPU_INT_CLEAR, NPU_INT_ALL);
		}
		clear_irq_pending(n);
	}
	if (bussgrf && firewall)
		keep_secure(bussgrf, firewall);
	dsb();
	pulse_npu_resets();
}

static TEE_Result get_caps(uint32_t types, TEE_Param params[TEE_NUM_PARAMS])
{
	struct pta_tnb_caps caps = {
		.abi_version = PTA_TNB_ABI_VERSION,
		.features = PTA_TNB_CAP_CALLER_BOUND | PTA_TNB_CAP_IOMMU_V2 |
				    PTA_TNB_CAP_SECURE_PTE |
				    PTA_TNB_CAP_FULL_SECURITY |
				    PTA_TNB_CAP_DIRECT_PHYS |
				    PTA_TNB_CAP_UNCACHED_COMMAND |
				    PTA_TNB_CAP_FIXED_TRAILER |
				    PTA_TNB_CAP_NONSECURE_WEIGHT |
				    PTA_TNB_CAP_MULTI_PAGE_IO |
				    PTA_TNB_CAP_REPREPARE |
				    PTA_TNB_CAP_MULTI_CORE |
				    PTA_TNB_CAP_STALLED_WEIGHT_REMAP |
				    PTA_TNB_CAP_FIXED_WEIGHT_PTES |
			    PTA_TNB_CAP_PROFILE |
			    PTA_TNB_CAP_FFN_WEIGHT_SLOTS |
			    PTA_TNB_CAP_MULTI_SLOT_CHAIN |
			    PTA_TNB_CAP_CACHED_FFN_SLOT_SET |
			    PTA_TNB_CAP_FUSED_SELECT_CHAIN,
		.max_templates = 1,
		.max_tensors = 3,
		.reserved = { PTA_TNB_BUILD_ID },
	};
	const uint32_t expected =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
				TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
				TEE_PARAM_TYPE_NONE);

	if (types != expected)
		return TEE_ERROR_BAD_PARAMETERS;
	if (params[0].memref.size < sizeof(caps)) {
		params[0].memref.size = sizeof(caps);
		return TEE_ERROR_SHORT_BUFFER;
	}
	memcpy(params[0].memref.buffer, &caps, sizeof(caps));
	params[0].memref.size = sizeof(caps);
	return TEE_SUCCESS;
}

static TEE_Result validate_prepare_params(
		uint32_t types, TEE_Param params[TEE_NUM_PARAMS],
		const struct pta_tnb_prepare **request_out,
		const uint64_t **commands_out)
{
	const uint32_t expected =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_MEMREF_INPUT);
	const struct pta_tnb_prepare *request = params[0].memref.buffer;
	const uint64_t *commands = params[1].memref.buffer;
	size_t command_blob_size = 0;

	if (types != expected || !request || !commands ||
	    !params[2].memref.buffer || !params[3].memref.buffer ||
	    params[0].memref.size != sizeof(*request) ||
	    request->command_count > PTA_TNB_MAX_COMMANDS)
		return TEE_ERROR_BAD_PARAMETERS;
	command_blob_size = (request->command_count +
			     PTA_TNB_COMMAND_TRAILER_COUNT) * sizeof(uint64_t);
	if (params[1].memref.size != command_blob_size ||
	    params[2].memref.size != request->input_size ||
	    params[3].memref.size != request->weight_size ||
	    !validate_template(request, commands) ||
	    ((request->reserved[0] & PTA_TNB_PREPARE_FLAG_NONSECURE_WEIGHT) &&
	     ((vaddr_t)params[3].memref.buffer & (TNB_PAGE_SIZE - 1))))
		return TEE_ERROR_BAD_PARAMETERS;
	*request_out = request;
	*commands_out = commands;
	return TEE_SUCCESS;
}

static bool validate_multi_request(const struct pta_tnb_prepare_multi *request,
				   const uint64_t *commands,
				   size_t command_blob_size)
{
	const uint32_t required_flags =
		PTA_TNB_PREPARE_FLAG_SECURE_PTE |
		PTA_TNB_PREPARE_FLAG_FULL_SECURITY |
		PTA_TNB_PREPARE_FLAG_UNCACHED_COMMAND |
		PTA_TNB_PREPARE_FLAG_NONSECURE_WEIGHT;
	size_t command_offset = 0;
	uint64_t weight_end = 0;
	uint64_t output_end = 0;
	size_t n = 0;

	if (!request || !commands ||
	    request->abi_version != PTA_TNB_ABI_VERSION ||
	    !request->template_id ||
	    request->core_count != PTA_TNB_MAX_CORES ||
	    !request->input_size ||
	    request->input_size > PTA_TNB_MAX_BATCH_INPUT_SIZE ||
	    request->weight_size != PTA_TNB_MAX_WEIGHT_SIZE ||
	    !request->output_size ||
	    request->output_size > PTA_TNB_MAX_BATCH_OUTPUT_SIZE ||
	    (request->flags & required_flags) != required_flags ||
	    (request->flags & ~(required_flags |
			       PTA_TNB_PREPARE_FLAG_TRACE_STEPS)) ||
	    !request->weight_cookie)
		return false;

	for (n = 0; n < PTA_TNB_MAX_CORES; n++) {
		const struct pta_tnb_segment *segment = &request->segment[n];
		size_t blob_size = 0;

		if (!segment->command_count ||
		    segment->command_count > PTA_TNB_MAX_COMMANDS)
			return false;
		blob_size = ((size_t)segment->command_count +
			     PTA_TNB_COMMAND_TRAILER_COUNT) * sizeof(uint64_t);

		if (command_offset > command_blob_size ||
		    blob_size > command_blob_size - command_offset ||
		    segment->interrupt_mask != 0x300 ||
		    segment->reserved || !segment->weight_size ||
		    (segment->weight_offset & (TNB_PAGE_SIZE - 1)) ||
		    (segment->weight_size & (TNB_PAGE_SIZE - 1)) ||
		    segment->weight_offset != weight_end ||
		    !segment->output_size ||
		    segment->output_offset != output_end ||
		    !validate_command_blob(segment->command_count,
					   segment->input_patch,
					   segment->weight_patch,
					   segment->output_patch,
					   commands + command_offset /
					   sizeof(uint64_t)))
			return false;
		command_offset += blob_size;
		weight_end += segment->weight_size;
		output_end += segment->output_size;
		if (weight_end > request->weight_size ||
		    output_end > request->output_size)
			return false;
	}

	return command_offset == command_blob_size &&
	       weight_end == request->weight_size &&
	       output_end == request->output_size;
}

static TEE_Result validate_prepare_multi_params(
		uint32_t types, TEE_Param params[TEE_NUM_PARAMS],
		const struct pta_tnb_prepare_multi **request_out,
		const uint64_t **commands_out)
{
	const uint32_t expected =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_MEMREF_INPUT);
	const struct pta_tnb_prepare_multi *request = params[0].memref.buffer;
	const uint64_t *commands = params[1].memref.buffer;

	if (types != expected || !request || !commands ||
	    !params[2].memref.buffer || !params[3].memref.buffer ||
	    params[0].memref.size != sizeof(*request) ||
	    params[2].memref.size != request->input_size ||
	    params[3].memref.size != request->weight_size ||
	    ((vaddr_t)params[3].memref.buffer & (TNB_PAGE_SIZE - 1)) ||
	    !validate_multi_request(request, commands,
				    params[1].memref.size))
		return TEE_ERROR_BAD_PARAMETERS;
	*request_out = request;
	*commands_out = commands;
	return TEE_SUCCESS;
}

static bool validate_ffn_slot_flags(uint32_t flags)
{
	const uint32_t required = PTA_TNB_PREPARE_FLAG_SECURE_PTE |
		PTA_TNB_PREPARE_FLAG_FULL_SECURITY |
		PTA_TNB_PREPARE_FLAG_UNCACHED_COMMAND |
		PTA_TNB_PREPARE_FLAG_NONSECURE_WEIGHT;

	return (flags & required) == required &&
	       !(flags & ~(required | PTA_TNB_PREPARE_FLAG_TRACE_STEPS));
}

static TEE_Result validate_ffn_slot_params(
		uint32_t types, TEE_Param params[TEE_NUM_PARAMS],
		const struct pta_tnb_ffn_slot **request_out)
{
	const uint32_t expected =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_NONE,
				TEE_PARAM_TYPE_NONE);
	const struct pta_tnb_ffn_slot *request = params[0].memref.buffer;

	if (types != expected || !request ||
	    params[0].memref.size != sizeof(*request) ||
	    request->abi_version != PTA_TNB_ABI_VERSION ||
	    !validate_ffn_slot_flags(request->flags) ||
	    request->weight_size != PTA_TNB_MAX_WEIGHT_SIZE ||
	    !request->weight_cookie ||
	    request->weight_slot >= PTA_TNB_FFN_WEIGHT_SLOT_COUNT ||
	    !zero_words(request->reserved, ARRAY_SIZE(request->reserved)) ||
	    !params[1].memref.buffer ||
	    params[1].memref.size != request->weight_size ||
	    ((vaddr_t)params[1].memref.buffer & (TNB_PAGE_SIZE - 1)))
		return TEE_ERROR_BAD_PARAMETERS;
	*request_out = request;
	return TEE_SUCCESS;
}

static TEE_Result validate_prepare_multi_slot_params(
		uint32_t types, TEE_Param params[TEE_NUM_PARAMS],
		const struct pta_tnb_prepare_multi_slot **request_out,
		const uint64_t **commands_out)
{
	const uint32_t expected =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_NONE);
	const struct pta_tnb_prepare_multi_slot *request =
		params[0].memref.buffer;
	const uint64_t *commands = params[1].memref.buffer;

	if (types != expected || !request || !commands ||
	    params[0].memref.size != sizeof(*request) ||
	    params[2].memref.size != request->prepare.input_size ||
	    request->weight_slot >= PTA_TNB_FFN_WEIGHT_SLOT_COUNT ||
	    !zero_words(request->reserved, ARRAY_SIZE(request->reserved)) ||
	    !validate_multi_request(&request->prepare, commands,
			params[1].memref.size))
		return TEE_ERROR_BAD_PARAMETERS;
	*request_out = request;
	*commands_out = commands;
	return TEE_SUCCESS;
}

static bool load_job_multi_slot(
		const struct pta_tnb_prepare_multi_slot *request,
		const uint64_t *commands, void *input)
{
	uint64_t weight_end = 0;
	uint64_t output_end = 0;
	size_t command_offset = 0;
	size_t n = 0;

	if (!dev.ffn_weight_slots || request->weight_slot >=
		PTA_TNB_FFN_WEIGHT_SLOT_COUNT ||
		request->prepare.weight_cookie !=
		dev.ffn_weight_cookie[request->weight_slot] ||
		request->prepare.weight_size != PTA_TNB_MAX_WEIGHT_SIZE)
		return false;
	dev.core_count = PTA_TNB_MAX_CORES;
	dev.multi_core = true;
	dev.nonsecure_weight = true;
	dev.pte_flags =
		(request->prepare.flags & PTA_TNB_PREPARE_FLAG_SECURE_PTE) ?
		TNB_PTE_SECURE : 0;
	if (!ensure_ffn_io_page_tables(request->prepare.input_size,
				       request->prepare.output_size))
		return false;
	dev.input_size = request->prepare.input_size;
	dev.weight_size = request->prepare.weight_size;
	dev.output_size = request->prepare.output_size;
	dev.template_id = request->prepare.template_id;
	dev.weight_cookie = request->prepare.weight_cookie;
	dev.weight_buffer = NULL;
	dev.weight_mapping_valid = true;
	memset(dev.debug, 0, sizeof(dev.debug));

	for (n = 0; n < dev.core_count; ++n) {
		const struct pta_tnb_segment *segment =
			&request->prepare.segment[n];
		size_t blob_size = ((size_t)segment->command_count +
			PTA_TNB_COMMAND_TRAILER_COUNT) * sizeof(uint64_t);
		uint64_t *secure_commands = command_buffer(n);

		memcpy(secure_commands,
		       (const uint8_t *)commands + command_offset, blob_size);
		secure_commands[segment->input_patch] =
			TNB_COMMAND_WITH_VALUE(secure_commands[segment->input_patch],
				TNB_IOVA_INPUT);
		secure_commands[segment->weight_patch] =
			TNB_COMMAND_WITH_VALUE(secure_commands[segment->weight_patch],
				TNB_IOVA_WEIGHT_SLOT(request->weight_slot) +
				segment->weight_offset);
		secure_commands[segment->output_patch] =
			TNB_COMMAND_WITH_VALUE(secure_commands[segment->output_patch],
				TNB_IOVA_OUTPUT + segment->output_offset);
		dev.command_count[n] = segment->command_count;
		dev.interrupt_mask[n] = segment->interrupt_mask;
		command_offset += blob_size;
		weight_end += segment->weight_size;
		output_end += segment->output_size;
	}
	if (!input || !request->prepare.input_size ||
	    command_offset > PTA_TNB_MAX_CORES * TNB_PAGE_SIZE)
		return false;
	if (weight_end != request->prepare.weight_size ||
	    output_end != request->prepare.output_size)
		return false;
	memcpy(input_pages, input, request->prepare.input_size);
	dev.debug[PTA_TNB_DEBUG_FLAGS] = request->prepare.flags;
	dev.debug[PTA_TNB_DEBUG_CMD_ADDR] = TNB_IOVA_CMD_CORE(0);
	dev.debug[PTA_TNB_DEBUG_INPUT_ADDR] = TNB_IOVA_INPUT;
	dev.debug[PTA_TNB_DEBUG_WEIGHT_ADDR] =
		TNB_IOVA_WEIGHT_SLOT(request->weight_slot);
	dev.debug[PTA_TNB_DEBUG_OUTPUT_ADDR] = TNB_IOVA_OUTPUT;
	dev.debug[PTA_TNB_DEBUG_DT_ADDR] = page_address(directory_table);
	dev.debug[PTA_TNB_DEBUG_PATCH_INPUT] = TNB_COMMAND_VALUE(
		command_buffer(0)[request->prepare.segment[0].input_patch]);
	dev.debug[PTA_TNB_DEBUG_PATCH_WEIGHT] = TNB_COMMAND_VALUE(
		command_buffer(0)[request->prepare.segment[0].weight_patch]);
	dev.debug[PTA_TNB_DEBUG_PATCH_OUTPUT] = TNB_COMMAND_VALUE(
		command_buffer(0)[request->prepare.segment[0].output_patch]);
	if (!dev.debug[PTA_TNB_DEBUG_DT_ADDR])
		return false;
	if (!(request->prepare.flags & PTA_TNB_PREPARE_FLAG_UNCACHED_COMMAND))
		dcache_clean_range(cached_command_page,
				   sizeof(cached_command_page));
	dcache_clean_range(input_pages, request->prepare.input_size);
	dsb();
	dev.ffn_job_loaded = true;
	return true;
}

static bool zap_iommu_caches(void)
{
	size_t n = 0;

	for (n = 0; n < 4; n++) {
		vaddr_t iommu = iommu_map(n);

		if (!iommu)
			return false;
		io_write32(iommu + IOMMU_COMMAND, IOMMU_CMD_ZAP_CACHE);
	}
	dsb();
	return !mmu_fault();
}

static bool stall_iommu_for_weight_remap(void)
{
	size_t n = 0;

	for (n = 0; n < 4; n++) {
		vaddr_t iommu = iommu_map(n);
		uint32_t status = iommu ? io_read32(iommu + IOMMU_STATUS) : 0;

		if (!iommu || !(status & IOMMU_STATUS_PAGING) ||
		    (status & IOMMU_STATUS_FAULT))
			return false;
	}
	for (n = 0; n < 4; n++)
		io_write32(iommu_map(n) + IOMMU_COMMAND,
			   IOMMU_CMD_ENABLE_STALL);
	dsb();
	if (!wait_iommu_status(IOMMU_STATUS_STALL, true) || mmu_fault())
		return false;
	for (n = 0; n < 4; n++)
		if (!(io_read32(iommu_map(n) + IOMMU_STATUS) &
		      IOMMU_STATUS_PAGING))
			return false;
	return true;
}

static bool unstall_iommu_after_weight_remap(void)
{
	size_t n = 0;

	for (n = 0; n < 4; n++)
		io_write32(iommu_map(n) + IOMMU_COMMAND,
			   IOMMU_CMD_DISABLE_STALL);
	dsb();
	if (!wait_iommu_status(IOMMU_STATUS_STALL, false) || mmu_fault())
		return false;
	for (n = 0; n < 4; n++)
		if (!(io_read32(iommu_map(n) + IOMMU_STATUS) &
		      IOMMU_STATUS_PAGING))
			return false;
	return true;
}

static bool remap_weight_while_active(void *weight, size_t size, bool secure)
{
	size_t weight_dte = dte_index(TNB_IOVA_WEIGHT);
	uint64_t profile_start = profile_now();
	bool ok = false;
	size_t n = 0;

	if (!secure_page(directory_table) ||
	    !prepare_weight_page_tables(weight, size, secure))
		goto out;
	for (n = 0; n < 3; n++) {
		paddr_t live_pa = virt_to_phys(page_tables[2 + n]);

		if (!secure_page(page_tables[2 + n]) || !live_pa ||
		    desc_address_v2(directory_table[weight_dte + n]) != live_pa)
			goto out;
	}
	if (!stall_iommu_for_weight_remap())
		goto out;
	for (n = 0; n < 3; n++) {
		memcpy(page_tables[2 + n], alternate_weight_page_tables[n],
		       sizeof(page_tables[2 + n]));
		dcache_clean_range(page_tables[2 + n],
				   sizeof(page_tables[2 + n]));
	}
	dsb();
	if (!zap_iommu_caches() || !unstall_iommu_after_weight_remap())
		goto out;
	for (n = 0; n < 3; n++) {
		paddr_t live_pa = virt_to_phys(page_tables[2 + n]);

		if (desc_address_v2(directory_table[weight_dte + n]) != live_pa)
			goto out;
	}
	dev.debug[PTA_TNB_DEBUG_PTE_WEIGHT] =
		page_tables[2][pte_index(TNB_IOVA_WEIGHT)];
	ok = true;
out:
	profile_add(&profile.weight_remap_calls, &profile.weight_remap_ticks,
		    profile_start);
	return ok;
}

static bool load_job(const struct pta_tnb_prepare *request,
			     const uint64_t *commands, TEE_Param params[TEE_NUM_PARAMS],
			     bool refresh_iommu)
{
	uint64_t *secure_commands = NULL;
	bool reuse_weight = refresh_iommu && dev.weight_mapping_valid &&
		dev.nonsecure_weight &&
		(request->reserved[0] & PTA_TNB_PREPARE_FLAG_NONSECURE_WEIGHT) &&
		!dev.multi_core &&
		dev.weight_size == request->weight_size &&
		dev.weight_buffer == params[3].memref.buffer;

	if (refresh_iommu) {
		memset(command_buffer(0), 0, TNB_PAGE_SIZE);
		memset(input_pages, 0, sizeof(input_pages));
		memset(output_pages, 0, sizeof(output_pages));
	} else {
		clear_work();
	}
	secure_commands = command_buffer(0);
	dev.core_count = 1;
	dev.multi_core = false;
	dev.nonsecure_weight =
		dev.prepare_flags & PTA_TNB_PREPARE_FLAG_NONSECURE_WEIGHT;
	dev.pte_flags = (dev.prepare_flags &
			 PTA_TNB_PREPARE_FLAG_SECURE_PTE) ? TNB_PTE_SECURE : 0;
	dev.input_size = request->input_size;
	dev.weight_size = request->weight_size;
	dev.output_size = request->output_size;
	memset(dev.debug, 0, sizeof(dev.debug));
	memcpy(secure_commands, commands, params[1].memref.size);
	memcpy(input_pages, params[2].memref.buffer, request->input_size);
	if (dev.nonsecure_weight) {
		if (!reuse_weight)
			dcache_clean_range(params[3].memref.buffer,
					   params[3].memref.size);
	} else {
		memcpy(weight_page, params[3].memref.buffer,
		       request->weight_size);
	}
	dev.debug[PTA_TNB_DEBUG_FLAGS] = dev.prepare_flags;
	dev.debug[PTA_TNB_DEBUG_CMD_ADDR] =
		(dev.prepare_flags & PTA_TNB_PREPARE_FLAG_DIRECT_PHYS) ?
		page_address(secure_commands) : TNB_IOVA_CMD_CORE(0);
	dev.debug[PTA_TNB_DEBUG_INPUT_ADDR] =
		(dev.prepare_flags & PTA_TNB_PREPARE_FLAG_DIRECT_PHYS) ?
		page_address(input_pages) : TNB_IOVA_INPUT;
	dev.debug[PTA_TNB_DEBUG_WEIGHT_ADDR] =
		(dev.prepare_flags & PTA_TNB_PREPARE_FLAG_DIRECT_PHYS) ?
		page_address(weight_page) : TNB_IOVA_WEIGHT;
	dev.debug[PTA_TNB_DEBUG_OUTPUT_ADDR] =
		(dev.prepare_flags & PTA_TNB_PREPARE_FLAG_DIRECT_PHYS) ?
		page_address(output_pages) : TNB_IOVA_OUTPUT;
	dev.debug[PTA_TNB_DEBUG_DT_ADDR] = page_address(directory_table);
	if (!dev.debug[PTA_TNB_DEBUG_CMD_ADDR] ||
	    !dev.debug[PTA_TNB_DEBUG_INPUT_ADDR] ||
	    !dev.debug[PTA_TNB_DEBUG_WEIGHT_ADDR] ||
	    !dev.debug[PTA_TNB_DEBUG_OUTPUT_ADDR] ||
	    !dev.debug[PTA_TNB_DEBUG_DT_ADDR])
		return false;
	secure_commands[request->input_patch] =
		TNB_COMMAND_WITH_VALUE(secure_commands[request->input_patch],
				       dev.debug[PTA_TNB_DEBUG_INPUT_ADDR]);
	secure_commands[request->weight_patch] =
		TNB_COMMAND_WITH_VALUE(secure_commands[request->weight_patch],
				       dev.debug[PTA_TNB_DEBUG_WEIGHT_ADDR]);
	secure_commands[request->output_patch] =
		TNB_COMMAND_WITH_VALUE(secure_commands[request->output_patch],
				       dev.debug[PTA_TNB_DEBUG_OUTPUT_ADDR]);
	dev.debug[PTA_TNB_DEBUG_PATCH_INPUT] =
		TNB_COMMAND_VALUE(secure_commands[request->input_patch]);
	dev.debug[PTA_TNB_DEBUG_PATCH_WEIGHT] =
		TNB_COMMAND_VALUE(secure_commands[request->weight_patch]);
	dev.debug[PTA_TNB_DEBUG_PATCH_OUTPUT] =
		TNB_COMMAND_VALUE(secure_commands[request->output_patch]);
	if (!refresh_iommu) {
		if (!prepare_page_tables(dev.nonsecure_weight ?
				 params[3].memref.buffer : weight_page))
			return false;
	} else if (dev.nonsecure_weight && !reuse_weight &&
		   !(dev.prepare_flags & PTA_TNB_PREPARE_FLAG_DIRECT_PHYS) &&
		   !remap_weight_while_active(params[3].memref.buffer,
					      request->weight_size, false)) {
		return false;
	}
	if (!(dev.prepare_flags & PTA_TNB_PREPARE_FLAG_UNCACHED_COMMAND))
		dcache_clean_range(cached_command_page,
				   sizeof(cached_command_page));
	dcache_clean_range(input_pages, sizeof(input_pages));
	if (!dev.nonsecure_weight)
		dcache_clean_range(weight_page, dev.weight_size);
	dcache_clean_range(output_pages, sizeof(output_pages));
	dsb();
	dev.command_count[0] = request->command_count;
	dev.interrupt_mask[0] = request->interrupt_mask;
	dev.template_id = request->template_id;
	dev.weight_buffer = params[3].memref.buffer;
	dev.weight_mapping_valid = true;
	return true;
}

static bool load_job_multi(const struct pta_tnb_prepare_multi *request,
			   const uint64_t *commands,
			   TEE_Param params[TEE_NUM_PARAMS], bool refresh_iommu)
{
	bool reuse_weight = refresh_iommu && dev.weight_mapping_valid &&
		dev.nonsecure_weight && dev.multi_core &&
		dev.weight_cookie == request->weight_cookie &&
		dev.weight_size == request->weight_size &&
		dev.weight_buffer == params[3].memref.buffer;
	size_t command_offset = 0;
	size_t n = 0;

	if (refresh_iommu) {
		memset(command_buffer(0), 0,
		       PTA_TNB_MAX_CORES * TNB_PAGE_SIZE);
		memset(input_pages, 0, sizeof(input_pages));
		memset(output_pages, 0, sizeof(output_pages));
	} else {
		clear_work();
	}

	dev.core_count = PTA_TNB_MAX_CORES;
	dev.multi_core = true;
	dev.nonsecure_weight = true;
	dev.pte_flags = TNB_PTE_SECURE;
	dev.input_size = request->input_size;
	dev.weight_size = request->weight_size;
	dev.output_size = request->output_size;
	memset(dev.debug, 0, sizeof(dev.debug));

	for (n = 0; n < dev.core_count; n++) {
		const struct pta_tnb_segment *segment = &request->segment[n];
		size_t blob_size = ((size_t)segment->command_count +
			PTA_TNB_COMMAND_TRAILER_COUNT) * sizeof(uint64_t);
		uint64_t *secure_commands = command_buffer(n);

		memcpy(secure_commands,
		       (const uint8_t *)commands + command_offset, blob_size);
		secure_commands[segment->input_patch] = TNB_COMMAND_WITH_VALUE(
			secure_commands[segment->input_patch], TNB_IOVA_INPUT);
		secure_commands[segment->weight_patch] = TNB_COMMAND_WITH_VALUE(
			secure_commands[segment->weight_patch],
			TNB_IOVA_WEIGHT + segment->weight_offset);
		secure_commands[segment->output_patch] = TNB_COMMAND_WITH_VALUE(
			secure_commands[segment->output_patch],
			TNB_IOVA_OUTPUT + segment->output_offset);
		dev.command_count[n] = segment->command_count;
		dev.interrupt_mask[n] = segment->interrupt_mask;
		command_offset += blob_size;
	}

	memcpy(input_pages, params[2].memref.buffer, request->input_size);
	if (!reuse_weight)
		dcache_clean_range(params[3].memref.buffer,
				   params[3].memref.size);
	dev.debug[PTA_TNB_DEBUG_FLAGS] = dev.prepare_flags;
	dev.debug[PTA_TNB_DEBUG_CMD_ADDR] = TNB_IOVA_CMD_CORE(0);
	dev.debug[PTA_TNB_DEBUG_INPUT_ADDR] = TNB_IOVA_INPUT;
	dev.debug[PTA_TNB_DEBUG_WEIGHT_ADDR] = TNB_IOVA_WEIGHT;
	dev.debug[PTA_TNB_DEBUG_OUTPUT_ADDR] = TNB_IOVA_OUTPUT;
	dev.debug[PTA_TNB_DEBUG_DT_ADDR] = page_address(directory_table);
	dev.debug[PTA_TNB_DEBUG_PATCH_INPUT] = TNB_COMMAND_VALUE(
		command_buffer(0)[request->segment[0].input_patch]);
	dev.debug[PTA_TNB_DEBUG_PATCH_WEIGHT] = TNB_COMMAND_VALUE(
		command_buffer(0)[request->segment[0].weight_patch]);
	dev.debug[PTA_TNB_DEBUG_PATCH_OUTPUT] = TNB_COMMAND_VALUE(
		command_buffer(0)[request->segment[0].output_patch]);
	if (!dev.debug[PTA_TNB_DEBUG_DT_ADDR])
		return false;
	if (!refresh_iommu) {
		if (!prepare_page_tables(params[3].memref.buffer))
			return false;
	} else if (!reuse_weight &&
		   !remap_weight_while_active(params[3].memref.buffer,
					      request->weight_size, false)) {
		return false;
	}
	if (!(dev.prepare_flags & PTA_TNB_PREPARE_FLAG_UNCACHED_COMMAND))
		dcache_clean_range(cached_command_page,
				   sizeof(cached_command_page));
	dcache_clean_range(input_pages, sizeof(input_pages));
	dcache_clean_range(output_pages, sizeof(output_pages));
	dsb();
	dev.template_id = request->template_id;
	dev.weight_cookie = request->weight_cookie;
	dev.weight_buffer = params[3].memref.buffer;
	dev.weight_mapping_valid = true;
	return true;
}

static TEE_Result prepare(void *session, uint32_t types,
			  TEE_Param params[TEE_NUM_PARAMS])
{
	const struct pta_tnb_prepare *request = NULL;
	const uint64_t *commands = NULL;
	TEE_Result res = TEE_SUCCESS;
	uint64_t profile_start = profile_now();

	res = validate_prepare_params(types, params, &request, &commands);
	if (res)
		return res;

	mutex_lock(&lock);
	if (dev.state == PTA_TNB_STATE_LOCKED) {
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (dev.state != PTA_TNB_STATE_READY) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	dev.prepare_flags = request->reserved[0];
	if ((dev.prepare_flags & PTA_TNB_PREPARE_FLAG_UNCACHED_COMMAND) &&
	    !init_uncached_command_page()) {
		res = TEE_ERROR_OUT_OF_MEMORY;
		goto out;
	}
	if ((dev.prepare_flags & PTA_TNB_PREPARE_FLAG_UNCACHED_COMMAND) &&
	    !set_command_page_uncached(false)) {
		res = TEE_ERROR_GENERIC;
		goto out;
	}
	if (!load_job(request, commands, params, false)) {
		clear_work();
		res = TEE_ERROR_GENERIC;
		goto out;
	}
	if ((dev.prepare_flags & PTA_TNB_PREPARE_FLAG_UNCACHED_COMMAND) &&
	    !set_command_page_uncached(true)) {
		res = TEE_ERROR_GENERIC;
		goto out;
	}
	dev.owner = session;
	dev.stage = PTA_TNB_STAGE_PREPARED;
	dev.fault = PTA_TNB_FAULT_NONE;
	dev.state = PTA_TNB_STATE_PREPARED;
out:
	if (res && dev.state == PTA_TNB_STATE_READY)
		dev.nonsecure_weight = false;
	profile_add(&profile.prepare_calls, &profile.prepare_ticks,
		    profile_start);
	mutex_unlock(&lock);
	return res;
}

static TEE_Result reprepare(void *session, uint32_t types,
			    TEE_Param params[TEE_NUM_PARAMS])
{
	const struct pta_tnb_prepare *request = NULL;
	const uint64_t *commands = NULL;
	TEE_Result res = TEE_SUCCESS;
	uint64_t profile_start = profile_now();

	res = validate_prepare_params(types, params, &request, &commands);
	if (res)
		return res;
	mutex_lock(&lock);
	if (dev.state == PTA_TNB_STATE_LOCKED) {
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (dev.owner != session) {
		res = TEE_ERROR_ACCESS_DENIED;
		goto out;
	}
	if (dev.state != PTA_TNB_STATE_ACTIVE || dev.multi_core) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	if (request->reserved[0] != dev.prepare_flags) {
		res = TEE_ERROR_BAD_PARAMETERS;
		goto out;
	}
	if ((dev.prepare_flags & PTA_TNB_PREPARE_FLAG_UNCACHED_COMMAND) &&
	    !set_command_page_uncached(false))
		goto lock;
	if (!load_job(request, commands, params, true))
		goto lock;
	if ((dev.prepare_flags & PTA_TNB_PREPARE_FLAG_UNCACHED_COMMAND) &&
	    !set_command_page_uncached(true))
		goto lock;
	dev.stage = PTA_TNB_STAGE_IOMMU;
	goto out;
lock:
	enter_locked(PTA_TNB_FAULT_RESTORE);
	res = TEE_ERROR_SECURITY;
out:
	profile_add(&profile.reprepare_calls, &profile.reprepare_ticks,
		    profile_start);
	mutex_unlock(&lock);
	return res;
}

static TEE_Result prepare_multi(void *session, uint32_t types,
				TEE_Param params[TEE_NUM_PARAMS])
{
	const struct pta_tnb_prepare_multi *request = NULL;
	const uint64_t *commands = NULL;
	TEE_Result res = validate_prepare_multi_params(types, params, &request,
						       &commands);
	uint64_t profile_start = profile_now();

	if (res)
		return res;
	mutex_lock(&lock);
	if (dev.state == PTA_TNB_STATE_LOCKED) {
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (dev.state != PTA_TNB_STATE_READY) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	dev.prepare_flags = request->flags;
	if (!init_uncached_command_page()) {
		res = TEE_ERROR_OUT_OF_MEMORY;
		goto out;
	}
	if (!set_command_page_uncached(false) ||
	    !load_job_multi(request, commands, params, false) ||
	    !set_command_page_uncached(true)) {
		clear_work();
		res = TEE_ERROR_GENERIC;
		goto out;
	}
	dev.owner = session;
	dev.stage = PTA_TNB_STAGE_PREPARED;
	dev.fault = PTA_TNB_FAULT_NONE;
	dev.state = PTA_TNB_STATE_PREPARED;
out:
	profile_add(&profile.prepare_calls, &profile.prepare_ticks,
		    profile_start);
	mutex_unlock(&lock);
	return res;
}

static TEE_Result prepare_ffn_slot_mobj(
	void *session, const struct pta_tnb_ffn_slot *request,
	struct mobj *weight_mobj, size_t weight_mobj_offset,
	void *weight_buffer, bool update_cache)
{
	struct tnb_session *tnb_session = session;
	struct tnb_cached_weight *cached = NULL;
	TEE_Result res = TEE_SUCCESS;
	uint64_t profile_start = profile_now();
	uint64_t weight_signature = 0;
	size_t cache_index = 0;
	size_t n = 0;

	if (!tnb_session || !request || !weight_mobj ||
	    request->weight_cookie > TNB_FFN_WEIGHT_CACHE_COUNT)
		return TEE_ERROR_BAD_PARAMETERS;
	mutex_lock(&lock);
	if (dev.state == PTA_TNB_STATE_LOCKED) {
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (request->weight_slot == PTA_TNB_FFN_WEIGHT_SLOT_GATE) {
		if (dev.state != PTA_TNB_STATE_READY) {
			res = TEE_ERROR_BAD_STATE;
			goto out;
		}
		dev.prepare_flags = request->flags;
		dev.pte_flags = request->flags &
			PTA_TNB_PREPARE_FLAG_SECURE_PTE ? TNB_PTE_SECURE : 0;
		dev.core_count = PTA_TNB_MAX_CORES;
		dev.multi_core = true;
		dev.nonsecure_weight = true;
		dev.input_size = PTA_TNB_MAX_BATCH_INPUT_SIZE;
		dev.weight_size = PTA_TNB_MAX_WEIGHT_SIZE;
		dev.output_size = PTA_TNB_MAX_BATCH_OUTPUT_SIZE;
		if (!init_uncached_command_page() ||
		    !set_command_page_uncached(false) ||
		    !prepare_ffn_page_tables()) {
			res = TEE_ERROR_GENERIC;
			goto reset;
		}
		dev.owner = session;
		dev.stage = PTA_TNB_STAGE_PREPARED;
		dev.fault = PTA_TNB_FAULT_NONE;
		dev.state = PTA_TNB_STATE_PREPARED;
	} else if (dev.owner != session ||
		   dev.state != PTA_TNB_STATE_PREPARED ||
		   dev.prepare_flags != request->flags ||
		   dev.ffn_weight_slots || dev.ffn_job_loaded) {
		res = dev.owner == session ? TEE_ERROR_BAD_STATE :
			TEE_ERROR_ACCESS_DENIED;
		goto out;
	}
	if (dev.ffn_weight_slot_mask != BIT(request->weight_slot) - 1) {
		res = TEE_ERROR_BAD_STATE;
		goto reset;
	}
	for (n = 0; n < request->weight_slot; ++n) {
		if (dev.ffn_weight_cookie[n] == request->weight_cookie) {
			res = TEE_ERROR_BAD_PARAMETERS;
			goto reset;
		}
	}
	cache_index = request->weight_cookie - 1;
	cached = &tnb_session->ffn_weight[cache_index];
	if (update_cache) {
		if (!prepare_ffn_weight_slot(request->weight_slot,
					     request->weight_cookie, weight_mobj,
					     weight_mobj_offset,
					     request->weight_size,
					     &weight_signature)) {
			res = TEE_ERROR_GENERIC;
			goto reset;
		}
		if (cached->mobj != weight_mobj ||
		    cached->mobj_offset != weight_mobj_offset ||
		    cached->signature != weight_signature) {
			struct mobj *new_mobj = mobj_get(weight_mobj);

			if (!new_mobj) {
				res = TEE_ERROR_OUT_OF_MEMORY;
				goto reset;
			}
			if (!cache_ffn_weight_slot(cached,
						   request->weight_slot)) {
				mobj_put(new_mobj);
				res = TEE_ERROR_GENERIC;
				goto reset;
			}
			mobj_put(cached->mobj);
			cached->mobj = new_mobj;
			cached->mobj_offset = weight_mobj_offset;
			cached->signature = weight_signature;
			if (weight_buffer)
				dcache_clean_range(weight_buffer,
						   request->weight_size);
			profile.slot_prepare_cold_calls++;
		} else {
			if (!cached->pte_valid &&
			    !cache_ffn_weight_slot(cached,
						   request->weight_slot)) {
				res = TEE_ERROR_GENERIC;
				goto reset;
			}
			profile.slot_prepare_cached_calls++;
		}
	} else if (cached->mobj != weight_mobj ||
		   cached->mobj_offset != weight_mobj_offset ||
		   !cached->pte_valid) {
		res = TEE_ERROR_SECURITY;
		goto reset;
	} else {
		if (!install_cached_ffn_weight_slot(cached,
						    request->weight_slot)) {
			res = TEE_ERROR_GENERIC;
			goto reset;
		}
		profile.slot_prepare_cached_calls++;
	}
	dsb();
	dev.ffn_weight_cookie[request->weight_slot] = request->weight_cookie;
	dev.ffn_weight_slot_mask |= BIT(request->weight_slot);
	dev.ffn_weight_slots = dev.ffn_weight_slot_mask ==
		BIT(PTA_TNB_FFN_WEIGHT_SLOT_COUNT) - 1;
	dev.ffn_job_loaded = false;
	goto out;
reset:
	clear_work();
	memset(&dev, 0, sizeof(dev));
out:
	profile_add(&profile.slot_prepare_calls, &profile.slot_prepare_ticks,
		    profile_start);
	mutex_unlock(&lock);
	return res;
}

static TEE_Result prepare_ffn_slot(void *session, uint32_t types,
				   TEE_Param params[TEE_NUM_PARAMS])
{
	const struct pta_tnb_ffn_slot *request = NULL;
	struct mobj *weight_mobj = NULL;
	TEE_Result res = validate_ffn_slot_params(types, params, &request);
	size_t weight_mobj_offset = 0;

	if (res)
		return res;
	if (!caller_buffer_mobj(params[1].memref.buffer, request->weight_size,
				&weight_mobj, &weight_mobj_offset))
		return TEE_ERROR_BAD_PARAMETERS;
	return prepare_ffn_slot_mobj(session, request, weight_mobj,
				     weight_mobj_offset,
				     params[1].memref.buffer, true);
}

static TEE_Result prepare_ffn_slot_set(void *session, uint32_t types,
				       TEE_Param params[TEE_NUM_PARAMS])
{
	const uint32_t expected =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_MEMREF_INPUT);
	const struct pta_tnb_ffn_slot_set *request = params[0].memref.buffer;
	TEE_Param slot_params[TEE_NUM_PARAMS] = { };
	struct pta_tnb_ffn_slot slot = { };
	TEE_Result res = TEE_SUCCESS;
	uint64_t profile_start = profile_now();
	size_t n = 0;
	size_t m = 0;

	if (types != expected || !request ||
	    params[0].memref.size != sizeof(*request) ||
	    request->abi_version != PTA_TNB_ABI_VERSION ||
	    !validate_ffn_slot_flags(request->flags) ||
	    request->weight_size != PTA_TNB_MAX_WEIGHT_SIZE ||
	    request->reserved ||
	    !zero_words(request->reserved2, ARRAY_SIZE(request->reserved2)))
		return TEE_ERROR_BAD_PARAMETERS;
	for (n = 0; n < PTA_TNB_FFN_WEIGHT_SLOT_COUNT; ++n) {
		if (!request->weight_cookie[n] || !params[n + 1].memref.buffer ||
		    params[n + 1].memref.size != request->weight_size ||
		    ((vaddr_t)params[n + 1].memref.buffer &
		     (TNB_PAGE_SIZE - 1)))
			return TEE_ERROR_BAD_PARAMETERS;
		for (m = 0; m < n; ++m)
			if (request->weight_cookie[m] == request->weight_cookie[n])
				return TEE_ERROR_BAD_PARAMETERS;
	}

	for (n = 0; n < PTA_TNB_FFN_WEIGHT_SLOT_COUNT; ++n) {
		memset(&slot, 0, sizeof(slot));
		memset(slot_params, 0, sizeof(slot_params));
		slot.abi_version = PTA_TNB_ABI_VERSION;
		slot.flags = request->flags;
		slot.weight_size = request->weight_size;
		slot.weight_cookie = request->weight_cookie[n];
		slot.weight_slot = n;
		slot_params[0].memref.buffer = &slot;
		slot_params[0].memref.size = sizeof(slot);
		slot_params[1] = params[n + 1];
		res = prepare_ffn_slot(session,
			TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
					TEE_PARAM_TYPE_MEMREF_INPUT,
					TEE_PARAM_TYPE_NONE,
					TEE_PARAM_TYPE_NONE), slot_params);
		if (res)
			break;
	}
	if (res)
		(void)close_device(session,
			TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
					TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE));
	mutex_lock(&lock);
	profile_add(&profile.slot_set_calls, &profile.slot_set_ticks,
		    profile_start);
	mutex_unlock(&lock);
	return res;
}

static TEE_Result select_ffn_slot_set_impl(void *session, uint32_t types,
					   TEE_Param params[TEE_NUM_PARAMS],
					   bool fused)
{
	const uint32_t expected =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
				TEE_PARAM_TYPE_NONE);
	const struct pta_tnb_ffn_slot_set *request = params[0].memref.buffer;
	struct tnb_session *tnb_session = session;
	struct pta_tnb_ffn_slot slot = { };
	TEE_Result res = TEE_SUCCESS;
	uint64_t profile_start = profile_now();
	size_t cache_index = 0;
	size_t n = 0;
	size_t m = 0;

	if (types != expected || !tnb_session || !request ||
	    params[0].memref.size != sizeof(*request) ||
	    request->abi_version != PTA_TNB_ABI_VERSION ||
	    !validate_ffn_slot_flags(request->flags) ||
	    request->weight_size != PTA_TNB_MAX_WEIGHT_SIZE || request->reserved ||
	    !zero_words(request->reserved2, ARRAY_SIZE(request->reserved2)))
		return TEE_ERROR_BAD_PARAMETERS;
	for (n = 0; n < PTA_TNB_FFN_WEIGHT_SLOT_COUNT; ++n) {
		if (!request->weight_cookie[n] ||
		    request->weight_cookie[n] > TNB_FFN_WEIGHT_CACHE_COUNT)
			return TEE_ERROR_BAD_PARAMETERS;
		for (m = 0; m < n; ++m)
			if (request->weight_cookie[m] == request->weight_cookie[n])
				return TEE_ERROR_BAD_PARAMETERS;
		cache_index = request->weight_cookie[n] - 1;
		if (!tnb_session->ffn_weight[cache_index].mobj)
			return TEE_ERROR_BAD_STATE;
	}
	for (n = 0; n < PTA_TNB_FFN_WEIGHT_SLOT_COUNT; ++n) {
		struct tnb_cached_weight *cached = NULL;

		cache_index = request->weight_cookie[n] - 1;
		cached = &tnb_session->ffn_weight[cache_index];
		memset(&slot, 0, sizeof(slot));
		slot.abi_version = PTA_TNB_ABI_VERSION;
		slot.flags = request->flags;
		slot.weight_size = request->weight_size;
		slot.weight_cookie = request->weight_cookie[n];
		slot.weight_slot = n;
		res = prepare_ffn_slot_mobj(session, &slot, cached->mobj,
					     cached->mobj_offset, NULL, false);
		if (res)
			break;
	}
	if (res)
		(void)close_device(session,
			TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
					TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE));
	mutex_lock(&lock);
	if (fused)
		profile_add(&profile.fused_slot_select_calls,
			    &profile.fused_slot_select_ticks, profile_start);
	else
		profile_add(&profile.slot_select_calls,
			    &profile.slot_select_ticks, profile_start);
	mutex_unlock(&lock);
	return res;
}

static TEE_Result select_ffn_slot_set(void *session, uint32_t types,
				      TEE_Param params[TEE_NUM_PARAMS])
{
	return select_ffn_slot_set_impl(session, types, params, false);
}

static TEE_Result prepare_multi_slot(void *session, uint32_t types,
				     TEE_Param params[TEE_NUM_PARAMS])
{
	const struct pta_tnb_prepare_multi_slot *request = NULL;
	const uint64_t *commands = NULL;
	TEE_Result res = validate_prepare_multi_slot_params(
		types, params, &request, &commands);
	uint64_t profile_start = profile_now();

	if (res)
		return res;
	mutex_lock(&lock);
	if (dev.state == PTA_TNB_STATE_LOCKED) {
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (dev.owner != session) {
		res = TEE_ERROR_ACCESS_DENIED;
		goto out;
	}
	if (dev.state != PTA_TNB_STATE_PREPARED || !dev.ffn_weight_slots ||
	    dev.ffn_job_loaded || request->prepare.flags != dev.prepare_flags) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	if (!set_command_page_uncached(false) ||
	    !load_job_multi_slot(request, commands,
				 params[2].memref.buffer) ||
	    !set_command_page_uncached(true)) {
		clear_work();
		memset(&dev, 0, sizeof(dev));
		res = TEE_ERROR_GENERIC;
		goto out;
	}
	dev.stage = PTA_TNB_STAGE_PREPARED;
out:
	profile_add(&profile.slot_repatch_calls, &profile.slot_repatch_ticks,
		    profile_start);
	mutex_unlock(&lock);
	return res;
}

static TEE_Result reprepare_multi_slot(void *session, uint32_t types,
				       TEE_Param params[TEE_NUM_PARAMS])
{
	const struct pta_tnb_prepare_multi_slot *request = NULL;
	const uint64_t *commands = NULL;
	TEE_Result res = validate_prepare_multi_slot_params(
		types, params, &request, &commands);
	uint64_t profile_start = profile_now();

	if (res)
		return res;
	mutex_lock(&lock);
	if (dev.state == PTA_TNB_STATE_LOCKED) {
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (dev.owner != session) {
		res = TEE_ERROR_ACCESS_DENIED;
		goto out;
	}
	if (dev.state != PTA_TNB_STATE_ACTIVE || !dev.ffn_weight_slots ||
	    !dev.ffn_job_loaded || request->prepare.flags != dev.prepare_flags) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	if (!set_command_page_uncached(false) ||
	    !load_job_multi_slot(request, commands,
				 params[2].memref.buffer) ||
	    !set_command_page_uncached(true))
		goto lock_device;
	dev.stage = PTA_TNB_STAGE_IOMMU;
	goto out;
lock_device:
	enter_locked(PTA_TNB_FAULT_RESTORE);
	res = TEE_ERROR_SECURITY;
out:
	profile_add(&profile.slot_repatch_calls, &profile.slot_repatch_ticks,
		    profile_start);
	mutex_unlock(&lock);
	return res;
}

static TEE_Result reprepare_multi(void *session, uint32_t types,
				  TEE_Param params[TEE_NUM_PARAMS])
{
	const struct pta_tnb_prepare_multi *request = NULL;
	const uint64_t *commands = NULL;
	TEE_Result res = validate_prepare_multi_params(types, params, &request,
						       &commands);
	uint64_t profile_start = profile_now();

	if (res)
		return res;
	mutex_lock(&lock);
	if (dev.state == PTA_TNB_STATE_LOCKED) {
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (dev.owner != session) {
		res = TEE_ERROR_ACCESS_DENIED;
		goto out;
	}
	if (dev.state != PTA_TNB_STATE_ACTIVE || !dev.multi_core) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	if (request->flags != dev.prepare_flags) {
		res = TEE_ERROR_BAD_PARAMETERS;
		goto out;
	}
	if (!set_command_page_uncached(false))
		goto lock_device;
	if (!load_job_multi(request, commands, params, true))
		goto lock_device;
	if (!set_command_page_uncached(true))
		goto lock_device;
	dev.stage = PTA_TNB_STAGE_IOMMU;
	goto out;
lock_device:
	enter_locked(PTA_TNB_FAULT_RESTORE);
	res = TEE_ERROR_SECURITY;
out:
	profile_add(&profile.reprepare_calls, &profile.reprepare_ticks,
		    profile_start);
	mutex_unlock(&lock);
	return res;
}

static TEE_Result activate(void *session, uint32_t types)
{
	const uint32_t expected =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
				TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
	vaddr_t bussgrf = io_map(BUSSGRF_BASE, BUSSGRF_SIZE);
	vaddr_t firewall = io_map(FIREWALL_DDR_BASE, FIREWALL_DDR_SIZE);
	TEE_Result res = TEE_SUCCESS;
	uint64_t profile_start = profile_now();
	size_t n = 0;

	if (types != expected || !bussgrf || !firewall)
		return TEE_ERROR_BAD_PARAMETERS;
	mutex_lock(&lock);
	if (dev.state == PTA_TNB_STATE_LOCKED) {
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (dev.owner != session || dev.state != PTA_TNB_STATE_PREPARED ||
	    (dev.ffn_weight_slots && !dev.ffn_job_loaded) ||
	    snapshot(&dev.before)) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	for (n = 0; n < dev.core_count; n++) {
		vaddr_t npu = npu_map(n);

		if (!npu) {
			res = TEE_ERROR_GENERIC;
			goto out;
		}
		dev.old_npu_mask[n] = io_read32(npu + NPU_INT_MASK);
	}
	for (n = 0; n < dev.core_count; n++) {
		if (!disable_irq(n, &dev.irq_enabled[n])) {
			size_t restore_count = n + 1;

			res = TEE_ERROR_GENERIC;
			while (restore_count--)
				if (!restore_irq(restore_count,
						 dev.irq_enabled[restore_count]))
					res = TEE_ERROR_SECURITY;
			goto out;
		}
	}
	dev.stage = PTA_TNB_STAGE_IRQ;
	dev.security_touched = true;
	if (!set_secure(bussgrf, firewall,
			dev.prepare_flags & PTA_TNB_PREPARE_FLAG_FULL_SECURITY)) {
		enter_locked(PTA_TNB_FAULT_SECURE);
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	dev.debug[PTA_TNB_DEBUG_BUSSGRF] =
		io_read32(bussgrf + BUSSGRF_FIREWALL_CON8);
	dev.debug[PTA_TNB_DEBUG_DDR_MST11] =
		io_read32(firewall + FIREWALL_DDR_MST(11));
	dev.debug[PTA_TNB_DEBUG_DDR_MST20] =
		io_read32(firewall + FIREWALL_DDR_MST(20));
	dev.debug[PTA_TNB_DEBUG_DDR_MST2] =
		io_read32(firewall + FIREWALL_DDR_MST(2));
	dev.debug[PTA_TNB_DEBUG_DDR_MST32] =
		io_read32(firewall + FIREWALL_DDR_MST(32));
	dev.stage = PTA_TNB_STAGE_SECURE;
	if (!disable_iommu(dev.mmu) ||
	    (!(dev.prepare_flags & PTA_TNB_PREPARE_FLAG_DIRECT_PHYS) &&
	     !install_iommu())) {
		enter_locked(PTA_TNB_FAULT_IOMMU);
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	dev.stage = PTA_TNB_STAGE_IOMMU;
	dev.state = PTA_TNB_STATE_ACTIVE;
out:
	profile_add(&profile.activate_calls, &profile.activate_ticks,
		    profile_start);
	mutex_unlock(&lock);
	return res;
}

static uint32_t command_address(size_t core)
{
	if (dev.prepare_flags & PTA_TNB_PREPARE_FLAG_DIRECT_PHYS)
		return page_address(command_buffer(core));
	return TNB_IOVA_CMD_CORE(core);
}

static bool prepare_npu_core(size_t core)
{
	vaddr_t npu = npu_map(core);
	uint32_t address = command_address(core);

	if (!npu || !address || core >= dev.core_count)
		return false;
	io_write32(npu + NPU_PC_DATA_ADDR, 1);
	dsb();
	io_write32(npu + NPU_CNA_POINTER, 0xe + 0x10000000 * core);
	io_write32(npu + NPU_CORE_POINTER, 0xe + 0x10000000 * core);
	io_write32(npu + NPU_PC_DATA_ADDR, address);
	io_write32(npu + NPU_PC_DATA_AMOUNT,
		   (dev.command_count[core] + 4 + 1) / 2 - 1);
	io_write32(npu + NPU_INT_MASK, dev.interrupt_mask[core]);
	io_write32(npu + NPU_INT_CLEAR, NPU_INT_ALL);
	/* Match the RK3588 PC driver: one task, ping-pong disabled. */
	io_write32(npu + NPU_PC_TASK_CONTROL, 0x6001);
	io_write32(npu + NPU_PC_DMA_BASE_ADDR, 0);
	return true;
}

static bool trigger_npu_core(size_t core)
{
	vaddr_t npu = npu_map(core);

	if (!npu)
		return false;
	io_write32(npu + NPU_PC_OP_EN, 1);
	dsb();
	io_write32(npu + NPU_PC_OP_EN, 0);
	dsb();
	return true;
}

static TEE_Result run(void *session, uint32_t types,
			      TEE_Param params[TEE_NUM_PARAMS])
{
	const uint32_t expected =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
				TEE_PARAM_TYPE_MEMREF_OUTPUT,
				TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
	struct pta_tnb_status status = { };
	vaddr_t npu = npu_map(0);
	TEE_Result res = TEE_SUCCESS;
	uint64_t profile_start = profile_now();
	uint64_t poll_start = 0;
	size_t n = 0;

	if (types != expected || !npu)
		return TEE_ERROR_BAD_PARAMETERS;
	if (params[0].memref.size < dev.output_size ||
	    params[1].memref.size < sizeof(status))
		return TEE_ERROR_SHORT_BUFFER;

	mutex_lock(&lock);
	if (dev.state == PTA_TNB_STATE_LOCKED) {
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (dev.owner != session || dev.state != PTA_TNB_STATE_ACTIVE) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	dev.state = PTA_TNB_STATE_RUNNING;
	dev.npu_status = 0;
	dev.npu_raw_status = 0;
	dev.completed_mask = 0;
	dev.completed_trace_valid = false;
	memset(&dev.completed_trace, 0, sizeof(dev.completed_trace));
	memset(&dev.failure_trace, 0, sizeof(dev.failure_trace));
	memset(dev.iommu_status, 0, sizeof(dev.iommu_status));
	dev.pc_task_status = 0;
	dev.pc_data_addr = 0;
	dev.pc_data_amount = 0;
	dev.pc_task_control = 0;
	dev.debug[PTA_TNB_DEBUG_PTE_FLAGS] = dev.pte_flags;
	dev.debug[PTA_TNB_DEBUG_NPU_MASK] =
		io_read32(npu + NPU_INT_MASK);
	capture_npu_debug(npu, PTA_TNB_DEBUG_PRE_STATUS);
	capture_command_registers(npu, PTA_TNB_DEBUG_PRE_CMD_INPUT);
	for (n = 0; n < dev.core_count; n++)
		if (!prepare_npu_core(n)) {
			capture_failure_npu_debug();
			enter_locked(PTA_TNB_FAULT_NPU);
			res = TEE_ERROR_SECURITY;
			goto out;
		}
	if (dev.prepare_flags & PTA_TNB_PREPARE_FLAG_TRACE_STEPS) {
		dsb();
		dev.debug[PTA_TNB_DEBUG_AFTER_CONTROL] =
			io_read32(npu + NPU_PC_TASK_CONTROL);
		dev.debug[PTA_TNB_DEBUG_AFTER_CONTROL_TASK] =
			io_read32(npu + NPU_PC_TASK_STATUS);
	}
	dsb();
	for (n = 0; n < dev.core_count; n++)
		if (!trigger_npu_core(n)) {
			capture_failure_npu_debug();
			enter_locked(PTA_TNB_FAULT_NPU);
			res = TEE_ERROR_SECURITY;
			goto out;
		}
	if (dev.prepare_flags & PTA_TNB_PREPARE_FLAG_TRACE_STEPS) {
		dsb();
		dev.debug[PTA_TNB_DEBUG_AFTER_OP_ENABLE] =
			io_read32(npu + NPU_PC_TASK_CONTROL);
		dev.debug[PTA_TNB_DEBUG_AFTER_OP_ENABLE_TASK] =
			io_read32(npu + NPU_PC_TASK_STATUS);
	}
	dsb();
	dev.debug[PTA_TNB_DEBUG_OP_ENABLE_READBACK] =
		io_read32(npu + NPU_PC_OP_EN);
	dev.debug[PTA_TNB_DEBUG_CNA_POINTER] =
		io_read32(npu + NPU_CNA_POINTER);
	dev.debug[PTA_TNB_DEBUG_CORE_POINTER] =
		io_read32(npu + NPU_CORE_POINTER);
	capture_npu_debug(npu, PTA_TNB_DEBUG_POST_STATUS);
	dev.stage = PTA_TNB_STAGE_SUBMITTED;
	poll_start = profile_now();
	dev.fault = poll_npus();
	profile_add(&profile.poll_calls, &profile.poll_ticks, poll_start);
	if (!dev.fault) {
		dev.debug[PTA_TNB_DEBUG_CNA_POINTER] =
			io_read32(npu + NPU_CNA_POINTER);
		dev.debug[PTA_TNB_DEBUG_CORE_POINTER] =
			io_read32(npu + NPU_CORE_POINTER);
		capture_command_registers(npu, PTA_TNB_DEBUG_POST_CMD_INPUT);
		capture_output_debug();
	}
	if (dev.fault) {
		enter_locked(dev.fault);
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	dev.stage = PTA_TNB_STAGE_COMPLETED;
	capture_run_status(npu);
	if (mmu_fault()) {
		dev.fault = PTA_TNB_FAULT_IOMMU;
		capture_failure_npu_debug();
		enter_locked(PTA_TNB_FAULT_IOMMU);
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (dev.multi_core && dev.completed_trace_valid) {
		dev.completed_trace.sequence = ++last_success_sequence;
		last_success_trace = dev.completed_trace;
	}
	memcpy(params[0].memref.buffer, output_pages, dev.output_size);
	params[0].memref.size = dev.output_size;
	dev.state = PTA_TNB_STATE_ACTIVE;
out:
	profile_add(&profile.run_calls, &profile.run_ticks, profile_start);
	fill_status(&status);
	if (params[1].memref.size >= sizeof(status)) {
		memcpy(params[1].memref.buffer, &status, sizeof(status));
		params[1].memref.size = sizeof(status);
	}
	mutex_unlock(&lock);
	return res;
}

static TEE_Result close_device(void *session, uint32_t types)
{
	const uint32_t expected =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
				TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
	struct tnb_hw_snapshot after = { };
	vaddr_t bussgrf = io_map(BUSSGRF_BASE, BUSSGRF_SIZE);
	vaddr_t firewall = io_map(FIREWALL_DDR_BASE, FIREWALL_DDR_SIZE);
	TEE_Result res = TEE_SUCCESS;
	uint64_t profile_start = profile_now();
	bool restored = true;
	size_t n = 0;

	if (types != expected || !bussgrf || !firewall)
		return TEE_ERROR_BAD_PARAMETERS;
	mutex_lock(&lock);
	if (dev.state == PTA_TNB_STATE_LOCKED) {
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (dev.owner != session) {
		res = TEE_ERROR_ACCESS_DENIED;
		goto out;
	}
	if (dev.state == PTA_TNB_STATE_PREPARED) {
		if ((dev.prepare_flags & PTA_TNB_PREPARE_FLAG_UNCACHED_COMMAND) &&
		    !set_command_page_uncached(false)) {
			dev.state = PTA_TNB_STATE_LOCKED;
			dev.owner = NULL;
			dev.fault = PTA_TNB_FAULT_RESTORE;
			res = TEE_ERROR_SECURITY;
			goto out;
		}
		clear_work();
		memset(&dev, 0, sizeof(dev));
		goto out;
	}
	if (dev.state != PTA_TNB_STATE_ACTIVE) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	for (n = 0; n < dev.core_count; n++) {
		vaddr_t npu = npu_map(n);

		if (!npu)
			goto lock;
		io_write32(npu + NPU_INT_MASK, 0);
		io_write32(npu + NPU_INT_CLEAR, NPU_INT_ALL);
	}
	dsb();
	if (!restore_iommu(dev.mmu))
		goto lock;
	if ((dev.prepare_flags & PTA_TNB_PREPARE_FLAG_UNCACHED_COMMAND) &&
	    !set_command_page_uncached(false))
		goto lock;
	clear_work();
	for (n = 0; n < dev.core_count; n++) {
		vaddr_t npu = npu_map(n);

		if (!npu) {
			restored = false;
			continue;
		}
		io_write32(npu + NPU_INT_MASK, dev.old_npu_mask[n]);
	}
	if (!restored || !restore_secure(bussgrf, firewall, &dev.before))
		goto lock;
	for (n = 0; n < dev.core_count; n++)
		restored = restore_irq(n, dev.irq_enabled[n]) && restored;
	if (!restored || snapshot(&after) ||
	    !snapshot_equal(&dev.before, &after))
		goto lock;
	dev.security_touched = false;
	memset(&dev, 0, sizeof(dev));
	dev.stage = PTA_TNB_STAGE_RESTORED;
	goto out;
lock:
	enter_locked(PTA_TNB_FAULT_RESTORE);
	res = TEE_ERROR_SECURITY;
out:
	profile_add(&profile.close_calls, &profile.close_ticks, profile_start);
	mutex_unlock(&lock);
	return res;
}

static TEE_Result exec_multi_slot_chain(void *session, uint32_t types,
					TEE_Param params[TEE_NUM_PARAMS])
{
	const uint32_t one_output =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
				TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_MEMREF_OUTPUT,
				TEE_PARAM_TYPE_NONE);
	const uint32_t two_outputs =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
				TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_MEMREF_OUTPUT,
				TEE_PARAM_TYPE_MEMREF_OUTPUT);
	struct pta_tnb_multi_slot_chain *chain = params[0].memref.buffer;
	struct pta_tnb_status run_status = { };
	TEE_Param call_params[TEE_NUM_PARAMS] = { };
	TEE_Result res = TEE_SUCCESS;
	uint64_t profile_start = profile_now();
	uint64_t completed_rows = 0;
	size_t input_bytes = 0;
	size_t output_bytes = 0;
	size_t job_index = 0;
	size_t row = 0;
	bool first = true;
	bool select_slots = false;

	if ((types != one_output && types != two_outputs) || !chain ||
	    params[0].memref.size != sizeof(*chain) ||
	    chain->abi_version != PTA_TNB_ABI_VERSION ||
	    chain->flags & ~PTA_TNB_CHAIN_FLAGS_MASK ||
	    !chain->job_count || chain->job_count > PTA_TNB_CHAIN_MAX_JOBS ||
	    !chain->rows || chain->rows > PTA_TNB_CHAIN_MAX_ROWS ||
	    !zero_words(chain->reserved, ARRAY_SIZE(chain->reserved)) ||
	    !params[1].memref.buffer ||
	    !params[2].memref.buffer ||
	    (chain->job_count == 1 && types != one_output) ||
	    (chain->job_count == 2 &&
	     (types != two_outputs || !params[3].memref.buffer)))
		return TEE_ERROR_BAD_PARAMETERS;
	select_slots =
		(chain->flags & PTA_TNB_CHAIN_FLAG_SELECT_FFN_SLOT_SET) != 0;
	if ((select_slots &&
	     (!(chain->flags & PTA_TNB_CHAIN_FLAG_BEGIN) ||
	      (chain->flags & PTA_TNB_CHAIN_FLAG_END) ||
	      chain->job_count != 2)) ||
	    (!select_slots &&
	     !zero_words(chain->select_weight_cookie,
			 ARRAY_SIZE(chain->select_weight_cookie))))
		return TEE_ERROR_BAD_PARAMETERS;

	for (job_index = 0; job_index < chain->job_count; ++job_index) {
		const struct pta_tnb_chain_job *job = &chain->job[job_index];
		const struct pta_tnb_prepare_multi *prepare = &job->request.prepare;
		size_t command_bytes = 0;

		if (!job->command_words ||
		    job->command_words > PTA_TNB_MULTI_COMMAND_WORDS ||
		    !zero_words(job->reserved, ARRAY_SIZE(job->reserved)) ||
		    !zero_words(job->request.reserved,
				ARRAY_SIZE(job->request.reserved)))
			return TEE_ERROR_BAD_PARAMETERS;
		command_bytes = (size_t)job->command_words * sizeof(uint64_t);
		if (!validate_multi_request(prepare, job->commands, command_bytes) ||
		    job->request.weight_slot >= PTA_TNB_FFN_WEIGHT_SLOT_COUNT ||
		    (job_index && prepare->input_size !=
				 chain->job[0].request.prepare.input_size))
			return TEE_ERROR_BAD_PARAMETERS;
	}
	if (select_slots) {
		size_t n = 0;
		size_t m = 0;

		for (n = 0; n < PTA_TNB_FFN_WEIGHT_SLOT_COUNT; ++n) {
			if (!chain->select_weight_cookie[n] ||
			    chain->select_weight_cookie[n] >
				    TNB_FFN_WEIGHT_CACHE_COUNT)
				return TEE_ERROR_BAD_PARAMETERS;
			for (m = 0; m < n; ++m)
				if (chain->select_weight_cookie[m] ==
				    chain->select_weight_cookie[n])
					return TEE_ERROR_BAD_PARAMETERS;
		}
		for (job_index = 0; job_index < chain->job_count; ++job_index) {
			const struct pta_tnb_chain_job *job =
				&chain->job[job_index];

			if (job->request.prepare.weight_cookie !=
			    chain->select_weight_cookie[job->request.weight_slot])
				return TEE_ERROR_BAD_PARAMETERS;
		}
	}

	input_bytes = (size_t)chain->rows *
		chain->job[0].request.prepare.input_size;
	if (params[1].memref.size != input_bytes)
		return TEE_ERROR_BAD_PARAMETERS;
	for (job_index = 0; job_index < chain->job_count; ++job_index) {
		output_bytes = (size_t)chain->rows *
			chain->job[job_index].request.prepare.output_size;
		if (params[job_index + 2].memref.size != output_bytes)
			return TEE_ERROR_BAD_PARAMETERS;
	}

	chain->failed_job = UINT32_MAX;
	chain->failed_row = UINT32_MAX;
	chain->result = TEE_SUCCESS;
	memset(&chain->status, 0, sizeof(chain->status));
	if (select_slots) {
		struct pta_tnb_ffn_slot_set request = {
			.abi_version = PTA_TNB_ABI_VERSION,
			.flags = chain->job[0].request.prepare.flags,
			.weight_size = PTA_TNB_MAX_WEIGHT_SIZE,
		};

		memcpy(request.weight_cookie, chain->select_weight_cookie,
		       sizeof(request.weight_cookie));
		memset(call_params, 0, sizeof(call_params));
		call_params[0].memref.buffer = &request;
		call_params[0].memref.size = sizeof(request);
		res = select_ffn_slot_set_impl(session,
			TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
					TEE_PARAM_TYPE_NONE,
					TEE_PARAM_TYPE_NONE,
					TEE_PARAM_TYPE_NONE),
			call_params, true);
		if (res)
			goto select_fail;
	}
	for (job_index = 0; job_index < chain->job_count; ++job_index) {
		struct pta_tnb_chain_job *job = &chain->job[job_index];
		size_t command_bytes =
			(size_t)job->command_words * sizeof(uint64_t);
		uint8_t *output = params[job_index + 2].memref.buffer;

		for (row = 0; row < chain->rows; ++row) {
			memset(call_params, 0, sizeof(call_params));
			call_params[0].memref.buffer = &job->request;
			call_params[0].memref.size = sizeof(job->request);
			call_params[1].memref.buffer = job->commands;
			call_params[1].memref.size = command_bytes;
			call_params[2].memref.buffer =
				(uint8_t *)params[1].memref.buffer +
				row * job->request.prepare.input_size;
			call_params[2].memref.size =
				job->request.prepare.input_size;
			res = first && (chain->flags & PTA_TNB_CHAIN_FLAG_BEGIN) ?
				prepare_multi_slot(session,
					TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
							TEE_PARAM_TYPE_MEMREF_INPUT,
							TEE_PARAM_TYPE_MEMREF_INPUT,
							TEE_PARAM_TYPE_NONE), call_params) :
				reprepare_multi_slot(session,
					TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
							TEE_PARAM_TYPE_MEMREF_INPUT,
							TEE_PARAM_TYPE_MEMREF_INPUT,
							TEE_PARAM_TYPE_NONE), call_params);
			if (res)
				goto fail;
			if (first && (chain->flags & PTA_TNB_CHAIN_FLAG_BEGIN)) {
				res = activate(session,
					TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE,
							TEE_PARAM_TYPE_NONE,
							TEE_PARAM_TYPE_NONE,
							TEE_PARAM_TYPE_NONE));
				if (res)
					goto fail;
			}
			first = false;
			memset(call_params, 0, sizeof(call_params));
			call_params[0].memref.buffer = output +
				row * job->request.prepare.output_size;
			call_params[0].memref.size =
				job->request.prepare.output_size;
			call_params[1].memref.buffer = &run_status;
			call_params[1].memref.size = sizeof(run_status);
			res = run(session,
				TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
						TEE_PARAM_TYPE_MEMREF_OUTPUT,
						TEE_PARAM_TYPE_NONE,
						TEE_PARAM_TYPE_NONE), call_params);
			if (res)
				goto fail;
			completed_rows++;
		}
	}
	if (chain->flags & PTA_TNB_CHAIN_FLAG_END) {
		res = close_device(session,
			TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
					TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE));
		if (res)
			goto fail;
	}
	goto out;

select_fail:
	chain->failed_job = UINT32_MAX;
	chain->failed_row = UINT32_MAX;
	chain->result = res;
	goto out;
fail:
	chain->failed_job = job_index;
	chain->failed_row = row;
	chain->result = res;
	(void)close_device(session,
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
				TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE));
out:
	mutex_lock(&lock);
	profile.chain_jobs += chain->job_count;
	profile.chain_rows += completed_rows;
	profile_add(&profile.chain_calls, &profile.chain_ticks, profile_start);
	fill_status(&chain->status);
	mutex_unlock(&lock);
	return res;
}

static TEE_Result get_status(uint32_t types,
			     TEE_Param params[TEE_NUM_PARAMS])
{
	const uint32_t expected =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
				TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
				TEE_PARAM_TYPE_NONE);
	struct pta_tnb_status status = { };

	if (types != expected)
		return TEE_ERROR_BAD_PARAMETERS;
	if (params[0].memref.size < sizeof(status)) {
		params[0].memref.size = sizeof(status);
		return TEE_ERROR_SHORT_BUFFER;
	}
	mutex_lock(&lock);
	fill_status(&status);
	memcpy(params[0].memref.buffer, &status, sizeof(status));
	params[0].memref.size = sizeof(status);
	mutex_unlock(&lock);
	return TEE_SUCCESS;
}

static TEE_Result lock_device(void *session, uint32_t types)
{
	const uint32_t expected =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
				TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
	TEE_Result res = TEE_SUCCESS;

	if (types != expected)
		return TEE_ERROR_BAD_PARAMETERS;
	mutex_lock(&lock);
	if (dev.state == PTA_TNB_STATE_LOCKED)
		res = TEE_ERROR_SECURITY;
	else if (dev.owner != session)
		res = TEE_ERROR_ACCESS_DENIED;
	else if (dev.state != PTA_TNB_STATE_ACTIVE &&
		 dev.state != PTA_TNB_STATE_RUNNING)
		res = TEE_ERROR_BAD_STATE;
	else {
		enter_locked(PTA_TNB_FAULT_VERIFY);
		res = TEE_ERROR_SECURITY;
	}
	mutex_unlock(&lock);
	return res;
}

static TEE_Result open_session(uint32_t types, TEE_Param params[TEE_NUM_PARAMS],
			       void **session)
{
	const TEE_UUID tnb_ta = PTA_TNB_USER_TA_UUID;
	const TEE_UUID layer_ta = PTA_TNB_LAYER_TA_UUID;
	const uint32_t expected =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
				TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
	struct tee_ta_session *caller =
		to_ta_session(ts_get_current_session());

	(void)params;
	if (types != expected)
		return TEE_ERROR_BAD_PARAMETERS;
	if (caller->clnt_id.login != TEE_LOGIN_TRUSTED_APP ||
	    (memcmp(&caller->clnt_id.uuid, &tnb_ta, sizeof(tnb_ta)) &&
	     memcmp(&caller->clnt_id.uuid, &layer_ta, sizeof(layer_ta))))
		return TEE_ERROR_ACCESS_DENIED;
	mutex_lock(&lock);
	memset(&profile, 0, sizeof(profile));
	profile.version = PTA_TNB_PROFILE_VERSION;
	profile.counter_frequency = read_cntfrq();
	mutex_unlock(&lock);
	*session = calloc(1, sizeof(struct tnb_session));
	if (!*session)
		return TEE_ERROR_OUT_OF_MEMORY;
	return TEE_SUCCESS;
}

static void close_session(void *session)
{
	struct tnb_session *tnb_session = session;
	size_t n = 0;

	mutex_lock(&lock);
	if (dev.owner == session && dev.state == PTA_TNB_STATE_PREPARED) {
		if ((dev.prepare_flags & PTA_TNB_PREPARE_FLAG_UNCACHED_COMMAND) &&
		    !set_command_page_uncached(false)) {
			dev.state = PTA_TNB_STATE_LOCKED;
			dev.owner = NULL;
			dev.fault = PTA_TNB_FAULT_RESTORE;
		} else {
			clear_work();
			memset(&dev, 0, sizeof(dev));
		}
	} else if (dev.owner == session &&
		   (dev.state == PTA_TNB_STATE_ACTIVE ||
		    dev.state == PTA_TNB_STATE_RUNNING)) {
		enter_locked(PTA_TNB_FAULT_CALLER);
	}
	mutex_unlock(&lock);
	if (tnb_session)
		for (n = 0; n < TNB_FFN_WEIGHT_CACHE_COUNT; ++n)
			mobj_put(tnb_session->ffn_weight[n].mobj);
	if (tnb_session)
		memset(tnb_session, 0, sizeof(*tnb_session));
	free(session);
}

static TEE_Result invoke_command(void *session, uint32_t command,
				 uint32_t types,
				 TEE_Param params[TEE_NUM_PARAMS])
{
	switch (command) {
	case PTA_TNB_CMD_GET_CAPS:
		return get_caps(types, params);
	case PTA_TNB_CMD_PREPARE:
		return prepare(session, types, params);
	case PTA_TNB_CMD_ACTIVATE:
		return activate(session, types);
	case PTA_TNB_CMD_RUN:
		return run(session, types, params);
	case PTA_TNB_CMD_CLOSE:
		return close_device(session, types);
	case PTA_TNB_CMD_GET_STATUS:
		return get_status(types, params);
	case PTA_TNB_CMD_LOCK:
		return lock_device(session, types);
	case PTA_TNB_CMD_REPREPARE:
		return reprepare(session, types, params);
	case PTA_TNB_CMD_PREPARE_MULTI:
		return prepare_multi(session, types, params);
	case PTA_TNB_CMD_REPREPARE_MULTI:
		return reprepare_multi(session, types, params);
	case PTA_TNB_CMD_PREPARE_FFN_SLOT:
		return prepare_ffn_slot(session, types, params);
	case PTA_TNB_CMD_PREPARE_MULTI_SLOT:
		return prepare_multi_slot(session, types, params);
	case PTA_TNB_CMD_REPREPARE_MULTI_SLOT:
		return reprepare_multi_slot(session, types, params);
	case PTA_TNB_CMD_PREPARE_FFN_SLOT_SET:
		return prepare_ffn_slot_set(session, types, params);
	case PTA_TNB_CMD_EXEC_MULTI_SLOT_CHAIN:
		return exec_multi_slot_chain(session, types, params);
	case PTA_TNB_CMD_SELECT_FFN_SLOT_SET:
		return select_ffn_slot_set(session, types, params);
	default:
		return TEE_ERROR_NOT_SUPPORTED;
	}
}

pseudo_ta_register(.uuid = PTA_TNB_UUID, .name = "tnb",
		   .flags = PTA_DEFAULT_FLAGS,
		   .open_session_entry_point = open_session,
		   .close_session_entry_point = close_session,
		   .invoke_command_entry_point = invoke_command);
