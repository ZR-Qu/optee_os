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
#include <malloc.h>
#include <mm/core_memprot.h>
#include <mm/mobj.h>
#include <mm/page_alloc.h>
#include <mm/phys_mem.h>
#include <mm/vm.h>
#include <platform_config.h>
#include <pta_tnb.h>
#include <stdlib.h>
#include <string.h>
#include <trace.h>

#include "tnb_gemm.h"

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
#define IOMMU_ZAP_ONE_LINE 0x10
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

#define GICD_CTLR 0x0
#define GICD_CTLR_RWP BIT(31)
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
#define TNB_IOVA_A 0x11000000
#define TNB_IOVA_B 0x12000000
#define TNB_IOVA_B2 0x14000000
#define TNB_IOVA_B3 0x15000000
#define TNB_IOVA_C 0x13000000
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
#define TNB_TRAILER_COUNT 4
#define TNB_MAX_B_SIZE 12582912
#define TNB_PAGE_TABLE_COUNT 12

#define TNB_FLAG_FULL_SECURITY BIT(1)
#define TNB_FLAG_UNCACHED_COMMAND BIT(4)
#define TNB_FLAGS (TNB_FLAG_FULL_SECURITY | TNB_FLAG_UNCACHED_COMMAND)

#define TNB_COMMAND_WITH_VALUE(entry, value) \
	(((entry) & ~GENMASK_64(47, 16)) | ((uint64_t)(value) << 16))

#define TNB_A_REG 0x1070
#define TNB_B_REG 0x1110
#define TNB_C_REG 0x4020

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
	uint32_t a_size;
	uint32_t b_size;
	uint32_t c_size;
	uint32_t mode;
	size_t a_span;
	size_t c_span;
	struct pta_tnb_batch batch;
	bool batched;
	struct mobj *b_mobj[PTA_TNB_MAX_B_INPUTS];
	size_t b_span[PTA_TNB_MAX_B_INPUTS];
	uint32_t old_npu_mask[PTA_TNB_MAX_CORES];
	bool irq_enabled[PTA_TNB_MAX_CORES];
	bool irq_held;
	bool secure_held;
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
static uint8_t *a_pages;
static uint8_t *c_pages;
static tee_mm_entry_t *staging_mm;
static uint8_t *staging_pages;

#define TNB_STAGING_SIZE \
	(ROUNDUP(TNB_GEMM_M_MAX * TNB_GEMM_K_MAX, TNB_PAGE_SIZE) + \
	 ROUNDUP(TNB_GEMM_M_MAX * TNB_GEMM_N_MAX * sizeof(int32_t), TNB_PAGE_SIZE))
static uint32_t directory_table[TNB_DT_ENTRIES] __aligned(TNB_PAGE_SIZE);
static uint32_t page_tables[TNB_PAGE_TABLE_COUNT][TNB_PT_ENTRIES]
	__aligned(TNB_PAGE_SIZE);
#ifdef CFG_TNB_SECURITY_TEST
static uint8_t secure_ram_test_page[PTA_TNB_SECURE_RAM_TEST_PAGE_SIZE]
	__aligned(TNB_PAGE_SIZE);
static bool secure_ram_test_active;
static void *secure_ram_test_owner;
static const uint8_t secure_ram_test_pattern[TNB_PAGE_SIZE / 128] = {
	0x4c, 0x4c, 0x4d, 0x2d, 0x54, 0x5a, 0x2d, 0x53,
	0x45, 0x43, 0x55, 0x52, 0x45, 0x2d, 0x56, 0x31,
	0x91, 0x27, 0xa4, 0x6e, 0x53, 0xd8, 0x0b, 0xf2,
	0x6a, 0xc1, 0x39, 0x7d, 0xe5, 0x48, 0xb0, 0x16,
};
#endif
static struct tnb_dev dev;
static struct pta_tnb_profile profile;
static struct mutex lock = MUTEX_INITIALIZER;

static TEE_Result close_device(void *session, uint32_t types, bool keep_command,
			       bool keep_work, bool keep_irq, bool keep_secure);

#ifdef CFG_TNB_SECURITY_TEST
static void secure_ram_test_clear(void)
{
	memset(secure_ram_test_page, 0, PTA_TNB_SECURE_RAM_TEST_PAGE_SIZE);
	dcache_clean_range(secure_ram_test_page,
			   PTA_TNB_SECURE_RAM_TEST_PAGE_SIZE);
	secure_ram_test_active = false;
	secure_ram_test_owner = NULL;
}

static TEE_Result secure_ram_test_begin(void *session, uint32_t types,
					TEE_Param params[TEE_NUM_PARAMS])
{
	struct pta_tnb_secure_ram_test_info info = { 0 };
	paddr_t pa;
	size_t n;
	const uint32_t expected = TEE_PARAM_TYPES(
		TEE_PARAM_TYPE_MEMREF_OUTPUT, TEE_PARAM_TYPE_NONE,
		TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);

	if (types != expected || !params[0].memref.buffer)
		return TEE_ERROR_BAD_PARAMETERS;
	if (params[0].memref.size < sizeof(info)) {
		params[0].memref.size = sizeof(info);
		return TEE_ERROR_SHORT_BUFFER;
	}

	mutex_lock(&lock);
	if (secure_ram_test_active || dev.state != PTA_TNB_STATE_READY) {
		mutex_unlock(&lock);
		return TEE_ERROR_BAD_STATE;
	}
	pa = virt_to_phys(secure_ram_test_page);
	if (!pa || (pa & (TNB_PAGE_SIZE - 1)) ||
	    !core_pbuf_is(CORE_MEM_SEC, pa,
			   PTA_TNB_SECURE_RAM_TEST_PAGE_SIZE)) {
		mutex_unlock(&lock);
		return TEE_ERROR_SECURITY;
	}
	for (n = 0; n < PTA_TNB_SECURE_RAM_TEST_PAGE_SIZE; n++)
		secure_ram_test_page[n] = secure_ram_test_pattern[
		n % sizeof(secure_ram_test_pattern)];
	dcache_clean_range(secure_ram_test_page,
			   PTA_TNB_SECURE_RAM_TEST_PAGE_SIZE);
	secure_ram_test_active = true;
	secure_ram_test_owner = session;
	info.version = PTA_TNB_SECURE_RAM_TEST_VERSION;
	info.page_size = PTA_TNB_SECURE_RAM_TEST_PAGE_SIZE;
	info.physical_address = (uint64_t)pa;
	info.token = UINT64_C(0x4c4c4d545a534543);
	memcpy(info.pattern, secure_ram_test_pattern, sizeof(info.pattern));
	memcpy(params[0].memref.buffer, &info, sizeof(info));
	params[0].memref.size = sizeof(info);
	mutex_unlock(&lock);
	return TEE_SUCCESS;
}

static TEE_Result secure_ram_test_check(void *session, uint32_t types,
					TEE_Param params[TEE_NUM_PARAMS])
{
	struct pta_tnb_secure_ram_test_check check = { 0 };
	const uint32_t expected = TEE_PARAM_TYPES(
		TEE_PARAM_TYPE_MEMREF_OUTPUT, TEE_PARAM_TYPE_NONE,
		TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
	size_t n;

	if (types != expected || !params[0].memref.buffer)
		return TEE_ERROR_BAD_PARAMETERS;
	if (params[0].memref.size < sizeof(check)) {
		params[0].memref.size = sizeof(check);
		return TEE_ERROR_SHORT_BUFFER;
	}

	mutex_lock(&lock);
	check.version = PTA_TNB_SECURE_RAM_TEST_VERSION;
	check.active = secure_ram_test_active ? 1 : 0;
	if (!secure_ram_test_active || secure_ram_test_owner != session) {
		mutex_unlock(&lock);
		return TEE_ERROR_BAD_STATE;
	}
	dcache_inv_range(secure_ram_test_page,
			   PTA_TNB_SECURE_RAM_TEST_PAGE_SIZE);
	for (n = 0; n < PTA_TNB_SECURE_RAM_TEST_PAGE_SIZE; n++)
		if (secure_ram_test_page[n] != secure_ram_test_pattern[
				n % sizeof(secure_ram_test_pattern)])
			check.changed_bytes++;
	check.unchanged = check.changed_bytes == 0;
	memcpy(params[0].memref.buffer, &check, sizeof(check));
	params[0].memref.size = sizeof(check);
	mutex_unlock(&lock);
	return TEE_SUCCESS;
}

static TEE_Result secure_ram_test_end(void *session, uint32_t types)
{
	const uint32_t expected = TEE_PARAM_TYPES(
		TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
		TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);

	if (types != expected)
		return TEE_ERROR_BAD_PARAMETERS;
	mutex_lock(&lock);
	if (!secure_ram_test_active || secure_ram_test_owner != session) {
		mutex_unlock(&lock);
		return TEE_ERROR_BAD_STATE;
	}
	secure_ram_test_clear();
	mutex_unlock(&lock);
	return TEE_SUCCESS;
}
#endif

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
	if ((dev.prepare_flags & TNB_FLAG_UNCACHED_COMMAND) &&
	    uncached_command_page)
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

static uint32_t b_address(uint32_t index)
{
	const uint32_t bases[] = { TNB_IOVA_B, TNB_IOVA_B2, TNB_IOVA_B3 };

	return bases[index];
}

static uint32_t *page_table_for_iova(uint32_t iova)
{
	size_t dte = dte_index(iova);
	size_t b_dte = dte_index(TNB_IOVA_B);

	if (dte == dte_index(TNB_IOVA_CMD))
		return page_tables[0];
	if (dte == dte_index(TNB_IOVA_A))
		return page_tables[1];
	if (dte >= b_dte && dte < b_dte + 3)
		return page_tables[2 + dte - b_dte];
	if (dte >= dte_index(TNB_IOVA_B2) &&
	    dte < dte_index(TNB_IOVA_B2) + 3)
		return page_tables[6 + dte - dte_index(TNB_IOVA_B2)];
	if (dte >= dte_index(TNB_IOVA_B3) &&
	    dte < dte_index(TNB_IOVA_B3) + 3)
		return page_tables[9 + dte - dte_index(TNB_IOVA_B3)];
	if (dte == dte_index(TNB_IOVA_C))
		return page_tables[5];
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

static bool map_secure_buffer(uint32_t iova, void *buffer, size_t size,
			      uint32_t flags)
{
	size_t offset = 0;

	if (!buffer || !size || ((vaddr_t)buffer & (TNB_PAGE_SIZE - 1)) ||
	    (iova & (TNB_PAGE_SIZE - 1)))
		return false;
	for (offset = 0; offset < ROUNDUP(size, TNB_PAGE_SIZE);
	     offset += TNB_PAGE_SIZE) {
		void *page = (uint8_t *)buffer + offset;
		paddr_t pa = virt_to_phys(page);

		if (!page_in_area(page, CORE_MEM_SEC) ||
		    !map_page(iova + offset, pa, flags, dev.pte_flags))
			return false;
	}
	return true;
}

static bool map_public_buffer(uint32_t iova, void *buffer, size_t size,
			      struct mobj **hold)
{
	struct ts_session *caller = ts_get_calling_session();
	struct user_mode_ctx *uctx = NULL;
	struct mobj *mobj = NULL;
	size_t offset = 0;
	size_t span = ROUNDUP(size, TNB_PAGE_SIZE);
	size_t n = 0;
	size_t phys_offset = 0;

	if (!caller || !is_user_ta_ctx(caller->ctx) || !buffer || !size ||
	    ((vaddr_t)buffer & (TNB_PAGE_SIZE - 1)))
		return false;
	uctx = to_user_mode_ctx(caller->ctx);
	if (vm_buf_to_mboj_offs(uctx, buffer, span, &mobj, &offset) ||
	    !mobj || !mobj_is_nonsec(mobj) ||
	    !mobj_check_offset_and_len(mobj, offset, span))
		return false;
	phys_offset = mobj_get_phys_offs(mobj, TNB_PAGE_SIZE);
	if (ADD_OVERFLOW(phys_offset, offset, &phys_offset) ||
	    (phys_offset & (TNB_PAGE_SIZE - 1)))
		return false;
	for (n = 0; n < span; n += TNB_PAGE_SIZE) {
		paddr_t pa = 0;

		if (mobj_get_pa(mobj, offset + n, TNB_PAGE_SIZE, &pa) ||
		    !map_page(iova + n, pa, TNB_PTE_READ, 0))
			return false;
	}
	if (hold)
		*hold = mobj_get(mobj);
	return true;
}

static bool prepare_page_tables(TEE_Param b[PTA_TNB_MAX_B_INPUTS])
{
	size_t n = 0;
	uint32_t index;
	uint32_t count = dev.batched ? dev.batch.b_count : 1;

	if (!secure_page(directory_table))
		return false;
	memset(directory_table, 0, sizeof(directory_table));
	memset(page_tables, 0, sizeof(page_tables));
	if (!install_page_table(TNB_IOVA_CMD) ||
	    !install_page_table(TNB_IOVA_A) ||
	    !install_page_table(TNB_IOVA_C))
		return false;
	if (!map_secure_buffer(TNB_IOVA_CMD, command_buffer(0), TNB_PAGE_SIZE,
			       TNB_PTE_READ) ||
	    !map_secure_buffer(TNB_IOVA_A, a_pages, dev.a_span, TNB_PTE_READ) ||
	    !map_secure_buffer(TNB_IOVA_C, c_pages, dev.c_span,
			       TNB_PTE_READ | TNB_PTE_WRITE))
		return false;
	for (index = 0; index < count; index++) {
		uint32_t iova = b_address(index);

		for (n = 0; n < ROUNDUP(dev.b_span[index], SIZE_M(4)); n += SIZE_M(4))
			if (!install_page_table(iova + n))
				return false;
		if (!map_public_buffer(iova, b[index].memref.buffer,
				       dev.b_span[index],
				       dev.batched ? &dev.b_mobj[index] : NULL))
			return false;
	}

	dcache_clean_range(directory_table, sizeof(directory_table));
	for (n = 0; n < TNB_PAGE_TABLE_COUNT; ++n)
		dcache_clean_range(page_tables[n], sizeof(page_tables[n]));
	dev.debug[PTA_TNB_DEBUG_PTE_CMD] =
		page_table_for_iova(TNB_IOVA_CMD)[pte_index(TNB_IOVA_CMD)];
	dev.debug[PTA_TNB_DEBUG_PTE_A] =
		page_table_for_iova(TNB_IOVA_A)[pte_index(TNB_IOVA_A)];
	dev.debug[PTA_TNB_DEBUG_PTE_B] =
		page_table_for_iova(TNB_IOVA_B)[pte_index(TNB_IOVA_B)];
	dev.debug[PTA_TNB_DEBUG_PTE_C] =
		page_table_for_iova(TNB_IOVA_C)[pte_index(TNB_IOVA_C)];
	return true;
}

static void clear_work(bool keep_command)
{
	size_t input_clear_size = a_pages ? dev.a_span : 0;
	size_t output_clear_size = c_pages ? dev.c_span : 0;
	uint64_t *cmd = command_buffer(0);

	if (!keep_command && cmd)
		memset(cmd, 0, TNB_PAGE_SIZE);
	if (!keep_command && uncached_command_page && command_page_is_mapped &&
	    !command_page_is_uncached)
		memset(uncached_command_page, 0, TNB_PAGE_SIZE);
	if (input_clear_size)
		memset(a_pages, 0, input_clear_size);
	if (output_clear_size)
		memset(c_pages, 0, output_clear_size);
	memset(directory_table, 0, sizeof(directory_table));
	memset(page_tables, 0, sizeof(page_tables));
	if (!keep_command)
		dcache_clean_range(cached_command_page, sizeof(cached_command_page));
	if (input_clear_size)
		dcache_clean_range(a_pages, input_clear_size);
	if (output_clear_size)
		dcache_clean_range(c_pages, output_clear_size);
	dcache_clean_range(directory_table, sizeof(directory_table));
	dcache_clean_range(page_tables, sizeof(page_tables));
	dsb();
	a_pages = NULL;
	c_pages = NULL;
	for (size_t index = 0; index < PTA_TNB_MAX_B_INPUTS; index++) {
		mobj_put(dev.b_mobj[index]);
		dev.b_mobj[index] = NULL;
	}
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

static bool wait_iommu_status_mask(uint32_t bit, bool set, uint32_t banks)
{
	uint64_t deadline = timeout_init_us(1000);

	while (!timeout_elapsed(deadline)) {
		bool ready = true;
		size_t n = 0;

		for (n = 0; n < 4; n++) {
			if (!(banks & BIT(n)))
				continue;
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

static bool wait_iommu_status(uint32_t bit, bool set)
{
	return wait_iommu_status_mask(bit, set, 0xf);
}

static bool wait_iommu_paging(const struct tnb_mmu_save saved[4])
{
	uint64_t deadline = timeout_init_us(1000);

	while (!timeout_elapsed(deadline)) {
		bool ready = true;
		size_t n = 0;

		for (n = 0; n < 4; n++) {
			vaddr_t iommu = iommu_map(n);

			if (!iommu ||
			    !!(io_read32(iommu + IOMMU_STATUS) &
			       IOMMU_STATUS_PAGING) != saved[n].paging)
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
	uint32_t paging_banks = 0;

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
		if (saved[n].paging) {
			paging_banks |= BIT(n);
			io_write32(iommu + IOMMU_COMMAND,
				   IOMMU_CMD_ENABLE_STALL);
		}
	}
	dsb();
	if (!wait_iommu_status_mask(IOMMU_STATUS_STALL, true, paging_banks))
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
	}
	dsb();
	/* Keep stall active until security attributes and TNB DTE are installed. */
	return !mmu_fault();
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
	for (n = 0; n < 4; n++)
		io_write32(iommu_map(n) + IOMMU_COMMAND,
			   IOMMU_CMD_DISABLE_STALL);
	dsb();
	if (!wait_iommu_status(IOMMU_STATUS_STALL, false))
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

static bool restore_secure(vaddr_t bussgrf, vaddr_t firewall,
			   const struct tnb_hw_snapshot *before);

static bool restore_iommu(const struct tnb_mmu_save saved[4],
			  vaddr_t bussgrf, vaddr_t firewall, bool keep_secure)
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
	for (n = 0; n < 4; n++)
		io_write32(iommu_map(n) + IOMMU_COMMAND,
			   IOMMU_CMD_DISABLE_PAGING);
	dsb();
	if (!wait_iommu_status(IOMMU_STATUS_PAGING, false))
		return false;
	for (n = 0; n < 4; n++) {
		vaddr_t iommu = iommu_map(n);

		io_write32(iommu + IOMMU_DTE, saved[n].dte);
		io_write32(iommu + IOMMU_COMMAND, IOMMU_CMD_ZAP_CACHE);
		io_write32(iommu + IOMMU_AUTO_GATING, saved[n].auto_gating);
	}
	dsb();
	/* Change security attributes before reopening Linux address translation. */
	if (!keep_secure && !restore_secure(bussgrf, firewall, &dev.before))
		return false;
	for (n = 0; n < 4; n++) {
		if (saved[n].paging)
			io_write32(iommu_map(n) + IOMMU_COMMAND,
				   IOMMU_CMD_ENABLE_PAGING);
	}
	dsb();
	if (!wait_iommu_paging(saved))
		return false;
	for (n = 0; n < 4; n++) {
		vaddr_t iommu = iommu_map(n);

		io_write32(iommu + IOMMU_INT_MASK, saved[n].int_mask);
		io_write32(iommu + IOMMU_COMMAND, IOMMU_CMD_DISABLE_STALL);
		if (io_read32(iommu + IOMMU_DTE) != saved[n].dte)
			return false;
	}
	dsb();
	if (!wait_iommu_status(IOMMU_STATUS_STALL, false))
		return false;
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

static bool wait_irq_mask(vaddr_t gic)
{
	uint64_t deadline = timeout_init_us(1000);

	while (io_read32(gic + GICD_CTLR) & GICD_CTLR_RWP) {
		if (timeout_elapsed(deadline))
			return false;
		udelay(1);
	}
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
	if (!wait_irq_mask(gic))
		return false;
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
	if (!enabled && !wait_irq_mask(gic))
		return false;
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
	bool full = dev.prepare_flags & TNB_FLAG_FULL_SECURITY;
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
	dev.failure_trace.mode = dev.mode;
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
		.mode = dev.mode,
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
			success_trace.valid = 1;
			success_trace.complete_mask = dev.completed_mask;
			dev.completed_trace = success_trace;
			dev.completed_trace_valid = true;
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

	dcache_inv_range(c_pages, dev.c_size);
	memcpy(words, c_pages, sizeof(words));
	for (n = 0; n < 4; n++)
		dev.debug[PTA_TNB_DEBUG_OUTPUT0 + n] = words[n];
}

static void capture_command_registers(vaddr_t npu, uint32_t base)
{
		dev.debug[base + 0] = io_read32(npu + TNB_A_REG);
	dev.debug[base + 1] = io_read32(npu + TNB_B_REG);
	dev.debug[base + 2] = io_read32(npu + 0x1040);
	dev.debug[base + 3] = io_read32(npu + 0x3010);
	dev.debug[base + 4] = io_read32(npu + 0x4004);
	dev.debug[base + 5] = io_read32(npu + TNB_C_REG);
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

static TEE_Result prepare_gemm(void *session, uint32_t types,
			       TEE_Param params[TEE_NUM_PARAMS], bool reuse_command,
			       bool batched)
{
	const uint32_t expected = TEE_PARAM_TYPES(
		batched ? TEE_PARAM_TYPE_MEMREF_INOUT : TEE_PARAM_TYPE_MEMREF_INPUT,
		TEE_PARAM_TYPE_MEMREF_INPUT, TEE_PARAM_TYPE_MEMREF_INPUT,
		batched ? TEE_PARAM_TYPE_MEMREF_INPUT : TEE_PARAM_TYPE_NONE);
	struct pta_tnb_batch *batch = params[0].memref.buffer;
	const struct pta_tnb_gemm *request;
	TEE_Param buffers[PTA_TNB_MAX_B_INPUTS] = { };
	uint64_t words[TNB_GEMM_WORDS];
	uint64_t a_size, b_size, c_size;
	size_t input_span = 0, output_span = 0;
	uint64_t *cmd = NULL;
	TEE_Result res = TEE_SUCCESS;
	uint64_t profile_start = profile_now();
	size_t word = 0;
	uint32_t index;
	void *a = batched ? NULL : params[1].memref.buffer;
	uint32_t first_b = TNB_IOVA_B;

	if (types != expected || !params[0].memref.buffer ||
	    params[0].memref.size != (batched ? sizeof(*batch) : sizeof(*request)))
		return TEE_ERROR_BAD_PARAMETERS;
	if (batched) {
		batch->oom_stage = 3;
		batch->oom_size = 0;
		if (!batch->job_count || batch->job_count > PTA_TNB_MAX_JOBS ||
		    !batch->b_count || batch->b_count > PTA_TNB_MAX_B_INPUTS)
			return TEE_ERROR_BAD_PARAMETERS;
		for (index = 0; index < batch->b_count; index++) {
			buffers[index] = params[index + 1];
			if (!buffers[index].memref.buffer || !buffers[index].memref.size ||
			    buffers[index].memref.size > TNB_MAX_B_SIZE ||
			    ((vaddr_t)buffers[index].memref.buffer & (TNB_PAGE_SIZE - 1)))
				return TEE_ERROR_BAD_PARAMETERS;
		}
		for (index = 0; index < batch->job_count; index++) {
			const struct pta_tnb_job *job = &batch->job[index];
			const struct pta_tnb_gemm *g = &job->gemm;

			if (g->abi_version != PTA_TNB_ABI_VERSION ||
			    g->mode != PTA_TNB_GEMM_W8A8_SINGLE ||
			    !tnb_gemm_emit(g->m, g->k, g->n, words) ||
			    job->b_index >= batch->b_count ||
			    job->b_offset % TNB_PAGE_SIZE ||
			    (uint64_t)job->b_offset + (uint64_t)g->k * g->n >
				buffers[job->b_index].memref.size)
				return TEE_ERROR_BAD_PARAMETERS;
			input_span = MAX(input_span, ROUNDUP((size_t)g->m * g->k,
							   TNB_PAGE_SIZE));
			output_span = MAX(output_span, ROUNDUP((size_t)g->m * g->n *
							    sizeof(int32_t), TNB_PAGE_SIZE));
		}
		request = &batch->job[0].gemm;
		first_b = b_address(batch->job[0].b_index) + batch->job[0].b_offset;
	} else {
		request = params[0].memref.buffer;
		buffers[0] = params[2];
		if (request->abi_version != PTA_TNB_ABI_VERSION ||
		    request->mode != PTA_TNB_GEMM_W8A8_SINGLE ||
		    !tnb_gemm_emit(request->m, request->k, request->n, words) ||
		    !params[1].memref.buffer ||
		    params[1].memref.size != (uint64_t)request->m * request->k ||
		    !buffers[0].memref.buffer ||
		    buffers[0].memref.size != (uint64_t)request->k * request->n ||
		    buffers[0].memref.size > TNB_MAX_B_SIZE ||
		    ((vaddr_t)buffers[0].memref.buffer & (TNB_PAGE_SIZE - 1)))
			return TEE_ERROR_BAD_PARAMETERS;
		input_span = ROUNDUP((size_t)request->m * request->k, TNB_PAGE_SIZE);
		output_span = ROUNDUP((size_t)request->m * request->n * sizeof(int32_t),
				      TNB_PAGE_SIZE);
	}
	tnb_gemm_emit(request->m, request->k, request->n, words);
	a_size = (uint64_t)request->m * request->k;
	b_size = (uint64_t)request->k * request->n;
	c_size = (uint64_t)request->m * request->n * sizeof(int32_t);

	mutex_lock(&lock);
	if (dev.state == PTA_TNB_STATE_LOCKED) {
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (dev.state != PTA_TNB_STATE_READY) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	dev.prepare_flags = TNB_FLAGS;
	words[TNB_GEMM_A] = TNB_COMMAND_WITH_VALUE(words[TNB_GEMM_A], TNB_IOVA_A);
	words[TNB_GEMM_B] = TNB_COMMAND_WITH_VALUE(words[TNB_GEMM_B], first_b);
	words[TNB_GEMM_C] = TNB_COMMAND_WITH_VALUE(words[TNB_GEMM_C], TNB_IOVA_C);
	if (reuse_command) {
		if (!command_page_is_mapped || !command_page_is_uncached) {
			res = TEE_ERROR_BAD_STATE;
			goto reset;
		}
		for (word = 0; word < TNB_GEMM_WORDS; word++) {
			if (io_read64((vaddr_t)command_buffer(0) + word * sizeof(uint64_t)) !=
			    words[word]) {
				res = TEE_ERROR_BAD_STATE;
				goto reset;
			}
		}
	}
	if (!init_uncached_command_page()) {
		if (batched) {
			batch->oom_stage = 4;
			batch->oom_size = PTA_TNB_MAX_CORES * TNB_PAGE_SIZE;
		}
		EMSG("OOM_F command_page size=%zu",
		     (size_t)PTA_TNB_MAX_CORES * TNB_PAGE_SIZE);
		res = TEE_ERROR_OUT_OF_MEMORY;
		goto reset;
	}
	if (!reuse_command && !set_command_page_uncached(false)) {
		if (batched) {
			batch->oom_stage = 5;
			batch->oom_size = PTA_TNB_MAX_CORES * TNB_PAGE_SIZE;
		}
		EMSG("OOM_F command_map size=%zu",
		     (size_t)PTA_TNB_MAX_CORES * TNB_PAGE_SIZE);
		res = TEE_ERROR_OUT_OF_MEMORY;
		goto reset;
	}
	clear_work(reuse_command);
	dev.a_span = input_span;
	dev.c_span = output_span;
	dev.batched = batched;
	if (batched)
		dev.batch = *batch;
	for (index = 0; index < PTA_TNB_MAX_B_INPUTS; index++)
		dev.b_span[index] = buffers[index].memref.size;
	if (!staging_mm) {
		staging_mm = phys_mem_ta_alloc(TNB_STAGING_SIZE);
		if (staging_mm) {
			staging_pages = phys_to_virt(tee_mm_get_smem(staging_mm),
				MEM_AREA_SEC_RAM_OVERALL, TNB_STAGING_SIZE);
			if (!staging_pages) {
				tee_mm_free(staging_mm);
				staging_mm = NULL;
			} else {
				memset(staging_pages, 0, TNB_STAGING_SIZE);
				dcache_clean_range(staging_pages, TNB_STAGING_SIZE);
			}
		}
	}
	if (staging_pages) {
		a_pages = staging_pages;
		c_pages = staging_pages + input_span;
	}
	if (!a_pages || !c_pages) {
		if (batched) {
			batch->oom_stage = !a_pages ? 6 : 7;
			batch->oom_size = TNB_STAGING_SIZE;
		}
		EMSG("OOM_F staging_pool size=%zu", (size_t)TNB_STAGING_SIZE);
		res = TEE_ERROR_OUT_OF_MEMORY;
		goto reset;
	}
	dev.mode = request->mode;
	dev.core_count = 1;
	dev.a_size = a_size;
	dev.b_size = b_size;
	dev.c_size = c_size;
	dev.pte_flags = TNB_PTE_SECURE;
	dev.command_count[0] = TNB_GEMM_COMMANDS;
	dev.interrupt_mask[0] = 0x300;
	cmd = command_buffer(0);
	if (!reuse_command)
		memcpy(cmd, words, sizeof(words));
	memset(a_pages, 0, input_span);
	if (a)
		memcpy(a_pages, a, a_size);
	memset(c_pages, 0, output_span);
	for (index = 0; index < (batched ? batch->b_count : 1); index++)
		dcache_clean_range(buffers[index].memref.buffer, dev.b_span[index]);
	dcache_clean_range(a_pages, input_span);
	dcache_clean_range(c_pages, output_span);
	dev.debug[PTA_TNB_DEBUG_FLAGS] = dev.prepare_flags;
	dev.debug[PTA_TNB_DEBUG_CMD_ADDR] = TNB_IOVA_CMD;
	dev.debug[PTA_TNB_DEBUG_A_ADDR] = TNB_IOVA_A;
	dev.debug[PTA_TNB_DEBUG_B_ADDR] = TNB_IOVA_B;
	dev.debug[PTA_TNB_DEBUG_C_ADDR] = TNB_IOVA_C;
	dev.debug[PTA_TNB_DEBUG_DT_ADDR] = page_address(directory_table);
	dev.debug[PTA_TNB_DEBUG_PATCH_A] = TNB_IOVA_A;
	dev.debug[PTA_TNB_DEBUG_PATCH_B] = TNB_IOVA_B;
	dev.debug[PTA_TNB_DEBUG_PATCH_C] = TNB_IOVA_C;
	if (!dev.debug[PTA_TNB_DEBUG_DT_ADDR] ||
	    !prepare_page_tables(buffers)) {
		res = TEE_ERROR_GENERIC;
		goto reset;
	}
	if (!reuse_command)
		dcache_clean_range(cmd, TNB_PAGE_SIZE);
	dsb();
	if (!reuse_command && !set_command_page_uncached(true)) {
		res = TEE_ERROR_GENERIC;
		goto reset;
	}
	dev.owner = session;
	dev.stage = PTA_TNB_STAGE_PREPARED;
	dev.fault = PTA_TNB_FAULT_NONE;
	dev.state = PTA_TNB_STATE_PREPARED;
	goto out;
reset:
	clear_work(false);
	memset(&dev, 0, sizeof(dev));
out:
	profile_add(&profile.prepare_calls, &profile.prepare_ticks,
		    profile_start);
	mutex_unlock(&lock);
	return res;
}

/* Change a job's contents, never its mappings or backing, while ACTIVE. */
static TEE_Result update_gemm(void *session, uint32_t types,
			      TEE_Param params[TEE_NUM_PARAMS])
{
	const uint32_t expected = TEE_PARAM_TYPES(
		TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_MEMREF_INPUT,
		TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
	const struct pta_tnb_gemm *request = NULL;
	uint64_t words[TNB_GEMM_WORDS];
	uint32_t index = params[0].value.a;
	TEE_Result res = TEE_SUCCESS;
	size_t word = 0;
	uint32_t b_iova;
	uint32_t m = params[0].value.b;

	if (types != expected || index >= PTA_TNB_MAX_JOBS ||
	    !params[1].memref.buffer)
		return TEE_ERROR_BAD_PARAMETERS;
	mutex_lock(&lock);
	if (dev.state == PTA_TNB_STATE_LOCKED) {
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (dev.owner != session || dev.state != PTA_TNB_STATE_ACTIVE ||
	    !dev.batched) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	if (index >= dev.batch.job_count) {
		res = TEE_ERROR_BAD_PARAMETERS;
		goto out;
	}
	request = &dev.batch.job[index].gemm;
	b_iova = b_address(dev.batch.job[index].b_index) + dev.batch.job[index].b_offset;
	if (!m || m > request->m ||
	    params[1].memref.size != (size_t)m * request->k ||
	    !tnb_gemm_emit(m, request->k, request->n, words)) {
		res = TEE_ERROR_BAD_PARAMETERS;
		goto out;
	}
	words[TNB_GEMM_A] = TNB_COMMAND_WITH_VALUE(words[TNB_GEMM_A], TNB_IOVA_A);
	words[TNB_GEMM_B] = TNB_COMMAND_WITH_VALUE(words[TNB_GEMM_B], b_iova);
	words[TNB_GEMM_C] = TNB_COMMAND_WITH_VALUE(words[TNB_GEMM_C], TNB_IOVA_C);
	/* The existing uncached CPU mapping stays in place for the whole window. */
	for (word = 0; word < TNB_GEMM_WORDS; word++)
		io_write64((vaddr_t)command_buffer(0) + word * sizeof(uint64_t),
			   words[word]);
	memset(a_pages, 0, dev.a_span);
	memcpy(a_pages, params[1].memref.buffer, params[1].memref.size);
	dcache_cleaninv_range(c_pages, dev.c_span);
	memset(c_pages, 0, dev.c_span);
	dcache_clean_range(a_pages, dev.a_span);
	dcache_clean_range(c_pages, dev.c_span);
	dsb();
	dev.a_size = m * request->k;
	dev.b_size = request->k * request->n;
	dev.c_size = m * request->n * sizeof(int32_t);
	dev.debug[PTA_TNB_DEBUG_B_ADDR] = b_iova;
	dev.debug[PTA_TNB_DEBUG_PATCH_B] = b_iova;
	dev.debug[PTA_TNB_DEBUG_PTE_B] =
		page_table_for_iova(b_iova)[pte_index(b_iova)];
out:
	mutex_unlock(&lock);
	return res;
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
			    PTA_TNB_CAP_PROFILE | PTA_TNB_CAP_GEMM
#ifdef CFG_TNB_SECURITY_TEST
			    | PTA_TNB_CAP_SECURE_RAM_TEST
#endif
			    ,
		.max_modes = 1,
		.max_tensors = 3,
		.mode = PTA_TNB_GEMM_W8A8_SINGLE,
		.n_min = TNB_GEMM_N_MIN,
		.n_max = TNB_GEMM_N_MAX,
		.n_align = TNB_GEMM_N_ALIGN,
		.b_max = TNB_GEMM_B_MAX,
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

static TEE_Result activate(void *session, uint32_t types)
{
	const uint32_t expected =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
				TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
	vaddr_t bussgrf = io_map(BUSSGRF_BASE, BUSSGRF_SIZE);
	vaddr_t firewall = io_map(FIREWALL_DDR_BASE, FIREWALL_DDR_SIZE);
	struct tnb_hw_snapshot before = { };
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
	    snapshot(&before)) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	if (!dev.secure_held) {
		dev.before = before;
	} else {
		/* Preserve the original security baseline; refresh Linux MMU state. */
		memcpy(dev.before.iommu_dte, before.iommu_dte,
		       sizeof(before.iommu_dte));
		memcpy(dev.before.iommu_status, before.iommu_status,
		       sizeof(before.iommu_status));
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
		/* The masked-handoff diagnostic retains the first saved IRQ state. */
		if (dev.irq_held)
			break;
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
	if (!disable_iommu(dev.mmu)) {
		enter_locked(PTA_TNB_FAULT_IOMMU);
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (!dev.secure_held && !set_secure(bussgrf, firewall,
			dev.prepare_flags & TNB_FLAG_FULL_SECURITY)) {
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
	if (!install_iommu()) {
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
		   (dev.command_count[core] + TNB_TRAILER_COUNT + 1) / 2 - 1);
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
	if (params[0].memref.size < dev.c_size ||
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
	dsb();
	dev.debug[PTA_TNB_DEBUG_AFTER_CONTROL] =
		io_read32(npu + NPU_PC_TASK_CONTROL);
	dev.debug[PTA_TNB_DEBUG_AFTER_CONTROL_TASK] =
		io_read32(npu + NPU_PC_TASK_STATUS);
	dsb();
	for (n = 0; n < dev.core_count; n++)
		if (!trigger_npu_core(n)) {
			capture_failure_npu_debug();
			enter_locked(PTA_TNB_FAULT_NPU);
			res = TEE_ERROR_SECURITY;
			goto out;
		}
	dsb();
	dev.debug[PTA_TNB_DEBUG_AFTER_OP_ENABLE] =
		io_read32(npu + NPU_PC_TASK_CONTROL);
	dev.debug[PTA_TNB_DEBUG_AFTER_OP_ENABLE_TASK] =
		io_read32(npu + NPU_PC_TASK_STATUS);
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
	if (dev.completed_trace_valid) {
		dev.completed_trace.sequence = ++last_success_sequence;
		last_success_trace = dev.completed_trace;
	}
	memcpy(params[0].memref.buffer, c_pages, dev.c_size);
	params[0].memref.size = dev.c_size;
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

/* Repeat diagnostic: CLOSE's NPU interrupt writes without hardware handoff. */
static TEE_Result cycle_irq(void *session, uint32_t types, bool paging,
			    uint32_t zap_mask, bool command_line)
{
	const uint32_t expected = TEE_PARAM_TYPES(
		TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
		TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
	TEE_Result res = TEE_SUCCESS;
	size_t n = 0;

	if (types != expected)
		return TEE_ERROR_BAD_PARAMETERS;
	mutex_lock(&lock);
	if (dev.state == PTA_TNB_STATE_LOCKED) {
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (dev.owner != session || dev.state != PTA_TNB_STATE_ACTIVE) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	for (n = 0; n < dev.core_count; n++) {
		vaddr_t npu = npu_map(n);

		if (!npu) {
			res = TEE_ERROR_GENERIC;
			goto out;
		}
		io_write32(npu + NPU_INT_MASK, 0);
		io_write32(npu + NPU_INT_CLEAR, NPU_INT_ALL);
	}
	dsb();
	if (paging) {
		for (n = 0; n < 4; n++) {
			io_write32(iommu_map(n) + IOMMU_INT_MASK, 0);
			io_write32(iommu_map(n) + IOMMU_COMMAND,
				   IOMMU_CMD_ENABLE_STALL);
		}
		dsb();
		if (!wait_iommu_status(IOMMU_STATUS_STALL, true))
			goto mmu_error;
		for (n = 0; n < 4; n++)
			io_write32(iommu_map(n) + IOMMU_COMMAND,
				   IOMMU_CMD_DISABLE_PAGING);
		dsb();
		if (!wait_iommu_status(IOMMU_STATUS_PAGING, false))
			goto mmu_error;
		if (zap_mask) {
			for (n = 0; n < 4; n++)
				if (zap_mask & BIT(n))
					io_write32(iommu_map(n) + IOMMU_COMMAND,
						   IOMMU_CMD_ZAP_CACHE);
			dsb();
		}
		for (n = 0; n < 4; n++)
			io_write32(iommu_map(n) + IOMMU_COMMAND,
				   IOMMU_CMD_ENABLE_PAGING);
		dsb();
		if (!wait_iommu_status(IOMMU_STATUS_PAGING, true))
			goto mmu_error;
		for (n = 0; n < 4; n++) {
			io_write32(iommu_map(n) + IOMMU_INT_MASK, IOMMU_FAULT);
			io_write32(iommu_map(n) + IOMMU_COMMAND,
				   IOMMU_CMD_DISABLE_STALL);
		}
		dsb();
		if (!wait_iommu_status(IOMMU_STATUS_STALL, false) || mmu_fault())
			goto mmu_error;
	} else if (zap_mask) {
		for (n = 0; n < 4; n++)
			if (zap_mask & BIT(n))
				io_write32(iommu_map(n) +
					   (command_line ? IOMMU_ZAP_ONE_LINE :
							   IOMMU_COMMAND),
					   command_line ? TNB_IOVA_CMD :
							  IOMMU_CMD_ZAP_CACHE);
		dsb();
		if (mmu_fault())
			goto mmu_error;
	}
	for (n = 0; n < dev.core_count; n++)
		io_write32(npu_map(n) + NPU_INT_MASK, dev.old_npu_mask[n]);
	dsb();
	goto out;
mmu_error:
	enter_locked(PTA_TNB_FAULT_IOMMU);
	res = TEE_ERROR_SECURITY;
out:
	mutex_unlock(&lock);
	return res;
}

static TEE_Result close_device(void *session, uint32_t types, bool keep_command,
			       bool keep_work, bool keep_irq, bool keep_secure)
{
	const uint32_t expected =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
				TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
	struct tnb_hw_snapshot after = { };
	struct tnb_hw_snapshot baseline = { };
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
		if (dev.secure_held &&
		    !restore_iommu(dev.mmu, bussgrf, firewall, false))
			goto lock;
		if (dev.irq_held) {
			for (n = 0; n < dev.core_count; n++)
				if (!restore_irq(n, dev.irq_enabled[n]))
					goto lock;
		}
		if (!keep_command && (dev.prepare_flags & TNB_FLAG_UNCACHED_COMMAND) &&
		    !set_command_page_uncached(false)) {
			dev.state = PTA_TNB_STATE_LOCKED;
			dev.owner = NULL;
			dev.fault = PTA_TNB_FAULT_RESTORE;
			res = TEE_ERROR_SECURITY;
			goto out;
		}
		clear_work(keep_command);
		memset(&dev, 0, sizeof(dev));
		goto out;
	}
	if (dev.state != PTA_TNB_STATE_ACTIVE) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	baseline = dev.before;
	if (keep_secure) {
		if (snapshot(&baseline))
			goto lock;
		memcpy(baseline.iommu_dte, dev.before.iommu_dte,
		       sizeof(baseline.iommu_dte));
		memcpy(baseline.iommu_status, dev.before.iommu_status,
		       sizeof(baseline.iommu_status));
	}
	for (n = 0; n < dev.core_count; n++) {
		vaddr_t npu = npu_map(n);

		if (!npu)
			goto lock;
		io_write32(npu + NPU_INT_MASK, 0);
		io_write32(npu + NPU_INT_CLEAR, NPU_INT_ALL);
	}
	dsb();
	if (!restore_iommu(dev.mmu, bussgrf, firewall, keep_secure))
		goto lock;
	if (!keep_work && !keep_command &&
	    (dev.prepare_flags & TNB_FLAG_UNCACHED_COMMAND) &&
	    !set_command_page_uncached(false))
		goto lock;
	if (!keep_work)
		clear_work(keep_command);
	for (n = 0; n < dev.core_count; n++) {
		vaddr_t npu = npu_map(n);

		if (!npu) {
			restored = false;
			continue;
		}
		io_write32(npu + NPU_INT_MASK, dev.old_npu_mask[n]);
	}
	if (!restored)
		goto lock;
	if (!keep_irq)
		for (n = 0; n < dev.core_count; n++)
			restored = restore_irq(n, dev.irq_enabled[n]) && restored;
	if (!restored || snapshot(&after) ||
	    !snapshot_equal(&baseline, &after))
		goto lock;
	if (keep_work) {
		dev.irq_held = keep_irq;
		dev.secure_held = keep_secure;
		dev.state = PTA_TNB_STATE_PREPARED;
		dev.stage = PTA_TNB_STAGE_PREPARED;
		goto out;
	}
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

static bool work_difference(struct pta_tnb_hw_state *state, uint32_t area,
			    uint32_t index, uint32_t expected, uint32_t actual)
{
	if (expected == actual)
		return false;
	state->memory_area = area;
	state->memory_index = index;
	state->memory_expected = expected;
	state->memory_actual = actual;
	return true;
}

/* Diagnostic only: observe work memory without cleaning or invalidating it. */
static TEE_Result check_work(const struct pta_tnb_gemm *desc, void *b,
			     size_t b_size, struct pta_tnb_hw_state *state)
{
	struct ts_session *caller = ts_get_calling_session();
	struct mobj *mobj = NULL;
	uint64_t words[TNB_GEMM_WORDS] = { };
	size_t offset = 0;
	size_t a_span = dev.a_span;
	size_t b_span = ROUNDUP(dev.b_size, TNB_PAGE_SIZE);
	size_t c_span = dev.c_span;
	size_t table = 0;
	size_t index = 0;

	if (!caller || !is_user_ta_ctx(caller->ctx) ||
	    desc->abi_version != PTA_TNB_ABI_VERSION || desc->mode != dev.mode ||
	    !tnb_gemm_emit(desc->m, desc->k, desc->n, words) ||
	    (uint64_t)desc->m * desc->k != dev.a_size ||
	    (uint64_t)desc->k * desc->n != dev.b_size ||
	    (uint64_t)desc->m * desc->n * sizeof(int32_t) != dev.c_size ||
	    b_size != dev.b_size ||
	    vm_buf_to_mboj_offs(to_user_mode_ctx(caller->ctx), b, b_span,
			       &mobj, &offset) || !mobj || !mobj_is_nonsec(mobj))
		return TEE_ERROR_BAD_PARAMETERS;
	words[TNB_GEMM_A] = TNB_COMMAND_WITH_VALUE(words[TNB_GEMM_A], TNB_IOVA_A);
	words[TNB_GEMM_B] = TNB_COMMAND_WITH_VALUE(words[TNB_GEMM_B], TNB_IOVA_B);
	words[TNB_GEMM_C] = TNB_COMMAND_WITH_VALUE(words[TNB_GEMM_C], TNB_IOVA_C);
	for (index = 0; index < TNB_PAGE_SIZE / sizeof(uint32_t); index++) {
		uint32_t expected = index < TNB_GEMM_WORDS * 2 ?
			(uint32_t)(words[index / 2] >> ((index % 2) * 32)) : 0;
		uint32_t actual = io_read32((vaddr_t)command_buffer(0) +
					   index * sizeof(uint32_t));

		if (work_difference(state, PTA_TNB_WORK_COMMAND, index,
				    expected, actual))
			return TEE_SUCCESS;
	}
	for (index = 0; index < ARRAY_SIZE(directory_table); index++) {
		uint32_t iova = index << TNB_IOVA_DTE_SHIFT;
		uint32_t expected = 0;

		if (iova == TNB_IOVA_CMD || iova == TNB_IOVA_A ||
		    iova == TNB_IOVA_C ||
		    (iova >= TNB_IOVA_B && iova < TNB_IOVA_B +
		     ROUNDUP(b_span, SIZE_M(4))))
			expected = make_desc_v2(virt_to_phys(page_table_for_iova(iova)));
		if (work_difference(state, PTA_TNB_WORK_DIRECTORY, index,
				    expected, directory_table[index]))
			return TEE_SUCCESS;
	}
	for (table = 0; table < TNB_PAGE_TABLE_COUNT; table++) {
		for (index = 0; index < ARRAY_SIZE(page_tables[table]); index++) {
			size_t byte = index * TNB_PAGE_SIZE;
			paddr_t pa = 0;
			uint32_t flags = TNB_PTE_READ;
			uint32_t expected = 0;
			uint32_t actual = page_tables[table][index];

			if (table == 0 && index == 0) {
				pa = virt_to_phys(command_buffer(0));
				flags |= TNB_PTE_SECURE;
			} else if (table == 1 && byte < a_span) {
				pa = virt_to_phys((uint8_t *)a_pages + byte);
				flags |= TNB_PTE_SECURE;
				state->a_pte[index] = actual;
			} else if (table >= 2 && table <= 4) {
				byte += (table - 2) * SIZE_M(4);
				if (byte < b_span &&
				    mobj_get_pa(mobj, offset + byte, TNB_PAGE_SIZE, &pa))
					return TEE_ERROR_GENERIC;
			} else if (table == 5 && byte < c_span) {
				pa = virt_to_phys((uint8_t *)c_pages + byte);
				flags |= TNB_PTE_WRITE | TNB_PTE_SECURE;
				state->c_pte[index] = actual;
			}
			if (pa)
				expected = make_desc_v2(pa) | flags;
			if (work_difference(state, PTA_TNB_WORK_PTE,
					    table * ARRAY_SIZE(page_tables[table]) + index,
					    expected, actual))
				return TEE_SUCCESS;
		}
	}
	return TEE_SUCCESS;
}

static TEE_Result read_hw(uint32_t types, TEE_Param params[TEE_NUM_PARAMS])
{
	const uint32_t expected =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
				TEE_PARAM_TYPE_MEMREF_INPUT, TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_NONE);
	struct pta_tnb_hw_state state = { };
	TEE_Result res = TEE_SUCCESS;
	vaddr_t npu = npu_map(0);
	size_t n = 0;

	if (types != expected || !npu ||
	    params[1].memref.size != sizeof(struct pta_tnb_gemm) ||
	    !params[1].memref.buffer || !params[2].memref.buffer)
		return TEE_ERROR_BAD_PARAMETERS;
	if (params[0].memref.size < sizeof(state)) {
		params[0].memref.size = sizeof(state);
		return TEE_ERROR_SHORT_BUFFER;
	}
	mutex_lock(&lock);
	if (dev.state == PTA_TNB_STATE_ACTIVE) {
		res = check_work(params[1].memref.buffer,
				 params[2].memref.buffer, params[2].memref.size, &state);
		if (res) {
			mutex_unlock(&lock);
			return res;
		}
	}
	state.npu[0] = io_read32(npu + NPU_INT_STATUS);
	state.npu[1] = io_read32(npu + NPU_INT_RAW_STATUS);
	state.npu[2] = io_read32(npu + NPU_PC_TASK_STATUS);
	state.npu[3] = io_read32(npu + NPU_PC_DATA_ADDR);
	state.npu[4] = io_read32(npu + NPU_PC_DATA_AMOUNT);
	state.npu[5] = io_read32(npu + NPU_PC_TASK_CONTROL);
	state.npu[6] = io_read32(npu + NPU_PC_DMA_BASE_ADDR);
	state.npu[7] = io_read32(npu + NPU_PC_OP_EN);
	state.npu[8] = io_read32(npu + NPU_CNA_POINTER);
	state.npu[9] = io_read32(npu + NPU_CORE_POINTER);
	for (n = 0; n < PTA_TNB_HW_IOMMUS; n++) {
		vaddr_t iommu = iommu_map(n);

		if (!iommu) {
			mutex_unlock(&lock);
			return TEE_ERROR_GENERIC;
		}
		state.iommu[n][0] = io_read32(iommu + IOMMU_DTE);
		state.iommu[n][1] = io_read32(iommu + IOMMU_STATUS);
		state.iommu[n][2] = io_read32(iommu + IOMMU_INT_RAW_STATUS);
		state.iommu[n][3] = io_read32(iommu + IOMMU_INT_STATUS);
		state.iommu[n][4] = io_read32(iommu + IOMMU_INT_MASK);
		state.iommu[n][5] = io_read32(iommu + IOMMU_AUTO_GATING);
	}
	mutex_unlock(&lock);
	memcpy(params[0].memref.buffer, &state, sizeof(state));
	params[0].memref.size = sizeof(state);
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
	*session = calloc(1, 1);
	if (!*session)
		return TEE_ERROR_OUT_OF_MEMORY;
	return TEE_SUCCESS;
}

static void close_session(void *session)
{
	bool irq_restored = true;
	size_t n = 0;

	mutex_lock(&lock);
#ifdef CFG_TNB_SECURITY_TEST
	if (secure_ram_test_owner == session)
		secure_ram_test_clear();
#endif
	if (dev.owner == session && dev.state == PTA_TNB_STATE_PREPARED) {
		if (dev.secure_held) {
			vaddr_t bussgrf = io_map(BUSSGRF_BASE, BUSSGRF_SIZE);
			vaddr_t firewall = io_map(FIREWALL_DDR_BASE, FIREWALL_DDR_SIZE);

			if (!bussgrf || !firewall ||
			    !restore_iommu(dev.mmu, bussgrf, firewall, false)) {
				enter_locked(PTA_TNB_FAULT_RESTORE);
				goto out;
			}
		}
		if (dev.irq_held)
			for (n = 0; n < dev.core_count; n++)
				irq_restored = restore_irq(n, dev.irq_enabled[n]) &&
					       irq_restored;
		if (!irq_restored ||
		    ((dev.prepare_flags & TNB_FLAG_UNCACHED_COMMAND) &&
		     !set_command_page_uncached(false))) {
			dev.state = PTA_TNB_STATE_LOCKED;
			dev.owner = NULL;
			dev.fault = PTA_TNB_FAULT_RESTORE;
		} else {
			clear_work(false);
			memset(&dev, 0, sizeof(dev));
		}
	} else if (dev.owner == session &&
		   (dev.state == PTA_TNB_STATE_ACTIVE ||
		    dev.state == PTA_TNB_STATE_RUNNING)) {
		enter_locked(PTA_TNB_FAULT_CALLER);
	}
out:
	mutex_unlock(&lock);
	free(session);
}

static TEE_Result invoke_command(void *session, uint32_t command,
				 uint32_t types,
				 TEE_Param params[TEE_NUM_PARAMS])
{
	switch (command) {
	case PTA_TNB_CMD_GET_CAPS:
		return get_caps(types, params);
	case PTA_TNB_CMD_GEMM:
		return prepare_gemm(session, types, params, false, false);
	case PTA_TNB_CMD_GEMM_REUSE:
		return prepare_gemm(session, types, params, true, false);
	case PTA_TNB_CMD_PREPARE_BATCH:
		return prepare_gemm(session, types, params, false, true);
	case PTA_TNB_CMD_UPDATE:
		return update_gemm(session, types, params);
	case PTA_TNB_CMD_ACTIVATE:
		return activate(session, types);
	case PTA_TNB_CMD_RUN:
		return run(session, types, params);
	case PTA_TNB_CMD_CLOSE:
		return close_device(session, types, false, false, false, false);
	case PTA_TNB_CMD_CLOSE_KEEP:
		return close_device(session, types, true, false, false, false);
	case PTA_TNB_CMD_SUSPEND:
		return close_device(session, types, false, true, false, false);
	case PTA_TNB_CMD_SUSPEND_MASKED:
		return close_device(session, types, false, true, true, false);
	case PTA_TNB_CMD_SUSPEND_SECURE:
		return close_device(session, types, false, true, true, true);
	case PTA_TNB_CMD_CYCLE_IRQ:
		return cycle_irq(session, types, false, 0, false);
	case PTA_TNB_CMD_CYCLE_PAGING:
		return cycle_irq(session, types, true, 0, false);
	case PTA_TNB_CMD_CYCLE_ZAP:
		return cycle_irq(session, types, true, 0xf, false);
	case PTA_TNB_CMD_CYCLE_ZAP_ON:
		return cycle_irq(session, types, false, 0xf, false);
	case PTA_TNB_CMD_CYCLE_ZAP0:
		return cycle_irq(session, types, false, BIT(0), false);
	case PTA_TNB_CMD_CYCLE_ZAP1:
		return cycle_irq(session, types, false, BIT(1), false);
	case PTA_TNB_CMD_CYCLE_ZAP_CMD:
		return cycle_irq(session, types, false, BIT(1), true);
	case PTA_TNB_CMD_GET_STATUS:
		return get_status(types, params);
	case PTA_TNB_CMD_READ_HW:
		return read_hw(types, params);
	case PTA_TNB_CMD_LOCK:
		return lock_device(session, types);
#ifdef CFG_TNB_SECURITY_TEST
	case PTA_TNB_CMD_SECURE_RAM_TEST_BEGIN:
		return secure_ram_test_begin(session, types, params);
	case PTA_TNB_CMD_SECURE_RAM_TEST_CHECK:
		return secure_ram_test_check(session, types, params);
	case PTA_TNB_CMD_SECURE_RAM_TEST_END:
		return secure_ram_test_end(session, types);
#endif
	default:
		return TEE_ERROR_NOT_SUPPORTED;
	}
}

pseudo_ta_register(.uuid = PTA_TNB_UUID, .name = "tnb",
		   .flags = PTA_DEFAULT_FLAGS,
		   .open_session_entry_point = open_session,
		   .close_session_entry_point = close_session,
		   .invoke_command_entry_point = invoke_command);
