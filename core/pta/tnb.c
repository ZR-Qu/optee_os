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

#include "tnb_npu.h"
#include "tnb_iommu.h"
#include "tnb_gemm.h"

#define NPU_INT_MASK 0x20
#define NPU_INT_CLEAR 0x24
#define NPU_INT_STATUS 0x28
#define NPU_INT_RAW_STATUS 0x2c
#define NPU_PC_OP_EN 0x8
#define NPU_PC_DATA_ADDR 0x10
#define NPU_PC_DATA_AMOUNT 0x14
#define NPU_PC_TASK_CONTROL 0x30
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

#define GICD_ISENABLER 0x100
#define GICD_ICENABLER 0x180
#define GICD_ICPENDR 0x280
#define NPU_IRQ_BASE (32 + 110)

#define TNB_PAGE_SIZE 4096
#define TNB_DT_ENTRIES 1024
#define TNB_PT_ENTRIES 1024
/* Match the IOVA window used by the Linux RKNPU allocator. */
#define TNB_IOVA_INPUT 0x11000000
#define TNB_IOVA_WEIGHT 0x12000000
#define TNB_IOVA_WEIGHT_SLOT(slot) \
	((slot) == PTA_TNB_FFN_WEIGHT_SLOT_GATE ? 0x12000000 : \
	 (slot) == PTA_TNB_FFN_WEIGHT_SLOT_UP ? 0x14000000 : 0x15000000)
#define TNB_IOVA_OUTPUT 0x13000000
#define TNB_IOVA_GEMM_B1 0x14000000
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
#define TNB_PAGE_TABLE_COUNT 2
#define TNB_FFN_WEIGHT_CACHE_COUNT 84
#define TNB_FFN_WEIGHT_PTE_TABLE_COUNT 3
#define TNB_INPUT_STORAGE_SIZE 57344
#define TNB_OUTPUT_STORAGE_SIZE PTA_TNB_MAX_BATCH_OUTPUT_SIZE

#define TNB_INPUT_REG 0x1070
#define TNB_WEIGHT_REG 0x1110
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

/* Secure PTE storage has a fixed budget, independent of the core heap.
 * Each page maps 4 MiB. No separate array of backing PAs is retained. */
#define TNB_REGISTERED_PT_PAGES 256
static uint32_t registered_pt[TNB_REGISTERED_PT_PAGES][TNB_PT_ENTRIES]
    __aligned(TNB_PAGE_SIZE);
static bool registered_pt_used[TNB_REGISTERED_PT_PAGES];
static uint32_t next_weight_handle;
struct tnb_registered_weight {
    struct tnb_registered_weight *next;
    struct mobj *mobj;
    size_t offset, size;
    uint32_t handle, first, count;
    paddr_t table_pa[3];
};
#define TNB_GENERIC_PLAN_CACHE_COUNT 4u
struct tnb_generic_plan {
    bool valid;
    struct pta_tnb_gemm_batch batch;
    struct tnb_reg config[PTA_TNB_GEMM_MAX_JOBS][TNB_GEMM_WRITES];
};
struct tnb_session {
    struct tnb_generic_plan *plans;
    uint32_t next_plan;
    struct tnb_registered_weight *weights;
};

struct tnb_dev {
    size_t input_used, output_used;
	void *owner;
	uint32_t generic_count;
	bool generic_selected;
	uint32_t generic_b[PTA_TNB_GEMM_MAX_JOBS];
	uint32_t generic_offset[PTA_TNB_GEMM_MAX_JOBS];
	struct pta_tnb_gemm generic_gemm[PTA_TNB_GEMM_MAX_JOBS];
	struct tnb_reg generic_config[PTA_TNB_GEMM_MAX_JOBS][TNB_GEMM_WRITES];
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
	uint32_t old_npu_mask[PTA_TNB_MAX_CORES];
	bool irq_enabled[PTA_TNB_MAX_CORES];
	bool security_touched;
	bool completed_trace_valid;
	struct pta_tnb_trace failure_trace;
	struct pta_tnb_trace completed_trace;
	struct tnb_hw_snapshot before;
	struct tnb_mmu_save mmu[4];
};

static struct tnb_reg config[PTA_TNB_MAX_CORES][TNB_GEMM_WRITES];
static struct pta_tnb_trace last_success_trace;
static uint32_t last_success_sequence;
static uint32_t last_failure_sequence;
static uint8_t input_pages[TNB_INPUT_STORAGE_SIZE] __aligned(TNB_PAGE_SIZE);
static uint8_t output_pages[TNB_OUTPUT_STORAGE_SIZE] __aligned(TNB_PAGE_SIZE);
static bool fixed_tables_ready;
static uint32_t fixed_dte[2];
static uint32_t directory_table[TNB_DT_ENTRIES] __aligned(TNB_PAGE_SIZE);
static uint32_t page_tables[TNB_PAGE_TABLE_COUNT][TNB_PT_ENTRIES]
	__aligned(TNB_PAGE_SIZE);
static struct tnb_dev dev;
static struct pta_tnb_profile profile;
static struct pta_tnb_batch_profile batch_profile;
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
    if (dte == dte_index(TNB_IOVA_INPUT))
        return page_tables[0];
    if (dte == dte_index(TNB_IOVA_OUTPUT))
        return page_tables[1];
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

/* Registered generic jobs share fixed secure A/C staging.
 * Preserve these two PTE pages; only the per-window DTE bindings change. */
static bool prepare_fixed_page_tables(void)
{
    const uint32_t iova[2] = { TNB_IOVA_INPUT, TNB_IOVA_OUTPUT };
    const uint32_t table[2] = { 0, 1 };

    if (!fixed_tables_ready) {
        for (uint32_t i = 0; i < 2; i++) {
            memset(page_tables[table[i]], 0, sizeof(page_tables[0]));
            if (!install_page_table(iova[i]))
                return false;
            fixed_dte[i] = directory_table[dte_index(iova[i])];
        }
        if (!map_buffer(TNB_IOVA_INPUT, input_pages, sizeof(input_pages),
                        TNB_PTE_READ, true) ||
            !map_buffer(TNB_IOVA_OUTPUT, output_pages, sizeof(output_pages),
                        TNB_PTE_READ | TNB_PTE_WRITE, true))
            return false;
        for (uint32_t i = 0; i < 2; i++)
            dcache_clean_range(page_tables[table[i]], sizeof(page_tables[0]));
        fixed_tables_ready = true;
    }
    for (uint32_t i = 0; i < 2; i++)
        directory_table[dte_index(iova[i])] = fixed_dte[i];
    dcache_clean_range(directory_table, sizeof(directory_table));
    dev.debug[PTA_TNB_DEBUG_PTE_INPUT] = page_tables[0][pte_index(TNB_IOVA_INPUT)];
    dev.debug[PTA_TNB_DEBUG_PTE_OUTPUT] = page_tables[1][pte_index(TNB_IOVA_OUTPUT)];
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

/* Keep the largest page-rounded A/C footprint of the entire window, not
 * just the final wave (Down and short tails commonly use less space). */
static void record_generic_work(void)
{
    dev.input_used = MAX(dev.input_used, ROUNDUP(dev.input_size, TNB_PAGE_SIZE));
    dev.output_used = MAX(dev.output_used, ROUNDUP(dev.output_size, TNB_PAGE_SIZE));
}

static void clear_work(void)
{
    bool locked = dev.state == PTA_TNB_STATE_LOCKED;
    size_t input_clear_size = locked ? sizeof(input_pages) : dev.input_used;
    size_t output_clear_size = locked ? sizeof(output_pages) : dev.output_used;

    if (locked)
        fixed_tables_ready = false;
    dev.input_used = 0;
    dev.output_used = 0;
    dev.template_id = 0;
    dev.weight_cookie = 0;
    dev.weight_buffer = NULL;
    dev.generic_count = 0;
    dev.generic_selected = false;
    if (input_clear_size) {
        memset(input_pages, 0, input_clear_size);
        dcache_clean_range(input_pages, input_clear_size);
    }
    if (output_clear_size) {
        memset(output_pages, 0, output_clear_size);
        dcache_clean_range(output_pages, output_clear_size);
    }
    memset(directory_table, 0, sizeof(directory_table));
    dcache_clean_range(directory_table, sizeof(directory_table));
    if (!fixed_tables_ready) {
        memset(page_tables, 0, sizeof(page_tables));
        dcache_clean_range(page_tables, sizeof(page_tables));
    }
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


/* CPU memory observations only: these do not prove what the NPU fetched.
 * NXR8 uses generic-only debug slots 52..55/59..62; no ABI-size change. */



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

static bool iommu_bases(vaddr_t bases[TNB_IOMMU_BANKS])
{
	size_t n = 0;

	for (n = 0; n < TNB_IOMMU_BANKS; n++) {
		bases[n] = iommu_map(n);
		if (!bases[n])
			return false;
	}
	return true;
}

static bool disable_iommu(struct tnb_mmu_save saved[4])
{
	vaddr_t bases[TNB_IOMMU_BANKS] = { };

	return iommu_bases(bases) && tnb_iommu_disable(bases, saved) &&
	       !mmu_fault();
}

static bool install_iommu(void)
{
	paddr_t dt_pa = virt_to_phys(directory_table);
	uint32_t dte = make_desc_v2(dt_pa);
	vaddr_t bases[TNB_IOMMU_BANKS] = { };
	size_t n = 0;

	if (!iommu_bases(bases) || !tnb_iommu_install(bases, dte) || mmu_fault())
		return false;
	for (n = 0; n < TNB_IOMMU_BANKS; n++)
		if (desc_address_v2(io_read32(bases[n] + IOMMU_DTE)) != dt_pa)
			return false;
	dev.debug[PTA_TNB_DEBUG_DTE_WRITTEN] = dte;
	for (n = 0; n < TNB_IOMMU_BANKS; n++) {
		dev.debug[PTA_TNB_DEBUG_DTE_READBACK0 + n] =
			io_read32(bases[n] + IOMMU_DTE);
		dev.debug[PTA_TNB_DEBUG_IOMMU_STATUS0 + n] =
			io_read32(bases[n] + IOMMU_STATUS);
	}
	return true;
}

static bool restore_iommu(const struct tnb_mmu_save saved[4])
{
	vaddr_t bases[TNB_IOMMU_BANKS] = { };

	return iommu_bases(bases) && tnb_iommu_restore(bases, saved);
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

static void capture_bank_debug(vaddr_t npu, uint32_t index, unsigned phase)
{
	dev.debug[index] = io_read32(npu + NPU_CNA_POINTER);
	dev.debug[index + 1] = io_read32(npu + NPU_CORE_POINTER);
	dev.debug[index + 2] = io_read32(npu + 0x4004);
	dev.debug[PTA_TNB_DEBUG_ENABLE_PRE + phase] = io_read32(npu + 0xf008);
	dev.debug[PTA_TNB_DEBUG_BANK_VALID] |= BIT(phase);
}

static void capture_address_debug(unsigned phase)
{
	static const uint32_t offsets[6] = {
		0x1070, 0x1110, 0x1040, 0x3010, 0x4004, 0x4020
	};
	vaddr_t npu = npu_map(0);

	if (!dev.generic_selected || phase >= 4)
		return;
	for (size_t i = 0; i < 6; i++)
		dev.debug[358 + phase * 6 + i] = io_read32(npu + offsets[i]);
	dev.debug[382] |= BIT(phase);
	if (phase == 3)
		for (size_t i = 0; i < TNB_IOMMU_BANKS; i++)
			dev.debug[383 + i] = io_read32(iommu_map(i) + IOMMU_DTE);
}

static void capture_failure_npu_debug(void)
{
	size_t n = 0;

	capture_address_debug(3);
	capture_bank_debug(npu_map(0), PTA_TNB_DEBUG_BANK_FAILURE, 4);
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
	uint32_t poll_index = 0, last_irq = 0, last_raw = 0;
	bool have_sample = false;

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
			if (!n) {
				poll_index++;
				if ((dev.prepare_flags & PTA_TNB_PREPARE_FLAG_TRACE_STEPS) &&
				    (!have_sample || status != last_irq || raw != last_raw)) {
					uint32_t count = dev.debug[PTA_TNB_DEBUG_POLL_COUNT];
					uint32_t index = PTA_TNB_DEBUG_POLL_HISTORY + (count % 8) * 3;
					dev.debug[index] = poll_index;
					dev.debug[index + 1] = status;
					dev.debug[index + 2] = raw;
					dev.debug[PTA_TNB_DEBUG_POLL_COUNT] = count + 1;
					last_irq = status;
					last_raw = raw;
					have_sample = true;
				}
			}
			dev.npu_status |= status;
			dev.npu_raw_status |= raw;
			if (!status || done[n])
				continue;
			if (normalize_status(status) != dev.interrupt_mask[n]) {
				capture_failure_npu_debug();
				capture_run_status(npu);
				return PTA_TNB_FAULT_NPU;
			}
			if (dev.prepare_flags & PTA_TNB_PREPARE_FLAG_TRACE_STEPS)
				capture_core_snapshot(npu, status, raw,
					      &success_trace.core[n]);
			else {
				success_trace.core[n].int_status = status;
				success_trace.core[n].int_raw_status = raw;
			}
			if (!n && (dev.prepare_flags & PTA_TNB_PREPARE_FLAG_TRACE_STEPS)) {
				dev.debug[PTA_TNB_DEBUG_BANK_COMPLETION] = success_trace.core[n].cna_pointer;
				dev.debug[PTA_TNB_DEBUG_BANK_COMPLETION + 1] = success_trace.core[n].core_pointer;
				dev.debug[PTA_TNB_DEBUG_BANK_COMPLETION + 2] = io_read32(npu + 0x4004);
				dev.debug[PTA_TNB_DEBUG_ENABLE_PRE + 3] = io_read32(npu + 0xf008);
				dev.debug[PTA_TNB_DEBUG_BANK_VALID] |= BIT(3);
			}
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

static size_t output_work_size(void);

static void capture_output_debug(void)
{
	uint32_t words[4] = { };
	size_t n = 0;

	dcache_inv_range(output_pages, output_work_size());
	if (!(dev.prepare_flags & PTA_TNB_PREPARE_FLAG_TRACE_STEPS))
		return;
	memcpy(words, output_pages, sizeof(words));
	for (n = 0; n < 4; n++)
		dev.debug[PTA_TNB_DEBUG_OUTPUT0 + n] = words[n];
}

static void capture_config_addresses(vaddr_t npu, uint32_t base)
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
                    PTA_TNB_CAP_SECURE_PTE | PTA_TNB_CAP_FULL_SECURITY |
                    PTA_TNB_CAP_NONSECURE_WEIGHT | PTA_TNB_CAP_MULTI_PAGE_IO |
                    PTA_TNB_CAP_MULTI_CORE | PTA_TNB_CAP_PROFILE |
                    PTA_TNB_CAP_GENERIC_GEMM,
        .max_templates = 0,
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

/* Translate each virtual page independently: TA/shared buffers need not
 * have physically contiguous backing.
 */
static bool buffer_in_area(const void *buffer, size_t size, uint32_t area)
{
	vaddr_t address = (vaddr_t)buffer;

	if (!buffer || !size || size > UINTPTR_MAX - address)
		return false;
	while (size) {
		size_t span = MIN(size, TNB_PAGE_SIZE - (address & (TNB_PAGE_SIZE - 1)));
		paddr_t pa = virt_to_phys((void *)address);

		if (!pa || !core_pbuf_is(area, pa, span))
			return false;
		address += span;
		size -= span;
	}
	return true;
}

/* The trusted caller keeps A/B backing alive through CLOSE. B is public,
 * page-aligned and page-rounded; A and the RUN output remain secure.
 */
/* Caller holds lock; descriptor and buffers have already been validated. */
static TEE_Result load_generic_gemm(void *session,
    const struct pta_tnb_gemm *gemm)
{
    uint64_t tick;
    bool cold = !fixed_tables_ready;

    if (dev.state == PTA_TNB_STATE_LOCKED)
        return TEE_ERROR_SECURITY;
    if (dev.state != PTA_TNB_STATE_READY)
        return TEE_ERROR_BAD_STATE;
    dev.core_count = 1;
    dev.prepare_flags = PTA_TNB_PREPARE_FLAG_SECURE_PTE |
        PTA_TNB_PREPARE_FLAG_FULL_SECURITY | PTA_TNB_PREPARE_FLAG_NONSECURE_WEIGHT;
    dev.pte_flags = TNB_PTE_SECURE;
    dev.input_size = gemm->m * gemm->k;
    dev.weight_size = gemm->k * gemm->n;
    dev.output_size = gemm->m * gemm->n * sizeof(int32_t);
    memset(dev.debug, 0, sizeof(dev.debug));
    dev.debug[PTA_TNB_DEBUG_FLAGS] = dev.prepare_flags;
    dev.debug[PTA_TNB_DEBUG_INPUT_ADDR] = TNB_IOVA_INPUT;
    dev.debug[PTA_TNB_DEBUG_WEIGHT_ADDR] = TNB_IOVA_WEIGHT;
    dev.debug[PTA_TNB_DEBUG_OUTPUT_ADDR] = TNB_IOVA_OUTPUT;
    dev.debug[PTA_TNB_DEBUG_DT_ADDR] = page_address(directory_table);
    dev.debug[PTA_TNB_DEBUG_PATCH_INPUT] = TNB_IOVA_INPUT;
    dev.debug[PTA_TNB_DEBUG_PATCH_WEIGHT] = TNB_IOVA_WEIGHT;
    dev.debug[PTA_TNB_DEBUG_PATCH_OUTPUT] = TNB_IOVA_OUTPUT;
    if (!dev.debug[PTA_TNB_DEBUG_DT_ADDR] || !secure_page(directory_table) ||
        !prepare_fixed_page_tables()) {
        clear_work();
        memset(&dev, 0, sizeof(dev));
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (cold) {
        tick = profile_now();
        dcache_clean_range(input_pages, sizeof(input_pages));
        dcache_clean_range(output_pages, sizeof(output_pages));
        dsb();
        batch_profile.begin_clean_ticks += profile_now() - tick;
    }
    dev.interrupt_mask[0] = 0x300;
    dev.template_id = PTA_TNB_TEMPLATE_GENERIC;
    record_generic_work();
    dev.owner = session;
    dev.stage = PTA_TNB_STAGE_PREPARED;
    dev.fault = PTA_TNB_FAULT_NONE;
    dev.state = PTA_TNB_STATE_PREPARED;
    return TEE_SUCCESS;
}

/* Both B ranges are mapped before ACTIVATE. The outer trusted invocation
 * retains their backing until CLOSE; no ACTIVE page-table update is allowed.
 */

/* Three independent public B windows, mapped once before ACTIVATE. */

static void release_registered_weights(struct tnb_session *ctx)
{
    while (ctx->weights) {
        struct tnb_registered_weight *weight = ctx->weights;
        ctx->weights = weight->next;
        for (size_t n = 0; n < weight->count; n++) {
            memset(registered_pt[weight->first + n], 0, TNB_PAGE_SIZE);
            registered_pt_used[weight->first + n] = false;
        }
        mobj_put(weight->mobj);
        free(weight);
    }
}

/* Registration is restricted to a trusted caller and owns the backing mobj.
 * Contents of public B are not immutable; registration fixes page identity. */
static TEE_Result register_weight(void *session, uint32_t types,
                                 TEE_Param params[TEE_NUM_PARAMS])
{
    struct tnb_session *ctx = session;
    struct tnb_registered_weight *weight = NULL;
    struct mobj *mobj = NULL;
    size_t offs = 0, size = params[0].memref.size, n, first, count;
    TEE_Result res = TEE_ERROR_BAD_PARAMETERS;
    uint64_t start = profile_now();

    if (types != TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                TEE_PARAM_TYPE_VALUE_OUTPUT, 0, 0) ||
        !size || size > SIZE_M(12) || (size & (TNB_PAGE_SIZE - 1)) ||
        !caller_buffer_mobj(params[0].memref.buffer, size, &mobj, &offs))
        return res;
    mutex_lock(&lock);
    if (dev.state != PTA_TNB_STATE_READY) {
        res = TEE_ERROR_BAD_STATE;
        goto out;
    }
    for (weight = ctx->weights; weight; weight = weight->next)
        if (weight->mobj == mobj && weight->offset == offs && weight->size == size) {
            params[1].value.a = weight->handle;
            params[1].value.b = 0;
            res = TEE_SUCCESS;
            goto out;
        }
    count = ROUNDUP(size, SIZE_M(4)) / SIZE_M(4);
    for (first = 0; first + count <= TNB_REGISTERED_PT_PAGES; first++) {
        for (n = 0; n < count && !registered_pt_used[first + n]; n++)
            ;
        if (n == count)
            break;
    }
    res = TEE_ERROR_OUT_OF_MEMORY;
    if (first + count > TNB_REGISTERED_PT_PAGES || next_weight_handle == UINT32_MAX)
        goto out;
    weight = calloc(1, sizeof(*weight));
    if (!weight)
        goto out;
    for (n = 0; n < count; n++) {
        if (!secure_page(registered_pt[first + n]))
            goto invalid;
        weight->table_pa[n] = virt_to_phys(registered_pt[first + n]);
        memset(registered_pt[first + n], 0, TNB_PAGE_SIZE);
    }
    for (n = 0; n < size / TNB_PAGE_SIZE; n++) {
        paddr_t pa = 0;
        if (mobj_get_pa(mobj, offs + n * TNB_PAGE_SIZE, TNB_PAGE_SIZE, &pa) ||
            !pa || (pa & (TNB_PAGE_SIZE - 1)) || pa >= BIT64(40) ||
            !core_pbuf_is(CORE_MEM_NON_SEC, pa, TNB_PAGE_SIZE))
            goto invalid;
        registered_pt[first + n / TNB_PT_ENTRIES][n % TNB_PT_ENTRIES] =
            make_desc_v2(pa) | TNB_PTE_READ;
    }
    weight->mobj = mobj_get(mobj);
    if (!weight->mobj)
        goto invalid;
    weight->offset = offs;
    weight->size = size;
    weight->first = first;
    weight->count = count;
    weight->handle = ++next_weight_handle;
    dcache_clean_range(params[0].memref.buffer, size);
    dcache_clean_range(registered_pt[first], count * TNB_PAGE_SIZE);
    dsb();
    for (n = 0; n < count; n++)
        registered_pt_used[first + n] = true;
    weight->next = ctx->weights;
    ctx->weights = weight;
    params[1].value.a = weight->handle;
    params[1].value.b = 0;
    profile_add(&batch_profile.register_calls, &batch_profile.register_ticks, start);
    batch_profile.register_pages += size / TNB_PAGE_SIZE;
    batch_profile.register_cache_bytes += size;
    res = TEE_SUCCESS;
    goto out;
invalid:
    free(weight);
    res = TEE_ERROR_BAD_PARAMETERS;
out:
    mutex_unlock(&lock);
    return res;
}

/* The cache contains only trusted, unbound generator output. Weight handles,
 * backing and address bindings are validated anew on every BEGIN/RUN.
 * Caller holds the core lock. Four entries bound per-session heap use. */
static TEE_Result get_registered_plan(struct tnb_session *ctx,
        const struct pta_tnb_gemm_batch *batch, struct tnb_generic_plan **out)
{
    struct tnb_generic_plan *plan;
    size_t bytes = batch->count * sizeof(batch->job[0]);

    if (!ctx->plans) {
        ctx->plans = calloc(TNB_GENERIC_PLAN_CACHE_COUNT, sizeof(*ctx->plans));
        if (!ctx->plans)
            return TEE_ERROR_OUT_OF_MEMORY;
    }
    for (uint32_t i = 0; i < TNB_GENERIC_PLAN_CACHE_COUNT; i++) {
        plan = &ctx->plans[i];
        if (plan->valid && plan->batch.count == batch->count &&
            !memcmp(plan->batch.job, batch->job, bytes)) {
            *out = plan;
            return TEE_SUCCESS;
        }
    }
    plan = &ctx->plans[ctx->next_plan];
    plan->valid = false;
    for (uint32_t i = 0; i < batch->count; i++) {
        const struct pta_tnb_gemm *g = &batch->job[i].gemm;
        if (!tnb_gemm_emit(g->m, g->k, g->n, plan->config[i]))
            return TEE_ERROR_BAD_PARAMETERS;
    }
    plan->batch = *batch;
    plan->valid = true;
    ctx->next_plan = (ctx->next_plan + 1) % TNB_GENERIC_PLAN_CACHE_COUNT;
    *out = plan;
    return TEE_SUCCESS;
}

/* No weight memref crosses this boundary. Only small session-owned handles
 * and a semantic batch descriptor are consumed in the steady-state path. */
static TEE_Result prepare_registered_batch(void *session, uint32_t types,
                                          TEE_Param params[TEE_NUM_PARAMS])
{
    struct tnb_session *ctx = session;
    struct tnb_registered_weight *weights[3] = { };
    struct pta_tnb_gemm_batch batch = { };
    struct tnb_generic_plan *plan = NULL;
    TEE_Result res = TEE_ERROR_BAD_PARAMETERS;
    size_t i, n;
    uint64_t start = profile_now(), tick;

    if (types != TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
            TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_VALUE_INPUT,
            TEE_PARAM_TYPE_VALUE_INPUT) || !params[0].memref.buffer ||
        params[0].memref.size != sizeof(batch))
        return res;
    memcpy(&batch, params[0].memref.buffer, sizeof(batch));
    if (!batch.count || batch.count > PTA_TNB_GEMM_MAX_JOBS || batch.reserved)
        return res;
    mutex_lock(&lock);
    if (dev.state != PTA_TNB_STATE_READY) {
        res = dev.state == PTA_TNB_STATE_LOCKED ? TEE_ERROR_SECURITY : TEE_ERROR_BAD_STATE;
        goto out;
    }
    for (i = 0; i < 3; i++) {
        if (params[i + 1].value.b)
            goto out;
        for (weights[i] = ctx->weights; weights[i]; weights[i] = weights[i]->next)
            if (weights[i]->handle == params[i + 1].value.a)
                break;
        if (!weights[i])
            goto out;
    }
    for (i = 0; i < batch.count; i++) {
        const struct pta_tnb_gemm_job *job = &batch.job[i];
        const struct pta_tnb_gemm *g = &job->gemm;
        if (g->abi_version != PTA_TNB_ABI_VERSION ||
            g->mode != PTA_TNB_GEMM_MODE_W8A8 || g->reserved[0] ||
            g->reserved[1] || g->reserved[2] ||
            job->b_index >= 3 || (job->b_offset & (TNB_PAGE_SIZE - 1)) ||
            (uint64_t)job->b_offset + (uint64_t)g->k * g->n > weights[job->b_index]->size)
            goto out;
    }
    res = get_registered_plan(ctx, &batch, &plan);
    if (res)
        goto out;
    tick = profile_now();
    res = load_generic_gemm(session, &batch.job[0].gemm);
    batch_profile.setup_ticks += profile_now() - tick;
    if (res)
        goto out;
    for (i = 0; i < 3; i++)
        for (n = 0; n < 3; n++)
            directory_table[dte_index(TNB_IOVA_WEIGHT_SLOT(i) + n * SIZE_M(4))] =
                n < weights[i]->count ? make_desc_v2(weights[i]->table_pa[n]) : 0;
    for (i = 0; i < batch.count; i++) {
        dev.generic_gemm[i] = batch.job[i].gemm;
        dev.generic_b[i] = batch.job[i].b_index;
        dev.generic_offset[i] = batch.job[i].b_offset;
        memcpy(dev.generic_config[i], plan->config[i],
               sizeof(dev.generic_config[i]));
    }
    dev.generic_count = batch.count;
    dev.debug[PTA_TNB_DEBUG_PTE_WEIGHT] = registered_pt[weights[0]->first][0];
    dcache_clean_range(directory_table, sizeof(directory_table));
    dsb();
out:
    profile_add(&batch_profile.calls, &batch_profile.ticks, start);
    mutex_unlock(&lock);
    return res;
}

/* Select trusted config in a stable ACTIVE address space. Only
 * CPU config and A/C data change; DTE/PTE/IOMMU remain untouched.
 */
static size_t output_work_size(void)
{
    if (dev.template_id != PTA_TNB_TEMPLATE_GENERIC)
        return sizeof(output_pages);
    return ROUNDUP(dev.output_size, TNB_PAGE_SIZE);
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
		    snapshot(&dev.before)) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	for (n = 0; n < PTA_TNB_MAX_CORES; n++) {
		vaddr_t npu = npu_map(n);

		if (!npu) {
			res = TEE_ERROR_GENERIC;
			goto out;
		}
		dev.old_npu_mask[n] = io_read32(npu + NPU_INT_MASK);
	}
	/* Test-only: save all restoration inputs before any device mutation. */
	for (n = 0; n < TNB_IOMMU_BANKS; n++) {
		vaddr_t mmu = iommu_map(n);

		if (!mmu) {
			res = TEE_ERROR_GENERIC;
			goto out;
		}
		dev.mmu[n].dte = dev.before.iommu_dte[n];
		dev.mmu[n].paging = !!(dev.before.iommu_status[n] & IOMMU_STATUS_PAGING);
		dev.mmu[n].int_mask = io_read32(mmu + IOMMU_INT_MASK);
		dev.mmu[n].auto_gating = io_read32(mmu + IOMMU_AUTO_GATING);
	}
	for (n = 0; n < PTA_TNB_MAX_CORES; n++)
		if (!irq_state(n, &dev.irq_enabled[n])) {
			res = TEE_ERROR_GENERIC;
			goto out;
		}
	/* IOMMU takeover includes all four banks, including NPU1/2, even
	 * when only core0 executes. Protect every shared NPU/IOMMU IRQ. */
	for (n = 0; n < PTA_TNB_MAX_CORES; n++) {
		if (!disable_irq(n, &dev.irq_enabled[n])) {
			size_t restore_count = n + 1;

			res = TEE_ERROR_GENERIC;
			while (restore_count--)
				if (!restore_irq(restore_count,
						 dev.irq_enabled[restore_count]))
					res = TEE_ERROR_SECURITY;
			if (res == TEE_ERROR_SECURITY)
				enter_locked(PTA_TNB_FAULT_IRQ);
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
	if (!disable_iommu(dev.mmu)) {
		enter_locked(PTA_TNB_FAULT_IOMMU);
		res = TEE_ERROR_SECURITY;
		goto out;
	}
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




static bool prepare_npu_core(size_t core)
{
	vaddr_t npu = npu_map(core);

	if (!npu || core >= dev.core_count ||
	    !dev.generic_count)
		return false;

	tnb_npu_prepare(npu, core, config[core],
			dev.interrupt_mask[core]);
	return true;
}

static bool trigger_npu_core(size_t core)
{
	vaddr_t npu = npu_map(core);

	if (!npu)
		return false;
	tnb_npu_start(npu);
	return true;
}

static TEE_Result run_locked(void *session, TEE_Param *output)
{
	vaddr_t npu = npu_map(0);
	TEE_Result res = TEE_SUCCESS;
	uint64_t profile_start = profile_now();
	uint64_t poll_start = 0;
	size_t n = 0;
	bool trace = dev.prepare_flags & PTA_TNB_PREPARE_FLAG_TRACE_STEPS;

	if (!npu)
		return TEE_ERROR_BAD_PARAMETERS;
	if (output->memref.size < dev.output_size)
		return TEE_ERROR_SHORT_BUFFER;

	if (dev.state == PTA_TNB_STATE_LOCKED) {
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (dev.owner != session || dev.state != PTA_TNB_STATE_ACTIVE) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	if (dev.template_id == PTA_TNB_TEMPLATE_GENERIC &&
	    (!(*output).memref.buffer ||
	     !buffer_in_area((*output).memref.buffer, dev.output_size, CORE_MEM_SEC))) {
		res = TEE_ERROR_BAD_PARAMETERS;
		goto out;
	}
	if (dev.generic_count && !dev.generic_selected) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	/* Experiment only: reuse the initially selected GEMM. */
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
	memset(dev.debug + PTA_TNB_DEBUG_BANK_PRE, 0, 46 * sizeof(uint32_t));
	if (trace)
		capture_bank_debug(npu, PTA_TNB_DEBUG_BANK_PRE, 0);
	if (trace)
		capture_npu_debug(npu, PTA_TNB_DEBUG_PRE_STATUS);
	dev.debug[382] = 0;
	if (trace)
		capture_address_debug(0);
	if (trace)
		capture_config_addresses(npu, PTA_TNB_DEBUG_PRE_CMD_INPUT);
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
	/* Experiment: PRE_CMD words now hold prepared/pre-OP_EN PC state. */
	if (trace)
		capture_npu_debug(npu, PTA_TNB_DEBUG_PRE_CMD_INPUT);
	if (trace)
		capture_bank_debug(npu, PTA_TNB_DEBUG_BANK_PREPARED, 1);
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
	if (trace)
		capture_npu_debug(npu, PTA_TNB_DEBUG_POST_STATUS);
	if (trace)
		capture_bank_debug(npu, PTA_TNB_DEBUG_BANK_LAUNCH, 2);
	if (trace)
		capture_address_debug(1);
	dev.stage = PTA_TNB_STAGE_SUBMITTED;
	poll_start = profile_now();
	dev.fault = poll_npus();
	profile_add(&profile.poll_calls, &profile.poll_ticks, poll_start);
	if (!dev.fault) {
		dev.debug[PTA_TNB_DEBUG_CNA_POINTER] =
			io_read32(npu + NPU_CNA_POINTER);
		dev.debug[PTA_TNB_DEBUG_CORE_POINTER] =
			io_read32(npu + NPU_CORE_POINTER);
		if (trace)
			capture_config_addresses(npu, PTA_TNB_DEBUG_POST_CMD_INPUT);
		if (trace)
			capture_address_debug(2);
		capture_output_debug();
	}
	if (dev.fault) {
		/* Preserve command-written registers before LOCKED resets the NPU. */
		capture_config_addresses(npu, PTA_TNB_DEBUG_POST_CMD_INPUT);
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
	memcpy((*output).memref.buffer, output_pages, dev.output_size);
	(*output).memref.size = dev.output_size;
	dev.state = PTA_TNB_STATE_ACTIVE;
out:
	profile_add(&profile.run_calls, &profile.run_ticks, profile_start);
	return res;
}


/* Caller holds the core lock and has validated the entire same-A group.
 * Independent semantic tiles share A, but each core has its own B/C binding.
 * C is tile-major, preserving the generic caller's existing result layout. */
static TEE_Result select_gemm_wave(uint32_t first, uint32_t count, uint32_t m,
                                  const TEE_Param *input, bool stage_input)
{
    size_t offset = 0;

    dev.generic_selected = false;
    for (uint32_t core = 0; core < count; core++) {
        uint32_t job = first + core;
        const struct pta_tnb_gemm *g = &dev.generic_gemm[job];
        struct tnb_reg *writes = config[core];

        if (m == g->m)
            memcpy(writes, dev.generic_config[job], sizeof(dev.generic_config[job]));
        else if (!tnb_gemm_emit(m, g->k, g->n, writes))
            return TEE_ERROR_NOT_SUPPORTED;
        writes[TNB_GEMM_A].value = TNB_IOVA_INPUT;
        writes[TNB_GEMM_B].value =
            TNB_IOVA_WEIGHT_SLOT(dev.generic_b[job]) + dev.generic_offset[job];
        writes[TNB_GEMM_C].value = TNB_IOVA_OUTPUT + offset;
        dev.interrupt_mask[core] = 0x300;
        offset += (size_t)m * g->n * sizeof(int32_t);
    }
    dev.core_count = count;
    dev.input_size = input->memref.size;
    dev.output_size = offset;
    record_generic_work();
    if (stage_input) {
        memcpy(input_pages, input->memref.buffer, dev.input_size);
        dcache_clean_range(input_pages, dev.input_size);
    }
    memset(output_pages, 0, output_work_size());
    dcache_clean_range(output_pages, output_work_size());
    dev.generic_selected = true;
    dsb();
    return TEE_SUCCESS;
}

/* One invocation, waves of up to three independent generic tiles. All
 * completion bits in a wave are required before copying its output. */
static TEE_Result exec_gemm_group(void *session, uint32_t types,
                                 TEE_Param params[TEE_NUM_PARAMS])
{
    struct pta_tnb_group_result *result = params[3].memref.buffer;
    uint32_t first = params[0].value.a;
    uint32_t count = params[0].value.b >> 16;
    uint32_t m = params[0].value.b & 0xffff;
    size_t total = 0, offset = 0;
    TEE_Result res = TEE_SUCCESS;
    TEE_Param output = { };

    if (types != TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT,
            TEE_PARAM_TYPE_MEMREF_INPUT, TEE_PARAM_TYPE_MEMREF_OUTPUT,
            TEE_PARAM_TYPE_MEMREF_OUTPUT) || !count || !m ||
        !result || params[3].memref.size != sizeof(*result) ||
        !buffer_in_area(result, sizeof(*result), CORE_MEM_SEC))
        return TEE_ERROR_BAD_PARAMETERS;
    memset(result, 0, sizeof(*result));
    mutex_lock(&lock);
    if (dev.state == PTA_TNB_STATE_LOCKED) {
        res = TEE_ERROR_SECURITY;
        goto out;
    }
    if (dev.owner != session || dev.state != PTA_TNB_STATE_ACTIVE ||
        dev.template_id != PTA_TNB_TEMPLATE_GENERIC ||
        first >= dev.generic_count || count > dev.generic_count - first) {
        res = TEE_ERROR_BAD_STATE;
        goto out;
    }
    /* Complete prevalidation; no device mutation for a malformed group. */
    for (uint32_t i = first; i < first + count; i++) {
        const struct pta_tnb_gemm *g = &dev.generic_gemm[i];
        size_t bytes = (size_t)m * g->n * sizeof(int32_t);
        if (m > g->m || params[1].memref.size != (size_t)m * g->k ||
            bytes > params[2].memref.size - total) {
            res = TEE_ERROR_BAD_PARAMETERS;
            goto out;
        }
        total += bytes;
    }
    if (total != params[2].memref.size ||
        !buffer_in_area(params[1].memref.buffer, params[1].memref.size, CORE_MEM_SEC) ||
        !buffer_in_area(params[2].memref.buffer, total, CORE_MEM_SEC)) {
        res = TEE_ERROR_BAD_PARAMETERS;
        goto out;
    }
    batch_profile.group_calls++;
    for (uint32_t i = first; i < first + count; ) {
        uint32_t width = MIN((uint32_t)PTA_TNB_MAX_CORES, first + count - i);
        size_t bytes = 0;

        for (uint32_t n = 0; n < width; n++)
            bytes += (size_t)m * dev.generic_gemm[i + n].n * sizeof(int32_t);
        result->attempted += width;
        result->phase = PTA_TNB_GROUP_SELECT;
        res = select_gemm_wave(i, width, m, &params[1], i == first);
        if (res)
            goto out;
        output.memref.buffer = (uint8_t *)params[2].memref.buffer + offset;
        output.memref.size = bytes;
        result->phase = PTA_TNB_GROUP_RUN;
        batch_profile.group_jobs += width;
        res = run_locked(session, &output);
        if (res)
            goto out;
        offset += bytes;
        result->completed += width;
        i += width;
    }
out:
    mutex_unlock(&lock);
    return res;
}


/* Caller holds lock; explicit END and session teardown share restoration. */
static TEE_Result close_locked(void *session)
{
	struct tnb_hw_snapshot after = { };
	vaddr_t bussgrf = io_map(BUSSGRF_BASE, BUSSGRF_SIZE);
	vaddr_t firewall = io_map(FIREWALL_DDR_BASE, FIREWALL_DDR_SIZE);
	TEE_Result res = TEE_SUCCESS;
	uint64_t profile_start = profile_now(), tick;
	bool restored = true;
	size_t n = 0;

	if (!bussgrf || !firewall)
		return TEE_ERROR_BAD_PARAMETERS;
	if (dev.state == PTA_TNB_STATE_LOCKED) {
		res = TEE_ERROR_SECURITY;
		goto out;
	}
	if (dev.owner != session) {
		res = TEE_ERROR_ACCESS_DENIED;
		goto out;
	}
	if (dev.state == PTA_TNB_STATE_PREPARED) {
		clear_work();
		memset(&dev, 0, sizeof(dev));
		goto out;
	}
	if (dev.state != PTA_TNB_STATE_ACTIVE) {
		res = TEE_ERROR_BAD_STATE;
		goto out;
	}
	for (n = 0; n < PTA_TNB_MAX_CORES; n++) {
		vaddr_t npu = npu_map(n);

		if (!npu)
			goto lock;
		io_write32(npu + NPU_INT_MASK, 0);
		io_write32(npu + NPU_INT_CLEAR, NPU_INT_ALL);
	}
	dsb();
	tick = profile_now();
	if (!restore_iommu(dev.mmu))
		goto lock;
	batch_profile.close_mmu_ticks += profile_now() - tick;
	tick = profile_now();
	clear_work();
	batch_profile.close_clear_ticks += profile_now() - tick;
	tick = profile_now();
	for (n = 0; n < PTA_TNB_MAX_CORES; n++) {
		vaddr_t npu = npu_map(n);

		if (!npu) {
			restored = false;
			continue;
		}
		io_write32(npu + NPU_INT_MASK, dev.old_npu_mask[n]);
	}
	if (!restored || !restore_secure(bussgrf, firewall, &dev.before))
		goto lock;
	for (n = 0; n < PTA_TNB_MAX_CORES; n++)
		restored = restore_irq(n, dev.irq_enabled[n]) && restored;
	if (!restored || snapshot(&after) ||
	    !snapshot_equal(&dev.before, &after))
		goto lock;
	batch_profile.close_restore_ticks += profile_now() - tick;
	dev.security_touched = false;
	memset(&dev, 0, sizeof(dev));
	dev.stage = PTA_TNB_STAGE_RESTORED;
	goto out;
lock:
	enter_locked(PTA_TNB_FAULT_RESTORE);
	res = TEE_ERROR_SECURITY;
out:
	profile_add(&profile.close_calls, &profile.close_ticks, profile_start);
	return res;
}

static TEE_Result close_device(void *session, uint32_t types)
{
    TEE_Result res;
    if (types != TEE_PARAM_TYPES(0,0,0,0))
        return TEE_ERROR_BAD_PARAMETERS;
    mutex_lock(&lock);
    res = close_locked(session);
    mutex_unlock(&lock);
    return res;
}

static TEE_Result get_batch_profile(uint32_t types,
	TEE_Param params[TEE_NUM_PARAMS])
{
	if (types != TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT, 0, 0, 0))
		return TEE_ERROR_BAD_PARAMETERS;
	if (params[0].memref.size < sizeof(batch_profile)) {
		params[0].memref.size = sizeof(batch_profile);
		return TEE_ERROR_SHORT_BUFFER;
	}
	mutex_lock(&lock);
	memcpy(params[0].memref.buffer, &batch_profile, sizeof(batch_profile));
	params[0].memref.size = sizeof(batch_profile);
	mutex_unlock(&lock);
	return TEE_SUCCESS;
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
	memset(&batch_profile, 0, sizeof(batch_profile));
	batch_profile.version = PTA_TNB_BATCH_PROFILE_VERSION;
	batch_profile.counter_frequency = profile.counter_frequency;
	mutex_unlock(&lock);
	*session = calloc(1, sizeof(struct tnb_session));
	if (!*session)
		return TEE_ERROR_OUT_OF_MEMORY;
	return TEE_SUCCESS;
}

static void close_session(void *session)
{
	struct tnb_session *tnb_session = session;

	mutex_lock(&lock);
	if (dev.owner == session) {
		if (dev.state == PTA_TNB_STATE_PREPARED ||
		    dev.state == PTA_TNB_STATE_ACTIVE) {
			if (close_locked(session) && dev.state != PTA_TNB_STATE_LOCKED)
				enter_locked(PTA_TNB_FAULT_RESTORE);
		} else if (dev.state == PTA_TNB_STATE_RUNNING) {
			enter_locked(PTA_TNB_FAULT_CALLER);
		}
	}
	/* A LOCKED device may retain PTE access; quarantine registrations. */
	if (dev.state != PTA_TNB_STATE_LOCKED)
		release_registered_weights(tnb_session);
	mutex_unlock(&lock);
	free(tnb_session->plans);
	memset(tnb_session, 0, sizeof(*tnb_session));
	free(session);
}

static TEE_Result invoke_command(void *session, uint32_t command,
                                uint32_t types, TEE_Param params[TEE_NUM_PARAMS])
{
    switch (command) {
    case PTA_TNB_CMD_REGISTER_WEIGHT:
        return register_weight(session, types, params);
    case PTA_TNB_CMD_PREPARE_REGISTERED_BATCH:
        return prepare_registered_batch(session, types, params);
    case PTA_TNB_CMD_EXEC_GEMM_GROUP:
        return exec_gemm_group(session, types, params);
    case PTA_TNB_CMD_ACTIVATE:
        return activate(session, types);
    case PTA_TNB_CMD_CLOSE:
        return close_device(session, types);
    case PTA_TNB_CMD_GET_CAPS:
        return get_caps(types, params);
    case PTA_TNB_CMD_GET_STATUS:
        return get_status(types, params);
    case PTA_TNB_CMD_GET_BATCH_PROFILE:
        return get_batch_profile(types, params);
    default:
        return TEE_ERROR_NOT_SUPPORTED;
    }
}

pseudo_ta_register(.uuid = PTA_TNB_UUID, .name = "tnb",
		   .flags = PTA_DEFAULT_FLAGS,
		   .open_session_entry_point = open_session,
		   .close_session_entry_point = close_session,
		   .invoke_command_entry_point = invoke_command);
