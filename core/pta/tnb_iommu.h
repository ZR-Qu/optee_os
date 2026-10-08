/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef TNB_IOMMU_H
#define TNB_IOMMU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <types_ext.h>

#define TNB_IOMMU_BANKS 4
#define IOMMU_DTE 0x0
#define IOMMU_STATUS 0x4
#define IOMMU_COMMAND 0x8
#define IOMMU_PAGE_FAULT_ADDR 0xc
#define IOMMU_INT_RAW_STATUS 0x14
#define IOMMU_INT_CLEAR 0x18
#define IOMMU_INT_MASK 0x1c
#define IOMMU_INT_STATUS 0x20
#define IOMMU_AUTO_GATING 0x24
#define IOMMU_STATUS_PAGING (1U << 0)
#define IOMMU_STATUS_FAULT (1U << 1)
#define IOMMU_STATUS_STALL (1U << 2)
#define IOMMU_FAULT ((1U << 0) | (1U << 1))
#define IOMMU_CMD_ENABLE_PAGING 0
#define IOMMU_CMD_DISABLE_PAGING 1
#define IOMMU_CMD_ENABLE_STALL 2
#define IOMMU_CMD_DISABLE_STALL 3
#define IOMMU_CMD_ZAP_CACHE 4
#define IOMMU_CMD_FORCE_RESET 6
#define IOMMU_DISABLE_FETCH_LIMIT (1U << 31)

struct tnb_mmu_save {
	uint32_t dte;
	uint32_t int_mask;
	uint32_t auto_gating;
	bool paging;
};

/* Caller provides powered/clocked banks and a quiescent device, holds the
 * device lock, and maintains page-table cache visibility and backing lifetime.
 * Install follows a successful disable with paging off; restore follows an
 * ACTIVE window with no further NPU submissions. These functions do not
 * provide Linux job/IRQ/domain exclusion or a ZAP completion acknowledgement.
 */
bool tnb_iommu_wait(const vaddr_t bases[TNB_IOMMU_BANKS], uint32_t banks,
		    uint32_t bit, bool set);
bool tnb_iommu_disable(const vaddr_t bases[TNB_IOMMU_BANKS],
		       struct tnb_mmu_save saved[TNB_IOMMU_BANKS]);
bool tnb_iommu_install(const vaddr_t bases[TNB_IOMMU_BANKS], uint32_t dte);
bool tnb_iommu_restore(const vaddr_t bases[TNB_IOMMU_BANKS],
		       const struct tnb_mmu_save saved[TNB_IOMMU_BANKS]);

#endif
