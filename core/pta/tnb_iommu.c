// SPDX-License-Identifier: BSD-2-Clause
#include <arm.h>
#include <io.h>
#include <kernel/delay.h>

#include "tnb_iommu.h"

#define ALL_BANKS ((1U << TNB_IOMMU_BANKS) - 1)

bool tnb_iommu_wait(const vaddr_t bases[TNB_IOMMU_BANKS], uint32_t banks,
		    uint32_t bit, bool set)
{
	uint64_t deadline = timeout_init_us(1000);

	while (!timeout_elapsed(deadline)) {
		bool ready = true;
		size_t n = 0;

		for (n = 0; n < TNB_IOMMU_BANKS; n++) {
			if (!(banks & (1U << n)))
				continue;
			if (!bases[n] ||
			    !!(io_read32(bases[n] + IOMMU_STATUS) & bit) != set)
				ready = false;
		}
		if (ready)
			return true;
		udelay(1);
	}
	return false;
}

static bool valid_bases(const vaddr_t bases[TNB_IOMMU_BANKS])
{
	size_t n = 0;

	for (n = 0; n < TNB_IOMMU_BANKS; n++)
		if (!bases[n])
			return false;
	return true;
}

/* Rockchip enable_stall explicitly requires paging to be enabled.
 * Wait for exactly the banks to which the stall request was sent.
 */
static bool stop_paging(const vaddr_t bases[TNB_IOMMU_BANKS])
{
	uint32_t paging_banks = 0;
	size_t n = 0;

	for (n = 0; n < TNB_IOMMU_BANKS; n++) {
		io_write32(bases[n] + IOMMU_INT_MASK, 0);
		if (io_read32(bases[n] + IOMMU_STATUS) & IOMMU_STATUS_PAGING) {
			paging_banks |= 1U << n;
			io_write32(bases[n] + IOMMU_COMMAND,
				   IOMMU_CMD_ENABLE_STALL);
		}
	}
	dsb();
	if (!tnb_iommu_wait(bases, paging_banks, IOMMU_STATUS_STALL, true))
		return false;
	for (n = 0; n < TNB_IOMMU_BANKS; n++)
		if (paging_banks & (1U << n))
			io_write32(bases[n] + IOMMU_COMMAND,
				   IOMMU_CMD_DISABLE_PAGING);
	dsb();
	return tnb_iommu_wait(bases, ALL_BANKS, IOMMU_STATUS_PAGING, false);
}

static bool unstall(const vaddr_t bases[TNB_IOMMU_BANKS])
{
	size_t n = 0;

	for (n = 0; n < TNB_IOMMU_BANKS; n++)
		io_write32(bases[n] + IOMMU_COMMAND, IOMMU_CMD_DISABLE_STALL);
	dsb();
	return tnb_iommu_wait(bases, ALL_BANKS, IOMMU_STATUS_STALL, false);
}

bool tnb_iommu_disable(const vaddr_t bases[TNB_IOMMU_BANKS],
		       struct tnb_mmu_save saved[TNB_IOMMU_BANKS])
{
	size_t n = 0;

	if (!valid_bases(bases))
		return false;
	for (n = 0; n < TNB_IOMMU_BANKS; n++) {
		saved[n].dte = io_read32(bases[n] + IOMMU_DTE);
		saved[n].int_mask = io_read32(bases[n] + IOMMU_INT_MASK);
		saved[n].auto_gating = io_read32(bases[n] + IOMMU_AUTO_GATING);
		saved[n].paging = !!(io_read32(bases[n] + IOMMU_STATUS) &
				    IOMMU_STATUS_PAGING);
	}
	if (!stop_paging(bases))
		return false;
	for (n = 0; n < TNB_IOMMU_BANKS; n++) {
		io_write32(bases[n] + IOMMU_DTE, 0);
		io_write32(bases[n] + IOMMU_INT_CLEAR, IOMMU_FAULT);
	}
	dsb();
	return unstall(bases);
}

/* Linux rk_iommu_enable resets the MMU before installing a domain. Use the
 * trusted DTE address as a nonzero marker while paging is off, so a stale
 * zero DTE cannot satisfy the reset acknowledgement before reset executes.
 * The v2 DTE register does not read back the descriptor valid bit.
 */
static bool reset_mmu(const vaddr_t bases[TNB_IOMMU_BANKS], uint32_t dte)
{
	uint64_t deadline = 0;
	size_t n = 0;

	for (n = 0; n < TNB_IOMMU_BANKS; n++) {
		io_write32(bases[n] + IOMMU_DTE, dte);
		if ((io_read32(bases[n] + IOMMU_DTE) & ~1U) != (dte & ~1U))
			return false;
	}
	for (n = 0; n < TNB_IOMMU_BANKS; n++)
		io_write32(bases[n] + IOMMU_COMMAND, IOMMU_CMD_FORCE_RESET);
	dsb();
	deadline = timeout_init_us(100000);
	while (!timeout_elapsed(deadline)) {
		bool done = true;

		for (n = 0; n < TNB_IOMMU_BANKS; n++)
			if (io_read32(bases[n] + IOMMU_DTE))
				done = false;
		if (done)
			return true;
		udelay(100);
	}
	return false;
}

bool tnb_iommu_install(const vaddr_t bases[TNB_IOMMU_BANKS], uint32_t dte)
{
	size_t n = 0;

	if (!valid_bases(bases) || !reset_mmu(bases, dte))
		return false;
	for (n = 0; n < TNB_IOMMU_BANKS; n++) {
		io_write32(bases[n] + IOMMU_DTE, dte);
		io_write32(bases[n] + IOMMU_COMMAND, IOMMU_CMD_ZAP_CACHE);
		io_write32(bases[n] + IOMMU_INT_CLEAR, IOMMU_FAULT);
		io_write32(bases[n] + IOMMU_INT_MASK, IOMMU_FAULT);
		io_setbits32(bases[n] + IOMMU_AUTO_GATING,
			     IOMMU_DISABLE_FETCH_LIMIT);
		io_write32(bases[n] + IOMMU_COMMAND, IOMMU_CMD_ENABLE_PAGING);
	}
	dsb();
	return tnb_iommu_wait(bases, ALL_BANKS, IOMMU_STATUS_PAGING, true);
}

bool tnb_iommu_restore(const vaddr_t bases[TNB_IOMMU_BANKS],
		       const struct tnb_mmu_save saved[TNB_IOMMU_BANKS])
{
	uint32_t paging_banks = 0;
	size_t n = 0;

	if (!valid_bases(bases) || !stop_paging(bases))
		return false;
	/* Do not replace DTE until paging-off has been observed on every bank. */
	for (n = 0; n < TNB_IOMMU_BANKS; n++) {
		io_write32(bases[n] + IOMMU_DTE, saved[n].dte);
		io_write32(bases[n] + IOMMU_COMMAND, IOMMU_CMD_ZAP_CACHE);
		io_write32(bases[n] + IOMMU_AUTO_GATING, saved[n].auto_gating);
		if (saved[n].paging) {
			paging_banks |= 1U << n;
			io_write32(bases[n] + IOMMU_COMMAND,
				   IOMMU_CMD_ENABLE_PAGING);
		}
	}
	dsb();
	if (!tnb_iommu_wait(bases, paging_banks, IOMMU_STATUS_PAGING, true) ||
	    !tnb_iommu_wait(bases, ALL_BANKS & ~paging_banks,
			    IOMMU_STATUS_PAGING, false))
		return false;
	for (n = 0; n < TNB_IOMMU_BANKS; n++)
		io_write32(bases[n] + IOMMU_INT_MASK, saved[n].int_mask);
	if (!unstall(bases))
		return false;
	for (n = 0; n < TNB_IOMMU_BANKS; n++)
		if (io_read32(bases[n] + IOMMU_DTE) != saved[n].dte)
			return false;
	return true;
}
