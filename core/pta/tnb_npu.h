/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef TNB_NPU_H
#define TNB_NPU_H

#include <stddef.h>
#include <stdint.h>
#include <types_ext.h>
#include "tnb_gemm.h"

/* Caller holds the device lock and supplies a validated ACTIVE job.
 * The trusted config is CPU-only; no command DMA or config cache clean.
 */
void tnb_npu_prepare(vaddr_t base, size_t core, const struct tnb_reg config[TNB_GEMM_WRITES],
		     uint32_t interrupt_mask);
void tnb_npu_start(vaddr_t base);

#endif
