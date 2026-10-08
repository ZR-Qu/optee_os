// SPDX-License-Identifier: BSD-2-Clause
#include <arm.h>
#include <io.h>
#include "tnb_npu.h"

/* Replay every trusted write in order, including repeated register values. */
void tnb_npu_prepare(vaddr_t base, size_t core, const struct tnb_reg config[TNB_GEMM_WRITES],
                     uint32_t interrupt_mask)
{
    size_t n = 0;

    io_write32(base + 0x10, 1); /* AHB/slave configuration mode */
    io_write32(base + 0x1004, 0xe + 0x10000000 * core);
    io_write32(base + 0x3004, 0xe + 0x10000000 * core);
    io_write32(base + 0x20, interrupt_mask);
    io_write32(base + 0x24, interrupt_mask);
    dsb();
    for (n = 0; n < TNB_GEMM_WRITES; n++) {
        io_write32(base + config[n].offset, config[n].value);
        dsb();
    }
}

void tnb_npu_start(vaddr_t base)
{
    /* 0xd is the trusted family's unit enable, not PC DMA start=1. */
    io_write32(base + 0xf008, 0xd);
    dsb();
    (void)io_read32(base + 0xf008);
}
