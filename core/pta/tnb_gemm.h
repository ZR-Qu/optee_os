/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef TNB_GEMM_H
#define TNB_GEMM_H

#include <stdbool.h>
#include <stdint.h>
#include <pta_tnb.h>

#define TNB_GEMM_WRITES 108

struct tnb_reg {
	uint32_t offset;
	uint32_t value;
};
#define TNB_GEMM_A 25
#define TNB_GEMM_B 34
#define TNB_GEMM_C 61

bool tnb_gemm_emit(uint32_t m, uint32_t k, uint32_t n,
		   struct tnb_reg out[TNB_GEMM_WRITES]);

#endif
