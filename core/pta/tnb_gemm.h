/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef TNB_GEMM_H
#define TNB_GEMM_H

#include <stdbool.h>
#include <stdint.h>

#define TNB_GEMM_COMMANDS 108
#define TNB_GEMM_WORDS 112
#define TNB_GEMM_A 25
#define TNB_GEMM_B 34
#define TNB_GEMM_C 61
#define TNB_GEMM_M_MAX 4u
#define TNB_GEMM_K_MAX 6208u
#define TNB_GEMM_B_MAX 4545536u
#define TNB_GEMM_N_MIN 672u
#define TNB_GEMM_N_MAX 2048u
#define TNB_GEMM_N_ALIGN 32u

bool tnb_gemm_emit(uint32_t m, uint32_t k, uint32_t n,
		   uint64_t out[TNB_GEMM_WORDS]);

#endif
