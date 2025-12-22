//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

extern "C"
{
  void vector_accum(int32_t *in_a, int32_t *in_b, int32_t *out_c,
                    int32_t N)
  {
    event0();

    // Initialize variables
    constexpr int vec_factor = 16;
    int32_t *__restrict pA1 = in_a;
    int32_t *__restrict pB1 = in_b;
    int32_t *__restrict pC1 = out_c;
    const int F = N / vec_factor;

    // Accumulate using vector operations
    aie::vector<int32_t, vec_factor> partial_sum = aie::zeros<int32_t, vec_factor>();
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int i = 0; i < F; i++)
    {
      aie::vector<int32_t, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
      pA1 += vec_factor;
      partial_sum = aie::add(partial_sum, A0);
    }

    // Accumulate to int32
    int sum = 0;
    for (int i = 0; i < vec_factor; i++)
    {
      sum += partial_sum[i];
    }

    // Store result
    out_c[0] += sum;
    event1();
  }

  void store_buff(int32_t *buff, int32_t offset, int32_t value)
  {
    buff[offset] = value;
  }

  void zeros(int32_t *buff, int32_t size)
  {
    for (int i = 0; i < size; i++)
    {
      buff[i] = 0;
    }
  }
} // extern "C"
