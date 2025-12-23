//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "xrt_test_wrapper.h"
#include <cstdint>
#include <cassert>

//*****************************************************************************
// Modify this section to customize buffer datatypes, initialization functions,
// and verify function. The other place to reconfigure your design is the
// Makefile.
//*****************************************************************************

// ------------------------------------------------------
// Configure this to match your buffer data type
// ------------------------------------------------------
using int32_t = std::int32_t;
using int32_t = std::int32_t;
using int32_t = std::int32_t;

const int VEC_B_PER_CORE = 1;
const int N_CORE_ROW = 2;
const int N_CORE_COL = 1;
const int N_SUBVEC = 4;
const int DEGREE = 4096;
const int DEGREE_PER_SUBVEC = DEGREE / N_SUBVEC;

void initialize_vec_A(int32_t *buff, int SIZE)
{
  assert(SIZE == DEGREE);
  for (int i = 0; i < DEGREE; i++)
  {
    buff[i] = i;
  }
}

void transform_input_matrix(int32_t *dst, int32_t *original)
{
  for (int subvec = 0; subvec < N_SUBVEC; subvec++)
  {
    int32_t *p_subvec = dst + subvec * N_CORE_COL * N_CORE_ROW * DEGREE_PER_SUBVEC * VEC_B_PER_CORE;
    for (int col = 0; col < N_CORE_COL; col++)
    {
      int32_t *p_col = p_subvec + col * N_CORE_ROW * DEGREE_PER_SUBVEC * VEC_B_PER_CORE;
      for (int row = 0; row < N_CORE_ROW; row++)
      {
        int32_t *p_row = p_col + row * DEGREE_PER_SUBVEC * VEC_B_PER_CORE;
        for (int i = 0; i < DEGREE_PER_SUBVEC; i++)
        {
          int orig_index = (col * N_CORE_ROW + row) * VEC_B_PER_CORE * DEGREE +
                           subvec * DEGREE_PER_SUBVEC + i;
          p_row[i] = original[orig_index];
        }
      }
    }
  }
}

void initialize_vec_B(int32_t *buff, int SIZE)
{
  assert(SIZE == N_CORE_COL * N_CORE_ROW * VEC_B_PER_CORE * DEGREE);

  for (int col = 0; col < N_CORE_COL; col++)
  {
    for (int row = 0; row < N_CORE_ROW; row++)
    {
      for (int elem = 0; elem < VEC_B_PER_CORE; elem++)
      {
        const int idx_core = col * N_CORE_ROW + row;
        int32_t *ptr = buff + idx_core * VEC_B_PER_CORE * DEGREE;
        for (int i = 0; i < DEGREE; i++)
        {
          ptr[i] = idx_core;
        }
      }
    }
  }
}

void initialize_bufOut(int32_t *bufOut, int SIZE)
{
  memset(bufOut, 0, SIZE);
}

// Functional correctness verifyer
int verify(int32_t *bufIn1, int32_t *bufIn2,
           int32_t *bufOut, int SIZE, int verbosity)
{
  std::cout << std::dec;
  int errors = 0;
  for (int row = 0; row < 2; row++)
  {
    std::cout << "out[row " << row << "] = " << bufOut[row] << std::endl;
  }

  for (int col = 0; col < N_CORE_COL; col++)
  {
    int32_t *p_col = bufIn2 + col * N_CORE_ROW * VEC_B_PER_CORE * DEGREE;
    for (int row = 0; row < N_CORE_ROW; row++)
    {
      int32_t *p_row = p_col + row * VEC_B_PER_CORE * DEGREE;
      for (int elem = 0; elem < VEC_B_PER_CORE; elem++)
      {
        int32_t *p_elem = p_row + elem * DEGREE;
        int sum = 0;
        for (int i = 0; i < DEGREE; i++)
        {
          sum += (p_elem[i] * bufIn1[i]);
        }

        // Check
        if (bufOut[row + col * N_CORE_ROW + elem] != sum)
        {
          if (verbosity >= 1)
          {
            std::cout << "Error in out[" << row + col * N_CORE_ROW + elem << "]="
                      << bufOut[row + col * N_CORE_ROW + elem]
                      << ", expected " << sum << std::endl;
          }
          errors++;
        }
        else
        {
          if (verbosity >= 1)
            std::cout << "Correct output " << bufOut[row + col * N_CORE_ROW + elem]
                      << " == " << sum << std::endl;
        }
      }
    }
  }

  return errors;
}

template <typename T1, typename T2, typename T3, void (*init_bufIn1)(T1 *, int),
          void (*init_bufIn2)(T2 *, int), void (*init_bufOut)(T3 *, int),
          int (*verify_results)(T1 *, T2 *, T3 *, int, int)>
int xx_setup_and_run_aie(int IN1_VOLUME, int IN2_VOLUME, int OUT_VOLUME,
                         struct args myargs, bool enable_ctrl_pkts = false)
{

  srand(time(NULL));

  // ==============================================
  // Load instruction sequence
  // ==============================================
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(myargs.instr);
  if (myargs.verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // ==============================================
  // Start the XRT context and load the kernel
  // ==============================================
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, myargs.verbosity,
                                   myargs.xclbin, myargs.kernel);

  // ==============================================
  // set up the buffer objects
  // ==============================================
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in1 = xrt::bo(device, IN1_VOLUME * sizeof(T1), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
  auto bo_in2 = xrt::bo(device, IN2_VOLUME * sizeof(T2), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));
  auto bo_out = xrt::bo(device, OUT_VOLUME * sizeof(T3), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(5));

  // If we enable control packets, then this is the input xrt buffer for that.
  // Otherwise, this is a dummy placedholder buffer.
  auto bo_ctrlpkts =
      xrt::bo(device, 8, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  // Workaround so we declare a really small trace buffer when one is not used
  // Second workaround for driver issue. Allocate large trace buffer *4
  // This includes the 8 bytes needed for control packet response.
  int tmp_trace_size = (myargs.trace_size > 0) ? myargs.trace_size * 4 : 1;
  auto bo_trace = xrt::bo(device, tmp_trace_size, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(7));

  if (myargs.verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // ==============================================
  // Initialize inputs
  // ==============================================
  T1 *bufIn1 = bo_in1.map<T1 *>();
  T1 *bufIn2 = bo_in2.map<T1 *>();
  T3 *bufOut = bo_out.map<T3 *>();
  T1 *bufIn2_original = new T1[IN2_VOLUME];
  char *bufTrace = bo_trace.map<char *>();
  uint32_t *bufCtrlPkts = bo_ctrlpkts.map<uint32_t *>();

  init_bufIn1(bufIn1, IN1_VOLUME);
  init_bufIn2(bufIn2_original, IN2_VOLUME);
  init_bufOut(bufOut, OUT_VOLUME);
  transform_input_matrix(bufIn2, bufIn2_original);

  if (myargs.trace_size > 0)
    memset(bufTrace, 0, myargs.trace_size);

  // Set control packet values
  if (myargs.trace_size > 0 && enable_ctrl_pkts)
  {
    bufCtrlPkts[0] = create_ctrl_pkt(1, 0, 0x32004); // core status
    bufCtrlPkts[1] = create_ctrl_pkt(1, 0, 0x320D8); // trace status
    if (myargs.verbosity >= 1)
    {
      std::cout << "bufCtrlPkts[0]:" << std::hex << bufCtrlPkts[0] << std::endl;
      std::cout << "bufCtrlPkts[1]:" << std::hex << bufCtrlPkts[1] << std::endl;
    }
  }

  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (myargs.trace_size > 0)
  {
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (enable_ctrl_pkts)
      bo_ctrlpkts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  // ==============================================
  // Initialize run configs
  // ==============================================
  unsigned num_iter = myargs.n_iterations + myargs.n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  int errors = 0;

  // ==============================================
  // Main run loop
  // ==============================================
  for (unsigned iter = 0; iter < num_iter; iter++)
  {

    if (myargs.verbosity >= 1)
      std::cout << "Running Kernel.\n";

    // ==============================================
    // Run kernel
    // ==============================================
    if (myargs.verbosity >= 1)
      std::cout << "Running Kernel.\n";
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in1, bo_in2, bo_out,
                      bo_ctrlpkts, bo_trace);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (myargs.trace_size > 0)
      bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < myargs.n_warmup_iterations)
      /* Warmup iterations do not count towards average runtime. */
      continue;

    // ==============================================
    // Verify results
    // ==============================================
    if (myargs.do_verify)
    {
      if (myargs.verbosity >= 1)
      {
        std::cout << "Verifying results ..." << std::endl;
      }
      auto vstart = std::chrono::system_clock::now();

      errors +=
          verify_results(bufIn1, bufIn2_original, bufOut, IN1_VOLUME, myargs.verbosity);

      auto vstop = std::chrono::system_clock::now();
      float vtime =
          std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
              .count();
      if (myargs.verbosity >= 1)
        std::cout << "Verify time: " << vtime << "secs." << std::endl;
    }
    else
    {
      if (myargs.verbosity >= 1)
        std::cout << "WARNING: results not verified." << std::endl;
    }

    // ==============================================
    // Output trace
    // ==============================================
    if (myargs.trace_size > 0 && iter == myargs.n_warmup_iterations)
    {
      test_utils::write_out_trace(((char *)bufTrace), myargs.trace_size,
                                  myargs.trace_file);
    }

    // ==============================================
    // Output control packet
    // ==============================================
    if (enable_ctrl_pkts)
    {
      uint32_t *ctrl_pkt_out =
          (uint32_t *)(((char *)bufTrace) + myargs.trace_size);
      if (myargs.verbosity >= 1)
      {
        std::cout << "ctrl_pkt_out[0]:" << std::hex << ctrl_pkt_out[0]
                  << std::endl;
        std::cout << "ctrl_pkt_out[1]:" << std::hex << ctrl_pkt_out[1]
                  << std::endl;
      }
      int col = (ctrl_pkt_out[0] >> 21) & 0x7F;
      int row = (ctrl_pkt_out[0] >> 16) & 0x1F;
      if ((ctrl_pkt_out[1] >> 8) == 3)
        std::cout << "WARNING: Trace overflow detected in tile(" << row << ","
                  << col << ". Trace results may be invalid." << std::endl;
    }

    // Accumulate run times
    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  // ==============================================
  // Print verification and timing results
  // ==============================================

  // TODO - Mac count to guide gflops
  float macs = 0;

  std::cout << std::endl
            << "Avg NPU time: " << npu_time_total / myargs.n_iterations << "us."
            << std::endl;
  if (macs > 0)
    std::cout << "Avg NPU gflops: "
              << macs / (1000 * npu_time_total / myargs.n_iterations)
              << std::endl;

  std::cout << std::endl
            << "Min NPU time: " << npu_time_min << "us." << std::endl;
  if (macs > 0)
    std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min)
              << std::endl;

  std::cout << std::endl
            << "Max NPU time: " << npu_time_max << "us." << std::endl;
  if (macs > 0)
    std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max)
              << std::endl;

  if (!errors)
  {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  else
  {
    std::cout << "\nError count: " << errors << "\n\n";
    std::cout << "\nFailed.\n\n";
    return 1;
  }
}

int main(int argc, const char *argv[])
{

  constexpr int IN1_VOLUME = DEGREE;
  constexpr int IN2_VOLUME = DEGREE * VEC_B_PER_CORE * N_CORE_ROW * N_CORE_COL;
  constexpr int OUT_VOLUME = VEC_B_PER_CORE * N_CORE_ROW * N_CORE_COL;

  args myargs = parse_args(argc, argv);

  int res = xx_setup_and_run_aie<int32_t, int32_t, int32_t,
                                 initialize_vec_A, initialize_vec_B,
                                 initialize_bufOut, verify>(
      IN1_VOLUME, IN2_VOLUME, OUT_VOLUME, myargs, true);
  return res;
}
