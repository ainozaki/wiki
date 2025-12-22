# vector_scalar_mul/vector_scalar_mul_placed.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from aie.extras.dialects.ext import memref, arith

dev = AIEDevice.npu1
N = 4096
N_bytes = 4096 * 4

def vec_matrix_mult(dev, trace_size):

    in1_dtype = np.int32
    out_dtype = np.int32
    in2_dtype = np.int32

    n_rows = 2
    n_cols = 1

    k_per_core = 1

    tensor_size = N
    matrix_size = N * n_rows
    num_sub_vectors = 4

    tiled_tensor_size = tensor_size // num_sub_vectors
    tiled_matrix_size = tiled_tensor_size * n_rows

    output_per_core = k_per_core
    output_size = n_rows * output_per_core
    
    vectorized = True


    @device(dev)
    def device_body():
        tile_ty = np.ndarray[(tiled_tensor_size,), np.dtype[in1_dtype]]    
        tile_matrix_ty = np.ndarray[(tiled_matrix_size,), np.dtype[in2_dtype]]    
        
        output_ty = np.ndarray[(output_size,), np.dtype[out_dtype]]
        output_ty_per_core = np.ndarray[(k_per_core,), np.dtype[out_dtype]]

        ctrl_pkt_ty = np.ndarray[(1,), np.dtype[np.uint32]]
        trace_ty = np.ndarray[(trace_size,), np.dtype[np.uint8]]

        buff_ty = np.ndarray[(128,), np.dtype[np.int32]]

    
        # ==============================================
        # AIE Core Function declarations
        # ==============================================
        ## void vector_accum(int32_t *in_a, int32_t *in_b, int32_t *out_c, int32_t N)
        accum_func = external_func(
            f"vector_accum",
            inputs=[tile_ty, tile_ty, output_ty_per_core, np.int32],
        )

        ## void store_buff(int32_t *buff, int32_t offset, int32_t value)
        store_buff_func = external_func(
            f"store_buff",
            inputs=[buff_ty, np.int32, np.int32],
        )

        ## void zeros(int32_t *buff, int32_t size)
        zeros_func = external_func(
            f"zeros",
            inputs=[output_ty_per_core, np.int32],
        )

        # ==============================================
        # Tile declarations
        # ==============================================
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        CtrlShimTile = tile(1, 0)
        ComputeTiles = [tile(0, 2 + i) for i in range(n_rows)]
        
        # ==============================================
        # Object FIFO declarations
        # ==============================================
        of_in_shim_mem = object_fifo(
            "in_shim_mem", ShimTile, MemTile, 2, tile_ty
        )
        of_in2_shim_mem = object_fifo(
            "in2_shim_mem", ShimTile, MemTile, 2, tile_matrix_ty
        )
        of_out_mem_shim = object_fifo(
            "out_mem_shim", MemTile, ShimTile, 2, output_ty
        )

        # in1
        of_in_broadcast = object_fifo("in_mem_ct_broadcast", MemTile, ComputeTiles, 2, tile_ty)
        object_fifo_link(of_in_shim_mem, [of_in_broadcast], [], [0])

        # in2
        of_in2_list = []
        for r in range(n_rows):
            of_in2_list.append(
                object_fifo(f"in2_mem_ct{r}", MemTile, ComputeTiles[r], 2, tile_ty)
            )
        object_fifo_link(of_in2_shim_mem, of_in2_list, [], [tiled_tensor_size * r for r in range(n_rows)])
        
        # out
        of_out_list = []
        for r in range(n_rows):
            of_out_list.append(
                object_fifo(f"out_ct{r}_mem", ComputeTiles[r], MemTile, 2, output_ty_per_core)
            )
        object_fifo_link(of_out_list, of_out_mem_shim, [k_per_core * r for r in range(n_rows)], [])

        # ==============================================
        # Buff declarations
        # ==============================================
        buffs = []
        for r in range(n_rows):
            buffs.append(buffer(ComputeTiles[r], np.ndarray[(128,), np.dtype[np.int32]] ,f"buff_ct{r}"))
        
        
        # ==============================================
        # Compute Tile Main Loops
        # ==============================================
        for r in range(n_rows):
            @core(ComputeTiles[r], "kernel.o")
            def core_body():
                # Effective while(1)
                store_buff_func(buffs[r], 0, 0)
                for _ in range_(sys.maxsize):
                    # Number of sub-vector "tile" iterations
                    elem_out = of_out_list[r].acquire(ObjectFifoPort.Produce, 1)
                    zeros_func(elem_out, 1)
                    for loop in range(num_sub_vectors):
                        elem_in = of_in_broadcast.acquire(ObjectFifoPort.Consume, 1)
                        elem_in2 = of_in2_list[r].acquire(ObjectFifoPort.Consume, 1)
                        accum_func(elem_in, elem_in2, elem_out, tiled_tensor_size)
                        of_in_broadcast.release(ObjectFifoPort.Consume, 1)
                        of_in2_list[r].release(ObjectFifoPort.Consume, 1)
                    of_out_list[r].release(ObjectFifoPort.Produce, 1)


        # ==============================================
        # Setup trace trace
        # ==============================================
        tiles_to_trace = [ComputeTiles[r] for r in range(n_rows)]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)
            trace_utils.configure_packet_ctrl_flow([ComputeTiles[0]], CtrlShimTile)

        trace_size_int32 = trace_size // np.dtype(np.int32).itemsize


        # ==============================================
        # Runtime Sequence
        # ==============================================
        tensor_ty = np.ndarray[(tensor_size,), np.dtype[in1_dtype]]
        matrix_ty = np.ndarray[(matrix_size,), np.dtype[in2_dtype]]
        @runtime_sequence(tensor_ty, matrix_ty, output_ty)
        def sequence(A, B, C):
            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTile,
                    trace_size=trace_size_int32,
                    coretile_events=[
                        CoreEvent.INSTR_EVENT_0,
                        CoreEvent.INSTR_EVENT_1,
                        CoreEvent.INSTR_VECTOR,
                        PortEvent(CoreEvent.PORT_RUNNING_0, 1, True),  # master(1)
                        PortEvent(CoreEvent.PORT_RUNNING_1, 2, True),  # master(2)
                        PortEvent(CoreEvent.PORT_RUNNING_2, 1, False),  # slave(1)
                        CoreEvent.INSTR_LOCK_ACQUIRE_REQ,
                        CoreEvent.LOCK_STALL,
                    ],
                    coremem_events=[
                        MemEvent.GROUP_MEMORY_CONFLICT,
                        MemEvent.DMA_MM2S_0_FINISHED_BD,
                        MemEvent.DMA_S2MM_0_FINISHED_BD,
                        MemEvent.DMA_S2MM_1_FINISHED_BD,
                        MemEvent.LOCK_3_REL,
                        MemEvent.DMA_MM2S_0_STREAM_BACKPRESSURE,
                        MemEvent.LOCK_SEL0_ACQ_GE,
                        MemEvent.LOCK_SEL1_ACQ_EQ,
                    ],
                )

            in_task = shim_dma_single_bd_task(
                of_in_shim_mem, A, sizes=[1, 1, 1, tensor_size], issue_token=True
            )
            in2_task = shim_dma_single_bd_task(
                of_in2_shim_mem, B, sizes=[1, 1, 1, matrix_size], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out_mem_shim, C, sizes=[1, 1, 1, output_size], issue_token=True
            )

            dma_start_task(in_task, in2_task, out_task)
            dma_await_task(in_task, in2_task, out_task)

            if trace_size > 0:
                trace_utils.config_ctrl_pkts_aie(
                    [ComputeTiles[0]], CtrlShimTile, output_offset=trace_size, num_pkts=2
                )

                trace_utils.gen_trace_done_aie2(ShimTile)



# ==============================================
# Options
# ==============================================
p = argparse.ArgumentParser()
p.add_argument(
    "-t",
    "--trace_size",
    required=False,
    dest="trace_size",
    default=0,
    help="Trace buffer size",
)

opts = p.parse_args(sys.argv[1:])
trace_size = int(opts.trace_size)

with mlir_mod_ctx() as ctx:
    vec_matrix_mult(dev, trace_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
