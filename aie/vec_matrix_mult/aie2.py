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
from aie.helpers.util import np_dtype_to_mlir_type

import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from aie.extras.dialects.ext import memref, arith

dev = AIEDevice.npu1
N = 4096
N_bytes = 4096 * 4

N_CORE_ROW = 2
N_CORE_COL = 4
N_SUBVEC = 16
DEGREE = 4096
DEGREE_PER_SUBVEC = DEGREE // N_SUBVEC
VEC_B_PER_CORE = DEGREE //  (N_CORE_ROW * N_CORE_COL)
# VEC_B_PER_CORE = 2

def vec_matrix_mult(dev, trace_size):

    in1_dtype = np.int32
    in2_dtype = np.int32
    out_dtype = np.int32
    
    OUTPUT_PER_COL = N_CORE_ROW * VEC_B_PER_CORE
    
    
    @device(dev)
    def device_body():
        tile_ty = np.ndarray[(DEGREE_PER_SUBVEC,), np.dtype[in1_dtype]]    
        tile_matrix_ty = np.ndarray[(DEGREE_PER_SUBVEC * N_CORE_ROW,), np.dtype[in2_dtype]]    
        
        output_ty = np.ndarray[(OUTPUT_PER_COL,), np.dtype[out_dtype]]
        output_ty_per_core = np.ndarray[(VEC_B_PER_CORE,), np.dtype[out_dtype]]

        ctrl_pkt_ty = np.ndarray[(1,), np.dtype[np.uint32]]
        trace_ty = np.ndarray[(trace_size,), np.dtype[np.uint8]]

        buff_ty = np.ndarray[(128,), np.dtype[np.int32]]

    
        # ==============================================
        # AIE Core Function declarations
        # ==============================================
        ## void vector_accum(int32_t *in_a, int32_t *in_b, int32_t *out_c, int32_t N)
        accum_func = external_func(
            f"vector_accum",
            inputs=[tile_ty, tile_ty, output_ty_per_core, np.int32, np.int32],
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
        ShimTiles = [tile(c, 0) for c in range(N_CORE_COL)]
        MemTiles = [tile(c, 1) for c in range(N_CORE_COL)]
        CtrlShimTile = tile(1, 0)
        ComputeTiles = [[tile(c, 2 + i) for i in range(N_CORE_ROW)] for c in range(N_CORE_COL)]
        
        # ==============================================
        # Object FIFO declarations
        # ==============================================
        of_in_shim_mem_list = []
        of_in2_shim_mem_list = []
        of_out_mem_shim_list = []
        for c in range(N_CORE_COL):
            of_in_shim_mem_list.append(object_fifo(
                f"in_shim_mem_{c}", ShimTiles[c], MemTiles[c], 2, tile_ty
            ))
            of_in2_shim_mem_list.append(object_fifo(
                f"in2_shim_mem_{c}", ShimTiles[c], MemTiles[c], 2, tile_matrix_ty
            ))
            of_out_mem_shim_list.append(object_fifo(
                f"out_mem_shim_{c}", MemTiles[c], ShimTiles[c], 2, output_ty
            ))

        # in1
        of_in_broadcast_list = []
        for c in range(N_CORE_COL):
            of_in_broadcast_list.append(object_fifo(f"in_mem_ct_broadcast_{c}", MemTiles[c], ComputeTiles[c], 2, tile_ty))
            object_fifo_link(of_in_shim_mem_list[c], [of_in_broadcast_list[c]], [], [0])

        # in2
        of_in2_list = []
        for c in range(N_CORE_COL):
            of_in2_list.append([])
            for r in range(N_CORE_ROW):
                of_in2_list[c].append(
                    object_fifo(f"in2_mem_ct{r}_col{c}", MemTiles[c], ComputeTiles[c][r], 2, tile_ty)
                )
            object_fifo_link(of_in2_shim_mem_list[c], of_in2_list[c], [], [DEGREE_PER_SUBVEC * r for r in range(N_CORE_ROW)])

        # out
        of_out_list = []
        for c in range(N_CORE_COL):
            of_out_list.append([])
            for r in range(N_CORE_ROW):
                of_out_list[c].append(
                    object_fifo(f"out_ct{r}_mem_col{c}", ComputeTiles[c][r], MemTiles[c], 2, output_ty_per_core)
                )
            object_fifo_link(of_out_list[c], of_out_mem_shim_list[c], [VEC_B_PER_CORE * r for r in range(N_CORE_ROW)], [])
        
        # ==============================================
        # Buff declarations
        # ==============================================
        # buffs = []
        # for r in range(N_CORE_ROW):
        #     buffs.append(buffer(ComputeTiles[r], np.ndarray[(128,), np.dtype[np.int32]] ,f"buff_ct{r}"))
        
        
        # ==============================================
        # Compute Tile Main Loops
        # ==============================================
        for c in range(N_CORE_COL):
            for r in range(N_CORE_ROW):
                @core(ComputeTiles[c][r], "kernel.o")
                def core_body():
                    # Effective while(1)
                    # store_buff_func(buffs[r], 0, 0)
                    for _ in range_(sys.maxsize):
                        elem_out = of_out_list[c][r].acquire(ObjectFifoPort.Produce, 1)
                        zeros_func(elem_out, VEC_B_PER_CORE)
                        # sub-vector loop
                        for _ in range_(N_SUBVEC):
                            elem_in = of_in_broadcast_list[c].acquire(ObjectFifoPort.Consume, 1)
                            # Vector B loop
                            for vec_b in range_(VEC_B_PER_CORE):
                                elem_in2 = of_in2_list[c][r].acquire(ObjectFifoPort.Consume, 1)
                                vec_b_i32 = arith.index_cast(vec_b, to=np_dtype_to_mlir_type(np.int32))
                                accum_func(elem_in, elem_in2, elem_out, DEGREE_PER_SUBVEC, vec_b_i32)
                                of_in2_list[c][r].release(ObjectFifoPort.Consume, 1)
                            of_in_broadcast_list[c].release(ObjectFifoPort.Consume, 1)
                        of_out_list[c][r].release(ObjectFifoPort.Produce, 1)


        # ==============================================
        # Setup trace trace
        # ==============================================
        tiles_to_trace = [ComputeTiles[0][0]]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTiles[0])
            # trace_utils.configure_packet_ctrl_flow([ComputeTiles[0][0]], CtrlShimTile)

        trace_size_int32 = trace_size // np.dtype(np.int32).itemsize


        # ==============================================
        # Runtime Sequence
        # ==============================================
        tensor_ty = np.ndarray[(DEGREE,), np.dtype[in1_dtype]]
        matrix_ty = np.ndarray[(DEGREE * VEC_B_PER_CORE * N_CORE_ROW * N_CORE_COL,), np.dtype[in2_dtype]]
        @runtime_sequence(tensor_ty, matrix_ty, output_ty)
        def sequence(A, B, C):
            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTiles[0],
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

            in_tasks = []
            in2_tasks = []
            out_tasks = []
            for c in range(N_CORE_COL):
                in_tasks.append(shim_dma_single_bd_task(
                    of_in_shim_mem_list[c], A, sizes=[1, 1, 1, DEGREE], issue_token=True
                ))
                in2_tasks.append(shim_dma_single_bd_task(
                    of_in2_shim_mem_list[c], B, 
                    sizes=[1, N_SUBVEC, VEC_B_PER_CORE, DEGREE_PER_SUBVEC * N_CORE_ROW], 
                    strides=[1, VEC_B_PER_CORE * N_CORE_COL * N_CORE_ROW * DEGREE_PER_SUBVEC, N_CORE_COL * N_CORE_ROW * DEGREE_PER_SUBVEC, 1],
                    offset=c * DEGREE_PER_SUBVEC * N_CORE_ROW,
                    issue_token=True
                ))
                out_tasks.append(shim_dma_single_bd_task(
                    of_out_mem_shim_list[c], C, sizes=[1, 1, 1, OUTPUT_PER_COL], offset=c * OUTPUT_PER_COL, issue_token=True
                ))

            dma_start_task(*in_tasks, *in2_tasks, *out_tasks)
            dma_await_task(*in_tasks, *in2_tasks, *out_tasks)

            if trace_size > 0:
                # trace_utils.config_ctrl_pkts_aie(
                #     [ComputeTiles[0]], CtrlShimTile, output_offset=trace_size, num_pkts=2
                # )

                trace_utils.gen_trace_done_aie2(ShimTiles[0])



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
