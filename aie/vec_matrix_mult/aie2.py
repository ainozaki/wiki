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

dev = AIEDevice.npu1
N = 4096
N_bytes = 4096 * 4

def vec_matrix_mult(dev, trace_size):

    in1_dtype = np.int32
    out_dtype = np.int32
    in2_dtype = np.int32

    in1_size = N
    in2_size = N
    out_size = N

    tensor_size = N
    num_sub_vectors = 4
    tile_size = tensor_size // num_sub_vectors

    assert out_size == in1_size, "Output buffer size must match input buffer size."

    vectorized = True

    @device(dev)
    def device_body():
        tensor_ty = np.ndarray[(tensor_size,), np.dtype[in1_dtype]]
        tile_ty = np.ndarray[(tile_size,), np.dtype[in1_dtype]]
        ctrl_pkt_ty = np.ndarray[(1,), np.dtype[np.uint32]]
        trace_ty = np.ndarray[(trace_size,), np.dtype[np.uint8]]

        # AIE Core Function declarations
        func_type = "vector" if vectorized else "scalar"
        add_vector = external_func(
            f"vector_vector_add",
            inputs=[tile_ty, tile_ty, tile_ty, np.int32],
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        CtrlShimTile = tile(1, 0)
        ComputeTile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, ComputeTile2, 2, tile_ty)
        of_in2 = object_fifo("in2", ShimTile, ComputeTile2, 2, tile_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, 2, tile_ty)

        # Set up compute tiles
        # Compute tile 2
        @core(ComputeTile2, "kernel.o")
        def core_body():
            # Effective while(1)
            for _ in range_(sys.maxsize):
                # Number of sub-vector "tile" iterations
                for _ in range_(num_sub_vectors):
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    elem_in2 = of_in2.acquire(ObjectFifoPort.Consume, 1)
                    add_vector(elem_in, elem_in2, elem_out, tile_size)
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_in2.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)


        # Set up a packet-switched flow from core to shim for tracing information
        tiles_to_trace = [ComputeTile2, ComputeTile2]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)
            trace_utils.configure_packet_ctrl_flow([ComputeTile2], CtrlShimTile)

        trace_size_int32 = trace_size // np.dtype(np.int32).itemsize

        # To/from AIE-array data movement
        # @runtime_sequence(tensor_ty, scalar_ty, tensor_ty, ctrl_pkt_ty, trace_ty)
        @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
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
                of_in, A, sizes=[1, 1, 1, tensor_size], issue_token=True
            )
            in2_task = shim_dma_single_bd_task(
                of_in2, B, sizes=[1, 1, 1, tensor_size], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out, C, sizes=[1, 1, 1, tensor_size], issue_token=True
            )

            dma_start_task(in_task, in2_task, out_task)
            dma_await_task(in_task, in2_task, out_task)

            if trace_size > 0:
                trace_utils.config_ctrl_pkts_aie(
                    [ComputeTile2], CtrlShimTile, output_offset=trace_size, num_pkts=2
                )

                trace_utils.gen_trace_done_aie2(ShimTile)


if len(sys.argv) < 5:
    raise ValueError(
        "[ERROR] Need at least 4 arguments (dev, in1_size, in2_size, out_size)"
    )


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument(
    "-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size"
)
p.add_argument(
    "-i2s", "--in2_size", required=True, dest="in2_size", help="Input 2 size"
)
p.add_argument("-os", "--out_size", required=True, dest="out_size", help="Output size")
p.add_argument(
    "-bw",
    "--int_bit_width",
    required=True,
    dest="int_bit_width",
    help="Integer Bit Width",
)
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
