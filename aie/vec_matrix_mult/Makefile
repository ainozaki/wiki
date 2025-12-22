##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ../makefile-common

MLIR_AIE_DIR=/home/projects/aie/mlir-aie/v1.1.3/

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = vector_scalar_mul
int_bit_width = 32

in1_size = 16384 # in bytes
out_size = 16384 # in bytes, should always be equal to in1_size
in2_size = 4 # in bytes, should always be 4 (1x int32)
trace_size = 8192
CHESS ?= false

data_size = in1_size
aie_py_src=aie2.py

all: build/final_${data_size}.xclbin build/insts_${data_size}.bin

build/%.o: %.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBIT_WIDTH=${int_bit_width} -c ../kernel.cc -o ${@F}; 
else 
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=${int_bit_width} -c ../kernel.cc -o ${@F}; 
endif

build/aie_${data_size}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< > $@

build/aie_trace_${data_size}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -t ${trace_size} > $@

build/final_${data_size}.xclbin: build/aie_${data_size}.mlir build/kernel.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu-insts --npu-insts-name=insts_${data_size}.bin $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    	  --no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=insts_${data_size}.bin $(<:%=../%)
endif

build/final_trace_${data_size}.xclbin: build/aie_trace_${data_size}.mlir build/kernel.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu-insts --npu-insts-name=insts_${data_size}.bin $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=insts_${data_size}.bin $(<:%=../%)
endif

${targetname}_${data_size}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname}_${data_size} -DIN1_SIZE=${in1_size} -DIN2_SIZE=${in2_size} -DOUT_SIZE=${out_size} -DINT_BIT_WIDTH=${int_bit_width} 
	cd _build && cmake --build . --config Release
	cp _build/${targetname}_${data_size} $@ 

run: ${targetname}_${data_size}.exe build/final_${data_size}.xclbin build/insts_${data_size}.bin 
	./$< -x build/final_${data_size}.xclbin -i build/insts_${data_size}.bin -k MLIR_AIE

trace: ${targetname}_${data_size}.exe build/final_trace_${data_size}.xclbin build/insts_${data_size}.bin 
	./$< -x build/final_trace_${data_size}.xclbin -i build/insts_${data_size}.bin -k MLIR_AIE -t ${trace_size} -v 2
	${MLIR_AIE_DIR}/python/utils/parse_trace.py --input trace.txt --mlir build/aie_trace_${data_size}.mlir --output trace.json
	${MLIR_AIE_DIR}/python/utils/get_trace_summary.py --input trace.json

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean: clean_trace
	rm -rf build _build ${targetname}*.exe
