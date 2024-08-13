GEN_PTX_CL=gen_ptx_from_cl.out

$(GEN_PTX_CL): gen_ptx_from_cl.c
	gcc ./gen_ptx_from_cl.c -o gen_ptx_from_cl.out -I /usr/local/cuda/include -L /usr/local/cuda/targets/x86_64-linux/lib/ -lOpenCL 

run-ntt:
	cd ntt && ./run_ptx.out ./ptx_from_cl.ptx test
	cd ntt && ./run_ptx.out ./ntt.ptx test

run-vectoradd:
	cd vector_add && ./run_ptx.out ./ptx_from_cl.ptx test
	cd vector_add && ./run_ptx.out ./vector_add.ptx test

vectoradd: $(GEN_PTX_CL) vector_add/vector_add.cu vector_add/vector_add.cl vector_add/run_ptx.cc
	cd vector_add && nvcc vector_add.cu -ptx
	cd vector_add && ../$(GEN_PTX_CL) ./vector_add.cl
	cd vector_add && g++ ./run_ptx.cc -o run_ptx.out -I/usr/local/cuda/include -L/usr/local/cuda/targets/x86_64-linux/lib/  -lcuda -lcudart

ntt: $(GEN_PTX_CL) ntt/ntt.cu ntt/ntt.cl ntt/run_ptx.cc
	cd ntt && nvcc ntt.cu -ptx
	cd ntt && ../$(GEN_PTX_CL) ./ntt.cl
	cd ntt && g++ ./run_ptx.cc -o run_ptx.out -I/usr/local/cuda/include -L/usr/local/cuda/targets/x86_64-linux/lib/  -lcuda -lcudart