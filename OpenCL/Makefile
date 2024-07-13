runptx:
	g++ ./run_ptx.cc -o run_ptx.out -I/usr/local/cuda/include -L/usr/local/cuda/targets/x86_64-linux/lib/  -lcuda -lcudart

gen-ptx-cl:
	gcc ./gen_ptx_from_cl.c -o gen_ptx_from_cl.out -I /usr/local/cuda/include -L /usr/local/cuda/targets/x86_64-linux/lib/ -lOpenCL 

gen-ptx-cu:
	nvcc ./vector_add.cu -ptx

run-cl:
	./run_ptx.out ./vector_add_from_cl.ptx test

run-cu:
	./run_ptx.out ./vector_add.ptx test