all:
	nvcc -o a.out main.cu -arch=sm_89
	./a.out

trace:
	nsys profile -o trace ./a.out

open-gui:
	nsight-sys ./trace.qdrep