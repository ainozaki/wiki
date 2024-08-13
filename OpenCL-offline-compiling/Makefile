oclc:
	g++ -I/usr/local/cuda/include/ -L/usr/local/cuda/targets/x86_64-linux/lib ./oclc.cpp -o oclc -lOpenCL

test:
	g++ -I/usr/local/cuda/include/ -L/usr/local/cuda/targets/x86_64-linux/lib ./main.cpp -o main -lOpenCL

nvvm:
	nvcc -I/usr/local/cuda-12.3/nvvm/include/ -L/usr/local/cuda-12.3/nvvm/lib64/ ./compile-nvvm.cpp -o compile-nvvm -lnvvm