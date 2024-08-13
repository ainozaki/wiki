#include <nvvm.h>

#include <iostream>
#include <fstream>
#include <cstring>

void compile_nvvm(const char* nvvm_code) {
    nvvmResult res;
    nvvmProgram prog;
    
    res = nvvmCreateProgram(&prog);
    if (res != NVVM_SUCCESS) {
        std::cout << "Error: nvvmCreateProgram failed" << std::endl;
        return;
    }

    res = nvvmAddModuleToProgram(prog, nvvm_code, strlen(nvvm_code), "module");
    std::cout << "nvvm_core size: " << strlen(nvvm_code) << std::endl;
    if (res != NVVM_SUCCESS) {
        std::cout << "Error: nvvmAddModuleToProgram failed" << std::endl;
        return;
    }

    const char *options[] = {"-arch=compute_50"};
    res = nvvmCompileProgram(prog, 1, options);
    if (res != NVVM_SUCCESS) {
        std::cout << "Error: nvvmCompileProgram failed" << std::endl;
        return;
    }
    std::cout << "Compiled NVVM code" << std::endl;

    size_t ptx_size;
    nvvmGetCompiledResultSize(prog, &ptx_size);
    char *ptx_code = (char*) malloc(ptx_size);
    nvvmGetCompiledResult(prog, ptx_code);

    printf("Compiled Result: %s\n", ptx_code);
    nvvmDestroyProgram(&prog);
}

int main(){
    // Read the NVVM code from a file
    std::ifstream file("kernel.bin");
    std::string nvvm_code((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    compile_nvvm(nvvm_code.c_str());
}