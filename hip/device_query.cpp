#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <string>

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                                \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__        \
                      << " code=" << static_cast<int>(err)                      \
                      << " (" << hipGetErrorString(err) << ")" << std::endl;   \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

int main() {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "No HIP devices found." << std::endl;
        return 0;
    }

    std::cout << "Number of HIP devices: " << deviceCount << "\n\n";

    for (int dev = 0; dev < deviceCount; ++dev) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, dev));

        std::cout << "Device " << dev << ":\n";
        std::cout << "  Name: " << prop.name << "\n";
        std::cout << "  PCI Bus ID: " << prop.pciBusID << "\n";
        std::cout << "  MultiProcessor Count: " << prop.multiProcessorCount << "\n";
        std::cout << "  Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) 
                  << " MB\n";
        std::cout << "  Clock Rate: " << prop.clockRate / 1000 << " MHz\n";

        // L2 cache size
        int l2CacheSize = 0;
        hipError_t attrErr = hipDeviceGetAttribute(
            &l2CacheSize,
            hipDeviceAttributeL2CacheSize,
            dev
        );

        if (attrErr == hipSuccess) {
            std::cout << "  L2 Cache Size: " << l2CacheSize / (1024 * 1024) << " MB\n";
        } else {
            std::cout << "  L2 Cache Size: (could not query: "
                      << hipGetErrorString(attrErr) << ")\n";
        }

        std::cout << std::endl;
    }

    return 0;
}