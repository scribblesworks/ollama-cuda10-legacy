#include <cuda_runtime_api.h>

#include <cstdio>

namespace {

constexpr size_t mib = 1024 * 1024;

bool check(cudaError_t status, const char * operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

} // namespace

int main() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
        return 0;
    }
    if (!check(status, "cudaGetDeviceCount")) {
        return 1;
    }
    for (int index = 0; index < count; ++index) {
        cudaDeviceProp device{};
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (!check(cudaGetDeviceProperties(&device, index), "cudaGetDeviceProperties") ||
            !check(cudaSetDevice(index), "cudaSetDevice") ||
            !check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo")) {
            return 1;
        }
        std::printf("%d\t%d\t%d\t%zu\t%zu\t%s\n", index, device.major, device.minor,
                    total_bytes / mib, free_bytes / mib, device.name);
    }
}
