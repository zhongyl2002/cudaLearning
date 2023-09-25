// 获取gpu属性，以确定最大线程数和最大网格数

/*
Device 0: NVIDIA GeForce GTX 1650
  Maximum threads per block: 1024
  Maximum grid size (x, y, z): (2147483647, 65535, 65535)
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount); // 获取可用的GPU设备数量

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i); // 获取设备属性

        printf("Device %d: %s\n", i, deviceProp.name);
        printf("  Maximum threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum grid size (x, y, z): (%ld, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        // 可以查看其他设备属性，如共享内存大小、全局内存大小等

        printf("\n");
    }

    return 0;
}
