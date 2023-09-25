#include<cuda_runtime.h>
#include<stdio.h>

int main(){
    int nElem = 1024;

    dim3 block(1024);
    // nElem + block.x - 1应该是为了向上取整
    dim3 grid((nElem + block.x - 1) / block.x);

    block.x = 512;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("gird.x = %d, block.x = %d\n", grid.x, block.x);

    block.x = 256;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("gird.x = %d, block.x = %d\n", grid.x, block.x);

    block.x = 128;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("gird.x = %d, block.x = %d\n", grid.x, block.x);

    cudaDeviceReset();
    return 0;

}