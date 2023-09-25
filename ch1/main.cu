#include<stdio.h>

__global__ void helloFromGPU(void){
    printf("hello from gpu from thread %d\n", threadIdx);
}

int main(){
    helloFromGPU <<<1, 10>>>();
    printf("Hello from cpu\n");
    // cudaDeviceReset();
    cudaDeviceSynchronize();
    return 0;
}