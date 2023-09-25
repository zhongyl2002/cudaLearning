#include<cuda_runtime.h>
#include<stdio.h>
#include<sys/time.h>

#define CHECK(call){                                    \
    const cudaError_t error = call;                     \
    if(error != cudaSuccess){                           \
        printf("Error: %s:%d, ", __FILE__, __LINE__);   \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(1);                                        \
    }\
}

void checkResult(float* hostRef, float *gpuRef, const int N){
    double e = 1.0e-8;
    bool match = true;
    for (int i = 0; i < N; i++)
    {
        if(abs(hostRef[i] - gpuRef[i]) > e){
            match = false;
            printf("NOT MATCH\n");
            printf("i = %d, cpuRef[i] = %5.2f, gpuRef[i] = %5.2f\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }
    if(match) printf("MATCH\n");
}

void sumArraysOnHost(float *A, float* B, float* C, const int N){
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }   
}

void initialData(float* ip, int size){
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = float((rand() & 0xff) / 10.0f);
    }
}

__global__ void sumArrayOnGPU(float *A, float *B, float *C, const int N){
    // int i = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) C[i] = A[i] + B[i];
}

void print(float* aa, int size){
    for (int i = 0; i < size; i++)
    {
        printf("%.3f\t", aa[i]);
    }
    printf("\n");
}

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.0e-6);
}

int main(int argc, char** argv){
    printf("%s Starting ...\n", argv[0]);

    int dev = 0;
    CHECK(cudaSetDevice(dev));

    int nElem = 0xfffffff;
    printf("Vector size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);
    
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // printf("Array A:\n");
    // print(h_A, nElem);
    // printf("Array B:\n");
    // print(h_B, nElem);

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    dim3 block(1024);
    dim3 grid((nElem + block.x - 1) / block.x);

    double iStart = cpuSecond();
    sumArrayOnGPU<<<grid, block>>> (d_A, d_B, d_C, nElem);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaStatus));
    }

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    double iElaps = cpuSecond() - iStart;
    printf("GPU - time cost : %f\n", iElaps);
    printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);


    iStart = cpuSecond();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = cpuSecond() - iStart;
    printf("CPU - time cost : %f\n", iElaps);

    checkResult(hostRef, gpuRef, nElem);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A), free(h_B), free(hostRef), free(gpuRef);

    return 0;
}