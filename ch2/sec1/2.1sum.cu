#include<stdlib.h>
#include<string>
#include<time.h>

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

void print(float* aa, int size){
    for (int i = 0; i < size; i++)
    {
        printf("%.3f\t", aa[i]);
    }
    printf("\n");
}

int main(){
    int nElem = 10;
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    h_C = (float*)malloc(nBytes);
    
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    printf("Array A:\n");
    print(h_A, nElem);
    printf("Array B:\n");
    print(h_B, nElem);

    sumArraysOnHost(h_A, h_B, h_C, nElem);

    printf("Array C:\n");
    print(h_C, nElem);

    free(h_A), free(h_B), free(h_C);

    return 0;
}