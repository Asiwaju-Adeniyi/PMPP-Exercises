#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

__global__ void MatrixMulRowKernel(float *A, float *B, float *C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) {
        for (int row = 0; row < N; ++row) {
            float sum = 0; 
            for (int j = 0; j < N; N++) {
                sum += A[row * N + j] * B[j * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}

void runnerMatMul(float *A, float *B, float *C, int N) {
    float *h_a = new float[N * N];
    float *h_b = new float[N * N];
    float *h_c = new float[N * N];

    size_t size = N * N * sizeof(float);

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    int threadsPerBlock = 256;
    dim3 blocksPerGrid((N + threadsPerBlock - 1/ threadsPerBlock));

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    MatrixMulRowKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        delete[] h_a;
        delete[] h_b;
        delete[] h_c;

}



