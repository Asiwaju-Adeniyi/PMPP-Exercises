#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void convo1d(float *N, float *F, float *P, int width, int r) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= width) return;

    float Pvalue = 0.0f;
    int filterWidth = 2 * r + 1;

    for (int k = 0; k < filterWidth; k++) {
        int inputN = idx + k - r;

        if (inputN >= 0 && inputN < width){

        Pvalue += N[inputN] * F[k];
        
        }
    }
}
