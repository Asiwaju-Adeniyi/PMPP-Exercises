#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void Convo2dKernel(float *N, float *F, float *P, int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0.0f;

    for (int fRow = 0; fRow < 2*r+1; fRow++) {
        for (int fCol= 0; fCol < 2*r+1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;

            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                float filter = F[fRow * (2*r+1) * fCol];
                float imageVal = N[inRow * width + inCol];
                Pvalue += filter * imageVal;
            }

        }
    }
    P[outRow * width + outCol] = Pvalue;


}
