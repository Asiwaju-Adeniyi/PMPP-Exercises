#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void convolution_3D_basic_kernel(float *N, float *F, float *P,
    int r, int width, int height, int depth) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    int outDepth = blockIdx.z*blockDim.z + threadIdx.z;

    float Pvalue = 0.0f;
    
    for (int fDepth = 0; fDepth < 2*r+1; fDepth++) {
        for (int fRow = 0; fRow < 2*r+1; fRow++) {
            for (int fCol = 0; fCol < 2*r+1; fCol++) {
                int inDepth = outDepth - r + fDepth;
                int inRow = outRow - r + fRow;
                int inCol = outCol - r + fCol;
                
                if (inRow >= 0 && inRow < height && 
                    inCol >= 0 && inCol < width &&
                    inDepth >= 0 && inDepth < depth) {
                    
                    int inIndex = inDepth*width*height + inRow*width + inCol;
                    int fIndex = fDepth*(2*r+1)*(2*r+1) + fRow*(2*r+1) + fCol;
                    
                    Pvalue += F[fIndex] * N[inIndex];
                }
            }
        }
    }
    
    P[outDepth*width*height + outRow*width + outCol] = Pvalue;
}
