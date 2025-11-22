#define IN_TILE_DIM 16  // Reduced from 32 due to 3D memory constraints
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__ void convolution_tiled_3D_const_mem_kernel(float *N, float *P,
                                                    int width, int height, int depth) {
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int dep = blockIdx.z*OUT_TILE_DIM + threadIdx.z - FILTER_RADIUS;

    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    
    if(dep>=0 && dep<depth && row>=0 && row<height && col>=0 && col<width) {
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = N[dep*width*height + row*width + col];
    } else {
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    __syncthreads();
    
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    int tileDep = threadIdx.z - FILTER_RADIUS;
    
    if (dep >= 0 && dep < depth && row >= 0 && row < height && col >= 0 && col < width) {
        if (tileCol>=0 && tileCol<OUT_TILE_DIM && 
            tileRow>=0 && tileRow<OUT_TILE_DIM &&
            tileDep>=0 && tileDep<OUT_TILE_DIM) {
            
            float Pvalue = 0.0f;
            
            for (int fDep = 0; fDep < 2*FILTER_RADIUS+1; fDep++) {
                for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
                    for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                        Pvalue += F_c[fDep][fRow][fCol] * 
                                 N_s[tileDep+fDep][tileRow+fRow][tileCol+fCol];
                    }
                }
            }
            
            P[dep*width*height + row*width + col] = Pvalue;
        }
    }
}
