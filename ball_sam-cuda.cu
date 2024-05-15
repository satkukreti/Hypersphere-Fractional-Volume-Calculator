#include <iostream>
#include <vector>
#include <cmath>
#include <curand_kernel.h>

const int n_bins = 100;
const int n_points = 3000;

__global__ void generatePoints(int dim, int *hist, int n_bins, int n_points){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n_points) {
        curandState state;
        curand_init(tid, 0, 0, &state); //rand num generator
        
        double sum_squares = 0;
	do {
    	sum_squares = 0.0;
           for (int i = 0; i < dim; ++i) {
               double p = curand_uniform(&state) * 2 - 1; //rand num between -1 and 1
               sum_squares += p * p;
	       if(sum_squares > 1.0)
	       	    break;
           }
	} while(sum_squares > 1.0);

	double distance = sqrtf(sum_squares);
        int bin = min((int)(distance * n_bins), n_bins - 1);
        atomicAdd(&hist[bin], 1);
        
    }
}

int main() {
    int *hist;
    cudaMallocManaged(&hist, n_bins * sizeof(int));

    for (int dim = 2; dim <= 16; ++dim) {
        cudaMemset(hist, 0, n_bins * sizeof(int));
        int blockSize = 256;
        int numBlocks = (n_points + blockSize - 1) / blockSize;
        generatePoints<<<numBlocks, blockSize>>>(dim, hist, n_bins, n_points);
        cudaDeviceSynchronize();

        // Print results
        std::cout << "Dimension: " << dim << "\n";
        for (int i = 0; i < n_bins; ++i) {
            std::cout << (double)(hist[i]) / n_points << " ";
        }
        std::cout << "\n\n";
    }

    cudaFree(hist);
    return 0;
}
