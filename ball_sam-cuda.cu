#include <stdio.h>

// CUDA kernel to print "Hello, World!" from each thread
__global__ void helloCUDA() {
    printf("Hello, World! I'm thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    // Launch the CUDA kernel with 1 block and 1 thread per block
    helloCUDA<<<1, 1>>>();
    
    // Synchronize threads to ensure all kernel executions are completed
    cudaDeviceSynchronize();

    // Check for any errors during kernel launch
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
        return 1;
    }

    return 0;
}
