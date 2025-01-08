#include <vector>
#include <cuda_runtime.h>
#include "wave_cuda.h"

// CUDA kernel to update the wave
__global__ void updateWaveKernel(float* u, float* u_prev, float* u_next, int grid_size, float c, float dt, float dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && j > 0 && i < grid_size - 1 && j < grid_size - 1) {
        int idx = i * grid_size + j;
        float laplacian = (u[(i + 1) * grid_size + j] + u[(i - 1) * grid_size + j] +
                           u[i * grid_size + (j + 1)] + u[i * grid_size + (j - 1)] - 4 * u[idx]) / (dx * dx);
        u_next[idx] = 2 * u[idx] - u_prev[idx] + c * c * dt * dt * laplacian;
        // u_next[idx]=1;
    }
}

__global__ void swapArraysKernel(float* a, float* b, int grid_size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && j > 0 && i < grid_size - 1 && j < grid_size - 1) {
        int idx = i * grid_size + j;
        float temp = a[idx];
        a[idx] = b[idx];
        b[idx] = temp;
    }
}

// Initialize the wave
void initializeWaveCuda(float* d_u, float* d_u_prev, int grid_size) {
    std::vector<float> u_prev_host(grid_size * grid_size, 0.0f);
    std::vector<float> u_host(grid_size * grid_size, 0.0f);

    int center = grid_size / 2;
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            float dist = std::sqrt((i - center) * (i - center) + (j - center) * (j - center));
            u_prev_host[i * grid_size + j] = std::exp(-dist * dist / 10.0f); // Gaussian disturbance
        }
    }

    // int center = grid_size / 4;
    // int center2 = grid_size / 2;
    // for (int i = 0; i < grid_size; ++i) {
    //     for (int j = 0; j < grid_size; ++j) {
    //         float dist = std::sqrt((i - center) * (i - center) + (j - center) * (j - center));
    //         float dist2 = std::sqrt((i - center2) * (i - center2) + (j - center2) * (j - center2));
    //         u_prev_host[i * grid_size + j] = std::exp(-dist * dist / 10.0f)+std::exp(-dist2 * dist2 / 10.0f)-1; // Gaussian disturbance
    //     }
    // }

    cudaMemcpy(d_u_prev, u_prev_host.data(), grid_size * grid_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u_prev_host.data(), grid_size * grid_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

// Update the wave simulation
void updateWaveCuda(float* d_u, float* d_u_prev, float* d_u_next, int grid_size, float c, float dt, float dx) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((grid_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (grid_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    updateWaveKernel<<<numBlocks, threadsPerBlock>>>(d_u, d_u_prev, d_u_next, grid_size, c, dt, dx);
    cudaDeviceSynchronize();

    // std::swap(d_u_prev, d_u);
    // std::swap(d_u, d_u_next);
    swapArraysKernel<<<numBlocks, threadsPerBlock>>>(d_u_prev, d_u, grid_size);
    cudaDeviceSynchronize();
    swapArraysKernel<<<numBlocks, threadsPerBlock>>>(d_u, d_u_next, grid_size);
    cudaDeviceSynchronize();
    // float *temp = d_u_prev;
    // d_u_prev = d_u;
    // d_u = d_u_next;
    // d_u_next = temp;
}