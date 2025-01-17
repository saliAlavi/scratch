#ifndef WAVE_CUDA_H
#define WAVE_CUDA_H

__global__ void updateWaveKernel(float* u, float* u_prev, float* u_next, int grid_size, float c, float dt, float dx);
void initializeWaveCuda(float* d_u, float* d_u_prev, int grid_size);
void updateWaveCuda(float* d_u, float* d_u_prev, float* d_u_next, int grid_size, float c, float dt, float dx);
#endif