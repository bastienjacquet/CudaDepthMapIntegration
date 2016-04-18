#ifndef _CudaReconstruction_
#define _CudaReconstruction_

__global__ void kernel(int *a, int *b)
{
  a[threadIdx.x] = b[threadIdx.x];
}

#define CudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

int cuda_reconstruction(
    double h_gridMatrix[16], double h_gridOrig[3], int h_gridDims[3], double h_gridSpacing[3],
    int h_depthMapDims[3], double* h_depths, double h_depthMapMatrixK[16], double h_depthMapMatrixTR[16],
    double* h_outScalar)
{
  const int N = 5;

  // create data into host
  int h_a[N] = { 0, 0, 0, 0, 0 };
  int h_b[N] = { 1, 1, 1, 1, 1 };

  // tranfer data from host to device
  int *d_a, *d_b;
  cudaMalloc((void**)&d_a, N * sizeof(int));
  cudaMalloc((void**)&d_b, N * sizeof(int));
  CudaErrorCheck(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));
  CudaErrorCheck(cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice));

  // organize threads into blocks and grids
  dim3 dimBlock(N, 1, 1);
  dim3 dimGrid(1, 1, 1);

  // run code into device
  kernel<<<dimGrid, dimBlock>>>(d_a, d_b);

  // transfer data from device to host
  CudaErrorCheck(cudaMemcpy(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; i++)
  {
    h_outScalar[i] = h_a[i];
  }

  // free memory
  cudaFree(d_a);
  cudaFree(d_b);

  return 1;
}

#endif
