#ifndef _CudaReconstruction_
#define _CudaReconstruction_

#include <stdio.h>
#include <vector>
#include "vtkPointData.h"
#include "vtkDoubleArray.h"
#include "vtkMatrix4x4.h"
#include <math.h>
#include "ReconstructionData.h"

// ----------------------------------------------------------------------------
// Define texture and constants
__constant__ double c_gridMatrix[16];
__constant__ double3 c_gridOrig;
__constant__ int3 c_gridDims;
__constant__ double3 c_gridSpacing;
__constant__ int2 c_depthMapDims;

// ----------------------------------------------------------------------------
#define CudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


// ----------------------------------------------------------------------------
__device__ void transformFrom4Matrix(double matrix[16], double point[3], double output[3])
{
  output[0] = matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3];
  output[1] = matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7];
  output[2] = matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11];
}


// ----------------------------------------------------------------------------
__device__ double norm(double vec[3])
{
  return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

// ----------------------------------------------------------------------------
__device__ double cumulFunction(double diff, double currentVal)
{
  double shift = 10 - 0.5 * std::abs(diff);
  if (shift < 0)
    shift = 0;
  return currentVal + shift;
}

// ----------------------------------------------------------------------------
__device__ int computeVoxelID(int coordinates[3], int type)
{
  int dimX = c_gridDims.x - 1;
  int dimY = c_gridDims.y - 1;
  if (type == 0)
    {
    dimX = c_depthMapDims.x;
    dimY = c_depthMapDims.y;
    }
  int i = coordinates[0];
  int j = coordinates[1];
  int k = coordinates[2];
  return (k*dimY + j)*dimX + i;
}

__device__ void computeVoxelCenter(int voxelCoordinate[3], double output[3])
{
  output[0] = c_gridOrig.x + (voxelCoordinate[0] + 0.5) * c_gridSpacing.x;
  output[1] = c_gridOrig.y + (voxelCoordinate[1] + 0.5) * c_gridSpacing.y;
  output[2] = c_gridOrig.z + (voxelCoordinate[2] + 0.5) * c_gridSpacing.z;
}


// ----------------------------------------------------------------------------
__global__ void depthMapKernel(double* depths, double matrixK[16], double matrixTR[16],
  double* output)
{
  // Get voxel coordinate according to thread id
  int i = threadIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;
  int voxelCoordinate[3] = { i, j, k };

  // Get the center of the voxel
  double voxelCenterTemp[3];
  computeVoxelCenter(voxelCoordinate, voxelCenterTemp);

  // Transform voxel from grid to real coord
  double voxelCenter[3];
  transformFrom4Matrix(c_gridMatrix, voxelCenterTemp, voxelCenter);

  // Transform voxel center from real coord to camera coords
  double voxelCenterCamera[3];
  transformFrom4Matrix(matrixTR, voxelCenter, voxelCenterCamera);

  // Transform voxel center from camera coords to depth map homogeneous coords
  double voxelCenterHomogen[3];
  transformFrom4Matrix(matrixK, voxelCenterCamera, voxelCenterHomogen);

  // Get voxel center on depth map coord
  double voxelCenterDepthMap[2];
  voxelCenterDepthMap[0] = voxelCenterHomogen[0] / voxelCenterHomogen[2];
  voxelCenterDepthMap[1] = voxelCenterHomogen[1] / voxelCenterHomogen[2];

  // Get real pixel position (approximation)
  int pixel[3];
  pixel[0] = round(voxelCenterDepthMap[0]);
  pixel[1] = round(voxelCenterDepthMap[1]);
  pixel[2] = 0;

  // Get the distance between voxel and camera
  double realDepth = norm(voxelCenterCamera);

  // Test if coordinate are inside depth map
  if (pixel[0] < 0 || pixel[1] < 0 || pixel[2] < 0 ||
    pixel[0] > c_depthMapDims.x - 1 ||
    pixel[1] > c_depthMapDims.y - 1)
    {
      return;
    }

  // Compute the ID on depthmap values according to pixel position and dpeth map dimensions
  int depthMapId = computeVoxelID(pixel, 0);
  int gridId = computeVoxelID(voxelCoordinate, 1);
  double depth = depths[depthMapId];
  double currentScalarValue = output[gridId];
  double newValue = cumulFunction(realDepth - depth, currentScalarValue);

  // Update the value to the output
  output[gridId] = newValue;
}


// ----------------------------------------------------------------------------
__host__ void vtkMatrixToDoubleTable(vtkMatrix4x4* matrix, double* output)
{
  output[0] = matrix->GetElement(0, 0);
  output[1] = matrix->GetElement(0, 1);
  output[2] = matrix->GetElement(0, 2);
  output[3] = matrix->GetElement(0, 3);
  output[4] = matrix->GetElement(1, 0);
  output[5] = matrix->GetElement(1, 1);
  output[6] = matrix->GetElement(1, 2);
  output[7] = matrix->GetElement(1, 3);
  output[8] = matrix->GetElement(2, 0);
  output[9] = matrix->GetElement(2, 1);
  output[10] = matrix->GetElement(2, 2);
  output[11] = matrix->GetElement(2, 3);
  output[12] = matrix->GetElement(3, 0);
  output[13] = matrix->GetElement(3, 1);
  output[14] = matrix->GetElement(3, 2);
  output[15] = matrix->GetElement(3, 3);
}


// ----------------------------------------------------------------------------
__host__ void vtkDoubleArrayToTable(vtkDoubleArray* doubleArray, double* output)
{
  for (int i = 0; i < doubleArray->GetNumberOfTuples(); i++)
  {
    output[i] = doubleArray->GetTuple1(i);
  }
}


// ----------------------------------------------------------------------------
__host__ void vtkImageDataToTable(vtkImageData* image, double* output)
{
  vtkDoubleArray* depths = vtkDoubleArray::SafeDownCast(image->GetPointData()->GetArray("Depths"));
  vtkDoubleArrayToTable(depths, output);
}


// ----------------------------------------------------------------------------
__host__ void doubleTableToVtkDoubleArray(double* table, vtkDoubleArray* output)
{
  int nbVoxels = output->GetNumberOfTuples();
  for (int i = 0; i < nbVoxels; i++)
  {
    output->SetTuple1(i, table[i]);
  }
}


// ----------------------------------------------------------------------------
/** Main function **/
int reconstruction(std::vector<ReconstructionData*> h_dataList, // List of depth matrix and associated matrix
                   vtkMatrix4x4* i_gridMatrix, // Matrix to transform grid voxel to real coordinates
                   int h_gridDims[3], // Dimensions of the output volume
                   double h_gridOrig[3], // Origin of the output volume
                   double h_gridSpacing[3], // Spacing of the output volume
                   vtkDoubleArray* io_outScalar)
{
  if (h_dataList.size() == 0)
    return -1;

  // Get usefull value for allocation of variables
  const int matrix4Size = 16;
  const int nbPixelOnDepthMap = h_dataList[0]->GetDepthMap()->GetNumberOfPoints();
  const int nbVoxels = io_outScalar->GetNumberOfTuples();

  // Fill texture and constant values
  double* h_gridMatrix = new double[16];
  vtkMatrixToDoubleTable(i_gridMatrix, h_gridMatrix);
  double* h_outScalar = new double[nbVoxels];
  vtkDoubleArrayToTable(io_outScalar, h_outScalar);


  // Create and fill device value
  cudaMemcpyToSymbol(c_gridMatrix, h_gridMatrix, 16 * sizeof(double));
  cudaMemcpyToSymbol(c_gridDims, h_gridDims, 3 * sizeof(int));
  cudaMemcpyToSymbol(c_gridOrig, h_gridOrig, 3 * sizeof(double));
  cudaMemcpyToSymbol(c_gridSpacing, h_gridSpacing, 3 * sizeof(double));
  double* d_outScalar;
  CudaErrorCheck(cudaMalloc((void**)&d_outScalar, nbVoxels * sizeof(double)));
  CudaErrorCheck(cudaMemcpy(d_outScalar, h_outScalar, nbVoxels * sizeof(double), cudaMemcpyHostToDevice));


  int h_dimDepthMap[3];
  h_dataList[0]->GetDepthMap()->GetDimensions(h_dimDepthMap);
  CudaErrorCheck(cudaMemcpyToSymbol(c_depthMapDims, h_dimDepthMap, 2 * sizeof(int)));

  // Organize threads into blocks and grids
  dim3 dimBlock(h_gridDims[0] - 1, 1, 1); // nb threads
  dim3 dimGrid(1, h_gridDims[1] - 1, h_gridDims[2] - 1); // nb blocks

  // Create device data from host value
  double *d_depthMap, *d_matrixK, *d_matrixRT;
  CudaErrorCheck(cudaMalloc((void**)&d_depthMap, nbPixelOnDepthMap * sizeof(double)));
  CudaErrorCheck(cudaMalloc((void**)&d_matrixK, matrix4Size * sizeof(double)));
  CudaErrorCheck(cudaMalloc((void**)&d_matrixRT, matrix4Size * sizeof(double)));
  
  for (int i = 0; i < h_dataList.size(); i++)
    {
    // Get data and transform its properties to atomic type
    ReconstructionData* currentData = h_dataList[i];
    double* h_depthMap = new double[nbPixelOnDepthMap];
    vtkImageDataToTable(currentData->GetDepthMap(), h_depthMap);
    double* h_matrixK = new double[16];
    vtkMatrixToDoubleTable(currentData->Get4MatrixK(), h_matrixK);
    double* h_matrixRT = new double[16];
    vtkMatrixToDoubleTable(currentData->GetMatrixTR(), h_matrixRT);

    CudaErrorCheck(cudaMemcpy(d_depthMap, h_depthMap, nbPixelOnDepthMap * sizeof(double), cudaMemcpyHostToDevice));
    CudaErrorCheck(cudaMemcpy(d_matrixK, h_matrixK, matrix4Size * sizeof(double), cudaMemcpyHostToDevice));
    CudaErrorCheck(cudaMemcpy(d_matrixRT, h_matrixRT, matrix4Size * sizeof(double), cudaMemcpyHostToDevice));

    // run code into device
    depthMapKernel << <dimGrid, dimBlock >> >(d_depthMap, d_matrixK, d_matrixRT, d_outScalar);

    // Wait that all threads have finished
    CudaErrorCheck(cudaDeviceSynchronize());

    // clean code
    delete(h_depthMap);
    delete(h_matrixK);
    delete(h_matrixRT);
    }

  // Transfer data from device to host
  cudaMemcpy(h_outScalar, d_outScalar, nbVoxels * sizeof(double), cudaMemcpyDeviceToHost);

  // Transfer host data to output
  doubleTableToVtkDoubleArray(h_outScalar, io_outScalar);

  // Clean memory
  cudaFree(d_outScalar);
  delete(h_gridMatrix);
  delete(h_outScalar);
  cudaFree(d_depthMap);
  cudaFree(d_matrixK);
  cudaFree(d_matrixRT);

  return 1;
}





// --------------------------------TEST----------------------------------------
// ----------------------------------------------------------------------------
__global__ void kernel(int *a, int *b)
{
  a[threadIdx.x] = b[threadIdx.x];
}
// ----------------------------------------------------------------------------
int cuda_reconstruction(
  double h_gridMatrix[16], double h_gridOrig[3], int h_gridDims[3], double h_gridSpacing[3],
  int h_depthMapDims[3], double* h_depths, double h_depthMapMatrixK[16], double h_depthMapMatrixTR[16],
  double* h_outScalar)
{
  const int N = 5;

  // create data into host
  int h_a[N] = { 0, 0, 0, 0, 0 };
  int h_b[N] = { 1, 2, 4, 5, 2 };

  // tranfer data from host to device
  int *d_a, *d_b;
  cudaMalloc((void**)&d_a, N * sizeof(int));
  cudaMalloc((void**)&d_b, N * sizeof(int));
  CudaErrorCheck(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));
  CudaErrorCheck(cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice));

  // organize threads into blocks and grids
  dim3 dimBlock(N, 1, 1); // nb threads per blocks
  dim3 dimGrid(1, 1, 1); // nb blocks

  // run code into device
  kernel << <dimGrid, dimBlock >> >(d_a, d_b);

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
