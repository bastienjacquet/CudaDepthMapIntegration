// Copyright(c) 2016, Kitware SAS
// www.kitware.fr
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met :
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation and
// / or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef _CudaReconstruction_
#define _CudaReconstruction_

// Project include
#include "ReconstructionData.h"

// STD include
#include <math.h>
#include <stdio.h>
#include <vector>

// VTK includes
#include "vtkPointData.h"
#include "vtkDoubleArray.h"
#include "vtkMatrix4x4.h"
#include "vtkImageData.h"

#define SizeMat4x4 16
#define SizePoint3D 3
#define SizeDim3D 3
// Apply to matrix, computes on 3D point
typedef double TypeCompute;

// ----------------------------------------------------------------------------
/* Define texture and constants */
__constant__ TypeCompute c_gridMatrix[SizeMat4x4]; // Matrix to transpose from basic axis to output volume axis
__constant__ int3 c_gridDims; // Dimensions of the output volume
__constant__ int2 c_depthMapDims; // Dimensions of all depths map
__constant__ TypeCompute c_rayPotentialThick; // Thickness threshold for the ray potential function
__constant__ TypeCompute c_rayPotentialRho; // Rho at the Y axis for the ray potential function
__constant__ TypeCompute c_rayPotentialEta;
__constant__ TypeCompute c_rayPotentialDelta;
int ch_gridDims[3];

// ----------------------------------------------------------------------------
/* Macro called to catch cuda error when cuda functions is called */
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
/* Apply a 4x4 matrix to a 3D points */
__device__ void transformFrom4Matrix(TypeCompute M[SizeMat4x4], TypeCompute point[SizePoint3D], TypeCompute output[SizePoint3D])
{
  output[0] = M[0 * 4 + 0] * point[0] + M[0 * 4 + 1] * point[1] + M[0 * 4 + 2] * point[2] + M[0 * 4 + 3];
  output[1] = M[1 * 4 + 0] * point[0] + M[1 * 4 + 1] * point[1] + M[1 * 4 + 2] * point[2] + M[1 * 4 + 3];
  output[2] = M[2 * 4 + 0] * point[0] + M[2 * 4 + 1] * point[1] + M[2 * 4 + 2] * point[2] + M[2 * 4 + 3];
}


// ----------------------------------------------------------------------------
/* Compute the norm of a table with 3 double */
__device__ TypeCompute norm(TypeCompute vec[SizeDim3D])
{
  return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

// ----------------------------------------------------------------------------
/* Ray potential function which computes the increment to the current voxel */
template<typename TVolumetric>
__device__ void rayPotential(TypeCompute realDistance, TypeCompute depthMapDistance, TVolumetric& res)
{
  TypeCompute diff = (realDistance - depthMapDistance);

  TypeCompute absoluteDiff = abs(diff);
  // Can't divide by zero
  int sign = diff != 0 ? diff / absoluteDiff : 0;

  if (absoluteDiff > c_rayPotentialDelta)
    res = diff > 0 ? 0 : - c_rayPotentialEta * c_rayPotentialRho;
  else if (absoluteDiff > c_rayPotentialThick)
    res = c_rayPotentialRho * sign;
  else
    res = (c_rayPotentialRho / c_rayPotentialThick) * diff;
}

// ----------------------------------------------------------------------------
/* Compute the voxel Id on a 1D table according to its 3D coordinates
  coordinates : 3D coordinates
*/
__device__ int computeVoxelIDGrid(int coordinates[SizePoint3D])
{
  int dimX = c_gridDims.x - 1;
  int dimY = c_gridDims.y - 1;
  int i = coordinates[0];
  int j = coordinates[1];
  int k = coordinates[2];
  return (k*dimY + j)*dimX + i;
}

// ----------------------------------------------------------------------------
/* Compute the pixel Id on a 1D table according to its 3D coordinates
  (third coordinate is not used)
coordinates : 3D coordinates
*/
__device__ int computeVoxelIDDepth(int coordinates[SizePoint3D])
{
  int dimX = c_depthMapDims.x;
  int dimY = c_depthMapDims.y;
  int x = coordinates[0];
  int y = coordinates[1];
  // /!\ vtkImageData has its origin at the bottom left, not top left
  return (dimX*(dimY-1-y)) + x;
}

// ----------------------------------------------------------------------------
/* Main function called inside the kernel
  depths : depth map values
  matrixK : matrixK
  matrixTR : matrixTR
  output : double table that will be filled at the end of function
*/
template<typename TVolumetric>
__global__ void depthMapKernel(TypeCompute* depths, TypeCompute matrixK[SizeMat4x4], TypeCompute matrixTR[SizeMat4x4],
  TVolumetric* output)
{
  // Get voxel coordinate according to thread id
  int voxelIndex[SizePoint3D] = { (int)threadIdx.x, (int)blockIdx.y, (int)blockIdx.z };
  TypeCompute voxelIndexTypeCompute[SizePoint3D] = { (TypeCompute)voxelIndex[0], (TypeCompute)voxelIndex[1], (TypeCompute)voxelIndex[2] };
  TypeCompute voxelCenter[SizePoint3D];
  transformFrom4Matrix(c_gridMatrix, voxelIndexTypeCompute, voxelCenter);

  // Transform voxel center from real coord to camera coords
  TypeCompute voxelCenterCamera[SizePoint3D];
  transformFrom4Matrix(matrixTR, voxelCenter, voxelCenterCamera);

  // Transform voxel center from camera coords to depth map homogeneous coords
  TypeCompute voxelCenterHomogen[SizePoint3D];
  transformFrom4Matrix(matrixK, voxelCenterCamera, voxelCenterHomogen);
  if (voxelCenterHomogen[2] < 0)
    {
    return;
    }
  // Get voxel center on depth map coord
  TypeCompute voxelCenterDepthMap[2];
  voxelCenterDepthMap[0] = voxelCenterHomogen[0] / voxelCenterHomogen[2];
  voxelCenterDepthMap[1] = voxelCenterHomogen[1] / voxelCenterHomogen[2];
  // Get real pixel position (approximation)
  int pixel[SizePoint3D];
  pixel[0] = round(voxelCenterDepthMap[0]);
  pixel[1] = round(voxelCenterDepthMap[1]);
  pixel[2] = 0;

  // Test if coordinate are inside depth map
  if (pixel[0] < 0 || pixel[1] < 0 || 
    pixel[0] >= c_depthMapDims.x ||
    pixel[1] >= c_depthMapDims.y )
    {
    return;
    }

  // Compute the ID on depthmap values according to pixel position and depth map dimensions
  int depthMapId = computeVoxelIDDepth(pixel);
  TypeCompute depth = depths[depthMapId];
  if (depth == -1)
    {
    return;
    }
  int gridId = computeVoxelIDGrid(voxelIndex);  // Get the distance between voxel and camera
  TypeCompute realDepth = voxelCenterCamera[2];
  TVolumetric newValue;
  rayPotential<TVolumetric>(realDepth, depth, newValue);
  // Update the value to the output
  output[gridId] += newValue;
}





// ----------------------------------------------------------------------------
/* Extract data from a 4x4 vtkMatrix and fill a double table with 16 space */
__host__ void vtkMatrixToTypeComputeTable(vtkMatrix4x4* matrix, TypeCompute* output)
{
  int cpt = 0;
  for (int i = 0; i < 4; i++)
    {
    for (int j = 0; j < 4; j++)
      {
      output[cpt++] = (TypeCompute)matrix->GetElement(i, j);
      }
    }
}


// ----------------------------------------------------------------------------
/* Extract double value from vtkDoubleArray and fill a double table (output) */
template <typename T>
__host__ void vtkDoubleArrayToTable(vtkDoubleArray* doubleArray, T* output)
{
  for (int i = 0; i < doubleArray->GetNumberOfTuples(); i++)
  {
    output[i] = (T)doubleArray->GetTuple1(i);
  }
}


// ----------------------------------------------------------------------------
/* Extract point data array (name 'Depths') from vtkImageData and fill a double table */
__host__ void vtkImageDataToTable(vtkImageData* image, TypeCompute* output)
{
  vtkDoubleArray* depths = vtkDoubleArray::SafeDownCast(image->GetPointData()->GetArray("Depths"));
  vtkDoubleArrayToTable<TypeCompute>(depths, output);
}


// ----------------------------------------------------------------------------
/* Fill a vtkDoubleArray from a double table */
template<typename TVolumetric>
__host__ void doubleTableToVtkDoubleArray(TVolumetric* table, vtkDoubleArray* output)
{
  int nbVoxels = output->GetNumberOfTuples();
  for (int i = 0; i < nbVoxels; i++)
  {
    output->SetTuple1(i, (double)table[i]);
  }
}


// ----------------------------------------------------------------------------
/* Initialize cuda constant */
void CudaInitialize(vtkMatrix4x4* i_gridMatrix, // Matrix to transform grid voxel to real coordinates
                int h_gridDims[SizeDim3D], // Dimensions of the output volume
                double h_rayPThick,
                double h_rayPRho,
                double h_rayPEta,
                double h_rayPDelta,
                int h_depthMapDims[2])
{
  TypeCompute* h_gridMatrix = new TypeCompute[SizeMat4x4];
  vtkMatrixToTypeComputeTable(i_gridMatrix, h_gridMatrix);

  cudaMemcpyToSymbol(c_gridMatrix, h_gridMatrix, SizeMat4x4 * sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_gridDims, h_gridDims, SizeDim3D * sizeof(int));
  cudaMemcpyToSymbol(c_rayPotentialThick, &h_rayPThick, sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_rayPotentialRho, &h_rayPRho, sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_rayPotentialEta, &h_rayPEta, sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_rayPotentialDelta, &h_rayPDelta, sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_depthMapDims, h_depthMapDims, 2 * sizeof(int));

  ch_gridDims[0] = h_gridDims[0];
  ch_gridDims[1] = h_gridDims[1];
  ch_gridDims[2] = h_gridDims[2];

  // Clean memory
  delete(h_gridMatrix);
}

// ----------------------------------------------------------------------------
/* Read all depth map and process each of them. Fill the output 'io_scalar' */
template <typename TVolumetric>
bool ProcessDepthMap(std::vector<std::string> vtiList,
                     std::vector<std::string> krtdList,
                     double thresholdBestCost,
                     vtkDoubleArray* io_scalar)
{
  if (vtiList.size() == 0 || krtdList.size() == 0)
    {
    std::cerr << "Error, no depthMap or KRTD matrix have been loaded" << std::endl;
    return false;
    }

  // Define usefull constant values
  ReconstructionData* data = new ReconstructionData(vtiList[0], krtdList[0]);
  const int nbPixelOnDepthMap = data->GetDepthMap()->GetNumberOfPoints();
  const int nbVoxels = io_scalar->GetNumberOfTuples();
  int nbDepthMap = (int)vtiList.size();

  std::cout << "START CUDA ON " << nbDepthMap << " Depth map" << std::endl;

  // Transform vtkDoubleArray to table
  TVolumetric* h_outScalar = new TVolumetric[nbVoxels];
  vtkDoubleArrayToTable<TVolumetric>(io_scalar, h_outScalar);
  TVolumetric* d_outScalar;
  CudaErrorCheck(cudaMalloc((void**)&d_outScalar, nbVoxels * sizeof(TVolumetric)));
  CudaErrorCheck(cudaMemcpy(d_outScalar, h_outScalar, nbVoxels * sizeof(TVolumetric), cudaMemcpyHostToDevice));

  // Organize threads into blocks and grids
  dim3 dimBlock(ch_gridDims[0] - 1, 1, 1); // nb threads on each block
  dim3 dimGrid(1, ch_gridDims[1] - 1, ch_gridDims[2] - 1); // nb blocks on a grid

  // Create device data from host value
  TypeCompute *d_depthMap, *d_matrixK, *d_matrixRT;
  CudaErrorCheck(cudaMalloc((void**)&d_depthMap, nbPixelOnDepthMap * sizeof(TypeCompute)));
  CudaErrorCheck(cudaMalloc((void**)&d_matrixK, SizeMat4x4 * sizeof(TypeCompute)));
  CudaErrorCheck(cudaMalloc((void**)&d_matrixRT, SizeMat4x4 * sizeof(TypeCompute)));

  TypeCompute* h_depthMap = new TypeCompute[nbPixelOnDepthMap];
  TypeCompute* h_matrixK = new TypeCompute[SizeMat4x4];
  TypeCompute* h_matrixRT = new TypeCompute[SizeMat4x4];

  for (int i = 0; i < nbDepthMap; i++)
    {
    std::cout << "\r" << (100 * i) / nbDepthMap << " %" << std::flush;

    ReconstructionData data(vtiList[i], krtdList[i]);
    data.ApplyDepthThresholdFilter(thresholdBestCost);

    // Get data and transform its properties to atomic type
    vtkImageDataToTable(data.GetDepthMap(), h_depthMap);
    vtkMatrixToTypeComputeTable(data.Get4MatrixK(), h_matrixK);
    vtkMatrixToTypeComputeTable(data.GetMatrixTR(), h_matrixRT);

    // Wait that all threads have finished
    CudaErrorCheck(cudaDeviceSynchronize());

    CudaErrorCheck(cudaMemcpy(d_depthMap, h_depthMap, nbPixelOnDepthMap * sizeof(TypeCompute), cudaMemcpyHostToDevice));
    CudaErrorCheck(cudaMemcpy(d_matrixK, h_matrixK, SizeMat4x4 * sizeof(TypeCompute), cudaMemcpyHostToDevice));
    CudaErrorCheck(cudaMemcpy(d_matrixRT, h_matrixRT, SizeMat4x4 * sizeof(TypeCompute), cudaMemcpyHostToDevice));

    // run code into device
    depthMapKernel<TVolumetric> << < dimGrid, dimBlock >> >(d_depthMap, d_matrixK, d_matrixRT, d_outScalar);

    }

  // Transfer data from device to host
  cudaMemcpy(h_outScalar, d_outScalar, nbVoxels * sizeof(TVolumetric), cudaMemcpyDeviceToHost);

  // Transfer host data to output
  doubleTableToVtkDoubleArray<TVolumetric>(h_outScalar, io_scalar);

  // Clean memory
  delete(h_outScalar);
  cudaFree(d_outScalar);
  cudaFree(d_depthMap);
  cudaFree(d_matrixK);
  cudaFree(d_matrixRT);
  delete(h_depthMap);
  delete(h_matrixK);
  delete(h_matrixRT);

  std::cout << "\r" << "100 %" << std::flush << std::endl << std::endl;

  return true;
}

// ----------------------------------------------------------------------------
// Define template for the compiler
template
bool ProcessDepthMap<float>(std::vector<std::string> vtiList,
std::vector<std::string> krtdList,
double thresholdBestCost,
vtkDoubleArray* io_scalar);

template
bool ProcessDepthMap<double>(std::vector<std::string> vtiList,
std::vector<std::string> krtdList,
double thresholdBestCost,
vtkDoubleArray* io_scalar);

#endif
