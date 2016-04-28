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

#include <stdio.h>
#include <vector>
#include "vtkPointData.h"
#include "vtkDoubleArray.h"
#include "vtkMatrix4x4.h"
#include <math.h>
#include "ReconstructionData.h"

#define Mat4x4 16
#define Point3D 3
#define Dim3D 3
// Apply to matrix, computes on 3D point
typedef double TCompute;

// ----------------------------------------------------------------------------
/* Define texture and constants */
__constant__ TCompute c_gridMatrix[Mat4x4]; // Matrix to transpose from basic axis to output volume axis
__constant__ TCompute c_gridOrig[Point3D]; // Origin of the output volume
__constant__ int3 c_gridDims; // Dimensions of the output volume
__constant__ TCompute c_gridSpacing[Dim3D]; // Spacing of the output volume
__constant__ int2 c_depthMapDims; // Dimensions of all depths map
__constant__ TCompute c_rayPotentialThick; // Thickness threshold for the ray potential function
__constant__ TCompute c_rayPotentialRho; // Rho at the Y axis for the ray potential function
__constant__ TCompute c_rayPotentialEta;
__constant__ TCompute c_rayPotentialDelta;
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
__device__ void transformFrom4Matrix(TCompute M[Mat4x4], TCompute point[Point3D], TCompute output[Point3D])
{
  output[0] = M[0 * 4 + 0] * point[0] + M[0 * 4 + 1] * point[1] + M[0 * 4 + 2] * point[2] + M[0 * 4 + 3];
  output[1] = M[1 * 4 + 0] * point[0] + M[1 * 4 + 1] * point[1] + M[1 * 4 + 2] * point[2] + M[1 * 4 + 3];
  output[2] = M[2 * 4 + 0] * point[0] + M[2 * 4 + 1] * point[1] + M[2 * 4 + 2] * point[2] + M[2 * 4 + 3];
}


// ----------------------------------------------------------------------------
/* Compute the norm of a table with 3 double */
__device__ TCompute norm(TCompute vec[Dim3D])
{
  return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

// ----------------------------------------------------------------------------
/* Ray potential function which computes the increment to the current voxel */
template<typename TVolumetric>
__device__ void rayPotential(TCompute realDistance, TCompute depthMapDistance, TVolumetric& res)
{
  TCompute diff = (realDistance - depthMapDistance);

  TCompute absolute = abs(diff);
  // Can't divide by zero
  int sign = diff != 0 ? diff / absolute : 0;

  if (absolute > c_rayPotentialDelta)
    res = diff > 0 ? 0 : -c_rayPotentialEta*c_rayPotentialRho;
  else if (abs(diff) > c_rayPotentialThick)
    res = c_rayPotentialRho*sign;
  else
    res = (c_rayPotentialRho / c_rayPotentialThick)* diff;
}

// ----------------------------------------------------------------------------
/* Compute the voxel Id on a 1D table according to its 3D coordinates
  coordinates : 3D coordinates
*/
__device__ int computeVoxelIDGrid(int coordinates[Point3D])
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
__device__ int computeVoxelIDDepth(int coordinates[Point3D])
{
  int dimX = c_depthMapDims.x;
  int dimY = c_depthMapDims.y;
  int x = coordinates[0];
  int y = coordinates[1];
  // /!\ vtkImageData has its origin at the bottom left, not top left
  return (dimX*(dimY-1-y)) + x;
}

// ----------------------------------------------------------------------------
/* Compute the middle of a voxel according to constant global value and the origin of the voxel */
__device__ void computeVoxelCenter(int voxelCoordinate[Point3D], TCompute output[Point3D])
{
  output[0] = c_gridOrig[0] + (voxelCoordinate[0] + 0.5) * c_gridSpacing[0];
  output[1] = c_gridOrig[1] + (voxelCoordinate[1] + 0.5) * c_gridSpacing[1];
  output[2] = c_gridOrig[2] + (voxelCoordinate[2] + 0.5) * c_gridSpacing[2];
}

// ----------------------------------------------------------------------------
/* Main function called inside the kernel
  depths : depth map values
  matrixK : matrixK
  matrixTR : matrixTR
  output : double table that will be filled at the end of function
*/
template<typename TVolumetric>
__global__ void depthMapKernel(TCompute* depths, TCompute matrixK[Mat4x4], TCompute matrixTR[Mat4x4],
  TVolumetric* output)
{
  // Get voxel coordinate according to thread id
  int i = threadIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;
  int voxelIndex[Point3D] = { i, j, k };

  // Get the center of the voxel
  TCompute voxelCenterCoordinate[Point3D];
  computeVoxelCenter(voxelIndex, voxelCenterCoordinate);

  // Transform voxel from grid to real coord
  TCompute voxelCenter[Point3D];
  transformFrom4Matrix(c_gridMatrix, voxelCenterCoordinate, voxelCenter);

  // Transform voxel center from real coord to camera coords
  TCompute voxelCenterCamera[Point3D];
  transformFrom4Matrix(matrixTR, voxelCenter, voxelCenterCamera);

  // Transform voxel center from camera coords to depth map homogeneous coords
  TCompute voxelCenterHomogen[Point3D];
  transformFrom4Matrix(matrixK, voxelCenterCamera, voxelCenterHomogen);
  if (voxelCenterHomogen[2] < 0)
    {
    return;
    }
  // Get voxel center on depth map coord
  TCompute voxelCenterDepthMap[2];
  voxelCenterDepthMap[0] = voxelCenterHomogen[0] / voxelCenterHomogen[2];
  voxelCenterDepthMap[1] = voxelCenterHomogen[1] / voxelCenterHomogen[2];
  // Get real pixel position (approximation)
  int pixel[Point3D];
  pixel[0] = round(voxelCenterDepthMap[0]);
  pixel[1] = round(voxelCenterDepthMap[1]);
  pixel[2] = 0;

  // Test if coordinate are inside depth map
  if (pixel[0] < 0 || pixel[1] < 0 || 
    pixel[0] > c_depthMapDims.x - 1 ||
    pixel[1] > c_depthMapDims.y - 1)
    {
    return;
    }

  // Compute the ID on depthmap values according to pixel position and depth map dimensions
  int depthMapId = computeVoxelIDDepth(pixel);
  TCompute depth = depths[depthMapId];
  if (depth == -1)
    {
    return;
    }
  int gridId = computeVoxelIDGrid(voxelIndex);  // Get the distance between voxel and camera
  TCompute realDepth = norm(voxelCenterCamera);
  TVolumetric newValue;
  rayPotential<TVolumetric>(realDepth, depth, newValue);
  // Update the value to the output
  output[gridId] += newValue;
}





// ----------------------------------------------------------------------------
/* Extract data from a 4x4 vtkMatrix and fill a double table with 16 space */
__host__ void vtkMatrixToTComputeTable(vtkMatrix4x4* matrix, TCompute* output)
{
  int cpt = 0;
  for (int i = 0; i < 4; i++)
    {
    for (int j = 0; j < 4; j++)
      {
      output[cpt++] = (TCompute)matrix->GetElement(i, j);
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
__host__ void vtkImageDataToTable(vtkImageData* image, TCompute* output)
{
  vtkDoubleArray* depths = vtkDoubleArray::SafeDownCast(image->GetPointData()->GetArray("Depths"));
  vtkDoubleArrayToTable<TCompute>(depths, output);
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
                int h_gridDims[Dim3D], // Dimensions of the output volume
                double h_gridOrig[Point3D], // Origin of the output volume
                double h_gridSpacing[Dim3D], // Spacing of the output volume
                double h_rayPThick,
                double h_rayPRho,
                double h_rayPEta,
                double h_rayPDelta,
                int h_depthMapDims[2])
{
  TCompute* h_gridMatrix = new TCompute[Mat4x4];
  vtkMatrixToTComputeTable(i_gridMatrix, h_gridMatrix);

  cudaMemcpyToSymbol(c_gridMatrix, h_gridMatrix, Mat4x4 * sizeof(TCompute));
  cudaMemcpyToSymbol(c_gridDims, h_gridDims, Dim3D * sizeof(int));
  cudaMemcpyToSymbol(c_gridOrig, h_gridOrig, Point3D * sizeof(TCompute));
  cudaMemcpyToSymbol(c_gridSpacing, h_gridSpacing, Dim3D * sizeof(TCompute));
  cudaMemcpyToSymbol(c_rayPotentialThick, &h_rayPThick, sizeof(TCompute));
  cudaMemcpyToSymbol(c_rayPotentialRho, &h_rayPRho, sizeof(TCompute));
  cudaMemcpyToSymbol(c_rayPotentialEta, &h_rayPEta, sizeof(TCompute));
  cudaMemcpyToSymbol(c_rayPotentialDelta, &h_rayPDelta, sizeof(TCompute));
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

  std::cout << "START CUDA ON " << std::to_string(nbDepthMap) << " Depth map" << std::endl;

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
  TCompute *d_depthMap, *d_matrixK, *d_matrixRT;
  CudaErrorCheck(cudaMalloc((void**)&d_depthMap, nbPixelOnDepthMap * sizeof(TCompute)));
  CudaErrorCheck(cudaMalloc((void**)&d_matrixK, Mat4x4 * sizeof(TCompute)));
  CudaErrorCheck(cudaMalloc((void**)&d_matrixRT, Mat4x4 * sizeof(TCompute)));


  for (int i = 0; i < nbDepthMap; i++)
    {
    if (i%3 == 0)
      std::cout << (100 * i) / nbDepthMap << " %" << std::endl;

    ReconstructionData data(vtiList[i], krtdList[i]);
    data.ApplyDepthThresholdFilter(thresholdBestCost);

    // Get data and transform its properties to atomic type
    TCompute* h_depthMap = new TCompute[nbPixelOnDepthMap];
    vtkImageDataToTable(data.GetDepthMap(), h_depthMap);
    TCompute* h_matrixK = new TCompute[Mat4x4];
    vtkMatrixToTComputeTable(data.Get4MatrixK(), h_matrixK);
    TCompute* h_matrixRT = new TCompute[Mat4x4];
    vtkMatrixToTComputeTable(data.GetMatrixTR(), h_matrixRT);

    // Wait that all threads have finished
    CudaErrorCheck(cudaDeviceSynchronize());

    CudaErrorCheck(cudaMemcpy(d_depthMap, h_depthMap, nbPixelOnDepthMap * sizeof(TCompute), cudaMemcpyHostToDevice));
    CudaErrorCheck(cudaMemcpy(d_matrixK, h_matrixK, Mat4x4 * sizeof(TCompute), cudaMemcpyHostToDevice));
    CudaErrorCheck(cudaMemcpy(d_matrixRT, h_matrixRT, Mat4x4 * sizeof(TCompute), cudaMemcpyHostToDevice));

    // clean memory
    delete(h_depthMap);
    delete(h_matrixK);
    delete(h_matrixRT);

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

  std::cout << "END CUDA : 100%" << std::endl;
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
