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
__device__ void transformFrom4Matrix(TCompute matrix[Mat4x4], TCompute point[Point3D], TCompute output[Point3D])
{
  output[0] = matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3];
  output[1] = matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7];
  output[2] = matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11];
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
    res = diff > 0 ? 0 : -c_rayPotentialEta;
  else if (abs(diff) > c_rayPotentialThick)
    res = c_rayPotentialRho*sign;
  else
    res = (c_rayPotentialRho / c_rayPotentialThick)* diff;
}


// ----------------------------------------------------------------------------
/* Compute the voxel Id on a 1D table according to its 3D coordinates
  coordinates : 3D coordinates
  type : Define if we want the Id from the grid matrix ( = 1) or the depth map ( = 0 )
         because we don't use  the same dimensions
  TOBEIMPROVED
*/
__device__ int computeVoxelID(int coordinates[Point3D], int type)
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


// ----------------------------------------------------------------------------
/* Compute the middle of a voxel according to constant global value and the origin of the voxel */
__device__ void computeVoxelCenter(int voxelCoordinate[Point3D], TCompute output[Point3D])
{
  output[0] = c_gridOrig[0] + (voxelCoordinate[0] + 0.5) * c_gridSpacing[0];
  output[1] = c_gridOrig[1] + (voxelCoordinate[1] + 0.5) * c_gridSpacing[1];
  output[2] = c_gridOrig[2] +(voxelCoordinate[2] + 0.5) * c_gridSpacing[2];
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
  int voxelCoordinate[Point3D] = { i, j, k };

  // Get the center of the voxel
  TCompute voxelCenterTemp[Point3D];
  computeVoxelCenter(voxelCoordinate, voxelCenterTemp);

  // Transform voxel from grid to real coord
  TCompute voxelCenter[Point3D];
  transformFrom4Matrix(c_gridMatrix, voxelCenterTemp, voxelCenter);

  // Transform voxel center from real coord to camera coords
  TCompute voxelCenterCamera[Point3D];
  transformFrom4Matrix(matrixTR, voxelCenter, voxelCenterCamera);

  // Transform voxel center from camera coords to depth map homogeneous coords
  TCompute voxelCenterHomogen[Point3D];
  transformFrom4Matrix(matrixK, voxelCenterCamera, voxelCenterHomogen);

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
  if (pixel[0] < 0 || pixel[1] < 0 || pixel[2] < 0 ||
    pixel[0] > c_depthMapDims.x - 1 ||
    pixel[1] > c_depthMapDims.y - 1)
    {
      return;
    }

  // Compute the ID on depthmap values according to pixel position and depth map dimensions
  int depthMapId = computeVoxelID(pixel, 0);
  int gridId = computeVoxelID(voxelCoordinate, 1);  // Get the distance between voxel and camera
  TCompute realDepth = norm(voxelCenterCamera);
  TCompute depth = depths[depthMapId];
  TVolumetric newValue;
  rayPotential<TVolumetric>(realDepth, depth, newValue);

  // Update the value to the output
  output[gridId] += newValue;
}


// ----------------------------------------------------------------------------
/* Extract data from a 4x4 vtkMatrix and fill a double table with 16 space */
__host__ void vtkMatrixToTComputeTable(vtkMatrix4x4* matrix, TCompute* output)
{
  output[0] = (TCompute)matrix->GetElement(0, 0);
  output[1] = (TCompute)matrix->GetElement(0, 1);
  output[2] = (TCompute)matrix->GetElement(0, 2);
  output[3] = (TCompute)matrix->GetElement(0, 3);
  output[4] = (TCompute)matrix->GetElement(1, 0);
  output[5] = (TCompute)matrix->GetElement(1, 1);
  output[6] = (TCompute)matrix->GetElement(1, 2);
  output[7] = (TCompute)matrix->GetElement(1, 3);
  output[8] = (TCompute)matrix->GetElement(2, 0);
  output[9] = (TCompute)matrix->GetElement(2, 1);
  output[10] = (TCompute)matrix->GetElement(2, 2);
  output[11] = (TCompute)matrix->GetElement(2, 3);
  output[12] = (TCompute)matrix->GetElement(3, 0);
  output[13] = (TCompute)matrix->GetElement(3, 1);
  output[14] = (TCompute)matrix->GetElement(3, 2);
  output[15] = (TCompute)matrix->GetElement(3, 3);
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
/** Main function **/
template <typename TVolumetric>
int reconstruction(std::vector<ReconstructionData*> h_dataList, // List of depth matrix and associated matrix
                   vtkMatrix4x4* i_gridMatrix, // Matrix to transform grid voxel to real coordinates
                   int h_gridDims[Dim3D], // Dimensions of the output volume
                   double h_gridOrig[Point3D], // Origin of the output volume
                   double h_gridSpacing[Dim3D], // Spacing of the output volume
                   double h_rayPThick,
                   double h_rayPRho,
                   double h_rayPEta,
                   double h_rayPDelta,
                   vtkDoubleArray* io_outScalar) // It will be filled inside function
{
  if (h_dataList.size() == 0)
    return -1;

  // Get usefull value for allocation of variables
  const int nbPixelOnDepthMap = h_dataList[0]->GetDepthMap()->GetNumberOfPoints();
  const int nbVoxels = io_outScalar->GetNumberOfTuples();

  // Fill texture and constant values
  TCompute* h_gridMatrix = new TCompute[Mat4x4];
  vtkMatrixToTComputeTable(i_gridMatrix, h_gridMatrix);
  TVolumetric* h_outScalar = new TVolumetric[nbVoxels];
  vtkDoubleArrayToTable<TVolumetric>(io_outScalar, h_outScalar);


  // Create and fill device value
  cudaMemcpyToSymbol(c_gridMatrix, h_gridMatrix, Mat4x4 * sizeof(TCompute));
  cudaMemcpyToSymbol(c_gridDims, h_gridDims, Dim3D * sizeof(int));
  cudaMemcpyToSymbol(c_gridOrig, h_gridOrig, Point3D * sizeof(TCompute));
  cudaMemcpyToSymbol(c_gridSpacing, h_gridSpacing, Dim3D * sizeof(TCompute));
  cudaMemcpyToSymbol(c_rayPotentialThick, &h_rayPThick, sizeof(TCompute));
  cudaMemcpyToSymbol(c_rayPotentialRho, &h_rayPRho, sizeof(TCompute));
  cudaMemcpyToSymbol(c_rayPotentialEta, &h_rayPEta, sizeof(TCompute));
  cudaMemcpyToSymbol(c_rayPotentialDelta, &h_rayPDelta, sizeof(TCompute));
  TVolumetric* d_outScalar;
  CudaErrorCheck(cudaMalloc((void**)&d_outScalar, nbVoxels * sizeof(TVolumetric)));
  CudaErrorCheck(cudaMemcpy(d_outScalar, h_outScalar, nbVoxels * sizeof(TVolumetric), cudaMemcpyHostToDevice));


  int h_dimDepthMap[Dim3D];
  h_dataList[0]->GetDepthMap()->GetDimensions(h_dimDepthMap);
  CudaErrorCheck(cudaMemcpyToSymbol(c_depthMapDims, h_dimDepthMap, 2 * sizeof(int)));

  // Organize threads into blocks and grids
  dim3 dimBlock(h_gridDims[0] - 1, 1, 1); // nb threads on each block
  dim3 dimGrid(1, h_gridDims[1] - 1, h_gridDims[2] - 1); // nb blocks on a grid

  // Create device data from host value
  TCompute *d_depthMap, *d_matrixK, *d_matrixRT;
  CudaErrorCheck(cudaMalloc((void**)&d_depthMap, nbPixelOnDepthMap * sizeof(TCompute)));
  CudaErrorCheck(cudaMalloc((void**)&d_matrixK, Mat4x4 * sizeof(TCompute)));
  CudaErrorCheck(cudaMalloc((void**)&d_matrixRT, Mat4x4 * sizeof(TCompute)));
  
  for (int i = 0; i < h_dataList.size(); i++)
    {
    // Get data and transform its properties to atomic type
    ReconstructionData* currentData = h_dataList[i];
    TCompute* h_depthMap = new TCompute[nbPixelOnDepthMap];
    vtkImageDataToTable(currentData->GetDepthMap(), h_depthMap);
    TCompute* h_matrixK = new TCompute[Mat4x4];
    vtkMatrixToTComputeTable(currentData->Get4MatrixK(), h_matrixK);
    TCompute* h_matrixRT = new TCompute[Mat4x4];
    vtkMatrixToTComputeTable(currentData->GetMatrixTR(), h_matrixRT);

    CudaErrorCheck(cudaMemcpy(d_depthMap, h_depthMap, nbPixelOnDepthMap * sizeof(TCompute), cudaMemcpyHostToDevice));
    CudaErrorCheck(cudaMemcpy(d_matrixK, h_matrixK, Mat4x4 * sizeof(TCompute), cudaMemcpyHostToDevice));
    CudaErrorCheck(cudaMemcpy(d_matrixRT, h_matrixRT, Mat4x4 * sizeof(TCompute), cudaMemcpyHostToDevice));

    // run code into device
    depthMapKernel<TVolumetric> <<< dimGrid, dimBlock >>>(d_depthMap, d_matrixK, d_matrixRT, d_outScalar);

    // Wait that all threads have finished
    CudaErrorCheck(cudaDeviceSynchronize());

    // clean memory
    delete(h_depthMap);
    delete(h_matrixK);
    delete(h_matrixRT);
    }

  // Transfer data from device to host
  cudaMemcpy(h_outScalar, d_outScalar, nbVoxels * sizeof(TVolumetric), cudaMemcpyDeviceToHost);

  // Transfer host data to output
  doubleTableToVtkDoubleArray<TVolumetric>(h_outScalar, io_outScalar);

  // Clean memory
  delete(h_gridMatrix);
  delete(h_outScalar);
  cudaFree(d_outScalar);
  cudaFree(d_depthMap);
  cudaFree(d_matrixK);
  cudaFree(d_matrixRT);

  return 1;
}


// ----------------------------------------------------------------------------
// Define template for the compiler
template
int reconstruction<double>(std::vector<ReconstructionData*> h_dataList,
  vtkMatrix4x4* i_gridMatrix, int h_gridDims[Dim3D],
  double h_gridOrig[Point3D],double h_gridSpacing[Dim3D],
  double h_rayPThick,double h_rayPRho, double h_rayPEta, double h_rayPDelta, vtkDoubleArray* io_outScalar);

template
int reconstruction <float>(std::vector<ReconstructionData*> h_dataList,
  vtkMatrix4x4* i_gridMatrix,int h_gridDims[Dim3D],
  double h_gridOrig[Point3D],double h_gridSpacing[Dim3D],
  double h_rayPThick, double h_rayPRho, double h_rayPEta, double h_rayPDelta, vtkDoubleArray* io_outScalar);

#endif
