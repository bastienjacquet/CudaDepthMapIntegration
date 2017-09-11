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
#include <cuda_runtime.h>
#include "ReconstructionData.h"

// STD include
#include <limits>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>

// VTK includes
#include "vtkCommand.h"
#include "vtkPointData.h"
#include "vtkDoubleArray.h"
#include "vtkMatrix4x4.h"
#include "vtkMath.h"
#include "vtkImageData.h"
#include "vtkNew.h"
#include "vtkCellData.h"
#include "vtkXMLImageDataWriter.h"
#include "vtkCudaReconstructionFilter.h"

#define SizeMat4x4 16
#define SizePoint3D 3
#define SizeDim3D 3
// Apply to matrix, computes on 3D point
typedef float TypeCompute;

// Type for a voxel : floating point type for the scalar, integer type for the count
// (used to compute a rolling mean of the ray potential on the voxel's scalar)
template <typename TVolumetric, typename TCount>
struct TypeVoxel
{
  TVolumetric scalar = 0.0;
  TCount count = 0;
};

// ----------------------------------------------------------------------------
/* Define texture and constants */
__constant__ TypeCompute c_gridMatrix[SizeMat4x4]; // Matrix to transpose from basic axis to output volume axis
__constant__ TypeCompute c_gridOrig[SizePoint3D]; // Origin of the output volume
__constant__ int3 c_gridNbVoxels; // Dimensions of the output volume
__constant__ TypeCompute c_gridSpacing[SizeDim3D]; // Spacing of the output volume
__constant__ int2 c_depthMapDims; // Dimensions of depth map
__constant__ int c_depthmapType; // Type of depthmaps : Structure from motion or spherical
__constant__ TypeCompute c_depthMapOrigin[SizePoint3D]; // Origin coordinates for spherical depthmaps
__constant__ TypeCompute c_depthMapSpacing[SizeDim3D]; // Spacing for spherical depthmaps
__constant__ TypeCompute c_matrixK[SizeMat4x4];
__constant__ TypeCompute c_matrixRT[SizeMat4x4];
__constant__ int3 c_tileNbVoxels; // Dimensions of the tiles
__constant__ int c_nbVoxels; // Total number of voxels
__constant__ TypeCompute c_rayPotentialThick; // Thickness threshold for the ray potential function
__constant__ TypeCompute c_rayPotentialRho; // Rho at the Y axis for the ray potential function
__constant__ TypeCompute c_rayPotentialEta;
__constant__ TypeCompute c_rayPotentialDelta;
__constant__ TypeCompute c_verticalAngleThreshold;
__constant__ size_t c_maxCountValue;

int ch_gridNbVoxels[SizeDim3D]; // Dimensions of the output volume
int ch_depthMapType; // Type of depthmaps : Structure from motion or spherical
int h_tileNbVoxels[SizeDim3D]; // Dimensions of the tiles
vtkCudaReconstructionFilter* ch_reconstructionFilter; // Used to invoke event on the progress bar

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
/* Compute the voxel's center coordinates from its integer coordinates */
__device__ void computeVoxelCenter(int voxelCoordinate[SizePoint3D], TypeCompute output[SizePoint3D])
{
  output[0] = c_gridOrig[0] + (voxelCoordinate[0] + 0.5) * c_gridSpacing[0];
  output[1] = c_gridOrig[1] + (voxelCoordinate[1] + 0.5) * c_gridSpacing[1];
  output[2] = c_gridOrig[2] + (voxelCoordinate[2] + 0.5) * c_gridSpacing[2];
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
__device__ TypeCompute norm(TypeCompute vec[3])
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
int computeVoxelIDGrid(int coordinates[SizePoint3D])
{
  int dimX = ch_gridNbVoxels[0];
  int dimY = ch_gridNbVoxels[1];
  int i = coordinates[0];
  int j = coordinates[1];
  int k = coordinates[2];
  return (k*dimY + j)*dimX + i;
}


// ----------------------------------------------------------------------------
/* Compute the voxel's 3D coordinates in tile according to its ID on a 1D table
  gridID : 1D coordinate
  coordinates : 3D coordinates
*/
void computeVoxel3DCoords(int gridId, int tileSize[SizeDim3D], int coordinates[SizePoint3D])
{
  coordinates[0] = gridId % tileSize[0];
  coordinates[1] = (gridId / tileSize[0]) % tileSize[1];
  coordinates[2] = ((gridId / tileSize[0]) / tileSize[1]) % tileSize[2];
}


// ----------------------------------------------------------------------------
/* Compute the tiles' origins as 3D coordinates according to their size and
 * the size of the voxel grid
  nbTilesXYZ : number of tiles in each dimension
  tileOrigin : 3D coordinates
*/
void computeTileOrigins(int nbTilesXYZ[SizeDim3D], int tileOrigin[][SizeDim3D])
{
  for (int x = 0; x < nbTilesXYZ[0]; x++)
  {
    for (int y = 0; y < nbTilesXYZ[1]; y++)
    {
      for (int z = 0; z < nbTilesXYZ[2]; z++)
      {
        int id = z + nbTilesXYZ[2]*(y + nbTilesXYZ[1]*x);
        tileOrigin[id][0] = x * h_tileNbVoxels[0];
        tileOrigin[id][1] = y * h_tileNbVoxels[1];
        tileOrigin[id][2] = z * h_tileNbVoxels[2];
      }
    }
  }
}


// ----------------------------------------------------------------------------
/* Compute the tiles' dimensions to use all GPUs
*/
template<typename TVoxel>
void computeTileNbVoxels(int nbDevices)
{
  // Initialize tile sizes such as there is one tile per device
  h_tileNbVoxels[0] = ch_gridNbVoxels[0];
  h_tileNbVoxels[1] = ch_gridNbVoxels[1];
  h_tileNbVoxels[2] = vtkMath::Ceil(static_cast<double>(ch_gridNbVoxels[2]) / nbDevices);

  size_t freeMemory, free, totalMemory;

  // Find the minimum amount of free memory on the devices
  for (int i = 0; i < nbDevices; i++)
  {
    CudaErrorCheck( cudaSetDevice(i) );
    CudaErrorCheck( cudaMemGetInfo(&free, &totalMemory) );

    if (i == 0)
    {
      freeMemory = free;
    }
    else
    {
      freeMemory = std::min(freeMemory, free);
    }
  }

  size_t voxelsPerTile = static_cast<size_t>(h_tileNbVoxels[0]) * h_tileNbVoxels[1] * h_tileNbVoxels[2];
  int usagePercent = 80;
  size_t freeVoxels = static_cast<size_t>(usagePercent * freeMemory)
                      / (100 * sizeof(TVoxel));

  // Use free GPU memory to reduce tile sizes if need be
  while (voxelsPerTile > freeVoxels)
  {
    // Subdivide the Z dimension
    if (h_tileNbVoxels[2] > 1)
    {
      h_tileNbVoxels[2] = vtkMath::Ceil(static_cast<double>(h_tileNbVoxels[2]) / 2.0);
    }
    else
    {
      // Subdivide the Y dimension
      if (h_tileNbVoxels[1] > 1)
      {
        h_tileNbVoxels[1] = vtkMath::Ceil(static_cast<double>(h_tileNbVoxels[1]) / 2.0);
      }
      // Subdivide the X dimension
      else
      {
        h_tileNbVoxels[0] = vtkMath::Ceil(static_cast<double>(h_tileNbVoxels[0]) / 2.0);
      }
    }

    voxelsPerTile = static_cast<size_t>(h_tileNbVoxels[0]) * h_tileNbVoxels[1] * h_tileNbVoxels[2];
  }
}


// ----------------------------------------------------------------------------
/* Copy the tile data to its spatial region in output scalar
  tileOrigin : 3D coordinates
*/
template<typename TVolumetric, typename TCount>
void copyTileDataToOutput(int nbVoxelsTile, int tileId, int tileOrigin[SizePoint3D],
  TypeVoxel<TVolumetric, TCount>* outTile, TVolumetric* outScalar, TCount* outCount)
{
  for (int k = 0; k < nbVoxelsTile; k++)
  {
    int voxelIndexRelative[SizePoint3D];
    computeVoxel3DCoords(k, h_tileNbVoxels, voxelIndexRelative);

    // Compute real voxel index from its relative index
    int voxelIndex[SizePoint3D];
    voxelIndex[0] = tileOrigin[0] + voxelIndexRelative[0];
    voxelIndex[1] = tileOrigin[1] + voxelIndexRelative[1];
    voxelIndex[2] = tileOrigin[2] + voxelIndexRelative[2];

    // Don't process out of bounds voxels
    if (voxelIndex[0] < ch_gridNbVoxels[0]
        && voxelIndex[1] < ch_gridNbVoxels[1]
        && voxelIndex[2] < ch_gridNbVoxels[2])
    {
      int gridId = computeVoxelIDGrid(voxelIndex);
      // Copy scalar and depthmap contribution count information to output
      outScalar[gridId] = outTile[k].scalar;
      outCount[gridId] = outTile[k].count;
    }
  }
}


// ----------------------------------------------------------------------------
/* Compute the voxel relative Id on a 1D table according to its 3D coordinates
  coordinates : 3D coordinates
*/
__device__ int computeVoxelRelativeIDGrid(int coordinates[SizePoint3D])
{
  int dimX = c_tileNbVoxels.x;
  int dimY = c_tileNbVoxels.y;
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
  tileOrigin : 3D coordinates
  depths : depth map values
  matrixK : matrixK
  matrixTR : matrixTR
  output : double table that will be filled at the end of function
*/
template<typename TVolumetric, typename TCount>
__global__ void depthMapKernel(int tileOrigin[SizePoint3D], TypeCompute* depths,
                               TypeVoxel<TVolumetric, TCount>* output)
{
  // Get relative voxel coordinates of the voxel according to thread id
  int voxelIndexRelative[SizePoint3D] = { (int)threadIdx.x, (int)blockIdx.y, (int)blockIdx.z };
  int gridIdRelative = computeVoxelRelativeIDGrid(voxelIndexRelative);

  // Get true voxel coordinates
  int voxelIndex[SizePoint3D];
  voxelIndex[0] = tileOrigin[0] + voxelIndexRelative[0];
  voxelIndex[1] = tileOrigin[1] + voxelIndexRelative[1];
  voxelIndex[2] = tileOrigin[2] + voxelIndexRelative[2];

  // Don't process out of bounds voxels
  if (voxelIndex[0] < c_gridNbVoxels.x
      && voxelIndex[1] < c_gridNbVoxels.y
      && voxelIndex[2] < c_gridNbVoxels.z)
  {
    int pixel[SizePoint3D] = {0};
    TypeCompute realDepth;
    int depthMapId;

    // Compute the voxel's center coordinates from its 3D index
    TypeCompute voxelCenterCoordinate[SizePoint3D];
    computeVoxelCenter(voxelIndex, voxelCenterCoordinate);

    // User defined change of cartesian coordinate system (defaults to identity)
    TypeCompute voxelCenter[SizePoint3D];
    transformFrom4Matrix(c_gridMatrix, voxelCenterCoordinate, voxelCenter);

    // Transform voxel center from real coord to camera coords
    TypeCompute voxelCenterCamera[SizePoint3D];
    transformFrom4Matrix(c_matrixRT, voxelCenter, voxelCenterCamera);

    // Structure from motion depthmap
    if (c_depthmapType == 0)
    {


      // Transform voxel center from camera coords to depth map homogeneous coords
      TypeCompute voxelCenterHomogen[SizePoint3D];
      transformFrom4Matrix(c_matrixK, voxelCenterCamera, voxelCenterHomogen);
      if (voxelCenterHomogen[2] < 0)
      {
        return;
      }
      // Get voxel center on depth map coord
      TypeCompute voxelCenterDepthMap[2];
      voxelCenterDepthMap[0] = voxelCenterHomogen[0] / voxelCenterHomogen[2];
      voxelCenterDepthMap[1] = voxelCenterHomogen[1] / voxelCenterHomogen[2];

      // Get real pixel position (approximation)
      pixel[0] = round(voxelCenterDepthMap[0]);
      pixel[1] = round(voxelCenterDepthMap[1]);
      //pixel[2] = 0;

      realDepth = voxelCenterCamera[2];

      // Compute the ID on depthmap values according to pixel position and depth map dimensions
      depthMapId = pixel[0] + c_depthMapDims.x * (c_depthMapDims.y - 1 - pixel[1]);


    }
    // Spherical depthmap
    else if (c_depthmapType == 1)
    {


      TypeCompute* cartesian = voxelCenterCamera;

      TypeCompute spherical[SizePoint3D];
      // Rho : distance (depth) in meters
      spherical[2] =  sqrt(cartesian[0] * cartesian[0]
                           + cartesian[1] * cartesian[1]
                           + cartesian[2] * cartesian[2]);

      // Phi : vertical angle (from XY-plane to vector) in ]-pi/2; pi/2]
      spherical[1] = asin(cartesian[2] / spherical[2]);
      // Theta : azimuth (horizontal angle from Y-axis to vector) in ]-pi; pi]
      spherical[0] = atan2(cartesian[0], cartesian[1]);

      // The convention used is that the pixel represent the center of a cell.
      pixel[0] = round((spherical[0] - c_depthMapOrigin[0])
                        / c_depthMapSpacing[0]);

      // Phi index candidate to read information from on depthmap
      TypeCompute phi_idx = (spherical[1] - c_depthMapOrigin[1])/ c_depthMapSpacing[1];
      pixel[1] = round(phi_idx);

      // Test if Phi is close enough to the closest depthmap phi
      if (abs(phi_idx - pixel[1]) * c_depthMapSpacing[1] > c_verticalAngleThreshold)
      {
        return;
      }
      //pixel[2] = 0;

      realDepth = spherical[2];

      // Compute the ID on depthmap values according to pixel position and depth map dimensions
      depthMapId = pixel[0] + c_depthMapDims.x * pixel[1];


    }

    // Test if coordinate are inside depth map
    if (pixel[0] < 0 || pixel[1] < 0
        || pixel[0] >= c_depthMapDims.x
        || pixel[1] >= c_depthMapDims.y)
    {
      return;
    }

    TypeCompute depth = depths[depthMapId];
    if (depth == -1.0)
    {
      return;
    }

    TVolumetric newValue;
    rayPotential<TVolumetric>(realDepth, depth, newValue);
    if (newValue == 0.0)
    {
      return;
    }
    // Increment the number of depthmap contributions for this point
    if (output[gridIdRelative].count < c_maxCountValue)
    {
      output[gridIdRelative].count++;
    }

    // Update the value to the output with a rolling average
    output[gridIdRelative].scalar *= TVolumetric(1.0 - 1.0/TVolumetric(output[gridIdRelative].count));
    output[gridIdRelative].scalar += newValue / TVolumetric(output[gridIdRelative].count);
  }
}


// ----------------------------------------------------------------------------
/* Convert a data array to TypeCompute table */
__host__ void doubleTableToTypeComputeTable(double* array, int numElements, TypeCompute* output)
{
  for (int i = 0; i < numElements; i++)
  {
    output[i] = static_cast<TypeCompute>(array[i]);
  }
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
      output[cpt++] = static_cast<TypeCompute>(matrix->GetElement(i, j));
    }
  }
}


// ----------------------------------------------------------------------------
/* Extract double value from vtkDoubleArray and fill a double table (output) */
template <typename T>
__host__ void vtkDataArrayToTable(vtkDataArray* dataArray, T* output)
{
  for (int i = 0; i < dataArray->GetNumberOfTuples(); i++)
  {
    output[i] = static_cast<T>(dataArray->GetTuple1(i));
  }
}


// ----------------------------------------------------------------------------
/* Extract point data array (name 'Depths') from vtkImageData and fill a double table */
__host__ void vtkImageDataToTypeComputeTable(vtkImageData* image, TypeCompute* output)
{
  vtkDataArray* depths = vtkDataArray::SafeDownCast(image->GetPointData()->GetArray("Depths"));
  if (depths != NULL)
  {
    vtkDataArrayToTable<TypeCompute>(depths, output);
  }
  else
  {
    std::cerr << "Could not get the Depths information from the vtkImageData depthmap." << std::endl;
  }

}


// ----------------------------------------------------------------------------
/* Fill a vtkDoubleArray from a double table */
//template<typename TVolumetric>
//__host__ void doubleTableToVtkDoubleArray(TVolumetric* table, vtkDoubleArray* output)
//{
//  int nbVoxels = output->GetNumberOfTuples();
//  for (int i = 0; i < nbVoxels; i++)
//  {
//    output->SetTuple1(i, (double)table[i]);
//  }
//}


// ----------------------------------------------------------------------------
/* Initialize cuda constant */
void CudaInitialize(vtkMatrix4x4* i_gridMatrix, // Matrix to transform grid voxel to real coordinates
                    int h_gridNbVoxels[SizeDim3D], // Dimensions of the output volume
                    double gridOrigin[SizePoint3D], // Origin of the output volume
                    double gridSpacing[SizeDim3D], // Spacing of the output volume
                    double rayPThick,
                    double rayPRho,
                    double rayPEta,
                    double rayPDelta,
                    int h_tilingSize[SizeDim3D],
                    int h_depthMapDims[2],
                    int h_depthMapType,
                    vtkCudaReconstructionFilter* h_reconstructionFilter) // Used to invoke progress event
{
  TypeCompute h_gridMatrix[SizeMat4x4];
  vtkMatrixToTypeComputeTable(i_gridMatrix, h_gridMatrix);

  TypeCompute h_gridOrigin[SizePoint3D];
  doubleTableToTypeComputeTable(gridOrigin, SizePoint3D, h_gridOrigin);
  TypeCompute h_gridSpacing[SizeDim3D];
  doubleTableToTypeComputeTable(gridSpacing, SizeDim3D, h_gridSpacing);
  TypeCompute h_rayPThick = static_cast<TypeCompute>(rayPThick);
  TypeCompute h_rayPRho = static_cast<TypeCompute>(rayPRho);
  TypeCompute h_rayPEta = static_cast<TypeCompute>(rayPEta);
  TypeCompute h_rayPDelta = static_cast<TypeCompute>(rayPDelta);

  // (Spherical Depthmaps) Allow 0.5 degrees threshold for vertical angle when reconstructing
  TypeCompute verticalAngleThreshold = 0.5 * vtkMath::Pi() / 180.0;

  // Device constants
  cudaMemcpyToSymbol(c_gridMatrix, h_gridMatrix, SizeMat4x4 * sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_gridNbVoxels, h_gridNbVoxels, SizeDim3D * sizeof(int));
  cudaMemcpyToSymbol(c_gridOrig, h_gridOrigin, SizePoint3D * sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_gridSpacing, h_gridSpacing, SizeDim3D * sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_rayPotentialThick, &h_rayPThick, sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_rayPotentialRho, &h_rayPRho, sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_rayPotentialEta, &h_rayPEta, sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_rayPotentialDelta, &h_rayPDelta, sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_depthMapDims, h_depthMapDims, 2 * sizeof(int));
  cudaMemcpyToSymbol(c_depthmapType, &h_depthMapType, sizeof(int));
  cudaMemcpyToSymbol(c_verticalAngleThreshold, &verticalAngleThreshold, sizeof(TypeCompute));

  // Host values
  for (int i = 0; i < 3; i++)
  {
    ch_gridNbVoxels[i] = h_gridNbVoxels[i];
    h_tileNbVoxels[i] = h_tilingSize[i];
  }

  ch_reconstructionFilter = h_reconstructionFilter;
  ch_depthMapType = h_depthMapType;
}


// ----------------------------------------------------------------------------
/* Read all depth map and process each of them. Fill the output 'io_scalar' */
template <typename TVolumetric, typename TCount>
bool ProcessDepthMap(std::vector<std::string> vtiList,
                     std::vector<std::string> krtdList,
                     double thresholdBestCost,
                     TVolumetric *io_scalar,
                     TCount* io_count)
{
  if (vtiList.size() == 0 || krtdList.size() == 0)
  {
    std::cerr << "Error, no depthMap or KRTD matrix have been loaded" << std::endl;
    return false;
  }

  // Define useful constant values
  ReconstructionData data0(vtiList[0].c_str(), krtdList[0].c_str());
  int nbPixelOnDepthMap = data0.GetDepthMap()->GetNumberOfPoints();

  size_t nbVoxels = static_cast<size_t>(ch_gridNbVoxels[0]) * ch_gridNbVoxels[1] * ch_gridNbVoxels[2];
  CudaErrorCheck( cudaMemcpyToSymbol(c_nbVoxels, &nbVoxels, sizeof(int)) );

  const int nbDepthMap = (int)vtiList.size();

  // Max value for TCount : rolling mean count information
  int maxCountValue = std::numeric_limits<TCount>::max();
  CudaErrorCheck( cudaMemcpyToSymbol(c_maxCountValue, &maxCountValue, sizeof(int)) );

  int nbDevices;
  CudaErrorCheck( cudaGetDeviceCount(&nbDevices) );

  std::cout << "START CUDA ON " << nbDepthMap << " Depth maps ("
  << nbDevices << " devices)" << std::endl;

  // Create depthmap device data from host value
  TypeCompute* h_depthMap = new TypeCompute[nbPixelOnDepthMap];
  TypeCompute* d_depthMap;
  CudaErrorCheck( cudaMalloc((void**)&d_depthMap, nbPixelOnDepthMap * sizeof(TypeCompute)) );

  // Depthmap parameters
  TypeCompute h_matrixRT[SizeMat4x4];
  // Structure from motion depthmaps
  TypeCompute h_matrixK[SizeMat4x4];
  // Spherical depthmaps
  int h_depthMapDimensions[SizeDim3D];
  TypeCompute h_depthMapOrigin[SizePoint3D];
  TypeCompute h_depthMapSpacing[SizeDim3D];


  // Runtime calculated tiling
  if (h_tileNbVoxels[0] == 0 && h_tileNbVoxels[1] == 0 && h_tileNbVoxels[2] == 0)
  {
    computeTileNbVoxels< TypeVoxel<TVolumetric, TCount> >(nbDevices);
  }
  CudaErrorCheck( cudaMemcpyToSymbol(c_tileNbVoxels, h_tileNbVoxels, 3 * sizeof(int)) );
  std::cout << "Tile size : " << h_tileNbVoxels[0] << "x" << h_tileNbVoxels[1] << "x"
  << h_tileNbVoxels[2] << " voxels" << std::endl;
  bool oneTileOnly = true;
  for (int i = 0; i < 3; i++)
  {
    if (ch_gridNbVoxels[i] != h_tileNbVoxels[i])
    {
      oneTileOnly = false;
      break;
    }
  }

  int nbTilesXYZ[SizeDim3D];
  // Compute the numbers of tiles needed to fill each dimension
  for (int i = 0; i < 3; i++)
  {
    nbTilesXYZ[i] = vtkMath::Ceil(static_cast<double>(ch_gridNbVoxels[i]) / h_tileNbVoxels[i]);
  }

  // Define tiling dimensions
  const size_t nbVoxelsTile = static_cast<size_t>(h_tileNbVoxels[0]) * h_tileNbVoxels[1] * h_tileNbVoxels[2];
  const int nbTiles = nbTilesXYZ[0] * nbTilesXYZ[1] * nbTilesXYZ[2];

  std::cout << "Tiling : " << nbTilesXYZ[0] << "x" << nbTilesXYZ[1]
  << "x" << nbTilesXYZ[2] << " (" << nbTiles << ")" << std::endl;

  // Compute and allocate tile information
  int tileOrigin[nbTiles][SizePoint3D];
  computeTileOrigins(nbTilesXYZ, tileOrigin);
  int* d_tileOrigin;
  CudaErrorCheck( cudaMalloc((void**)&d_tileOrigin, SizeDim3D * sizeof(int)) );
  TypeVoxel<TVolumetric, TCount>* h_outTile;
  h_outTile = new TypeVoxel<TVolumetric, TCount>[nbVoxelsTile];

  TypeVoxel<TVolumetric, TCount>* d_outTile;
  CudaErrorCheck( cudaMalloc((void**)&d_outTile,
                             nbVoxelsTile * sizeof(TypeVoxel<TVolumetric, TCount>)) );
  CudaErrorCheck( cudaMemset(d_outTile, 0,
                             nbVoxelsTile * sizeof(TypeVoxel<TVolumetric, TCount>)) );

  // Organize threads into blocks and grids
  dim3 dimBlock(h_tileNbVoxels[0], 1, 1); // nb threads on each block
  dim3 dimGrid(1, h_tileNbVoxels[1], h_tileNbVoxels[2]); // nb blocks on a grid

  // Define how the tiles are processed
  int nbConcurrent = std::min(nbTiles, nbDevices);
  int nbSequential = vtkMath::Ceil(static_cast<double>(nbTiles) / nbConcurrent);

  // Process the tiles
  for (int is = 0; is < nbSequential; is++)
  {
    std::cout << "\ntile: " << (nbConcurrent * is);
    if (nbConcurrent > 1)
    {
      std::cout << " to " << std::min(nbConcurrent, nbTiles - is * nbConcurrent);
    }
    std::cout << "\t(" << (nbConcurrent * is) +
    std::min(nbConcurrent, nbTiles - is * nbConcurrent) << "/" << nbTiles
    << ")" << std::endl;

    for (int j = 0; j < nbDepthMap; j++)
    {


      double progress = (static_cast<double>(is) / static_cast<double>(nbSequential))
      + (static_cast<double>(j) / static_cast<double>(nbDepthMap * nbSequential));

      // Update filter progress
      ch_reconstructionFilter->InvokeEvent(vtkCommand::ProgressEvent, &progress);

      if (false)
      {
        std::cout << "\r" << int(100 * progress) << "%\t("
                  << (100 * j) / nbDepthMap << " %)" << std::flush;
      }

      // Get data and transform its properties to atomic type
      ReconstructionData data(vtiList[j].c_str(), krtdList[j].c_str());

      if (ch_depthMapType == vtkCudaReconstructionFilter::STRUCTURE_FROM_MOTION)
      {
        data.ApplyDepthThresholdFilter(thresholdBestCost);
        vtkMatrixToTypeComputeTable(data.Get4MatrixK(), h_matrixK);
      }
      else if (ch_depthMapType == vtkCudaReconstructionFilter::SPHERICAL)
      {
        nbPixelOnDepthMap = data.GetDepthMap()->GetNumberOfPoints();

        // Reallocate space for the depthmaps of variable size
        delete(h_depthMap);
        cudaFree(d_depthMap);
        h_depthMap = new TypeCompute[nbPixelOnDepthMap];
        CudaErrorCheck( cudaMalloc((void**)&d_depthMap, nbPixelOnDepthMap * sizeof(TypeCompute)) );

        data.GetDepthMap()->GetDimensions(h_depthMapDimensions);

        double depthMapOrigin[SizePoint3D], depthMapSpacing[SizeDim3D];
        data.GetDepthMap()->GetOrigin(depthMapOrigin);
        data.GetDepthMap()->GetSpacing(depthMapSpacing);

        doubleTableToTypeComputeTable(depthMapOrigin, SizePoint3D, h_depthMapOrigin);
        doubleTableToTypeComputeTable(depthMapSpacing, SizeDim3D, h_depthMapSpacing);
      }

      vtkMatrixToTypeComputeTable(data.GetMatrixTR(), h_matrixRT);
      vtkImageDataToTypeComputeTable(data.GetDepthMap(), h_depthMap);

      // Copy data to devices and run kernels
      for (int ic = 0; ic < std::min(nbConcurrent, nbTiles - is * nbConcurrent); ic++)
      {
        int tileId = ic + nbConcurrent * is;

        CudaErrorCheck( cudaSetDevice(ic) );

        // Wait that all threads have finished on selected device
        CudaErrorCheck( cudaDeviceSynchronize() );

        // Copy data from host to device
        CudaErrorCheck( cudaMemcpy(d_tileOrigin, tileOrigin[tileId],
                                   SizeDim3D * sizeof(int), cudaMemcpyHostToDevice) );
        CudaErrorCheck( cudaMemcpy(d_depthMap, h_depthMap,
                                   nbPixelOnDepthMap * sizeof(TypeCompute),
                                   cudaMemcpyHostToDevice) );
        CudaErrorCheck( cudaMemcpyToSymbol(c_matrixRT, h_matrixRT,
                                           SizeMat4x4 * sizeof(TypeCompute)) );

        if (ch_depthMapType == vtkCudaReconstructionFilter::STRUCTURE_FROM_MOTION)
        {
          CudaErrorCheck( cudaMemcpyToSymbol(c_matrixK, h_matrixK,
                                             SizeMat4x4 * sizeof(TypeCompute)) );
        }
        // Spherical depthmaps : variable dimensions, origins and spacing attributes
        else if (ch_depthMapType == vtkCudaReconstructionFilter::SPHERICAL)
        {
          CudaErrorCheck( cudaMemcpyToSymbol(c_depthMapDims, h_depthMapDimensions,
                                             2 * sizeof(int)) );
          CudaErrorCheck( cudaMemcpyToSymbol(c_depthMapOrigin, h_depthMapOrigin,
                                             SizePoint3D * sizeof(TypeCompute)) );
          CudaErrorCheck( cudaMemcpyToSymbol(c_depthMapSpacing, h_depthMapSpacing,
                                             SizeDim3D * sizeof(TypeCompute)) );
        }

        // run code into device
        depthMapKernel<TVolumetric> << < dimGrid, dimBlock >> >
        (d_tileOrigin, d_depthMap, d_outTile);
      }
    }

    // Retrieve devices tile output and update host voxels
    for (int ic = 0; ic < std::min(nbConcurrent, nbTiles - is * nbConcurrent); ic++)
    {
      int tileId = ic + nbConcurrent * is;

      CudaErrorCheck( cudaSetDevice(ic) );

      // Wait that all threads have finished on selected device
      CudaErrorCheck( cudaDeviceSynchronize() );

      // Transfer tile data from device in host memory
      CudaErrorCheck( cudaMemcpy(h_outTile, d_outTile,
                                 nbVoxelsTile * sizeof(TypeVoxel<TVolumetric, TCount>),
                                 cudaMemcpyDeviceToHost) );

      // Copy data from tile to output TVolumetric array
      copyTileDataToOutput<TVolumetric, TCount>(nbVoxelsTile, tileId, tileOrigin[tileId],
                                        h_outTile, io_scalar, io_count);

      if (!oneTileOnly)
      {
        // Reset the current device voxel tile to 0s
        CudaErrorCheck( cudaMemset(d_outTile, 0,
                                   nbVoxelsTile * sizeof(TypeVoxel<TVolumetric, TCount>)) );
      }
    }
  }


  // Clean memory
  delete(h_outTile);
  cudaFree(d_outTile);
  cudaFree(d_depthMap);
  delete(h_depthMap);

  std::cout << "\r" << "100 %" << std::endl << std::endl;
  double progress = 1.0;
  ch_reconstructionFilter->InvokeEvent(vtkCommand::ProgressEvent, &progress);

  return true;
}

// ----------------------------------------------------------------------------
// Define template for the compiler
template
bool ProcessDepthMap<float, unsigned short>(std::vector<std::string> vtiList,
std::vector<std::string> krtdList,
double thresholdBestCost,
float* io_scalar,
unsigned short* io_count);

template
bool ProcessDepthMap<double, unsigned short>(std::vector<std::string> vtiList,
std::vector<std::string> krtdList,
double thresholdBestCost,
double* io_scalar,
unsigned short* io_count);


#endif
