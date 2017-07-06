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
#include <string>

// VTK includes
#include "vtkPointData.h"
#include "vtkDoubleArray.h"
#include "vtkMatrix4x4.h"
#include "vtkImageData.h"
#include "vtkNew.h"
#include "vtkCellData.h"
#include "vtkXMLImageDataWriter.h"

#define SizeMat4x4 16
#define SizePoint3D 3
#define SizeDim3D 3
// Apply to matrix, computes on 3D point
typedef double TypeCompute;

// ----------------------------------------------------------------------------
/* Define texture and constants */
__constant__ TypeCompute c_gridMatrix[SizeMat4x4]; // Matrix to transpose from basic axis to output volume axis
__constant__ TypeCompute c_gridOrig[SizePoint3D]; // Origin of the output volume
__constant__ int3 c_gridDims; // Dimensions of the output volume
__constant__ TypeCompute c_gridSpacing[SizeDim3D]; // Spacing of the output volume
__constant__ int2 c_depthMapDims; // Dimensions of all depths map
__constant__ int3 c_tileDims; // Dimensions of the tiles
__constant__ int c_nbVoxels; // Total number of voxels
__constant__ TypeCompute c_rayPotentialThick; // Thickness threshold for the ray potential function
__constant__ TypeCompute c_rayPotentialRho; // Rho at the Y axis for the ray potential function
__constant__ TypeCompute c_rayPotentialEta;
__constant__ TypeCompute c_rayPotentialDelta;
double ch_gridOrigin[3];
int ch_gridDims[3];
double ch_gridSpacing[3];
int h_tileDims[3];

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
int computeVoxelIDGrid(int coordinates[SizePoint3D])
{
  int dimX = ch_gridDims[0] - 1;
  int dimY = ch_gridDims[1] - 1;
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
void computeVoxel3DCoords(int gridId, int tileDims[3], int coordinates[SizePoint3D])
{
  coordinates[0] = gridId % tileDims[0];
  coordinates[1] = (gridId / tileDims[0]) % tileDims[1];
  coordinates[2] = ((gridId / tileDims[0]) / tileDims[1]) % tileDims[2];
}

// ----------------------------------------------------------------------------
/* Compute the tiles' origins as 3D coordinates according to their size and
 * the size of the voxel grid
  nbTilesXYZ : number of tiles in each dimension
  tileOrigin : 3D coordinates
*/
void computeTileOrigins(int nbTilesXYZ[3], int tileOrigin[][3])
{
  for (int x = 0; x < nbTilesXYZ[0]; x++)
  {
    for (int y = 0; y < nbTilesXYZ[1]; y++)
    {
      for (int z = 0; z < nbTilesXYZ[2]; z++)
      {
        int id = z + nbTilesXYZ[2]*(y + nbTilesXYZ[1]*x);
        tileOrigin[id][0] = x * h_tileDims[0];
        tileOrigin[id][1] = y * h_tileDims[1];
        tileOrigin[id][2] = z * h_tileDims[2];

        std::cout<<"tileOrigin["<<id<<"] : "<<tileOrigin[id][0]<<" "<<tileOrigin[id][1]<<" "<<tileOrigin[id][2]<<std::endl;
      }
    }
  }
}

// ----------------------------------------------------------------------------
/* Compute the tiles' dimensions to use all GPUs
*/
template<typename TVolumetric>
void computeTileDims(int nbDevices)
{
  // Initialize tile sizes such as there is one tile per device
  h_tileDims[0] = ch_gridDims[0] - 1;
  h_tileDims[1] = ch_gridDims[1] - 1;
  h_tileDims[2] = vtkMath::Ceil(double(ch_gridDims[2] - 1)) / nbDevices;

  size_t freeMemory, free, totalMemory;

  // Find the minimum amount of free memory on the devices
  for (int i = 0; i < nbDevices; i++)
  {
    CudaErrorCheck(cudaSetDevice(i));
    CudaErrorCheck(cudaMemGetInfo(&free, &totalMemory));

    if (i == 0)
    {
      freeMemory = free;
    }
    else
    {
      freeMemory = std::min(freeMemory, free);
    }
  }

  int voxelsPerTile = h_tileDims[0] * h_tileDims[1] * h_tileDims[2];
  int usagePercent = 80;
  int freeVoxels = double(usagePercent * freeMemory) / (100 * sizeof(TVolumetric));

  // Use free GPU memory to reduce tile sizes if need be
  while (voxelsPerTile > freeVoxels)
  {
    // Subdivide the Z dimension
    if (h_tileDims[2] > 1)
    {
      h_tileDims[2] = vtkMath::Ceil(double(h_tileDims[2]) / 2);
    }
    else
    {
      // Subdivide the Y dimension
      if (h_tileDims[1] > 1)
      {
        h_tileDims[1] = vtkMath::Ceil(double(h_tileDims[1]) / 2);
      }
      else
      {
        // Subdivide the X dimension
        h_tileDims[0] = vtkMath::Ceil(double(h_tileDims[0]) / 2);
      }
    }

    voxelsPerTile = h_tileDims[0] * h_tileDims[1] * h_tileDims[2];
  }
}

// ----------------------------------------------------------------------------
/* Copy the tile data to its spatial region in output scalar
  tileOrigin : 3D coordinates
*/
template<typename TVolumetric>
void copyTileDataToOutput(int nbVoxelsTile, int tileId, int tileOrigin[3], TVolumetric* outTile, TVolumetric* outScalar)
{
  for (int k = 0; k < nbVoxelsTile; k++)
  {
    int voxelIndexRelative[SizePoint3D];
    computeVoxel3DCoords(k, h_tileDims, voxelIndexRelative);

    int voxelIndex[SizePoint3D];
    voxelIndex[0] = tileOrigin[0] + voxelIndexRelative[0];
    voxelIndex[1] = tileOrigin[1] + voxelIndexRelative[1];
    voxelIndex[2] = tileOrigin[2] + voxelIndexRelative[2];

    if (voxelIndex[0] < ch_gridDims[0] - 1
    && voxelIndex[1] < ch_gridDims[1] - 1
    && voxelIndex[2] < ch_gridDims[2] - 1)
    {
      int gridId = computeVoxelIDGrid(voxelIndex);
      outScalar[gridId] = outTile[k];
    }
  }
}

// ----------------------------------------------------------------------------
/* Compute the voxel relative Id on a 1D table according to its 3D coordinates
  coordinates : 3D coordinates
*/
__device__ int computeVoxelRelativeIDGrid(int coordinates[SizePoint3D])
{
  int dimX = c_tileDims.x;
  int dimY = c_tileDims.y;
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
__global__ void depthMapKernel(int d_tileOrigin[3], TypeCompute* depths, TypeCompute matrixK[SizeMat4x4],
  TypeCompute matrixTR[SizeMat4x4], TVolumetric* output)
{
  // Get relative voxel coordinate of the voxel according to thread id
  int voxelIndexRelative[SizePoint3D] = { (int)threadIdx.x, (int)blockIdx.y, (int)blockIdx.z };
  int gridIdRelative = computeVoxelRelativeIDGrid(voxelIndexRelative);

  // Get true voxel coordinate
  int voxelIndex[SizePoint3D];
  voxelIndex[0] = d_tileOrigin[0] + voxelIndexRelative[0];
  voxelIndex[1] = d_tileOrigin[1] + voxelIndexRelative[1];
  voxelIndex[2] = d_tileOrigin[2] + voxelIndexRelative[2];

  // Don't process out of bounds voxels
  if (voxelIndex[0] < c_gridDims.x - 1
  && voxelIndex[1] < c_gridDims.y - 1
  && voxelIndex[2] < c_gridDims.z - 1)
  {
    TypeCompute voxelCenterCoordinate[SizePoint3D];
    computeVoxelCenter(voxelIndex, voxelCenterCoordinate);
    TypeCompute voxelCenter[SizePoint3D];
    transformFrom4Matrix(c_gridMatrix, voxelCenterCoordinate, voxelCenter);

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

    TypeCompute realDepth = voxelCenterCamera[2];
    TVolumetric newValue;
    rayPotential<TVolumetric>(realDepth, depth, newValue);
    // Update the value to the output
    output[gridIdRelative] += newValue;
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
                double h_gridOrig[SizePoint3D], // Origin of the output volume
                double h_gridSpacing[SizeDim3D], // Spacing of the output volume
                double h_rayPThick,
                double h_rayPRho,
                double h_rayPEta,
                double h_rayPDelta,
                int h_depthMapDims[2],
                int h_tilingDims[SizeDim3D])
{

  TypeCompute* h_gridMatrix = new TypeCompute[SizeMat4x4];
  vtkMatrixToTypeComputeTable(i_gridMatrix, h_gridMatrix);

  cudaMemcpyToSymbol(c_gridMatrix, h_gridMatrix, SizeMat4x4 * sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_gridDims, h_gridDims, SizeDim3D * sizeof(int));
  cudaMemcpyToSymbol(c_gridOrig, h_gridOrig, SizePoint3D * sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_gridSpacing, h_gridSpacing, SizeDim3D * sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_rayPotentialThick, &h_rayPThick, sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_rayPotentialRho, &h_rayPRho, sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_rayPotentialEta, &h_rayPEta, sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_rayPotentialDelta, &h_rayPDelta, sizeof(TypeCompute));
  cudaMemcpyToSymbol(c_depthMapDims, h_depthMapDims, 2 * sizeof(int));

  ch_gridOrigin[0] = h_gridOrig[0];
  ch_gridOrigin[1] = h_gridOrig[1];
  ch_gridOrigin[2] = h_gridOrig[2];

  ch_gridDims[0] = h_gridDims[0];
  ch_gridDims[1] = h_gridDims[1];
  ch_gridDims[2] = h_gridDims[2];

  ch_gridSpacing[0] = h_gridSpacing[0];
  ch_gridSpacing[1] = h_gridSpacing[1];
  ch_gridSpacing[2] = h_gridSpacing[2];

  h_tileDims[0] = h_tilingDims[0];
  h_tileDims[1] = h_tilingDims[1];
  h_tileDims[2] = h_tilingDims[2];

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
  CudaErrorCheck(cudaMemcpyToSymbol(c_nbVoxels, &nbVoxels, sizeof(int)));
  const int nbDepthMap = (int)vtiList.size();
  int nbDevices;
  cudaGetDeviceCount(&nbDevices);
  std::cout << "number of devices : " << nbDevices << std::endl;

  std::cout << "START CUDA ON " << nbDepthMap << " Depth maps" << std::endl;

  // Create depthmap device data from host value
  TypeCompute *d_depthMap, *d_matrixK, *d_matrixRT;
  CudaErrorCheck(cudaMalloc((void**)&d_depthMap, nbPixelOnDepthMap * sizeof(TypeCompute)));
  CudaErrorCheck(cudaMalloc((void**)&d_matrixK, SizeMat4x4 * sizeof(TypeCompute)));
  CudaErrorCheck(cudaMalloc((void**)&d_matrixRT, SizeMat4x4 * sizeof(TypeCompute)));
  TypeCompute* h_depthMap = new TypeCompute[nbPixelOnDepthMap];
  TypeCompute* h_matrixK = new TypeCompute[SizeMat4x4];
  TypeCompute* h_matrixRT = new TypeCompute[SizeMat4x4];

  // Runtime calculated tiling
  if (h_tileDims[0] == 0 && h_tileDims[1] == 0 && h_tileDims[2] == 0)
  {
    computeTileDims<TVolumetric>(nbDevices);
  }
  CudaErrorCheck(cudaMemcpyToSymbol(c_tileDims, h_tileDims, 3 * sizeof(int)));

  int nbTilesXYZ[3];
  // Compute the numbers of tiles needed to fill each dimension
  for (int i = 0; i < 3; i++)
  {
    nbTilesXYZ[i] = vtkMath::Ceil(double(ch_gridDims[i] - 1) / (h_tileDims[i]));
  }
  std::cout << "Tiling X : " << nbTilesXYZ[0] << ", Y : " << nbTilesXYZ[1]
  << ", Z : " << nbTilesXYZ[2] << std::endl;

  // Define tiling dimensions
  const int nbVoxelsTile = h_tileDims[0] * h_tileDims[1] * h_tileDims[2];
  const int nbTiles = nbTilesXYZ[0] * nbTilesXYZ[1] * nbTilesXYZ[2];

  // Compute and allocate tile information
  int tileOrigin[nbTiles][3];
  computeTileOrigins(nbTilesXYZ, tileOrigin);
  int* d_tileOrigin;
  CudaErrorCheck(cudaMalloc((void**)&d_tileOrigin, 3 * sizeof(int)));
  TVolumetric* h_outTile = new TVolumetric[nbVoxelsTile];
  TVolumetric* d_outTile;
  CudaErrorCheck(cudaMalloc((void**)&d_outTile, nbVoxelsTile * sizeof(TVolumetric)));
  CudaErrorCheck(cudaMemset(d_outTile, 0, nbVoxelsTile * sizeof(TVolumetric)));

  std::cout << "Tile : " << h_tileDims[0] << "x" << h_tileDims[1] << "x" << h_tileDims[2] << std::endl;
  size_t freeMemory, totalMemory;
  CudaErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));
  std::cout << "Free nb after tiling : " << freeMemory / sizeof(TVolumetric) << std::endl << std::endl;

  // Organize threads into blocks and grids
  dim3 dimBlock(h_tileDims[0], 1, 1); // nb threads on each block
  dim3 dimGrid(1, h_tileDims[1], h_tileDims[2]); // nb blocks on a grid

  // Transform vtkDoubleArray to table
  TVolumetric* h_outScalar = new TVolumetric[nbVoxels];
  vtkDoubleArrayToTable<TVolumetric>(io_scalar, h_outScalar);

  int nbConcurrent = std::min(nbTiles, nbDevices);
  int nbSequential = vtkMath::Ceil(double(nbTiles) / nbConcurrent);

  // Tiles output init
  double gridOrigin[3];
  vtkNew<vtkImageData> gridTile;
  gridTile->SetDimensions(h_tileDims);
  gridTile->SetSpacing(ch_gridSpacing);
  gridTile->SetOrigin(0.0, 0.0, 0.0);

  vtkNew<vtkDoubleArray> outScalarTile;
  outScalarTile->SetName("reconstruction_scalar");
  outScalarTile->SetNumberOfComponents(1);
  outScalarTile->SetNumberOfTuples(gridTile->GetNumberOfCells());
  outScalarTile->FillComponent(0, 0);

  gridTile->GetCellData()->AddArray(outScalarTile.Get());

  vtkNew<vtkXMLImageDataWriter> writer;
  writer->SetInputData(gridTile.Get());
  std::string tileFileName;
  //

  // Process the tilescalar
  for (int is = 0; is < nbSequential; is++)
  {
    std::cout << "\ntile: " << (nbConcurrent * is);
    if(nbConcurrent > 1)
    {
      std::cout << "to " << std::min(nbConcurrent, nbTiles - is * nbConcurrent);
    }
    std::cout << "\t(" << (nbConcurrent * is) + std::min(nbConcurrent, nbTiles - is * nbConcurrent) << "/" << nbTiles << ")" << std::endl;

    for (int j = 0; j < nbDepthMap; j++)
    {
      if (j % (nbDepthMap / 10) == 0)
      {
        std::cout << (100 * j) / nbDepthMap <<" %\t("
        << ((100 * is) / nbSequential) + ((j * 100) / (nbDepthMap * nbSequential)) << " %)" << std::flush;
      }

      // Init depthmap data to be transfered
      if(j == 0)
      {
        ReconstructionData data(vtiList[0], krtdList[0]);
        data.ApplyDepthThresholdFilter(thresholdBestCost);

        // Get data and transform its properties to atomic type
        vtkImageDataToTable(data.GetDepthMap(), h_depthMap);
        vtkMatrixToTypeComputeTable(data.Get4MatrixK(), h_matrixK);
        vtkMatrixToTypeComputeTable(data.GetMatrixTR(), h_matrixRT);
      }

      // Copy data to devices and run kernels
      for (int ic = 0; ic < std::min(nbConcurrent, nbTiles - is * nbConcurrent); ic++)
      {
        int tileId = ic + nbConcurrent * is;

        CudaErrorCheck(cudaSetDevice(ic));

        // Wait that all threads have finished on selected device
        CudaErrorCheck(cudaDeviceSynchronize());

        // Copy data from host to
        CudaErrorCheck(cudaMemcpy(d_tileOrigin, tileOrigin[tileId], 3 * sizeof(int), cudaMemcpyHostToDevice));
        CudaErrorCheck(cudaMemcpy(d_depthMap, h_depthMap, nbPixelOnDepthMap * sizeof(TypeCompute), cudaMemcpyHostToDevice));
        CudaErrorCheck(cudaMemcpy(d_matrixK, h_matrixK, SizeMat4x4 * sizeof(TypeCompute), cudaMemcpyHostToDevice));
        CudaErrorCheck(cudaMemcpy(d_matrixRT, h_matrixRT, SizeMat4x4 * sizeof(TypeCompute), cudaMemcpyHostToDevice));

        // run code into device
        depthMapKernel<TVolumetric> << < dimGrid, dimBlock >> >(d_tileOrigin, d_depthMap, d_matrixK, d_matrixRT, d_outTile);
      }

      // Prepare next depthmap data while kernel is running
      if (j < nbDepthMap - 1)
      {
        ReconstructionData data(vtiList[j+1], krtdList[j+1]);
        data.ApplyDepthThresholdFilter(thresholdBestCost);

        // Get data and transform its properties to atomic type
        vtkImageDataToTable(data.GetDepthMap(), h_depthMap);
        vtkMatrixToTypeComputeTable(data.Get4MatrixK(), h_matrixK);
        vtkMatrixToTypeComputeTable(data.GetMatrixTR(), h_matrixRT);
      }
    }

    // Retrieve devices tile output and update host voxels
    for (int ic = 0; ic < std::min(nbConcurrent, nbTiles - is * nbConcurrent); ic++)
    {
      int tileId = ic + nbConcurrent * is;

      CudaErrorCheck(cudaSetDevice(ic));

      // Wait that all threads have finished on selected device
      CudaErrorCheck(cudaDeviceSynchronize());

      // Transfer tile data from device in host memory
      CudaErrorCheck(cudaMemcpy(h_outTile, d_outTile, nbVoxelsTile * sizeof(TVolumetric), cudaMemcpyDeviceToHost));

      // Reset the current device voxel tile to 0s
      CudaErrorCheck(cudaMemset(d_outTile, 0, nbVoxelsTile * sizeof(TVolumetric)));

      ////
      // Convert
      doubleTableToVtkDoubleArray<TVolumetric>(h_outTile, outScalarTile.Get());
      // Save tile to vtkImageData
      // Set tile origin
      for (int k = 0; k < 3; k++)
      {
        gridOrigin[k] = ch_gridOrigin[k] + tileOrigin[tileId][k] * ch_gridSpacing[k];
      }
      gridTile->SetOrigin(gridOrigin);

      tileFileName = "tile_" + std::to_string(tileId) + ".vti";
      writer->SetFileName(tileFileName.c_str());
      writer->Write();

      // Copy data from tile to output double array
      copyTileDataToOutput<TVolumetric>(nbVoxelsTile, tileId, tileOrigin[tileId], h_outTile, h_outScalar);
    }
  }

  // Transfer host data to output
  doubleTableToVtkDoubleArray<TVolumetric>(h_outScalar, io_scalar);

  // Clean memory.
  delete(data);
  delete(h_outScalar);
  delete(h_outTile);
  cudaFree(d_outTile);
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
