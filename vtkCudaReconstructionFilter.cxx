// Copyright(c) 2016, Kitware SAS
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

#include "vtkCudaReconstructionFilter.h"

#include "vtkCell.h"
#include "vtkCellData.h"
#include "vtkDataSet.h"
#include "vtkDoubleArray.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMath.h"
#include "vtkMatrix3x3.h"
#include "vtkMatrix4x4.h"
#include "vtkNew.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkStructuredGrid.h"
#include "vtkTransform.h"

#include <cmath>
#include <vector>
#include <time.h>

vtkStandardNewMacro(vtkCudaReconstructionFilter);
vtkSetObjectImplementationMacro(vtkCudaReconstructionFilter, GridMatrix, vtkMatrix4x4);

// Define the function signature in .cu file in order to be recognize inside the file
int reconstruction(std::vector<ReconstructionData*> i_dataList, vtkMatrix4x4* i_gridMatrix, 
  int h_gridDims[3], double h_gridOrig[3], double h_gridSpacing[3], double h_rayPThick,
  double h_rayPRho, vtkDoubleArray* h_outScalar);

//----------------------------------------------------------------------------
vtkCudaReconstructionFilter::vtkCudaReconstructionFilter()
{
  this->SetNumberOfInputPorts(1);
  this->GridMatrix = 0;
  this->UseCuda = false;
  this->RayPotentialRho = 0;
  this->RayPotentialThickness = 0;
}

//----------------------------------------------------------------------------
vtkCudaReconstructionFilter::~vtkCudaReconstructionFilter()
{
  this->DataList.clear();
}

//----------------------------------------------------------------------------
void vtkCudaReconstructionFilter::SetDataList(std::vector<ReconstructionData*> list)
{
  this->DataList = list;
}

//----------------------------------------------------------------------------
void vtkCudaReconstructionFilter::UseCudaOn()
{
  this->SetUseCuda(true);
}

//----------------------------------------------------------------------------
void vtkCudaReconstructionFilter::UseCudaOff()
{
  this->SetUseCuda(false);
}

//----------------------------------------------------------------------------
int vtkCudaReconstructionFilter::RequestData(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
  // get the info objects
  vtkInformation *inGridInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *outGridInfo = outputVector->GetInformationObject(0);

  // get the input and output
  vtkImageData *inGrid = vtkImageData::SafeDownCast(
    inGridInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData *outGrid = vtkImageData::SafeDownCast(
    outGridInfo->Get(vtkDataObject::DATA_OBJECT()));

  if (this->DataList.size() == 0 || this->GridMatrix == 0)
    {
    std::cerr << "Error, some inputs have not been set." << std::endl;
    return 0;
    }

  // get grid info
  double gridOrig[3];
  inGrid->GetOrigin(gridOrig);
  int gridDims[3];
  inGrid->GetDimensions(gridDims);
  double gridSpacing[3];
  inGrid->GetSpacing(gridSpacing);


  // initialize output
  vtkNew<vtkDoubleArray> outScalar;
  outScalar->SetName("reconstruction_scalar");
  outScalar->SetNumberOfComponents(1);
  outScalar->SetNumberOfTuples(inGrid->GetNumberOfCells());
  outScalar->FillComponent(0, 0);
  outGrid->ShallowCopy(inGrid);
  outGrid->GetCellData()->AddArray(outScalar.Get());

  // computation
  if (!this->UseCuda)
    {
    clock_t start = clock();
    for (int i = 0; i < this->DataList.size(); i++)
      {
      ReconstructionData* currentData = this->DataList[i];
      vtkCudaReconstructionFilter::ComputeWithoutCuda(
        this->GridMatrix, gridOrig, gridDims, gridSpacing,
        currentData->GetDepthMap(), currentData->Get3MatrixK(), currentData->GetMatrixTR(),
        outScalar.Get());
      }
    clock_t end = clock();
    double diff = (double)(end - start) / CLOCKS_PER_SEC;
    std::cout << "Time WITHOUT CUDA : " << diff << " s" << std::endl;
    }
  else
    {
    // Check if all variables used in cuda are set
    if (this->RayPotentialRho == 0 && this->RayPotentialThickness == 0)
      {
      std::cerr << "Error : Ray potential Rho or Thickness or both have not been set" << std::endl;
      return 0;
      }

    clock_t start = clock();
    reconstruction(this->DataList, this->GridMatrix, gridDims, gridOrig, gridSpacing,
                   this->RayPotentialThickness, this->RayPotentialRho, outScalar.Get());
    clock_t end = clock();
    double diff = (double)(end - start) / CLOCKS_PER_SEC;
    std::cout << "Time WITH CUDA : " << diff << " s" << std::endl;
    }

  return 1;
}

//----------------------------------------------------------------------------
int vtkCudaReconstructionFilter::ComputeWithoutCuda(
    vtkMatrix4x4 *gridMatrix, double gridOrig[3], int gridDims[3], double gridSpacing[3],
    vtkImageData* depthMap, vtkMatrix3x3 *depthMapMatrixK, vtkMatrix4x4 *depthMapMatrixTR,
    vtkDoubleArray* outScalar)
{
  vtkIdType voxelsNb = outScalar->GetNumberOfTuples();

  // get depth scalars
  vtkDoubleArray* depths = vtkDoubleArray::SafeDownCast(depthMap->GetPointData()->GetArray("Depths"));
  if (!depths)
    {
    // todo error message
    std::cout << "Bad depths." << std::endl;
    return 0;
    }

  // create transforms from matrices
  vtkNew<vtkTransform> transformGridToRealCoords;
  transformGridToRealCoords->SetMatrix(gridMatrix);
  vtkNew<vtkTransform> transformSceneToCamera;
  transformSceneToCamera->SetMatrix(depthMapMatrixTR);
  // change the matrix 3x3 into matrix 4x4 to be compatible with vtkTransform
  vtkNew<vtkMatrix4x4> depthMapMatrixK4x4;
  depthMapMatrixK4x4->Identity();
  for (int i = 0; i < 3; i++)
    {
    for (int j = 0; j < 3; j++)
      {
      depthMapMatrixK4x4->SetElement(i, j, depthMapMatrixK->GetElement(i, j));
      }
    }
  vtkNew<vtkTransform> transformCameraToDepthMap;
  transformCameraToDepthMap->SetMatrix(depthMapMatrixK4x4.Get());

  for (vtkIdType i_vox = 0; i_vox < voxelsNb; i_vox++)
    {
    vtkIdType ijkVox[3];
    ijkVox[0] = i_vox % (gridDims[0] - 1);
    ijkVox[1] = (i_vox / (gridDims[0] - 1)) % (gridDims[1] - 1);
    ijkVox[2] = i_vox / ((gridDims[0] - 1) * (gridDims[1] - 1));

    // voxel center
    double voxCenterTemp[3];
    for (int i = 0; i < 3; i++)
      {
      voxCenterTemp[i] = gridOrig[i] + ((double)ijkVox[i] + 0.5) * gridSpacing[i];
      }
    double voxCenter[3];
    transformGridToRealCoords->TransformPoint(voxCenterTemp, voxCenter);

    // voxel center in camera coords
    double voxCameraCoords[3];
    transformSceneToCamera->TransformPoint(voxCenter, voxCameraCoords);

    // compute distance between voxel and camera
    double distanceVoxCam = vtkMath::Norm(voxCameraCoords);

    // voxel center in depth map homogeneous coords
    double voxDepthMapCoordsHomo[3];
    transformCameraToDepthMap->TransformVector(voxCameraCoords, voxDepthMapCoordsHomo);

    // voxel center in depth map coords
    double voxDepthMapCoords[2];
    voxDepthMapCoords[0] = voxDepthMapCoordsHomo[0] / voxDepthMapCoordsHomo[2];
    voxDepthMapCoords[1] = voxDepthMapCoordsHomo[1] / voxDepthMapCoordsHomo[2];

    // compute depth from depth map
    int ijk[3];
    ijk[0] = round(voxDepthMapCoords[0]);
    ijk[1] = round(voxDepthMapCoords[1]);
    ijk[2] = 0;
    int dim[3];
    depthMap->GetDimensions(dim);
    if (ijk[0] < 0 || ijk[0] > dim[0] - 1 || ijk[1] < 0 || ijk[1] > dim[1] - 1)
      {
      continue;
      }
    vtkIdType id = vtkStructuredData::ComputePointId(dim, ijk);
    if (0 > id && id >= depthMap->GetNumberOfPoints())
      {
      // todo error message
      std::cerr << "Bad conversion from ijk to id." << std::endl;
      continue;
      }
    double depth = depths->GetValue(id);

    // compute new val
    // todo replace by class function
    double currentVal = outScalar->GetValue(i_vox);
    double shift;
    this->RayPotential(distanceVoxCam, depth, shift);
    outScalar->SetValue(i_vox, currentVal + shift);
    }

  return 1;
}

//----------------------------------------------------------------------------
void vtkCudaReconstructionFilter::RayPotential(double realDistance,
                                                double depthMapDistance,
                                                double& shift)
{
  double diff = realDistance - depthMapDistance;

  shift = (this->RayPotentialThickness / this->RayPotentialRho) * diff;
  if (shift > this->RayPotentialRho)
    shift = this->RayPotentialRho;
  if (shift < -this->RayPotentialRho)
    shift = -this->RayPotentialRho;
}


//----------------------------------------------------------------------------
int vtkCudaReconstructionFilter::RequestInformation(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
  // get the info objects
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
               inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT()),
               6);

  return 1;
}

//----------------------------------------------------------------------------
int vtkCudaReconstructionFilter::RequestUpdateExtent(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
  return 1;
}

//----------------------------------------------------------------------------
void vtkCudaReconstructionFilter::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  //os << indent << "Depth Map: " << this->DepthMap << "\n";
}
