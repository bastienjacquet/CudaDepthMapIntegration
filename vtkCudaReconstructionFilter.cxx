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
#include <vtksys/SystemTools.hxx>
#include "vtkTransform.h"
#include "vtkXMLImageDataReader.h"

#include <cmath>
#include <vector>
#include <time.h>
#include <string>
#include <sstream>

vtkStandardNewMacro(vtkCudaReconstructionFilter);
vtkSetObjectImplementationMacro(vtkCudaReconstructionFilter, GridMatrix, vtkMatrix4x4);


void CudaInitialize(vtkMatrix4x4* i_gridMatrix, int h_gridDims[3],
  double h_gridOrig[3], double h_gridSpacing[3], double h_rayPThick, double h_rayPRho,
  double h_rayPEta, double h_rayPDelta, int h_depthMapDim[2]);

template <typename TVolumetric>
bool ProcessDepthMap(std::vector<std::string> vtiList,std::vector<std::string> krtdList,
  double thresholdBestCost, vtkDoubleArray* io_scalar);

//----------------------------------------------------------------------------
vtkCudaReconstructionFilter::vtkCudaReconstructionFilter()
{
  this->SetNumberOfInputPorts(1);
  this->GridMatrix = 0;
  this->UseCuda = false;
  this->RayPotentialRho = 0;
  this->RayPotentialThickness = 0;
  this->RayPotentialDelta = 0;
  this->RayPotentialEta = 0;
  this->ThresholdBestCost = 0;
  this->FilePathKRTD = 0;
  this->FilePathVTI = 0;
}

//----------------------------------------------------------------------------
vtkCudaReconstructionFilter::~vtkCudaReconstructionFilter()
{
  this->DataList.clear();
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
  this->ExecutionTime = -1;
  clock_t start = clock();

  // get the info objects
  vtkInformation *inGridInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *outGridInfo = outputVector->GetInformationObject(0);

  // get the input and output
  vtkImageData *inGrid = vtkImageData::SafeDownCast(
    inGridInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData *outGrid = vtkImageData::SafeDownCast(
    outGridInfo->Get(vtkDataObject::DATA_OBJECT()));

  if (this->FilePathKRTD == 0 || this->FilePathVTI == 0)
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
    // TODO
    //for (int i = 0; i < this->DataList.size(); i++)
    //  {
    //  ReconstructionData* currentData = this->DataList[i];
    //  vtkCudaReconstructionFilter::ComputeWithoutCuda(
    //    this->GridMatrix, gridOrig, gridDims, gridSpacing,
    //    currentData->GetDepthMap(), currentData->Get3MatrixK(), currentData->GetMatrixTR(),
    //    outScalar.Get());
    //  }
    }
  else
    {
    // Check if all variables used in cuda are set
    if (this->RayPotentialRho == 0 && this->RayPotentialThickness == 0)
      {
      std::cerr << "Error : Ray potential Rho or Thickness or both have not been set" << std::endl;
      return 0;
      }

    this->ComputeWithCuda(gridDims, gridOrig, gridSpacing, outScalar.Get());

    }

  clock_t end = clock();
  this->ExecutionTime = (double)(end - start) / CLOCKS_PER_SEC;

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
int vtkCudaReconstructionFilter::ComputeWithCuda(int gridDims[3], double gridOrig[3],
  double gridSpacing[3], vtkDoubleArray* outScalar)
{
  std::vector<std::string> vtiList = vtkCudaReconstructionFilter::ExtractAllFilePath(this->FilePathVTI);
  std::vector<std::string> krtdList = vtkCudaReconstructionFilter::ExtractAllFilePath(this->FilePathKRTD);

  if (vtiList.size() == 0 || krtdList.size() < vtiList.size())
    {
    std::cerr << "Error : There is no enough vti files, please check your vtiList.txt and krtdList.txt" << std::endl;
    return -1;
    }

  ReconstructionData data0(vtiList[0], krtdList[0]);
  int* depthMapGrid = data0.GetDepthMap()->GetDimensions();

  // Initialize Cuda constant
  CudaInitialize(this->GridMatrix, gridDims, gridOrig, gridSpacing,
    this->RayPotentialThickness, this->RayPotentialRho,
    this->RayPotentialEta, this->RayPotentialDelta, depthMapGrid);

  bool result = ProcessDepthMap<double>(vtiList, krtdList, this->ThresholdBestCost,
                                outScalar);

  return 0;
}

//----------------------------------------------------------------------------
void vtkCudaReconstructionFilter::RayPotential(double realDistance,
                                                double depthMapDistance,
                                                double& shift)
{
  double diff = realDistance - depthMapDistance;

  double absolute = abs(diff);
  int sign = diff / absolute;

  if (absolute > this->RayPotentialDelta)
    shift = diff > 0 ? 0 : -this->RayPotentialEta;
  else if (absolute > this->RayPotentialThickness)
    shift = this->RayPotentialRho*sign;
  else
    shift = (this->RayPotentialRho / this->RayPotentialThickness)* diff;
}

//----------------------------------------------------------------------------
std::vector<std::string> vtkCudaReconstructionFilter::ExtractAllFilePath(const char* globalPath)
{
  std::vector<std::string> pathList;

  // Open file which contains the list of all file
  std::ifstream container(globalPath);
  if (!container.is_open())
    {
    std::cerr << "Unable to open : " << globalPath << std::endl;
    return pathList;
    }

  // Extract path of globalPath from globalPath
  std::string directoryPath = vtksys::SystemTools::GetFilenamePath(std::string(globalPath));
  // Get current working directory
  if (directoryPath == "")
  {
    directoryPath = vtksys::SystemTools::GetCurrentWorkingDirectory();
  }

  std::string path;
  while (!container.eof())
  {
    std::getline(container, path);
    // only get the file name, not the whole path
    std::vector <std::string> elems;
    vtkCudaReconstructionFilter::SplitString(path, ' ', elems);

    // check if there are an empty line
    if (elems.size() == 0)
    {
      continue;
    }

    // Create the real data path to access depth map file
    pathList.push_back(directoryPath + "/" + elems[elems.size() - 1]);
  }

  return pathList;
}

//----------------------------------------------------------------------------
void  vtkCudaReconstructionFilter::SplitString( const std::string &s, char delim,
                                                std::vector<std::string> &elems)
{
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim))
  {
    elems.push_back(item);
  }
}

//----------------------------------------------------------------------------
bool vtkCudaReconstructionFilter::ReadKrtdFile(std::string filename,
                                               vtkMatrix3x3* matrixK,
                                               vtkMatrix4x4* matrixTR)
{
  // Open the file
  std::ifstream file(filename.c_str());
  if (!file.is_open())
  {
    std::cerr << "Unable to open krtd file : " << filename << std::endl;
    return false;
  }

  std::string line;

  // Get matrix K
  for (int i = 0; i < 3; i++)
  {
    getline(file, line);
    std::istringstream iss(line);

    for (int j = 0; j < 3; j++)
    {
      double value;
      iss >> value;
      matrixK->SetElement(i, j, value);
    }
  }

  getline(file, line);

  // Get matrix R
  for (int i = 0; i < 3; i++)
  {
    getline(file, line);
    std::istringstream iss(line);

    for (int j = 0; j < 3; j++)
    {
      double value;
      iss >> value;
      matrixTR->SetElement(i, j, value);
    }
  }

  getline(file, line);

  // Get matrix T
  getline(file, line);
  std::istringstream iss(line);
  for (int i = 0; i < 3; i++)
  {
    double value;
    iss >> value;
    matrixTR->SetElement(i, 3, value);
  }

  // Finalize matrix TR
  for (int j = 0; j < 4; j++)
  {
    matrixTR->SetElement(3, j, 0);
  }
  matrixTR->SetElement(3, 3, 1);

  return true;
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
