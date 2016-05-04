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

#include "ReconstructionData.h"

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

#include "Helper.h"

#include <cmath>
#include <string>
#include <sstream>
#include <time.h>
#include <vector>

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
  this->FilePathKRTD = 0;
  this->FilePathVTI = 0;
  this->GridMatrix = 0;
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

  // Check if all variables used in cuda are set
  if (this->RayPotentialRho == 0 && this->RayPotentialThickness == 0)
    {
    std::cerr << "Error : Ray potential Rho or Thickness or both have not been set" << std::endl;
    return 0;
    }

  this->Compute(gridDims, gridOrig, gridSpacing, outScalar.Get());


  clock_t end = clock();
  this->ExecutionTime = (double)(end - start) / CLOCKS_PER_SEC;

  return 1;
}


//----------------------------------------------------------------------------
int vtkCudaReconstructionFilter::Compute(int gridDims[3], double gridOrig[3],
  double gridSpacing[3], vtkDoubleArray* outScalar)
{
  std::vector<std::string> vtiList = help::ExtractAllFilePath(this->FilePathVTI);
  std::vector<std::string> krtdList = help::ExtractAllFilePath(this->FilePathKRTD);

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
