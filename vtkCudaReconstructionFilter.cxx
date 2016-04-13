#include "vtkCudaReconstructionFilter.h"

#include "vtkDataSet.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMatrix4x4.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include <vector>

vtkStandardNewMacro(vtkCudaReconstructionFilter);

vtkSetObjectImplementationMacro(vtkCudaReconstructionFilter, DepthMapMatrix, vtkMatrix4x4);

//----------------------------------------------------------------------------
vtkCudaReconstructionFilter::vtkCudaReconstructionFilter()
{
  this->SetNumberOfInputPorts(2);
  this->DepthMapMatrix = 0;
}

//----------------------------------------------------------------------------
vtkCudaReconstructionFilter::~vtkCudaReconstructionFilter()
{
}

//----------------------------------------------------------------------------
void vtkCudaReconstructionFilter::SetDepthMap(vtkDataObject *input)
{
  this->SetInputData(1, input);
}

//----------------------------------------------------------------------------
vtkDataObject *vtkCudaReconstructionFilter::GetDepthMap()
{
  if (this->GetNumberOfInputConnections(1) < 1)
    {
    return NULL;
    }

  return this->GetExecutive()->GetInputData(1, 0);
}

//----------------------------------------------------------------------------
int vtkCudaReconstructionFilter::RequestData(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
  // get the info objects
  vtkInformation *inGridInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *depthMapInfo = inputVector[1]->GetInformationObject(0);
  vtkInformation *outGridInfo = outputVector->GetInformationObject(0);

  // get the input and output
  vtkDataSet *inGrid = vtkDataSet::SafeDownCast(
    inGridInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkDataSet *depthMap = vtkDataSet::SafeDownCast(
    depthMapInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkDataSet *outGrid = vtkDataSet::SafeDownCast(
    outGridInfo->Get(vtkDataObject::DATA_OBJECT()));

  if (!depthMap || !this->DepthMapMatrix)
    {
    std::cout << "Bad input." << std::endl;
    return 0;
    }

  outGrid->ShallowCopy(inGrid);

  return 1;
}

//----------------------------------------------------------------------------
int vtkCudaReconstructionFilter::RequestInformation(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
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

  vtkDataObject *depthMap = this->GetDepthMap();
  os << indent << "Depth Map: " << depthMap << "\n";
}
