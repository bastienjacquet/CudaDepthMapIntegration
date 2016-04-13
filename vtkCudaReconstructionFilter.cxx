#include "vtkCudaReconstructionFilter.h"

#include "vtkCellData.h"
#include "vtkDataSet.h"
#include "vtkDoubleArray.h"
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

vtkStandardNewMacro(vtkCudaReconstructionFilter);

vtkSetObjectImplementationMacro(vtkCudaReconstructionFilter, DepthMapMatrixK, vtkMatrix3x3);
vtkSetObjectImplementationMacro(vtkCudaReconstructionFilter, DepthMapMatrixTR, vtkMatrix4x4);

//----------------------------------------------------------------------------
vtkCudaReconstructionFilter::vtkCudaReconstructionFilter()
{
  this->SetNumberOfInputPorts(2);
  this->DepthMapMatrixK = 0;
  this->DepthMapMatrixTR = 0;
}

//----------------------------------------------------------------------------
vtkCudaReconstructionFilter::~vtkCudaReconstructionFilter()
{
  if (this->DepthMapMatrixK)
    {
    this->DepthMapMatrixK->Delete();
    }
  if (this->DepthMapMatrixTR)
    {
    this->DepthMapMatrixTR->Delete();
    }
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
  vtkStructuredGrid *depthMap = vtkStructuredGrid::SafeDownCast(
    depthMapInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkDataSet *outGrid = vtkDataSet::SafeDownCast(
    outGridInfo->Get(vtkDataObject::DATA_OBJECT()));

  if (!depthMap || !this->DepthMapMatrixK || !this->DepthMapMatrixTR)
    {
    std::cout << "Bad input." << std::endl;
    return 0;
    }

  // get depth scalars
  vtkDoubleArray* depths = vtkDoubleArray::SafeDownCast(depthMap->GetPointData()->GetArray("Depth"));
  if (!depths)
    {
    std::cout << "Bad depths." << std::endl;
    return 0;
    }

  vtkIdType voxelsNb = inGrid->GetNumberOfCells();

  // copy input grid to output grid
  outGrid->ShallowCopy(inGrid);

  // add output scalar
  vtkNew<vtkDoubleArray> outScalar;
  outScalar->SetName("reconstruction_scalar");
  outScalar->SetNumberOfComponents(1);
  outScalar->SetNumberOfTuples(voxelsNb);
  outGrid->GetCellData()->AddArray(outScalar.Get());

  // create transforms from matrices
  vtkNew<vtkTransform> transformSceneToCamera;
  transformSceneToCamera->SetMatrix(this->DepthMapMatrixTR);
  // change the matrix 4x4 into matrix 3x3 to be compatible with vtkTransform
  vtkNew<vtkMatrix4x4> depthMapMatrixK;
  depthMapMatrixK->Identity();
  for (int i = 0; i < 3; i++)
    {
    for (int j = 0; j < 3; j++)
      {
      depthMapMatrixK->SetElement(i, j, this->DepthMapMatrixK->GetElement(i, j));
      }
    }
  vtkNew<vtkTransform> transformCameraToDepthMap;
  transformCameraToDepthMap->SetMatrix(depthMapMatrixK.Get());

  for (vtkIdType i_vox = 0; i_vox < voxelsNb; i_vox++)
    {
    // voxel bounds
    double voxBounds[3];
    inGrid->GetCellBounds(i_vox, voxBounds);

    // voxel center
    double voxCenter[4];
    for (int i = 0; i < 3; i++)
      {
      voxCenter[i] = (voxBounds[2 * i] + voxBounds[2 * i + 1]) / 2;
      }
    voxCenter[3] = 1;

    // voxel center in camera coords
    double voxCameraCoords[3];
    transformSceneToCamera->TransformPoint(voxCenter, voxCameraCoords);

    // compute distance between voxel and camera
    double distanceVoxCam = vtkMath::Norm(voxCameraCoords);

    // voxel center in depth map homogeneous coords
    double voxDepthMapCoordsHomo[3];
    transformCameraToDepthMap->TransformPoint(voxCameraCoords, voxDepthMapCoordsHomo);

    // voxel center in depth map coords
    double voxDepthMapCoords[2];
    voxDepthMapCoords[0] = voxDepthMapCoordsHomo[0] / voxDepthMapCoordsHomo[2];
    voxDepthMapCoords[1] = voxDepthMapCoordsHomo[1] / voxDepthMapCoordsHomo[2];

    // compute distance from depth map
    // todo improve with interpolation
    int ijk[3];
    ijk[0] = round(voxDepthMapCoords[0]);
    ijk[1] = round(voxDepthMapCoords[1]);
    ijk[2] = 0;
    int dim[3];
    depthMap->GetDimensions(dim);
    vtkIdType id = vtkStructuredData::ComputePointId(dim, ijk);
    if (id < depthMap->GetNumberOfPoints())
      {
      double depth = depths->GetValue(id);
      outScalar->SetValue(i_vox, distanceVoxCam - depth);
      }
    }

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
