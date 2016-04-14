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

vtkStandardNewMacro(vtkCudaReconstructionFilter);

vtkSetObjectImplementationMacro(vtkCudaReconstructionFilter, DepthMap, vtkImageData);
vtkSetObjectImplementationMacro(vtkCudaReconstructionFilter, DepthMapMatrixK, vtkMatrix3x3);
vtkSetObjectImplementationMacro(vtkCudaReconstructionFilter, DepthMapMatrixTR, vtkMatrix4x4);

//----------------------------------------------------------------------------
vtkCudaReconstructionFilter::vtkCudaReconstructionFilter()
{
  this->SetNumberOfInputPorts(1);
  this->DepthMap = 0;
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
  if (this->DepthMap)
    {
    this->DepthMap->Delete();
    }
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

  if (!this->DepthMap || !this->DepthMapMatrixK || !this->DepthMapMatrixTR)
    {
    std::cout << "Bad input." << std::endl;
    return 0;
    }

  // get depth scalars
  vtkDoubleArray* depths = vtkDoubleArray::SafeDownCast(this->DepthMap->GetPointData()->GetArray("Depths"));
  if (!depths)
    {
    std::cout << "Bad depths." << std::endl;
    return 0;
    }

  vtkIdType voxelsNb = inGrid->GetNumberOfCells();

  std::cout << "Initialize output." << std::endl;

  // copy input grid to output grid
  outGrid->ShallowCopy(inGrid);

  // add output scalar
  vtkNew<vtkDoubleArray> outScalar;
  outScalar->SetName("reconstruction_scalar");
  outScalar->SetNumberOfComponents(1);
  outScalar->SetNumberOfTuples(voxelsNb);
  outScalar->FillComponent(0, 0);
  outGrid->GetCellData()->AddArray(outScalar.Get());

  std::cout << "Create matrices." << std::endl;

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

  this->DepthMapMatrixTR->PrintSelf(std::cout, vtkIndent());
  depthMapMatrixK->PrintSelf(std::cout, vtkIndent());

  std::cout << "Fill output." << std::endl;

  for (vtkIdType i_vox = 0; i_vox < voxelsNb; i_vox++)
    {
    // voxel center
    // todo transform input image data into structured grid using this->GridVecX Y Z
    double pcoords[3];
    double voxCenter[3];
    vtkCell* vox = inGrid->GetCell(i_vox);
    int subId = vox->GetParametricCenter(pcoords);
    double *weights = new double [inGrid->GetMaxCellSize()];
    vox->EvaluateLocation(subId, pcoords, voxCenter, weights);

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

    // compute distance from depth map
    // todo improve with interpolation
    int ijk[3];
    ijk[0] = round(voxDepthMapCoords[0] / 1900 * 479);
    ijk[1] = round(voxDepthMapCoords[1] / 1000 * 269);
    ijk[2] = 0;

    int dim[3];
    this->DepthMap->GetDimensions(dim);

    if (ijk[0] >= 0 && ijk[0] < dim[0] && ijk[1] >= 0 && ijk[1] < dim[1])
      {

      vtkIdType id = vtkStructuredData::ComputePointId(dim, ijk);

      if (0 > id && id >= this->DepthMap->GetNumberOfPoints())
        {
        std::cout << "pb x" << std::endl;
        }

      double depth = depths->GetValue(id);
      double prevVal = outScalar->GetValue(i_vox);

      // compute new val
      // todo replace by class function
      double newVal = prevVal;
      if (std::abs(distanceVoxCam - depth) != 0)
        {
        newVal += 1 / std::abs(distanceVoxCam - depth);
        }
      else
        {
        newVal += 10;
        }

      //std::cout << "depth " << depth << std::endl;
      //std::cout << "val " << prevVal << " -> " << newVal << std::endl;

      outScalar->SetValue(i_vox, newVal);
      }
    /*
      std::cout << "voxCenter " << voxCenter[0] << " " << voxCenter[1] << " " << voxCenter[2] << std::endl;
      std::cout << "voxCameraCoords " << voxCameraCoords[0] << " " << voxCameraCoords[1] << " " << voxCameraCoords[2] << std::endl;
      std::cout << "voxDepthMapCoordsHomo " << voxDepthMapCoordsHomo[0] << " " << voxDepthMapCoordsHomo[1] << " " << voxDepthMapCoordsHomo[2] << " " << voxDepthMapCoordsHomo[3] << std::endl;
      std::cout << "voxDepthMapCoords " << voxDepthMapCoords[0] << " " << voxDepthMapCoords[1] << std::endl;
      std::cout << "ijk " << ijk[0] << " " << ijk[1] << " " << ijk[2] << std::endl;
      std::cout << "id " << id << std::endl;
      std::cout << "distanceVoxCam " << distanceVoxCam << std::endl;
      std::cout << std::endl;
     * */
    }

  return 1;
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

  os << indent << "Depth Map: " << this->DepthMap << "\n";
}
