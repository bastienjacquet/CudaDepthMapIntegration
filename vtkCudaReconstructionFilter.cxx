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
vtkSetObjectImplementationMacro(vtkCudaReconstructionFilter, GridMatrix, vtkMatrix4x4);

int cuda_reconstruction(
    double h_gridMatrix[16], double h_gridOrig[3], int h_gridDims[3], double h_gridSpacing[3],
    int h_depthMapDims[3], double* h_depths, double h_depthMapMatrixK[16], double h_depthMapMatrixTR[16],
    double* h_outScalar);

int reconstruction(std::vector<ReconstructionData*> i_dataList, vtkMatrix4x4* i_gridMatrix, 
  int h_gridDims[3], double h_gridOrig[3], double h_gridSpacing[3], vtkDoubleArray* h_outScalar);

//----------------------------------------------------------------------------
vtkCudaReconstructionFilter::vtkCudaReconstructionFilter()
{
  this->SetNumberOfInputPorts(1);
  this->GridMatrix = 0;
  this->useCuda = false;
}

//----------------------------------------------------------------------------
vtkCudaReconstructionFilter::~vtkCudaReconstructionFilter()
{
  this->DataList.clear();
}

void vtkCudaReconstructionFilter::SetDataList(std::vector<ReconstructionData*> list)
{
  this->DataList = list;
}

void vtkCudaReconstructionFilter::UseCudaOn()
{
  this->useCuda = true;
}

void vtkCudaReconstructionFilter::UseCudaOff()
{
  this->useCuda = false;
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

  if (this->DataList.size() == 0)
    {
    // todo error message
    std::cerr << "Error, input not set." << std::endl;
    return 0;
    }

  // get grid info
  double gridOrig[3];
  inGrid->GetOrigin(gridOrig);
  int gridDims[3];
  inGrid->GetDimensions(gridDims);
  double gridSpacing[3];
  inGrid->GetSpacing(gridSpacing);

  // todo remove
  std::cout << "Initialize output." << std::endl;


  // initialize output
  vtkNew<vtkDoubleArray> outScalar;
  outScalar->SetName("reconstruction_scalar");
  outScalar->SetNumberOfComponents(1);
  outScalar->SetNumberOfTuples(inGrid->GetNumberOfCells());
  outScalar->FillComponent(0, 0);
  outGrid->ShallowCopy(inGrid);
  outGrid->GetCellData()->AddArray(outScalar.Get());

  // computation
  if (!this->useCuda)
    {
      for (int i = 0; i < this->DataList.size(); i++)
        {
        ReconstructionData* currentData = this->DataList[i];
        vtkCudaReconstructionFilter::ComputeWithoutCuda(
          this->GridMatrix, gridOrig, gridDims, gridSpacing,
          currentData->GetDepthMap(), currentData->Get3MatrixK(), currentData->GetMatrixTR(),
          outScalar.Get());
        }
    }
  else
    {
     reconstruction(this->DataList, this->GridMatrix, gridDims, gridOrig, gridSpacing, outScalar.Get());
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

  // todo remove
  std::cout << "Create matrices." << std::endl;

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

  // todo remove
  std::cout << "Fill output." << std::endl;

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
    // todo improve with interpolation
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
      std::cout << "Bad conversion from ijk to id." << std::endl;
      continue;
      }
    double depth = depths->GetValue(id);

    // compute new val
    // todo replace by class function
    double val = outScalar->GetValue(i_vox);
    vtkCudaReconstructionFilter::FunctionCumul(distanceVoxCam - depth, val);
    outScalar->SetValue(i_vox, val);
    }

  return 1;
}

//----------------------------------------------------------------------------
void vtkCudaReconstructionFilter::FunctionCumul(double diff, double& val)
{
  //if (std::abs(diff) != 0)
    {
      val += 10 - 0.5*std::abs(diff);
    }
  //else
  //  {
  //  val += 10;
  //  }
  //if (val > 100)
  //  {
  //  val = 100;
  //  }
  if (val < 0)
    val = 0;
}

//----------------------------------------------------------------------------
int vtkCudaReconstructionFilter::ComputeWithCuda(
    vtkMatrix4x4 *gridMatrix, double gridOrig[3], int gridDims[3], double gridSpacing[3],
    vtkImageData* depthMap, vtkDoubleArray* outScalar)
{

  int res = reconstruction(this->DataList, gridMatrix, gridDims, gridOrig, gridSpacing, outScalar);

  //// todo convert gridMatrix, depthMapMatrixK, depthMapMatrixTR, outScalar into double*
  //double copy_gridMatrix[16];
  //double copy_depthMapMatrixK[16];
  //double copy_depthMapMatrixTR[16];
  //double copy_outScalar[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

  //// todo convert depthMap into double* + dims
  //int depthMapDims[3];
  //// todo this line is already done in ComputeWithoutCuda, must factorize it
  //vtkDoubleArray* depths = vtkDoubleArray::SafeDownCast(depthMap->GetPointData()->GetArray("Depths"));
  //if (!depths)
  //  {
  //  // todo error message
  //    std::cout << "Bad depths." << std::endl;
  //  return 0;
  //  }
  //double* copy_depths;
  //copy_depths = new double[3];
  //// call host function in cuda file
  //cuda_reconstruction(copy_gridMatrix, gridOrig, gridDims, gridSpacing,
  //                    depthMapDims, copy_depths, copy_depthMapMatrixK, copy_depthMapMatrixTR,
  //                    copy_outScalar);

  //// todo fill outScalar with copy_outScalar, maybe can be done directly
  //std::cout << copy_outScalar[0] << std::endl;
  //std::cout << copy_outScalar[1] << std::endl;
  //std::cout << copy_outScalar[2] << std::endl;
  //std::cout << copy_outScalar[3] << std::endl;
  //std::cout << copy_outScalar[4] << std::endl;

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

  //os << indent << "Depth Map: " << this->DepthMap << "\n";
}
