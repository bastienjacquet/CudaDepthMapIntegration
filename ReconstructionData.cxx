#include "ReconstructionData.h"

// VTK includes
#include "vtkDoubleArray.h"
#include "vtkPointData.h"

ReconstructionData::ReconstructionData()
{
  this->depthMap = nullptr;
  this->matrixK = nullptr;
  this->matrixTR = nullptr;
}

ReconstructionData::~ReconstructionData()
{
  if (this->depthMap)
    this->depthMap->Delete();
  if (this->matrixK)
    this->matrixK->Delete();
  if (this->matrixTR)
    this->matrixTR->Delete();
}

vtkImageData* ReconstructionData::GetDepthMap()
{
  return this->depthMap;
}

vtkMatrix3x3* ReconstructionData::Get3MatrixK()
{
  return this->matrixK;
}

vtkMatrix4x4* ReconstructionData::Get4MatrixK()
{
  return this->matrix4K;
}

vtkMatrix4x4* ReconstructionData::GetMatrixTR()
{
  return this->matrixTR;
}

void ReconstructionData::ApplyDepthThresholdFilter(double thresholdBestCost,
                                                   double thresholdUniqueness)
{
  if (this->depthMap == nullptr)
    return;

  vtkDoubleArray* depths =
    vtkDoubleArray::SafeDownCast(this->depthMap->GetPointData()->GetArray("Depths"));
  vtkDoubleArray* bestCost =
    vtkDoubleArray::SafeDownCast(this->depthMap->GetPointData()->GetArray("Best Cost Values"));
  vtkDoubleArray* uniqueness =
    vtkDoubleArray::SafeDownCast(this->depthMap->GetPointData()->GetArray("Uniqueness Ratios"));

  int nbTuples = depths->GetNumberOfTuples();

  if (bestCost->GetNumberOfTuples() != nbTuples &&
    uniqueness->GetNumberOfTuples() != nbTuples)
    return;

  for (int i = 0; i < nbTuples; i++)
  {
    double v_bestCost = bestCost->GetTuple1(i);
    double v_uniqueness = uniqueness->GetTuple1(i);
    if (v_bestCost > thresholdBestCost && v_uniqueness > thresholdUniqueness)
      depths->SetTuple1(i, -1);
  }
}

void ReconstructionData::SetDepthMap(vtkImageData* data)
{
  this->depthMap = data;
}

void ReconstructionData::SetMatrixK(vtkMatrix3x3* matrix)
{
  this->matrixK = matrix;
  this->matrix4K = vtkMatrix4x4::New();
  this->matrix4K->Identity();
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      this->matrix4K->SetElement(i, j, this->matrixK->GetElement(i, j));
    }
  }
}

void ReconstructionData::SetMatrixTR(vtkMatrix4x4* matrix)
{
  this->matrixTR = matrix;
}

