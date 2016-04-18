#include "ReconstructionData.h"

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

vtkMatrix3x3* ReconstructionData::GetMatrixK()
{
  return this->matrixK;
}

vtkMatrix4x4* ReconstructionData::GetMatrixTR()
{
  return this->matrixTR;
}

void ReconstructionData::SetDepthMap(vtkImageData* data)
{
  this->depthMap = data;
}

void ReconstructionData::SetMatrixK(vtkMatrix3x3* matrix)
{
  this->matrixK = matrix;
}

void ReconstructionData::SetMatrixTR(vtkMatrix4x4* matrix)
{
  this->matrixTR = matrix;
}

