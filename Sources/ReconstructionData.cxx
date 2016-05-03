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

#include "Helper.h"
#include "ReconstructionData.h"

#include <sstream>

// VTK includes
#include "vtkDoubleArray.h"
#include "vtkUnsignedCharArray.h"
#include "vtkPointData.h"
#include "vtkXMLImageDataReader.h"


ReconstructionData::ReconstructionData()
{
  this->DepthMap = nullptr;
  this->MatrixK = nullptr;
  this->MatrixTR = nullptr;
}

ReconstructionData::ReconstructionData(std::string depthPath,
                                       std::string matrixPath)
{
  // Read DEPTH MAP an fill this->DepthMap
  this->ReadDepthMap(depthPath);

  // Read KRTD FILE
  vtkMatrix3x3* K = vtkMatrix3x3::New();
  this->MatrixTR = vtkMatrix4x4::New();
  help::ReadKrtdFile(matrixPath, K, this->MatrixTR);
  // Set matrix K to  create matrix4x4 for K
  this->SetMatrixK(K);

}

ReconstructionData::~ReconstructionData()
{
  if (this->DepthMap)
    this->DepthMap->Delete();
  if (this->MatrixK)
    this->MatrixK->Delete();
  if (this->MatrixTR)
    this->MatrixTR->Delete();
  if (this->Matrix4K)
    this->Matrix4K->Delete();
}

void ReconstructionData::GetColorValue(int* pixelPosition, double rgb[3])
{
  vtkUnsignedCharArray* color =
    vtkUnsignedCharArray::SafeDownCast(this->DepthMap->GetPointData()->GetArray("Color"));

  if (color == nullptr)
    {
    std::cerr << "Error, no 'Color' array exists" << std::endl;
    return;
    }

  int* depthDims = this->DepthMap->GetDimensions();

  int pix[3];
  pix[0] = pixelPosition[0];
  pix[1] = depthDims[1] - 1 - pixelPosition[1];
  pix[2] = 0;

  int id = this->DepthMap->ComputePointId(pix);
  double* temp = color->GetTuple3(id);
  for (size_t i = 0; i < 3; i++)
  {
    rgb[i] = temp[i];
  }
}

vtkImageData* ReconstructionData::GetDepthMap()
{
  return this->DepthMap;
}

vtkMatrix3x3* ReconstructionData::Get3MatrixK()
{
  return this->MatrixK;
}

vtkMatrix4x4* ReconstructionData::Get4MatrixK()
{
  return this->Matrix4K;
}

vtkMatrix4x4* ReconstructionData::GetMatrixTR()
{
  return this->MatrixTR;
}

void ReconstructionData::ApplyDepthThresholdFilter(double thresholdBestCost)
{
  if (this->DepthMap == nullptr)
    return;

  vtkDoubleArray* depths =
    vtkDoubleArray::SafeDownCast(this->DepthMap->GetPointData()->GetArray("Depths"));
  vtkDoubleArray* bestCost =
    vtkDoubleArray::SafeDownCast(this->DepthMap->GetPointData()->GetArray("Best Cost Values"));

  if (depths == nullptr)
    {
    std::cerr << "Error during threshold, depths is empty" << std::endl;
    return;
    }

  int nbTuples = depths->GetNumberOfTuples();

  if (bestCost->GetNumberOfTuples() != nbTuples)
    return;

  for (int i = 0; i < nbTuples; i++)
    {
    double v_bestCost = bestCost->GetTuple1(i);
    if (v_bestCost > thresholdBestCost)
      {
      depths->SetTuple1(i, -1);
      }
    }
}

void ReconstructionData::SetDepthMap(vtkImageData* data)
{
  this->DepthMap = data;
}

void ReconstructionData::SetMatrixK(vtkMatrix3x3* matrix)
{
  this->MatrixK = matrix;
  this->Matrix4K = vtkMatrix4x4::New();
  this->Matrix4K->Identity();
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      this->Matrix4K->SetElement(i, j, this->MatrixK->GetElement(i, j));
    }
  }
}

void ReconstructionData::SetMatrixTR(vtkMatrix4x4* matrix)
{
  this->MatrixTR = matrix;
}

void ReconstructionData::ReadDepthMap(std::string path)
{
  vtkXMLImageDataReader* depthMapReader = vtkXMLImageDataReader::New();
  depthMapReader->SetFileName(path.c_str());
  depthMapReader->Update();
  this->DepthMap = depthMapReader->GetOutput();

}
