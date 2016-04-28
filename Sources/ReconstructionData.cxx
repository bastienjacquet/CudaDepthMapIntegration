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

#include "ReconstructionData.h"

// VTK includes
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "vtkXMLImageDataReader.h"

#include "Helper.h"

#include <sstream>

ReconstructionData::ReconstructionData()
{
  this->depthMap = nullptr;
  this->matrixK = nullptr;
  this->matrixTR = nullptr;
}

ReconstructionData::ReconstructionData(std::string depthPath,
                                       std::string matrixPath)
{
  // Read DEPTH MAP an fill this->depthMap
  this->ReadDepthMap(depthPath);

  // Read KRTD FILE
  vtkMatrix3x3* K = vtkMatrix3x3::New();
  this->matrixTR = vtkMatrix4x4::New();
  help::ReadKrtdFile(matrixPath, K, this->matrixTR);
  // Set matrix K to  create matrix4x4 for K
  this->SetMatrixK(K);

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

void ReconstructionData::ApplyDepthThresholdFilter(double thresholdBestCost)
{
  if (this->depthMap == nullptr)
    return;

  vtkDoubleArray* depths =
    vtkDoubleArray::SafeDownCast(this->depthMap->GetPointData()->GetArray("Depths"));
  vtkDoubleArray* bestCost =
    vtkDoubleArray::SafeDownCast(this->depthMap->GetPointData()->GetArray("Best Cost Values"));

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

void ReconstructionData::ReadDepthMap(std::string path)
{
  vtkXMLImageDataReader* depthMapReader = vtkXMLImageDataReader::New();
  depthMapReader->SetFileName(path.c_str());
  depthMapReader->Update();
  this->depthMap = depthMapReader->GetOutput();

}
