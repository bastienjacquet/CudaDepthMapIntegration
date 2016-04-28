#include "ReconstructionData.h"

// VTK includes
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "vtkXMLImageDataReader.h"

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
  this->ReadKRTD(matrixPath, K, this->matrixTR);
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

bool ReconstructionData::ReadKRTD(std::string path, vtkMatrix3x3* matrixK,
                                  vtkMatrix4x4* matrixTR)
{
  // Open the file
  std::ifstream file(path.c_str());
  if (!file.is_open())
  {
    std::cerr << "Unable to open krtd file : " << path << std::endl;
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
