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

// VTK includes
#include <vtksys/CommandLineArguments.hxx>
#include <vtksys/SystemTools.hxx>
#include "vtkDoubleArray.h"
#include "vtkNew.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkXMLPolyDataReader.h"
#include "vtkXMLPolyDataWriter.h"

#include "Helper.h"
#include "ReconstructionData.h"

#include <iostream>
#include <string>
#include <numeric>

//-----------------------------------------------------------------------------
// READ ARGUMENTS
//-----------------------------------------------------------------------------
std::string g_inputPath = "";
std::string g_outputPath = "";
std::string g_globalKRTDFilePath = "";
std::string g_globalVTIFilePath = "";
bool verbose = false;

//-----------------------------------------------------------------------------
// FUNCTIONS
//-----------------------------------------------------------------------------
bool ReadArguments(int argc, char ** argv);
void ShowInformation(std::string message);
void ShowFilledParameters();


//-----------------------------------------------------------------------------
/* Main function */
int main(int argc, char ** argv)
{
  if (!ReadArguments(argc, argv))
  {
    return EXIT_FAILURE;
  }

  ShowInformation("** Read input...");
  vtkNew<vtkXMLPolyDataReader> reader;
  reader->SetFileName(g_inputPath.c_str());
  reader->Update();


  ShowInformation("** Extract vti and krtd file path...");
  std::vector<std::string> vtiList = help::ExtractAllFilePath(g_globalVTIFilePath.c_str());
  std::vector<std::string> krtdList = help::ExtractAllFilePath(g_globalKRTDFilePath.c_str());
  if (krtdList.size() < vtiList.size())
    {
    std::cerr << "Error, no enough krtd file for each vti file" << std::endl;
    return EXIT_FAILURE;
    }

  int nbDepthMap = (int)vtiList.size();

  ShowInformation("** Read krtd and vti...");
  std::vector<ReconstructionData*> dataList;
  for (int id = 0; id < nbDepthMap; id++)
    {
    std::cout << "\r" << id * 100 / nbDepthMap << " %" << std::flush;
    ReconstructionData* data = new ReconstructionData(vtiList[id], krtdList[id]);
    dataList.push_back(data);
    }

  std::cout << "\r" << "100 %" << std::flush << std::endl << std::endl;
  vtkPolyData* mesh = reader->GetOutput();
  vtkPoints* meshPointList = mesh->GetPoints();
  vtkIdType nbMeshPoint = meshPointList->GetNumberOfPoints();
  //vtkIdType nbMeshPoint = 3;
  int* depthMapDimensions = dataList[0]->GetDepthMapDimensions();

  ShowInformation("** Process coloration for " + std::to_string(nbMeshPoint) + " points ...");


  std::vector<int> pointCount(nbMeshPoint);
  // Contains rgb values
  vtkUnsignedCharArray* meanValues = vtkUnsignedCharArray::New();
  meanValues->SetNumberOfComponents(3);
  meanValues->SetNumberOfTuples(nbMeshPoint);
  meanValues->FillComponent(0, 0);
  meanValues->FillComponent(1, 0);
  meanValues->FillComponent(2, 0);
  meanValues->SetName("meanColoration");

  vtkUnsignedCharArray* medianValues = vtkUnsignedCharArray::New();
  medianValues->SetNumberOfComponents(3);
  medianValues->SetNumberOfTuples(nbMeshPoint);
  medianValues->FillComponent(0, 0);
  medianValues->FillComponent(1, 0);
  medianValues->FillComponent(2, 0);
  medianValues->SetName("medianColoration");

  vtkDoubleArray* projectedDMValue = vtkDoubleArray::New();
  projectedDMValue->SetNumberOfComponents(1);
  projectedDMValue->SetNumberOfTuples(nbMeshPoint);
  projectedDMValue->FillComponent(0, 0);
  projectedDMValue->SetName("NbProjectedDepthMap");

  // Store each rgb value for each depth map
  std::vector<double> list0;
  std::vector<double> list1;
  std::vector<double> list2;

  for (vtkIdType id = 0; id < nbMeshPoint; id++)
    {
    std::cout << "\r" << id * 100 / nbMeshPoint << " %" << std::flush;

    list0.reserve(nbDepthMap);
    list1.reserve(nbDepthMap);
    list2.reserve(nbDepthMap);

    // Get mesh position from id
    double position[3];
    meshPointList->GetPoint(id, position);

    for (int idData = 0; idData < nbDepthMap; idData++)
      {
      ReconstructionData* data = dataList[idData];

      // Transform to pixel index
      int pixelPosition[2];
      help::WorldToDepthMap(data->GetMatrixTR(), data->Get4MatrixK(), position, pixelPosition);
      // Test if pixel is inside depth map
      if (pixelPosition[0] < 0 || pixelPosition[1] < 0 ||
        pixelPosition[0] >= depthMapDimensions[0] ||
        pixelPosition[1] >= depthMapDimensions[1])
        {
        continue;
        }

      double color[3];
      data->GetColorValue(pixelPosition, color);

      list0.push_back(color[0]);
      list1.push_back(color[1]);
      list2.push_back(color[2]);
      }

    // If we get elements
    if (list0.size() != 0)
      {
      double sum0 = std::accumulate(list0.begin(), list0.end(), 0);
      double sum1 = std::accumulate(list1.begin(), list1.end(), 0);
      double sum2 = std::accumulate(list2.begin(), list2.end(), 0);
      double nbVal = (double)list0.size();
      meanValues->SetTuple3(id, sum0 / (double)nbVal, sum1 / (double)nbVal, sum2 / (double)nbVal);
      double median0, median1, median2;
      help::ComputeMedian<double>(list0, median0);
      help::ComputeMedian<double>(list1, median1);
      help::ComputeMedian<double>(list2, median2);
      medianValues->SetTuple3(id, median0, median1, median2);
      projectedDMValue->SetTuple1(id, list0.size());
      }

    list0.clear();
    list1.clear();
    list2.clear();
    }

  mesh->GetPointData()->AddArray(meanValues);
  mesh->GetPointData()->AddArray(medianValues);
  mesh->GetPointData()->AddArray(projectedDMValue);

  std::cout << "\r" << "100 %" << std::flush << std::endl << std::endl;
  ShowInformation("** Write output image");
  vtkNew<vtkXMLPolyDataWriter> writer;
  writer->SetFileName(g_outputPath.c_str());
  writer->SetInputData(mesh);
  writer->Update();

  return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
/* Read input argument and check if they are valid */
bool ReadArguments(int argc, char ** argv)
{
  bool help = false;

  vtksys::CommandLineArguments arg;
  arg.Initialize(argc, argv);
  typedef vtksys::CommandLineArguments argT;

  arg.AddArgument("--input", argT::SPACE_ARGUMENT, &g_inputPath, "(required) Path to a .vtp file");
  arg.AddArgument("--output", argT::SPACE_ARGUMENT, &g_outputPath, "(required) Path of the output file (.vtp)");
  arg.AddArgument("--krtd", argT::SPACE_ARGUMENT, &g_globalKRTDFilePath, "(required) Path to the file which contains all krtd path");
  arg.AddArgument("--vti", argT::SPACE_ARGUMENT, &g_globalVTIFilePath, "(required) Path to the file which contains all vti path");
  arg.AddBooleanArgument("--verbose", &verbose, "(optional) Use to display debug information");
  arg.AddBooleanArgument("--help", &help, "Print help message");

  int result = arg.Parse();
  if (!result || help)
    {
    std::cerr << arg.GetHelp();
    return false;
    }

  // Check arguments
  if (g_inputPath == "" || g_outputPath == "" || g_globalKRTDFilePath == "" || g_globalVTIFilePath == "")
    {
    std::cerr << "Missing arguments..." << std::endl;
    std::cerr << arg.GetHelp();
    return false;
    }

  return true;
}


//-----------------------------------------------------------------------------
/* Show information on console if we are on verbose mode */
void ShowInformation(std::string information)
{
  if (verbose)
  {
    std::cout << information << "\n" << std::endl;
  }
}

void ShowFilledParameters()
{
  if (!verbose)
    return;

}