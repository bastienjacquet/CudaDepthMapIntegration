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
#include "vtkImageData.h"
#include "vtkNew.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include "vtkXMLPolyDataReader.h"
#include "vtkXMLPolyDataWriter.h"

#include "MeshColoration.h"
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

  MeshColoration* coloration = new MeshColoration(reader->GetOutput(), g_globalVTIFilePath, g_globalKRTDFilePath);
  bool process = coloration->ProcessColoration();

  if (process)
    {
    ShowInformation("** Write output image");
    vtkNew<vtkXMLPolyDataWriter> writer;
    writer->SetFileName(g_outputPath.c_str());
    writer->SetInputData(coloration->GetOutput());
    writer->Update();
    }
  else
    {
    ShowInformation("Error during coloration process...");
    }

  delete coloration;

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