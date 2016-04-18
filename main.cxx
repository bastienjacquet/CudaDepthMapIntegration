#include "vtkImageData.h"
#include "vtkMatrix3x3.h"
#include "vtkMatrix4x4.h"
#include "vtkNew.h"
#include "vtkPiecewiseFunction.h"
#include "vtkPolyData.h"
#include "vtkCudaReconstructionFilter.h"
#include "vtkStructuredGrid.h"
#include "vtkTransform.h"
#include "vtkTransformFilter.h"
#include "vtkUnstructuredGrid.h"
#include "vtkXMLImageDataReader.h"
#include "vtkXMLStructuredGridReader.h"
#include "vtkXMLStructuredGridWriter.h"

#include <vtksys/CommandLineArguments.hxx>
#include <vtksys/SystemTools.hxx>

#include "ReconstructionData.h"

#include <string>

// read arguments
std::vector<int> g_gridDims;
std::vector<double> g_gridSpacing;
std::vector<double> g_gridOrigin;
std::vector<double> g_gridVecX;
std::vector<double> g_gridVecY;
std::vector<double> g_gridVecZ;
std::string g_outputGridFilename;
std::string g_pathFolder; // Path to the folder which contains all data
std::string g_depthMapContainer = "vtiList.txt"; // File which contains all path of depth map
std::string g_KRTContainer = "kList.txt"; // File which contains all path of KRT matrix ofr each depth map

// filled attributes
std::vector<std::string> g_depthMapPathList; // Contains all depth map path
std::vector<std::string> g_KRTPathList; // Contains all KRT matrix path
std::vector<ReconstructionData*> g_dataList;
vtkMatrix4x4* g_gridMatrix;

bool read_arguments(int argc, char ** argv);
bool read_krtd(std::string filename, vtkMatrix3x3* matrixK, vtkMatrix4x4* matrixTR);
bool createData();
void createGridMatrix();
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

// todo remove
void init_arguments();

//cudareconstruction.exe --gridDims 100 100 100 --gridSpacing 0.1 0.1 0.1 --gridOrigin -5 -5 -5 --gridVecX 1 0 0 --gridVecY 0 1 0 --gridVecZ 0 0 1 --dataFolder C:\Dev\nda\TRG\Data --outputGridFilename C:\Dev\nda\TRG\Data\output.vts
int main(int argc, char ** argv)
{
  // arguments
  if (!read_arguments(argc, argv))
    {
    return EXIT_FAILURE;
    }

  // Read and create a list of ReconstructionData
  createData();

  // Create grid matrix from VecXYZ
  createGridMatrix();

  // generate grid from arguments
  vtkNew<vtkImageData> grid;
  grid->SetDimensions(&g_gridDims[0]);
  grid->SetSpacing(&g_gridSpacing[0]);
  grid->SetOrigin(&g_gridOrigin[0]);


  // todo remove
  std::cout << "Reconstruction filter." << std::endl;

  // reconstruction
  vtkNew<vtkCudaReconstructionFilter> cudaReconstructionFilter;
  cudaReconstructionFilter->UseCudaOff();
  cudaReconstructionFilter->SetInputData(grid.Get());
  cudaReconstructionFilter->SetDataList(g_dataList);
  cudaReconstructionFilter->SetGridMatrix(g_gridMatrix);
  cudaReconstructionFilter->Update();

  // todo remove
  std::cout << "Transform filter." << std::endl;

  // todo compute transform according to gridVecs
  vtkNew<vtkTransform> transform;
  transform->SetMatrix(g_gridMatrix);
  vtkNew<vtkTransformFilter> transformFilter;
  transformFilter->SetInputConnection(cudaReconstructionFilter->GetOutputPort());
  transformFilter->SetTransform(transform.Get());
  transformFilter->Update();
  vtkStructuredGrid* outputGrid = vtkStructuredGrid::SafeDownCast(transformFilter->GetOutput());

  // todo remove
  std::cout << "Write output." << std::endl;

  vtkNew<vtkXMLStructuredGridWriter> gridWriter;
  gridWriter->SetFileName(g_outputGridFilename.c_str());
  gridWriter->SetInputData(outputGrid);
  gridWriter->Write();

  // Clean pointers
  g_gridMatrix->Delete();
  g_dataList.clear();

  return EXIT_SUCCESS;
}

bool read_krtd(std::string filename, vtkMatrix3x3* matrixK, vtkMatrix4x4* matrixTR)
{
  // open the file
  std::ifstream file(filename.c_str());
  if (!file.is_open())
    {
    std::cerr << "Unable to open krtd file : " << filename  << std::endl;
    return false;
    }

  std::string line;

  // get matrix K
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

  // get matrix R
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

  // get matrix T
  getline(file, line);
  std::istringstream iss(line);
  for (int i = 0; i < 3; i++)
    {
    double value;
    iss >> value;
    matrixTR->SetElement(i, 3, value);
    }

  // finalize matrix TR
  for (int j = 0; j < 4; j++)
    {
    matrixTR->SetElement(3, j, 0);
    }
  matrixTR->SetElement(3, 3, 1);

  return true;
}

//-----------------------------------------------------------------------------
bool read_arguments(int argc, char ** argv)
{
  bool help = false;

  vtksys::CommandLineArguments arg;
  arg.Initialize(argc, argv);
  typedef vtksys::CommandLineArguments argT;

  arg.AddArgument("--gridDims", argT::MULTI_ARGUMENT, &g_gridDims, "Input grid dimensions (required)");
  arg.AddArgument("--gridSpacing", argT::MULTI_ARGUMENT, &g_gridSpacing, "Input grid spacing (required)");
  arg.AddArgument("--gridOrigin", argT::MULTI_ARGUMENT, &g_gridOrigin, "Input grid origin (required)");
  arg.AddArgument("--gridVecX", argT::MULTI_ARGUMENT, &g_gridVecX, "Input grid direction X (required)");
  arg.AddArgument("--gridVecY", argT::MULTI_ARGUMENT, &g_gridVecY, "Input grid direction Y (required)");
  arg.AddArgument("--gridVecZ", argT::MULTI_ARGUMENT, &g_gridVecZ, "Input grid direction Z (required)");
  arg.AddArgument("--outputGridFilename", argT::SPACE_ARGUMENT, &g_outputGridFilename, "Output grid filename (required)");
  arg.AddArgument("--dataFolder", argT::SPACE_ARGUMENT, &g_pathFolder, "Folder which contains all data (required)");
  arg.AddArgument("--depthMapFile", argT::SPACE_ARGUMENT, &g_depthMapContainer, "File which contains all the depth map path(default is vtiList.txt)");
  arg.AddArgument("--KRTFile", argT::SPACE_ARGUMENT, &g_KRTContainer, "File which contains all the KRTD path (default is kList.txt)");
  arg.AddBooleanArgument("--help", &help, "Print this help message");

  int result = arg.Parse();
  if (!result || help)
    {
    std::cout << arg.GetHelp() ;
    return false;
    }

  if (g_outputGridFilename == "" || g_depthMapContainer == "" || g_KRTContainer == "")
    {
    // todo error message
    std::cerr << "Problem parsing arguments." << std::endl;
    std::cerr << arg.GetHelp() ;
    return false;
    }

  return true;
}

//-----------------------------------------------------------------------------
bool createData()
{
  std::string dmapGlobalFile = g_pathFolder + "\\" + g_depthMapContainer;
  std::string krtGlobalFile = g_pathFolder + "\\" + g_KRTContainer;

  // open the file which contains depthMap path
  std::ifstream depthMapContainer(dmapGlobalFile.c_str());
  std::ifstream matrixContainer(krtGlobalFile.c_str());
  if (!depthMapContainer.is_open() || !matrixContainer.is_open())
  {
    std::cerr << "Unable to open file which contains depth map or matrix path." << std::endl;
    return false;
  }

  g_dataList.clear();

  std::string depthMapPath, matrixPath;
  while (!depthMapContainer.eof())
    {
    // DEPTH MAP
    std::getline(depthMapContainer, depthMapPath);
    // only get the file name, not the whole path
    std::vector <std::string> elems;
    split(depthMapPath, '/', elems);
    if (elems.size() == 0)
      {
      continue;
      }
    depthMapPath = g_pathFolder + "\\" + elems[elems.size() - 1];

    vtkXMLImageDataReader* depthMapReader = vtkXMLImageDataReader::New();
    depthMapReader->SetFileName(depthMapPath.c_str());
    depthMapReader->Update();

    // MATRIX
    std::getline(matrixContainer, matrixPath);
    // only get the file name, not the whole path
    elems.clear();
    split(matrixPath, '/', elems);
    matrixPath = g_pathFolder + "\\" + elems[elems.size() - 1];

    vtkMatrix3x3* depthMapMatrixK = vtkMatrix3x3::New();
    vtkMatrix4x4* depthMapMatrixTR = vtkMatrix4x4::New();
    bool isReadOk = read_krtd(matrixPath, depthMapMatrixK, depthMapMatrixTR);
    // Skip the creation of a new data if matrix is not readable
    if (!isReadOk)
      {
      continue;
      }

    // CREATE DATA
    ReconstructionData* data = new ReconstructionData();
    data->SetDepthMap(depthMapReader->GetOutput());
    data->SetMatrixK(depthMapMatrixK);
    data->SetMatrixTR(depthMapMatrixTR);

    g_dataList.push_back(data);
    }

  return true;
}

//-----------------------------------------------------------------------------
void createGridMatrix()
{
  vtkMatrix4x4* gridMatrix = vtkMatrix4x4::New();
  gridMatrix->Identity();

  //// Fill matrix
  //gridMatrix->SetElement(0, 0, g_gridVecX[0]);
  //gridMatrix->SetElement(0, 1, g_gridVecX[1]);
  //gridMatrix->SetElement(0, 2, g_gridVecX[2]);
  //gridMatrix->SetElement(1, 0, g_gridVecY[0]);
  //gridMatrix->SetElement(1, 1, g_gridVecY[1]);
  //gridMatrix->SetElement(1, 2, g_gridVecY[2]);
  //gridMatrix->SetElement(2, 0, g_gridVecZ[0]);
  //gridMatrix->SetElement(2, 1, g_gridVecZ[1]);
  //gridMatrix->SetElement(2, 2, g_gridVecZ[2]);

  g_gridMatrix = gridMatrix;
}

//-----------------------------------------------------------------------------
void init_arguments()
{
  g_gridDims.clear();
  g_gridDims.push_back(100);
  g_gridDims.push_back(100);
  g_gridDims.push_back(100);

  g_gridSpacing.clear();
  g_gridSpacing.push_back(0.1);
  g_gridSpacing.push_back(0.1);
  g_gridSpacing.push_back(0.1);

  g_gridOrigin.clear();
  g_gridOrigin.push_back(-5);
  g_gridOrigin.push_back(-5);
  g_gridOrigin.push_back(-5);

  g_gridVecX.clear();
  g_gridVecX.push_back(1);
  g_gridVecX.push_back(0);
  g_gridVecX.push_back(0);

  g_gridVecY.clear();
  g_gridVecY.push_back(0);
  g_gridVecY.push_back(1);
  g_gridVecY.push_back(0);

  g_gridVecZ.clear();
  g_gridVecZ.push_back(0);
  g_gridVecZ.push_back(0);
  g_gridVecZ.push_back(1);

  g_outputGridFilename = "C:/Dev/nda/TRG/CudaDepthMapIntegration/data/outputgrid.vts";
}

//-----------------------------------------------------------------------------
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems)
{
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim))
    {
    elems.push_back(item);
    }
  return elems;
}