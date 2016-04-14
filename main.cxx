#include "vtkColorTransferFunction.h"
#include "vtkDataSetTriangleFilter.h"
#include "vtkImageData.h"
#include "vtkMatrix3x3.h"
#include "vtkMatrix4x4.h"
#include "vtkNew.h"
#include "vtkPiecewiseFunction.h"
#include "vtkPolyData.h"
#include "vtkProjectedTetrahedraMapper.h"
#include "vtkCudaReconstructionFilter.h"
#include "vtkStructuredGrid.h"
#include "vtkThreshold.h"
#include "vtkTransform.h"
#include "vtkTransformFilter.h"
#include "vtkUnstructuredGrid.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkXMLImageDataReader.h"
#include "vtkXMLStructuredGridReader.h"
#include "vtkXMLStructuredGridWriter.h"

#include <vtksys/CommandLineArguments.hxx>
#include <vtksys/SystemTools.hxx>

// arguments
std::vector<int> g_gridDims(3);
std::vector<double> g_gridSpacing(3);
std::vector<double> g_gridOrigin(3);
std::vector<double> g_gridVecX(3);
std::vector<double> g_gridVecY(3);
std::vector<double> g_gridVecZ(3);
std::string g_depthMapFilename;
std::string g_matrixKRTDFilename;
std::string g_outputGridFilename;

bool read_arguments(int argc, char ** argv);
bool read_krtd(std::string filename, vtkMatrix3x3* matrixK, vtkMatrix4x4* matrixTR);

// todo remove
void init_arguments();

int main(int argc, char ** argv)
{
  // arguments
  // todo activate read_arguments and deactivate init_arguments
  /*
  if (!read_arguments(argc, argv))
    {
    return EXIT_FAILURE;
    }
   * */
  init_arguments();

  // read depth map
  vtkNew<vtkXMLImageDataReader> depthMapReader;
  depthMapReader->SetFileName(g_depthMapFilename.c_str());
  depthMapReader->Update();
  vtkImageData* depthMap = depthMapReader->GetOutput();

  // generate grid from arguments
  vtkNew<vtkImageData> grid;
  grid->SetDimensions(&g_gridDims[0]);
  grid->SetSpacing(&g_gridSpacing[0]);
  grid->SetOrigin(&g_gridOrigin[0]);

  // read depth map matrix
  vtkNew<vtkMatrix3x3> depthMapMatrixK;
  vtkNew<vtkMatrix4x4> depthMapMatrixTR;
  bool res = read_krtd(g_matrixKRTDFilename, depthMapMatrixK.Get(), depthMapMatrixTR.Get());
  if (!res)
    {
    return EXIT_FAILURE;
    }

  // todo compute matrix
  // compute transform matrix from gridVecs
  vtkNew<vtkMatrix4x4> gridMatrix;
  gridMatrix->Identity();

  // todo remove
  std::cout << "Reconstruction filter." << std::endl;

  // reconstruction
  vtkNew<vtkCudaReconstructionFilter> cudaReconstructionFilter;
  cudaReconstructionFilter->SetInputData(grid.Get());
  cudaReconstructionFilter->SetDepthMap(depthMap);
  cudaReconstructionFilter->SetDepthMapMatrixK(depthMapMatrixK.Get());
  cudaReconstructionFilter->SetDepthMapMatrixTR(depthMapMatrixTR.Get());
  cudaReconstructionFilter->SetGridMatrix(gridMatrix.Get());
  cudaReconstructionFilter->Update();

  // todo remove
  std::cout << "Transform filter." << std::endl;

  // todo compute transform according to gridVecs
  vtkNew<vtkTransform> transform;
  transform->SetMatrix(gridMatrix.Get());
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

  ////////// Setup visualization ////////

  // todo remove
  /*
  // structured to tetra
  vtkNew<vtkThreshold> thresholdFilter;
  thresholdFilter->SetInputData(outputGrid);
  thresholdFilter->ThresholdByUpper(-1);
  thresholdFilter->AllScalarsOff();
  vtkNew<vtkDataSetTriangleFilter> trifilter;
  trifilter->SetInputConnection(thresholdFilter->GetOutputPort());
  trifilter->Update();
  vtkUnstructuredGrid* uGrid = trifilter->GetOutput();
  std::cout << uGrid->GetNumberOfPoints() << std::endl;

  // mapper
  vtkNew<vtkProjectedTetrahedraMapper> gridMapper;
  gridMapper->SetInputData(uGrid);

  // Create transfer mapping scalar value to opacity.
  vtkNew<vtkPiecewiseFunction> opacityTransferFunction;
  opacityTransferFunction->AddPoint(00.0,  0.1);
  opacityTransferFunction->AddPoint(80.0,  0.2);
  opacityTransferFunction->AddPoint(120.0, 0.3);
  opacityTransferFunction->AddPoint(255.0, 0.4);

  // Create transfer mapping scalar value to color.
  vtkNew<vtkColorTransferFunction> colorTransferFunction;
  colorTransferFunction->AddRGBPoint(00.0,  1.0, 0.0, 0.0);
  colorTransferFunction->AddRGBPoint(80.0,  0.0, 0.0, 0.0);
  colorTransferFunction->AddRGBPoint(120.0, 0.0, 0.0, 1.0);
  colorTransferFunction->AddRGBPoint(160.0, 1.0, 0.0, 0.0);
  colorTransferFunction->AddRGBPoint(200.0, 0.0, 1.0, 0.0);
  colorTransferFunction->AddRGBPoint(255.0, 0.0, 1.0, 1.0);

  // The property describes how the data will look.
  vtkNew<vtkVolumeProperty> volumeProperty;
  volumeProperty->SetColor(colorTransferFunction.Get());
  volumeProperty->SetScalarOpacity(opacityTransferFunction.Get());
  volumeProperty->ShadeOff();
  volumeProperty->SetInterpolationTypeToLinear();

  // actor
  vtkNew<vtkVolume> gridActor;
  gridActor->SetMapper(gridMapper.Get());
  gridActor->SetProperty(volumeProperty.Get());

  vtkNew<vtkRenderer> renderer;
  vtkNew<vtkRenderWindow> renderWindow;
  renderWindow->AddRenderer(renderer.Get());
  vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
  renderWindowInteractor->SetRenderWindow(renderWindow.Get());

  vtkNew<vtkInteractorStyleTrackballCamera> style;
  renderWindowInteractor->SetInteractorStyle(style.Get());

  renderer->AddActor(gridActor.Get());

  renderWindow->Render();
  renderWindowInteractor->Start();
   * */

  return EXIT_SUCCESS;
}

bool read_krtd(std::string filename, vtkMatrix3x3* matrixK, vtkMatrix4x4* matrixTR)
{
  // open the file
  std::ifstream file(filename.c_str());
  if (!file.is_open())
    {
    // todo error message
    std::cout << "Unable to open krtd file." << std::endl;
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

  arg.AddArgument("--gridDims", argT::MULTI_ARGUMENT, &g_gridDims, "Specify the input grid dimensions (required)");
  arg.AddArgument("--gridSpacing", argT::MULTI_ARGUMENT, &g_gridSpacing, "Specify the input grid spacing (required)");
  arg.AddArgument("--gridOrigin", argT::MULTI_ARGUMENT, &g_gridOrigin, "Specify the input grid origin (required)");
  arg.AddArgument("--gridVecX", argT::MULTI_ARGUMENT, &g_gridVecX, "Specify the input grid direction X (required)");
  arg.AddArgument("--gridVecY", argT::MULTI_ARGUMENT, &g_gridVecY, "Specify the input grid direction Y (required)");
  arg.AddArgument("--gridVecZ", argT::MULTI_ARGUMENT, &g_gridVecZ, "Specify the input grid direction Z (required)");
  arg.AddArgument("--depthMapFilename", argT::SPACE_ARGUMENT, &g_depthMapFilename, "Specify the depth map filename (required)");
  arg.AddArgument("--matrixKRTDFilename", argT::SPACE_ARGUMENT, &g_matrixKRTDFilename, "Specify the depth map matrix filename (required)");
  arg.AddArgument("--outputGridFilename", argT::SPACE_ARGUMENT, &g_outputGridFilename, "Specify the output grid filename (required)");
  arg.AddBooleanArgument("--help", &help, "Print this help message");

  int result = arg.Parse();
  if (!result || help)
    {
    std::cout << arg.GetHelp() ;
    return false;
    }

  if (g_depthMapFilename == "" || g_matrixKRTDFilename == "" || g_outputGridFilename == "")
    {
    // todo error message
    std::cout << "Problem parsing arguments." << std::endl;
    std::cout << arg.GetHelp() ;
    return false;
    }

  return true;
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

  g_depthMapFilename = "/home/kitware/dev/cudareconstruction_sources/data/frame_0003_depth_map.0.vti";
  g_matrixKRTDFilename = "/home/kitware/dev/cudareconstruction_sources/data/frame_0003.krtd";
  g_outputGridFilename = "/home/kitware/dev/cudareconstruction_sources/data/outputgrid.vts";
}
