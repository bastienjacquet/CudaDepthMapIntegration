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
#include "vtkUnstructuredGrid.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkXMLPolyDataReader.h"
#include "vtkXMLStructuredGridReader.h"
#include "vtkXMLStructuredGridWriter.h"

bool ReadKRTD(std::string filename, vtkMatrix3x3* matrixK, vtkMatrix4x4* matrixTR);

int main(int, char *[])
{
  // arguments
  int gridDims[3] = { 21, 21, 21 };
  double gridSpacing[3] = { 1, 1, 1 };
  double gridOrigin[3] = { -10, -10, -10 };
  double gridVecX[3] = { 1, 0, 0 };
  double gridVecY[3] = { 0, 1, 0 };
  double gridVecZ[3] = { 0, 0, 1 };

  // read depth map
  vtkNew<vtkXMLPolyDataReader> depthMapReader;
  depthMapReader->SetFileName("/home/kitware/dev/cudareconstruction_sources/data/depthmap.vtp");
  depthMapReader->Update();
  vtkPolyData* depthMap = depthMapReader->GetOutput();
  std::cout << depthMap->GetNumberOfPoints() << std::endl;

  // generate grid from arguments
  vtkNew<vtkImageData> grid;
  grid->SetDimensions(gridDims);
  grid->SetSpacing(gridSpacing);
  grid->SetOrigin(gridOrigin);
  std::cout << grid->GetNumberOfPoints() << std::endl;

  // read depth map matrix
  vtkNew<vtkMatrix3x3> depthMapMatrixK;
  vtkNew<vtkMatrix4x4> depthMapMatrixTR;
  bool res = ReadKRTD("/home/kitware/dev/cudareconstruction_sources/data/matrix.krtd",
                      depthMapMatrixK.Get(), depthMapMatrixTR.Get());
  if (!res)
    {
    std::cout << "Read krtd failed." << std::endl;
    return EXIT_FAILURE;
    }

  // reconstruction
  vtkNew<vtkCudaReconstructionFilter> cudaReconstructionFilter;
  cudaReconstructionFilter->SetInputData(grid.Get());
  cudaReconstructionFilter->SetDepthMap(depthMap);
  cudaReconstructionFilter->SetDepthMapMatrixK(depthMapMatrixK.Get());
  cudaReconstructionFilter->SetDepthMapMatrixTR(depthMapMatrixTR.Get());
  cudaReconstructionFilter->SetGridVecX(gridVecX);
  cudaReconstructionFilter->SetGridVecY(gridVecY);
  cudaReconstructionFilter->SetGridVecZ(gridVecZ);
  cudaReconstructionFilter->Update();
  vtkStructuredGrid* outputGrid = vtkStructuredGrid::SafeDownCast(cudaReconstructionFilter->GetOutput());

  vtkNew<vtkXMLStructuredGridWriter> gridWriter;
  gridWriter->SetFileName("/home/kitware/dev/cudareconstruction_sources/data/outputgrid.vts");
  gridWriter->SetInputData(outputGrid);
  gridWriter->Write();

  ////////// Setup visualization ////////

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

bool ReadKRTD(std::string filename, vtkMatrix3x3* matrixK, vtkMatrix4x4* matrixTR)
{
  // open the file
  std::ifstream file(filename.c_str());
  if (!file.is_open())
    {
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