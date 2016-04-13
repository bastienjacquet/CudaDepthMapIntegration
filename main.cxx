#include <vtkVersion.h>
#include <vtkProperty.h>
#include <vtkDataSetMapper.h>
#include <vtkSmartPointer.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkTriangle.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkLine.h>
#include <vtkImageData.h>
#include <vtkProbeFilter.h>
#include <vtkDelaunay2D.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkDoubleArray.h>
#include <vtkMath.h>
#include <vtkCellLocator.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkFloatArray.h>
#include <vtkWarpScalar.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkInteractorStyleTrackballCamera.h>

#include "vtkColorTransferFunction.h"
#include "vtkDataSetTriangleFilter.h"
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

int main(int, char *[])
{
  // read depth map
  vtkNew<vtkXMLPolyDataReader> depthMapReader;
  depthMapReader->SetFileName("/home/kitware/dev/cudareconstruction_sources/data/depthmap.vtp");
  depthMapReader->Update();
  vtkPolyData* depthMap = depthMapReader->GetOutput();
  std::cout << depthMap->GetNumberOfPoints() << std::endl;

  // read grid
  vtkNew<vtkXMLStructuredGridReader> gridReader;
  gridReader->SetFileName("/home/kitware/dev/cudareconstruction_sources/data/grid.vts");
  gridReader->Update();
  vtkStructuredGrid* grid = gridReader->GetOutput();
  std::cout << grid->GetNumberOfPoints() << std::endl;

  // reconstruction
  vtkNew<vtkCudaReconstructionFilter> cudaReconstructionFilter;
  cudaReconstructionFilter->SetDepthMap(depthMap);
  cudaReconstructionFilter->SetInputData(grid);
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
