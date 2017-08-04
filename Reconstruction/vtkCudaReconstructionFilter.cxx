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
#include "vtkCudaReconstructionFilter.h"

#include "ReconstructionData.h"

//#include "vtkCell.h"
//#include "vtkCellData.h"
//#include "vtkCommand.h"
//#include "vtkDataSet.h"
#include "vtkDoubleArray.h"
#include "vtkExtentTranslator.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMath.h"
#include "vtkMathUtilities.h"
#include "vtkMatrix3x3.h"
#include "vtkMatrix4x4.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkNew.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkStreamingDemandDrivenPipeline.h"
//#include "vtkStructuredGrid.h"
#include <vtksys/SystemInformation.hxx>
#include <vtksys/SystemTools.hxx>
//#include "vtkTransform.h"
#include "vtkXMLImageDataReader.h"
#include "vtkXMLImageDataWriter.h"
#include "vtkXMLPImageDataWriter.h"

#include "Helper.h"

#include <cmath>
#include <string>
#include <time.h>
#include <vector>

vtkStandardNewMacro(vtkCudaReconstructionFilter);
vtkSetObjectImplementationMacro(vtkCudaReconstructionFilter, GridMatrix, vtkMatrix4x4);


void CudaInitialize(vtkMatrix4x4* i_gridMatrix, int h_gridDims[3],
  double h_gridOrig[3], double h_gridSpacing[3], double h_rayPThick, double h_rayPRho,
  double h_rayPEta, double h_rayPDelta, int h_tilingDims[3], int h_depthMapDim[2],
  vtkCudaReconstructionFilter* ch_cudaFilter);

template <typename TVolumetric>
bool ProcessDepthMap(std::vector<std::string> vtiList,std::vector<std::string> krtdList,
  double thresholdBestCost, TVolumetric* io_scalar);

//----------------------------------------------------------------------------
vtkCudaReconstructionFilter::vtkCudaReconstructionFilter()
{
  this->SetNumberOfInputPorts(0);
  this->SetNumberOfOutputPorts(1);

  // create default translator
  this->ExtentTranslator = vtkExtentTranslator::New();

  this->PImageDataWriter = vtkXMLPImageDataWriter::New();

  this->CurrentDate = 0;
  this->ForceCubicVoxels = false;
  this->WriteSummary = false;
  this->DataFolder = 0;
  this->DepthMapFile = 0;
  this->FilePathKRTD = 0;
  this->FilePathVTI = 0;
  this->KRTDFile = 0;

  for (int i = 0; i < 3; i++)
  {
    this->GridEnd[i] = VTK_DOUBLE_MIN;
    this->GridOrigin[i] = VTK_DOUBLE_MIN;
    this->GridSpacing[i] = VTK_DOUBLE_MIN;
    this->GridVecX[i] = 0.0;
    this->GridVecY[i] = 0.0;
    this->GridVecZ[i] = 0.0;
    this->GridNbVoxels[i] = VTK_INT_MIN;
    this->TilingSize[i] = 0;
  }
  this->GridVecX[0] = 1.0;
  this->GridVecY[1] = 1.0;
  this->GridVecZ[2] = 1.0;
  this->RayPotentialDelta = 0.0;
  this->RayPotentialEta = 0.0;
  this->RayPotentialRho = 0.0;
  this->RayPotentialThickness = 0.0;
  this->ThresholdBestCost = 0.0;
  this->GridPropertiesMode = DEFAULT_MODE;
  this->GridMatrix = 0;
  this->ExecutionTime = -1;
}

//----------------------------------------------------------------------------
vtkCudaReconstructionFilter::~vtkCudaReconstructionFilter()
{
  this->FilePathKRTD = 0;
  this->FilePathVTI = 0;
  this->GridMatrix = 0;
  this->DataFolder = 0;
  this->DepthMapFile = 0;
}

//----------------------------------------------------------------------------
int vtkCudaReconstructionFilter::ProcessRequest(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
//  if (request->Has(vtkDemandDrivenPipeline::REQUEST_DATA()))
//  {
//    std::cout<<" * Request data * "<<std::endl<<std::endl;
//  }

//  if(request->Has(vtkStreamingDemandDrivenPipeline::REQUEST_UPDATE_EXTENT()))
//  {
//    std::cout<<" * Request update extent * "<<std::endl<<std::endl;
//  }

//  if(request->Has(vtkStreamingDemandDrivenPipeline::REQUEST_INFORMATION()))
//  {
//    std::cout<<" * Request information * "<<std::endl<<std::endl;
//  }

  return this->Superclass::ProcessRequest(request, inputVector, outputVector);
}

//----------------------------------------------------------------------------
int vtkCudaReconstructionFilter::RequestInformation(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** vtkNotUsed(inputVector),
  vtkInformationVector* outputVector)
{
  this->ExecutionTime = clock();

  if (!this->CheckArguments())
  {
    return 0;
  }

  // Get the output data object
  vtkInformation* outInfo = outputVector->GetInformationObject(0);
  vtkImageData *output =
    vtkImageData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

  outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
               0, this->GridNbVoxels[0] - 1,
               0, this->GridNbVoxels[1] - 1,
               0, this->GridNbVoxels[2] - 1);

  output->SetSpacing(this->GridSpacing);
  output->SetOrigin(this->GridOrigin);

  // Use Estimated size and free memory to determine how many pieces are needed
  size_t estimatedSize = static_cast<size_t>(this->GridNbVoxels[0])
                         * this->GridNbVoxels[1] * this->GridNbVoxels[2] * sizeof(double);
  std::cout << "Estimated size (kiB) : " << estimatedSize << std::endl;
  vtksys::SystemInformation sysInfo;
  size_t availableSize = (sysInfo.GetHostMemoryTotal() - sysInfo.GetHostMemoryUsed()) * 1024;
  std::cout << "Available size (kiB) : " << availableSize << std::endl;
  double ratio = static_cast<double>(estimatedSize) / static_cast<double>(availableSize);
  int nbPieces = vtkMath::Ceil(ratio);
  std::cout << "nbPieces : " << nbPieces << std::endl;

  // Set the extent translator's properties
  vtkExtentTranslator *translator = this->ExtentTranslator;
  translator->SetWholeExtent(0, this->GridNbVoxels[0] - 1,
                             0, this->GridNbVoxels[1] - 1,
                             0, this->GridNbVoxels[2] - 1);
  translator->SetPiece(0);
  translator->SetNumberOfPieces(nbPieces);

  // Set the KRTD and VTI file paths
  std::string dmapGlobalFile = std::string(this->DataFolder) + "/" + std::string(this->DepthMapFile);
  std::string krtdGlobalFile = std::string(this->DataFolder) + "/" + std::string(this->KRTDFile);
  this->SetFilePathKRTD(krtdGlobalFile.c_str());
  this->SetFilePathVTI(dmapGlobalFile.c_str());

  this->CreateGridMatrixFromInput();

  this->SetCurrentDate(vtksys::SystemTools::GetCurrentDateTime("%d.%m.%Y-%Hh%M").c_str());

  return 1;
}

//----------------------------------------------------------------------------
int vtkCudaReconstructionFilter::RequestData(
  vtkInformation *request,
  vtkInformationVector **vtkNotUsed(inputVector),
  vtkInformationVector *outputVector)
{

  // Get the output data object
  vtkInformation* outInfo = outputVector->GetInformationObject(0);
  vtkImageData *output =
    vtkImageData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

  vtkExtentTranslator *translator = this->ExtentTranslator;

  int *outExt = outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT());

  std::cout << "Piece : " << translator->GetPiece() + 1 << "/"
            << translator->GetNumberOfPieces() << std::endl;

  if (translator->GetNumberOfPieces() > 1 && translator->GetPiece() == 0)
  {
    vtkOutputWindowDisplayDebugText("Not enough available memory to allocate the whole grid at once."
                                    "\nWriting the pieces to disk.\n");
  }

  // Allocate and initialize the output scalars
  output->SetExtent(outExt);
  output->AllocateScalars(VTK_DOUBLE, 1);
  output->GetPointData()->GetScalars()->FillComponent(0, 0.0);
  output->GetPointData()->GetScalars()->SetName("reconstruction_scalar");
  output->GetPointData()->SetActiveScalars("reconstruction_scalar");

  double* outScalar = static_cast<double *>(output->GetScalarPointerForExtent(outExt));

  int pieceNbVoxels[3];
  pieceNbVoxels[0] = outExt[1] - outExt[0] + 1;
  pieceNbVoxels[1] = outExt[3] - outExt[2] + 1;
  pieceNbVoxels[2] = outExt[5] - outExt[4] + 1;

  double pieceOrigin[3];
  for (int i = 0; i < 3; i++)
  {
    pieceOrigin[i] = this->GridOrigin[i] + outExt[2*i] * this->GridSpacing[i];
  }

  std::stringstream ss;
  ss << "Processing piece " << translator->GetPiece() << "/"
     << translator->GetNumberOfPieces();
  this->SetProgressText(ss.str().c_str());

  // Compute the reconstruction by cuda on the current piece
  if (this->Compute(outScalar, pieceNbVoxels, pieceOrigin) != 0)
  {
    return 0;
  }

  // Write pieces to disk if they cannot fit in memory
  if (translator->GetNumberOfPieces() > 1)
  {
    vtkNew<vtkXMLImageDataWriter> writer;
    std::stringstream outputFileName;
    outputFileName << "Reconstruction_output_" << this->CurrentDate
                   << "_" << translator->GetPiece() << ".vti";
    writer->SetInputData(output);
    writer->SetFileName(outputFileName.str().c_str());
    writer->Write();
  }

  // Last piece has been processed
  if (translator->GetPiece() + 1 == translator->GetNumberOfPieces())
  {
    clock_t end = clock();
    this->ExecutionTime = double(end - ExecutionTime) / CLOCKS_PER_SEC;
    std::cout << "Execution time : " << ExecutionTime << " s" << std::endl;

    if (this->WriteSummary)
    {
      this->WriteSummaryFile();
    }

    return 1;
  }

  output->GetPointData()->RemoveArray(0);
  translator->SetPiece(translator->GetPiece() + 1);

  return 1;
}

//-----------------------------------------------------------------------------
int vtkCudaReconstructionFilter::RequestUpdateExtent(
  vtkInformation *request,
  vtkInformationVector **vtkNotUsed(inputVector),
  vtkInformationVector *outputVector)
{
  // Get the output data object
  vtkInformation* outInfo = outputVector->GetInformationObject(0);

  vtkExtentTranslator* translator = this->ExtentTranslator;

  // Get the current piece's extent
  int outExt[6];
  if (translator->PieceToExtentByPoints())
  {
    translator->GetExtent(outExt);
  }
  outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), outExt, 6);

  return 1;
}

//-----------------------------------------------------------------------------
/* Check if input vectors are orthogonals (GridVecX, GridVecY, GridVecZ) */
bool vtkCudaReconstructionFilter::AreVectorsOrthogonal()
{
  double X[3] = { this->GridVecX[0], this->GridVecX[1], this->GridVecX[2] };
  double Y[3] = { this->GridVecY[0], this->GridVecY[1], this->GridVecY[2] };
  double Z[3] = { this->GridVecZ[0], this->GridVecZ[1], this->GridVecZ[2] };

  double XY = vtkMath::Dot(X, Y);
  double YZ = vtkMath::Dot(Y, Z);
  double ZX = vtkMath::Dot(Z, X);

  double epsilon = 10e-6;
  if (vtkMathUtilities::FuzzyCompare(XY, 0.0, epsilon)
      && vtkMathUtilities::FuzzyCompare(YZ, 0.0, epsilon)
      && vtkMathUtilities::FuzzyCompare(ZX, 0.0, epsilon))
  {
    return true;
  }

  return false;
}

//----------------------------------------------------------------------------
bool vtkCudaReconstructionFilter::CheckArguments()
{
  // Ray potential parameters check
  if (this->RayPotentialRho <= 0.0)
  {
    vtkErrorMacro("Ray potential rho must be > 0");
    return false;
  }
  if (this->RayPotentialEta < 0.0 || this->RayPotentialEta > 1.0)
  {
    vtkErrorMacro("Ray potential eta must be between 0 and 1");
    return false;
  }
  if (this->RayPotentialDelta <= 0.0)
  {
    vtkErrorMacro("Ray potential delta must be > 0");
    return false;
  }
  if (this->RayPotentialThickness <= 0.0)
  {
    vtkErrorMacro("Ray potential thickness must be > 0");
    return false;
  }
  if (this->RayPotentialDelta < this->RayPotentialThickness)
  {
    vtkErrorMacro("Ray potential thickness must be less than delta");
    return false;
  }


  if (this->ThresholdBestCost <= 0.0)
  {
    vtkErrorMacro("Best Cost Threshold must be > 0");
    return false;
  }


  // File containers parameters check
  if (this->DataFolder == 0)
  {
    vtkErrorMacro("Data folder must be specified");
    return false;
  }
  if (this->DepthMapFile == 0)
  {
    vtkErrorMacro("Depth Map filename must be specified");
    return false;
  }
  if (this->KRTDFile == 0)
  {
    vtkErrorMacro("KRTD filename must be specified");
    return false;
  }


  // Grid parameters check
  if (!this->AreVectorsOrthogonal())
  {
    vtkErrorMacro("Grid vectors are not orthogonal");
    return false;
  }

  // Exactly 3 out of the 4 grid properties must be set
  enum gridProperties_t {ORIGIN, END, NB_VOXELS, SPACING};
  bool gridPropertySet[4] = {false};

  if (this->GridPropertiesMode != END_NBVOX_SPAC
      && this->GridOrigin[0] != VTK_DOUBLE_MIN
      && this->GridOrigin[1] != VTK_DOUBLE_MIN
      && this->GridOrigin[2] != VTK_DOUBLE_MIN)
  {
    gridPropertySet[ORIGIN] = true;
  }

  if (this->GridPropertiesMode != ORIG_NBVOX_SPAC
      && this->GridEnd[0] != VTK_DOUBLE_MIN
      && this->GridEnd[1] != VTK_DOUBLE_MIN
      && this->GridEnd[2] != VTK_DOUBLE_MIN)
  {
    gridPropertySet[END] = true;
  }

  if (this->GridPropertiesMode != ORIG_END_SPAC
      && this->GridNbVoxels[0] != VTK_INT_MIN
      && this->GridNbVoxels[1] != VTK_INT_MIN
      && this->GridNbVoxels[2] != VTK_INT_MIN)
  {
    if (this->GridNbVoxels[0] > 0
        && this->GridNbVoxels[1] > 0
        && this->GridNbVoxels[2] > 0)
    {
      gridPropertySet[NB_VOXELS] = true;
    }
    else
    {
      vtkErrorMacro("Grid number of voxels must all be > 0");
      return false;
    }
  }

  if (this->GridPropertiesMode != ORIG_END_NBVOX
      && this->GridSpacing[0] != VTK_DOUBLE_MIN
      && this->GridSpacing[1] != VTK_DOUBLE_MIN
      && this->GridSpacing[2] != VTK_DOUBLE_MIN)
  {
    if (this->GridSpacing[0] > 0.0
        && this->GridSpacing[1] > 0.0
        && this->GridSpacing[2] > 0.0)
    {
      gridPropertySet[SPACING] = true;
    }
    else
    {
      vtkErrorMacro("Grid spacings must all be > 0");
      return false;
    }
  }


  if (this->GridPropertiesMode == DEFAULT_MODE)
  {
    // Count the number of grid properties set
    int nbGridPropertySet = 0;
    for (int i = 0; i < 4; i++)
    {
      nbGridPropertySet += gridPropertySet[i];
    }

    if (nbGridPropertySet != 3)
    {
      vtkErrorMacro("Exactly 3 grid properties must be set");
      return false;
    }
    else
    {
      // Find which grid properties have been set
      if (gridPropertySet[ORIGIN] == false)
      {
        this->GridPropertiesMode = END_NBVOX_SPAC;
      }
      if (gridPropertySet[END] == false)
      {
        this->GridPropertiesMode = ORIG_NBVOX_SPAC;
      }
      if (gridPropertySet[NB_VOXELS] == false)
      {
        this->GridPropertiesMode = ORIG_END_SPAC;
      }
      if (gridPropertySet[SPACING] == false)
      {
        this->GridPropertiesMode = ORIG_END_NBVOX;
      }
    }
  }

  if (this->ForceCubicVoxels && this->GridPropertiesMode != ORIG_END_NBVOX)
  {
    // Get the minimum spacing when it has been user specified
    double min = *std::min_element(this->GridSpacing, this->GridSpacing + 3);
    for (int i = 0; i < 3; i++)
    {
      this->GridSpacing[i] = min;
    }
  }

  // Fill the missing grid argument
  if (this->GridPropertiesMode == END_NBVOX_SPAC)
  {
    // Grid origin has to be computed
    this->GridOrigin[0] = this->GridEnd[0] - (this->GridSpacing[0] * this->GridNbVoxels[0]);
    this->GridOrigin[1] = this->GridEnd[1] - (this->GridSpacing[1] * this->GridNbVoxels[1]);
    this->GridOrigin[2] = this->GridEnd[2] - (this->GridSpacing[2] * this->GridNbVoxels[2]);
  }
  else if (this->GridPropertiesMode == ORIG_NBVOX_SPAC)
  {
    // Grid end has to be computed
    this->GridEnd[0] = this->GridOrigin[0] + (this->GridSpacing[0] * this->GridNbVoxels[0]);
    this->GridEnd[1] = this->GridOrigin[1] + (this->GridSpacing[1] * this->GridNbVoxels[1]);
    this->GridEnd[2] = this->GridOrigin[2] + (this->GridSpacing[2] * this->GridNbVoxels[2]);
  }
  else
  {
    // Get the requested grid size on each axis
    double sizeX = this->GridEnd[0] - this->GridOrigin[0];
    double sizeY = this->GridEnd[1] - this->GridOrigin[1];
    double sizeZ = this->GridEnd[2] - this->GridOrigin[2];

    if (sizeX <= 0.0 || sizeY <= 0.0 || sizeZ <= 0.0)
    {
      vtkErrorMacro("Grid end coordinates must be greater than the grid origin's");
      return false;
    }

    if (this->GridPropertiesMode == ORIG_END_SPAC)
    {
      // Compute the number of voxels accord to the spacing
      this->GridNbVoxels[0] = vtkMath::Ceil(sizeX / this->GridSpacing[0]);
      this->GridNbVoxels[1] = vtkMath::Ceil(sizeY / this->GridSpacing[1]);
      this->GridNbVoxels[2] = vtkMath::Ceil(sizeZ / this->GridSpacing[2]);

      if (this->GridNbVoxels[0] <= 0 || this->GridNbVoxels[1] <= 0 || this->GridNbVoxels[2] <= 0)
      {
        vtkErrorMacro("Grid spacing must not be bigger than the requested grid size");
        return false;
      }
    }
    else if (this->GridPropertiesMode == ORIG_END_NBVOX)
    {
      // Compute the spacing according to the dimensions
      this->GridSpacing[0] = sizeX / static_cast<double>(this->GridNbVoxels[0]);
      this->GridSpacing[1] = sizeY / static_cast<double>(this->GridNbVoxels[1]);
      this->GridSpacing[2] = sizeZ / static_cast<double>(this->GridNbVoxels[2]);

      if (this->ForceCubicVoxels)
      {
        // Get the minimum spacing
        double min = *std::min_element(this->GridSpacing, this->GridSpacing + 3);
        for (int i = 0; i < 3; i++)
        {
          this->GridSpacing[i] = min;
        }

        // Compute the new number of voxels
        this->GridNbVoxels[0] =  vtkMath::Ceil((this->GridEnd[0] - this->GridOrigin[0])
                                  / this->GridSpacing[0]);
        this->GridNbVoxels[1] =  vtkMath::Ceil((this->GridEnd[1] - this->GridOrigin[1])
                                  / this->GridSpacing[1]);
        this->GridNbVoxels[2] =  vtkMath::Ceil((this->GridEnd[2] - this->GridOrigin[2])
                                  / this->GridSpacing[2]);
      }
    }
    else
    {
      vtkErrorMacro("Wrong GridPropertiesMode value");
      return false;
    }
  }


  if (this->TilingSize[0] < 0
      && this->TilingSize[1] < 0
      && this->TilingSize[2] < 0)
  {
    vtkErrorMacro("Tiling size must be positive");
    return false;
  }
  // Tiling cannot be bigger than grid number of voxels
  for (int i = 0; i < 3; i++)
  {
    if (this->TilingSize[i] > this->GridNbVoxels[i])
    {
      this->TilingSize[i] = this->GridNbVoxels[i];
    }
  }

  return true;
}

//----------------------------------------------------------------------------
//int vtkCudaReconstructionFilter::Compute(vtkDoubleArray* outScalar)
int vtkCudaReconstructionFilter::Compute(double* outScalar, int pieceNbVoxels[3], double pieceOrigin[3])
{
  std::vector<std::string> vtiList = help::ExtractAllFilePath(this->FilePathVTI);
  std::vector<std::string> krtdList = help::ExtractAllFilePath(this->FilePathKRTD);

  if (vtiList.size() == 0 || krtdList.size() < vtiList.size())
  {
    vtkErrorMacro("Error : There is no enough vti files, please check your "
                  + std::string(this->DepthMapFile) + " and "
                  + std::string(this->KRTDFile));
    return 1;
  }

  ReconstructionData data0(vtiList[0].c_str(), krtdList[0].c_str());
  int* depthMapGrid = data0.GetDepthMap()->GetDimensions();

  // Initialize Cuda constant
  CudaInitialize(this->GridMatrix, pieceNbVoxels, pieceOrigin, this->GridSpacing,
                 this->RayPotentialThickness, this->RayPotentialRho, this->RayPotentialEta,
                 this->RayPotentialDelta, this->TilingSize, depthMapGrid, this);

  bool result = ProcessDepthMap<double>(vtiList, krtdList, this->ThresholdBestCost,
                                        outScalar);

  return 0;
}

//----------------------------------------------------------------------------
void vtkCudaReconstructionFilter::CreateGridMatrixFromInput()
{
  vtkNew<vtkMatrix4x4> gridMatrix;
  gridMatrix->Identity();

  // Fill matrix
  gridMatrix->SetElement(0, 0, this->GridVecX[0]);
  gridMatrix->SetElement(0, 1, this->GridVecX[1]);
  gridMatrix->SetElement(0, 2, this->GridVecX[2]);
  gridMatrix->SetElement(1, 0, this->GridVecY[0]);
  gridMatrix->SetElement(1, 1, this->GridVecY[1]);
  gridMatrix->SetElement(1, 2, this->GridVecY[2]);
  gridMatrix->SetElement(2, 0, this->GridVecZ[0]);
  gridMatrix->SetElement(2, 1, this->GridVecZ[1]);
  gridMatrix->SetElement(2, 2, this->GridVecZ[2]);

  this->SetGridMatrix(gridMatrix.Get());
}

//----------------------------------------------------------------------------
void vtkCudaReconstructionFilter::WriteSummaryFile()
{
  std::string path = std::string(this->DataFolder) + "/Reconstruction_summary"
                     + this->CurrentDate + ".txt";
  std::ofstream output(path.c_str());
  std::string file = "";

  output << "----------------------" << std::endl;
  output << "** OUTPUT GRID :" << std::endl;
  output << "----------------------" << std::endl;
  output << "--- Origin     : ( " << this->GridOrigin[0]<< ", " << this->GridOrigin[1]
         << ", " << this->GridOrigin[2] << " )" << std::endl;
  output << "--- End        : ( " << this->GridEnd[0] << ", " << this->GridEnd[1]
         << ", " << this->GridEnd[2] << " )" << std::endl;
  output << "--- Dimensions : ( " << this->GridNbVoxels[0] << ", "
         << this->GridNbVoxels[1] << ", " << this->GridNbVoxels[2] << " )" << std::endl;
  output << "--- Spacing    : ( " << this->GridSpacing[0] << ", "
         << this->GridSpacing[1] << ", " << this->GridSpacing[2] << " )" << std::endl;
  output << "--- Nb voxels  : " << this->GridNbVoxels[0] * this->GridNbVoxels[1]
         * this->GridNbVoxels[2] << std::endl;
  output << "--- Real volume size : ( " << (this->GridEnd[0] - this->GridOrigin[0])
         << ", " << (this->GridEnd[1] - this->GridOrigin[1]) << ", "
         << (this->GridEnd[2] - this->GridOrigin[2]) << ")" << std::endl;
  output << "--- Matrix :" << std::endl;
  output << this->GridVecX[0] << this->GridVecY[0] << this->GridVecZ[0] << std::endl;
  output << this->GridVecX[1] << this->GridVecY[1] << this->GridVecZ[1] << std::endl;
  output << this->GridVecX[2] << this->GridVecY[2] << this->GridVecZ[2] << std::endl;
  output << "----------------------" << std::endl;
  output << "** DEPTH MAP :" << std::endl;
  output << "----------------------" << std::endl;
  output << "--- Threshold for BestCost  : " << this->ThresholdBestCost << std::endl;
  output << std::endl;
  output << "----------------------" << std::endl;
  output << "** CUDA :" << std::endl;
  output << "----------------------" << std::endl;
  output << "--- Thickness ray potential : " << this->RayPotentialThickness << std::endl;
  output << "--- Rho ray potential :       " << this->RayPotentialRho << std::endl;
  output << "--- Eta ray potential :       " << this->RayPotentialEta << std::endl;
  output << "--- Delta ray potential :     " << this->RayPotentialDelta << std::endl;
  output << std::endl;
  output << "----------------------" << std::endl;
  output << "** TIME :" << std::endl;
  output << "----------------------" << std::endl;
  output << "--- Reconstruction : " << this->ExecutionTime << " s" << std::endl;
  output << std::endl;

  output << file;
  output.close();
}
