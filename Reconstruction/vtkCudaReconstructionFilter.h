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


// .NAME vtkCudaReconstructionFilter -
// .SECTION Description
//

#ifndef vtkCudaReconstructionFilter_h
#define vtkCudaReconstructionFilter_h

#include "vtkFiltersCoreModule.h" // For export macro
#include "vtkImageAlgorithm.h"

class vtkDoubleArray;
class vtkExtentTranslator;
class vtkImageData;
class vtkMatrix3x3;
class vtkMatrix4x4;

class vtkCudaReconstructionFilter : public vtkImageAlgorithm
{
public:
  static vtkCudaReconstructionFilter *New();
  vtkTypeMacro(vtkCudaReconstructionFilter, vtkImageAlgorithm);

  // Description:
  // Define the thickness threshold (X axis) of the ray potential function when using cuda
  vtkSetMacro(RayPotentialThickness, double);
  // Description:
  // Define the rho value (Y axis) of the ray potential function when using cuda
  vtkSetMacro(RayPotentialRho, double);
  // Description :
  // Define a percentage of rho value for supposed empty voxel between filled voxel and camera
  vtkSetMacro(RayPotentialEta, double);
  // Description :
  // Define accepted voxel before/after filled voxel
  vtkSetMacro(RayPotentialDelta, double);
  // Description :
  // Threshold that will be applied on vtkImageData depth map according to BestCost
  vtkSetMacro(ThresholdBestCost, double);
  // Description :
  // Set whether or not to write the summary
  vtkSetMacro(WriteSummary, bool);
  // Description :
  // Set whether or not to force cubic voxels
  vtkSetMacro(ForceCubicVoxels, bool);
  // Description :
  // Sets a value which specifies the 3 grid properties used
  vtkSetMacro(GridPropertiesMode, int);
  // Description :
  // Sets a value which specifies the input depthmaps type
  vtkSetMacro(DepthmapType, int);
  // Description :
  // Set the execution date and time
  vtkSetStringMacro(CurrentDate);
  // Description :
  // Set the data folder
  vtkSetStringMacro(DataFolder);
  // Description :
  // Set the depth map file container
  vtkSetStringMacro(DepthMapFile);
  // Description
  // Entire path to access file that contains all krtd file names
  // krtd files have to be in the same folder as FilePathKRTD
  vtkSetStringMacro(FilePathKRTD);
  // Description :
  // Entire path to access file that contains all vti file names
  // vti files have to be in the same folder as FilePathVTI
  vtkSetStringMacro(FilePathVTI);
  // Description :
  // Set the KRTD file container
  vtkSetStringMacro(KRTDFile);
  // Description :
  // Define voxel tiling in each dimension
  vtkSetVector3Macro(TilingSize, int);
  // Description :
  // Define grid end 3D coordinates
  vtkSetVector3Macro(GridEnd, double);
  // Description :
  // Define grid origin 3D coordinates
  vtkSetVector3Macro(GridOrigin, double);
  // Description :
  // Define grid spacing in each dimension
  vtkSetVector3Macro(GridSpacing, double);
  // Description :
  // Define grid dimensions
  vtkSetVector3Macro(GridNbVoxels, int);
  // Description :
  // Define grid X direction
  vtkSetVector3Macro(GridVecX, double);
  // Description :
  // Define grid Y direction
  vtkSetVector3Macro(GridVecY, double);
  // Description :
  // Define grid Z direction
  vtkSetVector3Macro(GridVecZ, double);


  //Description :
  // Get the execution time when update is launch (in seconds)
  vtkGetMacro(ExecutionTime, double);

  // Description
  // Define the matrix transform to orientate the output volume
  // to the right axis
  void SetGridMatrix(vtkMatrix4x4 *gridMatrix);

  enum DepthmapType_t
  {
    STRUCTURE_FROM_MOTION, SPHERICAL
  };

protected:
  vtkCudaReconstructionFilter();
  ~vtkCudaReconstructionFilter();

  virtual int RequestData(vtkInformation *,
                          vtkInformationVector **,
                          vtkInformationVector *);
  virtual int RequestInformation(vtkInformation*,
                                 vtkInformationVector**,
                                 vtkInformationVector*);
  virtual int RequestUpdateExtent(vtkInformation*,
                                  vtkInformationVector**,
                                  vtkInformationVector*);

  bool AreVectorsOrthogonal();
  bool CheckArguments();
  int Compute(float *outScalar, unsigned short* outCount, int pieceNbVoxels[3],
              double pieceOrigin[3]);
  // Write a file with all parameters used
  void WriteSummaryFile();
  void CreateGridMatrixFromInput();


  bool ForceCubicVoxels;
  bool WriteSummary;
  const char* DataFolder;
  const char* DepthMapFile;
  //BTX
  const char* CurrentDate;
  const char* FilePathKRTD;
  const char* FilePathVTI;
  //ETX
  const char* KRTDFile;
  double GridEnd[3];
  double GridOrigin[3];
  double GridSpacing[3];
  double GridVecX[3];
  double GridVecY[3];
  double GridVecZ[3];
  double ExecutionTime;
  double RayPotentialDelta;
  double RayPotentialEta;
  double RayPotentialRho;
  double RayPotentialThickness;
  double ThresholdBestCost;
  //BTX
  enum GridPropertiesMode_t
  {
    DEFAULT_MODE, ORIG_END_NBVOX, ORIG_END_SPAC, ORIG_NBVOX_SPAC,
    END_NBVOX_SPAC
  };
  //ETX
  int DepthmapType;
  int GridPropertiesMode;
  int GridNbVoxels[3];
  int TilingSize[3];
  //BTX
  vtkMatrix4x4* GridMatrix;
  vtkExtentTranslator* ExtentTranslator;
  //ETX

private:
  vtkCudaReconstructionFilter(const vtkCudaReconstructionFilter&);  // Not implemented.
  void operator=(const vtkCudaReconstructionFilter&);  // Not implemented.
};

#endif
