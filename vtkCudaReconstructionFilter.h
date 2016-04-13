
// .NAME vtkCudaReconstructionFilter -
// .SECTION Description
//

#ifndef vtkCudaReconstructionFilter_h
#define vtkCudaReconstructionFilter_h

#include "vtkFiltersCoreModule.h" // For export macro
#include "vtkDataSetAlgorithm.h"

class vtkMatrix3x3;
class vtkMatrix4x4;

class vtkCudaReconstructionFilter : public vtkDataSetAlgorithm
{
public:
  static vtkCudaReconstructionFilter *New();
  vtkTypeMacro(vtkCudaReconstructionFilter,vtkDataSetAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Specify the depth map.
  void SetDepthMap(vtkDataObject *depthMap);
  vtkDataObject *GetDepthMap();

  // Description:
  // Specify the depth map transform matrix: K, R, T.
  void SetDepthMapMatrixK(vtkMatrix3x3 *depthMapMatrixK);
  void SetDepthMapMatrixTR(vtkMatrix4x4 *depthMapMatrixTR);

  // Description:
  // Specify the grid directions.
  vtkSetVector3Macro(GridVecX, double);
  vtkSetVector3Macro(GridVecY, double);
  vtkSetVector3Macro(GridVecZ, double);

//BTX
protected:
  vtkCudaReconstructionFilter();
  ~vtkCudaReconstructionFilter();

  virtual int RequestData(vtkInformation *, vtkInformationVector **,
    vtkInformationVector *);
  virtual int RequestInformation(vtkInformation *, vtkInformationVector **,
    vtkInformationVector *);
  virtual int RequestUpdateExtent(vtkInformation *, vtkInformationVector **,
    vtkInformationVector *);

  vtkMatrix3x3 *DepthMapMatrixK;
  vtkMatrix4x4 *DepthMapMatrixTR;
  double GridVecX[3];
  double GridVecY[3];
  double GridVecZ[3];

private:
  vtkCudaReconstructionFilter(const vtkCudaReconstructionFilter&);  // Not implemented.
  void operator=(const vtkCudaReconstructionFilter&);  // Not implemented.

//ETX
};

#endif
