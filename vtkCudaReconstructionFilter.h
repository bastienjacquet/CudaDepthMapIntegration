
// .NAME vtkCudaReconstructionFilter -
// .SECTION Description
//

#ifndef vtkCudaReconstructionFilter_h
#define vtkCudaReconstructionFilter_h

#include "vtkFiltersCoreModule.h" // For export macro
#include "vtkImageAlgorithm.h"

#include "ReconstructionData.h"

#include <vector>

class vtkDoubleArray;
class vtkImageData;
class vtkMatrix3x3;
class vtkMatrix4x4;

class vtkCudaReconstructionFilter : public vtkImageAlgorithm
{
public:
  static vtkCudaReconstructionFilter *New();
  vtkTypeMacro(vtkCudaReconstructionFilter,vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  void UseCudaOn();
  void UseCudaOff();

  // Description:
  // Specify the depth map.
  void SetDepthMap(vtkImageData *depthMap);

  // Description:
  // Specify the depth map transform matrix: K, R, T.
  void SetDepthMapMatrixK(vtkMatrix3x3 *depthMapMatrixK);
  void SetDepthMapMatrixTR(vtkMatrix4x4 *depthMapMatrixTR);
  void SetGridMatrix(vtkMatrix4x4 *gridMatrix);
  // Description:
  // List all data with depthMap and KRT matrix
  void SetDataList(std::vector<ReconstructionData*> list);

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

  static int ComputeWithoutCuda(
    vtkMatrix4x4 *gridMatrix, double gridOrig[3], int gridDims[3], double gridSpacing[3],
    vtkImageData* depthMap, vtkMatrix3x3 *depthMapMatrixK, vtkMatrix4x4 *depthMapMatrixTR,
    vtkDoubleArray* outScalar);
  static void FunctionCumul(double diff, double& val);

  static int ComputeWithCuda(
    vtkMatrix4x4 *gridMatrix, double gridOrig[3], int gridDims[3], double gridSpacing[3],
    vtkImageData* depthMap, vtkMatrix3x3 *depthMapMatrixK, vtkMatrix4x4 *depthMapMatrixTR,
    vtkDoubleArray* outScalar);

  std::vector<ReconstructionData*> DataList;
  vtkMatrix4x4 *GridMatrix;
  bool useCuda;

private:
  vtkCudaReconstructionFilter(const vtkCudaReconstructionFilter&);  // Not implemented.
  void operator=(const vtkCudaReconstructionFilter&);  // Not implemented.

//ETX
};

#endif
