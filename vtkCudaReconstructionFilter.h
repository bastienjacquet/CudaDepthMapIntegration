
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

  // Description:
  // Define the thickness threshold (X axis) of the ray potential function when using cuda
  vtkSetMacro(RayPotentialThickness, double);
  // Description:
  // Define the rho value (Y axis) of the ray potential function when using cuda
  vtkSetMacro(RayPotentialRho, double);
  // Description :
  // Define if algorithm is launched on the GPU with cuda (or not)
  vtkSetMacro(UseCuda, bool);

  // Description :
  // The algorithm will be launched with cuda on GPU (fast)
  void UseCudaOn();
  // Description :
  // The algorithm will be launched withou cuda on CPU (slow)
  void UseCudaOff();

  // Description:
  // Specify the depth map.
  void SetDepthMap(vtkImageData *depthMap);

  // Description:
  // Specify the depth map transform matrix: K.
  void SetDepthMapMatrixK(vtkMatrix3x3 *depthMapMatrixK);
  // Description:
  // Specify the depth map transform matrix: R, T.
  void SetDepthMapMatrixTR(vtkMatrix4x4 *depthMapMatrixTR);
  // Description
  // Define the matrix transform to orientate the output volume
  // to the right axis
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

  int ComputeWithoutCuda(
    vtkMatrix4x4 *gridMatrix, double gridOrig[3], int gridDims[3], double gridSpacing[3],
    vtkImageData* depthMap, vtkMatrix3x3 *depthMapMatrixK, vtkMatrix4x4 *depthMapMatrixTR,
    vtkDoubleArray* outScalar);

  void RayPotential(double realDistance, double depthMapDistance, double& val);

  std::vector<ReconstructionData*> DataList;
  vtkMatrix4x4 *GridMatrix;
  double RayPotentialRho;
  double RayPotentialThickness;
  bool UseCuda;

private:
  vtkCudaReconstructionFilter(const vtkCudaReconstructionFilter&);  // Not implemented.
  void operator=(const vtkCudaReconstructionFilter&);  // Not implemented.

//ETX
};

#endif
