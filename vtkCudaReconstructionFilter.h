
// .NAME vtkCudaReconstructionFilter -
// .SECTION Description
//

#ifndef vtkCudaReconstructionFilter_h
#define vtkCudaReconstructionFilter_h

#include "vtkFiltersCoreModule.h" // For export macro
#include "vtkDataSetAlgorithm.h"

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
  // Specify the depth map transform matrix.
  void SetDepthMapMatrix(vtkMatrix4x4 *depthMapMatrix);

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

  vtkMatrix4x4 *DepthMapMatrix;

private:
  vtkCudaReconstructionFilter(const vtkCudaReconstructionFilter&);  // Not implemented.
  void operator=(const vtkCudaReconstructionFilter&);  // Not implemented.

//ETX
};

#endif
