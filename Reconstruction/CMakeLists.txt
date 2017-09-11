
# Allow to debug GPU function
cuda_compile( CUDA_COMPILE_O CudaReconstruction.cu OPTIONS -G )

cuda_add_library( CudaReconstructionLib SHARED ${CUDA_COMPILE_O} )

ADD_PARAVIEW_PLUGIN(CudaReconstructionFilter "1.0"
  SERVER_MANAGER_XML CudaReconstructionFilter.xml
  SERVER_MANAGER_SOURCES vtkCudaReconstructionFilter.cxx)

target_link_libraries( CudaReconstructionLib ReconstructionDataLib )
target_link_libraries( CudaReconstructionFilter PRIVATE ReconstructionDataLib CudaReconstructionLib ${VTK_LIBRARIES} )
