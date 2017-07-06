project( Reconstruction )

include( ${VTK_USE_FILE} )

# Allow to debug GPU function
cuda_compile( CUDA_COMPILE_O CudaReconstruction.cu OPTIONS -G -std=c++11 -D_MWAITXINTRIN_H_INCLUDED)

# Specify target & source files to compile
cuda_add_executable(
    ${PROJECT_NAME}
    main.cxx
    vtkCudaReconstructionFilter.h
    vtkCudaReconstructionFilter.cxx
    ${CUDA_COMPILE_O})

target_link_libraries(${PROJECT_NAME} ReconstructionLib ${VTK_LIBRARIES})
