
project(cudareconstruction)
cmake_minimum_required(VERSION 2.8)

find_package(VTK
   COMPONENTS
   vtkCommonCore
   vtkCommonDataModel
   vtkCommonExecutionModel
   vtkFiltersCore
   vtkFiltersGeneral
   vtkIOLegacy
   vtkIOXML
   REQUIRED)
include(${VTK_USE_FILE})

find_package(CUDA REQUIRED)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_20)

# Allow to debug GPU function
cuda_compile( CUDA_COMPILE_O CudaReconstruction.cu OPTIONS -G)

# Specify target & source files to compile
cuda_add_executable(
    ${PROJECT_NAME}
    main.cxx
    vtkCudaReconstructionFilter.h
    vtkCudaReconstructionFilter.cxx
    ReconstructionData.h
    ReconstructionData.cxx
    ${CUDA_COMPILE_O})

target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
