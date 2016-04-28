cmake_minimum_required(VERSION 3.0)

project( Reconstruction )

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

find_package(CUDA REQUIRED)
# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_20)

include_directories(${CMAKE_SOURCE_DIR}/Sources)

add_subdirectory( Sources )
add_subdirectory( Reconstruction )
add_subdirectory( Coloration )