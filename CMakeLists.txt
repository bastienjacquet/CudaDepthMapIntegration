cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
   vtkIOImage
   REQUIRED)

find_package(CUDA REQUIRED)
# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_20)

include_directories(${CMAKE_SOURCE_DIR}/Sources)

add_subdirectory( Sources )
add_subdirectory( Reconstruction )
add_subdirectory( Coloration )
