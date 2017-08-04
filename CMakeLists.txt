cmake_minimum_required( VERSION 3.0 )
project( Reconstruction )

set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

if (NOT ParaView_BINARY_DIR)
  find_package(ParaView REQUIRED)
  include(${PARAVIEW_USE_FILE})
endif()

include( ParaViewPlugins )

find_package( CUDA REQUIRED )

# Pass options to NVCC : allow relocation
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -fPIC
    -std=c++11 -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED
    --gpu-architecture sm_20)

include_directories( ${CMAKE_SOURCE_DIR}/Sources )

add_subdirectory( Sources )
add_subdirectory( Reconstruction )