set(Reconstruction_SRCS
  ReconstructionData.h
  ReconstructionData.cxx
  Helper.h
)

include( ${VTK_USE_FILE} )
add_library( ReconstructionLib ${Reconstruction_SRCS} )
target_link_libraries( ReconstructionLib ${VTK_LIBRARIES} )