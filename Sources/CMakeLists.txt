set(Reconstruction_SRCS
    ReconstructionData.h
    ReconstructionData.cxx
    Helper.h)

add_library( ReconstructionDataLib SHARED ${Reconstruction_SRCS} )

target_link_libraries( ReconstructionDataLib ${VTK_LIBRARIES} )