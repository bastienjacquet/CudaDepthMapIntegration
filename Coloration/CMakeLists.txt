project( Coloration )

include( ${VTK_USE_FILE} )

set( Coloration_SRCS
  main.cxx
  MeshColoration.cxx
  MeshColoration.h
)

add_executable( ${PROJECT_NAME} ${Coloration_SRCS} )
target_link_libraries(${PROJECT_NAME} ReconstructionLib ${VTK_LIBRARIES})
