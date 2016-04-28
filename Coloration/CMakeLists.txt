project( Coloration )

include( ${VTK_USE_FILE} )

set( Coloration_SRCS
  main.cxx
)

add_executable( ${PROJECT_NAME} ${Coloration_SRCS} )
target_link_libraries(${PROJECT_NAME} ReconstructionLib ${VTK_LIBRARIES})
