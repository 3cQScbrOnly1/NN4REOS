
include_directories(
  model
)

add_executable(NNCNNMaxReluLabeler NNCNNLabeler.cpp)
target_link_libraries(NNCNNMaxReluLabeler ${LIBS})
