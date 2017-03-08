
include_directories(
  model
)

add_executable(NNBiGRNNLabeler NNRNNLabeler.cpp)
target_link_libraries(NNBiGRNNLabeler ${LIBS})
