
include_directories(
  model
)

add_executable(NNBiRNNLabeler NNRNNLabeler.cpp)
target_link_libraries(NNBiRNNLabeler ${LIBS})
