
include_directories(
  model
)

add_executable(NNBiRNNLabeler-v2 NNRNNLabeler.cpp)
target_link_libraries(NNBiRNNLabeler-v2 ${LIBS})
