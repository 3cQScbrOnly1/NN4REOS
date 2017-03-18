
include_directories(
  model
)

add_executable(NNBiGRNNLabeler-v2 NNRNNLabeler.cpp)
target_link_libraries(NNBiGRNNLabeler-v2 ${LIBS})
