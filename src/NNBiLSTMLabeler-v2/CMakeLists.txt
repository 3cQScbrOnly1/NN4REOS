
include_directories(
  model
)

add_executable(NNBiLSTMLabeler-v2 NNRNNLabeler.cpp)
target_link_libraries(NNBiLSTMLabeler-v2 ${LIBS})
