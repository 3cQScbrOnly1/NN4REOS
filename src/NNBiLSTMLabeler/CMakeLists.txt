
include_directories(
  model
)

add_executable(NNBiLSTMLabeler NNRNNLabeler.cpp)
target_link_libraries(NNBiLSTMLabeler ${LIBS})
