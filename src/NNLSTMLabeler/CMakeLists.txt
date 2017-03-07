
include_directories(
  model
)

add_executable(NNLSTMLabeler NNRNNLabeler.cpp)
target_link_libraries(NNLSTMLabeler ${LIBS})
