
include_directories(
  model
)

add_executable(NNRNNLabeler NNRNNLabeler.cpp)
target_link_libraries(NNRNNLabeler ${LIBS})
