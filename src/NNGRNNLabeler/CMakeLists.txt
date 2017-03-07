
include_directories(
  model
)

add_executable(NNGRNNLabeler NNRNNLabeler.cpp)
target_link_libraries(NNGRNNLabeler ${LIBS})
