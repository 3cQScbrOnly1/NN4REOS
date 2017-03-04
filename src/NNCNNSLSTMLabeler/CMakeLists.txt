
include_directories(
  model
)

add_executable(NNCNNSLSTMLabeler NNCNNLSTMLabeler.cpp)
target_link_libraries(NNCNNSLSTMLabeler ${LIBS})
