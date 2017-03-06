
include_directories(
  model
)

add_executable(NNCNNLSTMLabeler NNCNNLSTMLabeler.cpp)
target_link_libraries(NNCNNLSTMLabeler ${LIBS})
