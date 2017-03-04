
include_directories(
  model
)

add_executable(NNCNNSparseLabeler NNCNNLabeler.cpp)
target_link_libraries(NNCNNSparseLabeler ${LIBS})
