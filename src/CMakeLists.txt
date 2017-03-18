
include_directories(
  basic
)

add_subdirectory(NNCNNLabeler)
add_subdirectory(NNRNNLabeler)
add_subdirectory(NNGRNNLabeler)
add_subdirectory(NNLSTMLabeler)


add_subdirectory(NNBiRNNLabeler)
add_subdirectory(NNBiRNNLabeler-v2)
add_subdirectory(NNBiGRNNLabeler)
add_subdirectory(NNBiGRNNLabeler-v2)
add_subdirectory(NNBiLSTMLabeler)
add_subdirectory(NNBiLSTMLabeler-v2)