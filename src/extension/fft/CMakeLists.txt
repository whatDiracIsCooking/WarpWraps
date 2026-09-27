# The error-handling and RAII-plan layer for GPU FFT. The execution wrappers
# that once shared this module were split out to gpumod.wrappers.fft; what is
# left here is fft_error + fft_plan, aggregated by interface.cppm.
gpumod_add_cxx_module_library(
  NAME
  gpumod.extension.fft
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  fft_error.cppm
  fft_plan.cppm
  LINK_PUBLIC
  gpumod.fft
  gpumod.extension.common
  gpumod.extension.common.handle
  IMPORT_STD
)
