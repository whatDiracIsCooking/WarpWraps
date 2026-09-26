# GPU FFT extension - type-safe wrappers over gpufft* (cuFFT or hipFFT, per
# GPUMOD_GPU_BACKEND). dispatch_macros.h sits next to the sources and is
# #include'd by relative path, so it needs no include directory.
gpumod_add_cxx_module_library(
  NAME
  gpumod.wrappers.fft
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  type_traits.cppm
  exec.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  gpumod.fft
  gpumod.wrappers.common
  IMPORT_STD
)
