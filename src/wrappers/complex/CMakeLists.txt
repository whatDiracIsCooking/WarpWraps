# C++ Module Library
#
# Backend-neutral: the ergonomic wwr::complex<T> value type over the gpu*
# complex layer. Imports only std, wwr.complex (the gpu* types, make_wwr*Complex
# and the wwrCreal*/wwrCimag* accessors) and wwr.wrappers.common (real_fp,
# RealToComplexType) -- never src/cuda or src/hip.
#
# The device-compile mirror device_complex.cuh is header-only and rides
# wwr.device's src/ include root (like wrappers/math/math.cuh), so it needs no
# target here; a kernel TU reaches it as "wrappers/complex/device_complex.cuh".
wwr_add_cxx_module_library(
  NAME
  wwr.wrappers.complex
  PRIMARY_INTERFACE
  complex.cppm
  LINK_PUBLIC
  wwr.complex
  wwr.wrappers.common
  IMPORT_STD
)
