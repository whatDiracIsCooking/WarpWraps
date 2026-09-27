# GPU BLAS extension - type-safe wrappers over wwrblas* (cuBLAS or hipBLAS,
# per WWR_GPU_BACKEND). dispatch_macros.h sits next to the sources and is
# #include'd same-dir, so it needs no include directory; the shared
# dispatch_sdcz.h it pulls in is reached through the src/ root that
# wwr.wrappers.common exports (linked PUBLIC below).
wwr_add_cxx_module_library(
  NAME
  wwr.wrappers.blas
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  type_traits.cppm
  level_1.cppm
  level_2.cppm
  level_3.cppm
  extension.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.blas
  wwr.complex
  wwr.fp16
  wwr.bf16
  wwr.wrappers.common
  IMPORT_STD
)

# Catch a wrapper whose IntT=int64_t branch calls the 32-bit entry point (or
# otherwise narrows a 64-bit size/index to 32 bits) at compile time instead of
# silently truncating at n > 2^31. See issue #135.
target_compile_options(wwr.wrappers.blas PRIVATE -Werror=shorten-64-to-32)
