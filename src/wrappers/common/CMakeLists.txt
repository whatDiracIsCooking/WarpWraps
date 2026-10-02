# C++ Module Library
#
# Backend-neutral: imports no raw backend module (never src/cuda or src/hip).
#
# INCLUDE_DIRS_PUBLIC roots src/ so the shared, non-module headers are reached
# by the owned-header spelling -- dispatch_sdcz.h as
# `#include "wrappers/common/dispatch_sdcz.h"`, and the fp_types.cppm global
# module fragment's fp_types.h -> complex.h / fp16.h / bf16.h as bare src/-root
# names -- not a `../common` relative climb, from any module that links this
# target (blas, sparse, solver). BUILD_INTERFACE is src/; INSTALL_INTERFACE is
# include/wwr, the mirror wwr_install.cmake ships src/'s shape under -- so the
# one spelling resolves both in-tree and for an installed consumer recompiling
# those module units. This is the root the install file's rule-3 comment already
# documents.
#
# wwr::backend is linked PRIVATE for the WWR_GPU_BACKEND_* define only:
# fp_types.cppm's GMF includes fp_types.h, whose complex.h / fp16.h / bf16.h
# reach selected_backend.h, and a host compile with no __CUDACC__/__HIP__ needs
# the define to pick the backend's types. It does NOT license a backend #if
# here -- the #if lives in those src/ headers (src/CMakeLists.txt). Re-added
# under $<INSTALL_INTERFACE:> below so an installed consumer recompiling
# fp_types.cppm gets it too, exactly as wwr.wrappers.thrust.scan does.
wwr_add_cxx_module_library(
  NAME
  wwr.wrappers.common
  PRIMARY_INTERFACE
  interface.cppm
  INCLUDE_DIRS_PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
  PARTITIONS
  fp_types.cppm
  int_types.cppm
  LINK_PUBLIC
  wwr::complex
  wwr::fp16
  wwr::bf16
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

# The WWR_GPU_BACKEND_* define is PRIVATE to the in-tree build (above) but must
# reach an installed consumer, since that consumer is the one recompiling
# fp_types.cppm's module unit. The src/ include root is already carried by
# INCLUDE_DIRS_PUBLIC's $<INSTALL_INTERFACE:>, so only the define is re-added
# here -- see wwr.wrappers.thrust.scan and the comment at the end of
# src/CMakeLists.txt.
target_link_libraries(
  wwr.wrappers.common PUBLIC $<INSTALL_INTERFACE:wwr_backend>
)
