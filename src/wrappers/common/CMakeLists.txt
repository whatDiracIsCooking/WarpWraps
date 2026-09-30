# C++ Module Library
#
# Backend-neutral: imports only std and src, never src/cuda or src/hip.
#
# INCLUDE_DIRS_PUBLIC roots src/ so the shared, non-module header
# dispatch_sdcz.h is reached by the owned-header spelling
# `#include "wrappers/common/dispatch_sdcz.h"` -- not a `../common` relative
# climb -- from any module that links this target (blas, sparse, solver).
# BUILD_INTERFACE is src/; INSTALL_INTERFACE is include/wwr, the mirror
# wwr_install.cmake ships src/'s shape under -- so the one spelling resolves
# both in-tree and for an installed consumer recompiling those module units.
# This is the root the install file's rule-3 comment already documents.
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
  IMPORT_STD
)
