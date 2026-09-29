// gpu_check_macros.h - identity checks for the wwr* layer compile-time tests
//
// Each src name must be exactly the backend name it stands for: the same
// type, the same constant (type and value), the same function. The expected
// backend name is spelled out in full at every use, rather than derived with
// the wwr* layer's own WWR_SELECT/prefix-pasting macros, so a mistake in those macros
// cannot be mirrored here and pass.

#pragma once

#include "test/shared/link_check.h"

// gpu is the same type as backend.
#define WWR_SAME_TYPE(gpu, backend)                                                             \
  static_assert(std::is_same_v<gpu, backend>, #gpu " is not " #backend);

// gpu has backend's (cv-stripped) type and value.
#define WWR_SAME_VALUE(gpu, backend)                                                            \
  static_assert(                                                                                   \
      std::is_same_v<std::remove_cvref_t<decltype(gpu)>, std::remove_cvref_t<decltype(backend)>>,  \
      #gpu " does not have the type of " #backend);                                                \
  static_assert(gpu == backend, #gpu " != " #backend);

// gpu refers to backend itself, and the symbol resolves at link time.
#define WWR_SAME_FUNCTION(gpu, backend)                                                         \
  static_assert(&gpu == &backend, #gpu " is not " #backend);                                       \
  WWR_LINK_CHECK(gpu)
