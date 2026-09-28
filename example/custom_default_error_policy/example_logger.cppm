// example_logger.cppm -- a user's OWN module, the kind of thing you might wrap a
// real logger (spdlog, a project sink, ...) in.
//
// This exists to make one point: the custom default-error-policy partition
// `import`s this module. The old header customization point could not -- a
// header pulled into a global module fragment cannot `import` anything, and this
// tree's convention is that imports live in the module purview, never the GMF
// (see src/gpu_backend.h / src/complex.cppm). Swapping the whole partition file
// restores that: the replacement is a real module unit and imports normally.
//
// It carries a little state (error_count) precisely to show this is a genuine
// module with linkage, not header-inlined boilerplate.

export module wwr.example.error_logger;

import std;

export namespace wwr::example {

/// @brief Running count of errors this sink has seen (real module-scoped state).
inline int &error_count() {
  static int n = 0;
  return n;
}

/// @brief Route one GPU-error message to stderr through this "sink".
inline void log_gpu_error(std::string_view message) {
  std::println(std::cerr, "[wwr.example.error_logger] #{}: {}", ++error_count(), message);
}

} // namespace wwr::example
