// custom_default_error_policy.cppm -- a custom DEFAULT error policy for wwr,
// supplied as a WHOLE MODULE PARTITION.
//
// Point the tree's cache variables at this file when you configure:
//   -DWWR_DEFAULT_ERROR_POLICY_MODULE=<this dir>/custom_default_error_policy.cppm
//   -DWWR_DEFAULT_ERROR_POLICY_LINK=wwr.example.error_logger
// and wwr::extension::DefaultErrorPolicy<T> BECOMES the type below, build-wide --
// every default-policy slot in the extension layer picks it up (the argless
// gpu_check, and the P_create / P_alloc / P_free / P_destroy template-argument
// defaults on the RAII buffer / handle / stream / plan wrappers) with no change
// at any of those call sites.
//
// This is a real module unit, so -- unlike the old header customization point --
// its purview imports whatever it needs:
//   * import :error_code;              wwr's own error_name / error_string /
//                                      success_code, which a global-module-fragment
//                                      header could not reach;
//   * import wwr.example.error_logger; a module of your own (the headline case).
// Imports go after the module declaration, exactly like every other module in
// this tree.
//
// CONTRACT -- handle_error MUST be noexcept. The destruction-slot policies
// (P_free / P_destroy) are constrained by nothrow_error_policy: they run from
// destructors, where a throwing handler would std::terminate. A custom default
// that threw would therefore fail to compile those slots. This one logs and
// RETURNS instead of throwing or aborting -- a "record the failure, don't crash"
// policy, the opposite end from the built-in default.

export module wwr.extension.common.error_handling:default_error_policy;

import std;
import :error_code;              // wwr's own success_code / error_name / error_string
import wwr.example.error_logger; // a user module -- the capability the GMF header lacked

export namespace wwr::extension {

// The customization point wwr resolves DefaultErrorPolicy to when the partition
// is swapped. Generic over the error code type T (wwrError_t and each library
// status type), exactly like the built-in default it replaces.
template <typename T>
class DefaultErrorPolicy {
public:
  using error_type = T;

  void handle_error(const T error, std::source_location location) noexcept {
    if (error != success_code<T>()) {
      wwr::example::log_gpu_error(
          std::format("GPU error {} ({}) at {}:{} in {} -- continuing (no abort)",
                      error_name(error), error_string(error), location.file_name(),
                      location.line(), location.function_name()));
    }
  }
};

} // namespace wwr::extension
