/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.tx
 *
 * A single RAII guard, ScopedRange, over the wwrtx* nested-range markers
 * (wwr.tx -> NVTX or rocTX, per WWR_GPU_BACKEND): it pushes a range on
 * construction and pops it on destruction, so a range cannot be left open on an
 * early return or a throw. The wwrtx* layer is already a usable C API, so the
 * push/pop pairing is the only thing worth wrapping -- ScopedRange calls
 * wwrtxRangePushA / wwrtxRangePop directly, and nothing else here re-exports the
 * raw markers.
 *
 * Messages are const char* -- NVTX and rocTX both take a null-terminated
 * string, and this layer forwards it unchanged rather than copy a
 * std::string_view to guarantee termination.
 *
 * Usage:
 *   import wwr.extension.tx;
 *
 *   {
 *     wwr::extension::ScopedRange region{"phase 1"};
 *     // ... work ...
 *   } // range popped here, even on early return or throw
 */

export module wwr.extension.tx;

import wwr.tx;
import std;

export namespace wwr::extension {

/**
 * @brief Scoped nested range: wwrtxRangePushA on construction, wwrtxRangePop at scope exit.
 *
 * Non-copyable and non-movable, like std::lock_guard: a push/pop pair is strictly
 * LIFO and bound to this scope, so there is no meaningful way to relocate one.
 */
class ScopedRange {
public:
  explicit ScopedRange(const char *message) { wwrtxRangePushA(message); }
  ~ScopedRange() { wwrtxRangePop(); }

  ScopedRange(const ScopedRange &) = delete;
  ScopedRange &operator=(const ScopedRange &) = delete;
  ScopedRange(ScopedRange &&) = delete;
  ScopedRange &operator=(ScopedRange &&) = delete;
};

} // namespace wwr::extension
