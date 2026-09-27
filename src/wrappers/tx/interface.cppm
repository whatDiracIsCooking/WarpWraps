/**
 * @file interface.cppm
 * @brief Primary interface for gpumod.wrappers.tx
 *
 * Type-safe, backend-neutral profiler annotations over the gputx* marker/range
 * names (gpumod.tx -> NVTX or rocTX, per WWR_GPU_BACKEND). It adds the C++
 * ergonomics the raw gputx* layer does not: named free functions and a
 * ScopedRange RAII guard that pushes a nested range on construction and pops it
 * on destruction, so a range cannot be left open on an early return or a throw.
 *
 * Unlike the blas/solver/fft/sparse wrappers there is no typed dispatch -- the
 * tools API is untyped -- so this is a single interface unit with no partitions,
 * instantiations, or dispatch table.
 *
 * Names live in wwr::tx, not bare wwr: mark / range_push / range_pop are
 * generic enough to want a scope of their own, and it reads as a cohesive API
 * (wwr::tx::ScopedRange). Messages are const char* -- NVTX and rocTX both take
 * a null-terminated string, and this layer forwards it unchanged rather than
 * copy a std::string_view to guarantee termination.
 *
 * Usage:
 *   import gpumod.wrappers.tx;
 *
 *   wwr::tx::mark("checkpoint");
 *   {
 *     wwr::tx::ScopedRange region{"phase 1"};
 *     // ... work ...
 *   } // range popped here, even on early return or throw
 */

module;

export module gpumod.wrappers.tx;

import gpumod.tx;
import std;

export namespace wwr::tx {

// ========================================================================
// Markers -- an instantaneous event at a point in time
// ========================================================================

/// @brief Record an instantaneous marker carrying an ASCII message.
void mark(const char *message) { gputxMarkA(message); }

// ========================================================================
// Ranges -- nested (stack) push/pop on the calling thread
// ========================================================================

/// @brief Begin a nested range on the calling thread.
/// @return the zero-based nesting depth begun, or a negative value on error.
int range_push(const char *message) { return gputxRangePushA(message); }

/// @brief End the innermost nested range on the calling thread.
/// @return the depth of the range ended, or a negative value on error.
int range_pop() { return gputxRangePop(); }

// ========================================================================
// Ranges -- process-wide asynchronous start/stop
// ========================================================================

/// @brief Begin an asynchronous range; pass the returned id to range_stop.
gputxRangeId_t range_start(const char *message) { return gputxRangeStartA(message); }

/// @brief End the asynchronous range identified by id.
void range_stop(gputxRangeId_t id) { gputxRangeStop(id); }

// ========================================================================
// RAII nested-range guard
// ========================================================================

/**
 * @brief Scoped nested range: range_push on construction, range_pop at scope exit.
 *
 * Non-copyable and non-movable, like std::lock_guard: a push/pop pair is strictly
 * LIFO and bound to this scope, so there is no meaningful way to relocate one.
 */
class ScopedRange {
public:
  explicit ScopedRange(const char *message) { range_push(message); }
  ~ScopedRange() { range_pop(); }

  ScopedRange(const ScopedRange &) = delete;
  ScopedRange &operator=(const ScopedRange &) = delete;
  ScopedRange(ScopedRange &&) = delete;
  ScopedRange &operator=(ScopedRange &&) = delete;
};

} // namespace wwr::tx
