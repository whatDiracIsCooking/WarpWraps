// api.cppm - Compile-time tests for gpumod.wrappers.tx's API shape

export module gpumod.test.wrappers.tx_api;

import std;
import gpumod.tx;
import gpumod.wrappers.tx;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// The tx wrapper is untyped -- no dispatch table, no type map -- so what it can
// get wrong is its API shape: the free-function signatures (a forward that
// dropped an argument or returned the wrong width would compile against the
// vendor but not against these), and ScopedRange's ownership contract. Both are
// constant expressions, so this is a static_assert-only module with no runtime
// half, like the fft/solver wrapper tests. The forwarding itself is checked by
// the wrapper compiling, and the gputx* -> vendor link is checked by
// test/gpu/tx.cppm.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::test {

using namespace wwr;

// ──────────────────────────────────────────────────────────────────────
// Free-function signatures
// ──────────────────────────────────────────────────────────────────────

static_assert(std::is_same_v<decltype(tx::mark), void(const char *)>);
static_assert(std::is_same_v<decltype(tx::range_push), int(const char *)>);
static_assert(std::is_same_v<decltype(tx::range_pop), int()>);
static_assert(std::is_same_v<decltype(tx::range_start), gputxRangeId_t(const char *)>);
static_assert(std::is_same_v<decltype(tx::range_stop), void(gputxRangeId_t)>);

// ──────────────────────────────────────────────────────────────────────
// ScopedRange ownership contract: constructed from a message, never copied,
// moved, or default-constructed (a push/pop pair is LIFO and scope-bound).
// ──────────────────────────────────────────────────────────────────────

static_assert(std::is_constructible_v<tx::ScopedRange, const char *>);
static_assert(!std::is_default_constructible_v<tx::ScopedRange>);
static_assert(!std::is_copy_constructible_v<tx::ScopedRange>);
static_assert(!std::is_move_constructible_v<tx::ScopedRange>);
static_assert(!std::is_copy_assignable_v<tx::ScopedRange>);
static_assert(!std::is_move_assignable_v<tx::ScopedRange>);
static_assert(std::is_nothrow_destructible_v<tx::ScopedRange>);

} // namespace wwr::test
