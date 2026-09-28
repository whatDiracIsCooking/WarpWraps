// tx.cppm - Compile-time tests for wwr.extension.tx's API shape

export module wwr.test.extension.tx;

import std;
import wwr.tx;
import wwr.extension.tx;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// The tx guard is untyped -- no dispatch table, no type map -- so what it can
// get wrong is ScopedRange's ownership contract. That is a set of constant
// expressions, so this is a static_assert-only module with no runtime half,
// like the fft/solver wrapper tests. The wwrtxRangePushA/wwrtxRangePop calls
// themselves are checked by the module compiling, and the wwrtx* -> vendor link
// is checked by test/gpu/tx.cppm.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::test {

using namespace wwr;

// ──────────────────────────────────────────────────────────────────────
// ScopedRange ownership contract: constructed from a message, never copied,
// moved, or default-constructed (a push/pop pair is LIFO and scope-bound).
// ──────────────────────────────────────────────────────────────────────

static_assert(std::is_constructible_v<extension::ScopedRange, const char *>);
static_assert(!std::is_default_constructible_v<extension::ScopedRange>);
static_assert(!std::is_copy_constructible_v<extension::ScopedRange>);
static_assert(!std::is_move_constructible_v<extension::ScopedRange>);
static_assert(!std::is_copy_assignable_v<extension::ScopedRange>);
static_assert(!std::is_move_assignable_v<extension::ScopedRange>);
static_assert(std::is_nothrow_destructible_v<extension::ScopedRange>);

} // namespace wwr::test
