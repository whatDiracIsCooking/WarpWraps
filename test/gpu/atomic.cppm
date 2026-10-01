// atomic.cppm - Compile-time test for wwr.atomic
//
// wwr.atomic is the thinnest wwr* module here: it exports two project-owned
// enums (wwrMemoryOrder, wwrThreadScope) and nothing else. The scoped atomic
// *operations* are __device__-only and live in atomic.h's device section, pinned
// by atomic.cu -- a host-compiled module test cannot name them. So unlike every
// other test here there is no backend entity to compare against (no vendor enum
// behind these -- the step past vector_types, which at least aliases a vendor
// global type), and gpu_check_macros.h's WWR_SAME_* macros do not apply.
//
// What this CAN break, and so is what it asserts: the two enums are re-exported
// intact across `import wwr.atomic` -- each is a scoped enum and every
// enumerator is nameable as a constant through the import. A dropped
// `using wwr::...` in atomic.cppm, or a renamed enumerator, reddens this.

export module wwr.test.gpu.atomic;

import std;
import wwr.atomic;

namespace wwr::test {

using wwr::wwrMemoryOrder;
using wwr::wwrThreadScope;

// Both arrive as scoped enums, not decayed to int or a plain enum.
static_assert(std::is_scoped_enum_v<wwrMemoryOrder>, "wwrMemoryOrder is not a scoped enum");
static_assert(std::is_scoped_enum_v<wwrThreadScope>, "wwrThreadScope is not a scoped enum");

// Every enumerator names a distinct constant through the import -- naming each in
// a constant context is the assertion that the re-export carried all of them; a
// dropped or renamed one fails to compile here.
constexpr wwrMemoryOrder kOrders[] = {
    wwrMemoryOrder::relaxed, wwrMemoryOrder::acquire, wwrMemoryOrder::release,
    wwrMemoryOrder::acq_rel, wwrMemoryOrder::seq_cst,
};
constexpr wwrThreadScope kScopes[] = {
    wwrThreadScope::thread,
    wwrThreadScope::block,
    wwrThreadScope::device,
    wwrThreadScope::system,
};

static_assert(std::size(kOrders) == 5, "wwrMemoryOrder enumerator set changed");
static_assert(std::size(kScopes) == 4, "wwrThreadScope enumerator set changed");

// The enumerators are distinct -- a spot check that they are not all folded to
// one value by a broken definition.
static_assert(wwrMemoryOrder::relaxed != wwrMemoryOrder::seq_cst, "wwrMemoryOrder is degenerate");
static_assert(wwrThreadScope::thread != wwrThreadScope::system, "wwrThreadScope is degenerate");

} // namespace wwr::test
