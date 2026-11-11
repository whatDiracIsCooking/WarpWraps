"""The `tx` (NVTX / rocTX) profiler-annotation surface, HAND-WRITTEN.

tx is the third hand-written surface after complex and comp, and -- like them --
because it has no vendor `decl` to source a signature from in the usual way, but
for a subtler reason worth recording (it is the #359 feasibility question).

NVTX v3 is HEADER-ONLY `static inline` (#292): nvtxMarkA and friends have
INTERNAL linkage, so they cannot be re-exported with `using ::` and there is no
libnvToolsExt to link. The raw module wwr.cuda.nvToolsExt therefore re-exports
each through a thin EXTERNAL-linkage forwarding function (exactly as wwr.complex
externalises cuComplex.h's static-inline arithmetic) -- see
src/cuda/nvToolsExt.cppm. rocTX, by contrast, IS a real shared library
(libroctx64.so), so wwr.hip.roctx re-exports its symbols with `using ::`. Either
way, `wwr::wwrtx<Name>` (the WWR_FUNCTION reference in src/tx.cppm) resolves to an
EXTERNAL-linkage symbol the capi thunk links by `import wwr.tx;` and calling it.

So tx IS wrappable as a C ABI: the feasibility verdict is YES. It is hand-written
(not fragment-sourced through the LibrarySpec path) only because its five
functions carry no status enum / handle / POD for that machinery to resolve --
the signatures are trivial (a `const char *` message, a `uint64_t` range id, and
`void` / `int` / `uint64_t` returns), so enumerating them here by hand is simpler
than registering an empty type set. The wwrtxRangeId_t handle (nvtxRangeId_t /
roctx_range_id_t, both uint64_t) is exposed as a plain `uint64_t` -- no vendor
type crosses the boundary.

Every tx call is host-callable and does NOT touch the device (a profiler marker),
so the thunks run fully on a no-GPU runner, like complex and rtc.
"""

from __future__ import annotations

# The range-id capi spelling: both backends' range id is uint64_t (nvtxRangeId_t
# / roctx_range_id_t), so the ABI exposes a plain uint64_t -- no vendor type.
RANGE_ID = "uint64_t"

# Functions. `wwr` is the function in `wwr.tx` the thunk forwards into; `ret` /
# `args` use plain C scalar types (a message string, the range id). The thunk is
# a pure pass-through forward -- no enum map, no POD conversion.
FUNCTIONS = [
    {"capi": "wwr_tx_mark", "wwr": "wwr::wwrtxMarkA",
     "ret": "void", "args": [("const char *", "message")]},
    {"capi": "wwr_tx_range_push", "wwr": "wwr::wwrtxRangePushA",
     "ret": "int", "args": [("const char *", "message")]},
    {"capi": "wwr_tx_range_pop", "wwr": "wwr::wwrtxRangePop",
     "ret": "int", "args": []},
    {"capi": "wwr_tx_range_start", "wwr": "wwr::wwrtxRangeStartA",
     "ret": RANGE_ID, "args": [("const char *", "message")]},
    {"capi": "wwr_tx_range_stop", "wwr": "wwr::wwrtxRangeStop",
     "ret": "void", "args": [(RANGE_ID, "id")]},
]
