"""The `fp16` numeric-type surface, HAND-WRITTEN (the complex / comp shape).

fp16 is a TYPE surface, not a vendor-library one: it exposes the half-precision
POD and the two host float<->half conversions wwr.fp16 exports (wwr::wwrFloat2Half
/ wwr::wwrHalf2Float). Those are wwr-AUTHORED inline forwarders (detail/fp16_names.h),
NOT vendor functions -- they appear in no vendor/*/*.json manifest, so there is
no `decl` to source a signature from, and the fragment-sourced library path
(spec/library.py) cannot cover them. So the surface is enumerated here by hand,
exactly as complex.py and comp.py are, against the neutral wwr* signatures.

The one thing a C ABI needs that the wwr names do not carry is a stable,
vendor-free POD for the half type (wwr::wwrHalf is the vendor __half, which is a
different by-value type on each backend). wwr_half is that POD -- 16 opaque bits,
layout-matched to the vendor half the thunk reinterpret_casts to (§3/§12). It is
SHARED (gen/common.h): the blas surface already takes __half BY POINTER and
declares the same POD, so fp16 declaring it too would be a duplicate; the shared
numeric vocabulary carries it once, alongside the complex PODs. See generate.py's
gen_common_h and SHARED_NUMERIC_PODS.

DEVICE-ONLY conversions are OUT OF SCOPE (the numeric-types decision): the C ABI
is a HOST surface, and fp16's conversions are host-callable inline forwarders
(fp16.cppm binds them `inline`), so both are wrapped -- nothing is device-only to
skip here. (The device section of fp16.h pastes the same two with
__device__ __forceinline__ for a .cu, which the C ABI never calls.)
"""

from __future__ import annotations

# The host conversions, forwarding into wwr.fp16. `wwr` is the function the thunk
# forwards into; `ret`/`args` use capi type names (the shared wwr_half POD or a
# plain float). The POD crosses the boundary by VALUE, reinterpret_cast to the
# neutral wwr::wwrHalf (layout-identical, §3/§12) -- generate.py's fp16 thunk
# emitter handles the cast.
FUNCTIONS = [
    {"capi": "wwr_float2half", "wwr": "wwr::wwrFloat2Half",
     "ret": "wwr_half", "args": [("float", "value")]},
    {"capi": "wwr_half2float", "wwr": "wwr::wwrHalf2Float",
     "ret": "float", "args": [("wwr_half", "value")]},
]
