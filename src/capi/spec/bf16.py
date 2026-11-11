"""The `bf16` numeric-type surface, HAND-WRITTEN (the fp16 twin).

bf16 is the bfloat16 counterpart to fp16.py, and hand-written for the same
reason: the host float<->bfloat16 conversions wwr.bf16 exports (wwr::wwrFloat2Bfloat16
/ wwr::wwrBfloat162Float) are wwr-AUTHORED inline forwarders (detail/bf16_names.h),
not vendor functions in any manifest -- so no `decl` to source, and the surface
is enumerated here by hand.

The bfloat16 type DIVERGES by backend (__nv_bfloat16 vs __hip_bfloat16, unlike
half which is spelled __half on both), which is exactly why a stable vendor-free
POD is needed: wwr_bf16, 16 opaque bits, layout-matched to the vendor bfloat16
the thunk reinterpret_casts to (§3/§12). It lives in gen/common.h alongside
wwr_half and the complex PODs -- the shared numeric vocabulary (SHARED_NUMERIC_PODS
in generate.py). bf16 is the only surface that names it today, but it is a numeric
scalar POD of the same family, so it is consolidated there rather than in the
unit header, matching the numeric-types decision.

DEVICE-ONLY conversions are OUT OF SCOPE: both bf16 conversions are host-callable
inline forwarders (bf16.cppm binds them `inline`), so both are wrapped -- nothing
is device-only to skip. bfloat16's arithmetic OPERATORS are static-inline on HIP
and live only in the raw module (never this neutral layer), so they are not part
of this surface at all.
"""

from __future__ import annotations

# The host conversions, forwarding into wwr.bf16. The wwr_bf16 POD crosses by
# value, reinterpret_cast to the neutral wwr::wwrBfloat16 (§3/§12) in the thunk.
FUNCTIONS = [
    {"capi": "wwr_float2bfloat16", "wwr": "wwr::wwrFloat2Bfloat16",
     "ret": "wwr_bf16", "args": [("float", "value")]},
    {"capi": "wwr_bfloat162float", "wwr": "wwr::wwrBfloat162Float",
     "ret": "float", "args": [("wwr_bf16", "value")]},
]
