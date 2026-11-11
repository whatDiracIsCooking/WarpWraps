"""The `extension_random_normal` DEVICE surface, HAND-WRITTEN -- the first C ABI
over wwr's device/template code (issue #347, architecture.md §25).

Every surface shipped before this one is a HOST surface: a thunk forwards into a
host-callable `wwr::*` function and the per-unit `.so` NEEDs only that surface's
vendor host library. This surface is different in kind. It wraps two
`wwr.extension` modules that LAUNCH KERNELS --

  - wwr.extension.random_normal: a host-callable template launcher
    `wwr::extension::random_normal<T>(wwrStream_t, size_t, wwrrandState*, T*, T)`,
    EXPLICITLY INSTANTIATED over a fixed six-type set (see
    src/extension/random_normal/instantiations.cpp -- that list is authoritative;
    the suffixes below are sourced from it, no type invented, none dropped);
  - wwr.extension.init_state: a single non-template entry point
    `wwr::extension::init_state(wwrStream_t, size_t, wwrrandState*, ull, ull, ull)`
    that seeds the states random_normal draws from (they pair in the usage flow).

A C ABI cannot export a template, so each random_normal INSTANTIATION becomes its
own monomorphised `extern "C"` entry, the type named by a suffix:

  | suffix | T (wwr)               | data POD        | scale by value          |
  |--------|-----------------------|-----------------|-------------------------|
  | f32    | float                 | float           | plain scalar            |
  | f64    | double                | double          | plain scalar            |
  | c64    | wwr::wwrFloatComplex   | wwr_complex64   | {re, im} -> .x/.y       |
  | c128   | wwr::wwrDoubleComplex  | wwr_complex128  | {re, im} -> .x/.y       |
  | f16    | wwr::wwrHalf           | wwr_half        | memcpy bits (§3/§12)    |
  | bf16   | wwr::wwrBfloat16       | wwr_bf16        | memcpy bits (§3/§12)    |

init_state is not a template, so it is ONE entry with no suffix.

The opaque runtime handles map to `void *`: the stream (wwrStream_t) and the
state array (wwrrandState *) are backend types with no vendor-free POD and no
field a C consumer reads -- a Rust caller holds them as `*mut c_void`, getting the
stream from the runtime_api surface and the state buffer as a device allocation it
fills with wwr_extension_init_state before the first draw. The data POD pointer
(wwr_complex64 * etc.) and the by-value scale carry the shared PODs the complex /
fp16 / bf16 surfaces already declare in gen/common.h, so NONE is redeclared here.

These EXECUTE on the device. The C ABI's gate is therefore COMPILE + LINK on both
backends (CI, no GPU); a real run needs a card and is GPU-gated. The unit `.so`
links the rand + runtime device libraries through the two wwr.extension modules
but a plain `.cu` kernel archive does NOT probe for a device on load (unlike
hipTensor), so merely loading the `.so` is safe on a no-GPU box -- the C /
Rust host drivers, which link only complex + runtime_api, are unaffected either way.

HAND-WRITTEN (not fragment-sourced): the launchers are wwr-authored templates /
functions, in no vendor/*/*.json manifest, so there is no `decl` to read a
signature from -- exactly why complex / comp / tx / the numeric types are
hand-written too. The thunk emitter lives in generate.py (gen_extension_random_
normal_*). The feasibility verdict for the rest of src/extension is recorded in
the PR for #347.
"""

from __future__ import annotations

# The random_normal instantiations, sourced 1:1 from
# src/extension/random_normal/instantiations.cpp. `suffix` names the C entry
# (wwr_extension_random_normal_<suffix>); `pod` is the capi type the data pointer
# and the by-value scale carry (a shared POD in gen/common.h, or a plain scalar);
# `wwr` is the neutral output type the template is instantiated over; `kind`
# tells the thunk emitter how to carry the by-value scale across the boundary:
#   - "scalar":  float / double, passed straight through;
#   - "complex": built from the POD's {re, im} and returned via the vendor .x/.y;
#   - "bits":    opaque 16-bit half / bfloat16, memcpy'd to/from the vendor type
#                (the fp16 surface's idiom -- the layout is identical, §3/§12).
INSTANTIATIONS = [
    {"suffix": "f32", "pod": "float", "wwr": "float", "kind": "scalar"},
    {"suffix": "f64", "pod": "double", "wwr": "double", "kind": "scalar"},
    {"suffix": "c64", "pod": "wwr_complex64",
     "wwr": "wwr::wwrFloatComplex", "kind": "complex"},
    {"suffix": "c128", "pod": "wwr_complex128",
     "wwr": "wwr::wwrDoubleComplex", "kind": "complex"},
    {"suffix": "f16", "pod": "wwr_half", "wwr": "wwr::wwrHalf", "kind": "bits"},
    {"suffix": "bf16", "pod": "wwr_bf16", "wwr": "wwr::wwrBfloat16", "kind": "bits"},
]

# The single init_state entry point. Not a template, so no suffix and no type
# table: the C entry is wwr_extension_init_state, forwarding into
# wwr::extension::init_state. The three ull defaults (seed / sequence_offset /
# offset) become required arguments in C -- a C ABI has no default arguments.
INIT_STATE = {
    "capi": "wwr_extension_init_state",
    "wwr": "wwr::extension::init_state",
    "args": [
        ("void *", "stream"),
        ("size_t", "count"),
        ("void *", "states"),
        ("unsigned long long", "seed"),
        ("unsigned long long", "sequence_offset"),
        ("unsigned long long", "offset"),
    ],
}
