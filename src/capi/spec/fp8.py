"""The `fp8` low-precision numeric-type surface, HAND-WRITTEN (the comp shape).

fp8 is a TYPE surface: the 8-bit float storage POD, the saturation /
interpretation value enums, and the two host narrowing conversions wwr.fp8
exports (wwr::wwrFloat2Fp8 / wwr::wwrDouble2Fp8). Those are wwr-AUTHORED forwarders
routing through the raw module's host wrappers (fp8.cppm), NOT vendor functions in
any manifest -- so no `decl` to source, and the surface is enumerated here by
hand, like comp.py (which also carries value enums the thunk maps).

SCOPE: host NARROWING only (float/double -> fp8 storage), which is exactly what
wwr.fp8 host-exports. Neither vendor has a direct storage -> float WIDENING call
(a value read back travels through the scalar struct type's own operator float, a
C++ member the C ABI cannot express), so there is no host widening to wrap; the
packed x2/x4 conversions and the fnuz/e8m0 formats stay in the raw modules. These
are DEVICE-ONLY / non-C-ABI surfaces, OUT OF SCOPE per the numeric-types decision,
and reported as explicit skips by generate.py -- never silently dropped.

fp4 / fp6 are the companion low-precision surfaces, but are BLOCKED ON THE HIP
BACKEND and so CANNOT be wrapped here: wwr.fp4 / wwr.fp6 are NOT registered in
src/CMakeLists.txt (their HIP raw modules wwr.hip.hip_fp4 / wwr.hip.hip_fp6 do not
build with this toolchain -- amd_detail/amd_hip_ocp_types.h #errors outside real
`-x hip` device mode). A capi fp4 / fp6 unit would `import wwr.fp4` / `wwr.fp6`,
which do not exist on HIP, so the HIP leg of the cross-backend compile gate would
fail. They flip on the day their HIP raw modules do (the same day src/CMakeLists.txt
registers wwr.fp4 / wwr.fp6), reported as a surface-level skip by generate.py.
Section 11 (fp4 and fp6 cannot share a translation unit) is moot while neither is
built -- when they unblock, each gets its OWN capi unit (importing its own
wwr.fp4 / wwr.fp6), which already honours the one-TU-per-type rule.

The storage POD wwr_fp8_storage is 8 opaque bits (the vendor __nv_fp8_storage_t /
__hip_fp8_storage_t is `unsigned char`), this unit's OWN POD (unit header, not
common.h -- no other surface names it). The two value enums are stable capi enums
the thunk maps capi<->neutral against the wwr:: constants, exactly as comp's
status / data-type enums.
"""

from __future__ import annotations

from captypes import ValueEnum

# The 8-bit storage POD: 8 opaque bits, layout-matched to the vendor storage the
# conversions return (an unsigned char on both backends). Rust sees a transparent
# u8; the thunk returns it straight (the wwr conversion already yields the vendor
# storage, which IS unsigned char, so no cast is needed -- a plain value).
STORAGE_POD = {"capi": "wwr_fp8_storage", "fields": [("uint8_t", "bits")]}

# The saturation mode taken when narrowing. A stable capi enum the thunk maps to
# the neutral wwr:: constants (wwrNosat / wwrSatfinite), the comp ValueEnum shape.
# vendor_types is unused here (hand-written surface, no manifest resolution), kept
# for the ValueEnum dataclass; the maps compare only the neutral constants.
SATURATION = ValueEnum(
    vendor_types=("__nv_saturation_t", "__hip_saturation_t"),
    capi_type="wwr_fp8_saturation",
    wwr_type="wwrSaturation",
    entries=(
        ("WWR_FP8_NOSAT", "wwrNosat"),
        ("WWR_FP8_SATFINITE", "wwrSatfinite"),
    ),
    fallback="WWR_FP8_NOSAT",
)

# The interpretation (format kind): the shared OCP E4M3 / E5M2 formats. Stable
# capi enum, mapped to wwr::wwrE4m3 / wwr::wwrE5m2.
INTERPRETATION = ValueEnum(
    vendor_types=("__nv_fp8_interpretation_t", "__hip_fp8_interpretation_t"),
    capi_type="wwr_fp8_interpretation",
    wwr_type="wwrFp8Interpretation",
    entries=(
        ("WWR_FP8_E4M3", "wwrE4m3"),
        ("WWR_FP8_E5M2", "wwrE5m2"),
    ),
    fallback="WWR_FP8_E4M3",
)

VALUE_ENUMS = (SATURATION, INTERPRETATION)

# The two host narrowing conversions, forwarding into wwr.fp8. Each `param` is
# (repr, capi_c_type, name): "scalar" passes through, "saturation" /
# "interpretation" are value enums the thunk maps capi -> neutral. Every call
# returns the wwr_fp8_storage POD (by value). The argument order -- (value,
# saturation, interpretation) -- is the one wwr.fp8 declares.
FUNCTIONS = [
    {"capi": "wwr_float2fp8", "wwr": "wwr::wwrFloat2Fp8",
     "params": [("scalar", "float", "value"),
                ("saturation", "wwr_fp8_saturation", "saturate"),
                ("interpretation", "wwr_fp8_interpretation", "interpretation")]},
    {"capi": "wwr_double2fp8", "wwr": "wwr::wwrDouble2Fp8",
     "params": [("scalar", "double", "value"),
                ("saturation", "wwr_fp8_saturation", "saturate"),
                ("interpretation", "wwr_fp8_interpretation", "interpretation")]},
]

# The surface-level skips, reported by generate.py (NO SILENT DROPS). A tuple of
# (what, reason). These are OUT OF SCOPE, not failures.
SKIPS = (
    ("fp4 (whole surface)",
     "wwr.fp4 is BLOCKED on HIP (wwr.hip.hip_fp4 does not build -- "
     "amd_detail/amd_hip_ocp_types.h #errors); not registered in src/CMakeLists.txt, "
     "so a capi unit importing wwr.fp4 would fail the HIP compile gate. Flips when "
     "the HIP raw module does."),
    ("fp6 (whole surface)",
     "wwr.fp6 is BLOCKED on HIP (wwr.hip.hip_fp6 does not build, same reason); "
     "not registered, so out of scope until the HIP raw module builds."),
    ("fp8 storage -> float widening",
     "no host widening call exists (widening is the scalar type's operator float, "
     "a C++ member the C ABI cannot express); device/non-ABI, out of scope."),
    ("fp8 x2/x4 packed + fnuz/e8m0 conversions",
     "not host-exported by wwr.fp8 (raw-module only); out of scope for the host C ABI"),
)
