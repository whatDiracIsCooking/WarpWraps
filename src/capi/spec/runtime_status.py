"""The `runtime_api` status slice as a FRAGMENT-SOURCED surface.

This is the thin per-surface definition the fragment-sourcing path needs, and the
shape #349 (blas) copies. It carries NO signatures and NO vendor names: it names

  * the fragment to source from (`detail/runtime_api_names.h`),
  * which fragment entries to expose and the capi name each gets, and
  * the one thing a name cannot carry -- the stable `wwr_status` enum map (§6).

Everything else -- each function's signature, the enum's backend values -- is
cross-referenced at generate time from the fragment (names) and the vendor
manifests (signatures / values). Delete `RuntimeGetVersion` from the fragment
and it drops from the generated header/thunk/`ffi.rs`; that is the drift the
hand-written `spec/complex.py` restatement could not prevent (architecture.md
§25, #328).

`complex` is NOT sourced this way: its thunks forward into wwr-AUTHORED inline
wrappers (`wwr::wwrCaddf`), which are not vendor functions and so appear in no
vendor manifest. It stays a documented hand-written special case in
`spec/complex.py`; see the generator and the README.
"""

from __future__ import annotations

from captypes import EnumMap

# The fragment this surface sources its names from (relative to src/).
FRAGMENT = "detail/runtime_api_names.h"

# The functions to expose, as (capi_name, fragment_wwr_name). The signature of
# each is read from the vendor manifest via the fragment's per-backend spelling;
# none is written here. These three are the status slice (#329 increment 2) --
# each returns the vendor error enum, mapped to wwr_status below.
FUNCTIONS = [
    ("wwr_runtime_get_version", "wwrRuntimeGetVersion"),
    ("wwr_get_device_count", "wwrGetDeviceCount"),
    ("wwr_set_device", "wwrSetDevice"),
]

# The stable status enum (§6). The vendor return type cudaError_t / hipError_t is
# classified as this EnumMap: a fixed-value capi enum with SUCCESS first (== 0),
# the curated error set, UNKNOWN last, and a `_map` that compares the NEUTRAL
# wwr:: constants. The values are backend-independent and frozen (architecture.md
# §21); each `entries` pair is (capi_enumerator, neutral wwr constant), and both
# the wwr constant and the SUCCESS/error vendor names are themselves fragment
# entries (WWR_RT_VALUE(Success), WWR_RT_VALUE(ErrorInvalidValue), ...), so this
# lists the MAPPING, not a second copy of the names.
STATUS = EnumMap(
    vendor_types=("cudaError_t", "hipError_t"),
    capi_type="wwr_status",
    success="WWR_STATUS_SUCCESS",
    unknown="WWR_STATUS_UNKNOWN",
    entries=(
        ("WWR_STATUS_INVALID_VALUE", "wwrErrorInvalidValue"),
        ("WWR_STATUS_INVALID_DEVICE", "wwrErrorInvalidDevice"),
        ("WWR_STATUS_NO_DEVICE", "wwrErrorNoDevice"),
        ("WWR_STATUS_NOT_READY", "wwrErrorNotReady"),
        ("WWR_STATUS_MEMORY_ALLOCATION", "wwrErrorMemoryAllocation"),
        ("WWR_STATUS_NOT_SUPPORTED", "wwrErrorNotSupported"),
    ),
)
