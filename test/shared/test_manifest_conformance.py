"""Tests for manifest_conformance.py, the floor-conformance gate (#117).

The gate's own failure mode is passing when it should fail -- a scoping rule that
quietly matches nothing reports "all conformant". So beyond asserting the real
tree is clean, these pin each rule the checker exists to enforce with a synthetic
source checked against the REAL committed hip_runtime_api manifests: an unguarded
pin-only use fails, the same use behind its version guard passes, a guard around a
symbol the floor already has fails, and a cross-library name is out of scope.

Runs on a bare runner -- no CUDA, no ROCm, no compiler -- because the checker
reads only committed manifests and source text, which is the whole point.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "shared"))

import manifest_conformance as mc  # noqa: E402

# The real pin/floor name sets for one library with a genuine floor delta.
_PIN = ROOT / "vendor" / "rocm-7.2.4" / "hip_runtime_api.json"
_FLOOR = ROOT / "vendor" / "rocm-7.1.0" / "hip_runtime_api.json"
PREFIX, PIN = mc.load_manifest(_PIN)
_, FLOOR = mc.load_manifest(_FLOOR)

# hipKernelGetName was added in ROCm 7.2 (pin-only); hipMalloc is at both.
PIN_ONLY = "hipKernelGetName"
AT_BOTH = "hipMalloc"


def check(src: str):
    return mc.check_source(src, PREFIX, PIN, FLOOR, "hip/hip_runtime_api", "rocm-7.1.0")


def test_real_tree_is_conformant():
    report = mc.check_all()
    assert report.violations == [], "\n" + mc.format_report(report)
    # The check must actually be doing work, not skipping everything.
    assert len(report.checked) > 20


def test_unguarded_pin_only_is_flagged():
    violations, _ = check(f"using ::{PIN_ONLY};\n")
    assert [v.name for v in violations] == [PIN_ONLY]
    assert "pin-only" in violations[0].kind
    assert violations[0].floor == "rocm-7.1.0"


def test_pin_only_behind_its_guard_passes():
    src = f"#if WWR_HIP_SINCE_7_2\nusing ::{PIN_ONLY};\n#endif\n"
    violations, _ = check(src)
    assert violations == []


def test_guard_around_floor_symbol_is_flagged():
    # A symbol the floor already ships does not need a since-guard; guarding it is
    # a wrong claim, and the bidirectional check catches it.
    src = f"#if WWR_HIP_SINCE_7_2\nusing ::{AT_BOTH};\n#endif\n"
    violations, _ = check(src)
    assert [v.name for v in violations] == [AT_BOTH]
    assert "unnecessary" in violations[0].kind


def test_guard_else_branch_requires_floor_symbol():
    # The #else of a since-guard is the floor path: a pin-only name there is
    # absent at the floor and must fail.
    src = f"#if WWR_HIP_SINCE_7_2\n#else\nusing ::{PIN_ONLY};\n#endif\n"
    violations, _ = check(src)
    assert [v.name for v in violations] == [PIN_ONLY]
    assert "#else" in violations[0].kind


def test_at_both_unguarded_is_fine():
    violations, _ = check(f"using ::{AT_BOTH};\n")
    assert violations == []


def test_cross_library_name_is_out_of_scope():
    # cudaStream_t leads with 'cuda' != prefix 'hip', so it is another manifest's
    # surface, not a false "absent from hip_runtime_api".
    violations, _ = check("using ::cudaStream_t;\n")
    assert violations == []


def test_nested_guard_is_still_guarded():
    src = (f"#ifdef SOMETHING\n#if WWR_HIP_SINCE_7_2\n"
           f"using ::{PIN_ONLY};\n#endif\n#endif\n")
    violations, _ = check(src)
    assert violations == []


def test_leading_prefix_matches_the_harvester():
    # The rule that decides manifest membership, copied from vendor_harvest.py.
    assert mc.leading_prefix("hipKernelGetName") == "hip"
    assert mc.leading_prefix("HIPRAND_FOO") == "hiprand"
    assert mc.leading_prefix("HIPBLASLT_EPILOGUE_BIAS") == "hipblaslt"
    assert mc.leading_prefix("_underscore") is None


def test_blind_spot_is_bounded():
    # Variant-library uppercase constants (HIPBLASLT_* etc.) are captured by no
    # manifest -- the harvester's prefix rule, not this checker's. Exactly two are
    # version-guarded (WWR_HIPBLASLT_SINCE_1_2's ...SIGMOID_*_EXT); pin the count
    # so a newly uncovered guarded constant is noticed here. See the module
    # docstring's KNOWN BLIND SPOT.
    report = mc.check_all()
    assert report.blind_spots == 2
