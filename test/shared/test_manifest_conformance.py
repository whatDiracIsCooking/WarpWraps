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

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "shared"))

import manifest_conformance as mc  # noqa: E402

# The real pin/floor name sets for one library with a genuine floor delta.
_PIN = ROOT / "vendor" / "rocm-7.2.4" / "hip_runtime_api.json"
_FLOOR = ROOT / "vendor" / "rocm-7.1.0" / "hip_runtime_api.json"
ACCEPTED, PIN = mc.load_manifest(_PIN)
_, FLOOR = mc.load_manifest(_FLOOR)

# hipKernelGetName was added in ROCm 7.2 (pin-only); hipMalloc is at both.
PIN_ONLY = "hipKernelGetName"
AT_BOTH = "hipMalloc"

# The real linkable data of the same library, for assertion 3. hipMalloc the .so
# exports; hipExternalMemoryGetMappedMipmappedArray it declares but does not
# export (the module's one live WWR_DECLARED_CHECK site).
_LINK = json.loads(_PIN.read_text())["linkable"]
LINKABLE = set(_LINK["linkable"])
DECLARED_ONLY = set(_LINK["declared_not_linkable"])
EXPORTED = "hipMalloc"
DECLARED_NOT_LINKABLE = "hipExternalMemoryGetMappedMipmappedArray"


def check(src: str):
    return mc.check_source(
        src, ACCEPTED, PIN, FLOOR, "hip/hip_runtime_api", "rocm-7.1.0")


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


def test_real_tree_link_complete():
    report = mc.check_link_completeness()
    assert report.gaps == [], "\n" + mc.format_link_report(report)
    assert len(report.checked) > 20  # actually verifying, not skipping everything


def test_link_exceptions_are_exactly_hiptensor_and_rccl():
    report = mc.check_link_completeness()
    assert set(report.excepted) == {"hip/hiptensor", "hip/rccl"}


def test_unchecked_flags_a_wrapped_function_without_a_check():
    # A function that is re-exported and declared but has no WWR_*_CHECK is a gap.
    assert mc.unchecked_functions({"fooBar"}, {"fooBar"}, set()) == ["fooBar"]
    # ... and is clean once a check names it.
    assert mc.unchecked_functions({"fooBar"}, {"fooBar"}, {"fooBar"}) == []


def test_unchecked_ignores_unwrapped_and_nonfunction_names():
    # Declared but not re-exported (no using::) -> not wwr's surface, not required.
    assert mc.unchecked_functions({"fooBar"}, set(), set()) == []
    # Re-exported but not a manifest FunctionDecl (a type/enum) -> not link-checked.
    assert mc.unchecked_functions(set(), {"fooType"}, set()) == []


def test_blind_spot_is_bounded():
    # #167 whitelisted each variant library's own token (HIPBLASLT_* -> hipblaslt
    # etc.) as an extra_prefix, so the once-blind guarded constants
    # (WWR_HIPBLASLT_SINCE_1_2's ...SIGMOID_*_EXT) are now in scope and validated.
    # The count is the RESIDUAL -- a guarded using:: on a companion token nobody
    # whitelisted yet -- pinned at 0 so the next one is noticed, not silently
    # unchecked. See the module docstring's BLIND SPOTS.
    report = mc.check_all()
    assert report.blind_spots == 0


# ── Assertion 3: macro correctness ──────────────────────────────────────────


def test_real_tree_macro_correct():
    report = mc.check_macro_correctness()
    assert report.mislabels == [], "\n" + mc.format_macro_report(report)
    assert len(report.checked) > 20  # actually judging, not skipping everything


def test_exported_symbol_must_use_link_check():
    # The converse of the payoff: WWR_DECLARED_CHECK on a symbol the .so exports
    # is wrong, and the check demands WWR_LINK_CHECK.
    assert mc.macro_violation(
        "DECLARED", EXPORTED, LINKABLE, DECLARED_ONLY) == "WWR_LINK_CHECK"
    # ... and WWR_LINK_CHECK on it is correct.
    assert mc.macro_violation("LINK", EXPORTED, LINKABLE, DECLARED_ONLY) is None


def test_declared_only_symbol_must_use_declared_check():
    # A symbol the .so declares but does not export: WWR_LINK_CHECK would fail the
    # link, so the check demands WWR_DECLARED_CHECK.
    assert mc.macro_violation(
        "LINK", DECLARED_NOT_LINKABLE, LINKABLE, DECLARED_ONLY
    ) == "WWR_DECLARED_CHECK"
    # ... and WWR_DECLARED_CHECK on it is correct.
    assert mc.macro_violation(
        "DECLARED", DECLARED_NOT_LINKABLE, LINKABLE, DECLARED_ONLY) is None


def test_macro_out_of_scope_name_is_not_a_violation():
    # A checked name in neither linkable set (a cross-library symbol, or one the
    # harvest never saw) cannot be judged and is never a violation.
    assert mc.macro_violation("LINK", "cudaStreamFoo", LINKABLE, DECLARED_ONLY) is None
    assert mc.macro_violation(
        "DECLARED", "cudaStreamFoo", LINKABLE, DECLARED_ONLY) is None


def test_macro_linkable_wins_a_tie():
    # If a name were in both sets, exported wins -- WWR_LINK_CHECK is demanded.
    assert mc.macro_violation("DECLARED", "dup", {"dup"}, {"dup"}) == "WWR_LINK_CHECK"
    assert mc.macro_violation("LINK", "dup", {"dup"}, {"dup"}) is None


def test_every_declared_check_site_validates():
    # Acceptance (#121): every live WWR_DECLARED_CHECK in the tree names a symbol
    # the pin manifest records as declared-but-not-linkable -- none is a stale
    # "someday" the vendor has since started exporting. The check reads test
    # source, so count the real sites too: this must be judging them, not zero.
    sites = 0
    for cppm in sorted((ROOT / "test").rglob("*.cppm")):
        kinds = mc._MACRO_CHECK.findall(cppm.read_text())
        sites += sum(k == "DECLARED" for k, _ in kinds)
    assert sites >= 10  # the #121 floor; the tree has more after the LINK fixes
    report = mc.check_macro_correctness()
    declared_mislabels = [m for m in report.mislabels if m.used == "WWR_DECLARED_CHECK"]
    assert declared_mislabels == [], "\n" + mc.format_macro_report(report)


def test_macro_unjudged_when_so_absent_at_harvest():
    # nvToolsExt's libnvToolsExt.so was dropped at CUDA 12, so its manifest has
    # linkable.available false: no truth to judge by, reported UNJUDGED not guessed.
    report = mc.check_macro_correctness()
    assert "cuda/nvToolsExt" in report.unjudged
