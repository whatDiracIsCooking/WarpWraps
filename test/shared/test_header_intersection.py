"""Tests for header_intersection.py, the shared-surface / coverage tool.

This is the SDK-FREE gate for that tool. header_intersection.py used to regex the
vendor headers at run time, so it only ran where CUDA / ROCm were installed. It
now reads the committed ``vendor/<pin>/<lib>.json`` manifests, so it -- and this
file -- run anywhere the repo is checked out: no CUDA, no ROCm, no compiler. That
is what lets the CI ``test`` job (a bare runner, no GPU SDK) exercise it.

A coverage tool's failure mode is passing when it should fail -- a parser that
silently matches nothing reports "all covered". So beyond checking the real
rand / blas surfaces intersect and are covered, these pin the two facts the
manifest rewrite exists to guarantee: the ``curandState*_t`` device types (which
live in ``curand_kernel.json``, the #115 completeness gap this work closed) are
in the rand intersection, and a shared name absent from a module source is
reported MISSING rather than silently dropped.

The numbers here are the MANIFEST-derived answer, which intentionally diverges
from -- and is more correct than -- the old single-header text scan; see
header_intersection.py's header for why (umbrella headers like cublas_v2.h hid
the bulk of the API from a one-file scan).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "devtools"))

import header_intersection as hi  # noqa: E402

CUDA = ROOT / "vendor" / "cuda-13.0.x"
ROCM = ROOT / "vendor" / "rocm-7.2.4"

RAND_CUDA = [CUDA / "curand.json", CUDA / "curand_kernel.json"]
RAND_HIP = [ROCM / "hiprand.json", ROCM / "hiprand_kernel.json"]
# detail/rand_names.h carries the host API (the WWR_FUNCTION_RAW aliases, moved
# out of rand.cppm into the fragment the module and wwr/rand.h both paste);
# rand.h carries the generator state types (the curandState*_t device types below)
# and, in its device-pass-gated section, the __device__ generators (folded in from
# the former rand.cuh). rand.cppm now just #includes the fragment and re-exports
# the state types, so a text scan sees its host names only via the fragment. All
# three are scanned so the whole shared surface is covered.
RAND_SRC = [
    ROOT / "src" / "rand.cppm",
    ROOT / "src" / "rand.h",
    ROOT / "src" / "detail" / "rand_names.h",
]

BLAS_CUDA = [CUDA / "cublas_v2.json"]
BLAS_HIP = [ROCM / "hipblas.json"]
# detail/blas_names.h carries the wwrblas* surface (the WWR_FUNCTION_RAW aliases,
# moved out of blas.cppm into the fragment the module and wwr/blas.h both paste);
# blas.cppm now just #includes it, so a text scan sees its names only via the
# fragment.
BLAS_SRC = [ROOT / "src" / "blas.cppm", ROOT / "src" / "detail" / "blas_names.h"]

# runtime_api's neutral surface binds its vendor names through the 1-arg
# WWR_RT_* paste, so the vendor identifier never appears literally -- the
# surface a raw text scan is blind to. The paste list lives in the shared
# fragment detail/runtime_api_names.h (included by runtime_api.cppm); the .cppm
# itself carries the macro definitions and the hand-written #if forwarders.
RT_CUDA = [CUDA / "cuda_runtime_api.json"]
RT_HIP = [ROCM / "hip_runtime_api.json"]
RT_SRC = [
    ROOT / "src" / "runtime_api.cppm",
    ROOT / "src" / "runtime_api.h",
    ROOT / "src" / "detail" / "runtime_api_names.h",
]


def _report(cuda, hip, coverage=None):
    return hi.build_report(hi.Side("cuda", cuda), hi.Side("hip", hip), coverage)


def test_manifests_are_present_and_sdk_free():
    # The whole point: this runs off committed data. If a manifest is missing the
    # test must fail loudly, not skip -- a skip here would hide a deleted file.
    for p in RAND_CUDA + RAND_HIP + BLAS_CUDA + BLAS_HIP:
        assert p.exists(), f"missing committed manifest {p}"


def test_rand_intersection_is_nonempty():
    report = _report(RAND_CUDA, RAND_HIP)
    assert report["cuda"]["prefix"] == "curand"
    assert report["hip"]["prefix"] == "hiprand"
    # A regression here means the manifest read silently matched nothing.
    assert len(report["intersection"]) >= 100


def test_rand_intersection_includes_curand_state_device_types():
    # These live in curand_kernel.json, the #115 gap this work closed. Without
    # that manifest the CUDA side never declares them, so they drop out of the
    # intersection -- exactly the bug this asserts against.
    report = _report(RAND_CUDA, RAND_HIP)
    state_t = {
        item["key"]
        for item in report["intersection"]
        if item["key"].startswith("state") and item["key"].endswith("_t")
    }
    expected = {
        "state_t", "statemrg32k3a_t", "statemtgp32_t", "statephilox4_32_10_t",
        "statescrambledsobol32_t", "statescrambledsobol64_t",
        "statesobol32_t", "statesobol64_t", "statexorwow_t",
    }
    assert expected <= state_t, f"missing shared curandState*_t: {expected - state_t}"


def test_rand_coverage_over_module_source():
    report = _report(RAND_CUDA, RAND_HIP, RAND_SRC)
    cov = report["coverage"]
    assert cov["intersection"] == len(report["intersection"])
    # rand is a whole-surface module: the state types must all be referenced.
    missing_keys = {m["key"] for m in cov["missing"]}
    state_keys = {i["key"] for i in report["intersection"]
                  if i["key"].startswith("state")}
    assert not (state_keys & missing_keys), (
        f"rand module leaves shared state types uncovered: "
        f"{sorted(state_keys & missing_keys)}"
    )


def test_blas_intersection_is_nonempty():
    report = _report(BLAS_CUDA, BLAS_HIP)
    assert report["cuda"]["prefix"] == "cublas"
    assert report["hip"]["prefix"] == "hipblas"
    # cublas_v2 <-> hipblas share a substantial GEMM/BLAS surface; the manifest
    # answer is larger than the old header-scan number (which the umbrella hid).
    assert len(report["intersection"]) >= 150


def test_blas_coverage_reports_a_reference_count():
    report = _report(BLAS_CUDA, BLAS_HIP, BLAS_SRC)
    cov = report["coverage"]
    covered = cov["intersection"] - len(cov["missing"])
    # blas is a curated-subset module: some shared names are unused, so `missing`
    # is a nonempty discovery menu -- but a real slice must be covered.
    assert covered > 0
    assert covered <= cov["intersection"]


def test_missing_symbol_is_reported_not_swallowed():
    # A coverage tool must FAIL to find a symbol that is not in the source. Feed
    # an empty source (a real file that names no vendor symbol) and every shared
    # symbol must land in `missing`.
    empty = ROOT / "vendor" / "README.md"  # committed, names no curand* symbol
    report = _report(RAND_CUDA, RAND_HIP, [empty])
    cov = report["coverage"]
    assert len(cov["missing"]) == cov["intersection"]
    assert cov["intersection"] > 0


def test_leading_prefix_normalises_both_spellings():
    assert hi.leading_prefix("curandCreateGenerator") == "curand"
    assert hi.leading_prefix("CURAND_STATUS_SUCCESS") == "curand"
    assert hi.leading_prefix("hiprandState_t") == "hiprand"


# --- The curated decisions file (devtools/coverage_decisions.json) ------------
# These pin the contract-class machine and the documented-omission pass/fail
# behaviour, off committed data -- no SDK, no GPU. The mechanism (contract class,
# omission matching, the fail verdict) is exercised with SYNTHETIC two-symbol
# manifests written to tmp so the pass/fail flip is unambiguous and independent
# of any real module's dispatch-macro scan gaps; the class assignment itself is
# checked against the real decisions file and the committed vendor manifests.


def _mini_manifest(prefix, names):
    """A minimal vendor_harvest.py-shaped manifest: just a prefix and symbols."""
    return {
        "harvest": {"prefix": prefix},
        "symbols": [{"name": n, "kind": "FunctionDecl"} for n in names],
    }


def _write(tmp_path, fname, prefix, names):
    p = tmp_path / fname
    p.write_text(json.dumps(_mini_manifest(prefix, names)))
    return p


def test_decisions_file_is_committed_and_well_formed():
    decisions = hi.load_decisions()
    assert decisions, "coverage_decisions.json missing or empty"
    mods = decisions["modules"]
    # Every module declares one of the two known contract classes.
    for name, block in mods.items():
        assert block["contract"] in (hi.WHOLE_SURFACE, hi.CURATED_SUBSET), name
    # The class split the docstring names must hold.
    whole = {n for n, b in mods.items() if b["contract"] == hi.WHOLE_SURFACE}
    subset = {n for n, b in mods.items() if b["contract"] == hi.CURATED_SUBSET}
    assert {"rand", "fft", "tx", "runtime_api"} <= whole
    assert {"blas", "solver", "sparse"} <= subset


def test_named_omissions_are_recorded_with_rationale():
    # The prose facts that moved into the file must actually be there, each with a
    # non-empty rationale and a source pointer -- this is what lets the prose point
    # here instead of restating them.
    mods = hi.load_decisions()["modules"]

    def names(module):
        out = set()
        for om in mods[module]["omissions"]:
            for side in ("cuda", "hip"):
                if om.get(side):
                    out |= set(om[side].split("|"))
            assert om["why"].strip(), f"{module} omission has no rationale"
            assert om["source"].strip(), f"{module} omission has no source"
        return out

    assert "cublasBfloat16" in names("blas")
    assert "hipsparseBfloat16" in names("blas")
    assert any(n.startswith("cusolverMg") for n in names("solver"))
    assert "cusparseSpMMOp" in names("sparse")
    assert "cusparseSgebsr2gebsc_bufferSizeExt" in names("sparse")
    assert "hipExternalMemoryGetMappedMipmappedArray" in names("runtime_api")


def test_whole_surface_fails_on_undocumented_gap(tmp_path):
    # Both backends share fooBar and fooBaz; the source names only fooBar.
    cuda = _write(tmp_path, "c.json", "foo", ["fooBar", "fooBaz"])
    hip = _write(tmp_path, "h.json", "bar", ["barBar", "barBaz"])
    src = tmp_path / "mod.cppm"
    src.write_text("fooBar barBar")  # fooBaz/barBaz unreferenced
    decision = {"contract": hi.WHOLE_SURFACE, "omissions": []}
    report = hi.build_report(hi.Side("cuda", [cuda]), hi.Side("hip", [hip]),
                             [src], decision)
    cov = report["coverage"]
    assert cov["fails"] is True
    assert {i["key"] for i in cov["undocumented_gaps"]} == {"baz"}
    assert cov["documented_omissions"] == []


def test_whole_surface_passes_when_gap_is_a_documented_omission(tmp_path):
    # Same surface, same source -- but now fooBaz is a documented omission, so the
    # gap becomes a deliberate, justified pass. This is the flip the issue asks to
    # demonstrate: removing the omission entry turns it back into a failure (the
    # test above), adding it makes it pass.
    cuda = _write(tmp_path, "c.json", "foo", ["fooBar", "fooBaz"])
    hip = _write(tmp_path, "h.json", "bar", ["barBar", "barBaz"])
    src = tmp_path / "mod.cppm"
    src.write_text("fooBar barBar")
    decision = {
        "contract": hi.WHOLE_SURFACE,
        "omissions": [{"key": "baz", "cuda": "fooBaz", "hip": "barBaz",
                       "why": "test", "source": "test"}],
    }
    report = hi.build_report(hi.Side("cuda", [cuda]), hi.Side("hip", [hip]),
                             [src], decision)
    cov = report["coverage"]
    assert cov["fails"] is False
    assert cov["undocumented_gaps"] == []
    assert {i["key"] for i in cov["documented_omissions"]} == {"baz"}


def test_curated_subset_reports_but_does_not_fail(tmp_path):
    # A curated-subset module never promised the full intersection: the same gap
    # is reported in `missing` but the verdict is not a failure.
    cuda = _write(tmp_path, "c.json", "foo", ["fooBar", "fooBaz"])
    hip = _write(tmp_path, "h.json", "bar", ["barBar", "barBaz"])
    src = tmp_path / "mod.cppm"
    src.write_text("fooBar barBar")
    decision = {"contract": hi.CURATED_SUBSET, "omissions": []}
    report = hi.build_report(hi.Side("cuda", [cuda]), hi.Side("hip", [hip]),
                             [src], decision)
    cov = report["coverage"]
    assert cov["fails"] is False
    assert {i["key"] for i in cov["missing"]} == {"baz"}


def test_no_decision_fails_on_any_gap_as_before(tmp_path):
    # An ad-hoc run with no --module keeps the historical behaviour: any gap fails.
    cuda = _write(tmp_path, "c.json", "foo", ["fooBar", "fooBaz"])
    hip = _write(tmp_path, "h.json", "bar", ["barBar", "barBaz"])
    src = tmp_path / "mod.cppm"
    src.write_text("fooBar barBar")
    report = hi.build_report(hi.Side("cuda", [cuda]), hi.Side("hip", [hip]),
                             [src], None)
    assert report["coverage"]["fails"] is True


def test_wildcard_omission_matches_by_prefix(tmp_path):
    # A `cusolverMg*`-style trailing wildcard covers every shared name that shares
    # the stem, so one entry documents a whole family.
    cuda = _write(tmp_path, "c.json", "foo",
                  ["fooBar", "fooMgOne", "fooMgTwo"])
    hip = _write(tmp_path, "h.json", "bar",
                 ["barBar", "barMgOne", "barMgTwo"])
    src = tmp_path / "mod.cppm"
    src.write_text("fooBar barBar")
    decision = {
        "contract": hi.WHOLE_SURFACE,
        "omissions": [{"key": "mg", "cuda": "fooMg*", "hip": "barMg*",
                       "why": "test", "source": "test"}],
    }
    report = hi.build_report(hi.Side("cuda", [cuda]), hi.Side("hip", [hip]),
                             [src], decision)
    cov = report["coverage"]
    assert cov["fails"] is False
    assert {i["key"] for i in cov["documented_omissions"]} == {"mgone", "mgtwo"}


def test_real_whole_surface_module_passes_clean(tmp_path):
    # fft is whole-surface and (unlike rand, whose dispatch macros hide names from
    # a literal scan) fully token-visible: it must pass with zero undocumented
    # gaps against committed data, proving the wiring works end to end on a real
    # module, not just synthetic fixtures.
    decision = hi.module_decision(hi.load_decisions(), "fft")
    assert decision is not None
    cuda = [CUDA / "cufft.json"]
    hip = [ROCM / "hipfft.json"]
    src = [ROOT / "src" / "cuda" / "cufft.cppm"]
    report = hi.build_report(hi.Side("cuda", cuda), hi.Side("hip", hip), src,
                             decision)
    cov = report["coverage"]
    assert cov["contract"] == hi.WHOLE_SURFACE
    assert cov["fails"] is False
    assert cov["undocumented_gaps"] == []


# --- The macro-argument-aware coverage matcher (#258) -------------------------
# runtime_api binds vendor names through the 1-arg WWR_RT_* paste, so the vendor
# identifier is never literal. These pin that --coverage resolves the paste (and
# the N-arg explicit forms), and that a vendor-deprecated shared symbol is never
# reported as a gap -- derived from the manifest flag, not hand-listed.


def test_resolved_names_reads_the_paste_and_explicit_macros():
    text = (
        "WWR_RT_FUNCTION(GetLastError)\n"
        "WWR_RT_VALUE(MemcpyHostToDevice)\n"
        "WWR_FUNCTION_RAW(wwrGetDeviceProperties, cudaGetDeviceProperties, "
        "hipGetDevicePropertiesR0600)\n"
    )
    got = hi.resolved_names(text)
    # 1-arg paste: the preprocessor synthesises cuda<Tail> / hip<Tail>.
    assert {"cudaGetLastError", "hipGetLastError"} <= got
    assert {"cudaMemcpyHostToDevice", "hipMemcpyHostToDevice"} <= got
    # N-arg explicit: the vendor names are the 2nd/3rd args verbatim.
    assert {"cudaGetDeviceProperties", "hipGetDevicePropertiesR0600"} <= got


def test_runtime_paste_surface_is_legible_to_coverage():
    # Before #258 the literal scan saw ~9/686 (blind to the WWR_RT_* paste). Now
    # the token-pasted error-handling pair resolves and reads COVERED:
    # GetLastError (clears the sticky error) and its sibling PeekAtLastError (the
    # #257 gap, wrapped in the same change that closed it).
    report = _report(RT_CUDA, RT_HIP, RT_SRC)
    cov = report["coverage"]
    covered = cov["intersection"] - len(cov["missing"])
    assert covered >= 80, f"paste surface still largely invisible: {covered}"
    missing = {m["key"] for m in cov["missing"]}
    assert "getlasterror" not in missing
    assert "peekatlasterror" not in missing
    # Gap detection itself (an unreferenced shared symbol lands in `missing`) is
    # pinned by the synthetic-fixture tests below, independent of which real
    # runtime symbols are wrapped at any given point in the milestone.


def test_runtime_api_whole_surface_passes_clean():
    # THE GATE (#267). runtime_api is now whole-surface: every non-deprecated shared
    # cuda∩hip runtime symbol must be either aliased in detail/runtime_api_names.h
    # or a documented omission in coverage_decisions.json. This runs off committed
    # manifests + source (no SDK), so CI's `test` job fails the day a vendor bump
    # adds a shared symbol nobody wrapped, or a wrapper is removed without an
    # omission. Mirrors test_real_whole_surface_module_passes_clean for fft.
    decision = hi.module_decision(hi.load_decisions(), "runtime_api")
    assert decision is not None
    report = hi.build_report(hi.Side("cuda", RT_CUDA), hi.Side("hip", RT_HIP),
                             RT_SRC, decision)
    cov = report["coverage"]
    assert cov["contract"] == hi.WHOLE_SURFACE
    assert cov["undocumented_gaps"] == [], (
        "unwrapped, undocumented shared runtime symbol(s): "
        f"{[g['cuda']['names'] for g in cov['undocumented_gaps']]}"
    )
    assert cov["fails"] is False


def test_deprecated_shared_symbol_is_not_a_gap(tmp_path):
    # A symbol flagged deprecated in EITHER manifest is auto-omitted from the gap
    # verdict. fooBar is live and referenced; fooBaz is deprecated and
    # unreferenced, so a whole-surface module with no omissions still passes.
    cuda = tmp_path / "c.json"
    cuda.write_text(json.dumps({
        "harvest": {"prefix": "foo"},
        "symbols": [
            {"name": "fooBar", "kind": "FunctionDecl"},
            {"name": "fooBaz", "kind": "FunctionDecl", "deprecated": True},
        ],
    }))
    hip = _write(tmp_path, "h.json", "bar", ["barBar", "barBaz"])
    src = tmp_path / "mod.cppm"
    src.write_text("fooBar barBar")
    decision = {"contract": hi.WHOLE_SURFACE, "omissions": []}
    report = hi.build_report(hi.Side("cuda", [cuda]), hi.Side("hip", [hip]),
                             [src], decision)
    cov = report["coverage"]
    assert cov["fails"] is False
    assert cov["undocumented_gaps"] == []
    assert {i["key"] for i in cov["deprecated_omissions"]} == {"baz"}
