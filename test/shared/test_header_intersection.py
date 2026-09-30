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

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "devtools"))

import header_intersection as hi  # noqa: E402

CUDA = ROOT / "vendor" / "cuda-13.0.x"
ROCM = ROOT / "vendor" / "rocm-7.2.4"

RAND_CUDA = [CUDA / "curand.json", CUDA / "curand_kernel.json"]
RAND_HIP = [ROCM / "hiprand.json", ROCM / "hiprand_kernel.json"]
RAND_SRC = [ROOT / "src" / "rand.cppm", ROOT / "src" / "rand.cuh"]

BLAS_CUDA = [CUDA / "cublas_v2.json"]
BLAS_HIP = [ROCM / "hipblas.json"]
BLAS_SRC = [ROOT / "src" / "blas.cppm"]


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
