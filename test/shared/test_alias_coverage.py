"""Tests for alias_coverage.py, the gpu* layer alias-coverage guard.

A coverage checker's failure mode is passing when it should fail -- a parser that
silently matches nothing reports "all covered". So beyond asserting the real tree
is fully covered, these pin the parser against the two shapes that actually fooled
an early cut of it: the GPUMOD_RT_FUNCTION prefix-paste and the macro #define line.
"""

from __future__ import annotations

from pathlib import Path

import alias_coverage as ac

ROOT = Path(__file__).resolve().parents[2]


def test_real_tree_is_fully_covered():
    modules = ac.discover_modules(ROOT)
    assert modules, "discovered no gpu modules -- alias parser is broken"
    gaps = {m: ac.check_module(ROOT, m) for m in modules}
    uncovered = {m: g for m, g in gaps.items() if g}
    assert not uncovered, f"uncovered gpu* aliases: {uncovered}"


def test_parses_gpu_function():
    src = "GPUMOD_FUNCTION(gpublasSgemm, cublasSgemm_v2, hipblasSgemm)\n"
    assert ac.parse_aliases(src) == {"gpublasSgemm"}


def test_parses_gpu_rt_function_prefix_paste():
    # GPUMOD_RT_FUNCTION(StreamCreate) -> alias gpuStreamCreate
    src = "GPUMOD_RT_FUNCTION(StreamCreate)\n"
    assert ac.parse_aliases(src) == {"gpuStreamCreate"}


def test_skips_macro_definition_lines():
    # The #define lines are not invocations and must not be read as aliases.
    src = (
        "#define GPUMOD_FUNCTION(gpu_name, cuda_name, hip_name) ...\n"
        "#define GPUMOD_RT_FUNCTION(x) GPUMOD_FUNCTION(gpu##x, cuda##x, hip##x)\n"
    )
    assert ac.parse_aliases(src) == set()


def test_parses_same_function():
    test = "GPUMOD_SAME_FUNCTION(gpublasCreate, cublasCreate_v2)\n"
    assert ac.parse_same_functions(test) == {"gpublasCreate"}


def test_parses_same_function_not_on_first_line():
    # Real test files have GPUMOD_SAME_FUNCTION on many lines, not line 1: a whole-text
    # ^-anchored regex without re.MULTILINE silently matches only the first.
    text = (
        "// header comment\n"
        "\n"
        "GPUMOD_SAME_FUNCTION(gpublasCreate, cublasCreate_v2)\n"
        "GPUMOD_SAME_FUNCTION(gpublasDestroy, hipblasDestroy)\n"
    )
    assert ac.parse_same_functions(text) == {"gpublasCreate", "gpublasDestroy"}


def test_parses_toml_targets_nested():
    data = {"dispatch": {"gemm<float, int>": ["gpublasSgemm"],
                         "asum<double, int>": ["gpublasDasum"]},
            "type_names": {"CUDA": {"float2": "gpuComplex"}}}
    targets = ac.parse_toml_targets(data)
    assert {"gpublasSgemm", "gpublasDasum"} <= targets


def test_gap_flags_uncovered_alias():
    aliases = {"gpublasSgemm", "gpublasDasum"}
    same = {"gpublasSgemm"}
    toml_targets: set[str] = set()
    assert ac.coverage_gap(aliases, same, toml_targets) == {"gpublasDasum"}


def test_gap_accepts_same_function_coverage():
    aliases = {"gpublasCreate"}
    assert ac.coverage_gap(aliases, {"gpublasCreate"}, set()) == set()


def test_gap_accepts_toml_coverage():
    aliases = {"gpublasSgemm"}
    assert ac.coverage_gap(aliases, set(), {"gpublasSgemm"}) == set()
