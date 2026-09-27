"""Tests for alias_coverage.py, the wwr* layer alias-coverage guard.

A coverage checker's failure mode is passing when it should fail -- a parser that
silently matches nothing reports "all covered". So beyond asserting the real tree
is fully covered, these pin the parser against the two shapes that actually fooled
an early cut of it: the WWR_RT_FUNCTION prefix-paste and the macro #define line.
"""

from __future__ import annotations

from pathlib import Path

import alias_coverage as ac

ROOT = Path(__file__).resolve().parents[2]


def test_real_tree_is_fully_covered():
    modules = ac.discover_modules(ROOT)
    assert modules, "discovered no wwr modules -- alias parser is broken"
    gaps = {m: ac.check_module(ROOT, m) for m in modules}
    uncovered = {m: g for m, g in gaps.items() if g}
    assert not uncovered, f"uncovered wwr* aliases: {uncovered}"


def test_parses_gpu_function():
    src = "WWR_FUNCTION(wwrblasSgemm, cublasSgemm_v2, hipblasSgemm)\n"
    assert ac.parse_aliases(src) == {"wwrblasSgemm"}


def test_parses_gpu_rt_function_prefix_paste():
    # WWR_RT_FUNCTION(StreamCreate) -> alias wwrStreamCreate
    src = "WWR_RT_FUNCTION(StreamCreate)\n"
    assert ac.parse_aliases(src) == {"wwrStreamCreate"}


def test_skips_macro_definition_lines():
    # The #define lines are not invocations and must not be read as aliases.
    src = (
        "#define WWR_FUNCTION(wwr_name, cuda_name, hip_name) ...\n"
        "#define WWR_RT_FUNCTION(x) WWR_FUNCTION(wwr##x, cuda##x, hip##x)\n"
    )
    assert ac.parse_aliases(src) == set()


def test_parses_same_function():
    test = "WWR_SAME_FUNCTION(wwrblasCreate, cublasCreate_v2)\n"
    assert ac.parse_same_functions(test) == {"wwrblasCreate"}


def test_parses_same_function_not_on_first_line():
    # Real test files have WWR_SAME_FUNCTION on many lines, not line 1: a whole-text
    # ^-anchored regex without re.MULTILINE silently matches only the first.
    text = (
        "// header comment\n"
        "\n"
        "WWR_SAME_FUNCTION(wwrblasCreate, cublasCreate_v2)\n"
        "WWR_SAME_FUNCTION(wwrblasDestroy, hipblasDestroy)\n"
    )
    assert ac.parse_same_functions(text) == {"wwrblasCreate", "wwrblasDestroy"}


def test_parses_toml_targets_nested():
    data = {"dispatch": {"gemm<float, int>": ["wwrblasSgemm"],
                         "asum<double, int>": ["wwrblasDasum"]},
            "type_names": {"CUDA": {"float2": "wwrComplex"}}}
    targets = ac.parse_toml_targets(data)
    assert {"wwrblasSgemm", "wwrblasDasum"} <= targets


def test_gap_flags_uncovered_alias():
    aliases = {"wwrblasSgemm", "wwrblasDasum"}
    same = {"wwrblasSgemm"}
    toml_targets: set[str] = set()
    assert ac.coverage_gap(aliases, same, toml_targets) == {"wwrblasDasum"}


def test_gap_accepts_same_function_coverage():
    aliases = {"wwrblasCreate"}
    assert ac.coverage_gap(aliases, {"wwrblasCreate"}, set()) == set()


def test_gap_accepts_toml_coverage():
    aliases = {"wwrblasSgemm"}
    assert ac.coverage_gap(aliases, set(), {"wwrblasSgemm"}) == set()
