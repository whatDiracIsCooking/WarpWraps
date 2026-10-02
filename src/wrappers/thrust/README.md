# src/wrappers/thrust — the portable Thrust algorithm layer

A typed, backend-neutral algorithm layer over Thrust (CUDA) and rocThrust (HIP):
sort/reorder, scan, reduce and transform, each dispatched through a stream-bound
execution policy so its work lands on the caller's stream. It is **not** a raw
`wwr.cuda.thrust` module over `wwr*` names — Thrust is header-only templates in
`namespace thrust::`, both backends already spell the calls `thrust::`, and a
device TU cannot `import` (`docs/architecture.md` §8). The design rationale lives
in `docs/architecture.md` §22 (why a wrappers-altitude layer, the per-family
shape); the version floors are §21.

## Layout

One subdirectory per algorithm family, each self-contained. The shared pieces at
this level are `execution_policy.cuh` (the `wwr::par_on(stream)` policy shim) and
the `wwr::thrust` INTERFACE target in `CMakeLists.txt` (the backend's Thrust
include dirs and link deps behind one name).

| Family | Module | Device TU | Instantiation list |
|---|---|---|---|
| `reorder/` | `wwr.wrappers.thrust.reorder` | `reorder.cu` | `reorder/instantiations.cpp` (a separate implementation unit) |
| `scan/` | `wwr.wrappers.thrust.scan` | `scan.cu` | folded into `scan.cu` and the `interface.cppm` `extern template` block |
| `reduce/` | `wwr.wrappers.thrust.reduce` | `reduce.cu` | folded into `reduce.cu`; the `extern template`s live in `interface.cppm` |
| `transform/` | `wwr.wrappers.thrust.transform` (+ `transform/algorithms.cuh`) | `transform.cu` | folded into `transform.cu` |

The families are **not internally uniform**: reorder keeps a standalone
`instantiations.cpp`, where the other three fold the explicit instantiations
directly into their device `.cu`. Transform additionally carries
`algorithms.cuh` — the header-template half for caller-supplied functors (see
"transform vs parallel_for" below). Do not assume a single template when editing
one; match the family you are in.

## Per-family element-type coverage, as implemented

Coverage is **per-family, not a uniform `sdcz` list** — each family instantiates
only the element types its algorithms are meaningful for. The sets below are
transcribed from the explicit instantiations in each family's sources (the `.cu`
`WWR_*_INSTANTIATE` lists and, for reorder, `instantiations.cpp`); a call with a
type outside its family's set fails to link (or, for scan, is a compile error via
the `scan_scalar` concept).

| Family | `float` | `double` | `int32` | `int64` | `uint32` | `uint64` | complex | Why it narrows |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **reorder** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | sort/unique/partition need an ordering or equality; complex has neither, so it is excluded family-wide. The unsigned integers *are* orderable, so they stay. |
| **reduce** | ✓ | ✓ | ✓ | ✓ | — | — | — | A total order for `min_element`/`max_element`, native device arithmetic for the sums, and no dependence on the unaudited `thrust::complex`. The unsigned ints are simply not needed by the reduce surface. |
| **scan** | ✓ | ✓ | ✓ (`int`) | ✓ | — | — | — | `thrust::plus`/`negate` and (for the `max` variant) a sensible order. Complex excluded: `cuComplex` is an operator-less `float2` aggregate, so `thrust::plus` over it does not compile on CUDA without a hand-written functor (out of arithmetic scope). |
| **transform** | ✓ | ✓ | ✓ (`int`) | ✓ (`long long`) | see note | see note | partial | Per-*algorithm*, not per-family — see the breakdown below. |

Notes on the integer spellings: reorder and reduce instantiate the fixed-width
`std::int32_t`/`std::int64_t` (plus the `std::uint*` for reorder); scan and
transform instantiate the built-in `int`/`long long`. These coincide with the
fixed-width types on this toolchain (LP64) but are spelled as the family's
sources spell them.

### transform is per-algorithm

The transform family's coverage varies by algorithm, because what is meaningful
differs:

| Algorithm | Element types instantiated | Complex? |
|---|---|:---:|
| `fill` | float, double, int, unsigned int, long long, unsigned long long, `wwrFloatComplex`, `wwrDoubleComplex` | ✓ |
| `replace` | same as `fill` | ✓ (via `replace_if` + a component-wise predicate — `cuComplex`/`hipComplex` have no `operator==`) |
| `transform_binary` (Plus/Minus/Multiply) | same as `fill` | ✓ (via the `wwrC*` device functors) |
| `transform_unary` Negate, Square | float, double, int, long long, `wwrFloatComplex`, `wwrDoubleComplex` | ✓ |
| `transform_unary` Abs | float, double, int, long long | — (complex magnitude is a different, real-valued type — not an in-place elementwise map) |
| `sequence` | float, double, int, unsigned int, long long, unsigned long long | — (no meaningful linear step for complex) |

The functor-taking algorithms (`for_each`, `generate`, `tabulate`, free-functor
`transform`) are header templates in `algorithms.cuh`, instantiated with the
caller's own type and functor in the caller's device TU — so they carry no fixed
type list at all.

> **Coverage home.** This table is the layer's own record. It is deliberately
> *not* in `devtools/coverage_decisions.json`, which governs the whole-surface
> vendor modules (`rand`, `fft`, …) — a machine-harvestable `.so` surface. This
> layer is header-only template code with no `.so` and no manifest, so its
> coverage is a hand-curated per-family decision recorded here.

## transform vs. parallel_for — reach for which

The transform family **does not replace** `src/extension/parallel_for`; they
coexist at different altitudes.

| Reach for… | When |
|---|---|
| `wwr.wrappers.thrust.transform` / `transform/algorithms.cuh` | A clean elementwise *map* over contiguous device buffer(s) — `out[i] = op(in[i])`, fills, sequences, replaces — where you want Thrust's typed range/iterator ergonomics and a one-line call. |
| `parallel_for.cuh` | The lower-level, hand-written index-per-thread kernel: manual indexing, multi-array gather/scatter, or any non-elementwise per-index work. |

`parallel_for` itself once *was* Thrust-backed and **dropped** that backend for
the hand-written `.cuh` (`src/extension/parallel_for/CMakeLists.txt` and memory
`project-thrust-algorithms-break-under-rdc`); `transform` now *is* Thrust, at the
higher range altitude. The two are complementary, not redundant.

## Extending the layer

The portable subset audited and wrapped today is the four A-families below.
Further families — search, gather, set operations — are extensible as sibling
subdirectories following the same `interface.cppm` + `.cu` + bridge-header shape,
and are out of scope for the current milestone.

---

## Appendix: the portability audit (CCCL 3.0.1 ∩ rocThrust 2.8.5)

This is the audit that **defined the signatures each A-family was built against**
(issues #238–#241). An entry listed as portable here is the contract those
issues built on; an entry flagged non-portable had to be avoided or guarded.

### What was verified, and how

Every claim below was checked against the **headers installed in the container**,
not from memory or upstream docs:

| Backend | Version (from `thrust/version.h`) | Header root read |
|---|---|---|
| CCCL Thrust | `THRUST_VERSION 300001` → **3.0.1** (CUDA 13.0) | `/usr/local/cuda-13.0/targets/x86_64-linux/include/cccl/thrust/` |
| rocThrust | `THRUST_VERSION 200805` → **2.8.5** (`ROCTHRUST_VERSION 400200`, ROCm 7.2.4) | `/opt/rocm-7.2.4/include/thrust/` |

Method: for every algorithm in the four families, the per-header free-function
declarations were enumerated in both trees and their overload counts compared;
then each family's headline policy-taking overload was extracted
(template-parameter line through the closing `)`) and **diffed after normalising
away the one cosmetic difference** — CCCL spells the attribute
`_CCCL_HOST_DEVICE`, rocThrust spells it `THRUST_HOST_DEVICE`. After that
normalisation **every in-scope policy overload is byte-identical** across the two
trees. This layer uses the execution-policy-first overload exclusively
(`thrust::<algo>(exec, ...)`); each is confirmed present in both.

The major-version gap (3.0 vs 2.8) did not remove any in-scope algorithm. The
only 3.0 removal that touches these headers is scalar `thrust::min`/`thrust::max`
(see caveats) — out of scope, since the families use `min_element`/`max_element`.

### Per-family status

Legend: **P** = portable, identical policy overload in both; the signature column
gives the policy-first overload shape (iterator/functor template params
abbreviated). Overload counts are policy-taking overloads confirmed in both.

#### sort/reorder (issue #238) — all portable

| Algorithm | Portable policy-first signature | Overloads (both) | Notes |
|---|---|---|---|
| `sort` | `sort(exec, first, last[, comp])` | 2 | P |
| `unique` | `unique(exec, first, last[, pred])` | 2 | P. `unique_copy`, `unique_by_key`, `unique_by_key_copy` also present (2 each); `unique_count` present (2). |
| `partition` | `partition(exec, first, last, pred)` | 2 | P. `partition_copy`, `stable_partition`, `stable_partition_copy` present (2 each); `is_partitioned`, `partition_point` present (1 each). |
| `remove` / `remove_if` | `remove_if(exec, first, last, pred)` | 2 | P. `remove` (1), `remove_copy` (1), `remove_copy_if` (2) present. |
| `copy_if` | `copy_if(exec, first, last[, stencil], out, pred)` | 2 | P — both the 4-arg and the stencil (5-arg) form exist in both. `copy` (1), `copy_n` (1) present. |
| `reverse` | `reverse(exec, first, last)` | 1 | P. `reverse_copy` (1) present. |

Also present and identical: `stable_sort`, `sort_by_key`, `stable_sort_by_key`
(2 each); `is_sorted`, `is_sorted_until` (2 each).

#### reduce (issue #239) — all portable

| Algorithm | Portable policy-first signature | Overloads (both) | Notes |
|---|---|---|---|
| `reduce` | `reduce(exec, first, last[, init[, binary_op]])` | 3 | P — all three init/op overloads agree. |
| `transform_reduce` | `transform_reduce(exec, first, last, unary_op, init, binary_op)` | 1 | P |
| `count` / `count_if` | `count(exec, first, last, value)` / `count_if(exec, first, last, pred)` | 1 each | P |
| `inner_product` | `inner_product(exec, first1, last1, first2, init[, binary_op1, binary_op2])` | 2 | P |
| `min_element` / `max_element` | `min_element(exec, first, last[, comp])` | 2 each | P. `minmax_element` present (2). |

Also present and identical: `reduce_by_key` (3).

#### scan (issue #240) — all portable

| Algorithm | Portable policy-first signature | Overloads (both) | Notes |
|---|---|---|---|
| `inclusive_scan` | `inclusive_scan(exec, first, last, out[, binary_op])` | 3 | P |
| `exclusive_scan` | `exclusive_scan(exec, first, last, out[, init[, binary_op]])` | 3 | P |
| `transform_inclusive_scan` | `transform_inclusive_scan(exec, first, last, out, unary_op, binary_op)` | 2 | P |
| `transform_exclusive_scan` | `transform_exclusive_scan(exec, first, last, out, unary_op, init, binary_op)` | 1 | P |

Also present and identical: `inclusive_scan_by_key` (3), `exclusive_scan_by_key` (4).

#### transform (issue #241) — all portable

| Algorithm | Portable policy-first signature | Overloads (both) | Notes |
|---|---|---|---|
| `transform` | `transform(exec, first, last, [first2,] out, op)` | 2 | P — unary and binary forms. |
| `for_each` | `for_each(exec, first, last, f)` → returns iterator | 1 | P. `for_each_n` (1) present; both return `InputIterator`. |
| `fill` / `fill_n` | `fill(exec, first, last, value)` | 1 each | P |
| `generate` / `generate_n` | `generate(exec, first, last, gen)` | 1 each | P |
| `sequence` | `sequence(exec, first, last[, init[, step]])` | 3 | P |
| `tabulate` | `tabulate(exec, first, last, op)` | 1 | P |
| `replace` | `replace(exec, first, last, old_value, new_value)` | 1 | P. `replace_if` (2), `replace_copy` (1), `replace_copy_if` (2) present. |

Also present and identical: `transform_if` (3).

### Caveats and must-avoid entries

- **No in-scope algorithm diverges.** Across all four families, every
  policy-taking overload is identical in count and argument shape. Nothing in the
  A-family scope needs a per-backend guard.
- **Scalar `thrust::min` / `thrust::max` are 2.8-only — do not call them.**
  rocThrust 2.8 still declares the scalar comparators in `extrema.h`
  (`T thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(a, b[, comp])`); CCCL 3.0
  **removed** that public spelling from `extrema.h` (it survives only inside
  `thrust/random/*`). This does **not** affect `min_element`/`max_element`, which
  are portable. If a family ever needs a scalar min/max, use
  `<algorithm>`/`cuda::std`, not `thrust::min`.
- **`THRUST_PREVENT_MACRO_SUBSTITUTION`** appears only on those scalar
  comparators; no in-scope algorithm uses it, so the layer need not reproduce the
  idiom.
- **Attribute macro name differs** (`_CCCL_HOST_DEVICE` vs `THRUST_HOST_DEVICE`).
  Irrelevant to callers — it is an implementation detail of the declaration, not
  part of the signature — but it is why a raw `diff` of the two headers is noisy
  and the audit normalised it away.
- **Deprecation notes are symmetric.** `transform.h` carries the same 15
  "deprecated" doc mentions in both trees (chiefly "relying on the address of a
  predicate's arguments is deprecated"); none is a removed overload. No action.

### Scope boundary of this audit

This covered the four A-family algorithm sets named in issue #236. It does
**not** audit iterator adaptors (`counting_iterator`, `transform_iterator`, …),
the `thrust::device_vector` container, `thrust::complex`, or the
execution-policy / stream-binding machinery itself — those are separate
surfaces. The `docs/architecture.md` section that records the design decision is
§22.
