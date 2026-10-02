# src/wrappers/thrust — portable Thrust 3.0 ∩ rocThrust 2.8 subset (audit)

This directory will provide the backend-neutral `wwr.wrappers.thrust` layer: a
single source, written against the `thrust::` names that exist and agree across
both GPU backends, dispatched through a stream-bound execution policy (that shim
is sibling work, issue #237). Unlike the other `src/wrappers/*` layers it is
**not** a raw module over `wwr*` names — device TUs cannot `import`, both
backends already spell the calls `thrust::`, and Thrust ships as headers in each
SDK. See `docs/architecture.md` §2 (why a wrappers-altitude layer), §8 (device
TUs `#include`, never `import`).

This file is the audit that **defines the signatures each A-family may use**
(issues #238–#241). An entry listed as portable here is the contract those
issues build against; an entry flagged non-portable must be avoided or guarded.

## What was verified, and how

Every claim below was checked against the **headers installed in this
container**, not from memory or upstream docs:

| Backend | Version (from `thrust/version.h`) | Header root read |
|---|---|---|
| CCCL Thrust | `THRUST_VERSION 300001` → **3.0.1** (CUDA 13.0) | `/usr/local/cuda-13.0/targets/x86_64-linux/include/cccl/thrust/` |
| rocThrust | `THRUST_VERSION 200805` → **2.8.5** (`ROCTHRUST_VERSION 400200`, ROCm 7.2.4) | `/opt/rocm-7.2.4/include/thrust/` |

Method: for every algorithm in the four families, the per-header free-function
declarations were enumerated in both trees and their overload counts compared;
then each family's headline policy-taking overload was extracted
(template-parameter line through the closing `)`) and **diffed after
normalising away the one cosmetic difference** — CCCL spells the attribute
`_CCCL_HOST_DEVICE`, rocThrust spells it `THRUST_HOST_DEVICE`. After that
normalisation **every in-scope policy overload is byte-identical** across the
two trees. This layer uses the execution-policy-first overload exclusively
(`thrust::<algo>(exec, ...)`); each is confirmed present in both.

The major-version gap (3.0 vs 2.8) did not remove any in-scope algorithm. The
only 3.0 removal that touches these headers is scalar `thrust::min`/`thrust::max`
(see caveats) — out of scope, since the families use `min_element`/`max_element`.

## Per-family status

Legend: **P** = portable, identical policy overload in both; signature column
gives the policy-first overload shape (iterator/functor template params
abbreviated). Overload counts are policy-taking overloads confirmed in both.

### sort/reorder (issue #238) — all portable

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

### reduce (issue #239) — all portable

| Algorithm | Portable policy-first signature | Overloads (both) | Notes |
|---|---|---|---|
| `reduce` | `reduce(exec, first, last[, init[, binary_op]])` | 3 | P — all three init/op overloads agree. |
| `transform_reduce` | `transform_reduce(exec, first, last, unary_op, init, binary_op)` | 1 | P |
| `count` / `count_if` | `count(exec, first, last, value)` / `count_if(exec, first, last, pred)` | 1 each | P |
| `inner_product` | `inner_product(exec, first1, last1, first2, init[, binary_op1, binary_op2])` | 2 | P |
| `min_element` / `max_element` | `min_element(exec, first, last[, comp])` | 2 each | P. `minmax_element` present (2). |

Also present and identical: `reduce_by_key` (3).

### scan (issue #240) — all portable

| Algorithm | Portable policy-first signature | Overloads (both) | Notes |
|---|---|---|---|
| `inclusive_scan` | `inclusive_scan(exec, first, last, out[, binary_op])` | 3 | P |
| `exclusive_scan` | `exclusive_scan(exec, first, last, out[, init[, binary_op]])` | 3 | P |
| `transform_inclusive_scan` | `transform_inclusive_scan(exec, first, last, out, unary_op, binary_op)` | 2 | P |
| `transform_exclusive_scan` | `transform_exclusive_scan(exec, first, last, out, unary_op, init, binary_op)` | 1 | P |

Also present and identical: `inclusive_scan_by_key` (3), `exclusive_scan_by_key` (4).

### transform (issue #241) — all portable

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

## Caveats and must-avoid entries

- **No in-scope algorithm diverges.** Across all four families, every
  policy-taking overload is identical in count and argument shape. Nothing in
  the A-family scope needs a per-backend guard.
- **Scalar `thrust::min` / `thrust::max` are 2.8-only — do not call them.**
  rocThrust 2.8 still declares the scalar comparators in `extrema.h`
  (`T thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(a, b[, comp])`); CCCL 3.0
  **removed** that public spelling from `extrema.h` (it survives only inside
  `thrust/random/*`). This does **not** affect `min_element`/`max_element`,
  which are portable. If the transform/reduce families ever need a scalar
  min/max, use `<algorithm>`/`cuda::std`, not `thrust::min`.
- **`THRUST_PREVENT_MACRO_SUBSTITUTION`** appears only on those scalar
  comparators; no in-scope algorithm uses it, so the layer need not reproduce
  the idiom.
- **Attribute macro name differs** (`_CCCL_HOST_DEVICE` vs `THRUST_HOST_DEVICE`).
  Irrelevant to callers — it is an implementation detail of the declaration, not
  part of the signature — but it is why a raw `diff` of the two headers is noisy
  and the audit normalised it away.
- **Deprecation notes are symmetric.** `transform.h` carries the same 15
  "deprecated" doc mentions in both trees (chiefly "relying on the address of a
  predicate's arguments is deprecated"); none is a removed overload. No action.

## A-transform: the elementwise family (issue #241)

The first family built on this audit. It ships at **two altitudes**, split by
whether the operation needs a caller-supplied functor:

- **Module `wwr.wrappers.thrust` (`elementwise.cppm`)** — the value-based and
  fixed-op surface: `fill`, `sequence`, `replace`, and `transform_unary` /
  `transform_binary` over a small portable op set (`UnaryOp::Negate|Abs|Square`,
  `BinaryOp::Plus|Minus|Multiply`). No caller functor crosses the host/device
  boundary, so each is instantiated once in `elementwise.cu`
  (`thrust::<algo>(wwr::par_on(stream), …)`) and bound by `extern template`,
  exactly as `src/extension/random_normal` is. Import it and call — no device TU
  on the consumer side.
- **Header template `algorithms.cuh`** — `for_each`, `generate`, `tabulate`, and
  a free-functor `transform`. Their whole point is an *arbitrary* caller op,
  which cannot be pre-instantiated, so they are header templates a consumer's own
  `.cu` `#include`s and instantiates with its functor (the same way
  `parallel_for.cuh` is consumed).

Per-algorithm element coverage, justified where it narrows: `fill` / `replace` /
`transform` (negate, square, binary) cover the full numeric set incl. complex
(complex handled through the `wwrC*` device primitives — `cuComplex`/`hipComplex`
have no operators or `operator==`, so `replace` on them routes through
`replace_if` with a component-wise predicate); `sequence` is reals + integers
(no meaningful linear step for complex); `transform` **abs** is reals + signed
integers only (complex magnitude is a different, real-valued type — not an
in-place elementwise map).

### transform vs. parallel_for — reach for which

This family **does not replace** `src/extension/parallel_for`; they coexist at
different altitudes:

| Reach for… | When |
|---|---|
| `wwr.wrappers.thrust` / `algorithms.cuh` | A clean elementwise *map* over contiguous device buffer(s) — `out[i] = op(in[i])`, fills, sequences, replaces — where you want Thrust's typed, range/iterator ergonomics and a one-line call. |
| `parallel_for.cuh` | The lower-level, hand-written index-per-thread kernel: manual indexing, multi-array gather/scatter, or any non-elementwise per-index work. |

`parallel_for` itself once *was* Thrust-backed and **dropped** that backend for
the hand-written `.cuh` (see `src/extension/parallel_for/CMakeLists.txt` and
memory `project-thrust-algorithms-break-under-rdc`); `transform` now *is* Thrust,
at the higher range altitude. The two are complementary, not redundant.

## Scope boundary of this audit

This covers the four A-family algorithm sets named in issue #236. It does **not**
audit iterator adaptors (`counting_iterator`, `transform_iterator`, …), the
`thrust::device_vector` container, `thrust::complex`, or the execution-policy /
stream-binding machinery itself (issue #237) — those are separate surfaces. The
`docs/architecture.md` section that records the design decision is finalised in
issue #242.
