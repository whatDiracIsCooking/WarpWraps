# vendor/ — committed vendor-surface manifests (machine output)

Every `*.json` here is the **declared surface of one wrapped vendor library**,
harvested by `devtools/vendor_harvest.py` from the pinned SDK headers. They are
**machine output — never hand-edit them.** Each file's `"harvest"` block records
the exact clang command, the config tuple (backend, SDK version, arch, defines,
includes), the clang version and the source header that produced it, so a
manifest is reproducible and a diff between two is meaningful only when their
tuples agree. See `devtools/vendor_harvest.py`'s header for what a manifest is
and is not.

## Layout

One directory per SDK pin, one file per wrapped library:

- `cuda-13.0.x/` — the CUDA backend, harvested against the toolkit the
  `:cuda-ci` image ships (CUDA 13.0.x; NCCL / cuTENSOR / nvCOMP arrive by apt).
- `rocm-7.2.4/` — the HIP backend, harvested against ROCm 7.2.4 (arch `gfx942`),
  plus hipCOMP from `/opt/rocm-ds`.

## Regenerating

`devtools/vendor_manifests.sh` is the single source of the harvest invocations
(prefix, defines, include roots per library). Run it **inside the matching
pinned `-ci` image** — the version dir names encode the pin, and the manifests
are only reproducible against those exact headers:

```bash
devtools/vendor_manifests.sh cuda   # rewrites cuda-13.0.x/
devtools/vendor_manifests.sh hip    # rewrites rocm-7.2.4/
devtools/vendor_manifests.sh both   # both (needs the combined image)
```

The `vendor-manifests` CI job regenerates into a throwaway tree and fails on any
diff, printing it — the honesty mechanism that keeps these files from going
stale-but-trusted. A red diff is either a manifest nobody regenerated after an
intended change, or a new SDK point release changing the surface under a fixed
pin.

## Coverage

`devtools/vendor_manifests.sh`'s header lists the deliberate gaps: the
type-wrapper modules (`cuda_fp16`/`bf16`/`fp8`/`fp4`/`fp6`, `cuComplex` and
their HIP twins) and `cufile` have no single vendor prefix the name-mode
harvester can key on, so no meaningful manifest exists for them. `cublasLt` /
`cublasXt` / `cusolverMg` / `cusolverSp` are documented supersets (their prefix
also captures the shared base surface).
