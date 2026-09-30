# devtools/harvest_aggregates

Hand-authored aggregation headers for `devtools/vendor_harvest.py`. They are
**inputs** to the harvester, not manifests — the only hand-written files in the
whole vendor-manifest pipeline.

A manifest harvests one header. A few wrapped libraries do not expose their full
surface through a single umbrella:

- **`nvcomp.h` / `hipcomp.h`** are thin umbrellas that pull in only
  `shared_types.h` / `version.h`, not the per-algorithm batched headers
  (`nvcomp/lz4.h`, …). The corresponding `.cppm` `#include`s each algorithm
  header explicitly, so its surface is the union of all of them.
- **`hiprand_kernel.h`** calls bare `printf` without `#include <cstdio>` of its
  own; a plain (non-`-x hip`) compile needs `<cstdio>` pre-included, exactly as
  `src/hip/hiprand_kernel.cppm` does.
- **`hiptensor.h`** has a version-dependent include set: 2.2.0 (ROCm 7.2, the
  pin) added the C-linkage `hiptensor.h`, while 2.1.0 (ROCm 7.1, the floor) ships
  only `hiptensor.hpp`, which — unlike the `.h` — does not pull
  `hiptensor-version.hpp`. The aggregate's `__has_include` picks the header the
  same way `src/hip/hiptensor.cppm` does and adds the version header on the floor
  path, so `hiptensorGetVersion` is declared at both ends; the load-bearing
  `<array>` pre-include is carried too.

Each file below mirrors the vendor `#include` set of the matching `.cppm`'s
global module fragment one-for-one, so the manifest is the surface that module
actually wraps. Keep them in lockstep with the `.cppm` if its include set
changes.
