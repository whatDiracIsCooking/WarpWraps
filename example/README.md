# `example/`

Two different things, and only one of them is part of this build.

| Directory | What it is | Who builds it |
|---|---|---|
| `warp_reduce/` | A warp-level reduction kernel written against the `gpu*` layer | the main build, on either backend |
| `consumer/` | A standalone project consuming an **installed** gpumod | `devtools/install-check.sh` |

`consumer/` has its own `project()` call and is not added by
`example/CMakeLists.txt`: the whole point is that it knows nothing about this
source tree and reaches gpumod only through `find_package(gpumod)`.

## `warp_reduce/` — an exemplar, not a library

It is the answer to "how do I write my own warp-level kernel against gpumod?".
One block of four warps, each thread cascading over the whole array, then a
`tile.shfl_down` ladder — Mark Harris's reduction in cooperative groups, so any
`count` is reduced by one kernel with no second pass and no cooperative launch.
The cost is occupancy: one block occupies one SM / CU.

It ships as an example rather than a `gpumod.extension.*` module on purpose. A
reduction has no single right shape — single-block versus two-pass, the
operator, the accumulator type, whether the answer is bit-reproducible — so a
shipped one would be a set of choices the caller cannot revisit. As source to
copy, the choices are theirs. `gpumod.extension.parallel_for` is what a
reduction *would* be built on, and it stays a library because an
index-per-thread map has no such choices in it.

It is also the in-tree consumer of `src/cooperative_groups.cuh`.
`test/gpu/cooperative_groups.cu` compiles every portable entity that header
reaches; this is the only thing that launches one and checks the answer.

## Running it

Needs a real device. Building it does not — that is what
`devtools/cross-backend-check.sh` does for the backend you are not on.

```bash
devtools/cpp-tier.sh                    # or any preset that builds the tree
./build/example/warp_reduce/gpumod.example.warp_reduce
```

Exit status is 0 on success, 1 on a wrong answer, and 77 (ctest's "skipped")
when no device is reachable. There is no ctest entry: the runtime tier has
nothing device-dependent in it, and registering this would turn every
driverless box red.
