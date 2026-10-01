# `example/`

Three things, and only the first two are part of this build.

| Directory | What it is | Who builds it |
|---|---|---|
| `warp_reduce/` | A warp-level reduction kernel written against the `wwr*` layer | the main build, on either backend |
| `buffer_suite/` | A tour of the memory-buffer suite — the policy **map** and views | the main build, on either backend |
| `consumer/` | A standalone project consuming an **installed** wwr | `devtools/install-check.sh` |

`consumer/` has its own `project()` call and is not added by
`example/CMakeLists.txt`: the whole point is that it knows nothing about this
source tree and reaches wwr only through `find_package(wwr)`.

## `warp_reduce/` — an exemplar, not a library

It is the answer to "how do I write my own warp-level kernel against wwr?".
One block of four warps, each thread cascading over the whole array, then a
`tile.shfl_down` ladder — Mark Harris's reduction in cooperative groups, so any
`count` is reduced by one kernel with no second pass and no cooperative launch.
The cost is occupancy: one block occupies one SM / CU.

It ships as an example rather than a `wwr.extension.*` module on purpose. A
reduction has no single right shape — single-block versus two-pass, the
operator, the accumulator type, whether the answer is bit-reproducible — so a
shipped one would be a set of choices the caller cannot revisit. As source to
copy, the choices are theirs. `wwr.extension.parallel_for` is what a
reduction *would* be built on, and it stays a library because an
index-per-thread map has no such choices in it.

It is also the in-tree consumer of `src/cooperative_groups.h`.
`test/gpu/cooperative_groups.cu` compiles every portable entity that header
reaches; this is the only thing that launches one and checks the answer.

## `buffer_suite/` — the suite's two entry points

It is the answer to "how do I stop writing the buffer alias prelude by hand?".
The `:suite` partition of `wwr.extension.memory_buffer` binds the per-kind buffer
aliases *and* their views from one place. `warp_reduce` already shows the terse
**convenience** (`device_buffers<P, H>`, one policy across every kind); this shows
the other entry point, the **map** (`device_buffer_suite<M, H>`), which is what you
reach for when the `alloc` and `free` slots want different policies — here, abort
on an allocation failure but only warn on a free, a split the single-policy
convenience cannot express. It also uses a `host_view`, which the suite derives
from its buffer alias so the two cannot drift apart.

No kernel and no `.cu`: it exercises only the host-callable RAII + copy surface (a
host→device→host round-trip and a sub-view check), so it is one ordinary host
module translation unit.

## Running it

Needs a real device. Building either does not — that is what
`devtools/cross-backend-check.sh` does for the backend you are not on.

```bash
devtools/cpp-tier.sh                    # or any preset that builds the tree
./build/example/warp_reduce/wwr.example.warp_reduce
./build/example/buffer_suite/wwr.example.buffer_suite
```

Exit status is 0 on success, 1 on a wrong answer, and 77 (ctest's "skipped")
when no device is reachable. There is no ctest entry for either: the runtime tier
has nothing device-dependent in it, and registering these would turn every
driverless box red.
