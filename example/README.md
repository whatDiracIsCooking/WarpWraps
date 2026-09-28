# `example/`

Two different things, and only one of them is part of this build.

| Directory | What it is | Who builds it |
|---|---|---|
| `warp_reduce/` | A warp-level reduction kernel written against the `gpu*` layer | the main build, on either backend |
| `custom_default_error_policy/` | A drop-in replacement for the extension layer's default error policy | the main build, on either backend |
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

It is also the in-tree consumer of `src/cooperative_groups.cuh`.
`test/gpu/cooperative_groups.cu` compiles every portable entity that header
reaches; this is the only thing that launches one and checks the answer.

## `custom_default_error_policy/` — swapping the built-in default policy

`wwr::extension::DefaultErrorPolicy<T>` is the policy every extension-layer
default-policy slot falls back to — the argless `gpu_check`, and the
`P_create` / `P_alloc` / `P_free` / `P_destroy` template-argument defaults on
the RAII wrappers. Its built-in body prints to `stderr` and `std::abort()`s. A
build can replace that body without editing any wwr source or touching a single
call site: point the CMake cache variables at a **whole partition file** of your
own when you configure the tree —

```bash
cmake --preset default \
  -DWWR_DEFAULT_ERROR_POLICY_MODULE="$PWD/example/custom_default_error_policy/custom_default_error_policy.cppm" \
  -DWWR_DEFAULT_ERROR_POLICY_LINK=wwr.example.error_logger
cmake --build build --target wwr.example.custom_default_error_policy
```

and the `:default_error_policy` partition — and so `DefaultErrorPolicy<T>` —
comes from your file instead of the built-in. `custom_default_error_policy.cppm`
here logs and *continues* rather than aborting; its comments carry the one hard
rule — `handle_error` must be `noexcept`, because the destruction-slot policies
are `nothrow_error_policy`.

Because the replacement is a real module unit, its purview `import`s whatever it
needs: `import :error_code;` to reuse wwr's own `error_name` / `error_string`
(which a global-module-fragment header could not reach), and
`import wwr.example.error_logger;` to route through a module of your own. Name any
such module target in `WWR_DEFAULT_ERROR_POLICY_LINK` so it is linked into the
error-handling module. This mirrors the tree's own convention — imports live in
the module purview, never the global module fragment (see `src/gpu_backend.h` and
`src/complex.cppm`).

The choice is **build-wide**: it re-compiles the one shared error-handling
module, so every target in the tree — including the test suites — gets the
swapped policy. That is why this ships as a configure option rather than a
per-target one.

### Running it

Needs no device. The demonstration is to run it *both ways* and compare:

```bash
# Default build (no -D): the built-in policy prints and aborts (exit 134).
./build/example/custom_default_error_policy/wwr.example.custom_default_error_policy

# Reconfigured with -DWWR_DEFAULT_ERROR_POLICY_MODULE=…/custom_default_error_policy.cppm
# (and -DWWR_DEFAULT_ERROR_POLICY_LINK=wwr.example.error_logger): the custom
# policy logs "[wwr.example.error_logger] …" and returns, so gpu_check yields
# false and the program exits 0.
```

There is no ctest entry: in a default build the program aborts on purpose (it
triggers an error to show the policy running), which would turn a driverless
CI run red.

> The matching CMake helper for **installed** (`find_package(wwr)`) consumers is
> not wired yet — the extension layer is built in-tree but not installed (see
> `cmake/wwr_install.cmake`). Until it is, this option is the from-source way to
> customize the default policy.

## Running it

Needs a real device. Building it does not — that is what
`devtools/cross-backend-check.sh` does for the backend you are not on.

```bash
devtools/cpp-tier.sh                    # or any preset that builds the tree
./build/example/warp_reduce/wwr.example.warp_reduce
```

Exit status is 0 on success, 1 on a wrong answer, and 77 (ctest's "skipped")
when no device is reachable. There is no ctest entry: the runtime tier has
nothing device-dependent in it, and registering this would turn every
driverless box red.
