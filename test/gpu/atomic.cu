// Compile-time test for src/atomic.cuh, the scoped/ordered atomic forwarders.
// This is the surface ABOVE the bare common atomics: test/gpu/atomics.cu (note
// the plural) keeps its own job unchanged -- the atomicAdd/CAS/... that both
// backends spell identically and wwr does NOT wrap. Here every operation
// carries an explicit memory order and thread scope, where the two backends
// diverge (cuda::atomic_ref<T, Scope> vs a __hip_atomic_* builtin), so a
// forwarder does work rather than renaming a name to itself.
//
// Building this .cu under the selected backend's device pass IS the test: every
// forwarder x every portable scope named below has to resolve through
// atomic.cuh's one #include switch on both backends. The kernel is never
// launched -- ordering *semantics* cannot be proven by a compile, and a runtime
// memory-model test is a flake generator, so the compile is the honest limit:
// it pins that every portable combination resolves, not that the orderings mean
// the same thing (that is the documentation claim in docs/architecture.md §20,
// read off both memory models). No ctest entry, no launch, matching
// atomics.cu / cooperative_groups.cu / wmma.cu.
//
// FP atomics ride the default (safe) codegen: -munsafe-fp-atomics is opt-in and
// not passed here, so atomicAdd on float/double lowers to AMD's correct CAS
// fallback. See docs/architecture.md section 15.
#include "atomic.cuh"

namespace {

using wwr::wwrMemoryOrder;
using wwr::wwrThreadScope;

// Every forwarder at one scope, across the type surface each operation supports
// on both backends: integer widths carry the whole set; floating-point carries
// fetch_add (the one FP atomic §15 fixes as portable and safe). min/max on FP
// has no portable guarantee here, so a TU wanting it names the vendor form.
// Templated on the scope so the four portable rows are each exercised by an
// instantiation in the kernel below.
template<wwrThreadScope Scope>
__device__ void wwr_atomic_exercise(int *i, unsigned *u, unsigned long long *ull, float *f,
                                    double *d) {
  // load / store / exchange, each with an explicit order.
  (void)wwr::wwrAtomicLoad<Scope>(i, wwrMemoryOrder::acquire);
  wwr::wwrAtomicStore<Scope>(i, *i, wwrMemoryOrder::release);
  (void)wwr::wwrAtomicExchange<Scope>(i, *i, wwrMemoryOrder::acq_rel);

  // compare-exchange, strong and weak, with distinct success/failure orders --
  // the two-order shape both backends take (the failure order stays a load
  // order, acquire here, as both memory models require).
  int expected = *i;
  (void)wwr::wwrAtomicCompareExchangeStrong<Scope>(i, expected, *i, wwrMemoryOrder::acq_rel,
                                                   wwrMemoryOrder::acquire);
  (void)wwr::wwrAtomicCompareExchangeWeak<Scope>(i, expected, *i, wwrMemoryOrder::acq_rel,
                                                 wwrMemoryOrder::acquire);

  // integer arithmetic and bitwise, on the widths both backends carry. A bare
  // call takes the default seq_cst order; an explicit one is spot-checked above.
  (void)wwr::wwrAtomicFetchAdd<Scope>(i, *i, wwrMemoryOrder::relaxed);
  (void)wwr::wwrAtomicFetchAdd<Scope>(u, *u);
  (void)wwr::wwrAtomicFetchAdd<Scope>(ull, *ull);
  (void)wwr::wwrAtomicFetchSub<Scope>(i, *i);
  (void)wwr::wwrAtomicFetchAnd<Scope>(u, *u);
  (void)wwr::wwrAtomicFetchOr<Scope>(u, *u);
  (void)wwr::wwrAtomicFetchXor<Scope>(u, *u);
  (void)wwr::wwrAtomicFetchMin<Scope>(i, *i);
  (void)wwr::wwrAtomicFetchMax<Scope>(ull, *ull);

  // floating-point add, default safe codegen (see the file header).
  (void)wwr::wwrAtomicFetchAdd<Scope>(f, *f, wwrMemoryOrder::relaxed);
  (void)wwr::wwrAtomicFetchAdd<Scope>(d, *d);
}

} // namespace

// Every forwarder x every portable scope -- thread, block, device, system --
// the instantiation is the assertion. The two vendor-only scopes (CUDA's
// cluster, HIP's wavefront) are deliberately absent: they have no portable
// counterpart, exactly as §15 leaves atomicAdd_block/_system out. Reached
// through wwr.device, which carries atomic.cuh and (on CUDA >= 13) the cccl
// include dir <cuda/atomic> needs.
__global__ void wwr_atomic_all_scopes(int *i, unsigned *u, unsigned long long *ull, float *f,
                                      double *d) {
  wwr_atomic_exercise<wwrThreadScope::thread>(i, u, ull, f, d);
  wwr_atomic_exercise<wwrThreadScope::block>(i, u, ull, f, d);
  wwr_atomic_exercise<wwrThreadScope::device>(i, u, ull, f, d);
  wwr_atomic_exercise<wwrThreadScope::system>(i, u, ull, f, d);
}
