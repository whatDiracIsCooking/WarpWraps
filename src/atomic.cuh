/**
 * @file atomic.cuh
 * @brief Scoped, memory-ordered device atomics for device-compiled TUs
 *
 * `#include`d into a .cu (CUDA) or `-x hip` device-compiled (HIP) TU; link
 * `wwr.device`. The surface ABOVE the common atomics: every operation carries
 * an explicit memory order and thread scope, where the two backends diverge in
 * spelling and so clear the bar the common atomics do not (docs/architecture.md
 * §15 -- those are spelled identically on both, ride the vendor runtime header
 * runtime.cuh switches, and are wrapped by nothing; test/gpu/atomics.cu pins
 * them). Here CUDA spells the operation `cuda::atomic_ref<T, Scope>` from
 * <cuda/atomic> and HIP spells it a `__hip_atomic_*` clang builtin -- two
 * spellings for one operation, so a forwarder does work rather than renaming a
 * name to itself.
 *
 * Two enums of the project's own, because neither vendor's spelling is
 * portable: `wwrMemoryOrder` and `wwrThreadScope`. The scope is a TEMPLATE
 * parameter and the order a FUNCTION argument -- the split libcu++ forces, its
 * `atomic_ref` taking the scope as a template argument while its `.fetch_add`
 * takes the order as a runtime argument; the HIP builtin takes both as function
 * arguments and accepts a runtime value for each (measured, see §20), so it
 * follows the shape libcu++ dictates without complaint.
 *
 * The mapping from `wwrThreadScope` to each vendor's scope constant is the
 * risk: a wrong row is silent -- it compiles on both and gives up an ordering
 * guarantee at runtime on one (the §16 class of trap). The four portable rows,
 * verified against both memory models, are in docs/architecture.md §20; the two
 * vendor-only scopes (CUDA's cluster, HIP's wavefront) are deliberately absent,
 * left to a TU that names the vendor form directly, exactly as §15 leaves
 * `atomicAdd_block`/`_system` and AMD's `unsafeAtomicAdd` exposed.
 *
 * A compile is the only honest test: ordering *semantics* cannot be proven by
 * one, and a runtime memory-model test is a flake generator. test/gpu/atomic.cu
 * pins that every forwarder x every portable scope resolves under both front
 * ends; the mapping's meaning is a documentation claim (§20), not a runtime
 * assertion. `-munsafe-fp-atomics` is untouched here, as §15 has it: a TU that
 * wants AMD's native FP-atomic codegen passes it on its own device library.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, the device-pass #error, and the vendor
// runtime header -- which on HIP declares the __HIP_MEMORY_SCOPE_* constants
// (amd_hip_atomic.h, reached through hip_runtime.h) the builtins take. Unlike
// cooperative_groups.cuh / wmma.cuh it is not WWR_WARP_SIZE that is wanted here
// but that runtime include, so the same base header serves.
#include "runtime.cuh"

#if defined(WWR_SELECTED_CUDA)

// libcu++. CUDA 13.0 relocated it from include/cuda/atomic to
// include/cccl/cuda/atomic; CMake 4.2's FindCUDAToolkit adds that cccl dir to
// CUDA::cudart's include interface, which wwr.device links, so no extra include
// dir is needed (docs/architecture.md §20 has the measurement).
#include <cuda/atomic>

#endif

namespace wwr {

// ========================================================================
// The portable order and scope -- neither vendor's spelling carries across, so
// these are the project's own (docs/architecture.md §20)
// ========================================================================

/// @brief Memory order for a scoped atomic. Maps to cuda::memory_order (CUDA)
/// or the __ATOMIC_* clang constant (HIP). `consume` is omitted -- both
/// vendors treat it as `acquire`.
enum class wwrMemoryOrder { relaxed, acquire, release, acq_rel, seq_cst };

/// @brief Thread scope an atomic is ordered against. Only the four rows both
/// backends share; CUDA's `cluster` and HIP's `wavefront` are vendor-only and
/// reached by naming the vendor form directly (docs/architecture.md §20).
enum class wwrThreadScope { thread, block, device, system };

#if defined(WWR_SELECTED_CUDA)

/// @brief wwrThreadScope -> cuda::thread_scope, the atomic_ref template arg.
template<wwrThreadScope Scope>
__host__ __device__ constexpr ::cuda::thread_scope wwrToVendorScope() {
  if constexpr (Scope == wwrThreadScope::thread) {
    return ::cuda::thread_scope_thread;
  } else if constexpr (Scope == wwrThreadScope::block) {
    return ::cuda::thread_scope_block;
  } else if constexpr (Scope == wwrThreadScope::device) {
    return ::cuda::thread_scope_device;
  } else {
    return ::cuda::thread_scope_system;
  }
}

/// @brief wwrMemoryOrder -> cuda::memory_order, the runtime order argument.
__host__ __device__ constexpr ::cuda::memory_order wwrToVendorOrder(const wwrMemoryOrder order) {
  switch (order) {
    case wwrMemoryOrder::relaxed: return ::cuda::memory_order_relaxed;
    case wwrMemoryOrder::acquire: return ::cuda::memory_order_acquire;
    case wwrMemoryOrder::release: return ::cuda::memory_order_release;
    case wwrMemoryOrder::acq_rel: return ::cuda::memory_order_acq_rel;
    case wwrMemoryOrder::seq_cst: return ::cuda::memory_order_seq_cst;
  }
  return ::cuda::memory_order_seq_cst;
}

#else

/// @brief wwrThreadScope -> the __HIP_MEMORY_SCOPE_* builtin argument.
template<wwrThreadScope Scope>
__host__ __device__ constexpr int wwrToVendorScope() {
  if constexpr (Scope == wwrThreadScope::thread) {
    return __HIP_MEMORY_SCOPE_SINGLETHREAD;
  } else if constexpr (Scope == wwrThreadScope::block) {
    return __HIP_MEMORY_SCOPE_WORKGROUP;
  } else if constexpr (Scope == wwrThreadScope::device) {
    return __HIP_MEMORY_SCOPE_AGENT;
  } else {
    return __HIP_MEMORY_SCOPE_SYSTEM;
  }
}

/// @brief wwrMemoryOrder -> the __ATOMIC_* builtin argument.
__host__ __device__ constexpr int wwrToVendorOrder(const wwrMemoryOrder order) {
  switch (order) {
    case wwrMemoryOrder::relaxed: return __ATOMIC_RELAXED;
    case wwrMemoryOrder::acquire: return __ATOMIC_ACQUIRE;
    case wwrMemoryOrder::release: return __ATOMIC_RELEASE;
    case wwrMemoryOrder::acq_rel: return __ATOMIC_ACQ_REL;
    case wwrMemoryOrder::seq_cst: return __ATOMIC_SEQ_CST;
  }
  return __ATOMIC_SEQ_CST;
}

#endif

// ========================================================================
// The forwarders -- one signature per operation, a two-line per-backend body.
// Scope defaults to `device` (the common cross-block case) and order to
// `seq_cst` (the safe default, as the standard and libcu++ have it).
// ========================================================================

/// @brief Atomically load `*ptr`.
template<wwrThreadScope Scope = wwrThreadScope::device, typename T>
__device__ __forceinline__ T wwrAtomicLoad(T *ptr, const wwrMemoryOrder order = wwrMemoryOrder::seq_cst) {
#if defined(WWR_SELECTED_CUDA)
  ::cuda::atomic_ref<T, wwrToVendorScope<Scope>()> ref{*ptr};
  return ref.load(wwrToVendorOrder(order));
#else
  return __hip_atomic_load(ptr, wwrToVendorOrder(order), wwrToVendorScope<Scope>());
#endif
}

/// @brief Atomically store `value` into `*ptr`.
template<wwrThreadScope Scope = wwrThreadScope::device, typename T>
__device__ __forceinline__ void wwrAtomicStore(T *ptr, const T value,
                                                const wwrMemoryOrder order = wwrMemoryOrder::seq_cst) {
#if defined(WWR_SELECTED_CUDA)
  ::cuda::atomic_ref<T, wwrToVendorScope<Scope>()> ref{*ptr};
  ref.store(value, wwrToVendorOrder(order));
#else
  __hip_atomic_store(ptr, value, wwrToVendorOrder(order), wwrToVendorScope<Scope>());
#endif
}

/// @brief Atomically replace `*ptr` with `value`, returning the old value.
template<wwrThreadScope Scope = wwrThreadScope::device, typename T>
__device__ __forceinline__ T wwrAtomicExchange(T *ptr, const T value,
                                                const wwrMemoryOrder order = wwrMemoryOrder::seq_cst) {
#if defined(WWR_SELECTED_CUDA)
  ::cuda::atomic_ref<T, wwrToVendorScope<Scope>()> ref{*ptr};
  return ref.exchange(value, wwrToVendorOrder(order));
#else
  return __hip_atomic_exchange(ptr, value, wwrToVendorOrder(order), wwrToVendorScope<Scope>());
#endif
}

/// @brief Strong compare-and-swap: if `*ptr == expected`, store `desired` and
/// return true; else load `*ptr` into `expected` and return false. `success` /
/// `failure` are the orders on the taken / not-taken path.
template<wwrThreadScope Scope = wwrThreadScope::device, typename T>
__device__ __forceinline__ bool
wwrAtomicCompareExchangeStrong(T *ptr, T &expected, const T desired,
                               const wwrMemoryOrder success = wwrMemoryOrder::seq_cst,
                               const wwrMemoryOrder failure = wwrMemoryOrder::seq_cst) {
#if defined(WWR_SELECTED_CUDA)
  ::cuda::atomic_ref<T, wwrToVendorScope<Scope>()> ref{*ptr};
  return ref.compare_exchange_strong(expected, desired, wwrToVendorOrder(success),
                                     wwrToVendorOrder(failure));
#else
  return __hip_atomic_compare_exchange_strong(ptr, &expected, desired, wwrToVendorOrder(success),
                                              wwrToVendorOrder(failure), wwrToVendorScope<Scope>());
#endif
}

/// @brief Weak compare-and-swap: like the strong form but allowed to fail
/// spuriously, so it belongs in a retry loop.
template<wwrThreadScope Scope = wwrThreadScope::device, typename T>
__device__ __forceinline__ bool
wwrAtomicCompareExchangeWeak(T *ptr, T &expected, const T desired,
                             const wwrMemoryOrder success = wwrMemoryOrder::seq_cst,
                             const wwrMemoryOrder failure = wwrMemoryOrder::seq_cst) {
#if defined(WWR_SELECTED_CUDA)
  ::cuda::atomic_ref<T, wwrToVendorScope<Scope>()> ref{*ptr};
  return ref.compare_exchange_weak(expected, desired, wwrToVendorOrder(success),
                                   wwrToVendorOrder(failure));
#else
  return __hip_atomic_compare_exchange_weak(ptr, &expected, desired, wwrToVendorOrder(success),
                                            wwrToVendorOrder(failure), wwrToVendorScope<Scope>());
#endif
}

/// @brief Atomically add `value` to `*ptr`, returning the old value.
template<wwrThreadScope Scope = wwrThreadScope::device, typename T>
__device__ __forceinline__ T wwrAtomicFetchAdd(T *ptr, const T value,
                                               const wwrMemoryOrder order = wwrMemoryOrder::seq_cst) {
#if defined(WWR_SELECTED_CUDA)
  ::cuda::atomic_ref<T, wwrToVendorScope<Scope>()> ref{*ptr};
  return ref.fetch_add(value, wwrToVendorOrder(order));
#else
  return __hip_atomic_fetch_add(ptr, value, wwrToVendorOrder(order), wwrToVendorScope<Scope>());
#endif
}

/// @brief Atomically subtract `value` from `*ptr`, returning the old value.
template<wwrThreadScope Scope = wwrThreadScope::device, typename T>
__device__ __forceinline__ T wwrAtomicFetchSub(T *ptr, const T value,
                                               const wwrMemoryOrder order = wwrMemoryOrder::seq_cst) {
#if defined(WWR_SELECTED_CUDA)
  ::cuda::atomic_ref<T, wwrToVendorScope<Scope>()> ref{*ptr};
  return ref.fetch_sub(value, wwrToVendorOrder(order));
#else
  return __hip_atomic_fetch_sub(ptr, value, wwrToVendorOrder(order), wwrToVendorScope<Scope>());
#endif
}

/// @brief Atomically bitwise-AND `value` into `*ptr`, returning the old value.
template<wwrThreadScope Scope = wwrThreadScope::device, typename T>
__device__ __forceinline__ T wwrAtomicFetchAnd(T *ptr, const T value,
                                               const wwrMemoryOrder order = wwrMemoryOrder::seq_cst) {
#if defined(WWR_SELECTED_CUDA)
  ::cuda::atomic_ref<T, wwrToVendorScope<Scope>()> ref{*ptr};
  return ref.fetch_and(value, wwrToVendorOrder(order));
#else
  return __hip_atomic_fetch_and(ptr, value, wwrToVendorOrder(order), wwrToVendorScope<Scope>());
#endif
}

/// @brief Atomically bitwise-OR `value` into `*ptr`, returning the old value.
template<wwrThreadScope Scope = wwrThreadScope::device, typename T>
__device__ __forceinline__ T wwrAtomicFetchOr(T *ptr, const T value,
                                              const wwrMemoryOrder order = wwrMemoryOrder::seq_cst) {
#if defined(WWR_SELECTED_CUDA)
  ::cuda::atomic_ref<T, wwrToVendorScope<Scope>()> ref{*ptr};
  return ref.fetch_or(value, wwrToVendorOrder(order));
#else
  return __hip_atomic_fetch_or(ptr, value, wwrToVendorOrder(order), wwrToVendorScope<Scope>());
#endif
}

/// @brief Atomically bitwise-XOR `value` into `*ptr`, returning the old value.
template<wwrThreadScope Scope = wwrThreadScope::device, typename T>
__device__ __forceinline__ T wwrAtomicFetchXor(T *ptr, const T value,
                                               const wwrMemoryOrder order = wwrMemoryOrder::seq_cst) {
#if defined(WWR_SELECTED_CUDA)
  ::cuda::atomic_ref<T, wwrToVendorScope<Scope>()> ref{*ptr};
  return ref.fetch_xor(value, wwrToVendorOrder(order));
#else
  return __hip_atomic_fetch_xor(ptr, value, wwrToVendorOrder(order), wwrToVendorScope<Scope>());
#endif
}

/// @brief Atomically store min(`*ptr`, `value`), returning the old value.
template<wwrThreadScope Scope = wwrThreadScope::device, typename T>
__device__ __forceinline__ T wwrAtomicFetchMin(T *ptr, const T value,
                                               const wwrMemoryOrder order = wwrMemoryOrder::seq_cst) {
#if defined(WWR_SELECTED_CUDA)
  ::cuda::atomic_ref<T, wwrToVendorScope<Scope>()> ref{*ptr};
  return ref.fetch_min(value, wwrToVendorOrder(order));
#else
  return __hip_atomic_fetch_min(ptr, value, wwrToVendorOrder(order), wwrToVendorScope<Scope>());
#endif
}

/// @brief Atomically store max(`*ptr`, `value`), returning the old value.
template<wwrThreadScope Scope = wwrThreadScope::device, typename T>
__device__ __forceinline__ T wwrAtomicFetchMax(T *ptr, const T value,
                                               const wwrMemoryOrder order = wwrMemoryOrder::seq_cst) {
#if defined(WWR_SELECTED_CUDA)
  ::cuda::atomic_ref<T, wwrToVendorScope<Scope>()> ref{*ptr};
  return ref.fetch_max(value, wwrToVendorOrder(order));
#else
  return __hip_atomic_fetch_max(ptr, value, wwrToVendorOrder(order), wwrToVendorScope<Scope>());
#endif
}

} // namespace wwr
