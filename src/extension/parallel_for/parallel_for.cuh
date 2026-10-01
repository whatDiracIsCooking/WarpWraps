/// @file
/// @brief Backend-neutral, header-only index-per-thread parallel_for kernel launcher
///
/// Executes a user-supplied functor on each index in `[0, count)` on the GPU,
/// via a hand-written `__global__` kernel -- index-per-thread, no grid-stride
/// loop. See project memory `project-thrust-algorithms-break-under-rdc` for
/// why this does not use Thrust.
///
/// `#include`d directly into a .cu (CUDA) or `-x hip` device-compiled (HIP)
/// translation unit, so it reaches the backend through the wwr* layer's
/// runtime.h (and device_guard.h for the device-pass gate) rather than
/// backend.h: there is no module involved at the point of use. Link
/// `wwr.device`.
///
/// A functor's `operator()` is plain `__device__` on both backends. Its
/// callability is constrained on the kernel template below -- a device entity,
/// where a `{ f(i) }` probe accepts a `__device__`-only call -- rather than on
/// the host-checked `device_functor` concept, where the same probe would run in
/// host context and reject it.
///
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// WWR_GRID_CONSTANT, WWR_WARP_SIZE and wwrStream_t for the signature below, all
// the wwr* layer's, reached bare through wwr.device's include path. runtime.h's
// device section carries the two macros and the full runtime; wwrStream_t is in
// its always-on part. device_guard.h is the device-pass gate -- it #errors
// outside a CUDA or HIP device compile, the guard runtime.h (host-safe) does not
// carry, so this device-only .cuh includes it directly.
#include "device_guard.h"
#include "runtime.h"

namespace wwr::extension {

/// @brief Constraint for an immutable, trivially-copyable callable invoked per thread index
///
/// The functor must not be mutable: under CUDA the kernel receives it via
/// `WWR_GRID_CONSTANT` (constant memory shared across the grid), and writes to
/// constant memory are undefined behaviour on either backend. The
/// `!is_copy_assignable` check is the compile-time proxy for that -- a const
/// member deletes the implicit copy-assignment operator.
///
/// That const member must be of SCALAR type. A const member of CLASS type
/// makes the enclosing class non-trivially-copyable under clang, failing this
/// concept's other requirement instead. See docs/architecture.md, section 13.
///
/// @tparam F Functor type -- a trivially-copyable class with at least one
///           const scalar member (never a const class-type member) and a
///           const-qualified `__device__ operator()(IndexType) -> void`
/// @tparam IndexType Integral index type passed to the functor
///
/// @note This concept deliberately does NOT probe `{ f(i) }`. That check lives
///       on the kernel template below, because it must be evaluated in device
///       context; see this file's header.
template<typename F, typename IndexType = std::size_t>
concept device_functor = std::integral<IndexType> && std::is_trivially_copyable_v<F> &&
                         std::is_class_v<F> && !std::is_copy_assignable_v<F>;

namespace device {

/// @brief [kernel] Index-per-thread mapping executing one functor call per element
///
/// The `requires` carries the callability check `device_functor` cannot: a
/// `__global__` template is a device entity, so `{ f(i) }` here accepts the
/// functor's `__device__`-only `operator()`.
template<typename Functor, typename IndexType>
  requires requires(const Functor f, const IndexType i) {
    { f(i) } -> std::same_as<void>;
  }
__global__ void parallel_for_kernel(WWR_GRID_CONSTANT const Functor f, IndexType count) {
  const IndexType i = static_cast<IndexType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    f(i);
  }
}

} // namespace device

/// @brief Launch a 1-D grid invoking functor(i) for each index in [0, count)
///
/// @tparam IndexType Integral type for the iteration index
/// @tparam Functor Trivially-copyable callable satisfying device_functor
/// @param stream GPU stream for async kernel launch
/// @param count Number of elements to process
/// @param functor Per-thread functor invoked with thread index
/// @return The synchronous launch status from wwrGetLastError -- wwrSuccess if
///         the grid was configured and queued (including the empty count < 1
///         case, which launches nothing). This reports LAUNCH errors only (bad
///         configuration, resource limits); a kernel's own execution errors are
///         asynchronous and surface at the next synchronization, not here.
template<std::integral IndexType = std::size_t, device_functor<IndexType> Functor>
wwrError_t parallel_for(wwrStream_t stream, const IndexType count, Functor functor) {
  if (count < 1) {
    return wwrSuccess;
  }

  // 4 warps per block. WWR_WARP_SIZE (runtime.h) is a configure-time
  // value -- -DWWR_WARP_SIZE, default 32, so this is 128 unless a
  // CDNA build sets 64 and makes it 256. Neither backend offers a warp size
  // that can be used in a constant expression, and HIP's one compile-time
  // spelling differs between its host and device passes; that header's
  // WWR_WARP_SIZE entry has the transcript.
  constexpr uint32_t kBlockSize = 4 * WWR_WARP_SIZE;
  static_assert(kBlockSize <= 1024,
                "block size exceeds the 1024 threads/block both backends cap at -- "
                "WWR_WARP_SIZE is too large");

  const uint32_t num_blocks = static_cast<uint32_t>((count + kBlockSize - 1) / kBlockSize);

  device::parallel_for_kernel<<<num_blocks, kBlockSize, 0, stream>>>(functor, count);

  return wwrGetLastError();
}

} // namespace wwr::extension
