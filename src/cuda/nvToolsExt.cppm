/**
 * @file nvToolsExt.cppm
 * @brief NVTX (NVIDIA Tools Extension) marker/range API module wrapper
 *
 * Wraps the marker-and-range core of nvtx3/nvToolsExt.h -- the profiler
 * annotation surface that has a one-to-one HIP counterpart in wwr.hip.roctx
 * (rocTX). Scoped deliberately to what both backends share, so a neutral gpu*
 * layer can sit on exactly this set.
 *
 * NVTX3's functions are header-only static inline -- NVTX_DECLSPEC resolves to
 * NVTX_INLINE_STATIC in normal usage, with definitions pulled in from
 * nvtxDetail/nvtxImpl.h -- so, like cuComplex.h, they have internal linkage and
 * cannot be re-exported with `using ::`. Each is re-exported through a thin
 * forwarding function instead. There is no libnvToolsExt to link and no
 * CUDA::nvToolsExt target; only the toolkit include path is needed
 * (INCLUDE_CUDA_TOOLKIT in src/cuda/CMakeLists.txt).
 *
 * Deliberately out of scope -- NVTX surface with no rocTX analogue, so nothing
 * for a neutral wrapper to map onto:
 *   - attributed events (nvtxEventAttributes_t) and the *Ex entry points
 *     (nvtxMarkEx, nvtxRangePushEx, nvtxRangeStartEx) -- color, category and
 *     payload; rocTX ranges are plain strings only.
 *   - domains (nvtxDomain*), registered strings (nvtxDomainRegisterString*),
 *     categories (nvtxNameCategory*), and resource naming (nvtxName*),
 *     including the CUDA-object variants in nvToolsExtCuda.h / CudaRt.h.
 *   - the wide-char (*W) overloads -- rocTX is ASCII-only.
 *   - nvtxInitialize -- rocTX has no explicit initialization call.
 *
 * Note the name shapes differ from rocTX's: the async-range terminator here is
 * nvtxRangeEnd (rocTX spells it roctxRangeStop), and the range-id type is
 * nvtxRangeId_t (rocTX: roctx_range_id_t). Both are uint64_t.
 *
 * Usage:
 *   import wwr.cuda.nvToolsExt;
 */

module;

#include <nvtx3/nvToolsExt.h>

export module wwr.cuda.nvToolsExt;

export namespace wwr::cuda {

// ========================================================================
// Types
// ========================================================================

// Opaque handle identifying a process-wide asynchronous range (uint64_t). A
// typedef has no linkage of its own, so -- unlike the functions below -- it
// re-exports directly.
using ::nvtxRangeId_t;

// ========================================================================
// Markers -- an instantaneous event at a point in time
// ========================================================================

// Record a marker carrying an ASCII message.
void nvtxMarkA(const char* message) {
  ::nvtxMarkA(message);
}

// ========================================================================
// Ranges -- nested (stack) push/pop on the calling thread
// ========================================================================

// Begin a nested range; returns the zero-based depth of the range begun, or a
// negative value on error.
int nvtxRangePushA(const char* message) {
  return ::nvtxRangePushA(message);
}

// End the innermost nested range on the calling thread; returns the depth of
// the range ended, or a negative value on error.
int nvtxRangePop() {
  return ::nvtxRangePop();
}

// ========================================================================
// Ranges -- process-wide asynchronous start/end
// ========================================================================

// Begin an asynchronous range; returns a handle to pass to nvtxRangeEnd.
nvtxRangeId_t nvtxRangeStartA(const char* message) {
  return ::nvtxRangeStartA(message);
}

// End the asynchronous range identified by id.
void nvtxRangeEnd(nvtxRangeId_t id) {
  ::nvtxRangeEnd(id);
}

} // namespace wwr::cuda
