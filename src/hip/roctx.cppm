/**
 * @file roctx.cppm
 * @brief rocTX (ROCm Tracing Extension) marker/range API module wrapper
 *
 * Wraps roctracer/roctx.h -- rocTX, the HIP counterpart to NVTX (see
 * wwr.cuda.nvToolsExt). This is nearly the whole of the classic roctx.h,
 * scoped to the marker-and-range surface both backends share, so a neutral
 * gpu* layer can sit on exactly this set.
 *
 * Unlike NVTX's header-only static-inline functions, rocTX's are real
 * extern "C" symbols in libroctx64, so they re-export by name with `using ::`.
 * roctracer has no CMake package (no /opt/rocm/lib/cmake/roctracer/), so
 * src/hip/CMakeLists.txt find_library's roctx64 and find_path's the include dir
 * directly, exactly as it already does for roctracer64. roctx.h needs neither
 * the HIP runtime nor its headers (only <stdint.h>), so this links no hip::
 * target.
 *
 * Note the name shapes differ from NVTX's: the async-range terminator here is
 * roctxRangeStop (NVTX spells it nvtxRangeEnd), and the range-id type is
 * roctx_range_id_t (NVTX: nvtxRangeId_t). Both are uint64_t.
 *
 * The `roctxMark` / `roctxRangePush` / `roctxRangeStart` names in the header are
 * function-like macro aliases for the A-suffixed functions; macros do not cross
 * a module boundary, so only the real A-suffixed functions are exported.
 *
 * Deliberately out of scope:
 *   - roctx_version_major / roctx_version_minor -- versioning queries with no
 *     counterpart in NVTX's common marker/range surface.
 *   - the richer surface that exists only in the rocprofiler-sdk-roctx variant
 *     of the header (a distinct library, librocprofiler-sdk-roctx):
 *     roctxProfilerPause/Resume, roctxNameOsThread, roctxNameHipDevice/HipStream,
 *     roctxNameHsaAgent, roctxGetThreadId.
 *
 * Usage:
 *   import wwr.hip.roctx;
 */

module;

#include <roctracer/roctx.h>

export module wwr.hip.roctx;

export namespace wwr::hip {

// ========================================================================
// Types
// ========================================================================

// Opaque handle identifying a process-wide asynchronous range (uint64_t).
using ::roctx_range_id_t;

// ========================================================================
// Markers -- an instantaneous event at a point in time
// ========================================================================

using ::roctxMarkA;

// ========================================================================
// Ranges -- nested (stack) push/pop on the calling thread
// ========================================================================

using ::roctxRangePop;
using ::roctxRangePushA;

// ========================================================================
// Ranges -- process-wide asynchronous start/stop
// ========================================================================

using ::roctxRangeStartA;
using ::roctxRangeStop;

} // namespace wwr::hip
