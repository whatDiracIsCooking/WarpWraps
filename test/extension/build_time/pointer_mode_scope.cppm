// pointer_mode_scope.cppm - Compile-time contract of PointerModeScope and the
// blas_handle concept it co-owns its handle through.
//
// The guard needs no live GPU to pin down: that its handle axis is the blas_handle
// concept (not the full BlasHandleWrapper policy list), that the concept actually
// discriminates a BLAS-handle-yielding type from any other handle, and that the
// guard is a non-movable RAII value taking a shared_ptr. The runtime behaviour --
// that it truly gets/sets/restores the mode on a real handle -- is a GPU test and
// lives beside the other blas handle tests.

export module wwr.test.extension.pointer_mode_scope;

import std;
import wwr.runtime_api;
import wwr.blas;
import wwr.extension.common;
import wwr.extension.runtime; // StreamWrapper -- a handle that yields the WRONG type
import wwr.extension.blas;
import wwr.test.shared.abort_policy;
import wwr.test.shared.device_handle; // DeviceHandle, the stream owner a BlasHandleWrapper binds to

namespace wwr::extension::test {

using Abort = AbortPolicy<wwrError_t>;
using BlasAbort = AbortPolicy<wwrblasStatus_t>;
using BlasHandle = BlasHandleWrapper<BlasAbort, BlasAbort, DeviceHandle, Abort>;

// The concept agrees with blas_handle.cppm: a BlasHandleWrapper yields a
// wwrblasHandle_t through get(), so it models blas_handle.
static_assert(blas_handle<BlasHandle>);

// ...and it discriminates: a StreamWrapper is a real handle wrapper, but its get()
// yields a wwrStream_t (ihipStream_t* on HIP, a distinct type from wwrblasHandle_t
// even where the latter is void*), so it is rejected -- as is anything with no
// get() at all. The concept's same_as (not convertible_to) is exactly what makes
// this hold on HIP; see blas_handle's own note.
static_assert(!blas_handle<StreamWrapper<Abort, Abort, Abort>>);
static_assert(!blas_handle<int>);

using Scope = PointerModeScope<BlasHandle, BlasAbort>;

// A move-only resource guard is in fact non-movable: NonCopyable deletes the copy,
// and the user-declared destructor suppresses the implicit move -- so the retained
// handle_ is never null after construction, which the restore in ~PointerModeScope
// relies on.
static_assert(!std::is_copy_constructible_v<Scope>);
static_assert(!std::is_move_constructible_v<Scope>);

// Constructed from a shared-owned handle plus the target mode, with the error
// policy defaulted or named. The shared_ptr is the whole point: the handle is
// co-owned, so it cannot be freed out from under the guard.
static_assert(std::is_constructible_v<Scope, std::shared_ptr<BlasHandle>, wwrblasPointerMode_t>);
static_assert(
    std::is_constructible_v<Scope, std::shared_ptr<BlasHandle>, wwrblasPointerMode_t, BlasAbort>);
// A bare wwrblasHandle_t is NOT accepted -- co-ownership is required, not a borrow.
static_assert(!std::is_constructible_v<Scope, wwrblasHandle_t, wwrblasPointerMode_t>);

} // namespace wwr::extension::test
