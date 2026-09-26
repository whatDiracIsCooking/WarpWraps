// roctracer.cppm - Compile-time tests for gpumod.hip.roctracer

module;

#include "test/shared/link_check.h"

export module gpumod.test.hip.roctracer;

import std;
import gpumod.hip.roctracer;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.hip.roctracer
//
// The module is pure re-export (using declarations only -- roctracer.h and
// the ext/prof_protocol.h types it pulls in are a plain extern "C" API, no
// convenience-template collisions like hip_runtime_api.h). Runtime tests
// would just test ROCtracer itself, and instrumenting a live HIP program is
// out of scope for a compile-time test; we verify at compile/link time that:
//   1. Enum types satisfy std::is_enum_v
//   2. Every enumerator value matches the compiled header value (generated
//      from the actual compiled values, not hand-copied)
//   3. The activity-record struct is trivially copyable (C-interop guarantee)
//   4. Callback/allocator function-pointer typedefs are actually pointers,
//      and the scalar typedefs (correlation ID, timestamp, opaque pool
//      handle) have the expected shape
//   5. Link-time symbol resolution for every exported function
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::hip::test {

using namespace gpumod::hip;

// ──────────────────────────────────────────────────────────────────────
// Enum type checks
// ──────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<activity_domain_t>);
static_assert(std::is_enum_v<activity_api_phase_t>);
static_assert(std::is_enum_v<roctracer_status_t>);
// roctracer_domain_t is a typedef alias of activity_domain_t, not a distinct
// enum -- confirm the alias, not re-check enum-ness.
static_assert(std::is_same_v<roctracer_domain_t, activity_domain_t>);

// ──────────────────────────────────────────────────────────────────────
// Enum values (generated from the compiled roctracer.h / prof_protocol.h values)
// ──────────────────────────────────────────────────────────────────────

// activity_domain_t
static_assert(static_cast<long long>(ACTIVITY_DOMAIN_HSA_API) == 0);
static_assert(static_cast<long long>(ACTIVITY_DOMAIN_HSA_OPS) == 1);
static_assert(static_cast<long long>(ACTIVITY_DOMAIN_HIP_OPS) == 2);
static_assert(static_cast<long long>(ACTIVITY_DOMAIN_HCC_OPS) == 2);
static_assert(static_cast<long long>(ACTIVITY_DOMAIN_HIP_VDI) == 2);
static_assert(static_cast<long long>(ACTIVITY_DOMAIN_HIP_API) == 3);
static_assert(static_cast<long long>(ACTIVITY_DOMAIN_KFD_API) == 4);
static_assert(static_cast<long long>(ACTIVITY_DOMAIN_EXT_API) == 5);
static_assert(static_cast<long long>(ACTIVITY_DOMAIN_ROCTX) == 6);
static_assert(static_cast<long long>(ACTIVITY_DOMAIN_HSA_EVT) == 7);
static_assert(static_cast<long long>(ACTIVITY_DOMAIN_NUMBER) == 8);

// activity_api_phase_t
static_assert(static_cast<long long>(ACTIVITY_API_PHASE_ENTER) == 0);
static_assert(static_cast<long long>(ACTIVITY_API_PHASE_EXIT) == 1);

// roctracer_status_t
static_assert(static_cast<long long>(ROCTRACER_STATUS_SUCCESS) == 0);
static_assert(static_cast<long long>(ROCTRACER_STATUS_ERROR) == -1);
static_assert(static_cast<long long>(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID) == -2);
static_assert(static_cast<long long>(ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT) == -3);
static_assert(static_cast<long long>(ROCTRACER_STATUS_ERROR_DEFAULT_POOL_UNDEFINED) == -4);
static_assert(static_cast<long long>(ROCTRACER_STATUS_ERROR_DEFAULT_POOL_ALREADY_DEFINED) == -5);
static_assert(static_cast<long long>(ROCTRACER_STATUS_ERROR_MEMORY_ALLOCATION) == -6);
static_assert(static_cast<long long>(ROCTRACER_STATUS_ERROR_MISMATCHED_EXTERNAL_CORRELATION_ID) ==
              -7);
static_assert(static_cast<long long>(ROCTRACER_STATUS_ERROR_NOT_IMPLEMENTED) == -8);
static_assert(static_cast<long long>(ROCTRACER_STATUS_UNINIT) == 2);
static_assert(static_cast<long long>(ROCTRACER_STATUS_BREAK) == 3);
static_assert(static_cast<long long>(ROCTRACER_STATUS_BAD_DOMAIN) == -2);
static_assert(static_cast<long long>(ROCTRACER_STATUS_BAD_PARAMETER) == -3);
static_assert(static_cast<long long>(ROCTRACER_STATUS_HIP_API_ERR) == 6);
static_assert(static_cast<long long>(ROCTRACER_STATUS_HIP_OPS_ERR) == 7);
static_assert(static_cast<long long>(ROCTRACER_STATUS_HCC_OPS_ERR) == 7);
static_assert(static_cast<long long>(ROCTRACER_STATUS_HSA_ERR) == 7);
static_assert(static_cast<long long>(ROCTRACER_STATUS_ROCTX_ERR) == 8);

// ──────────────────────────────────────────────────────────────────────
// Struct traits: trivial copyability (C-interop guarantee)
// ──────────────────────────────────────────────────────────────────────

static_assert(std::is_trivially_copyable_v<activity_record_t>);
static_assert(std::is_trivially_copyable_v<roctracer_record_t>);
static_assert(std::is_trivially_copyable_v<roctracer_properties_t>);

// ──────────────────────────────────────────────────────────────────────
// Scalar / callback typedef shapes
// ──────────────────────────────────────────────────────────────────────

// Callback and allocator types are function pointers
static_assert(std::is_pointer_v<activity_rtapi_callback_t>);
static_assert(std::is_pointer_v<activity_sync_callback_t>);
static_assert(std::is_pointer_v<activity_async_callback_t>);
static_assert(std::is_pointer_v<roctracer_rtapi_callback_t>);
static_assert(std::is_pointer_v<roctracer_allocator_t>);
static_assert(std::is_pointer_v<roctracer_buffer_callback_t>);

// Domain-op and correlation/timestamp scalars are unsigned integral typedefs
static_assert(sizeof(activity_kind_t) == sizeof(std::uint32_t));
static_assert(sizeof(activity_op_t) == sizeof(std::uint32_t));
static_assert(sizeof(activity_correlation_id_t) == sizeof(std::uint64_t));
static_assert(sizeof(roctracer_timestamp_t) == sizeof(std::uint64_t));

// roctracer_pool_t is `typedef void roctracer_pool_t;` -- an opaque handle
// type used only as roctracer_pool_t*, never a complete object type itself.
static_assert(std::is_void_v<roctracer_pool_t>);

// ──────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// ──────────────────────────────────────────────────────────────────────

// Versioning and error reporting
GPUMOD_LINK_CHECK(roctracer_version_major)
GPUMOD_LINK_CHECK(roctracer_version_minor)
GPUMOD_LINK_CHECK(roctracer_error_string)

// Domain queries and properties
GPUMOD_LINK_CHECK(roctracer_op_string)
GPUMOD_LINK_CHECK(roctracer_op_code)
GPUMOD_LINK_CHECK(roctracer_set_properties)

// Callback API
GPUMOD_LINK_CHECK(roctracer_enable_op_callback)
GPUMOD_LINK_CHECK(roctracer_enable_domain_callback)
GPUMOD_LINK_CHECK(roctracer_disable_op_callback)
GPUMOD_LINK_CHECK(roctracer_disable_domain_callback)

// Activity API: records and memory pools
GPUMOD_LINK_CHECK(roctracer_next_record)
GPUMOD_LINK_CHECK(roctracer_open_pool_expl)
GPUMOD_LINK_CHECK(roctracer_open_pool)
GPUMOD_LINK_CHECK(roctracer_close_pool_expl)
GPUMOD_LINK_CHECK(roctracer_close_pool)
GPUMOD_LINK_CHECK(roctracer_default_pool_expl)
GPUMOD_LINK_CHECK(roctracer_default_pool)

// Activity API: enable/disable/flush
GPUMOD_LINK_CHECK(roctracer_enable_op_activity_expl)
GPUMOD_LINK_CHECK(roctracer_enable_op_activity)
GPUMOD_LINK_CHECK(roctracer_enable_domain_activity_expl)
GPUMOD_LINK_CHECK(roctracer_enable_domain_activity)
GPUMOD_LINK_CHECK(roctracer_disable_op_activity)
GPUMOD_LINK_CHECK(roctracer_disable_domain_activity)
GPUMOD_LINK_CHECK(roctracer_flush_activity_expl)
GPUMOD_LINK_CHECK(roctracer_flush_activity)

// Timestamp
GPUMOD_LINK_CHECK(roctracer_get_timestamp)

} // namespace gpumod::hip::test
