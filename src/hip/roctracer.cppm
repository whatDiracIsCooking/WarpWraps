/**
 * @file roctracer.cppm
 * @brief ROCtracer (ROC Profiling Tools Interface) module wrapper for gpumod project
 *
 * Wraps roctracer/roctracer.h -- ROCm's runtime callback and
 * asynchronous-activity tracing API, HIP's rough counterpart to CUPTI (see
 * wwr.cuda.cupti). roctracer.h includes ext/prof_protocol.h, so both
 * headers' declarations are exported together. Like the other src/hip
 * management headers it is a pure C API, re-exported by name.
 *
 * Deliberately out of scope:
 *   - roctracer_hip.h, which defines one enum (hip_op_id_t) but pulls in the
 *     entire HIP runtime plus the generated hip_prof_str.h to do it. That
 *     surface is already wwr.hip.hip_runtime_api's.
 *   - roctracer_roctx.h, which types the ACTIVITY_DOMAIN_ROCTX callback
 *     payload but requires roctx.h -- a distinct marker API with its own
 *     library, wrapped separately in wwr.hip.roctx (its NVTX counterpart is
 *     wwr.cuda.nvToolsExt). The domain enumerator itself, from
 *     prof_protocol.h, IS exported here; only the ROCTX-specific callback-data
 *     struct is absent.
 *
 * Usage:
 *   import wwr.hip.roctracer;
 */

module;

#include <roctracer/roctracer.h>

export module wwr.hip.roctracer;

export namespace wwr::hip {

// ========================================================================
// Types and Enumerations (roctracer/ext/prof_protocol.h)
// ========================================================================

// activity_domain_t
using ::ACTIVITY_DOMAIN_EXT_API;
using ::ACTIVITY_DOMAIN_HCC_OPS;
using ::ACTIVITY_DOMAIN_HIP_API;
using ::ACTIVITY_DOMAIN_HIP_OPS;
using ::ACTIVITY_DOMAIN_HIP_VDI;
using ::ACTIVITY_DOMAIN_HSA_API;
using ::ACTIVITY_DOMAIN_HSA_EVT;
using ::ACTIVITY_DOMAIN_HSA_OPS;
using ::ACTIVITY_DOMAIN_KFD_API;
using ::ACTIVITY_DOMAIN_NUMBER;
using ::ACTIVITY_DOMAIN_ROCTX;
using ::activity_domain_t;

// API callback type and the domain-op scalar typedefs it (and activity
// records) are keyed on
using ::activity_kind_t;
using ::activity_op_t;
using ::activity_rtapi_callback_t;

// activity_api_phase_t
using ::ACTIVITY_API_PHASE_ENTER;
using ::ACTIVITY_API_PHASE_EXIT;
using ::activity_api_phase_t;

// Correlation ID and timestamp scalar typedefs
using ::activity_correlation_id_t;
using ::roctracer_timestamp_t;

// Activity record type, and the sync/async callback types it is delivered through
using ::activity_async_callback_t;
using ::activity_record_t;
using ::activity_sync_callback_t;

// ========================================================================
// Types and Enumerations (roctracer/roctracer.h)
// ========================================================================

// roctracer_status_t
using ::ROCTRACER_STATUS_ERROR;
using ::ROCTRACER_STATUS_ERROR_DEFAULT_POOL_ALREADY_DEFINED;
using ::ROCTRACER_STATUS_ERROR_DEFAULT_POOL_UNDEFINED;
using ::ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT;
using ::ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID;
using ::ROCTRACER_STATUS_ERROR_MEMORY_ALLOCATION;
using ::ROCTRACER_STATUS_ERROR_MISMATCHED_EXTERNAL_CORRELATION_ID;
using ::ROCTRACER_STATUS_ERROR_NOT_IMPLEMENTED;
using ::ROCTRACER_STATUS_SUCCESS;
using ::roctracer_status_t;
// Deprecated error codes, kept for parity with the header (some are
// equal-valued aliases of the codes above -- still distinct enumerator names
// the header declares, so still exported individually)
using ::ROCTRACER_STATUS_BAD_DOMAIN;
using ::ROCTRACER_STATUS_BAD_PARAMETER;
using ::ROCTRACER_STATUS_BREAK;
using ::ROCTRACER_STATUS_HCC_OPS_ERR;
using ::ROCTRACER_STATUS_HIP_API_ERR;
using ::ROCTRACER_STATUS_HIP_OPS_ERR;
using ::ROCTRACER_STATUS_HSA_ERR;
using ::ROCTRACER_STATUS_ROCTX_ERR;
using ::ROCTRACER_STATUS_UNINIT;

// roctracer_domain_t is `typedef activity_domain_t roctracer_domain_t;`
using ::roctracer_domain_t;

// roctracer_rtapi_callback_t is `typedef activity_rtapi_callback_t roctracer_rtapi_callback_t;`
using ::roctracer_rtapi_callback_t;

// roctracer_record_t is `typedef activity_record_t roctracer_record_t;`
using ::roctracer_record_t;

// Memory pool allocator/buffer callback types, properties struct, and the
// (opaque, `typedef void roctracer_pool_t;`) pool handle type
using ::roctracer_allocator_t;
using ::roctracer_buffer_callback_t;
using ::roctracer_pool_t;
using ::roctracer_properties_t;

// ========================================================================
// Functions
// ========================================================================

// Versioning and error reporting
using ::roctracer_error_string;
using ::roctracer_version_major;
using ::roctracer_version_minor;

// Domain queries and properties
using ::roctracer_op_code;
using ::roctracer_op_string;
using ::roctracer_set_properties;

// Callback API
using ::roctracer_disable_domain_callback;
using ::roctracer_disable_op_callback;
using ::roctracer_enable_domain_callback;
using ::roctracer_enable_op_callback;

// Activity API: records and memory pools
using ::roctracer_close_pool;
using ::roctracer_close_pool_expl;
using ::roctracer_default_pool;
using ::roctracer_default_pool_expl;
using ::roctracer_next_record;
using ::roctracer_open_pool;
using ::roctracer_open_pool_expl;

// Activity API: enable/disable/flush
using ::roctracer_disable_domain_activity;
using ::roctracer_disable_op_activity;
using ::roctracer_enable_domain_activity;
using ::roctracer_enable_domain_activity_expl;
using ::roctracer_enable_op_activity;
using ::roctracer_enable_op_activity_expl;
using ::roctracer_flush_activity;
using ::roctracer_flush_activity_expl;

// Timestamp
using ::roctracer_get_timestamp;

} // namespace wwr::hip
