/**
 * @file sort.cuh
 * @brief Backend-neutral wwr::thrust re-export of <thrust/sort.h>
 *
 * One leaf header per Thrust header, mirroring Thrust's own layout 1:1 so the
 * spelling a caller already knows carries over: <thrust/sort.h> becomes
 * "thrust/sort.cuh". A plain `using`-re-export into namespace wwr::thrust --
 * the names are identical on CUDA (CCCL) and HIP (rocThrust), so every overload
 * comes across. The leading `::` is load-bearing: inside namespace wwr::thrust
 * a bare `thrust` names THIS namespace.
 *
 * The one divergent spelling, the stream-bound execution policy, is owned by
 * wwr::par_on (execution_policy.cuh) and passed as the leading argument. Device-
 * includable header, not a module (docs/architecture.md §8); link wwr::thrust.
 */

#pragma once

#include "execution_policy.cuh"

#include <thrust/sort.h>

namespace wwr::thrust {

using ::thrust::is_sorted;
using ::thrust::is_sorted_until;
using ::thrust::sort;
using ::thrust::sort_by_key;
using ::thrust::stable_sort;
using ::thrust::stable_sort_by_key;

} // namespace wwr::thrust
