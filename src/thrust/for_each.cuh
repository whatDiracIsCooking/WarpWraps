/**
 * @file for_each.cuh
 * @brief Backend-neutral wwr::thrust re-export of <thrust/for_each.h>
 *
 * One leaf header per Thrust header, mirroring Thrust's own layout 1:1 so the
 * spelling a caller already knows carries over: <thrust/for_each.h> becomes
 * "thrust/for_each.cuh". A plain `using`-re-export into namespace wwr::thrust --
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

#include <thrust/for_each.h>

namespace wwr::thrust {

using ::thrust::for_each;
using ::thrust::for_each_n;

} // namespace wwr::thrust
