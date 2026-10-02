/**
 * @file reduce.cuh
 * @brief Backend-neutral wwr::thrust re-export of the reduce family
 *
 * reduce, transform_reduce, count, inner_product and the element extrema --
 * the algorithms that fold a range to a value (or a position). Plain `using`-
 * re-exports into namespace wwr::thrust; see reorder.cuh for the pattern, the
 * load-bearing leading `::`, and why the execution policy stays behind
 * wwr::par_on rather than being re-exported. Link wwr::thrust.
 *
 * Extrema are the element-position algorithms (min_element / max_element /
 * minmax_element), not scalar thrust::min / thrust::max -- the latter were
 * removed from CCCL 3.0's public extrema.h and are deliberately not re-exported.
 */

#pragma once

#include "execution_policy.cuh"

#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

namespace wwr::thrust {

using ::thrust::reduce;
using ::thrust::reduce_by_key;

using ::thrust::transform_reduce;

using ::thrust::count;
using ::thrust::count_if;

using ::thrust::inner_product;

using ::thrust::max_element;
using ::thrust::min_element;
using ::thrust::minmax_element;

} // namespace wwr::thrust
