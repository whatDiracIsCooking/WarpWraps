/**
 * @file scan.cuh
 * @brief Backend-neutral wwr::thrust re-export of the scan family
 *
 * Inclusive and exclusive prefix scans, their transform_*_scan fused variants,
 * and the by-key forms. Plain `using`-re-exports into namespace wwr::thrust;
 * see reorder.cuh for the pattern, the load-bearing leading `::`, and why the
 * execution policy stays behind wwr::par_on rather than being re-exported. Link
 * wwr::thrust.
 */

#pragma once

#include "execution_policy.cuh"

#include <thrust/scan.h>
#include <thrust/transform_scan.h>

namespace wwr::thrust {

using ::thrust::exclusive_scan;
using ::thrust::inclusive_scan;

using ::thrust::transform_exclusive_scan;
using ::thrust::transform_inclusive_scan;

using ::thrust::exclusive_scan_by_key;
using ::thrust::inclusive_scan_by_key;

} // namespace wwr::thrust
