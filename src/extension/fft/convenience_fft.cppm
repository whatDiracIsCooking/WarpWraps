/**
 * @file convenience_fft.cppm
 * @brief Default-policy alias for the GPU FFT plan wrapper
 *
 * `FftPlan` binds FftPlanWrapper to the default error policy. This partition is
 * the curated home for that default binding; alternative-policy aliases belong
 * here too.
 */

export module wwr.extension.fft:convenience_fft;

import :fft_plan;
import wwr.extension.common;

export namespace wwr::extension {

/**
 * @brief Convenient alias for FftPlanWrapper with default error policies
 *
 * Usage:
 *   FftPlan plan;  // Instead of FftPlanWrapper<>
 */
using FftPlan = FftPlanWrapper<>;

} // namespace wwr::extension
