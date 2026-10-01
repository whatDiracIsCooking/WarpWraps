/**
 * @file instantiations.cpp
 * @brief Explicit template instantiations for the GPU RNG wrappers
 *
 * This file contains explicit template instantiations to avoid code bloat
 * from implicit instantiation at every call site. Hand-written -- the
 * `extern template` declarations live in exec.cppm, next to each function.
 */

module wwr.wrappers.rand;

import wwr.rand;
import std;

namespace wwr {

// Function: generate_uniform
template wwrrandStatus_t generate_uniform<float>(wwrrandGenerator_t, float *, std::size_t);
template wwrrandStatus_t generate_uniform<double>(wwrrandGenerator_t, double *, std::size_t);

// Function: generate_normal
template wwrrandStatus_t generate_normal<float>(wwrrandGenerator_t, float *, std::size_t, float,
                                                float);
template wwrrandStatus_t generate_normal<double>(wwrrandGenerator_t, double *, std::size_t, double,
                                                 double);

// Function: generate_lognormal
template wwrrandStatus_t generate_lognormal<float>(wwrrandGenerator_t, float *, std::size_t, float,
                                                   float);
template wwrrandStatus_t generate_lognormal<double>(wwrrandGenerator_t, double *, std::size_t,
                                                    double, double);

} // namespace wwr
