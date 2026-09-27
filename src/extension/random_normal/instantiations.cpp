/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of random_normal per supported type
 *
 * Implementation unit of gpumod.extension.random_normal. Pairs with the
 * `extern template` declarations in interface.cppm: together they keep every
 * importer from instantiating the template again at each call site.
 *
 * These instantiate the exported WRAPPER. The device-side work it calls
 * (device::random_normal) is instantiated separately, in random_normal.cu,
 * because that one has to be compiled as device code.
 * Both lists cover the same six types and have to stay in step -- a type
 * added here without being added there links against nothing.
 */

module gpumod.extension.random_normal;

// An implementation unit implicitly imports its primary interface, but an
// import is not re-exported through it -- std::size_t in the signatures below
// is not visible without this.
import std;
import gpumod.runtime_api;
import gpumod.rand;
import gpumod.complex;
import gpumod.fp16;
import gpumod.bf16;

namespace wwr::extension {

template void random_normal<float>(gpuStream_t, std::size_t, gpurandState *, float *, float);
template void random_normal<double>(gpuStream_t, std::size_t, gpurandState *, double *, double);
template void random_normal<gpuFloatComplex>(gpuStream_t, std::size_t, gpurandState *,
                                             gpuFloatComplex *, gpuFloatComplex);
template void random_normal<gpuDoubleComplex>(gpuStream_t, std::size_t, gpurandState *,
                                              gpuDoubleComplex *, gpuDoubleComplex);
template void random_normal<gpuHalf>(gpuStream_t, std::size_t, gpurandState *, gpuHalf *, gpuHalf);
template void random_normal<gpuBfloat16>(gpuStream_t, std::size_t, gpurandState *, gpuBfloat16 *,
                                         gpuBfloat16);

} // namespace wwr::extension
