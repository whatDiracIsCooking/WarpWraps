/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of random_normal per supported type
 *
 * Implementation unit of wwr.extension.random_normal. Pairs with the
 * `extern template` declarations in interface.cppm: together they keep every
 * importer from instantiating the template again at each call site.
 *
 * These instantiate the exported WRAPPER. The device-side work it calls
 * (device::random_normal) is instantiated separately, in random_normal.cu,
 * because that one has to be compiled as device code.
 * Both lists cover the same six types and have to stay in step -- a type
 * added here without being added there links against nothing.
 */

module wwr.extension.random_normal;

// An implementation unit implicitly imports its primary interface, but an
// import is not re-exported through it -- std::size_t in the signatures below
// is not visible without this.
import std;
import wwr.runtime_api;
import wwr.rand;
import wwr.complex;
import wwr.fp16;
import wwr.bf16;

namespace wwr::extension {

template void random_normal<float>(wwrStream_t, std::size_t, wwrrandState *, float *, float);
template void random_normal<double>(wwrStream_t, std::size_t, wwrrandState *, double *, double);
template void random_normal<wwrFloatComplex>(wwrStream_t, std::size_t, wwrrandState *,
                                             wwrFloatComplex *, wwrFloatComplex);
template void random_normal<wwrDoubleComplex>(wwrStream_t, std::size_t, wwrrandState *,
                                              wwrDoubleComplex *, wwrDoubleComplex);
template void random_normal<wwrHalf>(wwrStream_t, std::size_t, wwrrandState *, wwrHalf *, wwrHalf);
template void random_normal<wwrBfloat16>(wwrStream_t, std::size_t, wwrrandState *, wwrBfloat16 *,
                                         wwrBfloat16);

} // namespace wwr::extension
