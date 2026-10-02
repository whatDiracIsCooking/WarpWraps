/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of each elementwise wrapper per type
 *
 * Implementation unit of wwr.wrappers.thrust. Pairs with the `extern template`
 * declarations in elementwise.cppm: together they keep every importer from
 * re-instantiating the (heavy) Thrust-backed wrapper at each call site.
 *
 * These instantiate the exported HOST wrappers. The device-side work each calls
 * (device::fill, device::transform_unary, ...) is instantiated separately, in
 * elementwise.cu, because that one has to be compiled as device code. The three
 * lists -- here, elementwise.cppm's extern templates, and elementwise.cu's
 * device instantiations -- cover the same sets and must stay in step.
 */

module wwr.wrappers.thrust;

// An implementation unit implicitly imports its primary interface, but an
// import is not re-exported through it -- std::size_t and wwrFloatComplex in the
// signatures below are not visible without these.
import std;
import wwr.runtime_api;
import wwr.complex;

namespace wwr::extension::thrust {

#define WWR_INST_FILL(T) template void fill<T>(wwrStream_t, T *, std::size_t, T)
WWR_INST_FILL(float);
WWR_INST_FILL(double);
WWR_INST_FILL(int);
WWR_INST_FILL(unsigned int);
WWR_INST_FILL(long long);
WWR_INST_FILL(unsigned long long);
WWR_INST_FILL(wwrFloatComplex);
WWR_INST_FILL(wwrDoubleComplex);
#undef WWR_INST_FILL

#define WWR_INST_SEQ(T) template void sequence<T>(wwrStream_t, T *, std::size_t, T, T)
WWR_INST_SEQ(float);
WWR_INST_SEQ(double);
WWR_INST_SEQ(int);
WWR_INST_SEQ(unsigned int);
WWR_INST_SEQ(long long);
WWR_INST_SEQ(unsigned long long);
#undef WWR_INST_SEQ

#define WWR_INST_REP(T) template void replace<T>(wwrStream_t, T *, std::size_t, T, T)
WWR_INST_REP(float);
WWR_INST_REP(double);
WWR_INST_REP(int);
WWR_INST_REP(unsigned int);
WWR_INST_REP(long long);
WWR_INST_REP(unsigned long long);
WWR_INST_REP(wwrFloatComplex);
WWR_INST_REP(wwrDoubleComplex);
#undef WWR_INST_REP

#define WWR_INST_UN(Op, T) template void transform_unary<Op, T>(wwrStream_t, const T *, T *, std::size_t)
WWR_INST_UN(UnaryOp::Negate, float);
WWR_INST_UN(UnaryOp::Negate, double);
WWR_INST_UN(UnaryOp::Negate, int);
WWR_INST_UN(UnaryOp::Negate, long long);
WWR_INST_UN(UnaryOp::Negate, wwrFloatComplex);
WWR_INST_UN(UnaryOp::Negate, wwrDoubleComplex);
WWR_INST_UN(UnaryOp::Square, float);
WWR_INST_UN(UnaryOp::Square, double);
WWR_INST_UN(UnaryOp::Square, int);
WWR_INST_UN(UnaryOp::Square, long long);
WWR_INST_UN(UnaryOp::Square, wwrFloatComplex);
WWR_INST_UN(UnaryOp::Square, wwrDoubleComplex);
WWR_INST_UN(UnaryOp::Abs, float);
WWR_INST_UN(UnaryOp::Abs, double);
WWR_INST_UN(UnaryOp::Abs, int);
WWR_INST_UN(UnaryOp::Abs, long long);
#undef WWR_INST_UN

#define WWR_INST_BIN(Op, T) \
  template void transform_binary<Op, T>(wwrStream_t, const T *, const T *, T *, std::size_t)
#define WWR_INST_BIN_ALL(Op) \
  WWR_INST_BIN(Op, float);   \
  WWR_INST_BIN(Op, double);  \
  WWR_INST_BIN(Op, int);     \
  WWR_INST_BIN(Op, long long); \
  WWR_INST_BIN(Op, wwrFloatComplex); \
  WWR_INST_BIN(Op, wwrDoubleComplex)
WWR_INST_BIN_ALL(BinaryOp::Plus);
WWR_INST_BIN_ALL(BinaryOp::Minus);
WWR_INST_BIN_ALL(BinaryOp::Multiply);
#undef WWR_INST_BIN_ALL
#undef WWR_INST_BIN

} // namespace wwr::extension::thrust
