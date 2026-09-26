# GPU FFT wrapper tests (backend-neutral: built for either GPUMOD_GPU_BACKEND)
#
# Build-time only. The typed exec wrappers pick their entry point by
# token-pasting a precision-specific transform-kind suffix (C2C/Z2Z, ...), which
# no static_assert can see, so fft_dispatch.toml checks it in the compiled
# objects. The wrappers are token-paste forwarders with no body of their own, so
# the pasted name is the whole of what they can get wrong -- there is no runtime
# half, and no type_traits static-assert companion the way solver has (FFT
# dispatch carries no runtime data-type value to check).

gpumod_add_dispatch_check(
  NAME
  gpumod.test.wrappers.fft_dispatch
  TARGET
  gpumod.wrappers.fft
  TABLE
  fft_dispatch.toml
  GPU_SOURCE
  src/fft.cppm
)
