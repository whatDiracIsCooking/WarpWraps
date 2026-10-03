# GPU FFT wrapper tests (backend-neutral: built for either WWR_GPU_BACKEND)
#
# Build-time only. The typed exec wrappers pick their entry point by
# token-pasting a precision-specific transform-kind suffix (C2C/Z2Z, ...), which
# no static_assert can see, so fft_dispatch.toml checks it in the compiled
# objects. The wrappers are token-paste forwarders with no body of their own, so
# the pasted name is the whole of what they can get wrong -- there is no runtime
# half, and no type_traits static-assert companion the way solver has (FFT
# dispatch carries no runtime data-type value to check).

# GPU_SOURCE is the file carrying the WWR_FUNCTION_RAW(wwrfft...) alias lines
# the check maps vendor symbols back through -- the shared fragment, not
# fft.cppm, since the surface moved there (fft.cppm now just #includes it, and
# dispatch.py scans one file's text, not its includes).
wwr_add_dispatch_check(
  NAME
  wwr.test.wrappers.fft_dispatch
  TARGET
  wwr.wrappers.fft
  TABLE
  fft_dispatch.toml
  GPU_SOURCE
  src/detail/fft_names.h
)
