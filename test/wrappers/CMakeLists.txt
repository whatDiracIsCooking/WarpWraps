# Backend-neutral extension tests -- src/wrappers's tests, written against gpu*
# names and built for either WWR_GPU_BACKEND.
#
# Each module's directory holds both kinds of test for that module: the
# build-time ones (static_asserts, and the dispatch checks on the compiled
# objects) and the GoogleTest runtime ones. The build-time half must be
# configured under WWR_COMPILE_TIME_ONLY too, so these directories are added
# unconditionally and each one returns early, after its build-time targets, when
# that option is on.

add_subdirectory(common)
add_subdirectory(blas)
add_subdirectory(solver)
add_subdirectory(sparse)
add_subdirectory(fft)
add_subdirectory(tx)
