# Backend-neutral extension modules -- higher-level wrappers and utilities
# written once against src/'s gpu* names. Built for either WWR_GPU_BACKEND;
# nothing here imports src/cuda or src/hip directly.
#
# error_handling is added first: it imports only std, and common links it.
# common is added next: every other module here links it. handle (the RAII
# handle layer) is added next -- it links common, and the runtime/memory_buffer/
# blas/solver/fft/sparse modules link it, so it must precede them. bridge (the
# header-only host/device type bridges) precedes parallel_for, which links it;
# parallel_for (the header-only device launcher) precedes init_state and
# random_normal, which use it.
add_subdirectory(error_handling)
add_subdirectory(common)
add_subdirectory(handle)
add_subdirectory(bridge)
add_subdirectory(parallel_for)
add_subdirectory(runtime)
add_subdirectory(memory_buffer)
add_subdirectory(blas)
add_subdirectory(solver)
add_subdirectory(fft)
add_subdirectory(sparse)
add_subdirectory(init_state)
add_subdirectory(random_normal)
