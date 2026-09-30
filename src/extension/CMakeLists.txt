# Backend-neutral extension modules -- higher-level wrappers and utilities
# written once against src/'s wwr* names. Built for either WWR_GPU_BACKEND;
# nothing here imports src/cuda or src/hip directly.
#
# error_handling is added first: it imports only std, and common links it.
# common is added next: every other module here links it. handle (the RAII
# handle layer) is added next -- it links common, and the runtime/memory_buffer/
# blas/solver/fft/sparse modules link it, so it must precede them.
# parallel_for (the header-only device launcher) precedes init_state and
# random_normal, which use it. tx is independent -- it links only wwr.tx (no
# extension module), so its position does not matter.
add_subdirectory(error_handling)
add_subdirectory(common)
add_subdirectory(handle)
add_subdirectory(parallel_for)
add_subdirectory(runtime)
add_subdirectory(memory_buffer)
add_subdirectory(blas)
add_subdirectory(solver)
add_subdirectory(fft)
add_subdirectory(sparse)
# texture/surface: thin BaseHandle RAII wrappers, each linking handle + common +
# runtime_api (all added above); independent of each other, order-free here.
add_subdirectory(texture)
add_subdirectory(surface)
add_subdirectory(init_state)
add_subdirectory(random_normal)
add_subdirectory(tx)
