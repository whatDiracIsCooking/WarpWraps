# Backend-neutral extension modules - higher-level wrappers and utilities
# written once against src's wwr* names. Built for either
# WWR_GPU_BACKEND; nothing here imports src/cuda or src/hip directly.

add_subdirectory(common)
add_subdirectory(blas)
add_subdirectory(solver)
add_subdirectory(sparse)
add_subdirectory(fft)
add_subdirectory(rand)
add_subdirectory(thrust)
