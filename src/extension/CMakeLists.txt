# Backend-neutral extension modules -- higher-level wrappers and utilities
# written once against src/'s wwr* names. Built for either WWR_GPU_BACKEND;
# nothing here imports src/cuda or src/hip directly.
#
# error_handling is added first: it imports only std, and common links it.
# common is added next: every other module here links it. handle (the RAII
# handle layer) is added next -- it links common, and the runtime/memory_buffer/
# blas/solver/fft/sparse/ccl modules link it, so it must precede them.
# parallel_for (the header-only device launcher) precedes init_state and
# random_normal, which use it. tx is independent -- it links only wwr.tx (no
# extension module), so its position does not matter.
add_subdirectory(error_handling)
add_subdirectory(common)
add_subdirectory(handle)
add_subdirectory(parallel_for)
add_subdirectory(runtime)
add_subdirectory(memory_buffer)
# comp (the compression error policy + scratch-buffer RAII) links memory_buffer
# for DeviceBufferWrapper, so it follows it.
if(WWR_WITH_COMP)
  add_subdirectory(comp)
endif()
add_subdirectory(blas)
# blaslt links wwr::extension::blas (it reuses blas's wwrblasStatus_t error
# policy, since wwrblasLtStatus_t is the same underlying type), so it follows
# blas here.
add_subdirectory(blaslt)
add_subdirectory(solver)
add_subdirectory(fft)
add_subdirectory(sparse)
# ccl (NCCL / RCCL collectives): a BaseHandle RAII communicator + typed errors,
# no layer-2 wrapper. Links handle + common like the handle layers above. Gated
# on WWR_WITH_CCL, which drops the wwr.ccl module this subdir links (see src/).
if(WWR_WITH_CCL)
  add_subdirectory(ccl)
endif()
# tensor: a library-handle layer like the four above -- RAII over the cuTENSOR /
# hipTensor handle and descriptor create/destroy pairs plus the status error
# policy. Links handle + common (added above); no layer-2 execution wrapper.
if(WWR_WITH_TENSOR)
  add_subdirectory(tensor)
endif()
# texture/surface: thin BaseHandle RAII wrappers, each linking handle + common +
# runtime_api (all added above); independent of each other, order-free here.
add_subdirectory(texture)
add_subdirectory(surface)
add_subdirectory(init_state)
add_subdirectory(random_normal)
add_subdirectory(tx)
