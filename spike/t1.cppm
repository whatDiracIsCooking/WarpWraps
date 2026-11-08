// Spike 1: does g++-16 accept `import std` in a module whose global module
// fragment pulls in a real CUDA vendor header, and re-export its decls?
module;
#include <cublas_v2.h>
export module t1;
import std;

export using ::cublasHandle_t;
export using ::cublasStatus_t;
export using ::cublasCreate_v2;
export using ::cublasDestroy_v2;
export using ::cublasSgemm_v2;

export std::string wwr_spike_name() { return std::string{"t1"}; }
