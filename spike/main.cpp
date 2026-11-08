import t1;
import std;
int main() {
  cublasHandle_t h{};
  std::print("spike: {} cublasCreate_v2={}\n", wwr_spike_name(),
             reinterpret_cast<void *>(&cublasCreate_v2));
  (void)h;
  return 0;
}
