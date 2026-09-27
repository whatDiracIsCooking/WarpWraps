/**
 * @file gpu_mem_pool.cppm
 * @brief RAII wrapper for GPU memory pool handles
 *
 * Provides GpuMemPool class for automatic GPU memory pool management.
 */

export module gpumod.extension.runtime:gpu_mem_pool;

import gpumod.runtime_api;
import gpumod.extension.common;
import gpumod.extension.handle;
import std;

export namespace gpumod::extension {

/**
 * @brief RAII wrapper for GPU memory pool
 *
 * Automatically creates a GPU memory pool on construction and destroys it on destruction.
 * Supports move semantics for transferring ownership.
 *
 * @tparam P_create Error policy type for creation (defaults to DefaultErrorPolicy<gpuError_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<gpuError_t> P_create = DefaultErrorPolicy<gpuError_t>,
         nothrow_error_policy<gpuError_t> P_destroy = P_create>
class GpuMemPoolWrapper
    : public DeviceBoundHandle<gpuMemPool_t, GpuMemPoolWrapper<P_create, P_destroy>, P_create,
                            P_destroy> {
private:
  using Base =
      DeviceBoundHandle<gpuMemPool_t, GpuMemPoolWrapper<P_create, P_destroy>, P_create, P_destroy>;

  /// @note Use 1MB as default release threshold
  static constexpr unsigned int default_threshold = 1024u * 1024u; // 1MB in bytes

  static gpuMemPoolProps make_default_props(int dev_idx) {
    gpuMemPoolProps props = {};
    props.allocType = gpuMemAllocationTypePinned;
    props.handleTypes = gpuMemHandleTypeNone;
    props.location.type = gpuMemLocationTypeDevice;
    props.location.id = dev_idx;
    return props;
  }

public:
  // The `GpuMemPool(int dev_idx = 0)` default/per-device constructor, inherited
  // from DeviceBoundHandle. With dev_idx as the mandatory first argument there is no
  // longer any collision with the `(dev_idx, release_threshold)` overload below,
  // so the base's device-index constructor is inherited like GpuStream/GpuEvent.
  using DeviceBoundHandle<gpuMemPool_t, GpuMemPoolWrapper<P_create, P_destroy>, P_create,
                       P_destroy>::DeviceBoundHandle;

  /// @brief Create a GPU memory pool on `dev_idx` with default properties and a custom release threshold
  /// @param dev_idx Device to create the pool on
  /// @param release_threshold Maximum bytes to hold in pool before returning memory to the OS
  /// @param location Source location where creation was requested
  GpuMemPoolWrapper(const int dev_idx, const unsigned int release_threshold,
                    std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_create_t{}) {
    Base::select_device(dev_idx, location);
    const auto props = make_default_props(dev_idx);
    gpu_check(gpuMemPoolCreate(&this->handle_, &props), this->policy_create_, location);
    const unsigned int threshold = release_threshold;
    gpu_check(gpuMemPoolSetAttribute(this->handle_, gpuMemPoolAttrReleaseThreshold,
                                     static_cast<void *>(const_cast<unsigned int *>(&threshold))),
              this->policy_create_, location);
    this->record_device();
  }

  /// @brief Create a GPU memory pool from explicit properties
  /// @param props Properties for memory pool creation; props.location.id names the device
  /// @param release_threshold Maximum bytes to hold in pool before returning memory to the OS
  /// @param location Source location where creation was requested
  GpuMemPoolWrapper(const gpuMemPoolProps &props,
                    const unsigned int release_threshold = default_threshold,
                    std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_create_t{}) {
    // Select the device the props name before creating, so the pool's device
    // and the current device stay consistent -- as the other constructors do.
    Base::select_device(props.location.id, location);
    gpu_check(gpuMemPoolCreate(&this->handle_, &props), this->policy_create_, location);
    const unsigned int threshold = release_threshold;
    gpu_check(gpuMemPoolSetAttribute(this->handle_, gpuMemPoolAttrReleaseThreshold,
                                     static_cast<void *>(const_cast<unsigned int *>(&threshold))),
              this->policy_create_, location);
    this->record_device();
  }

  /// @brief Create a GPU memory pool (default properties)
  /// @param handle Output parameter for the created memory pool
  /// @param location Source location where creation was requested
  void create(gpuMemPool_t *handle, std::source_location location) {
    // Reached through DeviceBoundHandle's default create path, which has already
    // made dev_idx the current device; read it back so props names it.
    int dev_idx = 0;
    gpuGetDevice(&dev_idx);
    auto props = make_default_props(dev_idx);
    gpu_check(gpuMemPoolCreate(handle, &props), this->policy_create_, location);
    const unsigned int threshold = default_threshold;
    gpu_check(gpuMemPoolSetAttribute(*handle, gpuMemPoolAttrReleaseThreshold,
                                     static_cast<void *>(const_cast<unsigned int *>(&threshold))),
              this->policy_create_, location);
  }

  /// @brief Destroy a GPU memory pool
  /// @param handle The memory pool to destroy
  void destroy(gpuMemPool_t handle) {
    if (handle != nullptr) {
      gpu_check(gpuMemPoolDestroy(handle), this->policy_destroy_);
    }
  }
};

/**
 * @brief Convenient alias for GpuMemPoolWrapper with default error policies
 *
 * Usage:
 *   GpuMemPool pool;  // Instead of GpuMemPoolWrapper<>
 */
using GpuMemPool = GpuMemPoolWrapper<>;

/// @brief Non-owning, copyable view of a memory pool handle (carries its device
///        index). Returned by GpuMemPool::view(); has no borrow-safe operations
///        of its own -- a pool handle is consumed by allocation calls.
using GpuMemPoolView = DeviceBoundHandleView<gpuMemPool_t>;

} // namespace gpumod::extension
