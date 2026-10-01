/**
 * @file ccl_comm.cppm
 * @brief RAII wrapper for a GPU collectives communicator
 *
 * Provides CommWrapper, which initializes a wwrcclComm_t (NCCL / RCCL
 * communicator) on construction and destroys it with wwrcclCommDestroy on scope
 * exit -- removing the leak-prone hand-paired init/destroy the raw API demands.
 *
 * A communicator is a pointer (ncclComm_t), so it rides BaseHandle's
 * null-sentinel liveness directly: no explicit ownership flag and no
 * device-access policy parameter (unlike the stream-bound library handles, which
 * select and record a device). But it is created with arguments -- rank/count or
 * a device list -- that the create() hook cannot carry, so it takes BaseHandle's
 * skip-default-create path like TextureObject/SurfaceObject: each constructor
 * runs the vendor init itself and records ownership with adopt(). adopt()'s bool
 * is ignored for a pointer handle (liveness is the null sentinel already in
 * handle_), so a failed init under a recoverable policy leaves handle_ null and
 * valid() false.
 *
 * Two construction paths are exposed:
 *  - single-process single-GPU, via wwrcclCommInitAll(&comm, 1, &device): one
 *    communicator over one device, which runs on a single card and is the
 *    construct/destroy path the test exercises;
 *  - the general rank path, via wwrcclCommInitRank(&comm, nranks, id, rank),
 *    which needs a wwrcclUniqueId broadcast across the participating ranks.
 * The multi-device wwrcclCommInitAll (ndev > 1) and the config/scalable/split
 * variants are left to the caller on get(); this layer owns the common cases.
 *
 * Usage:
 *   import wwr.extension.ccl;
 *
 *   // the kit's opt-in policy, or your own; the core forces none.
 *   using wwr::extension::kit::AbortPolicy;
 *   wwr::extension::CommWrapper<AbortPolicy<wwrcclResult_t>, AbortPolicy<wwrcclResult_t>>
 *       comm{wwr::extension::single_device, 0};  // one communicator on device 0
 *   wwrcclAllReduce(send, recv, n, WWRCCL_FLOAT32, WWRCCL_SUM, comm.get(), stream);
 *   // wwrcclCommDestroy on scope exit
 */

export module wwr.extension.ccl:ccl_comm;

import :ccl_error;
import wwr.ccl;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/// @brief Tag selecting CommWrapper's single-process single-GPU constructor,
///        which initializes one communicator over one device with
///        wwrcclCommInitAll(&comm, 1, &device). Distinguishes that overload from
///        the rank constructor, whose leading int would otherwise collide.
struct single_device_t {};
inline constexpr single_device_t single_device{};

/**
 * @brief RAII wrapper for a GPU collectives communicator
 *
 * @tparam P_create Error policy for initialization, typed to wwrcclResult_t
 * @tparam P_destroy Error policy for destruction, typed to wwrcclResult_t
 *
 * @note P_destroy MUST NOT THROW -- it is called from the destructor.
 */
template<error_policy<wwrcclResult_t> P_create, nothrow_error_policy<wwrcclResult_t> P_destroy>
class CommWrapper
    : public BaseHandle<wwrcclComm_t, CommWrapper<P_create, P_destroy>, P_create, P_destroy> {
  using Base = BaseHandle<wwrcclComm_t, CommWrapper<P_create, P_destroy>, P_create, P_destroy>;

public:
  /// @brief Initialize one communicator over a single device (single process,
  ///        single GPU) with wwrcclCommInitAll(&comm, 1, &device)
  /// @param device The device index this communicator runs on
  /// @param policy_create Error policy for the init call
  /// @param policy_destroy Error policy for the destroy call (must not throw)
  /// @param location Source location where creation was requested
  CommWrapper(single_device_t, int device, P_create policy_create = {}, P_destroy policy_destroy = {},
              std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_create_t{}, std::move(policy_create),
             std::move(policy_destroy)) {
    this->adopt(gpu_check(wwrcclCommInitAll(&this->handle_, 1, &device), this->create_policy(),
                          location));
  }

  /// @brief Initialize this rank's communicator with wwrcclCommInitRank
  /// @param nranks Total number of ranks in the communicator
  /// @param id The unique id (from wwrcclGetUniqueId) broadcast to every rank
  /// @param rank This process's rank, in [0, nranks)
  /// @param policy_create Error policy for the init call
  /// @param policy_destroy Error policy for the destroy call (must not throw)
  /// @param location Source location where creation was requested
  CommWrapper(int nranks, wwrcclUniqueId id, int rank, P_create policy_create = {},
              P_destroy policy_destroy = {},
              std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_create_t{}, std::move(policy_create),
             std::move(policy_destroy)) {
    this->adopt(gpu_check(wwrcclCommInitRank(&this->handle_, nranks, id, rank), this->create_policy(),
                          location));
  }

  /// @brief Destroy the communicator (BaseHandle destructor hook)
  /// @param handle The communicator to destroy
  void destroy(wwrcclComm_t handle) {
    gpu_check(wwrcclCommDestroy(handle), this->destroy_policy());
  }
};

} // namespace wwr::extension
