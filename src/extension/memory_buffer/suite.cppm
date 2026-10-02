/**
 * @file suite.cppm
 * @brief Buffer/view alias binders that emit a whole prelude from one place
 *
 * Every consumer of the memory-buffer wrappers writes the same alias prelude
 * before its first allocation: one alias per memory kind, each spelling the
 * wrapper's error policies, plus a matching view alias whose policies must be
 * kept identical to its buffer's (BufferViewWrapper's view constructors take the
 * *same* policy pair as the source buffer, so a drifted pair fails with a
 * no-matching-constructor wall rather than a legible error). This partition moves
 * that wrapper -> error-family mapping and the buffer/view pairing into the
 * library. It binds no policy of its own: the consumer names one (kit::AbortPolicy
 * or their own).
 *
 * Two entry points over one mechanism:
 *   - buffer_suite<M> / device_buffer_suite<M, H> take a policy MAP -- a type
 *     naming an alloc/free policy per error type (policy_map, error_handling) --
 *     for a consumer whose kinds want different policies.
 *   - buffers<P> / device_buffers<P, H> are the single-policy convenience over
 *     the map, for the common case of one policy template across all kinds.
 *
 * The split (host/pinned in buffer_suite, the handle-backed device and unified
 * added by device_buffer_suite) is what lets a host-only consumer skip the device
 * handle: H is constrained bare on device_buffer_suite and checked when it is
 * named, so there is no sentinel handle to invent and buffer_suite<M>::device /
 * ::unified simply do not exist.
 *
 * These live in the nested namespace wwr::extension::kit -- the opt-in layer of
 * ready-made helpers -- not in wwr::extension directly, so a plain
 * `using namespace wwr::extension;` does not pull generic names like `buffers` or
 * `device_buffers` into a consumer's scope. A consumer opts in with
 * `using namespace wwr::extension::kit;` or names `kit::` explicitly.
 *
 * Usage:
 *   import wwr.extension.memory_buffer;
 *   using namespace wwr::extension;
 *
 *   using Buf = kit::device_buffers<kit::AbortPolicy, MyDeviceHandle>; // AbortPolicy is template<class E>
 *   Buf::device<float> d(1024, dev);
 *   Buf::host<float>   h(1024);   // stdHostMemoryError_t, chosen by the suite
 */

export module wwr.extension.memory_buffer:suite;

import :base_buffer;
import :memory_kind;
import :host_memory;
import :device_buffer;
import :pinned_buffer;
import :unified_buffer;
import :host_buffer;
import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension::kit {

/**
 * @brief The non-owning view type over buffer B, carrying B's kind and policies
 *
 * @tparam B An owning or view buffer type
 *
 * @note Derived from B's published aliases rather than from a policy map, so a
 *       view always follows whatever its buffer alias is -- the point of #186's
 *       nested {alloc,free}_policy_type members alongside value_type/memory_kind.
 *       A suite spells
 *       its view aliases `view_of<host<T>>`, not by re-binding the map, precisely
 *       so a consumer that inherits the suite and shadows `host` cannot leave an
 *       inherited `host_view` pointing at the old policies.
 */
template<buffer_base B>
using view_of = BufferViewWrapper<typename B::value_type, B::memory_kind,
                                  typename B::alloc_policy_type, typename B::free_policy_type>;

/**
 * @brief Host/pinned buffer and view aliases bound from one policy map
 *
 * @tparam M A policy map: `template<class E> using alloc/free = ...;`. Feeds only
 *           buffer slots -- host speaks stdHostMemoryError_t, pinned speaks
 *           wwrError_t -- so the map is checked against both families.
 *
 * @note No device or unified buffer here: both need a handle, which
 *       device_buffer_suite adds. The requires-clause is checked eagerly when the
 *       specialization is named, so a malformed map fails at the consumer's
 *       `using` line rather than deep inside a later allocation.
 */
template<typename M>
  requires policy_map<M, stdHostMemoryError_t> && policy_map<M, wwrError_t>
struct buffer_suite {
  template<typename T>
  using host = HostBufferWrapper<T, typename M::template alloc<stdHostMemoryError_t>,
                                 typename M::template free<stdHostMemoryError_t>>;
  template<typename T>
  using pinned = PinnedBufferWrapper<T, typename M::template alloc<wwrError_t>,
                                     typename M::template free<wwrError_t>>;

  template<typename T>
  using host_view = view_of<host<T>>;
  template<typename T>
  using pinned_view = view_of<pinned<T>>;
};

/**
 * @brief buffer_suite plus the handle-backed kinds -- device and unified -- and
 *        their views, backed by handle H
 *
 * @tparam M A policy map, as for buffer_suite.
 * @tparam H The device handle backing the device and unified buffers. For device
 *           it also picks the alloc strategy by tier (see the device_handle ladder
 *           in wwr.extension.handle); unified reads only dev_idx() (managed memory
 *           has one alloc call and a sync-only free, so its tier is immaterial).
 *
 * @note Both buffers' third policy (P_device_access, the ScopedDeviceIndex switch)
 *       is bound from the map's free policy: it carries the same
 *       nothrow-on-the-destructor-path constraint, so the two-key map suffices
 *       and no separate slot is asked of the consumer.
 */
template<typename M, device_handle H>
struct device_buffer_suite : buffer_suite<M> {
  template<typename T>
  using device = DeviceBufferWrapper<T, typename M::template alloc<wwrError_t>,
                                     typename M::template free<wwrError_t>,
                                     typename M::template free<wwrError_t>, H>;
  template<typename T>
  using unified = UnifiedBufferWrapper<T, typename M::template alloc<wwrError_t>,
                                       typename M::template free<wwrError_t>,
                                       typename M::template free<wwrError_t>, H>;
  template<typename T>
  using device_view = view_of<device<T>>;
  template<typename T>
  using unified_view = view_of<unified<T>>;
};

/**
 * @brief A policy map that uses one policy template for both the alloc and free
 *        slots of every error type
 *
 * @tparam P A policy template on the error type (`template<class E> class P`).
 *
 * @note The adapter under the single-policy convenience below. P must be usable
 *       in the free slot -- a nothrow_error_policy -- since it serves both slots.
 */
template<template<typename> class P>
struct single_policy_map {
  template<typename E>
  using alloc = P<E>;
  template<typename E>
  using free = P<E>;
};

/**
 * @brief Single-policy convenience: host/pinned aliases from one policy
 *
 * @tparam P A policy template on the error type; used across every kind.
 *
 * @note The terse front door for the common case, a thin alias over
 *       buffer_suite<single_policy_map<P>> -- not a competing design.
 */
template<template<typename> class P>
using buffers = buffer_suite<single_policy_map<P>>;

/**
 * @brief Single-policy convenience with the handle-backed device and unified
 *        buffers added, backed by H
 *
 * @tparam P A policy template on the error type; used across every kind.
 * @tparam H The device handle backing the device and unified buffers.
 */
template<template<typename> class P, device_handle H>
using device_buffers = device_buffer_suite<single_policy_map<P>, H>;

} // namespace wwr::extension::kit
