/**
 * @file policy_slot.cppm
 * @brief Storage for a policy slot that collapses into an earlier same-type
 *        empty policy, so a redundant stateless slot costs nothing
 *
 * A wrapper (handle, buffer) carries several error-policy slots -- create/
 * destroy, alloc/free, plus the device slot the device-bound layer adds. The
 * common case is that two or more of them are the *same* stateless policy type.
 * [[no_unique_address]] alone does not make that free: [intro.object] forbids
 * two subobjects of the same type from sharing an address, so a second
 * same-type empty member still rounds the object up by a slot.
 *
 * policy_slot is the fix for a "second-or-later" slot. When its type is empty
 * and matches one of the slots that precede it, it stores nothing and resolve()
 * hands back the canonical instance the caller supplies (a sibling member, or a
 * base accessor across an inheritance boundary). Stateful or distinct policies
 * are stored as usual -- collapsing is gated on emptiness precisely because an
 * empty policy has no state to lose when one instance serves two roles.
 *
 * The first slot in a class is never wrapped: nothing precedes it, so it is
 * always stored and is the canonical everyone else routes to.
 *
 * Usage:
 *   import wwr.extension.common;
 *
 *   [[no_unique_address]] P_create policy_create_{};                 // canonical
 *   [[no_unique_address]] policy_slot<P_destroy, P_create> destroy_; // collapses into create
 *   P_destroy &destroy_policy() { return destroy_.resolve(policy_create_); }
 */

export module wwr.extension.common:policy_slot;

import std;

export namespace wwr::extension {

/// @brief Empty stand-in stored in place of a collapsed policy slot.
struct elided_policy {};

/**
 * @brief A policy slot that elides itself when it duplicates an earlier empty one
 *
 * @tparam Slot The policy type this slot holds.
 * @tparam Earlier The slots declared before this one, in order. The slot
 *         collapses when `Slot` is empty and identical to any of them.
 */
template<typename Slot, typename... Earlier>
class policy_slot {
  static constexpr bool collapsed =
      std::is_empty_v<Slot> && (std::same_as<Slot, Earlier> || ...);

  [[no_unique_address]] std::conditional_t<collapsed, elided_policy, Slot> storage_{};

public:
  policy_slot() = default;

  /// Store `s`, unless this slot collapsed -- then `s` is an empty stateless
  /// policy with nothing to carry, so it is discarded and the canonical serves.
  explicit policy_slot(Slot s) {
    if constexpr (!collapsed)
      storage_ = std::move(s);
  }

  /// The live policy for this slot: its own instance, or `canonical` when the
  /// slot collapsed. `canonical` is read only on the collapsed path (where its
  /// type is `Slot`); otherwise the argument is ignored, so any earlier slot
  /// may be passed.
  template<typename Canonical>
  Slot &resolve(Canonical &canonical) noexcept {
    if constexpr (collapsed)
      return canonical;
    else
      return storage_;
  }
  template<typename Canonical>
  const Slot &resolve(const Canonical &canonical) const noexcept {
    if constexpr (collapsed)
      return canonical;
    else
      return storage_;
  }
};

} // namespace wwr::extension
