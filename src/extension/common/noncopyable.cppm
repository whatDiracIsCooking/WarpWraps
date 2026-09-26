/**
 * @file noncopyable.cppm
 * @brief Mixin that deletes copy operations while allowing moves
 *
 * Usage:
 *   import gpumod.extension.common;
 *
 *   class MyResource : private NonCopyable { ... };
 */

export module gpumod.extension.common:noncopyable;

import std;

export namespace gpumod::extension {

/**
 * @brief Mixin that deletes copy operations while allowing moves
 *
 * Inherit privately. The protected destructor prevents polymorphic deletion
 * through a base pointer.
 */
class NonCopyable {
public:
  // Public special members (matches std::unique_ptr convention)
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
  // Explicitly allow move operations (required for derived classes to be
  // movable)
  NonCopyable(NonCopyable &&) = default;
  NonCopyable &operator=(NonCopyable &&) = default;

protected:
  ~NonCopyable() = default; // Protected: prevents deletion through base pointer
};

} // namespace gpumod::extension
