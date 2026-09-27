/**
 * @file base_buffer.cppm
 * @brief Base class for memory buffer management
 *
 * Provides RAII-based memory buffer management with support for different
 * memory kinds (Device, Pinned, Host, Unified).
 *
 * Usage:
 *   import wwr.extension.memory_buffer;
 *   using namespace wwr::extension;
 */

export module wwr.extension.memory_buffer:base_buffer;

import std;
import wwr.extension.common;
import :memory_kind;

export namespace wwr::extension {

// ============================================================================
// Reinterpreting-view tag
// ============================================================================

/**
 * @brief Tag selecting the reinterpreting (cross-type) view constructor
 *
 * Reinterpreting a buffer's bytes as a different element type is a sharp
 * operation, so it is never an implicit conversion: a view whose element type
 * differs from its source's is only ever formed by naming this tag, or through
 * the reinterpret_buffer_view() factory below (which names it for you).
 */
struct reinterpret_view_tag_t {
  explicit reinterpret_view_tag_t() = default;
};
inline constexpr reinterpret_view_tag_t reinterpret_view{};

// ============================================================================
// Buffer Base Class
// ============================================================================

/**
 * @brief Base class template for memory buffer management
 *
 * @tparam T The element type stored in the buffer
 * @tparam K The memory kind (Device, Pinned, Host, or Unified)
 * @tparam P_alloc The error policy type for allocation operations
 * @tparam P_free The error policy type for deallocation operations (defaults to P_alloc)
 *
 * @note Provides RAII-based memory management with appropriate allocation
 *       and deallocation strategies based on the memory kind
 * @note P_free MUST NOT THROW exceptions, as it is invoked from the destructor
 *       (a throwing handle_error would std::terminate). The nothrow_error_policy
 *       constraint on the P_free slot enforces this at compile time.
 * @note An owning Derived MUST (a) provide a static
 *       allocate(T**, std::size_t, P_alloc&, std::source_location), (b) provide
 *       a member deallocate(T*, std::size_t), and (c) call destroy_() from its
 *       own destructor. See the notes on ~BaseBuffer() and destroy_().
 */
template<typename T, MemoryKind K, typename Derived,
         error_policy<typename MemoryErrorType<K>::type> P_alloc,
         nothrow_error_policy<typename MemoryErrorType<K>::type> P_free = P_alloc, bool IsView = false>
class BaseBuffer {
public:
  /** @brief Element type stored in this buffer */
  using value_type = T;

  /** @brief Storage type: std::byte for void buffers, T otherwise */
  using storage_type = std::conditional_t<std::same_as<T, void>, std::byte, T>;

  // Every buffer here holds raw storage that it fills and copies bytewise
  // (memset/memcpy, or gpuMemcpy for device kinds) and never constructs a T.
  // A non-trivially-copyable element type would leave zeroed bytes masquerading
  // as live objects, so reject it at construction - as std::atomic does.
  static_assert(std::is_trivially_copyable_v<storage_type>,
                "gpumod buffers require a trivially-copyable element type");

  static constexpr MemoryKind memory_kind = K;
  /**
     * @brief Whether this buffer is a non-owning view
     *
     * @note Supplied as a template argument rather than read back out of the
     *       derived class. The constrained/deleted copy-constructor pair below
     *       is resolved while the derived class is still incomplete, so a
     *       constraint that queried it would never be satisfied there and every
     *       view would silently pick up the deleted copy constructor.
     */
  static constexpr bool is_view = IsView;
  static constexpr bool is_device = ((K == MemoryKind::Device) || (K == MemoryKind::Unified));
  static constexpr bool is_gpu_managed = (K != MemoryKind::Host);
  static constexpr bool is_host = (K != MemoryKind::Device);
  static constexpr std::size_t element_size = sizeof(storage_type);

  /**
     * @brief Largest element count whose byte size does not overflow std::size_t
     *
     * @note Requesting more than this is rejected at construction. Without the
     *       check, num_elements * element_size wraps and yields a tiny
     *       allocation behind a buffer that still reports the huge element
     *       count, so every subsequent bounds check passes and copy/memset run
     *       off the end of the block.
     */
  static constexpr std::size_t max_num_elements =
      std::numeric_limits<std::size_t>::max() / element_size;

  /**
     * @brief Default constructor - creates an empty buffer with no allocation
     *
     * Creates a buffer with data_ = nullptr and num_elements_ = 0.
     * Can be move-assigned later to take ownership of allocated memory.
     */
  BaseBuffer() noexcept = default;

  /**
     * @brief Construct buffer with automatic allocation
     *
     * @param num_elements Number of elements to allocate
     * @param location Source location for error reporting
     *
     * @note Only available for owning buffers (not views)
     */
  BaseBuffer(const std::size_t num_elements,
             const std::source_location location = std::source_location::current())
    requires(!IsView)
  {
    allocate_checked(num_elements, location);
  }

  /**
     * @brief Construct buffer with custom allocation policy
     *
     * @param num_elements Number of elements to allocate
     * @param policy Error policy for both allocation and deallocation
     * @param location Source location for error reporting
     *
     * @note Only available for owning buffers (not views)
     */
  BaseBuffer(const std::size_t num_elements, P_alloc policy,
             const std::source_location location = std::source_location::current())
    requires(!IsView)
      : policy_alloc_(std::move(policy)), policy_free_(policy_alloc_) {
    allocate_checked(num_elements, location);
  }

  /**
     * @brief Construct buffer with separate allocation and deallocation policies
     *
     * @param num_elements Number of elements to allocate
     * @param policy_alloc Error policy for allocation
     * @param policy_free Error policy for deallocation
     * @param location Source location for error reporting
     *
     * @note Only available for owning buffers (not views)
     */
  BaseBuffer(const std::size_t num_elements, P_alloc policy_alloc, P_free policy_free,
             const std::source_location location = std::source_location::current())
    requires(!IsView)
      : policy_alloc_(std::move(policy_alloc)), policy_free_(std::move(policy_free)) {
    allocate_checked(num_elements, location);
  }

  /**
     * @brief Construct a sub-view from any buffer of the same T and K
     *
     * @param src The source buffer to create a view of
     * @param offset Start offset (in elements) into the source buffer (default: 0)
     * @param count Number of elements in the view (default: src.num_elements() - offset)
     * @param location Source location for error reporting
     *
     * @note Only available for view types (IsView == true)
     * @note Reports an error via policy_alloc_ if the range is out of bounds,
     *       leaving the view empty
     * @note src is taken by non-const reference because a view hands out a
     *       mutable T*; a view over a const buffer would launder away the const
     */
  template<typename OtherDerived, bool OtherIsView>
  BaseBuffer(BaseBuffer<T, K, OtherDerived, P_alloc, P_free, OtherIsView> &src,
             const std::size_t offset, const std::size_t count,
             const std::source_location location = std::source_location::current())
    requires(IsView)
      : policy_alloc_(src.alloc_policy()), policy_free_(src.free_policy()) {
    // Written as two subtraction-free comparisons: "offset + count > n"
    // wraps for large counts and would accept an out-of-range view.
    if (count > src.num_elements() || offset > src.num_elements() - count) {
      policy_alloc_.handle_error(MemoryInvalidValue<K>::value, location);
    } else {
      data_ = reinterpret_cast<T *>(static_cast<storage_type *>(src.data()) + offset);
      num_elements_ = count;
    }
  }

  /**
     * @brief Construct a sub-view with offset only; count defaults to remaining elements
     *
     * @param src The source buffer to create a view of
     * @param offset Start offset (in elements) into the source buffer
     * @param location Source location for error reporting
     *
     * @note Only available for view types (IsView == true)
     */
  template<typename OtherDerived, bool OtherIsView>
  BaseBuffer(BaseBuffer<T, K, OtherDerived, P_alloc, P_free, OtherIsView> &src,
             const std::size_t offset,
             const std::source_location location = std::source_location::current())
    requires(IsView)
      // An offset past the end would underflow the remaining-element count, so
      // forward a count the delegated-to bounds check is guaranteed to reject.
      : BaseBuffer(src, offset,
                   offset <= src.num_elements() ? src.num_elements() - offset : std::size_t{1},
                   location) {}

  /**
     * @brief Construct a full-buffer view from any buffer of the same T and K
     *
     * @param src The source buffer to create a view of
     * @param location Source location for error reporting
     *
     * @note Only available for view types (IsView == true)
     */
  template<typename OtherDerived, bool OtherIsView>
  BaseBuffer(BaseBuffer<T, K, OtherDerived, P_alloc, P_free, OtherIsView> &src,
             const std::source_location location = std::source_location::current())
    requires(IsView)
      : BaseBuffer(src, 0, src.num_elements(), location) {}

  /**
     * @brief Construct a non-owning view that reinterprets a buffer's bytes as T
     *
     * @tparam U The source buffer's element type, which may differ from this
     *           view's T (this is the whole point of the overload)
     * @param src The source buffer whose storage is reinterpreted
     * @param location Source location for error reporting
     *
     * @note Only available for view types (IsView == true). The byte span is
     *       preserved and num_elements() becomes size_bytes() / element_size.
     * @note Selected only by naming reinterpret_view, so a same-K view of a
     *       different element type is never formed implicitly.
     * @note Reports an error via policy_alloc_ and leaves the view empty when
     *       src's byte size is not a whole multiple of element_size, or when its
     *       base pointer is not aligned for T. Either would let num_elements()
     *       and operator[] hand out storage that is not there or is misaligned -
     *       a misaligned sub-view is the reachable case, since an allocation is
     *       aligned for its own element type but an offset into it need not be.
     * @note src is a non-const reference for the same reason as the same-type
     *       view constructors: a view hands out a mutable T*.
     */
  template<typename U, typename OtherDerived, bool OtherIsView>
  BaseBuffer(reinterpret_view_tag_t,
             BaseBuffer<U, K, OtherDerived, P_alloc, P_free, OtherIsView> &src,
             const std::source_location location = std::source_location::current())
    requires(IsView)
      : policy_alloc_(src.alloc_policy()), policy_free_(src.free_policy()) {
    const std::size_t bytes = src.size_bytes();
    void *const raw = src.data();
    // A partial trailing element, or a base the target type cannot be read at,
    // would both surface only as bad accesses later - reject them up front and
    // leave the empty-buffer invariant (data_ == nullptr <=> num_elements_ == 0)
    // intact. reinterpret_cast to uintptr_t only inspects the address; it does
    // not dereference, so it is valid for a device pointer too.
    if (bytes % element_size != 0 ||
        reinterpret_cast<std::uintptr_t>(raw) % alignof(storage_type) != 0) {
      policy_alloc_.handle_error(MemoryInvalidValue<K>::value, location);
    } else {
      data_ = static_cast<T *>(raw);
      num_elements_ = bytes / element_size;
    }
  }

  /**
     * @brief Destructor
     *
     * @note This does NOT release the allocation. By the time a base destructor
     *       runs the derived sub-object has already been destroyed, so calling
     *       Derived::deallocate() from here is undefined behaviour
     *       ([class.cdtor]/4) - and it cannot see derived state such as
     *       DeviceBufferWrapper's stream. Owning buffers release from their own
     *       destructor via destroy_(); if data_ is still live here the derived
     *       class forgot to do so, which is reported rather than leaked silently.
     */
  ~BaseBuffer() {
    if constexpr (!IsView) {
      if (data_ != nullptr) {
        policy_free_.handle_error(MemoryInvalidValue<K>::value, std::source_location::current());
      }
    }
  }

  /**
     * @brief Move constructor
     */
  BaseBuffer(BaseBuffer &&other) noexcept
      : data_(other.data_), num_elements_(other.num_elements_),
        policy_alloc_(std::move(other.policy_alloc_)), policy_free_(std::move(other.policy_free_)) {
    // Take ownership by moving the data pointer
    // Leave other in valid null state
    other.data_ = nullptr;
    other.num_elements_ = 0;
  }

  /**
     * @brief Move assignment operator
     */
  BaseBuffer &operator=(BaseBuffer &&other) noexcept {
    // Self-assignment check
    if (this != &other) {
      // Release the current contents (owning only). Safe here, unlike in
      // the destructor: the derived sub-object is still alive.
      destroy_();

      // Take ownership from other
      data_ = other.data_;
      num_elements_ = other.num_elements_;
      other.data_ = nullptr;
      other.num_elements_ = 0;
      policy_alloc_ = std::move(other.policy_alloc_);
      policy_free_ = std::move(other.policy_free_);
    }

    return *this;
  }

  // Copy operations: enabled for views, deleted for owning buffers
  BaseBuffer(const BaseBuffer &) noexcept
    requires(IsView)
  = default;
  BaseBuffer &operator=(const BaseBuffer &) noexcept
    requires(IsView)
  = default;
  BaseBuffer(const BaseBuffer &) = delete;
  BaseBuffer &operator=(const BaseBuffer &) = delete;

  /**
     * @brief Implicit conversion to pointer
     * @return Pointer to the buffer data
     */
  operator T *() noexcept { return data_; }

  /**
     * @brief Implicit conversion to const pointer
     * @return Const pointer to the buffer data
     */
  operator const T *() const noexcept { return data_; }

  /**
     * @brief Get pointer to allocated memory
     * @return Pointer to the buffer data
     */
  T *data() noexcept { return data_; }

  /**
     * @brief Get const pointer to allocated memory
     * @return Const pointer to the buffer data
     */
  const T *data() const noexcept { return data_; }

  /**
     * @brief Array subscript operator for host-accessible buffers
     * @param index Element index
     * @return Reference to element at index
     * @note Only enabled for non-device buffers (Host, Pinned, Unified)
     * @note Declared in terms of storage_type, not T. A constraint does not stop
     *       the declaration from being instantiated with the class, and "void&"
     *       is ill-formed - spelling it T& made every void buffer uninstantiable.
     *       storage_type is T for every case this operator is enabled for.
     */
  storage_type &operator[](std::size_t index) noexcept
    requires(is_host && !std::same_as<T, void>)
  {
    return data_[index];
  }

  /**
     * @brief Array subscript operator for host-accessible buffers (const)
     * @param index Element index
     * @return Const reference to element at index
     * @note Only enabled for non-device, non-void buffers (Host, Pinned, Unified)
     */
  const storage_type &operator[](std::size_t index) const noexcept
    requires(is_host && !std::same_as<T, void>)
  {
    return data_[index];
  }

  /**
     * @brief Get number of elements in the buffer
     * @return Number of elements (not bytes)
     */
  std::size_t num_elements() const noexcept { return num_elements_; }

  /**
     * @brief Get total size in bytes
     * @return Size in bytes
     */
  std::size_t size_bytes() const noexcept { return num_elements_ * element_size; }

  /**
     * @brief Get size of a single element in bytes
     * @return Size of element type T in bytes
     */
  static constexpr std::size_t get_element_size() noexcept { return element_size; }

  /**
     * @brief Get the allocation error policy
     * @return Const reference to the allocation error policy
     */
  const P_alloc &alloc_policy() const noexcept { return policy_alloc_; }

  /**
     * @brief Get the deallocation error policy
     * @return Const reference to the deallocation error policy
     */
  const P_free &free_policy() const noexcept { return policy_free_; }

protected:
  T *data_ = nullptr;            ///< Pointer to allocated memory
  std::size_t num_elements_ = 0; ///< Number of elements (not bytes)
  P_alloc policy_alloc_{};       ///< Error policy for allocation operations
  P_free policy_free_{};         ///< Error policy for deallocation operations

  // Tag type for derived classes to skip default allocation
  struct skip_default_alloc_t {};

  // Protected constructor that skips automatic allocation
  // Allows derived classes to manually allocate memory with custom parameters
  BaseBuffer(skip_default_alloc_t) noexcept {}

  /**
     * @brief Release the owned allocation and reset to the empty state
     *
     * @note Every owning buffer MUST call this from its own destructor, while
     *       the derived sub-object - and any state its deallocate() needs, such
     *       as the stream a stream-ordered allocation must be returned on - is
     *       still alive. Idempotent, so a derived destructor and a later
     *       move-assignment cannot double free.
     */
  void destroy_() noexcept {
    if constexpr (!IsView) {
      if (data_ != nullptr) {
        static_cast<Derived *>(this)->deallocate(data_, num_elements_);
        data_ = nullptr;
        num_elements_ = 0;
      }
    }
  }

private:
  /**
     * @brief Size-check, then allocate, maintaining the empty-buffer invariant
     *
     * @param num_elements Number of elements to allocate
     * @param location Source location for error reporting
     *
     * @note Derived::allocate is called as a static function: the derived
     *       sub-object does not exist yet while a base constructor runs, so a
     *       member call on it would be undefined behaviour ([class.cdtor]/4).
     * @note Guarantees data_ == nullptr <=> num_elements_ == 0, so a failed
     *       allocation under a non-fatal error policy leaves a coherent empty
     *       buffer rather than a null pointer advertising num_elements_ elements.
     */
  void allocate_checked(const std::size_t num_elements, const std::source_location location) {
    if (num_elements > max_num_elements) {
      policy_alloc_.handle_error(MemoryInvalidValue<K>::value, location);
      return;
    }
    if (num_elements == 0) {
      return; // empty buffer; no allocator call, so no malloc(0) corner case
    }

    Derived::allocate(&data_, num_elements, policy_alloc_, location);

    if (data_ != nullptr) {
      num_elements_ = num_elements;
    }
  }
};

// ============================================================================
// Buffer View Wrapper
// ============================================================================

/**
 * @brief Non-owning view over any buffer of the same T and K
 *
 * Holds a T* + size without owning the memory. Constructible from any owning
 * or view buffer sharing the same value type and memory kind.
 * Supports both full-buffer and offset+count sub-views.
 *
 * @tparam T The element type
 * @tparam K The memory kind
 * @tparam P_alloc Error policy type for allocation operations (used for sub-view bounds checking)
 * @tparam P_free Error policy type for deallocation operations (defaults to P_alloc)
 *
 * @note BufferViewWrapper satisfies the buffer_base concept.
 * @note View types are copyable; copies share the same pointer without ownership transfer.
 */
template<typename T, MemoryKind K,
         error_policy<typename MemoryErrorType<K>::type> P_alloc =
             DefaultErrorPolicy<typename MemoryErrorType<K>::type>,
         nothrow_error_policy<typename MemoryErrorType<K>::type> P_free = P_alloc>
class BufferViewWrapper
    : public BaseBuffer<T, K, BufferViewWrapper<T, K, P_alloc, P_free>, P_alloc, P_free, true> {
public:
  using Base = BaseBuffer<T, K, BufferViewWrapper<T, K, P_alloc, P_free>, P_alloc, P_free, true>;

  // Inherit BaseBuffer's constructors: the view-from-buffer and
  // sub-view-from-buffer templates.
  //
  // Spelled `Base::Base`, not `Base::BaseBuffer`: the terminal name must
  // match the nested-name-specifier for the using-declarator to name the
  // CONSTRUCTORS. `Base::BaseBuffer` instead resolves to BaseBuffer's
  // injected-class-name -- a type -- and the declaration is then ill-formed
  // without `typename`.
  //
  // Default, copy and move constructors are NOT inherited (the standard
  // excludes them); BufferViewWrapper gets its own implicit ones, which
  // reach BaseBuffer's `requires (IsView)` copy/move overloads.
  //
  // The inherited set includes the reinterpret_view constructor, so
  // `BufferViewWrapper<T, K>(reinterpret_view, src)` resolves to it; the
  // reinterpret_buffer_view() factory below is the ergonomic front end.
  using Base::Base;
};

/**
 * @brief Make a non-owning view that reinterprets src's bytes as T
 *
 * @tparam T The view's element type; the only argument to spell out, the rest
 *           are deduced from src
 * @tparam U The source buffer's element type, which may differ from T
 * @param src Any owning or view buffer; its memory kind and error policies are
 *            carried onto the returned view
 * @param location Source location for error reporting
 * @return A BufferViewWrapper<T, K> over src's storage. The byte span is
 *         preserved (num_elements() == src.size_bytes() / sizeof(T)); see the
 *         reinterpreting view constructor for the whole-multiple and alignment
 *         checks and their empty-view failure mode.
 *
 * Example:
 *   HostBuffer<float> buf(16);
 *   auto bytes = reinterpret_buffer_view<std::byte>(buf);  // 64-element byte view
 */
template<typename T, typename U, MemoryKind K, typename OtherDerived,
         error_policy<typename MemoryErrorType<K>::type> P_alloc,
         nothrow_error_policy<typename MemoryErrorType<K>::type> P_free, bool OtherIsView>
[[nodiscard]] BufferViewWrapper<T, K, P_alloc, P_free>
reinterpret_buffer_view(BaseBuffer<U, K, OtherDerived, P_alloc, P_free, OtherIsView> &src,
                        const std::source_location location = std::source_location::current()) {
  return BufferViewWrapper<T, K, P_alloc, P_free>(reinterpret_view, src, location);
}

// ============================================================================
// Buffer Concept
// ============================================================================

/**
 * @brief Concept constraining types to buffer-like interfaces
 *
 * @tparam B The type to check
 *
 * @note Checks that B has:
 *       - value_type member type
 *       - memory_kind static member
 *       - data() method returning pointer to value_type
 *       - num_elements() method returning size
 *       - size_bytes() method returning size in bytes
 */
template<typename B>
concept buffer_base = requires(B buf, const B const_buf) {
  typename B::value_type;
  { B::memory_kind } -> std::convertible_to<MemoryKind>;
  { buf.data() } -> std::same_as<typename B::value_type *>;
  { const_buf.data() } -> std::same_as<const typename B::value_type *>;
  { const_buf.num_elements() } -> std::same_as<std::size_t>;
  { const_buf.size_bytes() } -> std::same_as<std::size_t>;
};

/**
 * @brief Concept for buffer types with a specific element type
 *
 * @tparam B The buffer type to check
 * @tparam T The expected element type
 *
 * @note Use this when you need to explicitly constrain the element type.
 *       Builds on buffer_base to ensure B is a valid buffer with value_type == T.
 *
 * Example:
 *   template <buffer_typename<float> B>
 *   void process_float_buffer(B& buffer) { ... }
 */
template<typename B, typename T>
concept buffer_typename = buffer_base<B> && std::same_as<typename B::value_type, T>;

/**
 * @brief Concept requiring two buffers have the same value_type
 *
 * @tparam B1 First buffer type
 * @tparam B2 Second buffer type
 *
 * @note Use this to ensure type compatibility between buffers in copy operations.
 *       Both types must satisfy buffer_base and have matching element types.
 *
 * Example:
 *   template <buffer_base B1, buffer_base B2>
 *     requires same_value_type<B1, B2>
 *   void copy_buffer(const B1& src, B2& dst) { ... }
 */
template<typename B1, typename B2>
concept same_value_type = buffer_base<B1> && buffer_base<B2> &&
                          std::same_as<typename B1::value_type, typename B2::value_type>;

} // namespace wwr::extension
