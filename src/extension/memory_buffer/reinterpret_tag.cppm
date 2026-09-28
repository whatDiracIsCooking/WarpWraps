/**
 * @file reinterpret_tag.cppm
 * @brief Disambiguation tag for the reinterpreting (cross-type) view constructor
 *
 * The tag is exported from this partition so the other partitions can name it,
 * but the primary interface unit imports it WITHOUT re-exporting (plain `import
 * :reinterpret_tag`, not `export import`). Consumers of the module therefore
 * cannot spell reinterpret_view: the reinterpreting view is reached only through
 * reinterpret_buffer_view(), while direct BufferViewWrapper(reinterpret_view,
 * src) construction stays an in-module implementation detail.
 */

export module wwr.extension.memory_buffer:reinterpret_tag;

export namespace wwr::extension {

/**
 * @brief Tag selecting the reinterpreting (cross-type) view constructor
 *
 * Reinterpreting a buffer's bytes as a different element type is a sharp
 * operation, so it is never an implicit conversion: a view whose element type
 * differs from its source's is only ever formed by naming this tag, which the
 * reinterpret_buffer_view() factory does on the caller's behalf.
 */
struct reinterpret_view_tag_t {
  explicit reinterpret_view_tag_t() = default;
};
inline constexpr reinterpret_view_tag_t reinterpret_view{};

} // namespace wwr::extension
