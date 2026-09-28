/**
 * @file stream_event_pair.cppm
 * @brief Pair of GPU stream and event handles
 *
 * Provides StreamEventPair class bundling a stream and an event together.
 */

export module wwr.extension.runtime:stream_event_pair;

import :gpu_stream;
import :gpu_event;
import wwr.extension.common;
import wwr.runtime_api;
import std;

export namespace wwr::extension {

/**
 * @brief Configuration for StreamEventPair construction.
 *
 * The stream and event are both created on `device` (default 0). The flag/
 * priority fields are optional — unset fields use the default GPU creation API.
 * If stream_priority is set without stream_flags, flags default to 0.
 */
struct StreamEventConfig {
  int device = 0;
  std::optional<unsigned int> stream_flags;
  std::optional<int> stream_priority;
  std::optional<unsigned int> event_flags;
};

/**
 * @brief Bundles a stream and an event together.
 *
 * Owns a stream and an associated event. Construction is controlled via
 * StreamEventConfig to allow optional flags and priority.
 */
class StreamEventPair : private NonCopyable {
private:
  GpuStreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> stream_;
  GpuEventWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> event_;

  static GpuStreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> make_stream(const StreamEventConfig &cfg) {
    if (cfg.stream_priority)
      return GpuStreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>>(cfg.device, cfg.stream_flags.value_or(wwrStreamDefault),
                       *cfg.stream_priority);
    if (cfg.stream_flags)
      return GpuStreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>>(cfg.device, *cfg.stream_flags);
    return GpuStreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>>(cfg.device);
  }

public:
  StreamEventPair() = default;

  explicit StreamEventPair(StreamEventConfig cfg)
      : stream_(make_stream(cfg)),
        event_(cfg.event_flags ? GpuEventWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>>(cfg.device, *cfg.event_flags) : GpuEventWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>>(cfg.device)) {}

  // Copy operations are implicitly deleted via the NonCopyable base.

  StreamEventPair(StreamEventPair &&) = default;
  StreamEventPair &operator=(StreamEventPair &&) = default;

  GpuStreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> &gpu_stream() { return stream_; }
  const GpuStreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> &gpu_stream() const { return stream_; }

  GpuEventWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> &gpu_event() { return event_; }
  const GpuEventWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> &gpu_event() const { return event_; }

  wwrStream_t stream_raw() const { return stream_.get(); }
  wwrEvent_t event_raw() const { return event_.get(); }

  // Borrow-safe ops are free functions now; qualify so the member `record`
  // below does not shadow the free `record` via ordinary (pre-ADL) lookup.
  wwrError_t record() { return wwr::extension::record(event_, stream_.get()); }
  wwrError_t record(const unsigned int flags) {
    return wwr::extension::record(event_, stream_.get(), flags);
  }
  wwrError_t stream_sync() { return wwr::extension::sync(stream_); }
  wwrError_t event_sync() { return wwr::extension::sync(event_); }
};

} // namespace wwr::extension
