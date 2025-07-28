#pragma once
#include "primus_turbo/macros.h"

#include <functional>
#include <hip/hip_runtime.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace primus_turbo::async_tp {

#define FOREACH_ALL_TURBO_ASYNC_TP_CACHED_TYPE(_)                                                  \
    _(0, kHIPStream, hipStream_t)                                                                  \
    _(1, kHIPEvent, hipEvent_t)

enum class CachedType {
#define DEFINE_ENUM_CACHED_TYPE(idx, dtype, ...) dtype = idx,
    FOREACH_ALL_TURBO_ASYNC_TP_CACHED_TYPE(DEFINE_ENUM_CACHED_TYPE)
#undef DEFINE_ENUM_CACHED_TYPE
};

template <CachedType T> struct CachedTypeToRealType;
template <typename T> struct RealTypeToCachedType;

#define DEFINE_CACHED_TYPE_TO_REAL_TYPE(idx, dtype, real_type, ...)                                \
    template <> struct CachedTypeToRealType<CachedType::dtype> {                                   \
        using value = real_type;                                                                   \
    };                                                                                             \
    template <> struct RealTypeToCachedType<real_type> {                                           \
        static constexpr CachedType value = CachedType::dtype;                                     \
    };

FOREACH_ALL_TURBO_ASYNC_TP_CACHED_TYPE(DEFINE_CACHED_TYPE_TO_REAL_TYPE)
#undef DEFINE_CACHED_TYPE_TO_REAL_TYPE

class AnyPool {

  public:
    static AnyPool *Instance() {
        static AnyPool instance;
        return &instance;
    }

    template <typename T> std::unique_ptr<T> get() {
        auto key        = RealTypeToCachedType<T>::value;
        auto destructor = [this, key](T *item) {
            std::lock_guard<std::mutex> lock(mutex_);
            res_[key].push_back(std::unique_ptr<T>(item));
        };

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto                        it = res_.find(key);
            if (it != res_.end() && !it->second.empty()) {
                auto *item = it->second.back().release();
                it->second.pop_back();
            }
        }
        return std::unique_ptr<T>(std::make_unique<T>().release(), destructor);
    }

  private:
    AnyPool() = default;

    struct Holder {
        std::unique_ptr<void, std::function<void(void *)>> ptr;

        template <typename T>
        Holder(std::unique_ptr<T> &&p, std::function<void(void *)> deleter)
            : ptr(p.release(), deleter) {}
    };

    alignas(64) std::mutex mutex_;
    std::unordered_map<CachedType, std::vector<Holder>> res_;
};
} // namespace primus_turbo::async_tp
