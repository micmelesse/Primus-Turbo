#include <deque>
#include <functional>
#include <memory>
#include <mutex>

namespace primus_turbo {
template <typename T> class AnyCached {
public:
    using ElementType = std::unique_ptr<T, std::function<void(T *)>>;

public:
    AnyCached() = default;

    template <typename... Args> auto get(Args... args) {
        auto destructor = [this](T *elem) {
            std::lock_guard<std::mutex> lock(mutex_);
            pool_.push_back(std::unique_ptr<T>(elem));
        };

        // Try to acquire an exists element from the pool.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!pool_.empty()) {
                auto *elem = pool_.front().release();
                pool_.pop_front();
                return ElementType(elem, destructor);
            }
        }
        // otherwise, allocate a new element that will be returned to the pool on
        // destruction.
        return ElementType(std::make_unique<T>(std::forward<Args>(args)...).release(), destructor);
    }

    void empty_cache() {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.clear();
    }

private:
    alignas(64) std::mutex mutex_;
    std::deque<std::unique_ptr<T>> pool_{};
};
} // namespace primus_turbo
