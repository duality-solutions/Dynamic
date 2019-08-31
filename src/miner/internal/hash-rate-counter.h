// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_INTERNAL_HASH_RATE_COUNTER_H
#define DYNAMIC_INTERNAL_HASH_RATE_COUNTER_H

#include <atomic>
#include <memory>


struct HashRateCounter;
using HashRateCounterRef = std::shared_ptr<HashRateCounter>;

/**
 * Hash rate counter struct.
 */
struct HashRateCounter : public std::enable_shared_from_this<HashRateCounter> {
private:
    std::atomic<int64_t> _count{0};
    std::atomic<int64_t> _timer_start{0};
    std::atomic<int64_t> _count_per_sec{0};

    HashRateCounterRef _parent;

public:
    explicit HashRateCounter() : _parent(nullptr){};
    explicit HashRateCounter(HashRateCounterRef parent) : _parent(parent){};

    // Returns hash rate per second
    operator int64_t() { return _count_per_sec; };

    // Creates new child counter
    HashRateCounterRef MakeChild() { return std::make_shared<HashRateCounter>(shared_from_this()); }

    // Increments counter
    void Increment(int64_t amount);

    // Resets counter and timer
    void Reset();

    // Returns start time
    int64_t start() const { return _timer_start; };
};

#endif // DYNAMIC_INTERNAL_HASH_RATE_COUNTER_H
