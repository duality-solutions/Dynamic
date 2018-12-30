// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_INTERNAL_MINERS_GROUP_H
#define DYNAMIC_INTERNAL_MINERS_GROUP_H

#include "miner/internal/miner-context.h"
#include "miner/internal/thread-group.h"


/**
 * Miners group threads controller.
 */
template <class T>
class MinersThreadGroup : public ThreadGroup<T, MinerContextRef>
{
public:
    MinersThreadGroup(MinerContextRef ctx)
        : ThreadGroup<T, MinerContextRef>(ctx){};

    // Shuts down all miner threads
    void Shutdown()
    {
        // Shutdown all threads
        SetSize(0);
        // It's not updated and instead of reading
        // system time and comparing with last update
        // it is just reset when all threads are shut
        this->_ctx->counter->Reset();
    };

    // Sets amount of threads
    void SetSize(uint8_t size)
    {
        // Set thread group size
        ThreadGroup<T, MinerContextRef>::SetSize(size);
        // Reset hash rate counter
        if (size == 0) {
            this->_ctx->counter->Reset();
        }
    };

    // Gets hash rate of all threads in the group
    int64_t GetHashRate() const { return *this->_ctx->counter; };
};


#endif // DYNAMIC_INTERNAL_MINERS_GROUP_H
