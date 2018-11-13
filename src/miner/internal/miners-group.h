// Copyright (c) 2018 Duality Blockchain Solutions Developers
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
        MinersThreadGroup<T>::Shutdown();
        // It's not updated and instead of reading
        // system time and comparing with last update
        // it is just reset when all threads are shut
        this->_ctx->counter->Reset();
    };

    // Sets amount of threads
    void SetNumThreads(uint8_t target)
    {
        // Set amount of target threads
        MinersThreadGroup<T>::SetNumThreads(target);
        // Reset hash rate counter
        this->_ctx->counter->Reset();
    };

    // Gets hash rate of all threads in the group
    int64_t GetHashRate() const { return *this->_ctx->counter; };
};


#endif // DYNAMIC_INTERNAL_MINERS_GROUP_H
