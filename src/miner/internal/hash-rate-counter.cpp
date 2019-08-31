
// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "miner/internal/hash-rate-counter.h"
#include "utiltime.h"


void HashRateCounter::Increment(int64_t amount)
{
    // Set start time if not set and return
    if (_timer_start == 0) {
        Reset();
        return;
    }
    // Increment hashes done
    _count += amount;
    if (_parent) {
        _parent->Increment(amount);
    }
    // Ignore until at least 4 seconds passed
    if (GetTimeMillis() - _timer_start < 4000) {
        return;
    }
    // Set count per second
    _count_per_sec = 1000.0 * _count / (GetTimeMillis() - _timer_start);
    // Reset timer and count
    _count = 0;
    _timer_start = GetTimeMillis();
}

void HashRateCounter::Reset()
{
    _count = 0;
    _count_per_sec = 0;
    _timer_start = GetTimeMillis();
}
