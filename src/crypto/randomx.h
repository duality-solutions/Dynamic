// Copyright (c) 2021 Duality Blockchain Solutions
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_CRYPTO_RANDOMX_H
#define BITCOIN_CRYPTO_RANDOMX_H

#include "sync.h"
#include "dbwrapper.h"

#include <map>
#include <cstdlib>
#include <uint256.h>

uint64_t GetHeight(const uint256& block_hash);

class RXEpochCache : public CDBWrapper {
private:
    mutable CCriticalSection cs;

    std::string phrase  = "cache";
    uint64_t time_period;
    uint32_t activation_wait;
    std::map<int64_t, uint256> cache;
public:
    RXEpochCache(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate)
    : CDBWrapper(GetDataDir() / "blocks" / "epoch", nCacheSize, fMemory, fWipe, obfuscate)
    {
        time_period = 2048;
        activation_wait = 64;
        if (!Exists(phrase)) {
            Write(phrase, *this);
        } else {
            Read(phrase, *this);
        }
    };

    ~RXEpochCache() = default;

    ADD_SERIALIZE_METHODS;
    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(cache);
        READWRITE(time_period);
        READWRITE(activation_wait);
    }

    bool AddEpoch(int64_t height, uint256 hash)
    {
        LOCK(cs);
        cache.insert({height, hash});
        return Update(phrase, *this);
    }

    bool RemoveEpoch(int64_t height, uint256 hash)
    {
        LOCK(cs);
        cache.erase(height);
        return Update(phrase, *this);
    }

    uint256 GetClosestEpoch(uint256 hash) const
    {
        LOCK(cs);
        int64_t height = GetHeight(hash);
        assert(cache.begin() != cache.end());
        auto low = cache.lower_bound(height);
        if (low == cache.begin()) { return low->second; }
        return std::abs(height - std::prev(low)->first) < std::abs(low->first - height) ?
               std::prev(low)->second : low->second;
    }

    std::map<int64_t, uint256> Get() const
    {
        return cache;
    }

    uint64_t GetPeriod() const
    {
        return time_period;
    }
};

uint256 RXHashFunction(const char* pbegin, const char* pend, const char* k_begin, const char* k_end);

extern RXEpochCache* epoch_cache;

#endif // BITCOIN_CRYPTO_RANDOMX_H