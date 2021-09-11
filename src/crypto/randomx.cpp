// Copyright (c) 2021 Duality Blockchain Solutions
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <crypto/randomx.h>

#include "sync.h"
#include "primitives/block.h"
#include "streams.h"
#include "utilstrencodings.h"

#include <thread>
#include <randomx.h>

namespace RandomX
{
class RXBuilder {
private:
    uint32_t dsic;
    uint256 hashResult;
    randomx_flags flags;

    randomx_vm *vm;
    randomx_cache *cache;
    randomx_dataset *dataset;

public:
    template<typename T1>
    RXBuilder(const T1 pbegin, const T1 pend, bool mining = false) {
        flags = randomx_get_flags();

        cache = randomx_alloc_cache(flags);
        assert (cache != nullptr);
        randomx_init_cache(cache, pbegin, pend - pbegin);

        dataset = randomx_alloc_dataset(flags);
        dsic = randomx_dataset_item_count();
        assert (dataset != nullptr);

        randomx_init_dataset(dataset, cache, 0, dsic / 2);
        randomx_init_dataset(dataset, cache, dsic / 2, dsic - dsic / 2);
        randomx_release_cache(cache);

        vm = randomx_create_vm(flags, cache, dataset);
        assert(vm != nullptr);
    }

    template<typename T1>
    uint256 GetHash(const T1 pbegin, const T1 pend)
    {
        randomx_calculate_hash(vm, pbegin, pend - pbegin, (uint8_t*)&hashResult);
        return hashResult;
    }

    ~RXBuilder()
    {
        randomx_destroy_vm(vm);
        randomx_release_dataset(dataset);
    }
};
} // namespace RandomX

uint256 RXHashFunction(const char* pbegin, const char* pend, const char* k_begin, const char* k_end)
{
    RandomX::RXBuilder bl(k_begin, k_end);
    return bl.GetHash(pbegin, pend);
}
