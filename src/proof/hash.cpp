// Copyright (c) 2021 Duality Blockchain Solutions
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <proof/hash.h>

#include "primitives/block.h"
#include "streams.h"
#include "utilstrencodings.h"

#include <randomx.h>

namespace RandomX
{
class RXCache;
class RXDataset;
class RXVMachine;

class RXCache {
private:
    randomx_cache* ptr;
protected:
    randomx_cache* Raw() { return ptr; }

    friend class RXDataset;
    friend class RXVMachine;
public:
    template<typename T1>
    RXCache(randomx_flags flg, const T1 pbegin, const T1 pend)
    {
        randomx_init_cache(ptr, pbegin, pend - pbegin);
    }
    ~RXCache()
    {
        randomx_release_cache(ptr);
    }
};

class RXDataset {
private:
    randomx_dataset* ptr = nullptr;
    unsigned long ds_ctr = randomx_dataset_item_count();

protected:
    randomx_dataset* Raw() { return ptr; }

    friend class RXCache;
    friend class RXVMachine;
public:
    RXDataset(randomx_flags flg, RXCache ch)
    {
        ptr = randomx_alloc_dataset(flg);
        randomx_init_dataset(ptr, ch.Raw(), 0, ds_ctr / 2);
        randomx_init_dataset(ptr, ch.Raw(), ds_ctr / 2, ds_ctr - ds_ctr / 2);
    }

    ~RXDataset()
    {
        randomx_release_dataset(ptr);
    }
};

class RXVMachine {
private:
    uint256 hashResult;
    randomx_vm* ptr = nullptr;
protected:
    randomx_vm* Raw() { return ptr; }

    friend class RXCache;
    friend class RXDataset;
public:
    RXVMachine(randomx_flags flg, RXDataset ds)
    {
        ptr = randomx_create_vm(flg, nullptr, ds.Raw());
    }
    ~RXVMachine()
    {
        randomx_destroy_vm(ptr);
    }

    template<typename T1>
    uint256 GetHash(const T1 pbegin, const T1 pend)
    {
        randomx_calculate_hash(ptr, pbegin, pend - pbegin, (uint8_t*)&hashResult);
        return hashResult;
    }
};
} // namespace RandomX

uint256 RXHashFunction(const char* pbegin, const char* pend, const unsigned char* k_begin, const unsigned char* k_end)
{
    using namespace RandomX;

    randomx_flags flags = randomx_get_flags();
    flags |= RANDOMX_FLAG_LARGE_PAGES;
    flags |= RANDOMX_FLAG_FULL_MEM;

    RXCache ch(flags, k_begin, k_end);
    RXDataset ds(flags, ch);
    RXVMachine vm(flags, ds);
    return vm.GetHash(pbegin, pend);
}
