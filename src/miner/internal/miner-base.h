// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_INTERNAL_MINER_BASE_H
#define DYNAMIC_INTERNAL_MINER_BASE_H

#include "arith_uint256.h"
#include "miner/internal/miner-context.h"

#include <cstddef>

class CBlock;
class CReserveScript;

/**
 * Base miner class for CPU and GPU miner.
 */
class MinerBase
{
public:
    MinerBase(MinerContextRef ctx, std::size_t device_index);
    virtual ~MinerBase() = default;

    // Starts miner loop
    void Loop();

    // Starts miner loop
    void operator()() { Loop(); };

    // Returns miner device name
    virtual const char* DeviceName() = 0;

protected:
    // Processes a new found solution
    void ProcessFoundSolution(const CBlock& block, const uint256& hash);

    // tries to mine a block
    virtual int64_t TryMineBlock(CBlock& block) = 0;

    // Solution must be lower or equal to
    arith_uint256 _hash_target = 0;

    // Miner context
    MinerContextRef _ctx;

private:
    // Miner device index
    std::size_t _device_index;

    // Extra block nonce
    unsigned int _extra_nonce = 0;

    // Miner coinbase script
    // Includes wallet payout key
    std::shared_ptr<CReserveScript> _coinbase_script{nullptr};
};

#endif // DYNAMIC_INTERNAL_MINER_BASE_H
