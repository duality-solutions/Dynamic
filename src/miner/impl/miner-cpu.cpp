
// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "miner/impl/miner-cpu.h"
#include "primitives/block.h"


CPUMiner::CPUMiner(MinerContextRef ctx, std::size_t device_index)
    : MinerBase(ctx, device_index){};

int64_t CPUMiner::TryMineBlock(CBlock& block)
{
    int64_t hashes_done = 0;
    while (true) {
        uint256 hash = block.GetHash();
        if (UintToArith256(hash) <= _hash_target) {
            this->ProcessFoundSolution(block, hash);
            break;
        }
        block.nNonce += 1;
        hashes_done += 1;
        if ((block.nNonce & 0xFF) == 0)
            break;
    }
    return hashes_done;
}
