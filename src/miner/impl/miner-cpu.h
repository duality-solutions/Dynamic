// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_MINER_IMPL_CPU_H
#define DYNAMIC_MINER_IMPL_CPU_H

#include "miner/internal/miner-base.h"


/**
 * Dynamic CPU miner.
 */
class CPUMiner final : public MinerBase
{
public:
    CPUMiner(MinerContextRef ctx, std::size_t device_index);
    virtual ~CPUMiner() = default;

    static unsigned int TotalDevices()
    {
        // Currently it's one until I'll solve
        // physical CPU vs CPU core count vs vCPU count
        // and get it fixed on Windows, OSx and Linux
        // Until then we will just use OS scheduler
        // (not a priority at all; here just for GPU)
        return 1;
    };

    virtual const char* DeviceName() override { return "CPU"; };

protected:
    virtual int64_t TryMineBlock(CBlock& block) override;
};

#endif // DYNAMIC_MINER_IMPL_CPU_H
