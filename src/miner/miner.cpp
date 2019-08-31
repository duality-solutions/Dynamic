// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "miner/miner.h"
#include "chainparams.h"
#include "consensus/consensus.h"
#include "consensus/validation.h"
#include "miner/internal/miners-controller.h"
#include "net.h"
#include "primitives/transaction.h"
#include "utilmoneystr.h"
#include "validation.h"
#include "validationinterface.h"

void InitMiners(const CChainParams& chainparams, CConnman& connman)
{
    if (!gMiners)
        gMiners.reset(new MinersController(chainparams, connman));
}

void StartMiners()
{
    assert(gMiners);
    gMiners->Start();
};

void ShutdownMiners()
{
    if (gMiners)
        gMiners->Shutdown();
};

void ShutdownCPUMiners()
{
    if (gMiners)
        gMiners->group_cpu().Shutdown();
};

void ShutdownGPUMiners()
{
#ifdef ENABLE_GPU
    if (gMiners)
        gMiners->group_gpu().Shutdown();
#endif // ENABLE_GPU
};

int64_t GetHashRate()
{
    if (gMiners)
        return gMiners->GetHashRate();
    return 0;
};

int64_t GetCPUHashRate()
{
    if (gMiners)
        return gMiners->group_cpu().GetHashRate();
    return 0;
};

int64_t GetGPUHashRate()
{
#ifdef ENABLE_GPU
    if (gMiners)
        return gMiners->group_gpu().GetHashRate();
#endif // ENABLE_GPU
    return 0;
};

void SetCPUMinerThreads(uint8_t target)
{
    assert(gMiners);
    gMiners->group_cpu().SetSize(target);
};

void SetGPUMinerThreads(uint8_t target)
{
#ifdef ENABLE_GPU
    assert(gMiners);
    gMiners->group_gpu().SetSize(target);
#endif // ENABLE_GPU
};

std::unique_ptr<MinersController> gMiners = {nullptr};
