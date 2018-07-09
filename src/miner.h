// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2018 The Dash Core Developers
// Copyright (c) 2009-2018 The Bitcoin Developers
// Copyright (c) 2009-2018 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_MINER_H
#define DYNAMIC_MINER_H

#include "primitives/block.h"

#include <stdint.h>
#include <memory>
#include <cstddef>

class CBlockIndex;
class CChainParams;
class CConnman;
class CReserveKey;
class CScript;
class CWallet;

namespace Consensus { struct Params; };

static const bool DEFAULT_GENERATE = false;
static const int DEFAULT_GENERATE_THREADS_CPU = -1;
// TODO(crackcom): Set to -1 after autotune tests
static const int DEFAULT_GENERATE_THREADS_GPU = -1;

static const bool DEFAULT_PRINTPRIORITY = false;

struct CBlockTemplate
{
    CBlock block;
    std::vector<CAmount> vTxFees;
    std::vector<int64_t> vTxSigOps;
};

#ifdef ENABLE_WALLET
/** Check mined block */
bool CheckWork(const CChainParams& chainparams, CBlock* pblock, CWallet& wallet, CReserveKey& reservekey, CConnman* connman);
#endif //ENABLE_WALLET
/** Run the miner threads */
void GenerateDynamics(int nCPUThreads, int nGPUThreads, const CChainParams& chainparams, CConnman& connman, bool fAutotune = false);
/** Shuts down all miner threads */
void ShutdownMiners();
/** Shuts down all CPU miner threads */
void ShutdownCPUMiners();
/** Shuts down all GPU miner threads */
void ShutdownGPUMiners();
/** Generate a new block, without valid proof-of-work */
std::unique_ptr<CBlockTemplate> CreateNewBlock(const CChainParams& chainparams, const CScript& scriptPubKeyIn);
/** Modify the extranonce in a block */
void IncrementExtraNonce(CBlock* pblock, const CBlockIndex* pindexPrev, unsigned int& nExtraNonce);
int64_t UpdateTime(CBlockHeader* pblock, const Consensus::Params& consensusParams, const CBlockIndex* pindexPrev);
/** Gets combined hash rate of GPU and CPU */
int64_t GetHashRate();
/** Gets hash rate of CPU */
int64_t GetCPUHashRate();
/** Gets hash rate of GPU */
int64_t GetGPUHashRate();

extern double dCPUHashesPerSec;
extern double dGPUHashesPerSec;
extern int64_t nHPSTimerStart;

#endif // DYNAMIC_MINER_H
