// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2018 The Dash Core Developers
// Copyright (c) 2009-2018 The Bitcoin Developers
// Copyright (c) 2009-2018 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_MINER_UTIL_H
#define DYNAMIC_MINER_UTIL_H

#include <atomic>
#include <memory>

// #include "chain/chain.h"
#include "primitives/block.h"

class CBlockIndex;
class CChainParams;
class CConnman;
class CReserveKey;
class CScript;
class CWallet;

namespace Consensus
{
struct Params;
};

static const bool DEFAULT_GENERATE = false;
static const int DEFAULT_GENERATE_THREADS_CPU = 0;
static const int DEFAULT_GENERATE_THREADS_GPU = 0;

static const bool DEFAULT_PRINTPRIORITY = false;

struct CBlockTemplate {
    CBlock block;
    std::vector<CAmount> vTxFees;
    std::vector<int64_t> vTxSigOps;
    CTxOut txoutDynode;                 // dynode payment
    std::vector<CTxOut> voutSuperblock; // dynode payment
};

#ifdef ENABLE_WALLET
/** Check mined block */
bool CheckWork(const CChainParams& chainParams, CBlock* pblock, CWallet& wallet, CReserveKey& reservekey, CConnman* connMan);
#endif // ENABLE_WALLET

/** Generate a new block, without valid proof-of-work */
std::unique_ptr<CBlockTemplate> CreateNewBlock(const CChainParams& chainParams, const CScript& scriptPubKeyIn);
/** Called by a miner when new block was found. */
bool ProcessBlockFound(const CBlock& block, const CChainParams& chainparams);

/** Modify the extranonce in a block */
void IncrementExtraNonce(CBlock& pblock, const CBlockIndex* pindexPrev, unsigned int& nExtraNonce);
int64_t UpdateTime(CBlockHeader& pblock, const Consensus::Params& consensusParams, const CBlockIndex* pindexPrev);

#endif // DYNAMIC_MINER_UTIL_H
