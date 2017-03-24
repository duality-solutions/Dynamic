// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_MINER_H
#define DYNAMIC_MINER_H

#include "primitives/block.h"

#include <stdint.h>
#include <cstddef>

class CBlockIndex;
class CChainParams;
class CReserveKey;
class CScript;
class CWallet;

namespace Consensus { struct Params; };

static const bool DEFAULT_GENERATE = false;
static const int DEFAULT_GENERATE_THREADS = 1;

static const bool DEFAULT_PRINTPRIORITY = false;

struct CBlockTemplate
{
    CBlock block;
    std::vector<CAmount> vTxFees;
    std::vector<int64_t> vTxSigOps;
};

/** ByteReverse Function used by GetWork */ // TODO: Shift to util
uint32_t ByteReverse(uint32_t value);
/** Do mining precalculation */
void FormatHashBuffers(CBlock* pblock, char* pmidstate, char* pdata, char* phash1);
/** Check mined block */
bool CheckWork(const CChainParams& chainparams, CBlock* pblock, CWallet& wallet, CReserveKey& reservekey);
/** Base sha256 mining transform */
void SHA256Transform(void* pstate, void* pinput, const void* pinit);

/** Run the miner threads */
void GenerateDynamics(bool fGenerate, int nThreads, const CChainParams& chainparams);
/** Generate a new block, without valid proof-of-work */
std::unique_ptr<CBlockTemplate> CreateNewBlock(const CChainParams& chainparams, const CScript& scriptPubKeyIn);
/** Modify the extranonce in a block */
void IncrementExtraNonce(CBlock* pblock, const CBlockIndex* pindexPrev, unsigned int& nExtraNonce);
int64_t UpdateTime(CBlockHeader* pblock, const Consensus::Params& consensusParams, const CBlockIndex* pindexPrev);

extern double dHashesPerSec;
extern int64_t nHPSTimerStart;

#endif // DYNAMIC_MINER_H
