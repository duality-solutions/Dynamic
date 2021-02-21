// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_POW_H
#define DYNAMIC_POW_H

#include <arith_uint256.h>
#include <chain.h>
#include <consensus/params.h>
#include <stdint.h>
#include <sync.h>
#include <util.h>

class arith_uint256;
class CBlockHeader;
class CBlockIndex;
class uint256;

#define PERCENT_FACTOR 100

#define BLOCK_TYPE CBlockHeader&
#define BLOCK_TIME(block) block->nTime
#define INDEX_TYPE CBlockIndex*
#define INDEX_HEIGHT(block) block->nHeight
#define INDEX_TIME(block) block->GetBlockTime()
#define INDEX_PREV(block) block->pprev
#define INDEX_TARGET(block) block->nBits
#define DIFF_SWITCHOVER(TEST, MAIN) (GetBoolArg("-testnet", false) ? TEST : MAIN)
#define DIFF_ABS std::abs
#define SET_COMPACT(EXPANDED, COMPACT) EXPANDED.SetCompact(COMPACT)
#define GET_COMPACT(EXPANDED) EXPANDED.GetCompact()
#define BIGINT_MULTIPLY(x, y) x* y
#define BIGINT_DIVIDE(x, y) x / y
#define BIGINT_GREATER_THAN(x, y) (x > y)

const CBlockIndex* GetLastBlockIndex(const CBlockIndex* pindex);

bool CheckForkIsTrue(const CBlockIndex* pindexLast, bool fTableFlip = false);

unsigned int LegacyRetargetBlock(const CBlockIndex* pindexLast, const CBlockHeader* pblock, const Consensus::Params&);
unsigned int GetNextWorkRequired(const INDEX_TYPE pindexLast, const BLOCK_TYPE block, const Consensus::Params&);

/** Check whether a block hash satisfies the proof-of-work requirement specified by nBits */
bool CheckProofOfWork(uint256 hash, unsigned int nBits, const Consensus::Params&);

#endif // DYNAMIC_POW_H
