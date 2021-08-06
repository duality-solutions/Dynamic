// Copyright (c) 2016-2021 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2021 The Dash Core Developers
// Copyright (c) 2009-2021 The Bitcoin Developers
// Copyright (c) 2009-2021 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "pow.h"

#include "arith_uint256.h"
#include "chain.h"
#include "chainparams.h"
#include "primitives/block.h"
#include "uint256.h"
#include "util.h"
#include "validation.h"

#include <algorithm>


const CBlockIndex* GetLastBlockIndex(const CBlockIndex* pindex)
{
    while (pindex && pindex->pprev)
        pindex = pindex->pprev;
    return pindex;
}

unsigned int GetNextWorkRequired(const CBlockIndex* pindexLast, const CBlockHeader& block, const Consensus::Params& params) {
    assert(pindexLast != nullptr);

    if (pindexLast->nHeight + 1 <= params.nUpdateDiffAlgoHeight)
        return UintToArith256(params.powLimit).GetCompact(); // genesis block and first x (nUpdateDiffAlgoHeight) blocks use the default difficulty
    
    if (pindexLast->nHeight + 1 <= Params().DifficultySwitchBlock())
        return DigiShield(pindexLast, 5, 5 * (2 * 64), ((5 * (2 * 64)) * (100 - 32)) / 100, ((5 * (2 * 64)) * (100 + 48)) / 100, params);

    return DigiShield(pindexLast, params.nPowAveragingWindow, params.AveragingWindowTimespan(), params.MinActualTimespan(), params.MaxActualTimespan(), params);
}

unsigned int DigiShield(const CBlockIndex* pindexLast, const int64_t AveragingWindow, const int64_t AveragingWindowTimespan, const int64_t MinActualTimespan, const int64_t MaxActualTimespan, const Consensus::Params& params)
{
    // Find the first block in the averaging interval
    const CBlockIndex* pindexFirst = pindexLast;
    arith_uint256 bnTot{0};

    for (int i = 0; pindexFirst && i < AveragingWindow; i++) {
        arith_uint256 bnTmp;
        bnTmp.SetCompact(pindexFirst->nBits);
        bnTot += bnTmp;
        pindexFirst = pindexFirst->pprev;
    }

    arith_uint256 bnAvg { bnTot / AveragingWindow };

    // Use medians to prevent time-warp attacks
    int64_t nLastBlockTime = pindexLast->GetMedianTimePast();
    int64_t nFirstBlockTime = pindexFirst->GetMedianTimePast();
    int64_t nActualTimespan = nLastBlockTime - nFirstBlockTime;
    nActualTimespan = AveragingWindowTimespan + (nActualTimespan - AveragingWindowTimespan) / 4;

    if (nActualTimespan < MinActualTimespan)
        nActualTimespan = MinActualTimespan;
    if (nActualTimespan > MaxActualTimespan)
        nActualTimespan = MaxActualTimespan;

    // Retarget
    const arith_uint256 bnPowLimit = UintToArith256(params.powLimit);
    arith_uint256 bnNew{bnAvg};
    bnNew /= AveragingWindowTimespan;
    bnNew *= nActualTimespan;

    if (bnNew > bnPowLimit)
        bnNew = bnPowLimit;

    return bnNew.GetCompact();
}


bool CheckProofOfWork(uint256 hash, unsigned int nBits, const Consensus::Params& params)
{
    bool fNegative;
    bool fOverflow;
    arith_uint256 bnTarget;

    bnTarget.SetCompact(nBits, &fNegative, &fOverflow);

    // Check range
    if (fNegative || bnTarget == 0 || fOverflow || bnTarget > UintToArith256(params.powLimit))
        return error("CheckProofOfWork(): nBits below minimum work");

    // Check proof of work matches claimed amount
    if (UintToArith256(hash) > bnTarget)
        return error("CheckProofOfWork(): hash doesn't match nBits");

    return true;
}
