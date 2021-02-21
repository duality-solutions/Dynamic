// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
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

unsigned int GetNextWorkRequired(const INDEX_TYPE pindexLast, const BLOCK_TYPE block, const Consensus::Params& params)
{
    assert(pindexLast != nullptr);
    if (pindexLast->nHeight + 1 <= params.nUpdateDiffAlgoHeight)
        return UintToArith256(params.powLimit).GetCompact(); // genesis block and first x (nUpdateDiffAlgoHeight) blocks use the default difficulty

    // Find the first block in the averaging interval
    const CBlockIndex* pindexFirst = pindexLast;
    arith_uint256 bnTot{0};
    for (int i = 0; pindexFirst && i < params.nPowAveragingWindow; i++) {
        arith_uint256 bnTmp;
        bnTmp.SetCompact(pindexFirst->nBits);
        bnTot += bnTmp;
        pindexFirst = pindexFirst->pprev;
    }
    arith_uint256 bnAvg{bnTot / params.nPowAveragingWindow};

    // Use medians to prevent time-warp attacks
    int64_t nLastBlockTime = pindexLast->GetMedianTimePast();
    int64_t nFirstBlockTime = pindexFirst->GetMedianTimePast();
    int64_t nActualTimespan = nLastBlockTime - nFirstBlockTime;
    LogPrint("pow", "  nActualTimespan = %d  before dampening\n", nActualTimespan);
    nActualTimespan = params.AveragingWindowTimespan() + (nActualTimespan - params.AveragingWindowTimespan()) / 4;
    LogPrint("pow", "  nActualTimespan = %d  before bounds\n", nActualTimespan);

    if (nActualTimespan < params.MinActualTimespan())
        nActualTimespan = params.MinActualTimespan();
    if (nActualTimespan > params.MaxActualTimespan())
        nActualTimespan = params.MaxActualTimespan();

    // Retarget
    const arith_uint256 bnPowLimit = UintToArith256(params.powLimit);
    arith_uint256 bnNew{bnAvg};
    bnNew /= params.AveragingWindowTimespan();
    bnNew *= nActualTimespan;

    if (bnNew > bnPowLimit)
        bnNew = bnPowLimit;

    /// debug print
    LogPrint("pow", "GetNextWorkRequired RETARGET\n");
    LogPrint("pow", "params.AveragingWindowTimespan() = %d    nActualTimespan = %d\n", params.AveragingWindowTimespan(), nActualTimespan);
    LogPrint("pow", "Current average: %08x  %s\n", bnAvg.GetCompact(), bnAvg.ToString());
    LogPrint("pow", "After:  %08x  %s\n", bnNew.GetCompact(), bnNew.ToString());

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
