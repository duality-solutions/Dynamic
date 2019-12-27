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

const CBlockIndex* GetLastBlockIndex(const CBlockIndex* pindex, bool fProofOfStake)
{
    while (pindex && pindex->pprev && (pindex->IsProofOfStake() != fProofOfStake))
        pindex = pindex->pprev;
    return pindex;
}

unsigned int GetNextStakeWorkRequired(const CBlockIndex* pindex, const Consensus::Params& params)
{
    const CBlockIndex* pindexLast = GetLastBlockIndex(pindex, true);
    if (!pindexLast)
        if (Params().NetworkIDString() == CBaseChainParams::REGTEST)
            return pindexLast->nBits;

    arith_uint256 targetLimit = UintToArith256(params.posLimit);

    const int64_t nTargetSpacing = params.nPosTargetSpacing;
    const int64_t nTargetTimespan = params.nTargetPosTimespan;

    int64_t nActualSpacing = 0;
    if (pindexLast->nHeight != 0)
        nActualSpacing = pindexLast->GetBlockTime() - pindexLast->pprev->GetBlockTime();
    if (nActualSpacing < 0)
        nActualSpacing = 1;
    if (nActualSpacing > nTargetSpacing*10)
        nActualSpacing = nTargetSpacing*10;

    // ppcoin: target change every block
    // ppcoin: retarget with exponential moving toward target spacing
    arith_uint256 bnNew;
    bnNew.SetCompact(pindexLast->nBits);

    int64_t nInterval = nTargetTimespan / nTargetSpacing;
    bnNew *= ((nInterval - 1) * nTargetSpacing + nActualSpacing + nActualSpacing);
    bnNew /= ((nInterval + 1) * nTargetSpacing);

    if (bnNew <= 0 || bnNew > targetLimit)
        bnNew = targetLimit;

    return bnNew.GetCompact();
}

unsigned int GetNextWorkRequired(const INDEX_TYPE pindexLast, const BLOCK_TYPE block, bool fProofOfStake, const Consensus::Params& params)
{
    assert(pindexLast != nullptr);

    if (fProofOfStake)
        return GetNextStakeWorkRequired(pindexLast, params);

    arith_uint256 targetLimit = UintToArith256(params.powLimit);

    if (pindexLast->nHeight + 1 <= params.nUpdateDiffAlgoHeight)
        return targetLimit.GetCompact(); // genesis block and first x (nUpdateDiffAlgoHeight) blocks use the default difficulty

    // Find the first block in the averaging interval
    const CBlockIndex* pindexFirst = pindexLast;
    arith_uint256 bnTot{0};
    int i = 0;
    int nPoWBlocks = 0;
    while (params.nPowAveragingWindow > nPoWBlocks) {
        if (pindexFirst->IsProofOfWork()) {
            arith_uint256 bnTmp;
            bnTmp.SetCompact(pindexFirst->nBits);
            bnTot += bnTmp;
            nPoWBlocks++;
        }
        pindexFirst = pindexFirst->pprev;
        if (pindexFirst->nHeight == 0)
            break;
        i++;
        if (i > 1000)
            break;
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
    const arith_uint256 ntargetLimit = UintToArith256(params.powLimit);
    arith_uint256 bnNew{bnAvg};
    bnNew /= params.AveragingWindowTimespan();
    bnNew *= nActualTimespan;

    if (bnNew > ntargetLimit)
        bnNew = ntargetLimit;

    /// debug print
    LogPrint("pow", "GetNextWorkRequired RETARGET\n");
    LogPrint("pow", "params.AveragingWindowTimespan() = %d    nActualTimespan = %d\n", params.AveragingWindowTimespan(), nActualTimespan);
    LogPrint("pow", "Current average: %08x  %s\n", bnAvg.GetCompact(), bnAvg.ToString());
    LogPrint("pow", "After:  %08x  %s\n", bnNew.GetCompact(), bnNew.ToString());

    return bnNew.GetCompact();
}

bool CheckProofOfWork(const uint256& hash, const unsigned int nBits, const Consensus::Params& params)
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
