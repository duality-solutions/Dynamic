// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Copyright (c) 2016-2017 The ZCash Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "pow.h"

#include "arith_uint256.h"
#include "primitives/block.h"
#include "chain.h"
#include "chainparams.h"
#include "checkforks.h"
#include "main.h"
#include "uint256.h"
#include "util.h"

#include <algorithm>

const CBlockIndex* GetLastBlockIndex(const CBlockIndex* pindex)
{
    while (pindex && pindex->pprev)
        pindex = pindex->pprev;
    return pindex;
}

unsigned int GetNextWorkRequired(const INDEX_TYPE pindexLast, const BLOCK_TYPE block, const Consensus::Params& params)
{
    if (CheckForkIsTrue(DELTA_RETARGET, pindexLast)) {

        unsigned int initalBlock = params.nUpdateDiffAlgoHeight;
        int64_t nRetargetTimespan = params.nPowTargetSpacing;
        const unsigned int nProofOfWorkLimit = UintToArith256(params.powLimit).GetCompact();

        const unsigned int nLastBlock = 1;
        const unsigned int nShortFrame = 3;
        const unsigned int nMiddleFrame = 24;
        const unsigned int nLongFrame = 576;

        const int64_t nLBWeight = 64;
        const int64_t nShortWeight = 8;
        int64_t nMiddleWeight = 2;
        int64_t nLongWeight = 1;

        const int64_t nLBMinGap = nRetargetTimespan / 6;
        const int64_t nLBMaxGap = nRetargetTimespan * 6;

        const int64_t nQBFrame = nShortFrame + 1;
        const int64_t nQBMinGap = (nRetargetTimespan * PERCENT_FACTOR / 120) * nQBFrame;

        const int64_t nBadTimeLimit = 0;
        const int64_t nBadTimeReplace = nRetargetTimespan / 10;

        const int64_t nLowTimeLimit = nRetargetTimespan * 90 / PERCENT_FACTOR;
        const int64_t nFloorTimeLimit = nRetargetTimespan * 65 / PERCENT_FACTOR;

        const int64_t nDrift = 1;
        int64_t nLongTimeLimit = ((6 * nDrift)) * 60;
        int64_t nLongTimeStep = nDrift * 60;

        unsigned int nMinimumAdjustLimit = (unsigned int)nRetargetTimespan * 75 / PERCENT_FACTOR;
        unsigned int nMaximumAdjustLimit = (unsigned int)nRetargetTimespan * 150 / PERCENT_FACTOR;

        int64_t nDeltaTimespan = 0;
        int64_t nLBTimespan = 0;
        int64_t nShortTimespan = 0;
        int64_t nMiddleTimespan = 0;
        int64_t nLongTimespan = 0;
        int64_t nQBTimespan = 0;

        int64_t nWeightedSum = 0;
        int64_t nWeightedDiv = 0;
        int64_t nWeightedTimespan = 0;

        const INDEX_TYPE pindexFirst = pindexLast; // multi algo - last block is selected on a per algo basis.

        if (pindexLast == NULL)
            return nProofOfWorkLimit;

        if (INDEX_HEIGHT(pindexLast) <= nQBFrame)
            return nProofOfWorkLimit;

        pindexFirst = INDEX_PREV(pindexLast);
        nLBTimespan = INDEX_TIME(pindexLast) - INDEX_TIME(pindexFirst);

        if (nLBTimespan > nBadTimeLimit && nLBTimespan < nLBMinGap)
            nLBTimespan = nLBTimespan * 50 / PERCENT_FACTOR;

        if (nLBTimespan <= nBadTimeLimit)
            nLBTimespan = nBadTimeReplace;

        if (nLBTimespan > nLBMaxGap)
            nLBTimespan = nLBTimespan * 150 / PERCENT_FACTOR;

        pindexFirst = pindexLast;
        for (unsigned int i = 1; pindexFirst != NULL && i <= nQBFrame; i++) {
            nDeltaTimespan = INDEX_TIME(pindexFirst) - INDEX_TIME(INDEX_PREV(pindexFirst));

            if (nDeltaTimespan <= nBadTimeLimit)
                nDeltaTimespan = nBadTimeReplace;

            if (i <= nShortFrame)
                nShortTimespan += nDeltaTimespan;
            nQBTimespan += nDeltaTimespan;
            pindexFirst = INDEX_PREV(pindexFirst);
        }

        if (INDEX_HEIGHT(pindexLast) - initalBlock <= nMiddleFrame) {
            nMiddleWeight = nMiddleTimespan = 0;
        }
        else {
            pindexFirst = pindexLast;
            for (unsigned int i = 1; pindexFirst != NULL && i <= nMiddleFrame; i++) {
                nDeltaTimespan = INDEX_TIME(pindexFirst) - INDEX_TIME(INDEX_PREV(pindexFirst));

                if (nDeltaTimespan <= nBadTimeLimit)
                    nDeltaTimespan = nBadTimeReplace;

                nMiddleTimespan += nDeltaTimespan;
                pindexFirst = INDEX_PREV(pindexFirst);
            }
        }

        if (INDEX_HEIGHT(pindexLast) - initalBlock <= nLongFrame) {
            nLongWeight = nLongTimespan = 0;
        }
        else {
            pindexFirst = pindexLast;
            for (unsigned int i = 1; pindexFirst != NULL && i <= nLongFrame; i++)
                pindexFirst = INDEX_PREV(pindexFirst);

            nLongTimespan = INDEX_TIME(pindexLast) - INDEX_TIME(pindexFirst);
        }

        if ((nQBTimespan > nBadTimeLimit) && (nQBTimespan < nQBMinGap) && (nLBTimespan < nRetargetTimespan * 40 / PERCENT_FACTOR)) {
            nMiddleWeight = nMiddleTimespan = nLongWeight = nLongTimespan = 0;
        }

        nWeightedSum = (nLBTimespan * nLBWeight) + (nShortTimespan * nShortWeight);
        nWeightedSum += (nMiddleTimespan * nMiddleWeight) + (nLongTimespan * nLongWeight);
        nWeightedDiv = (nLastBlock * nLBWeight) + (nShortFrame * nShortWeight);
        nWeightedDiv += (nMiddleFrame * nMiddleWeight) + (nLongFrame * nLongWeight);
        nWeightedTimespan = nWeightedSum / nWeightedDiv;

        if (DIFF_ABS(nLBTimespan - nRetargetTimespan) < nRetargetTimespan * 20 / PERCENT_FACTOR) {
            nMinimumAdjustLimit = (unsigned int)nRetargetTimespan * 90 / PERCENT_FACTOR;
            nMaximumAdjustLimit = (unsigned int)nRetargetTimespan * 110 / PERCENT_FACTOR;
        } else if (DIFF_ABS(nLBTimespan - nRetargetTimespan) < nRetargetTimespan * 30 / PERCENT_FACTOR) {
            nMinimumAdjustLimit = (unsigned int)nRetargetTimespan * 80 / PERCENT_FACTOR;
            nMaximumAdjustLimit = (unsigned int)nRetargetTimespan * 120 / PERCENT_FACTOR;
        }

        if (nWeightedTimespan < nMinimumAdjustLimit)
            nWeightedTimespan = nMinimumAdjustLimit;

        if (nWeightedTimespan > nMaximumAdjustLimit)
            nWeightedTimespan = nMaximumAdjustLimit;

        arith_uint256 bnNew;
        SET_COMPACT(bnNew, INDEX_TARGET(pindexLast));
        bnNew = BIGINT_MULTIPLY(bnNew, arith_uint256(nWeightedTimespan));
        bnNew = BIGINT_DIVIDE(bnNew, arith_uint256(nRetargetTimespan));

        nLBTimespan = INDEX_TIME(pindexLast) - INDEX_TIME(INDEX_PREV(pindexLast));
        arith_uint256 bnComp;
        SET_COMPACT(bnComp, INDEX_TARGET(pindexLast));
        if (nLBTimespan > 0 && nLBTimespan < nLowTimeLimit && BIGINT_GREATER_THAN(bnNew, bnComp)) {
            if (nLBTimespan < nFloorTimeLimit) {
                SET_COMPACT(bnNew, INDEX_TARGET(pindexLast));
                bnNew = BIGINT_MULTIPLY(bnNew, arith_uint256(95));
                bnNew = BIGINT_DIVIDE(bnNew, arith_uint256(PERCENT_FACTOR));
            }
            else {
                SET_COMPACT(bnNew, INDEX_TARGET(pindexLast));
            }
        }

        if ((BLOCK_TIME(block) - INDEX_TIME(pindexLast)) > nLongTimeLimit) {
            int64_t nNumMissedSteps = ((BLOCK_TIME(block) - INDEX_TIME(pindexLast) - nLongTimeLimit) / nLongTimeStep) + 1;
            for (int i = 0; i < nNumMissedSteps; ++i) {
                bnNew = BIGINT_MULTIPLY(bnNew, arith_uint256(110));
                bnNew = BIGINT_DIVIDE(bnNew, arith_uint256(PERCENT_FACTOR));
            }
        }

        SET_COMPACT(bnComp, nProofOfWorkLimit);
        if (BIGINT_GREATER_THAN(bnNew, bnComp))
            SET_COMPACT(bnNew, nProofOfWorkLimit);

        return GET_COMPACT(bnNew);

    } else { return LegacyRetargetBlock(pindexLast, block, params); }
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

arith_uint256 GetBlockProof(const CBlockIndex& block)
{
    arith_uint256 bnTarget;
    bool fNegative;
    bool fOverflow;
    bnTarget.SetCompact(block.nBits, &fNegative, &fOverflow);
    if (fNegative || fOverflow || bnTarget == 0)
        return 0;
    // We need to compute 2**256 / (bnTarget+1), but we can't represent 2**256
    // as it's too large for a arith_uint256. However, as 2**256 is at least as large
    // as bnTarget+1, it is equal to ((2**256 - bnTarget - 1) / (bnTarget+1)) + 1,
    // or ~bnTarget / (nTarget+1) + 1.
    return (~bnTarget / (bnTarget + 1)) + 1;
}

int64_t GetBlockProofEquivalentTime(const CBlockIndex& to, const CBlockIndex& from, const CBlockIndex& tip, const Consensus::Params& params)
{
    arith_uint256 r;
    int sign = 1;
    if (to.nChainWork > from.nChainWork) {
        r = to.nChainWork - from.nChainWork;
    } else {
        r = from.nChainWork - to.nChainWork;
        sign = -1;
    }
    r = r * arith_uint256(params.nPowTargetSpacing) / GetBlockProof(tip);
    if (r.bits() > 63) {
        return sign * std::numeric_limits<int64_t>::max();
    }
    return sign * r.GetLow64();
}
