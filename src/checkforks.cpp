// Copyright (c) 2016-2017 Plaxton/Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "checkforks.h"

/**
 * Fork Logic
 * 
 * This will return a true or false bool depending if certain fork parameters are met, if it is 
 * then it will return true, we have to avoid goof-ups so if there is an incorrect identifier, then
 * it will give a runtime error.
 **/

bool CheckForkIsTrue(ForkID identifier, const CBlockIndex* pindexLast, bool fTableFlip) {
	
	if(fTableFlip)
		return true;

	int32_t networkHeight = (pindexLast->nHeight + 1);

	int placeholderIntX = 2;
	int placeholderIntY = 3;

	const Consensus::Params& consensusParams = Params().GetConsensus();

	if (identifier == DELTA_RETARGET || identifier == FORK_SLOT_1 || identifier == FORK_SLOT_2 || identifier == FORK_SLOT_3 || identifier == FORK_SLOT_4) {  // Check if we are handling a valid fork

		if(networkHeight > consensusParams.nUpdateDiffAlgoHeight && chainActive.Height() > consensusParams.nUpdateDiffAlgoHeight && identifier == DELTA_RETARGET) { return true;}
		else if (placeholderIntX == placeholderIntY && identifier == FORK_SLOT_1) { return true; } // Empty Forking Slot I
		else if (placeholderIntX == placeholderIntY && identifier == FORK_SLOT_2) { return true; } // Empty Forking Slot II
		else if (placeholderIntX == placeholderIntY && identifier == FORK_SLOT_3) { return true; } // Empty Forking Slot III
		else if (placeholderIntX == placeholderIntY && identifier == FORK_SLOT_4) { return true; } // Empty Forking Slot IV
		else { return false; }

	} else { throw std::runtime_error(strprintf("%s: Unknown Fork Verification Cause! %s.", __func__, identifier)); }
}

/**
 * Legacy Code that has been replaced by its current counterpart
 **/

unsigned int LegacyRetargetBlock(const CBlockIndex* pindexLast, const CBlockHeader *pblock, const Consensus::Params& params)
{
	unsigned int nProofOfWorkLimit = UintToArith256(params.powLimit).GetCompact();

	if (pindexLast == NULL)
		return nProofOfWorkLimit; // genesis block

	const CBlockIndex* pindexPrev = GetLastBlockIndex(pindexLast);
	if (pindexPrev->pprev == NULL)
		return nProofOfWorkLimit; // first block
		
	const CBlockIndex* pindexPrevPrev = GetLastBlockIndex(pindexPrev->pprev);
	if (pindexPrevPrev->pprev == NULL)
		return nProofOfWorkLimit; // second block
	
	int64_t nTargetSpacing = params.nPowTargetTimespan;
	int64_t nActualSpacing = pindexPrev->GetBlockTime() - pindexPrevPrev->GetBlockTime();

	if (nActualSpacing < 0) {
		nActualSpacing = nTargetSpacing;
	}
	else if (nActualSpacing > nTargetSpacing * 10) {
		nActualSpacing = nTargetSpacing * 10;
	}

	// target change every block
	// retarget with exponential moving toward target spacing
	// Includes fix for wrong retargeting difficulty by Mammix2
	arith_uint256 bnNew;
	bnNew.SetCompact(pindexPrev->nBits);

	int64_t nInterval = 10;
	bnNew *= ((nInterval - 1) * nTargetSpacing + nActualSpacing + nActualSpacing);
	bnNew /= ((nInterval + 1) * nTargetSpacing);

    if (bnNew <= 0 || bnNew > nProofOfWorkLimit)
    bnNew = nProofOfWorkLimit;

	return bnNew.GetCompact();

}