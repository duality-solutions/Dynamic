// Copyright (c) 2016-2017 Plaxton/Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "checkforks.h"

bool CheckForkIsTrue(ForkID identifier, const CBlockIndex* pindexLast, bool fTableFlip) {
	bool booleanParam;
	const Consensus::Params& consensusParams = Params().GetConsensus();

	int placeholderIntX = 2;
	int placeholderIntY = 3;

	// Check if we are handling a valid fork
	if (identifier == DELTA_RETARGET || identifier == PRE_DELTA_RETARGET || identifier == START_DYNODE_PAYMENTS || identifier == FORK_SLOT_3 || identifier == FORK_SLOT_4) {  
		// Have we forked to the DELTA Retargeting Algorithm? (We're using pindexLast here because of logical reason)
		if((pindexLast->nHeight + 1) > consensusParams.nUpdateDiffAlgoHeight && identifier == DELTA_RETARGET) { booleanParam = true; }
		// Are we using the reward system before DELTA Retargeting's Fork?
		else if (chainActive.Height() < consensusParams.nUpdateDiffAlgoHeight && identifier == PRE_DELTA_RETARGET) { booleanParam = true; }
		// Have we now formally enabled Dynode Payments?
		else if (chainActive.Height() > consensusParams.nDynodePaymentsStartBlock && identifier == START_DYNODE_PAYMENTS) { booleanParam = true; }
		// Empty Forking Slot III
		else if (placeholderIntX == placeholderIntY && identifier == FORK_SLOT_3) { booleanParam = true; } 
		// Empty Forking Slot IV
		else if (placeholderIntX == placeholderIntY && identifier == FORK_SLOT_4) { booleanParam = true; } 
		// All parameters do not lead to forks!
		else { booleanParam = false; }
	// There seems to be an invalid entry!
	} else { throw std::runtime_error(strprintf("%s: Unknown Fork Verification Cause! %s.", __func__, identifier)); }

	return booleanParam;
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