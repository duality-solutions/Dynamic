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

std::string GetStringForIdentifier(ForkID identifier) {
	if (identifier == 1) { return "DELTA_RETARGET"; }
	else if (identifier == 2) { return "PRE_DELTA_RETARGET"; }
	else if (identifier == 3) { return "START_DYNODE_PAYMENTS"; }
	else if (identifier == 4) { return "FORK_SLOT_3"; }
	else if (identifier == 5) { return "FORK_SLOT_4"; }
	else { return "NO_IDENTIFER!"; }
}

bool CheckForkIsTrue(ForkID identifier, const CBlockIndex* pindexLast, bool fTableFlip) {

	bool booleanParam;
	bool trustable = true;
	const Consensus::Params& consensusParams = Params().GetConsensus();

	int placeholderIntX = 2;
	int placeholderIntY = 3;

	int32_t networkHeight = (pindexLast->nHeight);
	int32_t chainHeight = chainActive.Height();

	if(fTableFlip)
		return true;
	
	if(!fTableFlip)
	{
		// chainActive.Height() == 0 at all times equals to syncing, we need a workaround
		if(!(networkHeight == chainHeight) && chainActive.Height() != 0) {
			LogPrintf("CheckForkIsSane: Reported Network Height: %d vs Chain Height %d \n", networkHeight, chainHeight); 
			// Last minute negation function
			while((networkHeight+1) == chainHeight) { networkHeight++; }
			while((chainHeight+1) == networkHeight) { networkHeight--; }
			trustable = false;
		}

		// The genesis and the first block are fimble as anything, so we make an except and run our rule just in case
		if(pindexLast->nHeight >= 2 && chainActive.Height() == 0) {
			// We still need to maintain our forks even if we're just syncing, we cannot risk mess-ups
			chainHeight = networkHeight;
			trustable = false;
		}
	}

	// Check if we are handling a valid fork
	if (identifier == DELTA_RETARGET || identifier == PRE_DELTA_RETARGET || identifier == START_DYNODE_PAYMENTS || identifier == FORK_SLOT_3 || identifier == FORK_SLOT_4) {  
		// Have we forked to the DELTA Retargeting Algorithm?
		if(networkHeight > consensusParams.nUpdateDiffAlgoHeight && chainHeight > consensusParams.nUpdateDiffAlgoHeight && identifier == DELTA_RETARGET) { booleanParam = true; }
		// Are we using the reward system before DELTA Retargeting's Fork?
		else if (networkHeight < consensusParams.nUpdateDiffAlgoHeight && chainHeight < consensusParams.nUpdateDiffAlgoHeight && identifier == PRE_DELTA_RETARGET) { booleanParam = true; }
		// Have we now formally enabled Dynode Payments?
		else if (chainHeight > Params().GetConsensus().nDynodePaymentsStartBlock && identifier == START_DYNODE_PAYMENTS) { booleanParam = true; }
		// Empty Forking Slot III
		else if (placeholderIntX == placeholderIntY && identifier == FORK_SLOT_3) { booleanParam = true; } 
		// Empty Forking Slot IV
		else if (placeholderIntX == placeholderIntY && identifier == FORK_SLOT_4) { booleanParam = true; } 
		// All parameters do not lead to forks!
		else { booleanParam = false; }
	
		// Let's print
		LogPrintf("CheckForkIsTrue (%s): Reported Network Height: %d vs Chain Height %d : HaveWeForked to %s? %s \n", trustable?"CAN_TRUST":"CANT_TRUST", networkHeight, chainHeight, GetStringForIdentifier(identifier).c_str(), booleanParam?"true":"false");

	} else { throw std::runtime_error(strprintf("%s: Unknown Fork Verification Cause! %s.", __func__, identifier)); }

	if(!(pindexLast == NULL) && chainActive.Height() != 0)
		assert(chainHeight == networkHeight); // Well... are we even compairing with the correct parameters?

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