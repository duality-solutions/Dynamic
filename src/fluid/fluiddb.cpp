// Copyright (c) 2018 Duality Blockchain Solutions Developers

#include "fluiddb.h"

#include "fluid.h"
#include "fluiddynode.h"
#include "fluidmining.h"
#include "fluidmint.h"

CAmount GetFluidDynodeReward() 
{
    if (!CheckFluidDynodeDB())
        return GetStandardDynodePayment();

    CFluidDynode lastDynodeRecord;
    if (!pFluidDynodeDB->GetLastFluidDynodeRecord(lastDynodeRecord)) {
    	return GetStandardDynodePayment();
    }
    return lastDynodeRecord.DynodeReward;
}

CAmount GetFluidMiningReward() 
{
    if (!CheckFluidMiningDB())
        return GetStandardPoWBlockPayment();

    CFluidMining lastMiningRecord;
    if (!pFluidMiningDB->GetLastFluidMiningRecord(lastMiningRecord)) {
    	return GetStandardPoWBlockPayment();
    }
    return lastMiningRecord.MiningReward;
}

bool GetMintingInstructions(const int nHeight, CFluidMint& fluidMint)
{
    if (!CheckFluidMintDB())
        return false;
	
	CFluidMint getFluidMint;
    if (!pFluidMintDB->GetLastFluidMintRecord(getFluidMint)) {
        return false;
    }
	
	if ((int)getFluidMint.nHeight == (nHeight -1))
	{
		fluidMint = getFluidMint;
		return true;
	}
    return false;
}