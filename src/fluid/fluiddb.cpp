// Copyright (c) 2018 Duality Blockchain Solutions Developers

#include "fluiddb.h"

#include "base58.h"
#include "fluid.h"
#include "fluiddynode.h"
#include "fluidmining.h"
#include "fluidmint.h"
#include "fluidsovereign.h"

CAmount GetFluidDynodeReward() 
{
    if (!CheckFluidDynodeDB())
        return GetStandardDynodePayment();

    if (pFluidDynodeDB->IsEmpty())
        return GetStandardDynodePayment();

    CFluidDynode lastDynodeRecord;
    if (!pFluidDynodeDB->GetLastFluidDynodeRecord(lastDynodeRecord)) {
        return GetStandardDynodePayment();
    }
    LogPrintf("GetFluidDynodeReward: lastDynodeRecord.DynodeReward = %u\n", lastDynodeRecord.DynodeReward);
    return lastDynodeRecord.DynodeReward;
}

CAmount GetFluidMiningReward() 
{
    if (!CheckFluidMiningDB())
        return GetStandardPoWBlockPayment();

    if (pFluidMiningDB->IsEmpty())
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
    
    if (pFluidMintDB->IsEmpty())
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

/** Checks if any given address is a current sovereign wallet address (invoked by RPC) */
bool IsSovereignAddress(const CDynamicAddress& inputAddress) 
{
    if (!inputAddress.IsValid()) {
        return false;
    }
    
    if (!CheckFluidSovereignDB()) {
        return false;
    }

    CFluidSovereign lastSovereign;
    if (!pFluidSovereignDB->GetLastFluidSovereignRecord(lastSovereign)) {
        return false;
    }
    
    for (const std::vector<unsigned char>& vchAddress : lastSovereign.SovereignAddresses) {
        CDynamicAddress attemptKey(StringFromCharVector(vchAddress));
        if (attemptKey.IsValid() && inputAddress == attemptKey) {
            return true;
        }
    }
    return false;
}

bool GetAllFluidDynodeRecords(std::vector<CFluidDynode>& dynodeEntries)
{
    if (CheckFluidDynodeDB()) { 
        if (!pFluidDynodeDB->GetAllFluidDynodeRecords(dynodeEntries)) {
            return false;
        }
    }
    else {
        return false;
    }
    return true;
}

bool GetAllFluidMiningRecords(std::vector<CFluidMining>& miningEntries)
{
    if (CheckFluidMiningDB()) { 
        if (!pFluidMiningDB->GetAllFluidMiningRecords(miningEntries)) {
            return false;
        }
    }
    else {
        return false;
    }
    return true;
}

bool GetAllFluidMintRecords(std::vector<CFluidMint>& mintEntries)
{
    if (CheckFluidMintDB()) { 
        if (!pFluidMintDB->GetAllFluidMintRecords(mintEntries)) {
            return false;
        }
    }
    else {
        return false;
    }
    return true;
}

bool GetAllFluidSovereignRecords(std::vector<CFluidSovereign>& sovereignEntries)
{
    if (CheckFluidSovereignDB())
    {
        if (pFluidSovereignDB->IsEmpty()) {
            return false;
        }
        if (!pFluidSovereignDB->GetAllFluidSovereignRecords(sovereignEntries)) {
            return false;
        }
    }
    else {
        return false;
    }
    return true;
}