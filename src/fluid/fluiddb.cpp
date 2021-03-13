// Copyright (c) 2019 Duality Blockchain Solutions Developers

#include "fluiddb.h"

#include "base58.h"
#include "fluid.h"
#include "fluiddynode.h"
#include "fluidmining.h"
#include "fluidmint.h"
#include "fluidsovereign.h"

CAmount GetFluidDynodeReward(const int nHeight)
{
    if (fluid.FLUID_ACTIVATE_HEIGHT > nHeight)
        return GetStandardDynodePayment(nHeight);

    if (!CheckFluidDynodeDB())
        return GetStandardDynodePayment(nHeight);

    if (pFluidDynodeDB->IsEmpty())
        return GetStandardDynodePayment(nHeight);

    CFluidDynode lastDynodeRecord;
    if (!pFluidDynodeDB->GetLastFluidDynodeRecord(lastDynodeRecord, nHeight)) {
        return GetStandardDynodePayment(nHeight);
    }
    if (lastDynodeRecord.DynodeReward > 0) {
        return lastDynodeRecord.DynodeReward;
    } else {
        return GetStandardDynodePayment(nHeight);
    }
}

CAmount GetFluidMiningReward(const int nHeight)
{
    if (fluid.FLUID_ACTIVATE_HEIGHT > nHeight)
        return GetStandardPoWBlockPayment(nHeight);

    if (!CheckFluidMiningDB())
        return GetStandardPoWBlockPayment(nHeight);

    if (pFluidMiningDB->IsEmpty())
        return GetStandardPoWBlockPayment(nHeight);

    CFluidMining lastMiningRecord;
    if (!pFluidMiningDB->GetLastFluidMiningRecord(lastMiningRecord, nHeight)) {
        return GetStandardPoWBlockPayment(nHeight);
    }
    if (lastMiningRecord.MiningReward > 0) {
        return lastMiningRecord.MiningReward;
    } else {
        return GetStandardPoWBlockPayment(nHeight);
    }
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

    if ((int)getFluidMint.nHeight == (nHeight - 1)) {
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
    } else {
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
    } else {
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
    } else {
        return false;
    }
    return true;
}

bool GetAllFluidSovereignRecords(std::vector<CFluidSovereign>& sovereignEntries)
{
    if (CheckFluidSovereignDB()) {
        if (pFluidSovereignDB->IsEmpty()) {
            return false;
        }
        if (!pFluidSovereignDB->GetAllFluidSovereignRecords(sovereignEntries)) {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

bool GetLastFluidSovereignAddressStrings(std::vector<std::string>& sovereignAddresses)
{
    if (!CheckFluidSovereignDB()) {
        return false;
    }

    CFluidSovereign lastSovereign;
    if (!pFluidSovereignDB->GetLastFluidSovereignRecord(lastSovereign)) {
        return false;
    }
    sovereignAddresses = lastSovereign.SovereignAddressesStrings();
    return true;
}

/** Checks whether 3 of 5 sovereign addresses signed the token in the script to meet the quorum requirements */
bool CheckSignatureQuorum(const std::vector<unsigned char>& vchFluidScript, std::string& errMessage, bool individual)
{
    std::string consentToken = StringFromCharVector(vchFluidScript);
    std::vector<std::string> fluidSovereigns;
    if (!GetLastFluidSovereignAddressStrings(fluidSovereigns)) {
        return false;
    }

    std::pair<CDynamicAddress, bool> keyOne;
    std::pair<CDynamicAddress, bool> keyTwo;
    std::pair<CDynamicAddress, bool> keyThree;
    keyOne.second = false;
    keyTwo.second = false;
    keyThree.second = false;

    for (const std::string& sovereignAddress : fluidSovereigns) {
        CDynamicAddress attemptKey;
        CDynamicAddress xKey(sovereignAddress);

        if (!xKey.IsValid())
            return false;
        CFluid fluid;
        if (fluid.GenericVerifyInstruction(consentToken, attemptKey, errMessage, 1) && xKey == attemptKey) {
            keyOne = std::make_pair(attemptKey.ToString(), true);
        }

        if (fluid.GenericVerifyInstruction(consentToken, attemptKey, errMessage, 2) && xKey == attemptKey) {
            keyTwo = std::make_pair(attemptKey.ToString(), true);
        }

        if (fluid.GenericVerifyInstruction(consentToken, attemptKey, errMessage, 3) && xKey == attemptKey) {
            keyThree = std::make_pair(attemptKey.ToString(), true);
        }
    }

    bool fValid = (keyOne.first.ToString() != keyTwo.first.ToString() && keyTwo.first.ToString() != keyThree.first.ToString() && keyOne.first.ToString() != keyThree.first.ToString());

    LogPrint("fluid", "CheckSignatureQuorum(): Addresses validating this consent token are: %s, %s and %s\n", keyOne.first.ToString(), keyTwo.first.ToString(), keyThree.first.ToString());

    if (individual)
        return (keyOne.second || keyTwo.second || keyThree.second);
    else if (fValid)
        return (keyOne.second && keyTwo.second && keyThree.second);

    return false;
}