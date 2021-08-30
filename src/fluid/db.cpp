// Copyright (c) 2019-2021 Duality Blockchain Solutions Developers

#include "fluid/db.h"

#include "base58.h"
#include "fluid/fluid.h"
#include "fluid/dynode.h"
#include "fluid/mining.h"
#include "fluid/mint.h"
#include "fluid/sovereign.h"

CAmount GetFluidDynodeReward(const int nHeight)
{
    CFluidDynode lastDynodeRecord;

    assert(pFluidDynodeDB);
    if (pFluidDynodeDB->GetLastFluidDynodeRecord(lastDynodeRecord, nHeight) && !pFluidDynodeDB->IsEmpty()) {
        return (lastDynodeRecord.GetReward() > 0 && FLUID_ACTIVATE_HEIGHT > nHeight) ?
                lastDynodeRecord.GetReward() : GetStandardDynodePayment(nHeight);
    }

    return GetStandardDynodePayment(nHeight);
}

CAmount GetFluidMiningReward(const int nHeight)
{
    CFluidMining lastMiningRecord;

    assert(pFluidMiningDB);
    if (pFluidMiningDB->GetLastFluidMiningRecord(lastMiningRecord, nHeight) && !pFluidMiningDB->IsEmpty()) {
        return (lastMiningRecord.GetReward() > 0 && FLUID_ACTIVATE_HEIGHT > nHeight) ?
                lastMiningRecord.GetReward() : GetStandardPoWBlockPayment(nHeight);
    }

    return GetStandardDynodePayment(nHeight);
}

bool GetMintingInstructions(const int nHeight, CFluidMint& fluidMint)
{
    assert(pFluidDynodeDB);

    if (pFluidMintDB->IsEmpty())
        return false;

    CFluidMint getFluidMint;
    if (!pFluidMintDB->GetLastFluidMintRecord(getFluidMint)) {
        return false;
    }

    if (getFluidMint.GetHeight() == (nHeight - 1)) {
        getFluidMint = fluidMint;
        return true;
    }
    return false;
}

/** Checks if any given address is a current sovereign wallet address (invoked by RPC) */
bool IsSovereignAddress(const CDynamicAddress& inputAddress)
{
    CFluidSovereign lastSovereign;

    if (!inputAddress.IsValid()) {
        return false;
    }

    assert(pFluidSovereignDB);
    if (pFluidSovereignDB->GetLastFluidSovereignRecord(lastSovereign)) {
        for (const std::vector<unsigned char>& vchAddress : lastSovereign.obj_sigs) {
            CDynamicAddress attemptKey(StringFromCharVector(vchAddress));
            return attemptKey.IsValid() && inputAddress == attemptKey;
        }
    }

    return false;
}

bool GetAllFluidDynodeRecords(std::vector<CFluidDynode>& dynodeEntries)
{
    assert(pFluidDynodeDB);
    return pFluidDynodeDB->GetAllFluidDynodeRecords(dynodeEntries);
}

bool GetAllFluidMiningRecords(std::vector<CFluidMining>& miningEntries)
{
    assert(pFluidMiningDB);
    return pFluidMiningDB->GetAllFluidMiningRecords(miningEntries);
}

bool GetAllFluidMintRecords(std::vector<CFluidMint>& mintEntries)
{
    assert(pFluidMintDB);
    return pFluidMintDB->GetAllFluidMintRecords(mintEntries);
}

bool GetAllFluidSovereignRecords(std::vector<CFluidSovereign>& sovereignEntries)
{
    assert(pFluidSovereignDB);
    return pFluidSovereignDB->GetAllFluidSovereignRecords(sovereignEntries) && !pFluidSovereignDB->IsEmpty();
}

bool GetLastFluidSovereignAddressStrings(std::vector<std::string>& sovereignAddresses)
{
    assert(pFluidSovereignDB);
    CFluidSovereign lastSovereign;
    if (pFluidSovereignDB->GetLastFluidSovereignRecord(lastSovereign))
    {
        sovereignAddresses = lastSovereign.SovereignAddressesStrings();
        return true;
    }
    return false;
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