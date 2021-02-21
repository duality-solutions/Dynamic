// Copyright (c) 2019 Duality Blockchain Solutions Developers

#ifndef FLUID_DB_H
#define FLUID_DB_H

#include "amount.h"

class CDynamicAddress;
class CFluidDynode;
class CFluidMining;
class CFluidMint;
class CFluidSovereign;

CAmount GetFluidDynodeReward(const int nHeight);
CAmount GetFluidMiningReward(const int nHeight);
bool GetMintingInstructions(const int nHeight, CFluidMint& fluidMint);
bool IsSovereignAddress(const CDynamicAddress& inputAddress);
bool GetAllFluidDynodeRecords(std::vector<CFluidDynode>& dynodeEntries);
bool GetAllFluidMiningRecords(std::vector<CFluidMining>& miningEntries);
bool GetAllFluidMintRecords(std::vector<CFluidMint>& mintEntries);
bool GetAllFluidSovereignRecords(std::vector<CFluidSovereign>& sovereignEntries);
bool GetLastFluidSovereignAddressStrings(std::vector<std::string>& sovereignAddresses);
bool CheckSignatureQuorum(const std::vector<unsigned char>& vchFluidScript, std::string& errMessage, bool individual = false);

#endif // FLUID_DB_H
