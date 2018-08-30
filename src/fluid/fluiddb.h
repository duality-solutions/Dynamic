// Copyright (c) 2018 Duality Blockchain Solutions Developers

#ifndef FLUID_DB_H
#define FLUID_DB_H

#include "amount.h"

class CFluidMint;

CAmount GetFluidDynodeReward();
CAmount GetFluidMiningReward();
bool GetMintingInstructions(const int nHeight, CFluidMint& fluidMint);

#endif // FLUID_DYNODE_H
