// Copyright (c) 2021 Duality Blockchain Solutions

#ifndef BITCOIN_FLUID_SCRIPT_H
#define BITCOIN_FLUID_SCRIPT_H

#include "script/script.h"

#include <string>
#include <cstdlib>

bool WithinFluidRange(opcodetype op);

std::string TranslationTable(opcodetype op);
uint64_t TranslationTable(std::string str);

#endif // BITCOIN_FLUID_SCRIPT_H