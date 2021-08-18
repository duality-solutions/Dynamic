// Copyright (c) 2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "base58.h"

#include <algorithm>
#include <stdint.h>
#include <string.h>
#include <vector>

void ScrubString(std::string& input, bool forInteger = false);
void SeparateString(const std::string& input, std::vector<std::string>& output, bool subDelimiter = false);
void SeparateFluidOpString(const std::string& input, std::vector<std::string>& output);
std::string StitchString(const std::string& stringOne, const std::string& stringTwo, const bool subDelimiter = false);
std::string StitchString(const std::string& stringOne, const std::string& stringTwo, const std::string& stringThree, const bool subDelimiter = false);
std::string GetRidOfScriptStatement(const std::string& input, const int& position = 1);

extern std::string PrimaryDelimiter;
extern std::string SubDelimiter;
extern std::string SignatureDelimiter;

bool VerifyAddressOwnership(const CDynamicAddress& dynamicAddress);
bool SignTokenMessage(const CDynamicAddress& address, std::string unsignedMessage, std::string& stitchedMessage, bool stitch = true);
bool GenericSignMessage(const std::string& message, std::string& signedString, const CDynamicAddress& signer);

#endif // OPERATIONS_H
