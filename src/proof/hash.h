// Copyright (c) 2021 Duality Blockchain Solutions
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_PROOF_HASH_H
#define BITCOIN_PROOF_HASH_H

#include <uint256.h>

#include <vector>

uint256 RXHashFunction(const char* pbegin, const char* pend, const unsigned char* k_begin, const unsigned char* k_end);

#endif // BITCOIN_PROOF_HASH_H