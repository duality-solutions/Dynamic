// Copyright (c) 2021 Duality Blockchain Solutions
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_CRYPTO_RANDOMX_H
#define BITCOIN_CRYPTO_RANDOMX_H

#include <uint256.h>

uint256 RXHashFunction(const char* pbegin, const char* pend, const char* k_begin, const char* k_end);

#endif // BITCOIN_CRYPTO_RANDOMX_H