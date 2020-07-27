// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_AMOUNT_H
#define DYNAMIC_AMOUNT_H

#include "serialize.h"

#include <stdlib.h>
#include <string>

typedef int64_t CAmount;

static const CAmount COIN = 100000000;
static const CAmount CENT = 1000000;
static const CAmount BDAP_CREDIT = 100001; //= 0.00100001 DYN. Matches lowest PrivateSend denomination

extern const std::string CURRENCY_UNIT;

// No amount larger than this (in satoshi) is valid.
static const CAmount MAX_MONEY = std::numeric_limits<int64_t>::max();
inline bool MoneyRange(const CAmount& nValue) { return (nValue >= 0 && nValue <= MAX_MONEY); }

#endif //  DYNAMIC_AMOUNT_H
