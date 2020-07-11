// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_RPC_MINING_H
#define DYNAMIC_RPC_MINING_H

#include "script/script.h"

#include <univalue.h>

static const bool DEFAULT_GENERATE = false;
static const uint8_t DEFAULT_GENERATE_THREADS_CPU = 0;
static const uint8_t DEFAULT_GENERATE_THREADS_GPU = 0;

/** Generate blocks (mine) */
UniValue generateBlocks(std::shared_ptr<CReserveScript> coinbaseScript, int nGenerate, uint64_t nMaxTries, bool keepScript);

UniValue getgenerate(const UniValue& params, bool fHelp);

/** Check bounds on a command line confirm target */
unsigned int ParseConfirmTarget(const UniValue& value);

#endif // DYNAMIC_RPC_MINING_H