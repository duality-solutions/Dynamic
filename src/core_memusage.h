// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_CORE_MEMUSAGE_H
#define DYNAMIC_CORE_MEMUSAGE_H

#include "memusage.h"

#include "primitives/block.h"
#include "primitives/transaction.h"

static inline size_t RecursiveDynamicUsage(const CScript& script)
{
    return memusage::DynamicUsage(*static_cast<const CScriptBase*>(&script));
}

static inline size_t RecursiveDynamicUsage(const COutPoint& out)
{
    return 0;
}

static inline size_t RecursiveDynamicUsage(const CTxIn& in)
{
    return RecursiveDynamicUsage(in.scriptSig) + RecursiveDynamicUsage(in.prevout);
}

static inline size_t RecursiveDynamicUsage(const CTxOut& out)
{
    return RecursiveDynamicUsage(out.scriptPubKey);
}

static inline size_t RecursiveDynamicUsage(const CTransaction& tx)
{
    size_t mem = memusage::DynamicUsage(tx.vin) + memusage::DynamicUsage(tx.vout);
    for (std::vector<CTxIn>::const_iterator it = tx.vin.begin(); it != tx.vin.end(); it++) {
        mem += RecursiveDynamicUsage(*it);
    }
    for (std::vector<CTxOut>::const_iterator it = tx.vout.begin(); it != tx.vout.end(); it++) {
        mem += RecursiveDynamicUsage(*it);
    }
    return mem;
}

static inline size_t RecursiveDynamicUsage(const CMutableTransaction& tx)
{
    size_t mem = memusage::DynamicUsage(tx.vin) + memusage::DynamicUsage(tx.vout);
    for (std::vector<CTxIn>::const_iterator it = tx.vin.begin(); it != tx.vin.end(); it++) {
        mem += RecursiveDynamicUsage(*it);
    }
    for (std::vector<CTxOut>::const_iterator it = tx.vout.begin(); it != tx.vout.end(); it++) {
        mem += RecursiveDynamicUsage(*it);
    }
    return mem;
}

static inline size_t RecursiveDynamicUsage(const CBlock& block)
{
    size_t mem = memusage::DynamicUsage(block.vtx);
    for (const auto& tx : block.vtx) {
        mem += memusage::DynamicUsage(tx) + RecursiveDynamicUsage(*tx);
    }
    return mem;
}

static inline size_t RecursiveDynamicUsage(const CBlockLocator& locator)
{
    return memusage::DynamicUsage(locator.vHave);
}

#endif // DYNAMIC_CORE_MEMUSAGE_H
