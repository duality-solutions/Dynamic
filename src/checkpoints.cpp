// Copyright (c) 2016-2021 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2021 The Dash Core Developers
// Copyright (c) 2009-2021 The Bitcoin Developers
// Copyright (c) 2009-2021 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "checkpoints.h"

#include "chain.h"
#include "chainparams.h"
#include "uint256.h"
#include "validation.h"

#include <stdint.h>

#include <boost/foreach.hpp>

namespace Checkpoints
{
CBlockIndex* GetLastCheckpoint(const CCheckpointData& data)
{
    const MapCheckpoints& checkpoints = data.mapCheckpoints;

    BOOST_REVERSE_FOREACH (const MapCheckpoints::value_type& i, checkpoints) {
        const uint256& hash = i.second;
        BlockMap::const_iterator t = mapBlockIndex.find(hash);
        if (t != mapBlockIndex.end())
            return t->second;
    }
    return NULL;
}

bool fEnabled = true;

bool CheckBlock(int nHeight, const uint256& hash, bool fMatchesCheckpoint)
{
    if (!fEnabled)
        return true;

    const MapCheckpoints& checkpoints = Params().Checkpoints().mapCheckpoints;
    MapCheckpoints::const_iterator i = checkpoints.find(nHeight);
    // If looking for an exact match, then return false
    if (i == checkpoints.end()) return !fMatchesCheckpoint;
    return hash == i->second;
}
} // namespace Checkpoints