// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "miner/internal/miner-context.h"
#include "miner/miner-util.h"
#include "validation.h"

void MinerSharedContext::RecreateBlock()
{
    // First we increment flag
    // so all miner threads know new block is coming
    _block_flag += 1;
    // Then we acquire unique lock so that miners wait
    // for the new block to be created
    boost::unique_lock<boost::shared_mutex> guard(_mutex);
    _chain_tip = chainActive.Tip();
    _block_template = CreateNewBlock(chainparams);
}
