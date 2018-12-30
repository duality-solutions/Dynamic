// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "miner/internal/miner-context.h"
#include "miner/miner-util.h"
#include "txmempool.h"
#include "validation.h"

MinerContext::MinerContext(const CChainParams& chainparams_, CConnman& connman_)
    : counter(std::make_shared<HashRateCounter>()),
      shared(std::make_shared<MinerSharedContext>(chainparams_, connman_)){};

MinerContext::MinerContext(MinerSharedContextRef shared_, HashRateCounterRef counter_)
    : counter(counter_), shared(shared_){};

void MinerSharedContext::RecreateBlock()
{
    // Then we acquire unique lock so that miners wait
    // for the new block to be created
    boost::unique_lock<boost::shared_mutex> guard(_mutex);
    uint32_t txn_time = mempool.GetTransactionsUpdated();
    // pass if nothing changed
    if (_chain_tip == chainActive.Tip() && _last_txn == txn_time)
        return;
    _chain_tip = chainActive.Tip();
    _block_time = GetTime();
    _block_template = CreateNewBlock(chainparams);
    _last_txn = txn_time;
}
