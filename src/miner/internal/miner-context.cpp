// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2018 The Dash Core Developers
// Copyright (c) 2009-2018 The Bitcoin Developers
// Copyright (c) 2009-2018 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "miner/internal/miner-context.h"
#include "chain.h"
#include "miner/miner-util.h"
#include "net.h"
#include "txmempool.h"
#include "validation.h"
#include "validationinterface.h"

void MinerSharedContext::InitializeCoinbaseScript()
{
    boost::unique_lock<boost::shared_mutex> guard(_mutex);
    GetMainSignals().ScriptForMining(_coinbase_script);
    // Throw an error if no script was provided.  This can happen
    // due to some internal error but also if the keypool is empty.
    // In the latter case, already the pointer is NULL.
    if (!_coinbase_script || _coinbase_script->reserveScript.empty()) {
        throw std::runtime_error("No coinbase script available (mining requires a wallet)");
    }
}

void MinerSharedContext::RecreateBlock()
{
    boost::unique_lock<boost::shared_mutex> guard(_mutex);

    _tip = chainActive.Tip();
    _block_template = CreateNewBlock(chainparams, _coinbase_script->reserveScript);
}

CBlockIndex* MinerSharedContext::tip()
{
    boost::shared_lock<boost::shared_mutex> guard(_mutex);
    return _tip;
}

std::shared_ptr<CBlockTemplate> MinerSharedContext::block_template()
{
    boost::shared_lock<boost::shared_mutex> guard(_mutex);
    return _block_template;
}
