// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "miner/internal/miner-base.h"
#include "chainparams.h"
#include "miner/miner-util.h"
#include "primitives/block.h"
#include "util.h"
#include "validation.h"
#include "validationinterface.h"

#include <assert.h>

#include <boost/thread.hpp>

MinerBase::MinerBase(MinerContextRef ctx, std::size_t device_index)
    : _ctx(ctx),
      _device_index(device_index)
{
    GetMainSignals().ScriptForMining(_coinbase_script);
    // Throw an error if no script was provided.  This can happen
    // due to some internal error but also if the keypool is empty.
    // In the latter case, already the pointer is NULL.
    if (!_coinbase_script || _coinbase_script->reserveScript.empty()) {
        throw std::runtime_error("No coinbase script available (mining requires a wallet)");
    }
};

void MinerBase::Loop()
{
    LogPrintf("DynamicMiner -- started on %s#%d\n", DeviceName(), _device_index);
    SetThreadPriority(THREAD_PRIORITY_LOWEST);
    RenameThread(tfm::format("dynamic-%s-miner-%d", DeviceName(), _device_index).data());

    CBlock block;
    CBlockIndex* chain_tip = nullptr;
    int64_t block_time = 0;
    std::shared_ptr<CBlockTemplate> block_template = {nullptr};

    try {
        while (true) {
            // Update block and tip if changed
            if (block_time != _ctx->shared->block_time()) {
                // set new block template
                block_template = _ctx->shared->block_template();
                block = block_template->block;
                // set block reserve script
                SetBlockPubkeyScript(block, _coinbase_script->reserveScript);
                // set block flag only after template
                // so we've waited for RecreateBlock
                block_time = _ctx->shared->block_time();
                // block template chain tip
                chain_tip = _ctx->shared->tip();
            }
            // Make sure we have a tip
            assert(chain_tip != nullptr);
            assert(block_template != nullptr);
            // Increment nonce
            IncrementExtraNonce(block, chain_tip, _extra_nonce);
            LogPrintf("DynamicMiner -- Running miner on device %s#%d with %u transactions in block (%u bytes)\n", DeviceName(), _device_index, block.vtx.size(),
                GetSerializeSize(block, SER_NETWORK, PROTOCOL_VERSION));
            // set loop start for counter
            _hash_target = arith_uint256().SetCompact(block.nBits);
            // start mining the block
            while (true) {
                // try mining the block
                int64_t hashes = TryMineBlock(block);
                // increment hash statistics
                _ctx->counter->Increment(hashes);
                // Check for stop or if block needs to be rebuilt
                boost::this_thread::interruption_point();
                // Check if block was recreated
                if (block_time != _ctx->shared->block_time()) {
                    break;
                }
                // Recreate block if nonce too big
                if (block.nNonce >= 0xffff0000) {
                    _ctx->shared->RecreateBlock();
                    break;
                }
                // Update block time
                if (UpdateTime(block, _ctx->chainparams().GetConsensus(), chain_tip) < 0) {
                    // Recreate the block if the clock has run backwards,
                    // so that we can use the correct time.
                    _ctx->shared->RecreateBlock();
                    break;
                }
                if (_ctx->chainparams().GetConsensus().fPowAllowMinDifficultyBlocks) {
                    // Changing block.nTime can change work required on testnet:
                    _hash_target.SetCompact(block.nBits);
                }
            }
        }
    } catch (const boost::thread_interrupted&) {
        LogPrintf("DynamicMiner%s -- terminated\n", DeviceName());
        throw;
    } catch (const std::runtime_error& e) {
        LogPrintf("DynamicMiner%s -- runtime error: %s\n", DeviceName(), e.what());
        return;
    }
}

void MinerBase::ProcessFoundSolution(const CBlock& block, const uint256& hash)
{
    // Found a solution
    SetThreadPriority(THREAD_PRIORITY_NORMAL);
    LogPrintf("DynamicMiner%s:\n proof-of-work found  \n  hash: %s  \ntarget: %s\n", DeviceName(), hash.GetHex(), _hash_target.GetHex());
    ProcessBlockFound(block, _ctx->chainparams());
    SetThreadPriority(THREAD_PRIORITY_LOWEST);
    _coinbase_script->KeepScript();

    // TODO: it needs to close all miners
    // In regression test mode, stop mining after a block is found.
    if (_ctx->chainparams().MineBlocksOnDemand()) {
        throw boost::thread_interrupted();
    }
}
