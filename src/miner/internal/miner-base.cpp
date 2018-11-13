// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2018 The Dash Core Developers
// Copyright (c) 2009-2018 The Bitcoin Developers
// Copyright (c) 2009-2018 Satoshi Nakamoto
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


MinerBase::MinerBase(MinerContextRef ctx, std::size_t device_index) : _ctx(ctx), _device_index(device_index){};

void MinerBase::Loop()
{
    LogPrintf("DynamicMiner -- started on %s#%d\n", DeviceName(), _device_index);
    SetThreadPriority(THREAD_PRIORITY_LOWEST);
    RenameThread(tfm::format("dynamic-%s-miner-%d", DeviceName(), _device_index).data());

    CBlock block;
    CBlockIndex* current_tip = nullptr;
    std::shared_ptr<CBlockTemplate> block_template = {nullptr};

    try {
        while (true) {
            // Get blockchain tip
            CBlockIndex* next_tip = _ctx->tip();
            // Get shared block template
            auto next_template = _ctx->block_template();
            assert(next_template != nullptr);
            CBlock& next_block = next_template->block;
            // Update block and tip if changed
            if (current_tip != next_tip || (next_block.nBits != block.nBits && next_block.nTime > block.nTime)) {
                block = next_block;
                block_template = next_template;
                current_tip = next_tip;
            }
            // Make sure we have a tip
            assert(current_tip != nullptr);
            // Increment nonce
            IncrementExtraNonce(block, current_tip, _extra_nonce);
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
                if (block.nNonce >= 0xffff0000) {
                    _ctx->RecreateBlock();
                    break;
                }
                // Update block time
                if (UpdateTime(block, _ctx->chainparams().GetConsensus(), current_tip) < 0) {
                    // Recreate the block if the clock has run backwards,
                    // so that we can use the correct time.
                    _ctx->RecreateBlock();
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
    _ctx->coinbase_script()->KeepScript();

    // TODO: it needs to close all miners
    // In regression test mode, stop mining after a block is found.
    if (_ctx->chainparams().MineBlocksOnDemand()) {
        throw boost::thread_interrupted();
    }
}
