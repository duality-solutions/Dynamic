// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "miner/internal/miners-controller.h"
#include "chain.h"
#include "miner/internal/miner-context.h"
#include "miner/miner-util.h"
#include "net.h"
#include "txmempool.h"
#include "validation.h"
#include "validationinterface.h"

void ConnectMinerSignals(MinersController* miners_)
{
    miners_->ctx()->connman().ConnectSignalNode(boost::bind(&MinersController::NotifyNode, miners_, _1));
    GetMainSignals().UpdatedBlockTip.connect(boost::bind(&MinersController::NotifyBlock, miners_, _1, _2, _3));
    GetMainSignals().SyncTransaction.connect(boost::bind(&MinersController::NotifyTransaction, miners_, _1, _2, _3));
}

MinersController::MinersController(const CChainParams& chainparams, CConnman& connman)
    : MinersController(std::make_shared<MinerContext>(chainparams, connman)){};

MinersController::MinersController(MinerContextRef ctx)
    : _ctx(ctx),
      _group_cpu(_ctx->MakeChild()),
#ifdef ENABLE_GPU
      _group_gpu(_ctx->MakeChild()),
#endif // ENABLE_GPU
      _connected(!_ctx->chainparams().MiningRequiresPeers()),
      _downloaded(!_ctx->chainparams().MiningRequiresPeers())
{
    ConnectMinerSignals(this);
};

void MinersController::Start()
{
    _enable_start = true;
    if (can_start()) {
        _group_cpu.Start();
#ifdef ENABLE_GPU
        _group_gpu.Start();
#endif // ENABLE_GPU
    }
};

void MinersController::Shutdown()
{
    _group_cpu.Shutdown();
#ifdef ENABLE_GPU
    _group_gpu.Shutdown();
#endif // ENABLE_GPU
};

int64_t MinersController::GetHashRate() const
{
#ifdef ENABLE_GPU
    return _group_gpu.GetHashRate() + _group_cpu.GetHashRate();
#else
    return _group_cpu.GetHashRate();
#endif // ENABLE_GPU
}

void MinersController::NotifyNode(const CNode* node)
{
    _connected = true;
    Start();
};

void MinersController::NotifyBlock(const CBlockIndex* index_new, const CBlockIndex* index_fork, bool fInitialDownload)
{
    if (fInitialDownload)
        return;
    // Compare with current tip (checks for unexpected behaviour or old block)
    if (index_new != chainActive.Tip())
        return;
    // Create new block template for miners
    _last_sync_time = GetTime();
    _last_txn_time = mempool.GetTransactionsUpdated();
    _ctx->shared->RecreateBlock();
    // start miners
    if (can_start()) {
        _group_cpu.Start();
#ifdef ENABLE_GPU
        _group_gpu.Start();
#endif // ENABLE_GPU
    }
};

void MinersController::NotifyTransaction(const CTransaction& txn, const CBlockIndex* index, int posInBlock)
{
    // If blockchain hasn't synced do not allow miners to recreate blocks
    if (IsInitialBlockDownload())
        return;
    
    const int64_t latest_txn = mempool.GetTransactionsUpdated();
    if (latest_txn == _last_txn_time) {
        return;
    }
    if (GetTime() - _last_txn_time > 60) {
        _last_txn_time = latest_txn;
        _ctx->shared->RecreateBlock();
    }
};
