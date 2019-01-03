// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "miner/internal/miners-controller.h"
#include "chain.h"
#include "miner/internal/miner-context.h"
#include "miner/miner-util.h"
#include "net.h"
#include "validation.h"
#include "validationinterface.h"


MinersController::MinersController(const CChainParams& chainparams, CConnman& connman)
    : MinersController(std::make_shared<MinerContext>(chainparams, connman)){};

MinersController::MinersController(MinerContextRef ctx)
    : _ctx(ctx),
      _group_cpu(_ctx->MakeChild()),
#ifdef ENABLE_GPU
      _group_gpu(_ctx->MakeChild()),
#endif // ENABLE_GPU
      _connected(!_ctx->chainparams().MiningRequiresPeers()){};

void MinersController::Start()
{
    _connected = _ctx->connman().GetNodeCount(CConnman::CONNECTIONS_ALL) >= 2;
    _enable_start = true;
    _signals = std::make_shared<MinerSignals>(this);
    // initialize block template
    _ctx->shared->RecreateBlock();
    LogPrintf("MinersController::Start can_start = %v\n", can_start());

    if (can_start()) {
        _group_cpu.Start();
#ifdef ENABLE_GPU
        _group_gpu.Start();
#endif // ENABLE_GPU
    }
};

void MinersController::Shutdown()
{
    _enable_start = false;
    _signals = nullptr; // remove signals receiver

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

MinerSignals::MinerSignals(MinersController* ctr)
    : _ctr(ctr),
      _node(_ctr->ctx()->connman().ConnectSignalNode(boost::bind(&MinerSignals::NotifyNode, this, _1))),
      _block(GetMainSignals().UpdatedBlockTip.connect(boost::bind(&MinerSignals::NotifyBlock, this, _1, _2, _3))),
      _txn(GetMainSignals().SyncTransaction.connect(boost::bind(&MinerSignals::NotifyTransaction, this, _1, _2, _3))){};

void MinerSignals::NotifyNode(const CNode* node)
{
    if (_ctr->ctx()->connman().GetNodeCount(CConnman::CONNECTIONS_ALL) >= 2) {
        _ctr->_connected = true;
    } else if (_ctr->ctx()->connman().GetNodeCount(CConnman::CONNECTIONS_ALL) <= 1) {
        _ctr->_connected = false;
    }
};

void MinerSignals::NotifyBlock(const CBlockIndex* index_new, const CBlockIndex* index_fork, bool fInitialDownload)
{
    if (fInitialDownload)
        return;
    // Compare with current tip (checks for unexpected behaviour or old block)
    if (index_new != chainActive.Tip())
        return;
    // Create new block template for miners
    _ctr->_ctx->shared->RecreateBlock();
    // start miners
    if (_ctr->can_start()) {
        _ctr->_group_cpu.Start();
#ifdef ENABLE_GPU
        _ctr->_group_gpu.Start();
#endif // ENABLE_GPU
    }
};

void MinerSignals::NotifyTransaction(const CTransaction& txn, const CBlockIndex* index, int posInBlock)
{
    // check if blockchain has synced, has more than 1 peer and is enabled before recreating blocks
    if (IsInitialBlockDownload() || !_ctr->can_start())
        return;
    if (GetTime() - _ctr->_ctx->shared->last_txn() > 60) {
        _ctr->_ctx->shared->RecreateBlock();
    }
};
