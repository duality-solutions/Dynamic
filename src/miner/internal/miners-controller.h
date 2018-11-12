// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_INTERNAL_MINERS_CONTROLLER_H
#define DYNAMIC_INTERNAL_MINERS_CONTROLLER_H

#include "chainparams.h"
#include "miner/impl/miner-cpu.h"
#include "miner/impl/miner-gpu.h"
#include "miner/internal/miners-group.h"

class CNode;
class CConnman;
class CReserveScript;
class CChainParams;
class CBlockIndex;
class CTransaction;
struct CBlockTemplate;

class MinersController;

void ConnectMinerSignals(MinersController*);

/**
 * Miner controller for both CPU and GPU threads.
 */
class MinersController
{
private:
    MinerContextRef _ctx;
    MinersThreadGroup<CPUMiner> _group_cpu;
#ifdef ENABLE_GPU
    MinersThreadGroup<GPUMiner> _group_gpu;
#endif // ENABLE_GPU

    bool _connected;
    bool _downloaded;
    bool _enable_start = false;

    int64_t _last_txn_time = 0;
    int64_t _last_sync_time = 0;

public:
    MinersController(MinerContextRef ctx);
    MinersController(const CChainParams& chainparams, CConnman& connman);
    virtual ~MinersController() = default;

    /** Starts miners */
    void Start();

    /** Shuts down all miner threads */
    void Shutdown();

    /** Gets combined hash rate of GPU and CPU */
    int64_t GetHashRate() const;

    /** Returns CPU miners thread group */
    MinersThreadGroup<CPUMiner>& group_cpu() { return _group_cpu; }

#ifdef ENABLE_GPU
    /** Returns GPU miners thread group */
    MinersThreadGroup<GPUMiner>& group_gpu()
    {
        return _group_gpu;
    }
#endif // ENABLE_GPU

private:
    void StartIfEnabled();
    void InitializeCoinbaseScript();

    /** Returns true if can start */
    bool can_start() const { return _connected && _downloaded && _enable_start && _ctx->has_block(); }

protected:
    /** Returns shared miner context */
    MinerContextRef ctx() const { return _ctx; }

    /** Handles new node connection */
    virtual void NotifyNode(const CNode* node);

    /** Handles updated blockchain tip */
    virtual void NotifyBlock(const CBlockIndex* pindexNew, const CBlockIndex* pindexFork, bool fInitialDownload);

    /** Handles new transaction */
    virtual void NotifyTransaction(const CTransaction& tx, const CBlockIndex* pindex, int posInBlock);

    /** Connects signals to controller */
    friend void ConnectMinerSignals(MinersController*);
};

#endif // DYNAMIC_INTERNAL_MINERS_CONTROLLER_H
