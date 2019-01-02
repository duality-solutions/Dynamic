// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_INTERNAL_MINERS_CONTROLLER_H
#define DYNAMIC_INTERNAL_MINERS_CONTROLLER_H

#include <boost/signals2.hpp>

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

class MinerSignals;
class MinersController;

void ConnectMinerSignals(MinersController*);

/**
 * Miner controller for both CPU and GPU threads.
 */
class MinersController
{
public:
    MinersController(const CChainParams& chainparams, CConnman& connman);
    MinersController(MinerContextRef ctx);
    virtual ~MinersController() = default;

    // Starts miners
    void Start();

    // Shuts down all miner threads
    void Shutdown();

    // Gets combined hash rate of GPU and CPU
    int64_t GetHashRate() const;

    // Returns CPU miners thread group
    MinersThreadGroup<CPUMiner>& group_cpu() { return _group_cpu; }

#ifdef ENABLE_GPU
    // Returns GPU miners thread group
    MinersThreadGroup<GPUMiner>& group_gpu()
    {
        return _group_gpu;
    }
#endif // ENABLE_GPU

protected:
    // Returns shared miner context
    MinerContextRef ctx() const { return _ctx; }

    // Starts miner only if can
    void StartIfEnabled();

    // Returns true if enabled, connected and has block.
    bool can_start() const { return _connected && _enable_start && _ctx->shared->block_template(); }

    // Miner signals class
    friend class MinerSignals;

    // Optional miner signals
    // It can be empty when miner is shutdown
    std::shared_ptr<MinerSignals> _signals{nullptr};

    // Miner context
    MinerContextRef _ctx;
    // Miner CPU Thread group
    MinersThreadGroup<CPUMiner> _group_cpu;
#ifdef ENABLE_GPU
    // Miner GPU Thread group
    MinersThreadGroup<GPUMiner> _group_gpu;
#endif // ENABLE_GPU

    // Set to true when at least one node is connected
    bool _connected = false;

    // Set to true when user requested start
    bool _enable_start = false;

    // Time of last transaction signal
    int64_t _last_txn_time = 0;
    // Time of last time block template was created
    int64_t _last_sync_time = 0;
};

class MinerSignals
{
private:
    MinersController* _ctr;

    boost::signals2::scoped_connection _node;
    boost::signals2::scoped_connection _block;
    boost::signals2::scoped_connection _txn;

public:
    MinerSignals(MinersController* _ctr);
    virtual ~MinerSignals() = default;

private:
    // Handles new node connection
    virtual void NotifyNode(const CNode* node);

    // Handles updated blockchain tip
    virtual void NotifyBlock(const CBlockIndex* pindexNew, const CBlockIndex* pindexFork, bool fInitialDownload);

    // Handles new transaction
    virtual void NotifyTransaction(const CTransaction& txn, const CBlockIndex* pindex, int posInBlock);
};

#endif // DYNAMIC_INTERNAL_MINERS_CONTROLLER_H
