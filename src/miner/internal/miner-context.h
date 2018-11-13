// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_INTERNAL_MINER_CONTEXT_H
#define DYNAMIC_INTERNAL_MINER_CONTEXT_H

#include "miner/internal/hash-rate-counter.h"

#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

#include <atomic>

class CBlock;
class CChainParams;
class CConnman;
class CBlockIndex;
class CReserveScript;
struct CBlockTemplate;

class MinerBase;
class MinerContext;
class MinersController;

/** Miner context shared_ptr */
using MinerContextRef = std::shared_ptr<MinerContext>;

struct MinerSharedContext {
public:
    const CChainParams& chainparams;
    CConnman& connman;

    MinerSharedContext(const CChainParams& chainparams_, CConnman& connman_)
        : chainparams(chainparams_), connman(connman_){};

protected:
    friend class MinerContext;

    // recreates miners block template
    void RecreateBlock();
    void InitializeCoinbaseScript();

    CBlockIndex* tip() const { return _chain_tip; }
    uint64_t block_flag() const { return _block_flag; }
    std::shared_ptr<CReserveScript> coinbase_script() const { return _coinbase_script; }
    bool has_block() const { return _block_template != nullptr; }

    std::shared_ptr<CBlockTemplate> block_template()
    {
        boost::shared_lock<boost::shared_mutex> guard(_mutex);
        return _block_template;
    }

private:
    // current block chain tip
    std::atomic<CBlockIndex*> _chain_tip{nullptr};
    // atomic flag incremented on recreated block
    std::atomic<std::uint64_t> _block_flag{0};
    // shared block template for miners
    std::shared_ptr<CBlockTemplate> _block_template{nullptr};
    // coinbase script for all miners
    std::shared_ptr<CReserveScript> _coinbase_script{nullptr};
    // mutex protecting multiple threads recreating block
    mutable boost::shared_mutex _mutex;
};

using MinerSharedContextRef = std::shared_ptr<MinerSharedContext>;

/**
 * Miner context.
 */
class MinerContext
{
public:
    HashRateCounterRef counter;
    MinerSharedContextRef shared;

    MinerContext(const CChainParams& chainparams_, CConnman& connman_)
        : counter(HashRateCounter::Make()), shared(std::make_shared<MinerSharedContext>(chainparams_, connman_)){};

    MinerContext(MinerSharedContextRef shared_, HashRateCounterRef counter_)
        : counter(counter_), shared(shared_){};

    /* Constructs child context */
    explicit MinerContext(const MinerContext* ctx_)
        : MinerContext(ctx_->shared, ctx_->counter->MakeChild()){};

    /** Creates child context for group or miner */
    MinerContextRef MakeChild() const { return std::make_shared<MinerContext>(this); }

    /** Recreates block for all miners */
    void RecreateBlock() { shared->RecreateBlock(); }

    /** Initializes coinbase script for all miners */
    void InitializeCoinbaseScript() { shared->InitializeCoinbaseScript(); }

    CConnman& connman() const { return shared->connman; }
    const CChainParams& chainparams() const { return shared->chainparams; }
    CBlockIndex* tip() const { return shared->tip(); }
    uint64_t block_flag() const { return shared->block_flag(); }
    std::shared_ptr<CBlockTemplate> block_template() const { return shared->block_template(); }
    std::shared_ptr<CReserveScript> coinbase_script() const { return shared->coinbase_script(); }
    bool has_block() const { return shared->has_block(); }
};

#endif // DYNAMIC_INTERNAL_MINER_CONTEXT_H
