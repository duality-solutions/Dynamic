// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_VALIDATIONINTERFACE_H
#define DYNAMIC_VALIDATIONINTERFACE_H

#include "primitives/transaction.h" // CTransaction(Ref)

#include <boost/shared_ptr.hpp>
#include <boost/signals2/signal.hpp>
#include <memory>

class CBlock;
class CBlockIndex;
struct CBlockLocator;
class CConnman;
class CGovernanceVote;
class CGovernanceObject;
class CMessage;
class CReserveScript;
class CTransaction;
class CValidationInterface;
class CValidationState;
class uint256;

// These functions dispatch to one or all registered wallets

/** Register a wallet to receive updates from core */
void RegisterValidationInterface(CValidationInterface* pwalletIn);
/** Unregister a wallet from core */
void UnregisterValidationInterface(CValidationInterface* pwalletIn);
/** Unregister all wallets from core */
void UnregisterAllValidationInterfaces();

class CValidationInterface
{
protected:
    ~CValidationInterface() = default;
    virtual void AcceptedBlockHeader(const CBlockIndex* pindexNew) {}
    virtual void NotifyHeaderTip(const CBlockIndex* pindexNew, bool fInitialDownload) {}
    virtual void UpdatedBlockTip(const CBlockIndex* pindexNew, const CBlockIndex* pindexFork, bool fInitialDownload) {}
    /** Notifies listeners of a transaction having been added to mempool. */
    virtual void TransactionAddedToMempool(const CTransactionRef &ptxn) {}
    /**
     * Notifies listeners of a block being connected.
     * Provides a vector of transactions evicted from the mempool as a result.
     */
    virtual void BlockConnected(const std::shared_ptr<const CBlock> &block, const CBlockIndex *pindex, const std::vector<CTransactionRef> &txnConflicted) {}
    /** Notifies listeners of a block being disconnected */
    virtual void BlockDisconnected(const std::shared_ptr<const CBlock> &block) {}    virtual void NotifyTransactionLock(const CTransaction& tx) {}
    virtual void NotifyGovernanceVote(const CGovernanceVote& vote) {}
    virtual void NotifyGovernanceObject(const CGovernanceObject& object) {}
    virtual void NotifyInstantSendDoubleSpendAttempt(const CTransaction& currentTx, const CTransaction& previousTx) {}
    virtual void SetBestChain(const CBlockLocator& locator) {}
    virtual bool UpdatedTransaction(const uint256& hash) { return false; }
    virtual void Inventory(const uint256& hash) {}
    virtual void ResendWalletTransactions(int64_t nBestBlockTime, CConnman* connman) {}
    virtual void BlockChecked(const CBlock&, const CValidationState&) {}
    virtual void GetScriptForMining(std::shared_ptr<CReserveScript>&){};
    virtual void ResetRequestCount(const uint256& hash){};
    virtual void NewPoWValidBlock(const CBlockIndex* pindex, const std::shared_ptr<const CBlock>& block) {}
    virtual void NotifyBDAPUpdate(const char* value, const char* action) {}
    virtual void NewAssetMessage(const CMessage &message) {};
    friend void ::RegisterValidationInterface(CValidationInterface*);
    friend void ::UnregisterValidationInterface(CValidationInterface*);
    friend void ::UnregisterAllValidationInterfaces();
};

struct CMainSignals {
    /** Notifies listeners of accepted block header */
    boost::signals2::signal<void(const CBlockIndex*)> AcceptedBlockHeader;
    /** Notifies listeners of updated block header tip */
    boost::signals2::signal<void(const CBlockIndex*, bool fInitialDownload)> NotifyHeaderTip;
    /** Notifies listeners of updated block chain tip */
    boost::signals2::signal<void(const CBlockIndex*, const CBlockIndex*, bool fInitialDownload)> UpdatedBlockTip;
    /** A posInBlock value for SyncTransaction calls for tranactions not
     * included in connected blocks such as transactions removed from mempool,
     * accepted to mempool or appearing in disconnected blocks.*/
    static const int SYNC_TRANSACTION_NOT_IN_BLOCK = -1;
    boost::signals2::signal<void (const CTransactionRef &)> TransactionAddedToMempool;
    boost::signals2::signal<void (const std::shared_ptr<const CBlock> &, const CBlockIndex *pindex, const std::vector<CTransactionRef>&)> BlockConnected;
    boost::signals2::signal<void (const std::shared_ptr<const CBlock> &)> BlockDisconnected;
    /** Notifies listeners of an updated transaction lock without new data. */
    boost::signals2::signal<void(const CTransaction&)> NotifyTransactionLock;
    /** Notifies listeners of a new governance vote. */
    boost::signals2::signal<void(const CGovernanceVote&)> NotifyGovernanceVote;
    /** Notifies listeners of a new governance object. */
    boost::signals2::signal<void(const CGovernanceObject&)> NotifyGovernanceObject;
    /** Notifies listeners of a attempted InstantSend double spend*/
    boost::signals2::signal<void(const CTransaction& currentTx, const CTransaction& previousTx)> NotifyInstantSendDoubleSpendAttempt;
    /** Notifies listeners of an updated transaction without new data (for now: a coinbase potentially becoming visible). */
    boost::signals2::signal<bool(const uint256&)> UpdatedTransaction;
    /** Notifies listeners of a new active block chain. */
    boost::signals2::signal<void(const CBlockLocator&)> SetBestChain;
    /** Notifies listeners about an inventory item being seen on the network. */
    boost::signals2::signal<void(const uint256&)> Inventory;
    /** Tells listeners to broadcast their data. */
    boost::signals2::signal<void(int64_t nBestBlockTime, CConnman* connman)> Broadcast;
    /** Notifies listeners of a block validation result */
    boost::signals2::signal<void(const CBlock&, const CValidationState&)> BlockChecked;
    /** Notifies listeners that a key for mining is required (coinbase) */
    boost::signals2::signal<void(std::shared_ptr<CReserveScript>&)> ScriptForMining;
    /** Notifies listeners that a block has been successfully mined */
    boost::signals2::signal<void(const uint256&)> BlockFound;
    /**
     * Notifies listeners that a block which builds directly on our current tip
     * has been received and connected to the headers tree, though not validated yet */
    boost::signals2::signal<void(const CBlockIndex*, const std::shared_ptr<const CBlock>&)> NewPoWValidBlock;
    /** Notifies listeners of an updated BDAP action */
    boost::signals2::signal<void(const char* value, const char* action)> NotifyBDAPUpdate;
/* ASSET START */
    boost::signals2::signal<void (const CMessage &)> NewAssetMessage;
    boost::signals2::signal<void (const std::string &)> AssetInventory;
/* ASSET END  */
};

CMainSignals& GetMainSignals();

#endif // DYNAMIC_VALIDATIONINTERFACE_H
