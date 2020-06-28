// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_VALIDATIONINTERFACE_H
#define DYNAMIC_VALIDATIONINTERFACE_H

#include "primitives/transaction.h" // CTransaction(Ref)

#include <memory>

class CBlock;
class CBlockIndex;
struct CBlockLocator;
class CConnman;
class CGovernanceVote;
class CGovernanceObject;
class CMessage;
class CReserveScript;
class CScheduler;
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
    /**
    * Protected destructor so that instances can only be deleted by derived
    * classes. If that restriction is no longer desired, this should be made
    * public and virtual.
    */
    ~CValidationInterface() = default;
    virtual void AcceptedBlockHeader(const CBlockIndex* pindexNew) {}
    virtual void NotifyHeaderTip(const CBlockIndex* pindexNew, bool fInitialDownload) {}
    /** Notifies listeners of updated blockchain tip */
    virtual void UpdatedBlockTip(const CBlockIndex* pindexNew, const CBlockIndex* pindexFork, bool fInitialDownload) {}
    virtual void SyncTransactionNotInBlock(int NotInBlock) {}
    virtual void SyncTransaction(const CTransaction&, const CBlockIndex* pindex, int posInBlock) {}
    /** Notifies listeners of a transaction having been added to mempool. */
    virtual void TransactionAddedToMempool(const CTransactionRef &ptxn) {}
    /**
     * Notifies listeners of a block being connected.
     * Provides a vector of transactions evicted from the mempool as a result.
     */
    virtual void BlockConnected(const std::shared_ptr<const CBlock> &pblock, const CBlockIndex *pindex, const std::vector<CTransactionRef>& vtxConflicted) {}
    /** Notifies listeners of a block being disconnected */
    virtual void BlockDisconnected(const std::shared_ptr<const CBlock> &pblock) {}    virtual void NotifyTransactionLock(const CTransaction& tx) {}
    virtual void NotifyGovernanceVote(const CGovernanceVote& vote) {}
    virtual void NotifyGovernanceObject(const CGovernanceObject& object) {}
    virtual void NotifyInstantSendDoubleSpendAttempt(const CTransaction& currentTx, const CTransaction& previousTx) {}
    virtual void SetBestChain(const CBlockLocator& locator) {}
    virtual bool UpdatedTransaction(const uint256& hash) { return false; }
    virtual void Inventory(const uint256& hash) {}
    virtual void ResendWalletTransactions(int64_t nBestBlockTime, CConnman* connman) {}
    virtual void BlockChecked(const CBlock&, const CValidationState&) {}
    virtual void GetScriptForMining(std::shared_ptr<CReserveScript>&){}
    virtual void ResetRequestCount(const uint256& hash){}
    virtual void NewPoWValidBlock(const CBlockIndex* pindex, const std::shared_ptr<const CBlock>& block) {}
    virtual void NotifyBDAPUpdate(const char* value, const char* action) {}
    virtual void NewAssetMessage(const CMessage &message) {}

    friend void ::RegisterValidationInterface(CValidationInterface*);
    friend void ::UnregisterValidationInterface(CValidationInterface*);
    friend void ::UnregisterAllValidationInterfaces();
};

struct MainSignalsInstance;

class CMainSignals {
private:
    std::unique_ptr<MainSignalsInstance> m_internals;

    friend void ::RegisterValidationInterface(CValidationInterface*);
    friend void ::UnregisterValidationInterface(CValidationInterface*);
    friend void ::UnregisterAllValidationInterfaces();

public:
    /** Register a CScheduler to give callbacks which should run in the background (may only be called once) */
    void RegisterBackgroundSignalScheduler(CScheduler& scheduler);
    /** Unregister a CScheduler to give callbacks which should run in the background - these callbacks will now be dropped! */
    void UnregisterBackgroundSignalScheduler();
    /** Call any remaining callbacks on the calling thread */
    void FlushBackgroundCallbacks();

    void UpdatedBlockTip(const CBlockIndex *, const CBlockIndex *, bool fInitialDownload);
    void SyncTransactionNotInBlock(int NotInBlock);
    void SyncTransaction(const CTransaction &, const CBlockIndex *, int posInBlock);
    void TransactionAddedToMempool(const CTransactionRef &);
    void BlockConnected(const std::shared_ptr<const CBlock> &, const CBlockIndex *pindex, const std::vector<CTransactionRef> &);
    void NotifyTransactionLock(const CTransaction &);
    void BlockDisconnected(const std::shared_ptr<const CBlock> &);
    void NotifyGovernanceVote(const CGovernanceVote &);
    void NotifyGovernanceObject(const CGovernanceObject &);
    void NotifyInstantSendDoubleSpendAttempt(const CTransaction &, const CTransaction &);
    void SetBestChain(const CBlockLocator &);
    void Inventory(const uint256 &);
    void Broadcast(int64_t nBestBlockTime, CConnman* connman);
    void BlockChecked(const CBlock&, const CValidationState&);
    void ScriptForMining(std::shared_ptr<CReserveScript> &);
    void NewPoWValidBlock(const CBlockIndex *, const std::shared_ptr<const CBlock>&);
    void NotifyBDAPUpdate(const char*, const char*);
    void NewAssetMessage(const CMessage&);
};

CMainSignals& GetMainSignals();

#endif // DYNAMIC_VALIDATIONINTERFACE_H
