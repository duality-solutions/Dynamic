// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "validationinterface.h"

#include "init.h"
#include "primitives/block.h"
#include "scheduler.h"
#include "sync.h"
#include "util.h"

#include <list>
#include <atomic>

#include <boost/shared_ptr.hpp>
#include <boost/signals2/signal.hpp>

struct MainSignalsInstance {
    /** Notifies listeners of accepted block header */
    boost::signals2::signal<void(const CBlockIndex*)> AcceptedBlockHeader;
    /** Notifies listeners of updated block header tip */
    boost::signals2::signal<void(const CBlockIndex*, bool fInitialDownload)> NotifyHeaderTip;
    /** Notifies listeners of updated block chain tip */
    boost::signals2::signal<void(const CBlockIndex*, const CBlockIndex*, bool fInitialDownload)> UpdatedBlockTip;
    /** A posInBlock value for SyncTransaction calls for transactions not
     * included in connected blocks such as transactions removed from mempool,
     * accepted to mempool or appearing in disconnected blocks.*/
    boost::signals2::signal<void(int NotInBlock)> SyncTransactionNotInBlock;
    /** Notifies listeners of updated transaction data (transaction, and
     * optionally the block it is found in). Called with block data when
     * transaction is included in a connected block, and without block data when
     * transaction was accepted to mempool, removed from mempool (only when
     * removal was due to conflict from connected block), or appeared in a
     * disconnected block.*/
    boost::signals2::signal<void(const CTransaction&, const CBlockIndex* pindex, int posInBlock)> SyncTransaction;
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

    // We are not allowed to assume the scheduler only runs in one thread,
    // but must ensure all callbacks happen in-order, so we end up creating
    // our own queue here :(
    SingleThreadedSchedulerClient m_schedulerClient;

    explicit MainSignalsInstance(CScheduler *pscheduler) : m_schedulerClient(pscheduler) {}
};

static CMainSignals g_signals;

void CMainSignals::RegisterBackgroundSignalScheduler(CScheduler& scheduler) {
    assert(!m_internals);
    m_internals.reset(new MainSignalsInstance(&scheduler));
}

void CMainSignals::UnregisterBackgroundSignalScheduler() {
    m_internals.reset(nullptr);
}

void CMainSignals::FlushBackgroundCallbacks() {
    m_internals->m_schedulerClient.EmptyQueue();
}

CMainSignals& GetMainSignals()
{
    return g_signals;
}

void RegisterValidationInterface(CValidationInterface* pwalletIn)
{
    g_signals.AcceptedBlockHeader.connect(boost::bind(&CValidationInterface::AcceptedBlockHeader, pwalletIn, _1));
    g_signals.UpdatedBlockTip.connect(boost::bind(&CValidationInterface::UpdatedBlockTip, pwalletIn, _1, _2, _3));
    g_signals.BlockConnected.connect(boost::bind(&CValidationInterface::BlockConnected, pwalletIn, _1, _2, _3));
    g_signals.NotifyTransactionLock.connect(boost::bind(&CValidationInterface::NotifyTransactionLock, pwalletIn, _1));
    g_signals.NotifyGovernanceVote.connect(boost::bind(&CValidationInterface::NotifyGovernanceVote, pwalletIn, _1));
    g_signals.NotifyGovernanceObject.connect(boost::bind(&CValidationInterface::NotifyGovernanceObject, pwalletIn, _1));
    g_signals.NotifyInstantSendDoubleSpendAttempt.connect(boost::bind(&CValidationInterface::NotifyInstantSendDoubleSpendAttempt, pwalletIn, _1, _2));
    g_signals.UpdatedTransaction.connect(boost::bind(&CValidationInterface::UpdatedTransaction, pwalletIn, _1));
    g_signals.SetBestChain.connect(boost::bind(&CValidationInterface::SetBestChain, pwalletIn, _1));
    g_signals.Inventory.connect(boost::bind(&CValidationInterface::Inventory, pwalletIn, _1));
    g_signals.Broadcast.connect(boost::bind(&CValidationInterface::ResendWalletTransactions, pwalletIn, _1, _2));
    g_signals.BlockChecked.connect(boost::bind(&CValidationInterface::BlockChecked, pwalletIn, _1, _2));
    g_signals.ScriptForMining.connect(boost::bind(&CValidationInterface::GetScriptForMining, pwalletIn, _1));
    g_signals.BlockFound.connect(boost::bind(&CValidationInterface::ResetRequestCount, pwalletIn, _1));
    g_signals.NewPoWValidBlock.connect(boost::bind(&CValidationInterface::NewPoWValidBlock, pwalletIn, _1, _2));
    g_signals.NotifyBDAPUpdate.connect(boost::bind(&CValidationInterface::NotifyBDAPUpdate, pwalletIn, _1, _2));
/* ASSET START */
    g_signals.NewAssetMessage.connect(boost::bind(&CValidationInterface::NewAssetMessage, pwalletIn, _1));
/* ASSET END */
}

void UnregisterValidationInterface(CValidationInterface* pwalletIn)
{
    g_signals.AcceptedBlockHeader.disconnect(boost::bind(&CValidationInterface::AcceptedBlockHeader, pwalletIn, _1));
    g_signals.NotifyHeaderTip.disconnect(boost::bind(&CValidationInterface::NotifyHeaderTip, pwalletIn, _1, _2));
    g_signals.UpdatedBlockTip.disconnect(boost::bind(&CValidationInterface::UpdatedBlockTip, pwalletIn, _1, _2, _3));
    g_signals.BlockConnected.disconnect(boost::bind(&CValidationInterface::BlockConnected, pwalletIn, _1, _2, _3));
    g_signals.NotifyTransactionLock.disconnect(boost::bind(&CValidationInterface::NotifyTransactionLock, pwalletIn, _1));
    g_signals.NotifyGovernanceVote.disconnect(boost::bind(&CValidationInterface::NotifyGovernanceVote, pwalletIn, _1));
    g_signals.NotifyGovernanceObject.disconnect(boost::bind(&CValidationInterface::NotifyGovernanceObject, pwalletIn, _1));
    g_signals.NotifyInstantSendDoubleSpendAttempt.disconnect(boost::bind(&CValidationInterface::NotifyInstantSendDoubleSpendAttempt, pwalletIn, _1, _2));
    g_signals.UpdatedTransaction.disconnect(boost::bind(&CValidationInterface::UpdatedTransaction, pwalletIn, _1));
    g_signals.SetBestChain.disconnect(boost::bind(&CValidationInterface::SetBestChain, pwalletIn, _1));
    g_signals.Inventory.disconnect(boost::bind(&CValidationInterface::Inventory, pwalletIn, _1));
    g_signals.Broadcast.disconnect(boost::bind(&CValidationInterface::ResendWalletTransactions, pwalletIn, _1, _2));
    g_signals.BlockChecked.disconnect(boost::bind(&CValidationInterface::BlockChecked, pwalletIn, _1, _2));
    g_signals.ScriptForMining.disconnect(boost::bind(&CValidationInterface::GetScriptForMining, pwalletIn, _1));
    g_signals.BlockFound.disconnect(boost::bind(&CValidationInterface::ResetRequestCount, pwalletIn, _1));
    g_signals.NewPoWValidBlock.disconnect(boost::bind(&CValidationInterface::NewPoWValidBlock, pwalletIn, _1, _2));
    g_signals.NotifyBDAPUpdate.disconnect(boost::bind(&CValidationInterface::NotifyBDAPUpdate, pwalletIn, _1, _2));
/* ASSET START */
    g_signals.NewAssetMessage.disconnect(boost::bind(&CValidationInterface::NewAssetMessage, pwalletIn, _1));
/* ASSET END */
}

void UnregisterAllValidationInterfaces()
{
    g_signals.AcceptedBlockHeader.disconnect_all_slots();
    g_signals.NotifyHeaderTip.disconnect_all_slots();
    g_signals.UpdatedBlockTip.disconnect_all_slots();
    g_signals.BlockConnected.disconnect_all_slots();
    g_signals.NotifyTransactionLock.disconnect_all_slots();
    g_signals.NotifyGovernanceVote.disconnect_all_slots();
    g_signals.NotifyGovernanceObject.disconnect_all_slots();
    g_signals.NotifyInstantSendDoubleSpendAttempt.disconnect_all_slots();
    g_signals.UpdatedTransaction.disconnect_all_slots();
    g_signals.SetBestChain.disconnect_all_slots();
    g_signals.Inventory.disconnect_all_slots();
    g_signals.Broadcast.disconnect_all_slots();
    g_signals.BlockChecked.disconnect_all_slots();
    g_signals.ScriptForMining.disconnect_all_slots();
    g_signals.BlockFound.disconnect_all_slots();
    g_signals.NewPoWValidBlock.disconnect_all_slots();
    g_signals.NotifyBDAPUpdate.disconnect_all_slots();
/* ASSET START */
    g_signals.NewAssetMessage.disconnect_all_slots();
/* ASSET END */
}

void CMainSignals::UpdatedBlockTip(const CBlockIndex *pindexNew, const CBlockIndex *pindexFork, bool fInitialDownload) {
    m_internals->UpdatedBlockTip(pindexNew, pindexFork, fInitialDownload);
}

void CMainSignals::SyncTransactionNotInBlock(int NotInBlock) {
    m_internals->SyncTransactionNotInBlock(NotInBlock)
}

void CMainSignals::SyncTransaction(const CTransaction &tx, const CBlockIndex* pindex, int posInBlock) {
    m_internals->SyncTransaction(tx)
}

void CMainSignals::TransactionAddedToMempool(const CTransactionRef &ptx) {
    m_internals->TransactionAddedToMempool(ptx);
}

void CMainSignals::BlockConnected(const std::shared_ptr<const CBlock> &pblock, const CBlockIndex *pindex, const std::vector<CTransactionRef>& vtxConflicted) {
    m_internals->BlockConnected(pblock, pindex, vtxConflicted);
}

void CMainSignals::NotifyTransactionLock(const CTransaction &tx) {
    m_internals->NotifyTransactionLock(tx);
}

void CMainSignals::BlockDisconnected(const std::shared_ptr<const CBlock> &pblock) {
    m_internals->BlockDisconnected(pblock);
}

void CMainSignals::SetBestChain(const CBlockLocator &locator) {
    m_internals->SetBestChain(locator);
}

void CMainSignals::SetBestChain(const CBlockLocator &locator) {
    m_internals->SetBestChain(locator);
}

void CMainSignals::SetBestChain(const CBlockLocator &locator) {
    m_internals->SetBestChain(locator);
}

void CMainSignals::SetBestChain(const CBlockLocator &locator) {
    m_internals->SetBestChain(locator);
}

void CMainSignals::Inventory(const uint256 &hash) {
    m_internals->Inventory(hash);
}

void CMainSignals::Broadcast(int64_t nBestBlockTime, CConnman* connman) {
    m_internals->Broadcast(nBestBlockTime, connman);
}

void CMainSignals::BlockChecked(const CBlock& block, const CValidationState& state) {
    m_internals->BlockChecked(block, state);
}

void CMainSignals::ScriptForMining(std::shared_ptr<CReserveScript> &ScriptForMining) {
    m_internals->ScriptForMining(ScriptForMining);
}

void CMainSignals::NewPoWValidBlock(const CBlockIndex *pindex, const std::shared_ptr<const CBlock> &block) {
    m_internals->NewPoWValidBlock(pindex, block);
}

void CMainSignals::NotifyBDAPUpdate(const char* value, const char* action) {
    m_internals->NotifyBDAPUpdate(value, action);
}

void CMainSignals::NewAssetMessage(const CMessage& message) {
    m_internals->NewAssetMessage(message);
}