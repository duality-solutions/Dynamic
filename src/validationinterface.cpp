// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "validationinterface.h"

#include <boost/bind/bind.hpp>

static CMainSignals g_signals;

CMainSignals& GetMainSignals()
{
    return g_signals;
}

void RegisterValidationInterface(CValidationInterface* pwalletIn)
{
    g_signals.AcceptedBlockHeader.connect(boost::bind(&CValidationInterface::AcceptedBlockHeader, pwalletIn, boost::placeholders::_1));
    g_signals.UpdatedBlockTip.connect(boost::bind(&CValidationInterface::UpdatedBlockTip, pwalletIn, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3));
    g_signals.SyncTransaction.connect(boost::bind(&CValidationInterface::SyncTransaction, pwalletIn, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3));
    g_signals.NotifyTransactionLock.connect(boost::bind(&CValidationInterface::NotifyTransactionLock, pwalletIn, boost::placeholders::_1));
    g_signals.UpdatedTransaction.connect(boost::bind(&CValidationInterface::UpdatedTransaction, pwalletIn, boost::placeholders::_1));
    g_signals.SetBestChain.connect(boost::bind(&CValidationInterface::SetBestChain, pwalletIn, boost::placeholders::_1));
    g_signals.Inventory.connect(boost::bind(&CValidationInterface::Inventory, pwalletIn, boost::placeholders::_1));
    g_signals.Broadcast.connect(boost::bind(&CValidationInterface::ResendWalletTransactions, pwalletIn, boost::placeholders::_1, boost::placeholders::_2));
    g_signals.BlockChecked.connect(boost::bind(&CValidationInterface::BlockChecked, pwalletIn, boost::placeholders::_1, boost::placeholders::_2));
    g_signals.ScriptForMining.connect(boost::bind(&CValidationInterface::GetScriptForMining, pwalletIn, boost::placeholders::_1));
    g_signals.BlockFound.connect(boost::bind(&CValidationInterface::ResetRequestCount, pwalletIn, boost::placeholders::_1));
    g_signals.NewPoWValidBlock.connect(boost::bind(&CValidationInterface::NewPoWValidBlock, pwalletIn, boost::placeholders::_1, boost::placeholders::_2));
    g_signals.NotifyGovernanceObject.connect(boost::bind(&CValidationInterface::NotifyGovernanceObject, pwalletIn, boost::placeholders::_1));
    g_signals.NotifyGovernanceVote.connect(boost::bind(&CValidationInterface::NotifyGovernanceVote, pwalletIn, boost::placeholders::_1));
    g_signals.NotifyBDAPUpdate.connect(boost::bind(&CValidationInterface::NotifyBDAPUpdate, pwalletIn, boost::placeholders::_1, boost::placeholders::_2));
    g_signals.NotifyInstantSendDoubleSpendAttempt.connect(boost::bind(&CValidationInterface::NotifyInstantSendDoubleSpendAttempt, pwalletIn, boost::placeholders::_1, boost::placeholders::_2));
}

void UnregisterValidationInterface(CValidationInterface* pwalletIn)
{
    g_signals.BlockFound.disconnect(boost::bind(&CValidationInterface::ResetRequestCount, pwalletIn, boost::placeholders::_1));
    g_signals.ScriptForMining.disconnect(boost::bind(&CValidationInterface::GetScriptForMining, pwalletIn, boost::placeholders::_1));
    g_signals.BlockChecked.disconnect(boost::bind(&CValidationInterface::BlockChecked, pwalletIn, boost::placeholders::_1, boost::placeholders::_2));
    g_signals.Broadcast.disconnect(boost::bind(&CValidationInterface::ResendWalletTransactions, pwalletIn, boost::placeholders::_1, boost::placeholders::_2));
    g_signals.Inventory.disconnect(boost::bind(&CValidationInterface::Inventory, pwalletIn, boost::placeholders::_1));
    g_signals.SetBestChain.disconnect(boost::bind(&CValidationInterface::SetBestChain, pwalletIn, boost::placeholders::_1));
    g_signals.UpdatedTransaction.disconnect(boost::bind(&CValidationInterface::UpdatedTransaction, pwalletIn, boost::placeholders::_1));
    g_signals.NotifyTransactionLock.disconnect(boost::bind(&CValidationInterface::NotifyTransactionLock, pwalletIn, boost::placeholders::_1));
    g_signals.SyncTransaction.disconnect(boost::bind(&CValidationInterface::SyncTransaction, pwalletIn, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3));
    g_signals.UpdatedBlockTip.disconnect(boost::bind(&CValidationInterface::UpdatedBlockTip, pwalletIn, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3));
    g_signals.NewPoWValidBlock.disconnect(boost::bind(&CValidationInterface::NewPoWValidBlock, pwalletIn, boost::placeholders::_1, boost::placeholders::_2));
    g_signals.NotifyHeaderTip.disconnect(boost::bind(&CValidationInterface::NotifyHeaderTip, pwalletIn, boost::placeholders::_1, boost::placeholders::_2));
    g_signals.AcceptedBlockHeader.disconnect(boost::bind(&CValidationInterface::AcceptedBlockHeader, pwalletIn, boost::placeholders::_1));
    g_signals.NotifyGovernanceObject.disconnect(boost::bind(&CValidationInterface::NotifyGovernanceObject, pwalletIn, boost::placeholders::_1));
    g_signals.NotifyGovernanceVote.disconnect(boost::bind(&CValidationInterface::NotifyGovernanceVote, pwalletIn, boost::placeholders::_1));
    g_signals.NotifyBDAPUpdate.disconnect(boost::bind(&CValidationInterface::NotifyBDAPUpdate, pwalletIn, boost::placeholders::_1, boost::placeholders::_2));
    g_signals.NotifyInstantSendDoubleSpendAttempt.disconnect(boost::bind(&CValidationInterface::NotifyInstantSendDoubleSpendAttempt, pwalletIn, boost::placeholders::_1, boost::placeholders::_2));
}

void UnregisterAllValidationInterfaces()
{
    g_signals.BlockFound.disconnect_all_slots();
    g_signals.ScriptForMining.disconnect_all_slots();
    g_signals.BlockChecked.disconnect_all_slots();
    g_signals.Broadcast.disconnect_all_slots();
    g_signals.Inventory.disconnect_all_slots();
    g_signals.SetBestChain.disconnect_all_slots();
    g_signals.UpdatedTransaction.disconnect_all_slots();
    g_signals.NotifyTransactionLock.disconnect_all_slots();
    g_signals.SyncTransaction.disconnect_all_slots();
    g_signals.UpdatedBlockTip.disconnect_all_slots();
    g_signals.NewPoWValidBlock.disconnect_all_slots();
    g_signals.NotifyHeaderTip.disconnect_all_slots();
    g_signals.AcceptedBlockHeader.disconnect_all_slots();
    g_signals.NotifyGovernanceObject.disconnect_all_slots();
    g_signals.NotifyGovernanceVote.disconnect_all_slots();
    g_signals.NotifyBDAPUpdate.disconnect_all_slots();
    g_signals.NotifyInstantSendDoubleSpendAttempt.disconnect_all_slots();
}