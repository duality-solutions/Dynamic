// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "psnotificationinterface.h"

#include "chainparams.h"
#include "dynodeman.h"
#include "dynode-payments.h"
#include "dynode-sync.h"
#include "governance.h"
#include "instantsend.h"
#include "privatesend.h"
#ifdef ENABLE_WALLET
#include "privatesend-client.h"
#endif // ENABLE_WALLET

void CPSNotificationInterface::InitializeCurrentBlockTip()
{
    LOCK(cs_main);
    UpdatedBlockTip(chainActive.Tip(), NULL, IsInitialBlockDownload());
}

void CPSNotificationInterface::AcceptedBlockHeader(const CBlockIndex *pindexNew)
{
    dynodeSync.AcceptedBlockHeader(pindexNew);
}

void CPSNotificationInterface::NotifyHeaderTip(const CBlockIndex *pindexNew, bool fInitialDownload)
{
    dynodeSync.NotifyHeaderTip(pindexNew, fInitialDownload, connman);
}

void CPSNotificationInterface::UpdatedBlockTip(const CBlockIndex *pindexNew, const CBlockIndex *pindexFork, bool fInitialDownload)
{
    if (pindexNew == pindexFork) // blocks were disconnected without any new ones
        return;

    dynodeSync.UpdatedBlockTip(pindexNew, fInitialDownload, connman);

    if (fInitialDownload)
        return;

    dnodeman.UpdatedBlockTip(pindexNew);
    CPrivateSend::UpdatedBlockTip(pindexNew);
#ifdef ENABLE_WALLET
    privateSendClient.UpdatedBlockTip(pindexNew);
#endif // ENABLE_WALLET
    instantsend.UpdatedBlockTip(pindexNew);
    dnpayments.UpdatedBlockTip(pindexNew, connman);
    governance.UpdatedBlockTip(pindexNew, connman);
}

void CPSNotificationInterface::SyncTransaction(const CTransaction &tx, const CBlock *pblock)
{
    instantsend.SyncTransaction(tx, pblock);
    CPrivateSend::SyncTransaction(tx, pblock);
}
