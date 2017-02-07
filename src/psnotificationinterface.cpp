// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "psnotificationinterface.h"
#include "privatesend.h"
#include "instantsend.h"
#include "governance.h"
#include "stormnodeman.h"
#include "stormnode-payments.h"
#include "stormnode-sync.h"

CPSNotificationInterface::CPSNotificationInterface()
{
}

CPSNotificationInterface::~CPSNotificationInterface()
{
}

void CPSNotificationInterface::UpdatedBlockTip(const CBlockIndex *pindex)
{
    snodeman.UpdatedBlockTip(pindex);
    privateSendPool.UpdatedBlockTip(pindex);
    instantsend.UpdatedBlockTip(pindex);
    snpayments.UpdatedBlockTip(pindex);
    governance.UpdatedBlockTip(pindex);
    stormnodeSync.UpdatedBlockTip(pindex);
}

void CPSNotificationInterface::SyncTransaction(const CTransaction &tx, const CBlock *pblock)
{
    instantsend.SyncTransaction(tx, pblock);
}
