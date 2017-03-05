// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "psnotificationinterface.h"

#include "dynodeman.h"
#include "dynode-payments.h"
#include "dynode-sync.h"
#include "governance.h"
#include "instantsend.h"
#include "privatesend.h"

CPSNotificationInterface::CPSNotificationInterface()
{
}

CPSNotificationInterface::~CPSNotificationInterface()
{
}

void CPSNotificationInterface::UpdatedBlockTip(const CBlockIndex *pindex)
{
    dnodeman.UpdatedBlockTip(pindex);
    privateSendPool.UpdatedBlockTip(pindex);
    instantsend.UpdatedBlockTip(pindex);
    dnpayments.UpdatedBlockTip(pindex);
    governance.UpdatedBlockTip(pindex);
    dynodeSync.UpdatedBlockTip(pindex);
}

void CPSNotificationInterface::SyncTransaction(const CTransaction &tx, const CBlock *pblock)
{
    instantsend.SyncTransaction(tx, pblock);
}
