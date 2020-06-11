// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_PSNOTIFICATIONINTERFACE_H
#define DYNAMIC_PSNOTIFICATIONINTERFACE_H

#include "validationinterface.h"

class CPSNotificationInterface : public CValidationInterface
{
public:
    CPSNotificationInterface(CConnman& connmanIn) : connman(connmanIn) {}
    virtual ~CPSNotificationInterface() = default;

    // a small helper to initialize current block height in sub-modules on startup
    void InitializeCurrentBlockTip();

protected:
    // CValidationInterface
    void AcceptedBlockHeader(const CBlockIndex* pindexNew) override;
    void NotifyHeaderTip(const CBlockIndex* pindexNew, bool fInitialDownload) override;
    void UpdatedBlockTip(const CBlockIndex* pindexNew, const CBlockIndex* pindexFork, bool fInitialDownload) override;
    void TransactionAddedToMempool(const CTransactionRef &ptxn) {};
    void BlockConnected(const std::shared_ptr<const CBlock> &block, const CBlockIndex *pindex, const std::vector<CTransactionRef> &txnConflicted) {};
    void BlockDisconnected(const std::shared_ptr<const CBlock> &block) {}    virtual void NotifyTransactionLock(const CTransaction& tx) {};
private:
    CConnman& connman;
};

#endif // DYNAMIC_PSNOTIFICATIONINTERFACE_H
