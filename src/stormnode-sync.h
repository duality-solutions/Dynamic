// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DARKSILK_STORMNODE_SYNC_H
#define DARKSILK_STORMNODE_SYNC_H

#include "chain.h"
#include "net.h"

#include <univalue.h>

class CStormnodeSync;

static const int STORMNODE_SYNC_FAILED          = -1;
static const int STORMNODE_SYNC_INITIAL         = 0;
static const int STORMNODE_SYNC_SPORKS          = 1;
static const int STORMNODE_SYNC_LIST            = 2;
static const int STORMNODE_SYNC_SNW             = 3;
static const int STORMNODE_SYNC_GOVERNANCE      = 4;
static const int STORMNODE_SYNC_GOVOBJ          = 10;
static const int STORMNODE_SYNC_GOVOBJ_VOTE     = 11;
static const int STORMNODE_SYNC_FINISHED        = 999;

static const int STORMNODE_SYNC_TICK_SECONDS    = 6;
static const int STORMNODE_SYNC_TIMEOUT_SECONDS = 10; // our blocks are 64 seconds, this needs to be fast

static const int STORMNODE_SYNC_ENOUGH_PEERS    = 10;

extern CStormnodeSync stormnodeSync;

//
// CStormnodeSync : Sync stormnode assets in stages
//

class CStormnodeSync
{
private:
    // Keep track of current asset
    int nRequestedStormnodeAssets;
    // Count peers we've requested the asset from
    int nRequestedStormnodeAttempt;

    // Time when current stormnode asset sync started
    int64_t nTimeAssetSyncStarted;

    // Last time when we received some stormnode asset ...
    int64_t nTimeLastStormnodeList;
    int64_t nTimeLastPaymentVote;
    int64_t nTimeLastGovernanceItem;
    // ... or failed
    int64_t nTimeLastFailure;

    // How many times we failed
    int nCountFailures;

    // Keep track of current block index
    const CBlockIndex *pCurrentBlockIndex;

    bool CheckNodeHeight(CNode* pnode, bool fDisconnectStuckNodes = false);
    void Fail();
    void ClearFulfilledRequests();

public:
    CStormnodeSync() { Reset(); }

    void AddedStormnodeList() { nTimeLastStormnodeList = GetTime(); }
    void AddedPaymentVote() { nTimeLastPaymentVote = GetTime(); }
    void AddedGovernanceItem() { nTimeLastGovernanceItem = GetTime(); };

    bool IsFailed() { return nRequestedStormnodeAssets == STORMNODE_SYNC_FAILED; }
    bool IsBlockchainSynced(bool fBlockAccepted = false);
    bool IsStormnodeListSynced() { return nRequestedStormnodeAssets > STORMNODE_SYNC_LIST; }
    bool IsWinnersListSynced() { return nRequestedStormnodeAssets > STORMNODE_SYNC_SNW; }
    bool IsSynced() { return nRequestedStormnodeAssets == STORMNODE_SYNC_FINISHED; }

    int GetAssetID() { return nRequestedStormnodeAssets; }
    int GetAttempt() { return nRequestedStormnodeAttempt; }
    std::string GetAssetName();
    std::string GetSyncStatus();

    void Reset();
    void SwitchToNextAsset();

    void ProcessMessage(CNode* pfrom, std::string& strCommand, CDataStream& vRecv);
    void ProcessTick();

    void UpdatedBlockTip(const CBlockIndex *pindex);
};

#endif // DARKSILK_STORMNODE_SYNC_H
