// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DYNODE_SYNC_H
#define DYNAMIC_DYNODE_SYNC_H

#include "chain.h"
#include "net.h"

#include <univalue.h>

class CDynodeSync;

static const int DYNODE_SYNC_FAILED = -1;
static const int DYNODE_SYNC_INITIAL = 0; // sync just started, was reset recently or still in IDB
static const int DYNODE_SYNC_WAITING = 1; // waiting after initial to see if we can get more headers/blocks
static const int DYNODE_SYNC_LIST = 2;
static const int DYNODE_SYNC_DNW = 3;
static const int DYNODE_SYNC_GOVERNANCE = 4;
static const int DYNODE_SYNC_GOVOBJ = 10;
static const int DYNODE_SYNC_GOVOBJ_VOTE = 11;
static const int DYNODE_SYNC_FINISHED = 999;

static const int DYNODE_SYNC_TICK_SECONDS = 6;
static const int DYNODE_SYNC_TIMEOUT_SECONDS = 25;

static const int DYNODE_SYNC_ENOUGH_PEERS = 10;

extern CDynodeSync dynodeSync;

//
// CDynodeSync : Sync Dynode assets in stages
//

class CDynodeSync
{
private:
    // Keep track of current asset
    int nRequestedDynodeAssets;
    // Count peers we've requested the asset from
    int nRequestedDynodeAttempt;

    // Time when current Dynode asset sync started
    int64_t nTimeAssetSyncStarted;

    // ... last bumped
    int64_t nTimeLastBumped;

    // ... or failed
    int64_t nTimeLastFailure;

    void Fail();

public:
    CDynodeSync() { Reset(); }

    void SendGovernanceSyncRequest(CNode* pnode, CConnman& connman);

    bool IsFailed() { return nRequestedDynodeAssets == DYNODE_SYNC_FAILED; }
    bool IsBlockchainSynced() { return nRequestedDynodeAssets > DYNODE_SYNC_WAITING; }
    bool IsDynodeListSynced() { return nRequestedDynodeAssets > DYNODE_SYNC_LIST; }
    bool IsWinnersListSynced() { return nRequestedDynodeAssets > DYNODE_SYNC_DNW; }
    bool IsSynced() { return nRequestedDynodeAssets == DYNODE_SYNC_FINISHED; }

    int GetAssetID() { return nRequestedDynodeAssets; }
    int GetAttempt() { return nRequestedDynodeAttempt; }
    void BumpAssetLastTime(const std::string strFuncName);
    int64_t GetAssetStartTime() { return nTimeAssetSyncStarted; }
    std::string GetAssetName();
    std::string GetSyncStatus();

    void Reset();
    void SwitchToNextAsset(CConnman& connman);

    void ProcessMessage(CNode* pfrom, const std::string& strCommand, CDataStream& vRecv);
    void ProcessTick(CConnman& connman);

    void AcceptedBlockHeader(const CBlockIndex* pindexNew);
    void NotifyHeaderTip(const CBlockIndex* pindexNew, bool fInitialDownload, CConnman& connman);
    void UpdatedBlockTip(const CBlockIndex* pindexNew, bool fInitialDownload, CConnman& connman);

    void DoMaintenance(CConnman& connman);
    double SyncProgress();

};

#endif // DYNAMIC_DYNODE_SYNC_H
