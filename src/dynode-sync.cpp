// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dynode-sync.h"

#include "activedynode.h"
#include "checkpoints.h"
#include "governance.h"
#include "dynode.h"
#include "dynode-payments.h"
#include "dynodeman.h"
#include "main.h"
#include "netfulfilledman.h"
#include "spork.h"
#include "util.h"

class CDynodeSync;
CDynodeSync dynodeSync;

bool CDynodeSync::CheckNodeHeight(CNode* pnode, bool fDisconnectStuckNodes)
{
    CNodeStateStats stats;
    if(!GetNodeStateStats(pnode->id, stats) || stats.nCommonHeight == -1 || stats.nSyncHeight == -1) return false; // not enough info about this peer

    // Check blocks and headers, allow a small error margin of 1 block
    if(pCurrentBlockIndex->nHeight - 1 > stats.nCommonHeight) {
        // This peer probably stuck, don't sync any additional data from it
        if(fDisconnectStuckNodes) {
            // Disconnect to free this connection slot for another peer.
            pnode->fDisconnect = true;
            LogPrintf("CDynodeSync::CheckNodeHeight -- disconnecting from stuck peer, nHeight=%d, nCommonHeight=%d, peer=%d\n",
                        pCurrentBlockIndex->nHeight, stats.nCommonHeight, pnode->id);
        } else {
            LogPrintf("CDynodeSync::CheckNodeHeight -- skipping stuck peer, nHeight=%d, nCommonHeight=%d, peer=%d\n",
                        pCurrentBlockIndex->nHeight, stats.nCommonHeight, pnode->id);
        }
        return false;
    }
    else if(pCurrentBlockIndex->nHeight < stats.nSyncHeight - 1) {
        // This peer announced more headers than we have blocks currently
        LogPrintf("CDynodeSync::CheckNodeHeight -- skipping peer, who announced more headers than we have blocks currently, nHeight=%d, nSyncHeight=%d, peer=%d\n",
                    pCurrentBlockIndex->nHeight, stats.nSyncHeight, pnode->id);
        return false;
    }

    return true;
}

bool CDynodeSync::IsBlockchainSynced(bool fBlockAccepted)
{
    static bool fBlockchainSynced = false;
    static int64_t nTimeLastProcess = GetTime();
    static int nSkipped = 0;
    static bool fFirstBlockAccepted = false;

    // if the last call to this function was more than 60 minutes ago (client was in sleep mode) reset the sync process
    if(GetTime() - nTimeLastProcess > 60*60) {
        Reset();
        fBlockchainSynced = false;
    }

    if(!pCurrentBlockIndex || !pindexBestHeader || fImporting || fReindex) return false;

    if (IsInitialBlockDownload()) {
        if(fBlockAccepted) {
            // this should be only triggered while we are still syncing
            if(!IsSynced()) {
                // we are trying to download smth, reset blockchain sync status
                if(fDebug) LogPrintf("CDynodeSync::IsBlockchainSynced -- reset\n");
                fFirstBlockAccepted = true;
                fBlockchainSynced = false;
                nTimeLastProcess = GetTime();
                return false;
            }
        } else {
            // skip if we already checked less than 1 tick ago
            if(GetTime() - nTimeLastProcess < DYNODE_SYNC_TICK_SECONDS - DYNODE_SYNC_TICK_SECONDS_INITIAL) {
                nSkipped++;
                return fBlockchainSynced;
           }
        }
    }
    else 
    if (!IsInitialBlockDownload())
    {
        if(fBlockAccepted) {
            // this should be only triggered while we are still syncing
            if(!IsSynced()) {
                // we are trying to download smth, reset blockchain sync status
                if(fDebug) LogPrintf("CDynodeSync::IsBlockchainSynced -- reset\n");
                fFirstBlockAccepted = true;
                fBlockchainSynced = false;
                nTimeLastProcess = GetTime();
                return false;
            }
        } else {
            // skip if we already checked less than 1 tick ago
            if(GetTime() - nTimeLastProcess < DYNODE_SYNC_TICK_SECONDS) {
                nSkipped++;
                return fBlockchainSynced;
           }
        }
    }

    if(fDebug) LogPrintf("CDynodeSync::IsBlockchainSynced -- state before check: %ssynced, skipped %d times\n", fBlockchainSynced ? "" : "not ", nSkipped);

    nTimeLastProcess = GetTime();
    nSkipped = 0;

    if(fBlockchainSynced) return true;
    if(fCheckpointsEnabled && pCurrentBlockIndex->nHeight < Checkpoints::GetTotalBlocksEstimate(Params().Checkpoints()))
        return false;

    std::vector<CNode*> vNodesCopy = CopyNodeVector();

    // We have enough peers and assume most of them are synced
    if(vNodesCopy.size() >= DYNODE_SYNC_ENOUGH_PEERS) {
        // Check to see how many of our peers are (almost) at the same height as we are
        int nNodesAtSameHeight = 0;
        BOOST_FOREACH(CNode* pnode, vNodesCopy)
        {
            // Make sure this peer is presumably at the same height
            if(!CheckNodeHeight(pnode)) continue;
            nNodesAtSameHeight++;
            // if we have decent number of such peers, most likely we are synced now
            if(nNodesAtSameHeight >= DYNODE_SYNC_ENOUGH_PEERS) {
                LogPrintf("CDynodeSync::IsBlockchainSynced -- found enough peers on the same height as we are, done\n");
                fBlockchainSynced = true;
                ReleaseNodeVector(vNodesCopy);
                return true;
            }
        }
    }
    ReleaseNodeVector(vNodesCopy);

    // wait for at least one new block to be accepted
    if(!fFirstBlockAccepted) return false;

    // same as !IsInitialBlockDownload() but no cs_main needed here
    int64_t nMaxBlockTime = std::max(pCurrentBlockIndex->GetBlockTime(), pindexBestHeader->GetBlockTime());
    fBlockchainSynced = pindexBestHeader->nHeight - pCurrentBlockIndex->nHeight < 24*6 &&
                        GetTime() - nMaxBlockTime < Params().MaxTipAge();

    return fBlockchainSynced;
}

void CDynodeSync::Fail()
{
    nTimeLastFailure = GetTime();
    nRequestedDynodeAssets = DYNODE_SYNC_FAILED;
}

void CDynodeSync::Reset()
{
    nRequestedDynodeAssets = DYNODE_SYNC_INITIAL;
    nRequestedDynodeAttempt = 0;
    nTimeAssetSyncStarted = GetTime();
    nTimeLastDynodeList = GetTime();
    nTimeLastPaymentVote = GetTime();
    nTimeLastGovernanceItem = GetTime();
    nTimeLastFailure = 0;
    nCountFailures = 0;
}

std::string CDynodeSync::GetAssetName()
{
    switch(nRequestedDynodeAssets)
    {
        case(DYNODE_SYNC_INITIAL):      return "DYNODE_SYNC_INITIAL";
        case(DYNODE_SYNC_SPORKS):       return "DYNODE_SYNC_SPORKS";
        case(DYNODE_SYNC_LIST):         return "DYNODE_SYNC_LIST";
        case(DYNODE_SYNC_DNW):          return "DYNODE_SYNC_DNW";
        case(DYNODE_SYNC_GOVERNANCE):   return "DYNODE_SYNC_GOVERNANCE";
        case(DYNODE_SYNC_FAILED):       return "DYNODE_SYNC_FAILED";
        case DYNODE_SYNC_FINISHED:      return "DYNODE_SYNC_FINISHED";
        default:                           return "UNKNOWN";
    }
}

void CDynodeSync::SwitchToNextAsset()
{
    switch(nRequestedDynodeAssets)
    {
        case(DYNODE_SYNC_FAILED):
            throw std::runtime_error("Can't switch to next asset from failed, should use Reset() first!");
            break;
        case(DYNODE_SYNC_INITIAL):
            ClearFulfilledRequests();
            nRequestedDynodeAssets = DYNODE_SYNC_SPORKS;
            LogPrintf("CDynodeSync::SwitchToNextAsset -- Starting %s\n", GetAssetName());
            break;
        case(DYNODE_SYNC_SPORKS):
            nTimeLastDynodeList = GetTime();
            nRequestedDynodeAssets = DYNODE_SYNC_LIST;
            LogPrintf("CDynodeSync::SwitchToNextAsset -- Starting %s\n", GetAssetName());
            break;
        case(DYNODE_SYNC_LIST):
            nTimeLastPaymentVote = GetTime();
            nRequestedDynodeAssets = DYNODE_SYNC_DNW;
            LogPrintf("CDynodeSync::SwitchToNextAsset -- Starting %s\n", GetAssetName());
            break;
        case(DYNODE_SYNC_DNW):
            nTimeLastGovernanceItem = GetTime();
            nRequestedDynodeAssets = DYNODE_SYNC_GOVERNANCE;
            LogPrintf("CDynodeSync::SwitchToNextAsset -- Starting %s\n", GetAssetName());
            break;
        case(DYNODE_SYNC_GOVERNANCE):
            LogPrintf("CDynodeSync::SwitchToNextAsset -- Sync has finished\n");
            nRequestedDynodeAssets = DYNODE_SYNC_FINISHED;
            uiInterface.NotifyAdditionalDataSyncProgressChanged(1);
            //try to activate our Dynode if possible
            activeDynode.ManageState();

            TRY_LOCK(cs_vNodes, lockRecv);
            if(!lockRecv) return;

            BOOST_FOREACH(CNode* pnode, vNodes) {
                netfulfilledman.AddFulfilledRequest(pnode->addr, "full-sync");
            }

            break;
    }
    nRequestedDynodeAttempt = 0;
    nTimeAssetSyncStarted = GetTime();
}

std::string CDynodeSync::GetSyncStatus()
{
    switch (dynodeSync.nRequestedDynodeAssets) {
        case DYNODE_SYNC_INITIAL:       return _("Synchronization pending...");
        case DYNODE_SYNC_SPORKS:        return _("Synchronizing sporks...");
        case DYNODE_SYNC_LIST:          return _("Synchronizing Dynodes...");
        case DYNODE_SYNC_DNW:           return _("Synchronizing Dynode payments...");
        case DYNODE_SYNC_GOVERNANCE:    return _("Synchronizing governance objects...");
        case DYNODE_SYNC_FAILED:        return _("Synchronization failed");
        case DYNODE_SYNC_FINISHED:      return _("Synchronization finished");
        default:                           return "";
    }
}

void CDynodeSync::ProcessMessage(CNode* pfrom, std::string& strCommand, CDataStream& vRecv)
{
    if (strCommand == NetMsgType::SYNCSTATUSCOUNT) { //Sync status count

        //do not care about stats if sync process finished or failed
        if(IsSynced() || IsFailed()) return;

        int nItemID;
        int nCount;
        vRecv >> nItemID >> nCount;

        LogPrintf("SYNCSTATUSCOUNT -- got inventory count: nItemID=%d  nCount=%d  peer=%d\n", nItemID, nCount, pfrom->id);
    }
}

void CDynodeSync::ClearFulfilledRequests()
{
    TRY_LOCK(cs_vNodes, lockRecv);
    if(!lockRecv) return;

    BOOST_FOREACH(CNode* pnode, vNodes)
    {
        netfulfilledman.RemoveFulfilledRequest(pnode->addr, "spork-sync");
        netfulfilledman.RemoveFulfilledRequest(pnode->addr, "dynode-list-sync");
        netfulfilledman.RemoveFulfilledRequest(pnode->addr, "dynode-payment-sync");
        netfulfilledman.RemoveFulfilledRequest(pnode->addr, "governance-sync");
        netfulfilledman.RemoveFulfilledRequest(pnode->addr, "full-sync");
    }
}

void CDynodeSync::ProcessTick()
{
    static int nTick = 0;

    if (IsInitialBlockDownload()) {
        if(nTick++ % (DYNODE_SYNC_TICK_SECONDS - DYNODE_SYNC_TICK_SECONDS_INITIAL) != 0) return;
    }
    else
    if (!IsInitialBlockDownload()) {
        if(nTick++ % DYNODE_SYNC_TICK_SECONDS != 0) return;
    }

    if(!pCurrentBlockIndex) return;

    //the actual count of Dynodes we have currently
    int nDnCount = dnodeman.CountDynodes();

    if(fDebug) LogPrintf("CDynodeSync::ProcessTick -- nTick %d nDnCount %d\n", nTick, nDnCount);

    // RESET SYNCING INCASE OF FAILURE
    {
        if(IsSynced()) {
            /*
                Resync if we lost all dynodes from sleep/wake or failed to sync originally
            */
            if(nDnCount == 0) {
                LogPrintf("CDynodeSync::ProcessTick -- WARNING: not enough data, restarting sync\n");
                Reset();
            } else {
                std::vector<CNode*> vNodesCopy = CopyNodeVector();
                governance.RequestGovernanceObjectVotes(vNodesCopy);
                ReleaseNodeVector(vNodesCopy);
                return;
            }
        }

        //try syncing again
        if(IsFailed()) {
            if(nTimeLastFailure + (1*60) < GetTime()) { // 1 minute cooldown after failed sync
                Reset();
            }
            return;
        }
    }

    // INITIAL SYNC SETUP / LOG REPORTING
    double nSyncProgress = double(nRequestedDynodeAttempt + (nRequestedDynodeAssets - 1) * 8) / (8*4);
    LogPrintf("CDynodeSync::ProcessTick -- nTick %d nRequestedDynodeAssets %d nRequestedDynodeAttempt %d nSyncProgress %f\n", nTick, nRequestedDynodeAssets, nRequestedDynodeAttempt, nSyncProgress);
    uiInterface.NotifyAdditionalDataSyncProgressChanged(nSyncProgress);

    // sporks synced but blockchain is not, wait until we're almost at a recent block to continue
    if(Params().NetworkIDString() != CBaseChainParams::REGTEST &&
            !IsBlockchainSynced() && nRequestedDynodeAssets > DYNODE_SYNC_SPORKS)
    {
        LogPrintf("CDynodeSync::ProcessTick -- nTick %d nRequestedDynodeAssets %d nRequestedDynodeAttempt %d -- blockchain is not synced yet\n", nTick, nRequestedDynodeAssets, nRequestedDynodeAttempt);
        nTimeLastDynodeList = GetTime();
        nTimeLastPaymentVote = GetTime();
        nTimeLastGovernanceItem = GetTime();
        return;
    }

    if(nRequestedDynodeAssets == DYNODE_SYNC_INITIAL ||
        (nRequestedDynodeAssets == DYNODE_SYNC_SPORKS && IsBlockchainSynced()))
    {
        SwitchToNextAsset();
    }

    std::vector<CNode*> vNodesCopy = CopyNodeVector();

    BOOST_FOREACH(CNode* pnode, vNodesCopy)    {
        // Don't try to sync any data from outbound "dynode" connections -
        // they are temporary and should be considered unreliable for a sync process.
        // Inbound connection this early is most likely a "dynode" connection
        // initialted from another node, so skip it too.
        if(pnode->fDynode || (fDyNode && pnode->fInbound)) continue;
        // QUICK MODE (REGTEST ONLY!)
        if(Params().NetworkIDString() == CBaseChainParams::REGTEST)
        {
            if(nRequestedDynodeAttempt <= 2) {
                pnode->PushMessage(NetMsgType::GETSPORKS); //get current network sporks
            } else if(nRequestedDynodeAttempt < 4) {
                dnodeman.SsegUpdate(pnode);
            } else if(nRequestedDynodeAttempt < 6) {
                int nDnCount = dnodeman.CountDynodes();
                pnode->PushMessage(NetMsgType::DYNODEPAYMENTSYNC, nDnCount); //sync payment votes
                SendGovernanceSyncRequest(pnode);
            } else {
                nRequestedDynodeAssets = DYNODE_SYNC_FINISHED;
            }
            nRequestedDynodeAttempt++;
            ReleaseNodeVector(vNodesCopy);
            return;
        }

        // NORMAL NETWORK MODE - TESTNET/MAINNET
        {
            if(netfulfilledman.HasFulfilledRequest(pnode->addr, "full-sync")) {
                // We already fully synced from this node recently,
                // disconnect to free this connection slot for another peer.
                pnode->fDisconnect = true;
                LogPrintf("CDynodeSync::ProcessTick -- disconnecting from recently synced peer %d\n", pnode->id);
                continue;
            }

            // SPORK : ALWAYS ASK FOR SPORKS AS WE SYNC (we skip this mode now)

            if(!netfulfilledman.HasFulfilledRequest(pnode->addr, "spork-sync")) {
                // only request once from each peer
                netfulfilledman.AddFulfilledRequest(pnode->addr, "spork-sync");
                // get current network sporks
                pnode->PushMessage(NetMsgType::GETSPORKS);
                LogPrintf("CDynodeSync::ProcessTick -- nTick %d nRequestedDynodeAssets %d -- requesting sporks from peer %d\n", nTick, nRequestedDynodeAssets, pnode->id);
                continue; // always get sporks first, switch to the next node without waiting for the next tick
            }

            // MNLIST : SYNC DYNODE LIST FROM OTHER CONNECTED CLIENTS

            if(nRequestedDynodeAssets == DYNODE_SYNC_LIST) {
                LogPrint("Dynode", "CDynodeSync::ProcessTick -- nTick %d nRequestedDynodeAssets %d nTimeLastDynodeList %lld GetTime() %lld diff %lld\n", nTick, nRequestedDynodeAssets, nTimeLastDynodeList, GetTime(), GetTime() - nTimeLastDynodeList);
                if (IsInitialBlockDownload()) {
                    // check for timeout first
                    if(nTimeLastDynodeList < GetTime() - (DYNODE_SYNC_TIMEOUT_SECONDS - DYNODE_SYNC_TIMEOUT_SECONDS_INITIAL)) {
                        LogPrintf("CDynodeSync::ProcessTick -- nTick %d nRequestedDynodeAssets %d -- timeout\n", nTick, nRequestedDynodeAssets);
                        if (nRequestedDynodeAttempt == 0) {
                            LogPrintf("CDynodeSync::ProcessTick -- ERROR: failed to sync %s\n", GetAssetName());
                            // there is no way we can continue without Dynode list, fail here and try later
                            Fail();
                            ReleaseNodeVector(vNodesCopy);
                            return;
                        }
                        SwitchToNextAsset();
                        ReleaseNodeVector(vNodesCopy);
                        return;
                    }
                }
                else
                if (!IsInitialBlockDownload()) {
                    // check for timeout first
                    if(nTimeLastDynodeList < GetTime() - DYNODE_SYNC_TIMEOUT_SECONDS) {
                        LogPrintf("CDynodeSync::ProcessTick -- nTick %d nRequestedDynodeAssets %d -- timeout\n", nTick, nRequestedDynodeAssets);
                        if (nRequestedDynodeAttempt == 0) {
                            LogPrintf("CDynodeSync::ProcessTick -- ERROR: failed to sync %s\n", GetAssetName());
                            // there is no way we can continue without Dynode list, fail here and try later
                            Fail();
                            ReleaseNodeVector(vNodesCopy);
                            return;
                        }
                        SwitchToNextAsset();
                        ReleaseNodeVector(vNodesCopy);
                        return;
                    }
                }

                // only request once from each peer
                if(netfulfilledman.HasFulfilledRequest(pnode->addr, "dynode-list-sync")) continue;
                netfulfilledman.AddFulfilledRequest(pnode->addr, "dynode-list-sync");

                if (pnode->nVersion < dnpayments.GetMinDynodePaymentsProto()) continue;
                nRequestedDynodeAttempt++;

                dnodeman.SsegUpdate(pnode);

                ReleaseNodeVector(vNodesCopy);
                return; //this will cause each peer to get one request each six seconds for the various assets we need
            }

            // DNW : SYNC DYNODE PAYMENT VOTES FROM OTHER CONNECTED CLIENTS

            if(nRequestedDynodeAssets == DYNODE_SYNC_DNW) {
                LogPrint("dnpayments", "CDynodeSync::ProcessTick -- nTick %d nRequestedDynodeAssets %d nTimeLastPaymentVote %lld GetTime() %lld diff %lld\n", nTick, nRequestedDynodeAssets, nTimeLastPaymentVote, GetTime(), GetTime() - nTimeLastPaymentVote);
                // check for timeout first
                // This might take a lot longer than DYNODE_SYNC_TIMEOUT_SECONDS minutes due to new blocks,
                // but that should be OK and it should timeout eventually.
                if (IsInitialBlockDownload()) {
                    if(nTimeLastPaymentVote < GetTime() - (DYNODE_SYNC_TIMEOUT_SECONDS - DYNODE_SYNC_TIMEOUT_SECONDS_INITIAL)) {
                        LogPrintf("CDynodeSync::ProcessTick -- nTick %d nRequestedDynodeAssets %d -- timeout\n", nTick, nRequestedDynodeAssets);
                        if (nRequestedDynodeAttempt == 0) {
                            LogPrintf("CDynodeSync::ProcessTick -- ERROR: failed to sync %s\n", GetAssetName());
                            // probably not a good idea to proceed without winner list
                            Fail();
                            ReleaseNodeVector(vNodesCopy);
                            return;
                        }
                        SwitchToNextAsset();
                        ReleaseNodeVector(vNodesCopy);
                        return;
                    }
                }
                else
                if (!IsInitialBlockDownload()) {
                    if(nTimeLastPaymentVote < GetTime() - (DYNODE_SYNC_TIMEOUT_SECONDS)) {
                        LogPrintf("CDynodeSync::ProcessTick -- nTick %d nRequestedDynodeAssets %d -- timeout\n", nTick, nRequestedDynodeAssets);
                        if (nRequestedDynodeAttempt == 0) {
                            LogPrintf("CDynodeSync::ProcessTick -- ERROR: failed to sync %s\n", GetAssetName());
                            // probably not a good idea to proceed without winner list
                            Fail();
                            ReleaseNodeVector(vNodesCopy);
                            return;
                        }
                        SwitchToNextAsset();
                        ReleaseNodeVector(vNodesCopy);
                        return;
                    }
                }
                // check for data
                // if dnpayments already has enough blocks and votes, switch to the next asset
                // try to fetch data from at least two peers though
                if(nRequestedDynodeAttempt > 1 && dnpayments.IsEnoughData()) {
                    LogPrintf("CDynodeSync::ProcessTick -- nTick %d nRequestedDynodeAssets %d -- found enough data\n", nTick, nRequestedDynodeAssets);
                    SwitchToNextAsset();
                    ReleaseNodeVector(vNodesCopy);
                    return;
                }

                // only request once from each peer
                if(netfulfilledman.HasFulfilledRequest(pnode->addr, "dynode-payment-sync")) continue;
                netfulfilledman.AddFulfilledRequest(pnode->addr, "dynode-payment-sync");

                if(pnode->nVersion < dnpayments.GetMinDynodePaymentsProto()) continue;
                nRequestedDynodeAttempt++;

                // ask node for all payment votes it has (new nodes will only return votes for future payments)
                pnode->PushMessage(NetMsgType::DYNODEPAYMENTSYNC, dnpayments.GetStorageLimit());
                // ask node for missing pieces only (old nodes will not be asked)
                dnpayments.RequestLowDataPaymentBlocks(pnode);

                ReleaseNodeVector(vNodesCopy);
                return; //this will cause each peer to get one request each six seconds for the various assets we need
            }

            // GOVOBJ : SYNC GOVERNANCE ITEMS FROM OUR PEERS

            if(nRequestedDynodeAssets == DYNODE_SYNC_GOVERNANCE) {
                LogPrint("gobject", "CDynodeSync::ProcessTick -- nTick %d nRequestedDynodeAssets %d nTimeLastGovernanceItem %lld GetTime() %lld diff %lld\n", nTick, nRequestedDynodeAssets, nTimeLastGovernanceItem, GetTime(), GetTime() - nTimeLastGovernanceItem);

                // check for timeout first
                if (IsInitialBlockDownload()) {
                    if(GetTime() - nTimeLastGovernanceItem > (DYNODE_SYNC_TIMEOUT_SECONDS - DYNODE_SYNC_TIMEOUT_SECONDS_INITIAL)) {
                        LogPrintf("CDynodeSync::ProcessTick -- nTick %d nRequestedDynodeAssets %d -- timeout\n", nTick, nRequestedDynodeAssets);
                        if(nRequestedDynodeAttempt == 0) {
                            LogPrintf("CDynodeSync::ProcessTick -- WARNING: failed to sync %s\n", GetAssetName());
                            // it's kind of ok to skip this for now, hopefully we'll catch up later?
                        }
                        SwitchToNextAsset();
                        ReleaseNodeVector(vNodesCopy);
                        return;
                    }
                }
                else
                if (!IsInitialBlockDownload()) {
                    if(GetTime() - nTimeLastGovernanceItem > DYNODE_SYNC_TIMEOUT_SECONDS) {
                        LogPrintf("CDynodeSync::ProcessTick -- nTick %d nRequestedDynodeAssets %d -- timeout\n", nTick, nRequestedDynodeAssets);
                        if(nRequestedDynodeAttempt == 0) {
                            LogPrintf("CDynodeSync::ProcessTick -- WARNING: failed to sync %s\n", GetAssetName());
                            // it's kind of ok to skip this for now, hopefully we'll catch up later?
                        }
                        SwitchToNextAsset();
                        ReleaseNodeVector(vNodesCopy);
                        return;
                    }
                }
                // only request obj sync once from each peer, then request votes on per-obj basis
                if(netfulfilledman.HasFulfilledRequest(pnode->addr, "governance-sync")) {
                    governance.RequestGovernanceObjectVotes(pnode);
                    int nObjsLeftToAsk = governance.RequestGovernanceObjectVotes(pnode);
                    static int64_t nTimeNoObjectsLeft = 0;
                    // check for data
                    if(nObjsLeftToAsk == 0) {
                        static int nLastTick = 0;
                        static int nLastVotes = 0;
                        if(nTimeNoObjectsLeft == 0) {
                            // asked all objects for votes for the first time
                            nTimeNoObjectsLeft = GetTime();
                        }
                        // make sure the condition below is checked only once per tick
                        if(nLastTick == nTick) continue;

                        if (IsInitialBlockDownload()) {
                            if(GetTime() - nTimeNoObjectsLeft > DYNODE_SYNC_TIMEOUT_SECONDS &&
                                governance.GetVoteCount() - nLastVotes < std::max(int(0.0001 * nLastVotes), (DYNODE_SYNC_TICK_SECONDS - DYNODE_SYNC_TICK_SECONDS_INITIAL))
                            ) {
                                // We already asked for all objects, waited for DYNODE_SYNC_TIMEOUT_SECONDS
                                // after that and less then 0.01% or DYNODE_SYNC_TICK_SECONDS
                                // (i.e. 1 per second) votes were recieved during the last tick.
                                // We can be pretty sure that we are done syncing.
                                LogPrintf("CDynodeSync::ProcessTick -- nTick %d nRequestedDynodeAssets %d -- asked for all objects, nothing to do\n", nTick, nRequestedDynodeAssets);
                                // reset nTimeNoObjectsLeft to be able to use the same condition on resync
                                nTimeNoObjectsLeft = 0;
                                SwitchToNextAsset();
                                ReleaseNodeVector(vNodesCopy);
                                return;
                            }
                        }
                        else
                        if (!IsInitialBlockDownload()) {
                            if(GetTime() - nTimeNoObjectsLeft > DYNODE_SYNC_TIMEOUT_SECONDS &&
                                governance.GetVoteCount() - nLastVotes < std::max(int(0.0001 * nLastVotes), DYNODE_SYNC_TICK_SECONDS)
                            ) {
                                // We already asked for all objects, waited for DYNODE_SYNC_TIMEOUT_SECONDS
                                // after that and less then 0.01% or DYNODE_SYNC_TICK_SECONDS
                                // (i.e. 1 per second) votes were recieved during the last tick.
                                // We can be pretty sure that we are done syncing.
                                LogPrintf("CDynodeSync::ProcessTick -- nTick %d nRequestedDynodeAssets %d -- asked for all objects, nothing to do\n", nTick, nRequestedDynodeAssets);
                                // reset nTimeNoObjectsLeft to be able to use the same condition on resync
                                nTimeNoObjectsLeft = 0;
                                SwitchToNextAsset();
                                ReleaseNodeVector(vNodesCopy);
                                return;
                            }
                        }  

                        nLastTick = nTick;
                        nLastVotes = governance.GetVoteCount();
                    }
                 }

                netfulfilledman.AddFulfilledRequest(pnode->addr, "governance-sync");

                if (pnode->nVersion < MIN_GOVERNANCE_PEER_PROTO_VERSION) continue;
                nRequestedDynodeAttempt++;

                SendGovernanceSyncRequest(pnode);

                ReleaseNodeVector(vNodesCopy);
                return; //this will cause each peer to get one request each six seconds for the various assets we need
            }
        }
    }
    // looped through all nodes, release them
    ReleaseNodeVector(vNodesCopy);
}

void CDynodeSync::SendGovernanceSyncRequest(CNode* pnode)
{
    CBloomFilter filter;
    filter.clear();

    pnode->PushMessage(NetMsgType::DNGOVERNANCESYNC, uint256(), filter);
}

void CDynodeSync::UpdatedBlockTip(const CBlockIndex *pindex)
{
    pCurrentBlockIndex = pindex;
}
