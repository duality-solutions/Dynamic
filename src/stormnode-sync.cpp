// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "activestormnode.h"
#include "checkpoints.h"
#include "governance.h"
#include "main.h"
#include "stormnode.h"
#include "stormnode-payments.h"
#include "stormnode-sync.h"
#include "stormnodeman.h"
#include "netfulfilledman.h"
#include "spork.h"
#include "util.h"

class CStormnodeSync;
CStormnodeSync stormnodeSync;

bool CStormnodeSync::IsBlockchainSynced()
{
    static bool fBlockchainSynced = false;
    static int64_t nTimeLastProcess = GetTime();

    // if the last call to this function was more than 60 minutes ago (client was in sleep mode) reset the sync process
    if(GetTime() - nTimeLastProcess > 60*60) {
        Reset();
        fBlockchainSynced = false;
    }
    nTimeLastProcess = GetTime();

    if(fBlockchainSynced) return true;
    if(!pCurrentBlockIndex || !pindexBestHeader || fImporting || fReindex) return false;
    if(fCheckpointsEnabled && pCurrentBlockIndex->nHeight < Checkpoints::GetTotalBlocksEstimate(Params().Checkpoints()))
        return false;

    // same as !IsInitialBlockDownload() but no cs_main needed here
    int64_t nMaxBlockTime = std::max(pCurrentBlockIndex->GetBlockTime(), pindexBestHeader->GetBlockTime());
    fBlockchainSynced = pindexBestHeader->nHeight - pCurrentBlockIndex->nHeight &&
                        GetTime() - nMaxBlockTime < Params().MaxTipAge();

    return fBlockchainSynced;
}

void CStormnodeSync::Fail()
{
    nTimeLastFailure = GetTime();
    nRequestedStormnodeAssets = STORMNODE_SYNC_FAILED;
}

void CStormnodeSync::Reset()
{
    nRequestedStormnodeAssets = STORMNODE_SYNC_INITIAL;
    nRequestedStormnodeAttempt = 0;
    nTimeAssetSyncStarted = GetTime();
    nTimeLastStormnodeList = GetTime();
    nTimeLastPaymentVote = GetTime();
    nTimeLastGovernanceItem = GetTime();
    nTimeLastFailure = 0;
    nCountFailures = 0;
}

std::string CStormnodeSync::GetAssetName()
{
    switch(nRequestedStormnodeAssets)
    {
        case(STORMNODE_SYNC_INITIAL):      return "STORMNODE_SYNC_INITIAL";
        case(STORMNODE_SYNC_SPORKS):       return "STORMNODE_SYNC_SPORKS";
        case(STORMNODE_SYNC_LIST):         return "STORMNODE_SYNC_LIST";
        case(STORMNODE_SYNC_SNW):          return "STORMNODE_SYNC_SNW";
        case(STORMNODE_SYNC_GOVERNANCE):   return "STORMNODE_SYNC_GOVERNANCE";
        case(STORMNODE_SYNC_FAILED):       return "STORMNODE_SYNC_FAILED";
        case STORMNODE_SYNC_FINISHED:      return "STORMNODE_SYNC_FINISHED";
        default:                           return "UNKNOWN";
    }
}

void CStormnodeSync::SwitchToNextAsset()
{
    switch(nRequestedStormnodeAssets)
    {
        case(STORMNODE_SYNC_FAILED):
            throw std::runtime_error("Can't switch to next asset from failed, should use Reset() first!");
            break;
        case(STORMNODE_SYNC_INITIAL):
            ClearFulfilledRequests();
            nRequestedStormnodeAssets = STORMNODE_SYNC_SPORKS;
            break;
        case(STORMNODE_SYNC_SPORKS):
            nTimeLastStormnodeList = GetTime();
            nRequestedStormnodeAssets = STORMNODE_SYNC_LIST;
            break;
        case(STORMNODE_SYNC_LIST):
            nTimeLastPaymentVote = GetTime();
            nRequestedStormnodeAssets = STORMNODE_SYNC_SNW;
            break;
        case(STORMNODE_SYNC_SNW):
            nTimeLastGovernanceItem = GetTime();
            nRequestedStormnodeAssets = STORMNODE_SYNC_GOVERNANCE;
            break;
        case(STORMNODE_SYNC_GOVERNANCE):
            LogPrintf("CStormnodeSync::SwitchToNextAsset -- Sync has finished\n");
            nRequestedStormnodeAssets = STORMNODE_SYNC_FINISHED;
            uiInterface.NotifyAdditionalDataSyncProgressChanged(1);
            //try to activate our stormnode if possible
            activeStormnode.ManageState();

            TRY_LOCK(cs_vNodes, lockRecv);
            if(!lockRecv) return;

            BOOST_FOREACH(CNode* pnode, vNodes) {
                netfulfilledman.AddFulfilledRequest(pnode->addr, "full-sync");
            }

            break;
    }
    nRequestedStormnodeAttempt = 0;
    nTimeAssetSyncStarted = GetTime();
}

std::string CStormnodeSync::GetSyncStatus()
{
    switch (stormnodeSync.nRequestedStormnodeAssets) {
        case STORMNODE_SYNC_INITIAL:       return _("Synchronization pending...");
        case STORMNODE_SYNC_SPORKS:        return _("Synchronizing sporks...");
        case STORMNODE_SYNC_LIST:          return _("Synchronizing stormnodes...");
        case STORMNODE_SYNC_SNW:           return _("Synchronizing stormnode payments...");
        case STORMNODE_SYNC_GOVERNANCE:    return _("Synchronizing governance objects...");
        case STORMNODE_SYNC_FAILED:        return _("Synchronization failed");
        case STORMNODE_SYNC_FINISHED:      return _("Synchronization finished");
        default:                           return "";
    }
}

void CStormnodeSync::ProcessMessage(CNode* pfrom, std::string& strCommand, CDataStream& vRecv)
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

void CStormnodeSync::ClearFulfilledRequests()
{
    TRY_LOCK(cs_vNodes, lockRecv);
    if(!lockRecv) return;

    BOOST_FOREACH(CNode* pnode, vNodes)
    {
        netfulfilledman.RemoveFulfilledRequest(pnode->addr, "spork-sync");
        netfulfilledman.RemoveFulfilledRequest(pnode->addr, "stormnode-list-sync");
        netfulfilledman.RemoveFulfilledRequest(pnode->addr, "stormnode-payment-sync");
        netfulfilledman.RemoveFulfilledRequest(pnode->addr, "governance-sync");
        netfulfilledman.RemoveFulfilledRequest(pnode->addr, "full-sync");
    }
}

void ReleaseNodes(const std::vector<CNode*> &vNodesCopy)
{
    LOCK(cs_vNodes);
    BOOST_FOREACH(CNode* pnode, vNodesCopy)
        pnode->Release();
}

void CStormnodeSync::ProcessTick()
{
    static int nTick = 0;
    if(nTick++ % 6 != 0) return;
    if(!pCurrentBlockIndex) return;

    //the actual count of stormnodes we have currently
    int nSnCount = snodeman.CountStormnodes();

    if(fDebug) LogPrintf("CStormnodeSync::ProcessTick -- nTick %d nSnCount %d\n", nTick, nSnCount);

    // RESET SYNCING INCASE OF FAILURE
    {
        if(IsSynced()) {
            /*
                Resync if we lose all stormnodes from sleep/wake or failure to sync originally
            */
            if(nSnCount == 0) {
                LogPrintf("CStormnodeSync::ProcessTick -- WARNING: not enough data, restarting sync\n");
                Reset();
            } else {
                //if syncing is complete and we have stormnodes, return
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
    double nSyncProgress = double(nRequestedStormnodeAttempt + (nRequestedStormnodeAssets - 1) * 8) / (8*4);
    LogPrintf("CStormnodeSync::ProcessTick -- nTick %d nRequestedStormnodeAssets %d nRequestedStormnodeAttempt %d nSyncProgress %f\n", nTick, nRequestedStormnodeAssets, nRequestedStormnodeAttempt, nSyncProgress);
    uiInterface.NotifyAdditionalDataSyncProgressChanged(nSyncProgress);

    // sporks synced but blockchain is not, wait until we're almost at a recent block to continue
    if(Params().NetworkIDString() != CBaseChainParams::REGTEST &&
            !IsBlockchainSynced() && nRequestedStormnodeAssets > STORMNODE_SYNC_SPORKS)
    {
        LogPrintf("CStormnodeSync::ProcessTick -- nTick %d nRequestedStormnodeAssets %d nRequestedStormnodeAttempt %d -- blockchain is not synced yet\n", nTick, nRequestedStormnodeAssets, nRequestedStormnodeAttempt);
        return;
    }

    if(nRequestedStormnodeAssets == STORMNODE_SYNC_INITIAL ||
        (nRequestedStormnodeAssets == STORMNODE_SYNC_SPORKS && IsBlockchainSynced()))
    {
        SwitchToNextAsset();
    }

    std::vector<CNode*> vNodesCopy;
    {
        LOCK(cs_vNodes);
        vNodesCopy = vNodes;
        BOOST_FOREACH(CNode* pnode, vNodesCopy)
            pnode->AddRef();
    }

    BOOST_FOREACH(CNode* pnode, vNodesCopy)    {
        // QUICK MODE (REGTEST ONLY!)
        if(Params().NetworkIDString() == CBaseChainParams::REGTEST)
        {
            if(nRequestedStormnodeAttempt <= 2) {
                pnode->PushMessage(NetMsgType::GETSPORKS); //get current network sporks
            } else if(nRequestedStormnodeAttempt < 4) {
                snodeman.SsegUpdate(pnode);
            } else if(nRequestedStormnodeAttempt < 6) {
                int nSnCount = snodeman.CountStormnodes();
                pnode->PushMessage(NetMsgType::STORMNODEPAYMENTSYNC, nSnCount); //sync payment votes
                uint256 n = uint256();
                pnode->PushMessage(NetMsgType::SNGOVERNANCESYNC, n); //sync stormnode votes
            } else {
                nRequestedStormnodeAssets = STORMNODE_SYNC_FINISHED;
            }
            nRequestedStormnodeAttempt++;
            ReleaseNodes(vNodesCopy);
            return;
        }

        // NORMAL NETWORK MODE - TESTNET/MAINNET
        {
            if(netfulfilledman.HasFulfilledRequest(pnode->addr, "full-sync")) {
                // we already fully synced from this node recently,
                // disconnect to free this connection slot for a new node
                pnode->fDisconnect = true;
                LogPrintf("CStormnodeSync::ProcessTick -- disconnecting from recently synced peer %d\n", pnode->id);
                continue;
            }

            // SPORK : ALWAYS ASK FOR SPORKS AS WE SYNC (we skip this mode now)

            if(!netfulfilledman.HasFulfilledRequest(pnode->addr, "spork-sync")) {
                // only request once from each peer
                netfulfilledman.AddFulfilledRequest(pnode->addr, "spork-sync");
                // get current network sporks
                pnode->PushMessage(NetMsgType::GETSPORKS);
                LogPrintf("CStormnodeSync::ProcessTick -- nTick %d nRequestedStormnodeAssets %d -- requesting sporks from peer %d\n", nTick, nRequestedStormnodeAssets, pnode->id);
                continue; // always get sporks first, switch to the next node without waiting for the next tick
            }

            // MNLIST : SYNC STORMNODE LIST FROM OTHER CONNECTED CLIENTS

            if(nRequestedStormnodeAssets == STORMNODE_SYNC_LIST) {
                LogPrint("stormnode", "CStormnodeSync::ProcessTick -- nTick %d nRequestedStormnodeAssets %d nTimeLastStormnodeList %lld GetTime() %lld diff %lld\n", nTick, nRequestedStormnodeAssets, nTimeLastStormnodeList, GetTime(), GetTime() - nTimeLastStormnodeList);
                // check for timeout first
                if(nTimeLastStormnodeList < GetTime() - STORMNODE_SYNC_TIMEOUT_SECONDS) {
                    LogPrintf("CStormnodeSync::ProcessTick -- nTick %d nRequestedStormnodeAssets %d -- timeout\n", nTick, nRequestedStormnodeAssets);
                    if (nRequestedStormnodeAttempt == 0) {
                        LogPrintf("CStormnodeSync::ProcessTick -- ERROR: failed to sync %s\n", GetAssetName());
                        // there is no way we can continue without stormnode list, fail here and try later
                        Fail();
                        ReleaseNodes(vNodesCopy);
                        return;
                    }
                    SwitchToNextAsset();
                    ReleaseNodes(vNodesCopy);
                    return;
                }

                // check for data
                // if we have enough stormnodes in our list, switch to the next asset
                /* Note: Is this activing up? It's probably related to int CStormnodeMan::GetEstimatedStormnodes(int nBlock)
                   Surely doesn't work right for testnet currently */
                // try to fetch data from at least two peers though
                int nSnCountEstimated = snodeman.GetEstimatedStormnodes(pCurrentBlockIndex->nHeight)*0.9;
                LogPrintf("CStormnodeSync::ProcessTick -- nTick %d nSnCount %d nSnCountEstimated %d\n",
                          nTick, nSnCount, nSnCountEstimated);
                if(nRequestedStormnodeAttempt > 1 && nSnCount > nSnCountEstimated) {
                    LogPrintf("CStormnodeSync::ProcessTick -- nTick %d nRequestedStormnodeAssets %d -- found enough data\n", nTick, nRequestedStormnodeAssets);
                    SwitchToNextAsset();
                    ReleaseNodes(vNodesCopy);
                    return;
                }

                // only request once from each peer
                if(netfulfilledman.HasFulfilledRequest(pnode->addr, "stormnode-list-sync")) continue;
                netfulfilledman.AddFulfilledRequest(pnode->addr, "stormnode-list-sync");

                if (pnode->nVersion < snpayments.GetMinStormnodePaymentsProto()) continue;
                nRequestedStormnodeAttempt++;

                snodeman.SsegUpdate(pnode);

                ReleaseNodes(vNodesCopy);
                return; //this will cause each peer to get one request each six seconds for the various assets we need
            }

            // SNW : SYNC STORMNODE PAYMENT VOTES FROM OTHER CONNECTED CLIENTS

            if(nRequestedStormnodeAssets == STORMNODE_SYNC_SNW) {
                LogPrint("snpayments", "CStormnodeSync::ProcessTick -- nTick %d nRequestedStormnodeAssets %d nTimeLastPaymentVote %lld GetTime() %lld diff %lld\n", nTick, nRequestedStormnodeAssets, nTimeLastPaymentVote, GetTime(), GetTime() - nTimeLastPaymentVote);
                // check for timeout first
                // This might take a lot longer than STORMNODE_SYNC_TIMEOUT_SECONDS minutes due to new blocks,
                // but that should be OK and it should timeout eventually.
                if(nTimeLastPaymentVote < GetTime() - STORMNODE_SYNC_TIMEOUT_SECONDS) {
                    LogPrintf("CStormnodeSync::ProcessTick -- nTick %d nRequestedStormnodeAssets %d -- timeout\n", nTick, nRequestedStormnodeAssets);
                    if (nRequestedStormnodeAttempt == 0) {
                        LogPrintf("CStormnodeSync::ProcessTick -- ERROR: failed to sync %s\n", GetAssetName());
                        // probably not a good idea to proceed without winner list
                        Fail();
                        ReleaseNodes(vNodesCopy);
                        return;
                    }
                    SwitchToNextAsset();
                    ReleaseNodes(vNodesCopy);
                    return;
                }

                // check for data
                // if snpayments already has enough blocks and votes, switch to the next asset
                // try to fetch data from at least two peers though
                if(nRequestedStormnodeAttempt > 1 && snpayments.IsEnoughData()) {
                    LogPrintf("CStormnodeSync::ProcessTick -- nTick %d nRequestedStormnodeAssets %d -- found enough data\n", nTick, nRequestedStormnodeAssets);
                    SwitchToNextAsset();
                    ReleaseNodes(vNodesCopy);
                    return;
                }

                // only request once from each peer
                if(netfulfilledman.HasFulfilledRequest(pnode->addr, "stormnode-payment-sync")) continue;
                netfulfilledman.AddFulfilledRequest(pnode->addr, "stormnode-payment-sync");

                if(pnode->nVersion < snpayments.GetMinStormnodePaymentsProto()) continue;
                nRequestedStormnodeAttempt++;

                // ask node for all payment votes it has (new nodes will only return votes for future payments)
                pnode->PushMessage(NetMsgType::STORMNODEPAYMENTSYNC, snpayments.GetStorageLimit());
                // ask node for missing pieces only (old nodes will not be asked)
                snpayments.RequestLowDataPaymentBlocks(pnode);

                ReleaseNodes(vNodesCopy);
                return; //this will cause each peer to get one request each six seconds for the various assets we need
            }

            // GOVOBJ : SYNC GOVERNANCE ITEMS FROM OUR PEERS

            if(nRequestedStormnodeAssets == STORMNODE_SYNC_GOVERNANCE) {
                LogPrint("snpayments", "CStormnodeSync::ProcessTick -- nTick %d nRequestedStormnodeAssets %d nTimeLastPaymentVote %lld GetTime() %lld diff %lld\n", nTick, nRequestedStormnodeAssets, nTimeLastPaymentVote, GetTime(), GetTime() - nTimeLastPaymentVote);

                // check for timeout first
                if(GetTime() - nTimeLastGovernanceItem > STORMNODE_SYNC_TIMEOUT_SECONDS) {
                    LogPrintf("CStormnodeSync::ProcessTick -- nTick %d nRequestedStormnodeAssets %d -- timeout\n", nTick, nRequestedStormnodeAssets);
                    if(nRequestedStormnodeAttempt == 0) {
                        LogPrintf("CStormnodeSync::ProcessTick -- WARNING: failed to sync %s\n", GetAssetName());
                        // it's kind of ok to skip this for now, hopefully we'll catch up later?
                    }
                    SwitchToNextAsset();
                    ReleaseNodes(vNodesCopy);
                    return;
                }

                // check for data
                // if(nCountBudgetItemProp > 0 && nCountBudgetItemFin)
                // {
                //     if(governance.CountProposalInventoryItems() >= (nSumBudgetItemProp / nCountBudgetItemProp)*0.9)
                //     {
                //         if(governance.CountFinalizedInventoryItems() >= (nSumBudgetItemFin / nCountBudgetItemFin)*0.9)
                //         {
                //             SwitchToNextAsset();
                //             return;
                //         }
                //     }
                // }

                // only request once from each peer
                if(netfulfilledman.HasFulfilledRequest(pnode->addr, "governance-sync")) continue;
                netfulfilledman.AddFulfilledRequest(pnode->addr, "governance-sync");

                if (pnode->nVersion < MIN_GOVERNANCE_PEER_PROTO_VERSION) continue;
                nRequestedStormnodeAttempt++;

                pnode->PushMessage(NetMsgType::SNGOVERNANCESYNC, uint256()); //sync stormnode votes

                ReleaseNodes(vNodesCopy);
                return; //this will cause each peer to get one request each six seconds for the various assets we need
            }
        }
    }
    // looped through all nodes, release them
    ReleaseNodes(vNodesCopy);
}

void CStormnodeSync::UpdatedBlockTip(const CBlockIndex *pindex)
{
    pCurrentBlockIndex = pindex;
}
