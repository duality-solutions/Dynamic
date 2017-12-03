// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "privatesend-client.h"

#include "coincontrol.h"
#include "consensus/validation.h"
#include "core_io.h"
#include "init.h"
#include "dynode-sync.h"
#include "dynodeman.h"
#include "script/sign.h"
#include "txmempool.h"
#include "util.h"
#include "utilmoneystr.h"

#include <memory>

CPrivateSendClient privateSendClient;

void CPrivateSendClient::ProcessMessage(CNode* pfrom, std::string& strCommand, CDataStream& vRecv, CConnman& connman)
{
    if(fDyNode) return;
    if(fLiteMode) return; // ignore all Dynamic related functionality
    if(!dynodeSync.IsBlockchainSynced()) return;

    if(strCommand == NetMsgType::PSQUEUE) {
        TRY_LOCK(cs_privatesend, lockRecv);
        if(!lockRecv) return;

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrint("privatesend", "PSQUEUE -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            return;
        }

        CPrivatesendQueue psq;
        vRecv >> psq;

        // process every psq only once
        BOOST_FOREACH(CPrivatesendQueue q, vecPrivatesendQueue) {
            if(q == psq) {
                // LogPrint("privatesend", "PSQUEUE -- %s seen\n", psq.ToString());
                return;
            }
        }

        LogPrint("privatesend", "PSQUEUE -- %s new\n", psq.ToString());

        if(psq.IsExpired()) return;

        dynode_info_t infoDn;
        if(!dnodeman.GetDynodeInfo(psq.vin.prevout, infoDn)) return;

        if(!psq.CheckSignature(infoDn.pubKeyDynode)) {
            // we probably have outdated info
            dnodeman.AskForDN(pfrom, psq.vin.prevout, connman);
            return;
        }

        // if the queue is ready, submit if we can
        if(psq.fReady) {
            if(!infoMixingDynode.fInfoValid) return;
            if(infoMixingDynode.addr != infoDn.addr) {
                LogPrintf("PSQUEUE -- message doesn't match current Dynode: infoMixingDynode=%s, addr=%s\n", infoMixingDynode.addr.ToString(), infoDn.addr.ToString());
                return;
            }

            if(nState == POOL_STATE_QUEUE) {
                LogPrint("privatesend", "DSQUEUE -- PrivateSend queue (%s) is ready on dynode %s\n", psq.ToString(), infoDn.addr.ToString());
                SubmitDenominate(connman);
            }
        } else {
            BOOST_FOREACH(CPrivatesendQueue q, vecPrivatesendQueue) {
                if(q.vin == psq.vin) {
                    // no way same dn can send another "not yet ready" psq this soon
                    LogPrint("privatesend", "DSQUEUE -- Dynode %s is sending WAY too many psq messages\n", infoDn.addr.ToString());
                    return;
                }
            }

            int nThreshold = infoDn.nLastPsq + dnodeman.CountEnabled(MIN_PRIVATESEND_PEER_PROTO_VERSION)/5;
            LogPrint("privatesend", "DSQUEUE -- nLastPsq: %d  threshold: %d  nPsqCount: %d\n", infoDn.nLastPsq, nThreshold, dnodeman.nPsqCount);
            //don't allow a few nodes to dominate the queuing process
            if(infoDn.nLastPsq != 0 && nThreshold > dnodeman.nPsqCount) {
                LogPrint("privatesend", "DSQUEUE -- Dynode %s is sending too many psq messages\n", infoDn.addr.ToString());
                return;
            }

            if(!dnodeman.AllowMixing(psq.vin.prevout)) return;

            LogPrint("privatesend", "DSQUEUE -- new PrivateSend queue (%s) from dynode %s\n", psq.ToString(), infoDn.addr.ToString());
            if(infoMixingDynode.fInfoValid && infoMixingDynode.vin.prevout == psq.vin.prevout) {
                psq.fTried = true;
            }
            vecPrivatesendQueue.push_back(psq);
            psq.Relay(connman);
        }

    } else if(strCommand == NetMsgType::PSSTATUSUPDATE) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("PSSTATUSUPDATE -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            return;
        }

        if(!infoMixingDynode.fInfoValid) return;
        if(infoMixingDynode.addr != pfrom->addr) {
            //LogPrintf("PSSTATUSUPDATE -- message doesn't match current Dynode: infoMixingDynode %s addr %s\n", infoMixingDynode.addr.ToString(), pfrom->addr.ToString());
            return;
        }

        int nMsgSessionID;
        int nMsgState;
        int nMsgEntriesCount;
        int nMsgStatusUpdate;
        int nMsgMessageID;
        vRecv >> nMsgSessionID >> nMsgState >> nMsgEntriesCount >> nMsgStatusUpdate >> nMsgMessageID;

        LogPrint("privatesend", "PSSTATUSUPDATE -- nMsgSessionID %d  nMsgState: %d  nEntriesCount: %d  nMsgStatusUpdate: %d  nMsgMessageID %d\n",
                nMsgSessionID, nMsgState, nEntriesCount, nMsgStatusUpdate, nMsgMessageID);

        if(nMsgState < POOL_STATE_MIN || nMsgState > POOL_STATE_MAX) {
            LogPrint("privatesend", "PSSTATUSUPDATE -- nMsgState is out of bounds: %d\n", nMsgState);
            return;
        }

        if(nMsgStatusUpdate < STATUS_REJECTED || nMsgStatusUpdate > STATUS_ACCEPTED) {
            LogPrint("privatesend", "PSSTATUSUPDATE -- nMsgStatusUpdate is out of bounds: %d\n", nMsgStatusUpdate);
            return;
        }

        if(nMsgMessageID < MSG_POOL_MIN || nMsgMessageID > MSG_POOL_MAX) {
            LogPrint("privatesend", "PSSTATUSUPDATE -- nMsgMessageID is out of bounds: %d\n", nMsgMessageID);
            return;
        }

        LogPrint("privatesend", "PSSTATUSUPDATE -- GetMessageByID: %s\n", CPrivateSend::GetMessageByID(PoolMessage(nMsgMessageID)));

        if(!CheckPoolStateUpdate(PoolState(nMsgState), nMsgEntriesCount, PoolStatusUpdate(nMsgStatusUpdate), PoolMessage(nMsgMessageID), nMsgSessionID)) {
            LogPrint("privatesend", "PSSTATUSUPDATE -- CheckPoolStateUpdate failed\n");
        }

    } else if(strCommand == NetMsgType::PSFINALTX) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("PSFINALTX -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            return;
        }

        if(!infoMixingDynode.fInfoValid) return;
        if((CNetAddr)infoMixingDynode.addr != (CNetAddr)pfrom->addr) {
            //LogPrintf("PSFINALTX -- message doesn't match current Dynode: infoMixingDynode %s addr %s\n", infoMixingDynode.addr.ToString(), pfrom->addr.ToString());
            return;
        }

        int nMsgSessionID;
        CTransaction txNew;
        vRecv >> nMsgSessionID >> txNew;

        if(nSessionID != nMsgSessionID) {
            LogPrint("privatesend", "PSFINALTX -- message doesn't match current PrivateSend session: nSessionID: %d  nMsgSessionID: %d\n", nSessionID, nMsgSessionID);
            return;
        }

        LogPrint("privatesend", "PSFINALTX -- txNew %s", txNew.ToString());

        //check to see if input is spent already? (and probably not confirmed)
        SignFinalTransaction(txNew, pfrom, connman);

    } else if(strCommand == NetMsgType::PSCOMPLETE) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("PSCOMPLETE -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            return;
        }

        if(!infoMixingDynode.fInfoValid) return;
        if(infoMixingDynode.addr != pfrom->addr) {
            LogPrint("privatesend", "PSCOMPLETE -- message doesn't match current Dynode: infoMixingDynode=%s  addr=%s\n", infoMixingDynode.addr.ToString(), pfrom->addr.ToString());
            return;
        }

        int nMsgSessionID;
        int nMsgMessageID;
        vRecv >> nMsgSessionID >> nMsgMessageID;

        if(nMsgMessageID < MSG_POOL_MIN || nMsgMessageID > MSG_POOL_MAX) {
            LogPrint("privatesend", "PSCOMPLETE -- nMsgMessageID is out of bounds: %d\n", nMsgMessageID);
            return;
        }

        if(nSessionID != nMsgSessionID) {
            LogPrint("privatesend", "PSCOMPLETE -- message doesn't match current PrivateSend session: nSessionID: %d  nMsgSessionID: %d\n", nSessionID, nMsgSessionID);
            return;
        }

        LogPrint("privatesend", "PSCOMPLETE -- nMsgSessionID %d  nMsgMessageID %d (%s)\n", nMsgSessionID, nMsgMessageID, CPrivateSend::GetMessageByID(PoolMessage(nMsgMessageID)));

        CompletedTransaction(PoolMessage(nMsgMessageID));
    }
}

void CPrivateSendClient::ResetPool()
{
    nCachedLastSuccessBlock = 0;
    txMyCollateral = CMutableTransaction();
    vecDynodesUsed.clear();
    UnlockCoins();
    keyHolderStorage.ReturnAll();
    SetNull();
}

void CPrivateSendClient::SetNull()
{
    // Client side
    nEntriesCount = 0;
    fLastEntryAccepted = false;
    infoMixingDynode = dynode_info_t();

    CPrivateSendBase::SetNull();
}

//
// Unlock coins after mixing fails or succeeds
//
void CPrivateSendClient::UnlockCoins()
{
    while(true) {
        TRY_LOCK(pwalletMain->cs_wallet, lockWallet);
        if(!lockWallet) {MilliSleep(50); continue;}
        BOOST_FOREACH(COutPoint outpoint, vecOutPointLocked)
            pwalletMain->UnlockCoin(outpoint);
        break;
    }

    vecOutPointLocked.clear();
}

std::string CPrivateSendClient::GetStatus()
{
    static int nStatusMessageProgress = 0;
    nStatusMessageProgress += 10;
    std::string strSuffix = "";

    if(WaitForAnotherBlock() || !dynodeSync.IsBlockchainSynced())
        return strAutoDenomResult;

    switch(nState) {
        case POOL_STATE_IDLE:
            return _("PrivateSend is idle.");
        case POOL_STATE_QUEUE:
            if(     nStatusMessageProgress % 70 <= 30) strSuffix = ".";
            else if(nStatusMessageProgress % 70 <= 50) strSuffix = "..";
            else if(nStatusMessageProgress % 70 <= 70) strSuffix = "...";
            return strprintf(_("Submitted to dynode, waiting in queue %s"), strSuffix);;
        case POOL_STATE_ACCEPTING_ENTRIES:
            if(nEntriesCount == 0) {
                nStatusMessageProgress = 0;
                return strAutoDenomResult;
            } else if(fLastEntryAccepted) {
                if(nStatusMessageProgress % 10 > 8) {
                    fLastEntryAccepted = false;
                    nStatusMessageProgress = 0;
                }
                return _("PrivateSend request complete:") + " " + _("Your transaction was accepted into the pool!");
            } else {
                if(     nStatusMessageProgress % 70 <= 40) return strprintf(_("Submitted following entries to dynode: %u / %d"), nEntriesCount, CPrivateSend::GetMaxPoolTransactions());
                else if(nStatusMessageProgress % 70 <= 50) strSuffix = ".";
                else if(nStatusMessageProgress % 70 <= 60) strSuffix = "..";
                else if(nStatusMessageProgress % 70 <= 70) strSuffix = "...";
                return strprintf(_("Submitted to dynode, waiting for more entries ( %u / %d ) %s"), nEntriesCount, CPrivateSend::GetMaxPoolTransactions(), strSuffix);
            }
        case POOL_STATE_SIGNING:
            if(     nStatusMessageProgress % 70 <= 40) return _("Found enough users, signing ...");
            else if(nStatusMessageProgress % 70 <= 50) strSuffix = ".";
            else if(nStatusMessageProgress % 70 <= 60) strSuffix = "..";
            else if(nStatusMessageProgress % 70 <= 70) strSuffix = "...";
            return strprintf(_("Found enough users, signing ( waiting %s )"), strSuffix);
        case POOL_STATE_ERROR:
            return _("PrivateSend request incomplete:") + " " + strLastMessage + " " + _("Will retry...");
        case POOL_STATE_SUCCESS:
            return _("PrivateSend request complete:") + " " + strLastMessage;
       default:
            return strprintf(_("Unknown state: id = %u"), nState);
    }
}

//
// Check the mixing progress and send client updates if a Dynode
//
void CPrivateSendClient::CheckPool()
{
    // reset if we're here for 10 seconds
    if((nState == POOL_STATE_ERROR || nState == POOL_STATE_SUCCESS) && GetTimeMillis() - nTimeLastSuccessfulStep >= 10000) {
        LogPrint("privatesend", "CPrivateSendClient::CheckPool -- timeout, RESETTING\n");
        UnlockCoins();
        if (nState == POOL_STATE_ERROR) {
            keyHolderStorage.ReturnAll();
        } else {
            keyHolderStorage.KeepAll();
        }
        SetNull();
    }
}

//
// Check for various timeouts (queue objects, mixing, etc)
//
void CPrivateSendClient::CheckTimeout()
{
    {
        TRY_LOCK(cs_privatesend, lockDS);
        if(!lockDS) return; // it's ok to fail here, we run this quite frequently

        // check mixing queue objects for timeouts
        std::vector<CPrivatesendQueue>::iterator it = vecPrivatesendQueue.begin();
        while(it != vecPrivatesendQueue.end()) {
            if((*it).IsExpired()) {
                LogPrint("privatesend", "CPrivateSendClient::CheckTimeout -- Removing expired queue (%s)\n", (*it).ToString());
                it = vecPrivatesendQueue.erase(it);
            } else ++it;
        }
    }

    if(!fEnablePrivateSend && !fDyNode) return;

    // catching hanging sessions
    if(!fDyNode) {
        switch(nState) {
            case POOL_STATE_ERROR:
                LogPrint("privatesend", "CPrivateSendClient::CheckTimeout -- Pool error -- Running CheckPool\n");
                CheckPool();
                break;
            case POOL_STATE_SUCCESS:
                LogPrint("privatesend", "CPrivateSendClient::CheckTimeout -- Pool success -- Running CheckPool\n");
                CheckPool();
                break;
            default:
                break;
        }
    }

    int nLagTime = fDyNode ? 0 : 10000; // if we're the client, give the server a few extra seconds before resetting.
    int nTimeout = (nState == POOL_STATE_SIGNING) ? PRIVATESEND_SIGNING_TIMEOUT : PRIVATESEND_QUEUE_TIMEOUT;
    bool fTimeout = GetTimeMillis() - nTimeLastSuccessfulStep >= nTimeout*1000 + nLagTime;

    if(nState != POOL_STATE_IDLE && fTimeout) {
        LogPrint("privatesend", "CPrivateSendClient::CheckTimeout -- %s timed out (%ds) -- restting\n",
                (nState == POOL_STATE_SIGNING) ? "Signing" : "Session", nTimeout);
        UnlockCoins();
        keyHolderStorage.ReturnAll();
        SetNull();
        SetState(POOL_STATE_ERROR);
        strLastMessage = _("Session timed out.");
    }
}

//
// Execute a mixing denomination via a Dynode.
// This is only ran from clients
//
bool CPrivateSendClient::SendDenominate(const std::vector<CTxIn>& vecTxIn, const std::vector<CTxOut>& vecTxOut, CConnman& connman)
{
    if(fDyNode) {
        LogPrintf("CPrivateSendClient::SendDenominate -- PrivateSend from a Dynode is not supported currently.\n");
        return false;
    }

    if(txMyCollateral == CMutableTransaction()) {
        LogPrintf("CPrivateSendClient:SendDenominate -- PrivateSend collateral not set\n");
        return false;
    }

    // lock the funds we're going to use
    BOOST_FOREACH(CTxIn txin, txMyCollateral.vin)
        vecOutPointLocked.push_back(txin.prevout);

    BOOST_FOREACH(CTxIn txin, vecTxIn)
        vecOutPointLocked.push_back(txin.prevout);

    // we should already be connected to a Dynode
    if(!nSessionID) {
        LogPrintf("CPrivateSendClient::SendDenominate -- No Dynode has been selected yet.\n");
        UnlockCoins();
        keyHolderStorage.ReturnAll();
        SetNull();
        return false;
    }

    if(!CheckDiskSpace()) {
        UnlockCoins();
        keyHolderStorage.ReturnAll();
        SetNull();
        fEnablePrivateSend = false;
        LogPrintf("CPrivateSendClient::SendDenominate -- Not enough disk space, disabling PrivateSend.\n");
        return false;
    }

    SetState(POOL_STATE_ACCEPTING_ENTRIES);
    strLastMessage = "";

    LogPrintf("CPrivateSendClient::SendDenominate -- Added transaction to pool.\n");

    //check it against the memory pool to make sure it's valid
    {
        CValidationState validationState;
        CMutableTransaction tx;

        BOOST_FOREACH(const CTxIn& txin, vecTxIn) {
            LogPrint("privatesend", "CPrivateSendClient::SendDenominate -- txin=%s\n", txin.ToString());
            tx.vin.push_back(txin);
        }

        BOOST_FOREACH(const CTxOut& txout, vecTxOut) {
            LogPrint("privatesend", "CPrivateSendClient::SendDenominate -- txout=%s\n", txout.ToString());
            tx.vout.push_back(txout);
        }

        LogPrintf("CPrivateSendClient::SendDenominate -- Submitting partial tx %s", tx.ToString());

        mempool.PrioritiseTransaction(tx.GetHash(), tx.GetHash().ToString(), 1000, 0.1*COIN);
        TRY_LOCK(cs_main, lockMain);
        if(!lockMain || !AcceptToMemoryPool(mempool, validationState, CTransaction(tx), false, NULL, false, true, true)) {
            LogPrintf("CPrivateSendClient::SendDenominate -- AcceptToMemoryPool() failed! tx=%s", tx.ToString());
            UnlockCoins();
            keyHolderStorage.ReturnAll();
            SetNull();
            return false;
        }
    }

    // store our entry for later use
    CPrivateSendEntry entry(vecTxIn, vecTxOut, txMyCollateral);
    vecEntries.push_back(entry);
    RelayIn(entry, connman);
    nTimeLastSuccessfulStep = GetTimeMillis();

    return true;
}

// Incoming message from Dynode updating the progress of mixing
bool CPrivateSendClient::CheckPoolStateUpdate(PoolState nStateNew, int nEntriesCountNew, PoolStatusUpdate nStatusUpdate, PoolMessage nMessageID, int nSessionIDNew)
{
    if(fDyNode) return false;

    // do not update state when mixing client state is one of these
    if(nState == POOL_STATE_IDLE || nState == POOL_STATE_ERROR || nState == POOL_STATE_SUCCESS) return false;

    strAutoDenomResult = _("Dynode:") + " " + CPrivateSend::GetMessageByID(nMessageID);

    // if rejected at any state
    if(nStatusUpdate == STATUS_REJECTED) {
        LogPrintf("CPrivateSendClient::CheckPoolStateUpdate -- entry is rejected by Dynode\n");
        UnlockCoins();
        keyHolderStorage.ReturnAll();
        SetNull();
        SetState(POOL_STATE_ERROR);
        strLastMessage = CPrivateSend::GetMessageByID(nMessageID);
        return true;
    }

    if(nStatusUpdate == STATUS_ACCEPTED && nState == nStateNew) {
        if(nStateNew == POOL_STATE_QUEUE && nSessionID == 0 && nSessionIDNew != 0) {
            // new session id should be set only in POOL_STATE_QUEUE state
            nSessionID = nSessionIDNew;
            nTimeLastSuccessfulStep = GetTimeMillis();
            LogPrintf("CPrivateSendClient::CheckPoolStateUpdate -- set nSessionID to %d\n", nSessionID);
            return true;
        }
        else if(nStateNew == POOL_STATE_ACCEPTING_ENTRIES && nEntriesCount != nEntriesCountNew) {
            nEntriesCount = nEntriesCountNew;
            nTimeLastSuccessfulStep = GetTimeMillis();
            fLastEntryAccepted = true;
            LogPrintf("CPrivateSendClient::CheckPoolStateUpdate -- new entry accepted!\n");
            return true;
        }
    }

    // only situations above are allowed, fail in any other case
    return false;
}

//
// After we receive the finalized transaction from the Dynode, we must
// check it to make sure it's what we want, then sign it if we agree.
// If we refuse to sign, it's possible we'll be charged collateral
//
bool CPrivateSendClient::SignFinalTransaction(const CTransaction& finalTransactionNew, CNode* pnode, CConnman& connman)
{
    if(fDyNode || pnode == NULL) return false;

    finalMutableTransaction = finalTransactionNew;
    LogPrintf("CPrivateSendClient::SignFinalTransaction -- finalMutableTransaction=%s", finalMutableTransaction.ToString());

    std::vector<CTxIn> sigs;

    //make sure my inputs/outputs are present, otherwise refuse to sign
    BOOST_FOREACH(const CPrivateSendEntry entry, vecEntries) {
        BOOST_FOREACH(const CTxPSIn txdsin, entry.vecTxPSIn) {
            /* Sign my transaction and all outputs */
            int nMyInputIndex = -1;
            CScript prevPubKey = CScript();
            CTxIn txin = CTxIn();

            for(unsigned int i = 0; i < finalMutableTransaction.vin.size(); i++) {
                if(finalMutableTransaction.vin[i] == txdsin) {
                    nMyInputIndex = i;
                    prevPubKey = txdsin.prevPubKey;
                    txin = txdsin;
                }
            }

            if(nMyInputIndex >= 0) { //might have to do this one input at a time?
                int nFoundOutputsCount = 0;
                CAmount nValue1 = 0;
                CAmount nValue2 = 0;

                for(unsigned int i = 0; i < finalMutableTransaction.vout.size(); i++) {
                    BOOST_FOREACH(const CTxOut& txout, entry.vecTxPSOut) {
                        if(finalMutableTransaction.vout[i] == txout) {
                            nFoundOutputsCount++;
                            nValue1 += finalMutableTransaction.vout[i].nValue;
                        }
                    }
                }

                BOOST_FOREACH(const CTxOut txout, entry.vecTxPSOut)
                    nValue2 += txout.nValue;

                int nTargetOuputsCount = entry.vecTxPSOut.size();
                if(nFoundOutputsCount < nTargetOuputsCount || nValue1 != nValue2) {
                    // in this case, something went wrong and we'll refuse to sign. It's possible we'll be charged collateral. But that's
                    // better then signing if the transaction doesn't look like what we wanted.
                    LogPrintf("CPrivateSendClient::SignFinalTransaction -- My entries are not correct! Refusing to sign: nFoundOutputsCount: %d, nTargetOuputsCount: %d\n", nFoundOutputsCount, nTargetOuputsCount);
                    UnlockCoins();
                    keyHolderStorage.ReturnAll();
                    SetNull();

                    return false;
                }

                const CKeyStore& keystore = *pwalletMain;

                LogPrint("privatesend", "CPrivateSendClient::SignFinalTransaction -- Signing my input %i\n", nMyInputIndex);
                if(!SignSignature(keystore, prevPubKey, finalMutableTransaction, nMyInputIndex, int(SIGHASH_ALL|SIGHASH_ANYONECANPAY))) { // changes scriptSig
                    LogPrint("privatesend", "CPrivateSendClient::SignFinalTransaction -- Unable to sign my own transaction!\n");
                    // not sure what to do here, it will timeout...?
                }

                sigs.push_back(finalMutableTransaction.vin[nMyInputIndex]);
                LogPrint("privatesend", "CPrivateSendClient::SignFinalTransaction -- nMyInputIndex: %d, sigs.size(): %d, scriptSig=%s\n", nMyInputIndex, (int)sigs.size(), ScriptToAsmStr(finalMutableTransaction.vin[nMyInputIndex].scriptSig));
            }
        }
    }

    if(sigs.empty()) {
        LogPrintf("CPrivateSendClient::SignFinalTransaction -- can't sign anything!\n");
        UnlockCoins();
        keyHolderStorage.ReturnAll();
        SetNull();

        return false;
    }

    // push all of our signatures to the Dynode
    LogPrintf("CPrivateSendClient::SignFinalTransaction -- pushing sigs to the dynode, finalMutableTransaction=%s", finalMutableTransaction.ToString());
    g_connman->PushMessage(pnode, NetMsgType::PSSIGNFINALTX, sigs);
    SetState(POOL_STATE_SIGNING);
    nTimeLastSuccessfulStep = GetTimeMillis();

    return true;
}

void CPrivateSendClient::NewBlock()
{
    static int64_t nTimeNewBlockReceived = 0;

    //we we're processing lots of blocks, we'll just leave
    if(GetTime() - nTimeNewBlockReceived < 10) return;
    nTimeNewBlockReceived = GetTime();
    LogPrint("privatesend", "CPrivateSendClient::NewBlock\n");

    CheckTimeout();
}

// mixing transaction was completed (failed or successful)
void CPrivateSendClient::CompletedTransaction(PoolMessage nMessageID)
{
    if(fDyNode) return;

    if(nMessageID == MSG_SUCCESS) {
        LogPrintf("CompletedTransaction -- success\n");
        nCachedLastSuccessBlock = nCachedBlockHeight;
        keyHolderStorage.KeepAll();
    } else {
        LogPrintf("CompletedTransaction -- error\n");
        keyHolderStorage.ReturnAll();
    }
    UnlockCoins();
    SetNull();
    strLastMessage = CPrivateSend::GetMessageByID(nMessageID);
}

bool CPrivateSendClient::WaitForAnotherBlock()
{
    if(!dynodeSync.IsDynodeListSynced())
        return true;

    if(fPrivateSendMultiSession)
        return false;

    return nCachedBlockHeight - nCachedLastSuccessBlock < nMinBlocksToWait;
}

bool CPrivateSendClient::CheckAutomaticBackup()
{
    switch(nWalletBackups) {
        case 0:
            LogPrint("privatesend", "CPrivateSendClient::CheckAutomaticBackup -- Automatic backups disabled, no mixing available.\n");
            strAutoDenomResult = _("Automatic backups disabled") + ", " + _("no mixing available.");
            fEnablePrivateSend = false; // stop mixing
            pwalletMain->nKeysLeftSinceAutoBackup = 0; // no backup, no "keys since last backup"
            return false;
        case -1:
            // Automatic backup failed, nothing else we can do until user fixes the issue manually.
            // There is no way to bring user attention in daemon mode so we just update status and
            // keep spaming if debug is on.
            LogPrint("privatesend", "CPrivateSendClient::CheckAutomaticBackup -- ERROR! Failed to create automatic backup.\n");
            strAutoDenomResult = _("ERROR! Failed to create automatic backup") + ", " + _("see debug.log for details.");
            return false;
        case -2:
            // We were able to create automatic backup but keypool was not replenished because wallet is locked.
            // There is no way to bring user attention in daemon mode so we just update status and
            // keep spaming if debug is on.
            LogPrint("privatesend", "CPrivateSendClient::CheckAutomaticBackup -- WARNING! Failed to create replenish keypool, please unlock your wallet to do so.\n");
            strAutoDenomResult = _("WARNING! Failed to replenish keypool, please unlock your wallet to do so.") + ", " + _("see debug.log for details.");
            return false;
    }

    if(pwalletMain->nKeysLeftSinceAutoBackup < PRIVATESEND_KEYS_THRESHOLD_STOP) {
        // We should never get here via mixing itself but probably smth else is still actively using keypool
        LogPrint("privatesend", "CPrivateSendClient::CheckAutomaticBackup -- Very low number of keys left: %d, no mixing available.\n", pwalletMain->nKeysLeftSinceAutoBackup);
        strAutoDenomResult = strprintf(_("Very low number of keys left: %d") + ", " + _("no mixing available."), pwalletMain->nKeysLeftSinceAutoBackup);
        // It's getting really dangerous, stop mixing
        fEnablePrivateSend = false;
        return false;
    } else if(pwalletMain->nKeysLeftSinceAutoBackup < PRIVATESEND_KEYS_THRESHOLD_WARNING) {
        // Low number of keys left but it's still more or less safe to continue
        LogPrint("privatesend", "CPrivateSendClient::CheckAutomaticBackup -- Very low number of keys left: %d\n", pwalletMain->nKeysLeftSinceAutoBackup);
        strAutoDenomResult = strprintf(_("Very low number of keys left: %d"), pwalletMain->nKeysLeftSinceAutoBackup);

        if(fCreateAutoBackups) {
            LogPrint("privatesend", "CPrivateSendClient::CheckAutomaticBackup -- Trying to create new backup.\n");
            std::string warningString;
            std::string errorString;

            if(!AutoBackupWallet(pwalletMain, "", warningString, errorString)) {
                if(!warningString.empty()) {
                    // There were some issues saving backup but yet more or less safe to continue
                    LogPrintf("CPrivateSendClient::CheckAutomaticBackup -- WARNING! Something went wrong on automatic backup: %s\n", warningString);
                }
                if(!errorString.empty()) {
                    // Things are really broken
                    LogPrintf("CPrivateSendClient::CheckAutomaticBackup -- ERROR! Failed to create automatic backup: %s\n", errorString);
                    strAutoDenomResult = strprintf(_("ERROR! Failed to create automatic backup") + ": %s", errorString);
                    return false;
                }
            }
        } else {
            // Wait for smth else (e.g. GUI action) to create automatic backup for us
            return false;
        }
    }

    LogPrint("privatesend", "CPrivateSendClient::CheckAutomaticBackup -- Keys left since latest backup: %d\n", pwalletMain->nKeysLeftSinceAutoBackup);

    return true;
}

//
// Passively run mixing in the background to anonymize funds based on the given configuration.
//
bool CPrivateSendClient::DoAutomaticDenominating(CConnman& connman, bool fDryRun)
{
    if(fDyNode) return false; // no client-side mixing on dynodes
    if(!fEnablePrivateSend) return false;
    if(!pwalletMain || pwalletMain->IsLocked(true)) return false;
    if(nState != POOL_STATE_IDLE) return false;

    if(!dynodeSync.IsDynodeListSynced()) {
        strAutoDenomResult = _("Can't mix while sync in progress.");
        return false;
    }

    if(!CheckAutomaticBackup())
        return false;

    if(GetEntriesCount() > 0) {
        strAutoDenomResult = _("Mixing in progress...");
        return false;
    }

    TRY_LOCK(cs_privatesend, lockDS);
    if(!lockDS) {
        strAutoDenomResult = _("Lock is already in place.");
        return false;
    }

    if(!fDryRun && pwalletMain->IsLocked(true)) {
        strAutoDenomResult = _("Wallet is locked.");
        return false;
    }

    if(WaitForAnotherBlock()) {
        LogPrintf("CPrivateSendClient::DoAutomaticDenominating -- Last successful PrivateSend action was too recent\n");
        strAutoDenomResult = _("Last successful PrivateSend action was too recent.");
        return false;
    }

    if(dnodeman.size() == 0) {
        LogPrint("privatesend", "CPrivateSendClient::DoAutomaticDenominating -- No Dynodes detected\n");
        strAutoDenomResult = _("No Dynodes detected.");
        return false;
    }

    CAmount nValueMin = CPrivateSend::GetSmallestDenomination();

    // if there are no confirmed DS collateral inputs yet
    if(!pwalletMain->HasCollateralInputs()) {
        // should have some additional amount for them
        nValueMin += CPrivateSend::GetMaxCollateralAmount();
    }

    // including denoms but applying some restrictions
    CAmount nBalanceNeedsAnonymized = pwalletMain->GetNeedsToBeAnonymizedBalance(nValueMin);

    // anonymizable balance is way too small
    if(nBalanceNeedsAnonymized < nValueMin) {
        LogPrintf("CPrivateSendClient::DoAutomaticDenominating -- Not enough funds to anonymize\n");
        strAutoDenomResult = _("Not enough funds to anonymize.");
        return false;
    }

    // excluding denoms
    CAmount nBalanceAnonimizableNonDenom = pwalletMain->GetAnonymizableBalance(true);
    // denoms
    CAmount nBalanceDenominatedConf = pwalletMain->GetDenominatedBalance();
    CAmount nBalanceDenominatedUnconf = pwalletMain->GetDenominatedBalance(true);
    CAmount nBalanceDenominated = nBalanceDenominatedConf + nBalanceDenominatedUnconf;

    LogPrint("privatesend", "CPrivateSendClient::DoAutomaticDenominating -- nValueMin: %f, nBalanceNeedsAnonymized: %f, nBalanceAnonimizableNonDenom: %f, nBalanceDenominatedConf: %f, nBalanceDenominatedUnconf: %f, nBalanceDenominated: %f\n",
            (float)nValueMin/COIN,
            (float)nBalanceNeedsAnonymized/COIN,
            (float)nBalanceAnonimizableNonDenom/COIN,
            (float)nBalanceDenominatedConf/COIN,
            (float)nBalanceDenominatedUnconf/COIN,
            (float)nBalanceDenominated/COIN);

    if(fDryRun) return true;

    // Check if we have should create more denominated inputs i.e.
    // there are funds to denominate and denominated balance does not exceed
    // max amount to mix yet.
    if(nBalanceAnonimizableNonDenom >= nValueMin + CPrivateSend::GetCollateralAmount() && nBalanceDenominated < nPrivateSendAmount*COIN)
        return CreateDenominated(connman);

    //check if we have the collateral sized inputs
    if(!pwalletMain->HasCollateralInputs())
        return !pwalletMain->HasCollateralInputs(false) && MakeCollateralAmounts(connman);

    if(nSessionID) {
        strAutoDenomResult = _("Mixing in progress...");
        return false;
    }

    // Initial phase, find a Dynode
    // Clean if there is anything left from previous session
    UnlockCoins();
    keyHolderStorage.ReturnAll();
    SetNull();

    // should be no unconfirmed denoms in non-multi-session mode
    if(!fPrivateSendMultiSession && nBalanceDenominatedUnconf > 0) {
        LogPrintf("CPrivateSendClient::DoAutomaticDenominating -- Found unconfirmed denominated outputs, will wait till they confirm to continue.\n");
        strAutoDenomResult = _("Found unconfirmed denominated outputs, will wait till they confirm to continue.");
        return false;
    }

    //check our collateral and create new if needed
    std::string strReason;
    if(txMyCollateral == CMutableTransaction()) {
        if(!pwalletMain->CreateCollateralTransaction(txMyCollateral, strReason)) {
            LogPrintf("CPrivateSendClient::DoAutomaticDenominating -- create collateral error:%s\n", strReason);
            return false;
        }
    } else {
        if(!CPrivateSend::IsCollateralValid(txMyCollateral)) {
            LogPrintf("CPrivateSendClient::DoAutomaticDenominating -- invalid collateral, recreating...\n");
            if(!pwalletMain->CreateCollateralTransaction(txMyCollateral, strReason)) {
                LogPrintf("CPrivateSendClient::DoAutomaticDenominating -- create collateral error: %s\n", strReason);
                return false;
            }
        }
    }

    int nDnCountEnabled = dnodeman.CountEnabled(MIN_PRIVATESEND_PEER_PROTO_VERSION);

    // If we've used 90% of the Dynode list then drop the oldest first ~30%
    int nThreshold_high = nDnCountEnabled * 0.9;
    int nThreshold_low = nThreshold_high * 0.7;
    LogPrint("privatesend", "Checking vecDynodesUsed: size: %d, threshold: %d\n", (int)vecDynodesUsed.size(), nThreshold_high);

    if((int)vecDynodesUsed.size() > nThreshold_high) {
        vecDynodesUsed.erase(vecDynodesUsed.begin(), vecDynodesUsed.begin() + vecDynodesUsed.size() - nThreshold_low);
        LogPrint("privatesend", "  vecDynodesUsed: new size: %d, threshold: %d\n", (int)vecDynodesUsed.size(), nThreshold_high);
    }

    bool fUseQueue = GetRandInt(100) > 33;
    // don't use the queues all of the time for mixing unless we are a liquidity provider
    if((nLiquidityProvider || fUseQueue) && JoinExistingQueue(nBalanceNeedsAnonymized, connman))
        return true;

    // do not initiate queue if we are a liquidity provider to avoid useless inter-mixing
    if(nLiquidityProvider) return false;

    if(StartNewQueue(nValueMin, nBalanceNeedsAnonymized, connman))
        return true;

    strAutoDenomResult = _("No compatible Dynode found.");
    return false;
}

bool CPrivateSendClient::JoinExistingQueue(CAmount nBalanceNeedsAnonymized, CConnman& connman)
{
    std::vector<CAmount> vecStandardDenoms = CPrivateSend::GetStandardDenominations();
    // Look through the queues and see if anything matches
    BOOST_FOREACH(CPrivatesendQueue& psq, vecPrivatesendQueue) {
        // only try each queue once
        if(psq.fTried) continue;
        psq.fTried = true;

        if(psq.IsExpired()) continue;

        dynode_info_t infoDn;

        if(!dnodeman.GetDynodeInfo(psq.vin.prevout, infoDn)) {
            LogPrintf("CPrivateSendClient::JoinExistingQueue -- psq dynode is not in dynode list, dynode=%s\n", psq.vin.prevout.ToStringShort());
            continue;
        }

        if(infoDn.nProtocolVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) continue;

        std::vector<int> vecBits;
        if(!CPrivateSend::GetDenominationsBits(psq.nDenom, vecBits)) {
            // incompatible denom
            continue;
        }

        // mixing rate limit i.e. nLastPsq check should already pass in DSQUEUE ProcessMessage
        // in order for psq to get into vecPrivatesendQueue, so we should be safe to mix already,
        // no need for additional verification here

        LogPrint("privatesend", "CPrivateSendClient::JoinExistingQueue -- found valid queue: %s\n", psq.ToString());

        CAmount nValueInTmp = 0;
        std::vector<CTxIn> vecTxInTmp;
        std::vector<COutput> vCoinsTmp;

        // Try to match their denominations if possible, select at least 1 denominations
        if(!pwalletMain->SelectCoinsByDenominations(psq.nDenom, vecStandardDenoms[vecBits.front()], nBalanceNeedsAnonymized, vecTxInTmp, vCoinsTmp, nValueInTmp, 0, nPrivateSendRounds)) {
            LogPrintf("CPrivateSendClient::JoinExistingQueue -- Couldn't match denominations %d %d (%s)\n", vecBits.front(), psq.nDenom, CPrivateSend::GetDenominationsToString(psq.nDenom));
            continue;
        }

        vecDynodesUsed.push_back(psq.vin.prevout);

        CNode* pnodeFound = NULL;
        bool fDisconnect = false;
        g_connman->ForNode(infoDn.addr, CConnman::AllNodes, [&pnodeFound, &fDisconnect](CNode* pnode) {
            pnodeFound = pnode;
            if(pnodeFound->fDisconnect) {
                fDisconnect = true;
            } else {
                pnodeFound->AddRef();
            }
            return true;
        });
        if (fDisconnect)
            continue;

        LogPrintf("CPrivateSendClient::JoinExistingQueue -- attempt to connect to dynode from queue, addr=%s\n", infoDn.addr.ToString());
        // connect to Dynode and submit the queue request
        // TODO: Pass CConnman instance somehow and don't use global variable.
        CNode* pnode = (pnodeFound && pnodeFound->fDynode) ? pnodeFound : g_connman->ConnectNode(CAddress(infoDn.addr, NODE_NETWORK), NULL, true);
        if(pnode) {
            infoMixingDynode = infoDn;
            nSessionDenom = psq.nDenom;

            g_connman->PushMessage(pnode, NetMsgType::PSACCEPT, nSessionDenom, txMyCollateral);
            LogPrintf("CPrivateSendClient::JoinExistingQueue -- connected (from queue), sending PSACCEPT: nSessionDenom: %d (%s), addr=%s\n",
                    nSessionDenom, CPrivateSend::GetDenominationsToString(nSessionDenom), pnode->addr.ToString());
            strAutoDenomResult = _("Mixing in progress...");
            SetState(POOL_STATE_QUEUE);
            nTimeLastSuccessfulStep = GetTimeMillis();
            if(pnodeFound) {
                pnodeFound->Release();
            }
            return true;
        } else {
            LogPrintf("CPrivateSendClient::JoinExistingQueue -- can't connect, addr=%s\n", infoDn.addr.ToString());
            strAutoDenomResult = _("Error connecting to Dynode.");
            continue;
        }
    }
    return false;
}

bool CPrivateSendClient::StartNewQueue(CAmount nValueMin, CAmount nBalanceNeedsAnonymized, CConnman& connman)
{
    int nTries = 0;
    int nDnCountEnabled = dnodeman.CountEnabled(MIN_PRIVATESEND_PEER_PROTO_VERSION);

    // ** find the coins we'll use
    std::vector<CTxIn> vecTxIn;
    CAmount nValueInTmp = 0;
    if(!pwalletMain->SelectCoinsMix(nValueMin, nBalanceNeedsAnonymized, vecTxIn, nValueInTmp, 0, nPrivateSendRounds)) {
        // this should never happen
        LogPrintf("CPrivateSendClient::StartNewQueue -- Can't mix: no compatible inputs found!\n");
        strAutoDenomResult = _("Can't mix: no compatible inputs found!");
        return false;
    }

    // otherwise, try one randomly
    while(nTries < 10) {
        dynode_info_t infoDn = dnodeman.FindRandomNotInVec(vecDynodesUsed, MIN_PRIVATESEND_PEER_PROTO_VERSION);
        if(!infoDn.fInfoValid) {
            LogPrintf("CPrivateSendClient::StartNewQueue -- Can't find random dynode!\n");
            strAutoDenomResult = _("Can't find random Dynode.");
            return false;
        }
        vecDynodesUsed.push_back(infoDn.vin.prevout);

        if(infoDn.nLastPsq != 0 && infoDn.nLastPsq + nDnCountEnabled/5 > dnodeman.nPsqCount) {
            LogPrintf("CPrivateSendClient::StartNewQueue -- Too early to mix on this dynode!"
                        " dynode=%s  addr=%s  nLastPsq=%d  CountEnabled/5=%d  nPsqCount=%d\n",
                        infoDn.vin.prevout.ToStringShort(), infoDn.addr.ToString(), infoDn.nLastPsq,
                        nDnCountEnabled/5, dnodeman.nPsqCount);
            nTries++;
            continue;
        }

        CNode* pnodeFound = NULL;
        bool fDisconnect = false;
        g_connman->ForNode(infoDn.addr, CConnman::AllNodes, [&pnodeFound, &fDisconnect](CNode* pnode) {
            pnodeFound = pnode;
            if(pnodeFound->fDisconnect) {
                fDisconnect = true;
            } else {
                pnodeFound->AddRef();
            }
            return true;
        });
        if (fDisconnect) {
            nTries++;
            continue;
        }

        LogPrintf("CPrivateSendClient::StartNewQueue -- attempt %d connection to Dynode %s\n", nTries, infoDn.addr.ToString());
        // TODO: Pass CConnman instance somehow and don't use global variable.
        CNode* pnode = (pnodeFound && pnodeFound->fDynode) ? pnodeFound : g_connman->ConnectNode(CAddress(infoDn.addr, NODE_NETWORK), NULL, true);
        if(pnode) {
            LogPrintf("CPrivateSendClient::StartNewQueue -- connected, addr=%s\n", infoDn.addr.ToString());
            infoMixingDynode = infoDn;

            std::vector<CAmount> vecAmounts;
            pwalletMain->ConvertList(vecTxIn, vecAmounts);
            // try to get a single random denom out of vecAmounts
            while(nSessionDenom == 0) {
                nSessionDenom = CPrivateSend::GetDenominationsByAmounts(vecAmounts);
            }

            g_connman->PushMessage(pnode, NetMsgType::PSACCEPT, nSessionDenom, txMyCollateral);
            LogPrintf("CPrivateSendClient::StartNewQueue -- connected, sending PSACCEPT, nSessionDenom: %d (%s)\n",
                    nSessionDenom, CPrivateSend::GetDenominationsToString(nSessionDenom));
            strAutoDenomResult = _("Mixing in progress...");
            SetState(POOL_STATE_QUEUE);
            nTimeLastSuccessfulStep = GetTimeMillis();
            if(pnodeFound) {
                pnodeFound->Release();
            }
            return true;
        } else {
            LogPrintf("CPrivateSendClient::StartNewQueue -- can't connect, addr=%s\n", infoDn.addr.ToString());
            nTries++;
            continue;
        }
    }
    return false;
}

bool CPrivateSendClient::SubmitDenominate(CConnman& connman)
{
    std::string strError;
    std::vector<CTxIn> vecTxInRet;
    std::vector<CTxOut> vecTxOutRet;

    // Submit transaction to the pool if we get here
    // Try to use only inputs with the same number of rounds starting from the highest number of rounds possible
    for(int i = nPrivateSendRounds; i > 0; i--) {
        if(PrepareDenominate(i - 1, i, strError, vecTxInRet, vecTxOutRet)) {
            LogPrintf("CPrivateSendClient::SubmitDenominate -- Running PrivateSend denominate for %d rounds, success\n", i);
            return SendDenominate(vecTxInRet, vecTxOutRet, connman);
        }
        LogPrint("privatesend", "CPrivateSendClient::SubmitDenominate -- Running PrivateSend denominate for %d rounds, error: %s\n", i, strError);
    }

    // We failed? That's strange but let's just make final attempt and try to mix everything
    if(PrepareDenominate(0, nPrivateSendRounds, strError, vecTxInRet, vecTxOutRet)) {
        LogPrintf("CPrivateSendClient::SubmitDenominate -- Running PrivateSend denominate for all rounds, success\n");
        return SendDenominate(vecTxInRet, vecTxOutRet, connman);
    }

    // Should never actually get here but just in case
    LogPrintf("CPrivateSendClient::SubmitDenominate -- Running PrivateSend denominate for all rounds, error: %s\n", strError);
    strAutoDenomResult = strError;
    return false;
}

bool CPrivateSendClient::PrepareDenominate(int nMinRounds, int nMaxRounds, std::string& strErrorRet, std::vector<CTxIn>& vecTxInRet, std::vector<CTxOut>& vecTxOutRet)
{
    if(!pwalletMain) {
        strErrorRet = "Wallet is not initialized";
        return false;
    }

    if (pwalletMain->IsLocked(true)) {
        strErrorRet = "Wallet locked, unable to create transaction!";
        return false;
    }

    if (GetEntriesCount() > 0) {
        strErrorRet = "Already have pending entries in the PrivateSend pool";
        return false;
    }

    // make sure returning vectors are empty before filling them up
    vecTxInRet.clear();
    vecTxOutRet.clear();

    // ** find the coins we'll use
    std::vector<CTxIn> vecTxIn;
    std::vector<COutput> vCoins;
    CAmount nValueIn = 0;

    /*
        Select the coins we'll use

        if nMinRounds >= 0 it means only denominated inputs are going in and coming out
    */
    std::vector<int> vecBits;
    if (!CPrivateSend::GetDenominationsBits(nSessionDenom, vecBits)) {
        strErrorRet = "Incorrect session denom";
        return false;
    }
    std::vector<CAmount> vecStandardDenoms = CPrivateSend::GetStandardDenominations();
    bool fSelected = pwalletMain->SelectCoinsByDenominations(nSessionDenom, vecStandardDenoms[vecBits.front()], CPrivateSend::GetMaxPoolAmount(), vecTxIn, vCoins, nValueIn, nMinRounds, nMaxRounds);
    if (nMinRounds >= 0 && !fSelected) {
        strErrorRet = "Can't select current denominated inputs";
        return false;
    }

    LogPrintf("CPrivateSendClient::PrepareDenominate -- max value: %f\n", (double)nValueIn/COIN);

    {
        LOCK(pwalletMain->cs_wallet);
        BOOST_FOREACH(CTxIn txin, vecTxIn) {
            pwalletMain->LockCoin(txin.prevout);
        }
    }

    CAmount nValueLeft = nValueIn;

    // Try to add every needed denomination, repeat up to 5-PRIVATESEND_ENTRY_MAX_SIZE times.
    // NOTE: No need to randomize order of inputs because they were
    // initially shuffled in CWallet::SelectCoinsByDenominations already.
    int nStep = 0;
    int nStepsMax = 5 + GetRandInt(PRIVATESEND_ENTRY_MAX_SIZE-5+1);

    while (nStep < nStepsMax) {
        BOOST_FOREACH(int nBit, vecBits) {
            CAmount nValueDenom = vecStandardDenoms[nBit];
            if (nValueLeft - nValueDenom < 0) continue;

            // Note: this relies on a fact that both vectors MUST have same size
            std::vector<CTxIn>::iterator it = vecTxIn.begin();
            std::vector<COutput>::iterator it2 = vCoins.begin();
            while (it2 != vCoins.end()) {
                // we have matching inputs
                if ((*it2).tx->vout[(*it2).i].nValue == nValueDenom) {
                    // add new input in resulting vector
                    vecTxInRet.push_back(*it);
                    // remove corresponting items from initial vectors
                    vecTxIn.erase(it);
                    vCoins.erase(it2);

                    CScript scriptDenom = keyHolderStorage.AddKey(pwalletMain).GetScriptForDestination();

                    // add new output
                    CTxOut txout(nValueDenom, scriptDenom);
                    vecTxOutRet.push_back(txout);

                    // subtract denomination amount
                    nValueLeft -= nValueDenom;

                    // step is complete
                    break;
                }
                ++it;
                ++it2;
            }
        }
        if(nValueLeft == 0) break;
        nStep++;
    }

    {
        // unlock unused coins
        LOCK(pwalletMain->cs_wallet);
        BOOST_FOREACH(CTxIn txin, vecTxIn) {
            pwalletMain->UnlockCoin(txin.prevout);
        }
    }

    if (CPrivateSend::GetDenominations(vecTxOutRet) != nSessionDenom) {
        // unlock used coins on failure
        LOCK(pwalletMain->cs_wallet);
        BOOST_FOREACH(CTxIn txin, vecTxInRet) {
            pwalletMain->UnlockCoin(txin.prevout);
        }
        keyHolderStorage.ReturnAll();
        strErrorRet = "Can't make current denominated outputs";
        return false;
    }

    // We also do not care about full amount as long as we have right denominations
    return true;
}

// Create collaterals by looping through inputs grouped by addresses
bool CPrivateSendClient::MakeCollateralAmounts(CConnman& connman)
{
    std::vector<CompactTallyItem> vecTally;
    if(!pwalletMain->SelectCoinsGrouppedByAddresses(vecTally, false)) {
        LogPrint("privatesend", "CPrivateSendClient::MakeCollateralAmounts -- SelectCoinsGrouppedByAddresses can't find any inputs!\n");
        return false;
    }

    // First try to use only non-denominated funds
    BOOST_FOREACH(CompactTallyItem& item, vecTally) {
        if(!MakeCollateralAmounts(item, false, connman)) continue;
        return true;
    }

    // There should be at least some denominated funds we should be able to break in pieces to continue mixing
    BOOST_FOREACH(CompactTallyItem& item, vecTally) {
        if(!MakeCollateralAmounts(item, true, connman)) continue;
        return true;
    }

    // If we got here then smth is terribly broken actually
    LogPrintf("CPrivateSendClient::MakeCollateralAmounts -- ERROR: Can't make collaterals!\n");
    return false;
}

// Split up large inputs or create fee sized inputs
bool CPrivateSendClient::MakeCollateralAmounts(const CompactTallyItem& tallyItem, bool fTryDenominated, CConnman& connman)
{

    LOCK2(cs_main, pwalletMain->cs_wallet);

    // denominated input is always a single one, so we can check its amount directly and return early
    if(!fTryDenominated && tallyItem.vecTxIn.size() == 1 && pwalletMain->IsDenominatedAmount(tallyItem.nAmount))
        return false;

    CWalletTx wtx;
    CAmount nFeeRet = 0;
    int nChangePosRet = -1;
    std::string strFail = "";
    std::vector<CRecipient> vecSend;

    // make our collateral address
    CReserveKey reservekeyCollateral(pwalletMain);
    // make our change address
    CReserveKey reservekeyChange(pwalletMain);

    CScript scriptCollateral;
    CPubKey vchPubKey;
    assert(reservekeyCollateral.GetReservedKey(vchPubKey, false)); // should never fail, as we just unlocked
    scriptCollateral = GetScriptForDestination(vchPubKey.GetID());

    vecSend.push_back((CRecipient){scriptCollateral, CPrivateSend::GetMaxCollateralAmount(), false});

    // try to use non-denominated and not dn-like funds first, select them explicitly
    CCoinControl coinControl;
    coinControl.fAllowOtherInputs = false;
    coinControl.fAllowWatchOnly = false;
    // send change to the same address so that we were able create more denoms out of it later
    coinControl.destChange = tallyItem.txdest;
    BOOST_FOREACH(const CTxIn& txin, tallyItem.vecTxIn)
        coinControl.Select(txin.prevout);

    bool fSuccess = pwalletMain->CreateTransaction(vecSend, wtx, reservekeyChange,
            nFeeRet, nChangePosRet, strFail, &coinControl, true, ONLY_NONDENOMINATED_NOT1000IFDN);
    if(!fSuccess) {
        LogPrintf("CPrivateSendClient::MakeCollateralAmounts -- ONLY_NONDENOMINATED_NOT1000IFDN Error: %s\n", strFail);
        // If we failed then most likeky there are not enough funds on this address.
        if(fTryDenominated) {
            // Try to also use denominated coins (we can't mix denominated without collaterals anyway).
            // DN-like funds should not be touched in any case.
            if(!pwalletMain->CreateTransaction(vecSend, wtx, reservekeyChange,
                                nFeeRet, nChangePosRet, strFail, &coinControl, true, ONLY_NOT1000IFDN)) {
                LogPrintf("CPrivateSendClient::MakeCollateralAmounts -- ONLY_NOT1000IFDN Error: %s\n", strFail);
                reservekeyCollateral.ReturnKey();
                return false;
            }
        } else {
            // Nothing else we can do.
            reservekeyCollateral.ReturnKey();
            return false;
        }
    }

    reservekeyCollateral.KeepKey();

    LogPrintf("CPrivateSendClient::MakeCollateralAmounts -- txid=%s\n", wtx.GetHash().GetHex());

    // use the same nCachedLastSuccessBlock as for DS mixinx to prevent race
    if(!pwalletMain->CommitTransaction(wtx, reservekeyChange, &connman)) {
        LogPrintf("CPrivateSendClient::MakeCollateralAmounts -- CommitTransaction failed!\n");
        return false;
    }

    nCachedLastSuccessBlock = nCachedBlockHeight;

    return true;
}

// Create denominations by looping through inputs grouped by addresses
bool CPrivateSendClient::CreateDenominated(CConnman& connman)
{
    LOCK2(cs_main, pwalletMain->cs_wallet);

    std::vector<CompactTallyItem> vecTally;
    if(!pwalletMain->SelectCoinsGrouppedByAddresses(vecTally)) {
        LogPrint("privatesend", "CPrivateSendClient::CreateDenominated -- SelectCoinsGrouppedByAddresses can't find any inputs!\n");
        return false;
    }

    bool fCreateMixingCollaterals = !pwalletMain->HasCollateralInputs();

    BOOST_FOREACH(CompactTallyItem& item, vecTally) {
        if(!CreateDenominated(item, fCreateMixingCollaterals, connman)) continue;
        return true;
    }

    LogPrintf("CPrivateSendClient::CreateDenominated -- failed!\n");
    return false;
}

// Create denominations
bool CPrivateSendClient::CreateDenominated(const CompactTallyItem& tallyItem, bool fCreateMixingCollaterals, CConnman& connman)
{
    std::vector<CRecipient> vecSend;
    CKeyHolderStorage keyHolderStorageDenom;

    CAmount nValueLeft = tallyItem.nAmount;
    nValueLeft -= CPrivateSend::GetCollateralAmount(); // leave some room for fees

    LogPrintf("CreateDenominated0 nValueLeft: %f\n", (float)nValueLeft/COIN);

    // ****** Add an output for mixing collaterals ************ /

    if(fCreateMixingCollaterals) {
        CScript scriptCollateral = keyHolderStorageDenom.AddKey(pwalletMain).GetScriptForDestination();
        vecSend.push_back((CRecipient){ scriptCollateral, CPrivateSend::GetMaxCollateralAmount(), false });
        nValueLeft -= CPrivateSend::GetMaxCollateralAmount();
    }

    // ****** Add outputs for denoms ************ /

    // try few times - skipping smallest denoms first if there are too many of them already, if failed - use them too

    int nOutputsTotal = 0;
    bool fSkip = true;
    do {
        std::vector<CAmount> vecStandardDenoms = CPrivateSend::GetStandardDenominations();

        BOOST_REVERSE_FOREACH(CAmount nDenomValue, vecStandardDenoms) {

            if(fSkip) {
                // Note: denoms are skipped if there are already DENOMS_COUNT_MAX of them
                // and there are still larger denoms which can be used for mixing

                // check skipped denoms
                if(IsDenomSkipped(nDenomValue)) continue;

                // find new denoms to skip if any (ignore the largest one)
                if(nDenomValue != vecStandardDenoms.front() && pwalletMain->CountInputsWithAmount(nDenomValue) > DENOMS_COUNT_MAX) {
                    strAutoDenomResult = strprintf(_("Too many %f denominations, removing."), (float)nDenomValue/COIN);
                    LogPrintf("CPrivateSendClient::CreateDenominated -- %s\n", strAutoDenomResult);
                    vecDenominationsSkipped.push_back(nDenomValue);
                    continue;
                }
            }

            int nOutputs = 0;

            // add each output up to 11 times until it can't be added again
            while(nValueLeft - nDenomValue >= 0 && nOutputs <= 10) {
                CScript scriptDenom = keyHolderStorageDenom.AddKey(pwalletMain).GetScriptForDestination();

                vecSend.push_back((CRecipient){ scriptDenom, nDenomValue, false });

                //increment outputs and subtract denomination amount
                nOutputs++;
                nValueLeft -= nDenomValue;
                LogPrintf("CreateDenominated1: totalOutputs: %d, nOutputsTotal: %d, nOutputs: %d, nValueLeft: %f\n", nOutputsTotal + nOutputs, nOutputsTotal, nOutputs, (float)nValueLeft/COIN);
            }

            nOutputsTotal += nOutputs;
            if(nValueLeft == 0) break;
        }
        LogPrintf("CreateDenominated2: nOutputsTotal: %d, nValueLeft: %f\n", nOutputsTotal, (float)nValueLeft/COIN);
        // if there were no outputs added, start over without skipping
        fSkip = !fSkip;
    } while (nOutputsTotal == 0 && !fSkip);
    LogPrintf("CreateDenominated3: nOutputsTotal: %d, nValueLeft: %f\n", nOutputsTotal, (float)nValueLeft/COIN);

    // if we have anything left over, it will be automatically send back as change - there is no need to send it manually

    CCoinControl coinControl;
    coinControl.fAllowOtherInputs = false;
    coinControl.fAllowWatchOnly = false;
    // send change to the same address so that we were able create more denoms out of it later
    coinControl.destChange = tallyItem.txdest;
    BOOST_FOREACH(const CTxIn& txin, tallyItem.vecTxIn)
        coinControl.Select(txin.prevout);

    CWalletTx wtx;
    CAmount nFeeRet = 0;
    int nChangePosRet = -1;
    std::string strFail = "";
    // make our change address
    CReserveKey reservekeyChange(pwalletMain);

    bool fSuccess = pwalletMain->CreateTransaction(vecSend, wtx, reservekeyChange,
            nFeeRet, nChangePosRet, strFail, &coinControl, true, ONLY_NONDENOMINATED_NOT1000IFDN);
    if(!fSuccess) {
        LogPrintf("CPrivateSendClient::CreateDenominated -- Error: %s\n", strFail);
        keyHolderStorageDenom.ReturnAll();
        return false;
    }

    keyHolderStorageDenom.KeepAll();

    if(!pwalletMain->CommitTransaction(wtx, reservekeyChange, &connman)) {
        LogPrintf("CPrivateSendClient::CreateDenominated -- CommitTransaction failed!\n");
        return false;
    }

    // use the same nCachedLastSuccessBlock as for DS mixing to prevent race
    nCachedLastSuccessBlock = nCachedBlockHeight;
    LogPrintf("CPrivateSendClient::CreateDenominated -- txid=%s\n", wtx.GetHash().GetHex());

    return true;
}

void CPrivateSendClient::RelayIn(const CPrivateSendEntry& entry, CConnman& connman)
{
    if(!infoMixingDynode.fInfoValid) return;

    g_connman->ForNode(infoMixingDynode.addr, [&entry](CNode* pnode) {
        LogPrintf("CPrivateSendClient::RelayIn -- found master, relaying message to %s\n", pnode->addr.ToString());
        g_connman->PushMessage(pnode, NetMsgType::PSVIN, entry);
        return true;
    });
}

void CPrivateSendClient::SetState(PoolState nStateNew)
{
    LogPrintf("CPrivateSendClient::SetState -- nState: %d, nStateNew: %d\n", nState, nStateNew);
    nState = nStateNew;
}

void CPrivateSendClient::UpdatedBlockTip(const CBlockIndex *pindex)
{
    nCachedBlockHeight = pindex->nHeight;
    LogPrint("privatesend", "CPrivateSendClient::UpdatedBlockTip -- nCachedBlockHeight: %d\n", nCachedBlockHeight);

    if(!fLiteMode && dynodeSync.IsDynodeListSynced()) {
        NewBlock();
    }

    CPrivateSend::CheckPSTXes(pindex->nHeight);
}

//TODO: Rename/move to core
void ThreadCheckPrivateSendClient(CConnman& connman)
{
    if(fLiteMode) return; // disable all Dynamic specific functionality

    static bool fOneThread;
    if(fOneThread) return;
    fOneThread = true;

    // Make this thread recognisable as the PrivateSend thread
    RenameThread("dynamic-ps-client");

    unsigned int nTick = 0;
    unsigned int nDoAutoNextRun = nTick + PRIVATESEND_AUTO_TIMEOUT_MIN;

    while (true)
    {
        MilliSleep(1000);

        if(dynodeSync.IsBlockchainSynced() && !ShutdownRequested()) {
            nTick++;
            privateSendClient.CheckTimeout();
            if(nDoAutoNextRun == nTick) {
                privateSendClient.DoAutomaticDenominating(connman);
                nDoAutoNextRun = nTick + PRIVATESEND_AUTO_TIMEOUT_MIN + GetRandInt(PRIVATESEND_AUTO_TIMEOUT_MAX - PRIVATESEND_AUTO_TIMEOUT_MIN);
            }
        }
    }
}