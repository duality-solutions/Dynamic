// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "privatesend-client.h"

#include "wallet/coincontrol.h"
#include "consensus/validation.h"
#include "core_io.h"
#include "init.h"
#include "dynode-payments.h"
#include "dynode-sync.h"
#include "dynodeman.h"
#include "netmessagemaker.h"
#include "script/sign.h"
#include "txmempool.h"
#include "util.h"
#include "utilmoneystr.h"

#include <memory>

CPrivateSendClient privateSendClient;

void CPrivateSendClient::ProcessMessage(CNode* pfrom, const std::string& strCommand, CDataStream& vRecv, CConnman& connman)
{
    if(fDynodeMode) return;
    if(fLiteMode) return; // ignore all Dynamic related functionality
    if(!dynodeSync.IsBlockchainSynced()) return;

    if(strCommand == NetMsgType::PSQUEUE) {
        TRY_LOCK(cs_privatesend, lockRecv);
        if(!lockRecv) return;

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrint("privatesend", "PSQUEUE -- peer=%d using obsolete version %i\n", pfrom->id, pfrom->nVersion);
            connman.PushMessage(pfrom, CNetMsgMaker(pfrom->GetSendVersion()).Make(NetMsgType::REJECT, strCommand, REJECT_OBSOLETE,
                               strprintf("Version must be %d or greater", MIN_PRIVATESEND_PEER_PROTO_VERSION)));
            return;
        }

        CPrivateSendQueue psq;
        vRecv >> psq;

        // process every psq only once
        for (const auto& q : vecPrivateSendQueue) {
            if(q == psq) {
                // LogPrint("privatesend", "PSQUEUE -- %s seen\n", psq.ToString());
                return;
            }
        }

        LogPrint("privatesend", "PSQUEUE -- %s new\n", psq.ToString());

        if(psq.IsExpired()) return;
        if(psq.nInputCount < 0 || psq.nInputCount > PRIVATESEND_ENTRY_MAX_SIZE) return;

        dynode_info_t infoDn;
        if(!dnodeman.GetDynodeInfo(psq.dynodeOutpoint, infoDn)) return;

        if(!psq.CheckSignature(infoDn.pubKeyDynode)) {
            // we probably have outdated info
            dnodeman.AskForDN(pfrom, psq.dynodeOutpoint, connman);
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
                LogPrint("privatesend", "PSQUEUE -- PrivateSend queue (%s) is ready on dynode %s\n", psq.ToString(), infoDn.addr.ToString());
                SubmitDenominate(connman);
            }
        } else {
            for (const auto& q : vecPrivateSendQueue) {
                if(q.dynodeOutpoint == psq.dynodeOutpoint) {
                    // no way same dn can send another "not yet ready" psq this soon
                    LogPrint("privatesend", "PSQUEUE -- Dynode %s is sending WAY too many psq messages\n", infoDn.addr.ToString());
                    return;
                }
            }

            int nThreshold = infoDn.nLastPsq + dnodeman.CountEnabled(MIN_PRIVATESEND_PEER_PROTO_VERSION)/5;
            LogPrint("privatesend", "PSQUEUE -- nLastPsq: %d  threshold: %d  nPsqCount: %d\n", infoDn.nLastPsq, nThreshold, dnodeman.nPsqCount);
            //don't allow a few nodes to dominate the queuing process
            if(infoDn.nLastPsq != 0 && nThreshold > dnodeman.nPsqCount) {
                LogPrint("privatesend", "PSQUEUE -- Dynode %s is sending too many psq messages\n", infoDn.addr.ToString());
                return;
            }

            if(!dnodeman.AllowMixing(psq.dynodeOutpoint)) return;

            LogPrint("privatesend", "PSQUEUE -- new PrivateSend queue (%s) from dynode %s\n", psq.ToString(), infoDn.addr.ToString());
            if(infoMixingDynode.fInfoValid && infoMixingDynode.outpoint == psq.dynodeOutpoint) {
                psq.fTried = true;
            }
            vecPrivateSendQueue.push_back(psq);
            psq.Relay(connman);
        }

    } else if(strCommand == NetMsgType::PSSTATUSUPDATE) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrint("privatesend", "PSSTATUSUPDATE -- peer=%d using obsolete version %i\n", pfrom->id, pfrom->nVersion);
            connman.PushMessage(pfrom, CNetMsgMaker(pfrom->GetSendVersion()).Make(NetMsgType::REJECT, strCommand, REJECT_OBSOLETE,
                               strprintf("Version must be %d or greater", MIN_PRIVATESEND_PEER_PROTO_VERSION)));
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

        LogPrint("privatesend", "PSSTATUSUPDATE -- nMsgSessionID %d  nMsgState: %d  nEntriesCount: %d  nMsgStatusUpdate: %d  nMsgMessageID %d (%s)\n",
                nMsgSessionID, nMsgState, nEntriesCount, nMsgStatusUpdate, nMsgMessageID, CPrivateSend::GetMessageByID(PoolMessage(nMsgMessageID)));

        if(!CheckPoolStateUpdate(PoolState(nMsgState), nMsgEntriesCount, PoolStatusUpdate(nMsgStatusUpdate), PoolMessage(nMsgMessageID), nMsgSessionID)) {
            LogPrint("privatesend", "PSSTATUSUPDATE -- CheckPoolStateUpdate failed\n");
        }

    } else if(strCommand == NetMsgType::PSFINALTX) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrint("privatesend", "PSFINALTX -- peer=%d using obsolete version %i\n", pfrom->id, pfrom->nVersion);
            connman.PushMessage(pfrom, CNetMsgMaker(pfrom->GetSendVersion()).Make(NetMsgType::REJECT, strCommand, REJECT_OBSOLETE,
                               strprintf("Version must be %d or greater", MIN_PRIVATESEND_PEER_PROTO_VERSION)));
            return;
        }

        if(!infoMixingDynode.fInfoValid) return;
        if(infoMixingDynode.addr != pfrom->addr) {
            //LogPrintf("PSFINALTX -- message doesn't match current Dynode: infoMixingDynode %s addr %s\n", infoMixingDynode.addr.ToString(), pfrom->addr.ToString());
            return;
        }

        int nMsgSessionID;
        vRecv >> nMsgSessionID;
        CTransaction txNew(deserialize, vRecv);

        if(nSessionID != nMsgSessionID) {
            LogPrint("privatesend", "PSFINALTX -- message doesn't match current PrivateSend session: nSessionID: %d  nMsgSessionID: %d\n", nSessionID, nMsgSessionID);
            return;
        }

        LogPrint("privatesend", "PSFINALTX -- txNew %s", txNew.ToString());

        //check to see if input is spent already? (and probably not confirmed)
        SignFinalTransaction(txNew, pfrom, connman);

    } else if(strCommand == NetMsgType::PSCOMPLETE) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrint("privatesend", "PSCOMPLETE -- peer=%d using obsolete version %i\n", pfrom->id, pfrom->nVersion);
            connman.PushMessage(pfrom, CNetMsgMaker(pfrom->GetSendVersion()).Make(NetMsgType::REJECT, strCommand, REJECT_OBSOLETE,
                               strprintf("Version must be %d or greater", MIN_PRIVATESEND_PEER_PROTO_VERSION)));
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
    pendingPsaRequest = CPendingPsaRequest();

    CPrivateSendBase::SetNull();
}

//
// Unlock coins after mixing fails or succeeds
//
void CPrivateSendClient::UnlockCoins()
{
    if (!pwalletMain) return; 

    while(true) {
        TRY_LOCK(pwalletMain->cs_wallet, lockWallet);
        if(!lockWallet) {MilliSleep(50); continue;}
        for (const auto& outpoint : vecOutPointLocked)
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

bool CPrivateSendClient::GetMixingDynodeInfo(dynode_info_t& dnInfoRet)
{
    dnInfoRet = infoMixingDynode.fInfoValid ? infoMixingDynode : dynode_info_t();
    return infoMixingDynode.fInfoValid;
}

bool CPrivateSendClient::IsMixingDynode(const CNode* pnode)
{
    return infoMixingDynode.fInfoValid && pnode->addr == infoMixingDynode.addr;
}

//
// Check the mixing progress and send client updates if a Dynode
//
void CPrivateSendClient::CheckPool()
{
    // reset if we're here for 10 seconds
    if((nState == POOL_STATE_ERROR || nState == POOL_STATE_SUCCESS) && GetTime() - nTimeLastSuccessfulStep >= 10) {
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
    if(fDynodeMode) return;

    CheckQueue();

    if(!fEnablePrivateSend) return;

    // catching hanging sessions
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

    int nLagTime = 10; // give the server a few extra seconds before resetting.
    int nTimeout = (nState == POOL_STATE_SIGNING) ? PRIVATESEND_SIGNING_TIMEOUT : PRIVATESEND_QUEUE_TIMEOUT;
    bool fTimeout = GetTime() - nTimeLastSuccessfulStep >= nTimeout + nLagTime;

    if(nState != POOL_STATE_IDLE && fTimeout) {
        LogPrint("privatesend", "CPrivateSendClient::CheckTimeout -- %s timed out (%ds) -- resetting\n",
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
bool CPrivateSendClient::SendDenominate(const std::vector<CTxPSIn>& vecTxPSIn, const std::vector<CTxOut>& vecTxOut, CConnman& connman)
{
    if(fDynodeMode) {
        LogPrintf("CPrivateSendClient::SendDenominate -- PrivateSend from a Dynode is not supported currently.\n");
        return false;
    }

    if(txMyCollateral == CMutableTransaction()) {
        LogPrintf("CPrivateSendClient:SendDenominate -- PrivateSend collateral not set\n");
        return false;
    }

    // lock the funds we're going to use
    for (const auto& txin : txMyCollateral.vin)
        vecOutPointLocked.push_back(txin.prevout);

    for (const auto& txdsin : vecTxPSIn)
        vecOutPointLocked.push_back(txdsin.prevout);

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

    {
        // construct a pseudo tx, for debugging purpuses only

        CMutableTransaction tx;

        for (const auto& txdsin : vecTxPSIn) {
            LogPrint("privatesend", "CPrivateSendClient::SendDenominate -- txdsin=%s\n", txdsin.ToString());
            tx.vin.push_back(txdsin);
        }

        for (const CTxOut& txout : vecTxOut) {
            LogPrint("privatesend", "CPrivateSendClient::SendDenominate -- txout=%s\n", txout.ToString());
            tx.vout.push_back(txout);
        }

        LogPrintf("CPrivateSendClient::SendDenominate -- Submitting partial tx %s", tx.ToString());
    }

    // store our entry for later use
    CPrivateSendEntry entry(vecTxPSIn, vecTxOut, txMyCollateral);
    vecEntries.push_back(entry);
    RelayIn(entry, connman);
    nTimeLastSuccessfulStep = GetTime();

    return true;
}

// Incoming message from Dynode updating the progress of mixing
bool CPrivateSendClient::CheckPoolStateUpdate(PoolState nStateNew, int nEntriesCountNew, PoolStatusUpdate nStatusUpdate, PoolMessage nMessageID, int nSessionIDNew)
{
    if(fDynodeMode) return false;

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
            nTimeLastSuccessfulStep = GetTime();
            LogPrintf("CPrivateSendClient::CheckPoolStateUpdate -- set nSessionID to %d\n", nSessionID);
            return true;
        }
        else if(nStateNew == POOL_STATE_ACCEPTING_ENTRIES && nEntriesCount != nEntriesCountNew) {
            nEntriesCount = nEntriesCountNew;
            nTimeLastSuccessfulStep = GetTime();
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
     if (!pwalletMain) return false; 

    if(fDynodeMode || pnode == nullptr) return false;

    finalMutableTransaction = finalTransactionNew;
    LogPrintf("CPrivateSendClient::SignFinalTransaction -- finalMutableTransaction=%s", finalMutableTransaction.ToString());

    // Make sure it's BIP69 compliant
    sort(finalMutableTransaction.vin.begin(), finalMutableTransaction.vin.end(), CompareInputBIP69());
    sort(finalMutableTransaction.vout.begin(), finalMutableTransaction.vout.end(), CompareOutputBIP69());

    if(finalMutableTransaction.GetHash() != finalTransactionNew.GetHash()) {
        LogPrintf("CPrivateSendClient::SignFinalTransaction -- WARNING! Dynode %s is not BIP69 compliant!\n", infoMixingDynode.outpoint.ToStringShort());
        UnlockCoins();
        keyHolderStorage.ReturnAll();
        SetNull();
        return false;
    }

    std::vector<CTxIn> sigs;

    //make sure my inputs/outputs are present, otherwise refuse to sign
    for (const auto& entry : vecEntries) {
        for (const auto& txdsin : entry.vecTxPSIn) {
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

                for (const auto& txoutFinal : finalMutableTransaction.vout) {
                    for (const auto& txout: entry.vecTxOut) {
                        if(txoutFinal == txout) {
                            nFoundOutputsCount++;
                            nValue1 += txoutFinal.nValue;
                        }
                    }
                }

                for (const auto& txout : entry.vecTxOut)
                    nValue2 += txout.nValue;

                int nTargetOuputsCount = entry.vecTxOut.size();
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
    CNetMsgMaker msgMaker(pnode->GetSendVersion());
    connman.PushMessage(pnode, msgMaker.Make(NetMsgType::PSSIGNFINALTX, sigs));
    SetState(POOL_STATE_SIGNING);
    nTimeLastSuccessfulStep = GetTime();

    return true;
}

// mixing transaction was completed (failed or successful)
void CPrivateSendClient::CompletedTransaction(PoolMessage nMessageID)
{
    if(fDynodeMode) return;

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

bool CPrivateSendClient::IsDenomSkipped(CAmount nDenomValue)
{
    return std::find(vecDenominationsSkipped.begin(), vecDenominationsSkipped.end(), nDenomValue) != vecDenominationsSkipped.end();
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

    if (!pwalletMain) {
        LogPrint("privatesend", "CPrivateSendClient::CheckAutomaticBackup -- Wallet is not initialized, no mixing available.\n");
        strAutoDenomResult = _("Wallet is not initialized") + ", " + _("no mixing available.");
        fEnablePrivateSend = false; // no mixing
        return false;
    }

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
            // keep spamming if debug is on.
            LogPrint("privatesend", "CPrivateSendClient::CheckAutomaticBackup -- ERROR! Failed to create automatic backup.\n");
            strAutoDenomResult = _("ERROR! Failed to create automatic backup") + ", " + _("see debug.log for details.");
            return false;
        case -2:
            // We were able to create automatic backup but keypool was not replenished because wallet is locked.
            // There is no way to bring user attention in daemon mode so we just update status and
            // keep spamming if debug is on.
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
    if(fDynodeMode) return false; // no client-side mixing on dynodes
    if(!fEnablePrivateSend) return false;
    if(nState != POOL_STATE_IDLE) return false;

    if(!dynodeSync.IsDynodeListSynced()) {
        strAutoDenomResult = _("Can't mix while sync in progress.");
        return false;
    }

    if (!pwalletMain) {
        strAutoDenomResult = _("Wallet is not initialized");
        return false;
    }
    if(!fDryRun && pwalletMain->IsLocked(true)) {
        strAutoDenomResult = _("Wallet is locked.");
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
    if (!pwalletMain)  return false;

    std::vector<CAmount> vecStandardDenoms = CPrivateSend::GetStandardDenominations();
    // Look through the queues and see if anything matches
    for (auto& psq : vecPrivateSendQueue) {
        // only try each queue once
        if(psq.fTried) continue;
        psq.fTried = true;

        if(psq.IsExpired()) continue;

        dynode_info_t infoDn;

        if(!dnodeman.GetDynodeInfo(psq.dynodeOutpoint, infoDn)) {
            LogPrintf("CPrivateSendClient::JoinExistingQueue -- psq dynode is not in dynode list, dynode=%s\n", psq.dynodeOutpoint.ToStringShort());
            continue;
        }

        if(infoDn.nProtocolVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) continue;

        // skip next dn payments winners
        if (dnpayments.IsScheduled(infoDn, 0)) {
            LogPrintf("CPrivateSendClient::JoinExistingQueue -- skipping winner, dynode=%s\n", infoDn.outpoint.ToStringShort());
            continue;
        }

        std::vector<int> vecBits;
        if(!CPrivateSend::GetDenominationsBits(psq.nDenom, vecBits)) {
            // incompatible denom
            continue;
        }

        // mixing rate limit i.e. nLastPsq check should already pass in PSQUEUE ProcessMessage
        // in order for psq to get into vecPrivateSendQueue, so we should be safe to mix already,
        // no need for additional verification here

        LogPrint("privatesend", "CPrivateSendClient::JoinExistingQueue -- found valid queue: %s\n", psq.ToString());

        CAmount nValueInTmp = 0;
        std::vector<CTxPSIn> vecTxPSInTmp;
        std::vector<COutput> vCoinsTmp;
        CAmount nMinAmount = vecStandardDenoms[vecBits.front()];
        CAmount nMaxAmount = nBalanceNeedsAnonymized;
        // nInputCount is not covered by legacy signature, require SPORK_6_NEW_SIGS to activate to use new algo
        // (to make sure nInputCount wasn't modified by some intermediary node)
        bool fNewAlgo = infoDn.nProtocolVersion > 70900 && sporkManager.IsSporkActive(SPORK_6_NEW_SIGS);

        if (fNewAlgo && psq.nInputCount != 0) {
            nMinAmount = nMaxAmount = psq.nInputCount * vecStandardDenoms[vecBits.front()];
        }
        // Try to match their denominations if possible, select exact number of denominations
        if(!pwalletMain->SelectCoinsByDenominations(psq.nDenom, nMinAmount, nMaxAmount, vecTxPSInTmp, vCoinsTmp, nValueInTmp, 0, nPrivateSendRounds)) {
            LogPrintf("CPrivateSendClient::JoinExistingQueue -- Couldn't match %d denominations %d %d (%s)\n", psq.nInputCount, vecBits.front(), psq.nDenom, CPrivateSend::GetDenominationsToString(psq.nDenom));
            continue;
        }

        vecDynodesUsed.push_back(psq.dynodeOutpoint);

        if (connman.IsDynodeOrDisconnectRequested(infoDn.addr)) {
            LogPrintf("CPrivateSendClient::JoinExistingQueue -- skipping dynode connection, addr=%s\n", infoDn.addr.ToString());
            continue;
        }

        nSessionDenom = psq.nDenom;
        nSessionInputCount = fNewAlgo ? psq.nInputCount : 0;
        infoMixingDynode = infoDn;
        pendingPsaRequest = CPendingPsaRequest(infoDn.addr, CPrivateSendAccept(nSessionDenom, nSessionInputCount, txMyCollateral));
        connman.AddPendingDynode(infoDn.addr);
        // TODO: add new state POOL_STATE_CONNECTING and bump MIN_PRIVATESEND_PEER_PROTO_VERSION
        SetState(POOL_STATE_QUEUE);
        nTimeLastSuccessfulStep = GetTime();
        LogPrintf("CPrivateSendClient::JoinExistingQueue -- pending connection (from queue): nSessionDenom: %d (%s), nSessionInputCount: %d, addr=%s\n",
                  nSessionDenom, CPrivateSend::GetDenominationsToString(nSessionDenom), nSessionInputCount, infoDn.addr.ToString());
        strAutoDenomResult = _("Trying to connect...");
        return true;
    }
    strAutoDenomResult = _("Failed to find mixing queue to join");
    return false;
}

bool CPrivateSendClient::StartNewQueue(CAmount nValueMin, CAmount nBalanceNeedsAnonymized, CConnman& connman)
{
    if (!pwalletMain) return false;

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

        // skip next dn payments winners
        if (dnpayments.IsScheduled(infoDn, 0)) {
            LogPrintf("CPrivateSendClient::StartNewQueue -- skipping winner, dynode=%s\n", infoDn.outpoint.ToStringShort());
            nTries++;
            continue;
        }

        vecDynodesUsed.push_back(infoDn.outpoint);

        if(infoDn.nLastPsq != 0 && infoDn.nLastPsq + nDnCountEnabled/5 > dnodeman.nPsqCount) {
            LogPrintf("CPrivateSendClient::StartNewQueue -- Too early to mix on this dynode!"
                        " dynode=%s  addr=%s  nLastPsq=%d  CountEnabled/5=%d  nPsqCount=%d\n",
                        infoDn.outpoint.ToStringShort(), infoDn.addr.ToString(), infoDn.nLastPsq,
                        nDnCountEnabled/5, dnodeman.nPsqCount);
            nTries++;
            continue;
        }

        if (connman.IsDynodeOrDisconnectRequested(infoDn.addr)) {
            LogPrintf("CPrivateSendClient::StartNewQueue -- skipping dynode connection, addr=%s\n", infoDn.addr.ToString());
            nTries++;
            continue;
        }

        LogPrintf("CPrivateSendClient::StartNewQueue -- attempt %d connection to Dynode %s\n", nTries, infoDn.addr.ToString());

        std::vector<CAmount> vecAmounts;
        pwalletMain->ConvertList(vecTxIn, vecAmounts);
        // try to get a single random denom out of vecAmounts
        while(nSessionDenom == 0) {
            nSessionDenom = CPrivateSend::GetDenominationsByAmounts(vecAmounts);
        }

        // Count available denominations.
        // Should never really fail after this point, since we just selected compatible inputs ourselves.
        std::vector<int> vecBits;
        if (!CPrivateSend::GetDenominationsBits(nSessionDenom, vecBits)) {
            return false;
        }

        CAmount nValueInTmp = 0;
        std::vector<CTxPSIn> vecTxPSInTmp;
        std::vector<COutput> vCoinsTmp;
        std::vector<CAmount> vecStandardDenoms = CPrivateSend::GetStandardDenominations();

        bool fSelected = pwalletMain->SelectCoinsByDenominations(nSessionDenom, vecStandardDenoms[vecBits.front()], vecStandardDenoms[vecBits.front()] * PRIVATESEND_ENTRY_MAX_SIZE, vecTxPSInTmp, vCoinsTmp, nValueInTmp, 0, nPrivateSendRounds);
        if (!fSelected) {
            return false;
        }

        // nInputCount is not covered by legacy signature, require SPORK_6_NEW_SIGS to activate to use new algo
        // (to make sure nInputCount wasn't modified by some intermediary node)
        bool fNewAlgo = infoDn.nProtocolVersion > 70900 && sporkManager.IsSporkActive(SPORK_6_NEW_SIGS);
        nSessionInputCount = fNewAlgo
                ? std::min(vecTxPSInTmp.size(), size_t(5 + GetRand(PRIVATESEND_ENTRY_MAX_SIZE - 5 + 1)))
                : 0;
        infoMixingDynode = infoDn;
        connman.AddPendingDynode(infoDn.addr);
        pendingPsaRequest = CPendingPsaRequest(infoDn.addr, CPrivateSendAccept(nSessionDenom, nSessionInputCount, txMyCollateral));
        // TODO: add new state POOL_STATE_CONNECTING and bump MIN_PRIVATESEND_PEER_PROTO_VERSION
        SetState(POOL_STATE_QUEUE);
        nTimeLastSuccessfulStep = GetTime();
        LogPrintf("CPrivateSendClient::StartNewQueue -- pending connection, nSessionDenom: %d (%s), nSessionInputCount: %d, addr=%s\n",
                nSessionDenom, CPrivateSend::GetDenominationsToString(nSessionDenom), nSessionInputCount, infoDn.addr.ToString());
        strAutoDenomResult = _("Trying to connect...");
        return true;
    }
    strAutoDenomResult = _("Failed to start a new mixing queue");
    return false;
}

void CPrivateSendClient::ProcessPendingPsaRequest(CConnman& connman)
{
    if (!pendingPsaRequest) return;

    bool fDone = connman.ForNode(pendingPsaRequest.GetAddr(), [&](CNode* pnode) {
        LogPrint("privatesend", "-- processing psa queue for addr=%s\n", pnode->addr.ToString());
        nTimeLastSuccessfulStep = GetTime();
        // TODO: this vvvv should be here after new state POOL_STATE_CONNECTING is added and MIN_PRIVATESEND_PEER_PROTO_VERSION is bumped
        // SetState(POOL_STATE_QUEUE);
        strAutoDenomResult = _("Mixing in progress...");
        CNetMsgMaker msgMaker(pnode->GetSendVersion());
        connman.PushMessage(pnode, msgMaker.Make(NetMsgType::PSACCEPT, pendingPsaRequest.GetPSA()));
        return true;
    });

    if (fDone) {
        pendingPsaRequest = CPendingPsaRequest();
    } else if (pendingPsaRequest.IsExpired()) {
        LogPrint("privatesend", "CPrivateSendClient::%s -- failed to connect to %s\n", __func__, pendingPsaRequest.GetAddr().ToString());
        SetNull();
    }
}

bool CPrivateSendClient::SubmitDenominate(CConnman& connman)
{
    std::string strError;
    std::vector<CTxPSIn> vecTxPSInRet;
    std::vector<CTxOut> vecTxOutRet;

    // Submit transaction to the pool if we get here
    if (nLiquidityProvider) {
        // Try to use only inputs with the same number of rounds starting from the lowest number of rounds possible
        for(int i = 0; i< nPrivateSendRounds; i++) {
            if(PrepareDenominate(i, i + 1, strError, vecTxPSInRet, vecTxOutRet)) {
                LogPrintf("CPrivateSendClient::SubmitDenominate -- Running PrivateSend denominate for %d rounds, success\n", i);
                return SendDenominate(vecTxPSInRet, vecTxOutRet, connman);
            }
            LogPrint("privatesend", "CPrivateSendClient::SubmitDenominate -- Running PrivateSend denominate for %d rounds, error: %s\n", i, strError);
        }
    } else {
        // Try to use only inputs with the same number of rounds starting from the highest number of rounds possible
        for(int i = nPrivateSendRounds; i > 0; i--) {
            if(PrepareDenominate(i - 1, i, strError, vecTxPSInRet, vecTxOutRet)) {
                LogPrintf("CPrivateSendClient::SubmitDenominate -- Running PrivateSend denominate for %d rounds, success\n", i);
                return SendDenominate(vecTxPSInRet, vecTxOutRet, connman);
            }
            LogPrint("privatesend", "CPrivateSendClient::SubmitDenominate -- Running PrivateSend denominate for %d rounds, error: %s\n", i, strError);
        }
    }

    // We failed? That's strange but let's just make final attempt and try to mix everything
    if(PrepareDenominate(0, nPrivateSendRounds, strError, vecTxPSInRet, vecTxOutRet)) {
        LogPrintf("CPrivateSendClient::SubmitDenominate -- Running PrivateSend denominate for all rounds, success\n");
        return SendDenominate(vecTxPSInRet, vecTxOutRet, connman);
    }

    // Should never actually get here but just in case
    LogPrintf("CPrivateSendClient::SubmitDenominate -- Running PrivateSend denominate for all rounds, error: %s\n", strError);
    strAutoDenomResult = strError;
    return false;
}

bool CPrivateSendClient::PrepareDenominate(int nMinRounds, int nMaxRounds, std::string& strErrorRet, std::vector<CTxPSIn>& vecTxPSInRet, std::vector<CTxOut>& vecTxOutRet)
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
    vecTxPSInRet.clear();
    vecTxOutRet.clear();

    // ** find the coins we'll use
    std::vector<CTxPSIn> vecTxPSIn;
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
    bool fSelected = pwalletMain->SelectCoinsByDenominations(nSessionDenom, vecStandardDenoms[vecBits.front()], vecStandardDenoms[vecBits.front()] * PRIVATESEND_ENTRY_MAX_SIZE, vecTxPSIn, vCoins, nValueIn, nMinRounds, nMaxRounds);
    if (nMinRounds >= 0 && !fSelected) {
        strErrorRet = "Can't select current denominated inputs";
        return false;
    }

    LogPrintf("CPrivateSendClient::PrepareDenominate -- max value: %f\n", (double)nValueIn/COIN);

    {
        LOCK(pwalletMain->cs_wallet);
        for (const auto& txin : vecTxPSIn) {
            pwalletMain->LockCoin(txin.prevout);
        }
    }

    CAmount nValueLeft = nValueIn;

    // Try to add every needed denomination, repeat up to 5-PRIVATESEND_ENTRY_MAX_SIZE times.
    // NOTE: No need to randomize order of inputs because they were
    // initially shuffled in CWallet::SelectCoinsByDenominations already.
    int nStep = 0;
    int nStepsMax = nSessionInputCount != 0 ? nSessionInputCount : (5 + GetRandInt(PRIVATESEND_ENTRY_MAX_SIZE - 5 + 1));

    while (nStep < nStepsMax) {
        for (const auto& nBit : vecBits) {
            CAmount nValueDenom = vecStandardDenoms[nBit];
            if (nValueLeft - nValueDenom < 0) continue;

            // Note: this relies on a fact that both vectors MUST have same size
            std::vector<CTxPSIn>::iterator it = vecTxPSIn.begin();
            std::vector<COutput>::iterator it2 = vCoins.begin();
            while (it2 != vCoins.end()) {
                // we have matching inputs
                if ((*it2).tx->tx->vout[(*it2).i].nValue == nValueDenom) {
                    // add new input in resulting vector
                    vecTxPSInRet.push_back(*it);
                    // remove corresponding items from initial vectors
                    vecTxPSIn.erase(it);
                    vCoins.erase(it2);

                    CScript scriptDenom = keyHolderStorage.AddKey(pwalletMain);

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
        nStep++;
        if(nValueLeft == 0) break;
    }

    {
        // unlock unused coins
        LOCK(pwalletMain->cs_wallet);
        for (const auto& txin : vecTxPSIn) {
            pwalletMain->UnlockCoin(txin.prevout);
        }
    }

    if (CPrivateSend::GetDenominations(vecTxOutRet) != nSessionDenom || (nSessionInputCount != 0 && vecTxOutRet.size() != nSessionInputCount)) {
        {
            // unlock used coins on failure
            LOCK(pwalletMain->cs_wallet);
            for (const auto& txin : vecTxPSInRet) {
                pwalletMain->UnlockCoin(txin.prevout);
            }
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
    if (!pwalletMain) return false;

    std::vector<CompactTallyItem> vecTally;
    if(!pwalletMain->SelectCoinsGrouppedByAddresses(vecTally, false)) {
        LogPrint("privatesend", "CPrivateSendClient::MakeCollateralAmounts -- SelectCoinsGrouppedByAddresses can't find any inputs!\n");
        return false;
    }

    // First try to use only non-denominated funds
    for (const auto& item : vecTally) {
        if(!MakeCollateralAmounts(item, false, connman)) continue;
        return true;
    }

    // There should be at least some denominated funds we should be able to break in pieces to continue mixing
    for (const auto& item : vecTally) {
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
    if (!pwalletMain) return false;

    LOCK2(cs_main, pwalletMain->cs_wallet);

    // denominated input is always a single one, so we can check its amount directly and return early
    if(!fTryDenominated && tallyItem.vecOutPoints.size() == 1 && CPrivateSend::IsDenominatedAmount(tallyItem.nAmount))
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
    for (const auto& outpoint : tallyItem.vecOutPoints)
        coinControl.Select(outpoint);

    bool fSuccess = pwalletMain->CreateTransaction(vecSend, wtx, reservekeyChange,
            nFeeRet, nChangePosRet, strFail, &coinControl, true, ONLY_NONDENOMINATED);
    if(!fSuccess) {
        LogPrintf("CPrivateSendClient::MakeCollateralAmounts -- ONLY_NONDENOMINATED: %s\n", strFail);
        // If we failed then most likely there are not enough funds on this address.
        if(fTryDenominated) {
            // Try to also use denominated coins (we can't mix denominated without collaterals anyway).
            if(!pwalletMain->CreateTransaction(vecSend, wtx, reservekeyChange,
                                nFeeRet, nChangePosRet, strFail, &coinControl, true, ALL_COINS)) {
                LogPrintf("CPrivateSendClient::MakeCollateralAmounts -- ALL_COINS Error: %s\n", strFail);
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

    // use the same nCachedLastSuccessBlock as for DS mixing to prevent race
    CValidationState state;
    if(!pwalletMain->CommitTransaction(wtx, reservekeyChange, &connman, state)) {
        LogPrintf("CPrivateSendClient::MakeCollateralAmounts -- CommitTransaction failed! Reason given: %s\n", state.GetRejectReason());
        return false;
    }

    nCachedLastSuccessBlock = nCachedBlockHeight;

    return true;
}

// Create denominations by looping through inputs grouped by addresses
bool CPrivateSendClient::CreateDenominated(CConnman& connman)
{
    if (!pwalletMain) return false;

    LOCK2(cs_main, pwalletMain->cs_wallet);

    std::vector<CompactTallyItem> vecTally;
    if(!pwalletMain->SelectCoinsGrouppedByAddresses(vecTally)) {
        LogPrint("privatesend", "CPrivateSendClient::CreateDenominated -- SelectCoinsGrouppedByAddresses can't find any inputs!\n");
        return false;
    }

    bool fCreateMixingCollaterals = !pwalletMain->HasCollateralInputs();

    for (const auto& item : vecTally) {
        if(!CreateDenominated(item, fCreateMixingCollaterals, connman)) continue;
        return true;
    }

    LogPrintf("CPrivateSendClient::CreateDenominated -- failed!\n");
    return false;
}

// Create denominations
bool CPrivateSendClient::CreateDenominated(const CompactTallyItem& tallyItem, bool fCreateMixingCollaterals, CConnman& connman)
{
    if (!pwalletMain) return false;

    std::vector<CRecipient> vecSend;
    CKeyHolderStorage keyHolderStorageDenom;

    CAmount nValueLeft = tallyItem.nAmount;
    nValueLeft -= CPrivateSend::GetCollateralAmount(); // leave some room for fees

    LogPrintf("CreateDenominated0: %s nValueLeft: %f\n", CDynamicAddress(tallyItem.txdest).ToString(), (float)nValueLeft/COIN);

    // ****** Add an output for mixing collaterals ************ /

    if(fCreateMixingCollaterals) {
        CScript scriptCollateral = keyHolderStorageDenom.AddKey(pwalletMain);
        vecSend.push_back((CRecipient){ scriptCollateral, CPrivateSend::GetMaxCollateralAmount(), false });
        nValueLeft -= CPrivateSend::GetMaxCollateralAmount();
    }

    // ****** Add outputs for denoms ************ /

    // try few times - skipping smallest denoms first if there are too many of them already, if failed - use them too
    int nOutputsTotal = 0;
    bool fSkip = true;
    do {
        std::vector<CAmount> vecStandardDenoms = CPrivateSend::GetStandardDenominations();

        for (auto it = vecStandardDenoms.rbegin(); it != vecStandardDenoms.rend(); ++it) {
            CAmount nDenomValue = *it;

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
                CScript scriptDenom = keyHolderStorageDenom.AddKey(pwalletMain);

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
    for (const auto& outpoint : tallyItem.vecOutPoints)
        coinControl.Select(outpoint);

    CWalletTx wtx;
    CAmount nFeeRet = 0;
    int nChangePosRet = -1;
    std::string strFail = "";
    // make our change address
    CReserveKey reservekeyChange(pwalletMain);

    bool fSuccess = pwalletMain->CreateTransaction(vecSend, wtx, reservekeyChange,
            nFeeRet, nChangePosRet, strFail, &coinControl, true, ONLY_NONDENOMINATED);
    if(!fSuccess) {
        LogPrintf("CPrivateSendClient::CreateDenominated -- Error: %s\n", strFail);
        keyHolderStorageDenom.ReturnAll();
        return false;
    }

    keyHolderStorageDenom.KeepAll();

    CValidationState state;
    if(!pwalletMain->CommitTransaction(wtx, reservekeyChange, &connman, state)) {
        LogPrintf("CPrivateSendClient::CreateDenominated -- CommitTransaction failed! Reason given: %s\n", state.GetRejectReason());
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

    connman.ForNode(infoMixingDynode.addr, [&entry, &connman](CNode* pnode) {
        LogPrintf("CPrivateSendClient::RelayIn -- found master, relaying message to %s\n", pnode->addr.ToString());
        CNetMsgMaker msgMaker(pnode->GetSendVersion());
        connman.PushMessage(pnode, msgMaker.Make(NetMsgType::PSVIN, entry));
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

}

void CPrivateSendClient::DoMaintenance(CConnman& connman)
{
    if(fLiteMode) return; // disable all Dynamic specific functionality
    if(fDynodeMode) return; // no client-side mixing on dynodes

    if(!dynodeSync.IsBlockchainSynced() || ShutdownRequested())
        return;

    static unsigned int nTick = 0;
    static unsigned int nDoAutoNextRun = nTick + PRIVATESEND_AUTO_TIMEOUT_MIN;

    nTick++;
    privateSendClient.CheckTimeout();
    privateSendClient.ProcessPendingPsaRequest(connman);
    if(nDoAutoNextRun == nTick) {
        privateSendClient.DoAutomaticDenominating(connman);
        nDoAutoNextRun = nTick + PRIVATESEND_AUTO_TIMEOUT_MIN + GetRandInt(PRIVATESEND_AUTO_TIMEOUT_MAX - PRIVATESEND_AUTO_TIMEOUT_MIN);
    }
}
