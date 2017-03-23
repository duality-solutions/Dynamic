// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "privatesend.h"

#include "activedynode.h"
#include "coincontrol.h"
#include "dynode-payments.h"
#include "dynode-sync.h"
#include "dynodeman.h"
#include "governance.h"
#include "init.h"
#include "instantsend.h"
#include "messagesigner.h"
#include "script/sign.h"
#include "txmempool.h"
#include "util.h"
#include "utilmoneystr.h"
#include "consensus/validation.h"

#include <boost/lexical_cast.hpp>

int nPrivateSendRounds = DEFAULT_PRIVATESEND_ROUNDS;
int nPrivateSendAmount = DEFAULT_PRIVATESEND_AMOUNT;
int nLiquidityProvider = DEFAULT_PRIVATESEND_LIQUIDITY;
bool fEnablePrivateSend = false;
bool fPrivateSendMultiSession = DEFAULT_PRIVATESEND_MULTISESSION;

CPrivatesendPool privateSendPool;
std::map<uint256, CPrivatesendBroadcastTx> mapPrivatesendBroadcastTxes;
std::vector<CAmount> vecPrivateSendDenominations;

void CPrivatesendPool::ProcessMessage(CNode* pfrom, std::string& strCommand, CDataStream& vRecv)
{
    if(fLiteMode) return; // ignore all Dynamic related functionality
    if(!dynodeSync.IsBlockchainSynced()) return;

    if(strCommand == NetMsgType::PSACCEPT) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("PSACCEPT -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            PushStatus(pfrom, STATUS_REJECTED, ERR_VERSION);
            return;
        }

        if(!fDyNode) {
            LogPrintf("PSACCEPT -- not a Dynode!\n");
            PushStatus(pfrom, STATUS_REJECTED, ERR_NOT_A_DN);
            return;
        }

        if(IsSessionReady()) {
            // too many users in this session already, reject new ones
            LogPrintf("PSACCEPT -- queue is already full!\n");
            PushStatus(pfrom, STATUS_ACCEPTED, ERR_QUEUE_FULL);
            return;
        }

        int nDenom;
        CTransaction txCollateral;
        vRecv >> nDenom >> txCollateral;

        LogPrint("privatesend", "PSACCEPT -- nDenom %d (%s)  txCollateral %s", nDenom, GetDenominationsToString(nDenom), txCollateral.ToString());

        CDynode* pdn = dnodeman.Find(activeDynode.vin);
        if(pdn == NULL) {
            PushStatus(pfrom, STATUS_REJECTED, ERR_DN_LIST);
            return;
        }

        if(vecSessionCollaterals.size() == 0 && pdn->nLastSsq != 0 &&
            pdn->nLastSsq + dnodeman.CountEnabled(MIN_PRIVATESEND_PEER_PROTO_VERSION)/5 > dnodeman.nSsqCount)
        {
            LogPrintf("PSACCEPT -- last psq too recent, must wait: addr=%s\n", pfrom->addr.ToString());
            PushStatus(pfrom, STATUS_REJECTED, ERR_RECENT);
            return;
        }

        PoolMessage nMessageID = MSG_NOERR;

        bool fResult = nSessionID == 0  ? CreateNewSession(nDenom, txCollateral, nMessageID)
                                        : AddUserToExistingSession(nDenom, txCollateral, nMessageID);
        if(fResult) {
            LogPrintf("PSACCEPT -- is compatible, please submit!\n");
            PushStatus(pfrom, STATUS_ACCEPTED, nMessageID);
            return;
        } else {
            LogPrintf("PSACCEPT -- not compatible with existing transactions!\n");
            PushStatus(pfrom, STATUS_REJECTED, nMessageID);
            return;
        }

    } else if(strCommand == NetMsgType::PSQUEUE) {
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

        if(psq.IsExpired() || psq.nTime > GetTime() + PRIVATESEND_QUEUE_TIMEOUT) return;

        CDynode* pdn = dnodeman.Find(psq.vin);
        if(pdn == NULL) return;

        if(!psq.CheckSignature(pdn->pubKeyDynode)) {
            // we probably have outdated info
            dnodeman.AskForDN(pfrom, psq.vin);
            return;
        }

        // if the queue is ready, submit if we can
        if(psq.fReady) {
            if(!pSubmittedToDynode) return;
            if((CNetAddr)pSubmittedToDynode->addr != (CNetAddr)pdn->addr) {
                LogPrintf("PSQUEUE -- message doesn't match current Dynode: pSubmittedToDynode=%s, addr=%s\n", pSubmittedToDynode->addr.ToString(), pdn->addr.ToString());
                return;
            }

            if(nState == POOL_STATE_QUEUE) {
                LogPrint("privatesend", "PSQUEUE -- PrivateSend queue (%s) is ready on Dynode %s\n", psq.ToString(), pdn->addr.ToString());
                SubmitDenominate();
            }
        } else {
            BOOST_FOREACH(CPrivatesendQueue q, vecPrivatesendQueue) {
                if(q.vin == psq.vin) {
                    // no way same DN can send another "not yet ready" psq this soon
                    LogPrint("privatesend", "PSQUEUE -- Dynode %s is sending WAY too many psq messages\n", pdn->addr.ToString());
                    return;
                }
            }

            int nThreshold = pdn->nLastSsq + dnodeman.CountEnabled(MIN_PRIVATESEND_PEER_PROTO_VERSION)/5;
            LogPrint("privatesend", "PSQUEUE -- nLastSsq: %d  threshold: %d  nSsqCount: %d\n", pdn->nLastSsq, nThreshold, dnodeman.nSsqCount);
            //don't allow a few nodes to dominate the queuing process
            if(pdn->nLastSsq != 0 && nThreshold > dnodeman.nSsqCount) {
                LogPrint("privatesend", "PSQUEUE -- Dynode %s is sending too many psq messages\n", pdn->addr.ToString());
                return;
            }
            dnodeman.nSsqCount++;
            pdn->nLastSsq = dnodeman.nSsqCount;
            pdn->fAllowMixingTx = true;

            LogPrint("privatesend", "PSQUEUE -- new PrivateSend queue (%s) from Dynode %s\n", psq.ToString(), pdn->addr.ToString());
            if(pSubmittedToDynode && pSubmittedToDynode->vin.prevout == psq.vin.prevout) {
                psq.fTried = true;
            }
            vecPrivatesendQueue.push_back(psq);
            psq.Relay();
        }

    } else if(strCommand == NetMsgType::PSVIN) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("PSVIN -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            PushStatus(pfrom, STATUS_REJECTED, ERR_VERSION);
            return;
        }

        if(!fDyNode) {
            LogPrintf("PSVIN -- not a Dynode!\n");
            PushStatus(pfrom, STATUS_REJECTED, ERR_NOT_A_DN);
            return;
        }

        //do we have enough users in the current session?
        if(!IsSessionReady()) {
            LogPrintf("PSVIN -- session not complete!\n");
            PushStatus(pfrom, STATUS_REJECTED, ERR_SESSION);
            return;
        }

        CPrivateSendEntry entry;
        vRecv >> entry;

        LogPrint("privatesend", "PSVIN -- txCollateral %s", entry.txCollateral.ToString());

        if(entry.vecTxPSIn.size() > PRIVATESEND_ENTRY_MAX_SIZE) {
            LogPrintf("PSVIN -- ERROR: too many inputs! %d/%d\n", entry.vecTxPSIn.size(), PRIVATESEND_ENTRY_MAX_SIZE);
            PushStatus(pfrom, STATUS_REJECTED, ERR_MAXIMUM);
            return;
        }

        if(entry.vecTxPSOut.size() > PRIVATESEND_ENTRY_MAX_SIZE) {
            LogPrintf("PSVIN -- ERROR: too many outputs! %d/%d\n", entry.vecTxPSOut.size(), PRIVATESEND_ENTRY_MAX_SIZE);
            PushStatus(pfrom, STATUS_REJECTED, ERR_MAXIMUM);
            return;
        }

        //do we have the same denominations as the current session?
        if(!IsOutputsCompatibleWithSessionDenom(entry.vecTxPSOut)) {
            LogPrintf("PSVIN -- not compatible with existing transactions!\n");
            PushStatus(pfrom, STATUS_REJECTED, ERR_EXISTING_TX);
            return;
        }

        //check it like a transaction
        {
            CAmount nValueIn = 0;
            CAmount nValueOut = 0;

            CMutableTransaction tx;

            BOOST_FOREACH(const CTxOut txout, entry.vecTxPSOut) {
                nValueOut += txout.nValue;
                tx.vout.push_back(txout);

                if(txout.scriptPubKey.size() != 25) {
                    LogPrintf("PSVIN -- non-standard pubkey detected! scriptPubKey=%s\n", ScriptToAsmStr(txout.scriptPubKey));
                    PushStatus(pfrom, STATUS_REJECTED, ERR_NON_STANDARD_PUBKEY);
                    return;
                }
                if(!txout.scriptPubKey.IsNormalPaymentScript()) {
                    LogPrintf("PSVIN -- invalid script! scriptPubKey=%s\n", ScriptToAsmStr(txout.scriptPubKey));
                    PushStatus(pfrom, STATUS_REJECTED, ERR_INVALID_SCRIPT);
                    return;
                }
            }

            BOOST_FOREACH(const CTxIn txin, entry.vecTxPSIn) {
                tx.vin.push_back(txin);

                LogPrint("privatesend", "PSVIN -- txin=%s\n", txin.ToString());

                CTransaction txPrev;
                uint256 hash;
                if(GetTransaction(txin.prevout.hash, txPrev, Params().GetConsensus(), hash, true)) {
                    if(txPrev.vout.size() > txin.prevout.n)
                        nValueIn += txPrev.vout[txin.prevout.n].nValue;
                } else {
                    LogPrintf("PSVIN -- missing input! tx=%s", tx.ToString());
                    PushStatus(pfrom, STATUS_REJECTED, ERR_MISSING_TX);
                    return;
                }
            }

            // There should be no fee in mixing tx
            CAmount nFee = nValueIn - nValueOut;
            if(nFee != 0) {
                LogPrintf("PSVIN -- there should be no fee in mixing tx! fees: %lld, tx=%s", nFee, tx.ToString());
                PushStatus(pfrom, STATUS_REJECTED, ERR_FEES);
                return;
            }

            {
                LOCK(cs_main);
                CValidationState validationState;
                mempool.PrioritiseTransaction(tx.GetHash(), tx.GetHash().ToString(), 1000, 0.1*COIN);
                if(!AcceptToMemoryPool(mempool, validationState, CTransaction(tx), false, NULL, false, true, true)) {
                    LogPrintf("PSVIN -- transaction not valid! tx=%s", tx.ToString());
                    PushStatus(pfrom, STATUS_REJECTED, ERR_INVALID_TX);
                    return;
                }
            }
        }

        PoolMessage nMessageID = MSG_NOERR;

        if(AddEntry(entry, nMessageID)) {
            PushStatus(pfrom, STATUS_ACCEPTED, nMessageID);
            CheckPool();
            RelayStatus(STATUS_ACCEPTED);
        } else {
            PushStatus(pfrom, STATUS_REJECTED, nMessageID);
            SetNull();
        }

    } else if(strCommand == NetMsgType::PSSTATUSUPDATE) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("PSSTATUSUPDATE -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            return;
        }

        if(fDyNode) {
            // LogPrintf("PSSTATUSUPDATE -- Can't run on a Dynode!\n");
            return;
        }

        if(!pSubmittedToDynode) return;
        if((CNetAddr)pSubmittedToDynode->addr != (CNetAddr)pfrom->addr) {
            //LogPrintf("PSSTATUSUPDATE -- message doesn't match current Dynode: pSubmittedToDynode %s addr %s\n", pSubmittedToDynode->addr.ToString(), pfrom->addr.ToString());
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

        LogPrint("privatesend", "PSSTATUSUPDATE -- GetMessageByID: %s\n", GetMessageByID(PoolMessage(nMsgMessageID)));

        if(!CheckPoolStateUpdate(PoolState(nMsgState), nMsgEntriesCount, PoolStatusUpdate(nMsgStatusUpdate), PoolMessage(nMsgMessageID), nMsgSessionID)) {
            LogPrint("privatesend", "PSSTATUSUPDATE -- CheckPoolStateUpdate failed\n");
        }

    } else if(strCommand == NetMsgType::PSSIGNFINALTX) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("PSSIGNFINALTX -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            return;
        }

        if(!fDyNode) {
            LogPrintf("PSSIGNFINALTX -- not a Dynode!\n");
            return;
        }

        std::vector<CTxIn> vecTxIn;
        vRecv >> vecTxIn;

        LogPrint("privatesend", "PSSIGNFINALTX -- vecTxIn.size() %s\n", vecTxIn.size());

        int nTxInIndex = 0;
        int nTxInsCount = (int)vecTxIn.size();

        BOOST_FOREACH(const CTxIn txin, vecTxIn) {
            nTxInIndex++;
            if(!AddScriptSig(txin)) {
                LogPrint("privatesend", "PSSIGNFINALTX -- AddScriptSig() failed at %d/%d, session: %d\n", nTxInIndex, nTxInsCount, nSessionID);
                RelayStatus(STATUS_REJECTED);
                return;
            }
            LogPrint("privatesend", "PSSIGNFINALTX -- AddScriptSig() %d/%d success\n", nTxInIndex, nTxInsCount);
        }
        // all is good
        CheckPool();

    } else if(strCommand == NetMsgType::PSFINALTX) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("PSFINALTX -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            return;
        }

        if(fDyNode) {
            // LogPrintf("PSFINALTX -- Can't run on a Dynode!\n");
            return;
        }

        if(!pSubmittedToDynode) return;
        if((CNetAddr)pSubmittedToDynode->addr != (CNetAddr)pfrom->addr) {
            //LogPrintf("PSFINALTX -- message doesn't match current Dynode: pSubmittedToDynode %s addr %s\n", pSubmittedToDynode->addr.ToString(), pfrom->addr.ToString());
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
        SignFinalTransaction(txNew, pfrom);

    } else if(strCommand == NetMsgType::PSCOMPLETE) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("PSCOMPLETE -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            return;
        }

        if(fDyNode) {
            // LogPrintf("PSCOMPLETE -- Can't run on a Dynode!\n");
            return;
        }

        if(!pSubmittedToDynode) return;
        if((CNetAddr)pSubmittedToDynode->addr != (CNetAddr)pfrom->addr) {
            LogPrint("privatesend", "PSCOMPLETE -- message doesn't match current Dynode: pSubmittedToDynode=%s  addr=%s\n", pSubmittedToDynode->addr.ToString(), pfrom->addr.ToString());
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
            LogPrint("privatesend", "PSCOMPLETE -- message doesn't match current PrivateSend session: nSessionID: %d  nMsgSessionID: %d\n", privateSendPool.nSessionID, nMsgSessionID);
            return;
        }

        LogPrint("privatesend", "PSCOMPLETE -- nMsgSessionID %d  nMsgMessageID %d (%s)\n", nMsgSessionID, nMsgMessageID, GetMessageByID(PoolMessage(nMsgMessageID)));

        CompletedTransaction(PoolMessage(nMsgMessageID));
    }
}

void CPrivatesendPool::InitDenominations()
{
    vecPrivateSendDenominations.clear();
    /* Denominations

        A note about convertability. Within mixing pools, each denomination
        is convertable to another.

        For example:
        1DYN+1000 == (.1DYN+100)*10
        10DYN+10000 == (1DYN+1000)*10
    */
    vecPrivateSendDenominations.push_back( (10       * COIN)+10000 );
    vecPrivateSendDenominations.push_back( (1        * COIN)+1000 );
    vecPrivateSendDenominations.push_back( (.1       * COIN)+100 );
    vecPrivateSendDenominations.push_back( (.01      * COIN)+10 );
}

void CPrivatesendPool::ResetPool()
{
    nCachedLastSuccessBlock = 0;
    txMyCollateral = CMutableTransaction();
    vecDynodesUsed.clear();
    UnlockCoins();
    SetNull();
}

void CPrivatesendPool::SetNull()
{
    // DN side
    vecSessionCollaterals.clear();

    // Client side
    nEntriesCount = 0;
    fLastEntryAccepted = false;
    pSubmittedToDynode = NULL;

    // Both sides
    nState = POOL_STATE_IDLE;
    nSessionID = 0;
    nSessionDenom = 0;
    vecEntries.clear();
    finalMutableTransaction.vin.clear();
    finalMutableTransaction.vout.clear();
    nTimeLastSuccessfulStep = GetTimeMillis();
}

//
// Unlock coins after mixing fails or succeeds
//
void CPrivatesendPool::UnlockCoins()
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

std::string CPrivatesendPool::GetStateString() const
{
    switch(nState) {
        case POOL_STATE_IDLE:                   return "IDLE";
        case POOL_STATE_QUEUE:                  return "QUEUE";
        case POOL_STATE_ACCEPTING_ENTRIES:      return "ACCEPTING_ENTRIES";
        case POOL_STATE_SIGNING:                return "SIGNING";
        case POOL_STATE_ERROR:                  return "ERROR";
        case POOL_STATE_SUCCESS:                return "SUCCESS";
        default:                                return "UNKNOWN";
    }
}

std::string CPrivatesendPool::GetStatus()
{
    static int nStatusMessageProgress = 0;
    nStatusMessageProgress += 10;
    std::string strSuffix = "";

    if((pCurrentBlockIndex && pCurrentBlockIndex->nHeight - nCachedLastSuccessBlock < nMinBlockSpacing) || !dynodeSync.IsBlockchainSynced())
        return strAutoDenomResult;

    switch(nState) {
        case POOL_STATE_IDLE:
            return _("PrivateSend is idle.");
        case POOL_STATE_QUEUE:
            if(     nStatusMessageProgress % 70 <= 30) strSuffix = ".";
            else if(nStatusMessageProgress % 70 <= 50) strSuffix = "..";
            else if(nStatusMessageProgress % 70 <= 70) strSuffix = "...";
            return strprintf(_("Submitted to Dynode, waiting in queue %s"), strSuffix);;
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
                if(     nStatusMessageProgress % 70 <= 40) return strprintf(_("Submitted following entries to Dynode: %u / %d"), nEntriesCount, GetMaxPoolTransactions());
                else if(nStatusMessageProgress % 70 <= 50) strSuffix = ".";
                else if(nStatusMessageProgress % 70 <= 60) strSuffix = "..";
                else if(nStatusMessageProgress % 70 <= 70) strSuffix = "...";
                return strprintf(_("Submitted to Dynode, waiting for more entries ( %u / %d ) %s"), nEntriesCount, GetMaxPoolTransactions(), strSuffix);
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
void CPrivatesendPool::CheckPool()
{
    if(fDyNode) {
        LogPrint("privatesend", "CPrivatesendPool::CheckPool -- entries count %lu\n", GetEntriesCount());

        // If entries are full, create finalized transaction
        if(nState == POOL_STATE_ACCEPTING_ENTRIES && GetEntriesCount() >= GetMaxPoolTransactions()) {
            LogPrint("privatesend", "CPrivatesendPool::CheckPool -- FINALIZE TRANSACTIONS\n");
            CreateFinalTransaction();
            return;
        }

        // If we have all of the signatures, try to compile the transaction
        if(nState == POOL_STATE_SIGNING && IsSignaturesComplete()) {
            LogPrint("privatesend", "CPrivatesendPool::CheckPool -- SIGNING\n");
            CommitFinalTransaction();
            return;
        }
    }

    // reset if we're here for 10 seconds
    if((nState == POOL_STATE_ERROR || nState == POOL_STATE_SUCCESS) && GetTimeMillis() - nTimeLastSuccessfulStep >= 10000) {
        LogPrint("privatesend", "CPrivatesendPool::CheckPool -- timeout, RESETTING\n");
        UnlockCoins();
        SetNull();
    }
}

void CPrivatesendPool::CreateFinalTransaction()
{
    LogPrint("privatesend", "CPrivatesendPool::CreateFinalTransaction -- FINALIZE TRANSACTIONS\n");

    CMutableTransaction txNew;

    // make our new transaction
    for(int i = 0; i < GetEntriesCount(); i++) {
        BOOST_FOREACH(const CTxPSOut& txpsout, vecEntries[i].vecTxPSOut)
            txNew.vout.push_back(txpsout);

        BOOST_FOREACH(const CTxPSIn& txpsin, vecEntries[i].vecTxPSIn)
            txNew.vin.push_back(txpsin);
    }

    // BIP69 https://github.com/kristovatlas/bips/blob/master/bip-0069.mediawiki
    sort(txNew.vin.begin(), txNew.vin.end());
    sort(txNew.vout.begin(), txNew.vout.end());

    finalMutableTransaction = txNew;
    LogPrint("privatesend", "CPrivatesendPool::CreateFinalTransaction -- finalMutableTransaction=%s", txNew.ToString());

    // request signatures from clients
    RelayFinalTransaction(finalMutableTransaction);
    SetState(POOL_STATE_SIGNING);
}

void CPrivatesendPool::CommitFinalTransaction()
{
    if(!fDyNode) return; // check and relay final tx only on Dynode

    CTransaction finalTransaction = CTransaction(finalMutableTransaction);
    uint256 hashTx = finalTransaction.GetHash();

    LogPrint("privatesend", "CPrivatesendPool::CommitFinalTransaction -- finalTransaction=%s", finalTransaction.ToString());

    {
        // See if the transaction is valid
        TRY_LOCK(cs_main, lockMain);
        CValidationState validationState;
        mempool.PrioritiseTransaction(hashTx, hashTx.ToString(), 1000, 0.1*COIN);
        if(!lockMain || !AcceptToMemoryPool(mempool, validationState, finalTransaction, false, NULL, false, true, true))
        {
            LogPrintf("CPrivatesendPool::CommitFinalTransaction -- AcceptToMemoryPool() error: Transaction not valid\n");
            SetNull();
            // not much we can do in this case, just notify clients
            RelayCompletedTransaction(ERR_INVALID_TX);
            return;
        }
    }

    LogPrintf("CPrivatesendPool::CommitFinalTransaction -- CREATING PSTX\n");

    // create and sign Dynode pstx transaction
    if(!mapPrivatesendBroadcastTxes.count(hashTx)) {
        CPrivatesendBroadcastTx pstx(finalTransaction, activeDynode.vin, GetAdjustedTime());
        pstx.Sign();
        mapPrivatesendBroadcastTxes.insert(std::make_pair(hashTx, pstx));
    }

    LogPrintf("CPrivatesendPool::CommitFinalTransaction -- TRANSMITTING PSTX\n");

    CInv inv(MSG_PSTX, hashTx);
    RelayInv(inv);

    // Tell the clients it was successful
    RelayCompletedTransaction(MSG_SUCCESS);

    // Randomly charge clients
    ChargeRandomFees();

    // Reset
    LogPrint("privatesend", "CPrivatesendPool::CommitFinalTransaction -- COMPLETED -- RESETTING\n");
    SetNull();
}

//
// Charge clients a fee if they're abusive
//
// Why bother? PrivateSend uses collateral to ensure abuse to the process is kept to a minimum.
// The submission and signing stages are completely separate. In the cases where
// a client submits a transaction then refused to sign, there must be a cost. Otherwise they
// would be able to do this over and over again and bring the mixing to a hault.
//
// How does this work? Messages to Dynodes come in via NetMsgType::PSVIN, these require a valid collateral
// transaction for the client to be able to enter the pool. This transaction is kept by the Dynodes
// until the transaction is either complete or fails.
//
void CPrivatesendPool::ChargeFees()
{
    if(!fDyNode) return;

    //we don't need to charge collateral for every offence.
    if(GetRandInt(100) > 33) return;

    std::vector<CTransaction> vecOffendersCollaterals;

    if(nState == POOL_STATE_ACCEPTING_ENTRIES) {
        BOOST_FOREACH(const CTransaction& txCollateral, vecSessionCollaterals) {
            bool fFound = false;
            BOOST_FOREACH(const CPrivateSendEntry& entry, vecEntries)
                if(entry.txCollateral == txCollateral)
                    fFound = true;

            // This queue entry didn't send us the promised transaction
            if(!fFound) {
                LogPrintf("CPrivatesendPool::ChargeFees -- found uncooperative node (didn't send transaction), found offence\n");
                vecOffendersCollaterals.push_back(txCollateral);
            }
        }
    }

    if(nState == POOL_STATE_SIGNING) {
        // who didn't sign?
        BOOST_FOREACH(const CPrivateSendEntry entry, vecEntries) {
            BOOST_FOREACH(const CTxPSIn txpsin, entry.vecTxPSIn) {
                if(!txpsin.fHasSig) {
                    LogPrintf("CPrivatesendPool::ChargeFees -- found uncooperative node (didn't sign), found offence\n");
                    vecOffendersCollaterals.push_back(entry.txCollateral);
                }
            }
        }
    }

    // no offences found
    if(vecOffendersCollaterals.empty()) return;

    //mostly offending? Charge sometimes
    if((int)vecOffendersCollaterals.size() >= Params().PoolMaxTransactions() - 1 && GetRandInt(100) > 33) return;

    //everyone is an offender? That's not right
    if((int)vecOffendersCollaterals.size() >= Params().PoolMaxTransactions()) return;

    //charge one of the offenders randomly
    std::random_shuffle(vecOffendersCollaterals.begin(), vecOffendersCollaterals.end());

    if(nState == POOL_STATE_ACCEPTING_ENTRIES || nState == POOL_STATE_SIGNING) {
        LogPrintf("CPrivatesendPool::ChargeFees -- found uncooperative node (didn't %s transaction), charging fees: %s\n",
                (nState == POOL_STATE_SIGNING) ? "sign" : "send", vecOffendersCollaterals[0].ToString());

        LOCK(cs_main);

        CValidationState state;
        bool fMissingInputs;
        if(!AcceptToMemoryPool(mempool, state, vecOffendersCollaterals[0], false, &fMissingInputs, false, true)) {
            // should never really happen
            LogPrintf("CPrivatesendPool::ChargeFees -- ERROR: AcceptToMemoryPool failed!\n");
        } else {
            RelayTransaction(vecOffendersCollaterals[0]);
        }
    }
}

/*
    Charge the collateral randomly.
    Mixing is completely free, to pay miners we randomly pay the collateral of users.

    Collateral Fee Charges:

    Being that mixing has "no fees" we need to have some kind of cost associated
    with using it to stop abuse. Otherwise it could serve as an attack vector and
    allow endless transaction that would bloat Dynamic and make it unusable. To
    stop these kinds of attacks 1 in 10 successful transactions are charged. This
    adds up to a cost of 0.001DYN per transaction on average.
*/
void CPrivatesendPool::ChargeRandomFees()
{
    if(!fDyNode) return;

    LOCK(cs_main);

    BOOST_FOREACH(const CTransaction& txCollateral, vecSessionCollaterals) {

        if(GetRandInt(100) > 10) return;

        LogPrintf("CPrivatesendPool::ChargeRandomFees -- charging random fees, txCollateral=%s", txCollateral.ToString());

        CValidationState state;
        bool fMissingInputs;
        if(!AcceptToMemoryPool(mempool, state, txCollateral, false, &fMissingInputs, false, true)) {
            // should never really happen
            LogPrintf("CPrivatesendPool::ChargeRandomFees -- ERROR: AcceptToMemoryPool failed!\n");
        } else {
            RelayTransaction(txCollateral);
        }
    }
}

//
// Check for various timeouts (queue objects, mixing, etc)
//
void CPrivatesendPool::CheckTimeout()
{
    {
        TRY_LOCK(cs_privatesend, lockPS);
        if(!lockPS) return; // it's ok to fail here, we run this quite frequently

       int c = 0;
       vector<CPrivatesendQueue>::iterator it = vecPrivatesendQueue.begin();
       while(it != vecPrivatesendQueue.end()){
           if((*it).IsExpired()){
               LogPrint("privatesend", "CPrivatesendPool::CheckTimeout() : Removing expired queue entry - %d\n", c);
               it = vecPrivatesendQueue.erase(it);
           } else ++it;
           c++;
       }
    }

    if(!fEnablePrivateSend && !fDyNode) return;

    // catching hanging sessions
    if(!fDyNode) {
        switch(nState) {
            case POOL_STATE_ERROR:
                LogPrint("privatesend", "CPrivatesendPool::CheckTimeout -- Pool error -- Running CheckPool\n");
                CheckPool();
                break;
            case POOL_STATE_SUCCESS:
                LogPrint("privatesend", "CPrivatesendPool::CheckTimeout -- Pool success -- Running CheckPool\n");
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
        LogPrint("privatesend", "CPrivatesendPool::CheckTimeout -- %s timed out (%ds) -- restting\n",
                (nState == POOL_STATE_SIGNING) ? "Signing" : "Session", nTimeout);
        ChargeFees();
        UnlockCoins();
        SetNull();
        SetState(POOL_STATE_ERROR);
        strLastMessage = _("Session timed out.");
    }
}

/*
    Check to see if we're ready for submissions from clients
    After receiving multiple psa messages, the queue will switch to "accepting entries"
    which is the active state right before merging the transaction
*/
void CPrivatesendPool::CheckForCompleteQueue()
{
    if(!fEnablePrivateSend && !fDyNode) return;

    if(nState == POOL_STATE_QUEUE && IsSessionReady()) {
        SetState(POOL_STATE_ACCEPTING_ENTRIES);

        CPrivatesendQueue psq(nSessionDenom, activeDynode.vin, GetTime(), true);
        LogPrint("privatesend", "CPrivatesendPool::CheckForCompleteQueue -- queue is ready, signing and relaying (%s)\n", psq.ToString());
        psq.Sign();
        psq.Relay();
    }
}

// Check to make sure a given input matches an input in the pool and its scriptSig is valid
bool CPrivatesendPool::IsInputScriptSigValid(const CTxIn& txin)
{
    CMutableTransaction txNew;
    txNew.vin.clear();
    txNew.vout.clear();

    int i = 0;
    int nTxInIndex = -1;
    CScript sigPubKey = CScript();

    BOOST_FOREACH(CPrivateSendEntry& entry, vecEntries) {

        BOOST_FOREACH(const CTxPSOut& txpsout, entry.vecTxPSOut)
            txNew.vout.push_back(txpsout);

        BOOST_FOREACH(const CTxPSIn& txpsin, entry.vecTxPSIn) {
            txNew.vin.push_back(txpsin);

            if(txpsin.prevout == txin.prevout) {
                nTxInIndex = i;
                sigPubKey = txpsin.prevPubKey;
            }
            i++;
        }
    }

    if(nTxInIndex >= 0) { //might have to do this one input at a time?
        txNew.vin[nTxInIndex].scriptSig = txin.scriptSig;
        LogPrint("privatesend", "CPrivatesendPool::IsInputScriptSigValid -- verifying scriptSig %s\n", ScriptToAsmStr(txin.scriptSig).substr(0,24));
        if(!VerifyScript(txNew.vin[nTxInIndex].scriptSig, sigPubKey, SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_STRICTENC, MutableTransactionSignatureChecker(&txNew, nTxInIndex))) {
            LogPrint("privatesend", "CPrivatesendPool::IsInputScriptSigValid -- VerifyScript() failed on input %d\n", nTxInIndex);
            return false;
        }
    } else {
        LogPrint("privatesend", "CPrivatesendPool::IsInputScriptSigValid -- Failed to find matching input in pool, %s\n", txin.ToString());
        return false;
    }

    LogPrint("privatesend", "CPrivatesendPool::IsInputScriptSigValid -- Successfully validated input and scriptSig\n");
    return true;
}

// check to make sure the collateral provided by the client is valid
bool CPrivatesendPool::IsCollateralValid(const CTransaction& txCollateral)
{
    if(txCollateral.vout.empty()) return false;
    if(txCollateral.nLockTime != 0) return false;

    CAmount nValueIn = 0;
    CAmount nValueOut = 0;
    bool fMissingTx = false;

    BOOST_FOREACH(const CTxOut txout, txCollateral.vout) {
        nValueOut += txout.nValue;

        if(!txout.scriptPubKey.IsNormalPaymentScript()) {
            LogPrintf ("CPrivatesendPool::IsCollateralValid -- Invalid Script, txCollateral=%s", txCollateral.ToString());
            return false;
        }
    }

    BOOST_FOREACH(const CTxIn txin, txCollateral.vin) {
        CTransaction txPrev;
        uint256 hash;
        if(GetTransaction(txin.prevout.hash, txPrev, Params().GetConsensus(), hash, true)) {
            if(txPrev.vout.size() > txin.prevout.n)
                nValueIn += txPrev.vout[txin.prevout.n].nValue;
        } else {
            fMissingTx = true;
        }
    }

    if(fMissingTx) {
        LogPrint("privatesend", "CPrivatesendPool::IsCollateralValid -- Unknown inputs in collateral transaction, txCollateral=%s", txCollateral.ToString());
        return false;
    }

    //collateral transactions are required to pay out PRIVATESEND_COLLATERAL as a fee to the miners
    if(nValueIn - nValueOut < PRIVATESEND_COLLATERAL) {
        LogPrint("privatesend", "CPrivatesendPool::IsCollateralValid -- did not include enough fees in transaction: fees: %d, txCollateral=%s", nValueOut - nValueIn, txCollateral.ToString());
        return false;
    }

    LogPrint("privatesend", "CPrivatesendPool::IsCollateralValid -- %s", txCollateral.ToString());

    {
        LOCK(cs_main);
        CValidationState validationState;
        if(!AcceptToMemoryPool(mempool, validationState, txCollateral, false, NULL, false, true, true)) {
            LogPrint("privatesend", "CPrivatesendPool::IsCollateralValid -- didn't pass AcceptToMemoryPool()\n");
            return false;
        }
    }

    return true;
}


//
// Add a clients transaction to the pool
//
bool CPrivatesendPool::AddEntry(const CPrivateSendEntry& entryNew, PoolMessage& nMessageIDRet)
{
    if(!fDyNode) return false;

    BOOST_FOREACH(CTxIn txin, entryNew.vecTxPSIn) {
        if(txin.prevout.IsNull()) {
            LogPrint("privatesend", "CPrivatesendPool::AddEntry -- input not valid!\n");
            nMessageIDRet = ERR_INVALID_INPUT;
            return false;
        }
    }

    if(!IsCollateralValid(entryNew.txCollateral)) {
        LogPrint("privatesend", "CPrivatesendPool::AddEntry -- collateral not valid!\n");
        nMessageIDRet = ERR_INVALID_COLLATERAL;
        return false;
    }

    if(GetEntriesCount() >= GetMaxPoolTransactions()) {
        LogPrint("privatesend", "CPrivatesendPool::AddEntry -- entries is full!\n");
        nMessageIDRet = ERR_ENTRIES_FULL;
        return false;
    }

    BOOST_FOREACH(CTxIn txin, entryNew.vecTxPSIn) {
        LogPrint("privatesend", "looking for txin -- %s\n", txin.ToString());
        BOOST_FOREACH(const CPrivateSendEntry& entry, vecEntries) {
            BOOST_FOREACH(const CTxPSIn& txpsin, entry.vecTxPSIn) {
                if(txpsin.prevout == txin.prevout) {
                    LogPrint("privatesend", "CPrivatesendPool::AddEntry -- found in txin\n");
                    nMessageIDRet = ERR_ALREADY_HAVE;
                    return false;
                }
            }
        }
    }

    vecEntries.push_back(entryNew);

    LogPrint("privatesend", "CPrivatesendPool::AddEntry -- adding entry\n");
    nMessageIDRet = MSG_ENTRIES_ADDED;
    nTimeLastSuccessfulStep = GetTimeMillis();

    return true;
}

bool CPrivatesendPool::AddScriptSig(const CTxIn& txinNew)
{
    LogPrint("privatesend", "CPrivatesendPool::AddScriptSig -- scriptSig=%s\n", ScriptToAsmStr(txinNew.scriptSig).substr(0,24));

    BOOST_FOREACH(const CPrivateSendEntry& entry, vecEntries) {
        BOOST_FOREACH(const CTxPSIn& txpsin, entry.vecTxPSIn) {
            if(txpsin.scriptSig == txinNew.scriptSig) {
                LogPrint("privatesend", "CPrivatesendPool::AddScriptSig -- already exists\n");
                return false;
            }
        }
    }

    if(!IsInputScriptSigValid(txinNew)) {
        LogPrint("privatesend", "CPrivatesendPool::AddScriptSig -- Invalid scriptSig\n");
        return false;
    }

    LogPrint("privatesend", "CPrivatesendPool::AddScriptSig -- scriptSig=%s new\n", ScriptToAsmStr(txinNew.scriptSig).substr(0,24));

    BOOST_FOREACH(CTxIn& txin, finalMutableTransaction.vin) {
        if(txinNew.prevout == txin.prevout && txin.nSequence == txinNew.nSequence) {
            txin.scriptSig = txinNew.scriptSig;
            txin.prevPubKey = txinNew.prevPubKey;
            LogPrint("privatesend", "CPrivatesendPool::AddScriptSig -- adding to finalMutableTransaction, scriptSig=%s\n", ScriptToAsmStr(txinNew.scriptSig).substr(0,24));
        }
    }
    for(int i = 0; i < GetEntriesCount(); i++) {
        if(vecEntries[i].AddScriptSig(txinNew)) {
            LogPrint("privatesend", "CPrivatesendPool::AddScriptSig -- adding to entries, scriptSig=%s\n", ScriptToAsmStr(txinNew.scriptSig).substr(0,24));
            return true;
        }
    }

    LogPrintf("CPrivatesendPool::AddScriptSig -- Couldn't set sig!\n" );
    return false;
}

// Check to make sure everything is signed
bool CPrivatesendPool::IsSignaturesComplete()
{
    BOOST_FOREACH(const CPrivateSendEntry& entry, vecEntries)
        BOOST_FOREACH(const CTxPSIn& txpsin, entry.vecTxPSIn)
            if(!txpsin.fHasSig) return false;

    return true;
}

//
// Execute a mixing denomination via a Dynode.
// This is only ran from clients
//
bool CPrivatesendPool::SendDenominate(const std::vector<CTxIn>& vecTxIn, const std::vector<CTxOut>& vecTxOut)
{
    if(fDyNode) {
        LogPrintf("CPrivatesendPool::SendDenominate -- PrivateSend from a Dynode is not supported currently.\n");
        return false;
    }

    if(txMyCollateral == CMutableTransaction()) {
        LogPrintf("CPrivatesendPool:SendDenominate -- PrivateSend collateral not set\n");
        return false;
    }

    // lock the funds we're going to use
    BOOST_FOREACH(CTxIn txin, txMyCollateral.vin)
        vecOutPointLocked.push_back(txin.prevout);

    BOOST_FOREACH(CTxIn txin, vecTxIn)
        vecOutPointLocked.push_back(txin.prevout);

    // we should already be connected to a Dynode
    if(!nSessionID) {
        LogPrintf("CPrivatesendPool::SendDenominate -- No Dynode has been selected yet.\n");
        UnlockCoins();
        SetNull();
        return false;
    }

    if(!CheckDiskSpace()) {
        UnlockCoins();
        SetNull();
        fEnablePrivateSend = false;
        LogPrintf("CPrivatesendPool::SendDenominate -- Not enough disk space, disabling PrivateSend.\n");
        return false;
    }

    SetState(POOL_STATE_ACCEPTING_ENTRIES);
    strLastMessage = "";

    LogPrintf("CPrivatesendPool::SendDenominate -- Added transaction to pool.\n");

    //check it against the memory pool to make sure it's valid
    {
        CValidationState validationState;
        CMutableTransaction tx;

        BOOST_FOREACH(const CTxIn& txin, vecTxIn) {
            LogPrint("privatesend", "CPrivatesendPool::SendDenominate -- txin=%s\n", txin.ToString());
            tx.vin.push_back(txin);
        }

        BOOST_FOREACH(const CTxOut& txout, vecTxOut) {
            LogPrint("privatesend", "CPrivatesendPool::SendDenominate -- txout=%s\n", txout.ToString());
            tx.vout.push_back(txout);
        }

        LogPrintf("CPrivatesendPool::SendDenominate -- Submitting partial tx %s", tx.ToString());

        mempool.PrioritiseTransaction(tx.GetHash(), tx.GetHash().ToString(), 1000, 0.1*COIN);
        TRY_LOCK(cs_main, lockMain);
        if(!lockMain || !AcceptToMemoryPool(mempool, validationState, CTransaction(tx), false, NULL, false, true, true)) {
            LogPrintf("CPrivatesendPool::SendDenominate -- AcceptToMemoryPool() failed! tx=%s", tx.ToString());
            UnlockCoins();
            SetNull();
            return false;
        }
    }

    // store our entry for later use
    CPrivateSendEntry entry(vecTxIn, vecTxOut, txMyCollateral);
    vecEntries.push_back(entry);
    RelayIn(entry);
    nTimeLastSuccessfulStep = GetTimeMillis();

    return true;
}

// Incoming message from Dynode updating the progress of mixing
bool CPrivatesendPool::CheckPoolStateUpdate(PoolState nStateNew, int nEntriesCountNew, PoolStatusUpdate nStatusUpdate, PoolMessage nMessageID, int nSessionIDNew)
{
    if(fDyNode) return false;

    // do not update state when mixing client state is one of these
    if(nState == POOL_STATE_IDLE || nState == POOL_STATE_ERROR || nState == POOL_STATE_SUCCESS) return false;

    strAutoDenomResult = _("Dynode:") + " " + GetMessageByID(nMessageID);

    // if rejected at any state
    if(nStatusUpdate == STATUS_REJECTED) {
        LogPrintf("CPrivatesendPool::CheckPoolStateUpdate -- entry is rejected by Dynode\n");
        UnlockCoins();
        SetNull();
        SetState(POOL_STATE_ERROR);
        strLastMessage = GetMessageByID(nMessageID);
        return true;
    }

    if(nStatusUpdate == STATUS_ACCEPTED && nState == nStateNew) {
        if(nStateNew == POOL_STATE_QUEUE && nSessionID == 0 && nSessionIDNew != 0) {
            // new session id should be set only in POOL_STATE_QUEUE state
            nSessionID = nSessionIDNew;
            nTimeLastSuccessfulStep = GetTimeMillis();
            LogPrintf("CPrivatesendPool::CheckPoolStateUpdate -- set nSessionID to %d\n", nSessionID);
            return true;
        }
        else if(nStateNew == POOL_STATE_ACCEPTING_ENTRIES && nEntriesCount != nEntriesCountNew) {
            nEntriesCount = nEntriesCountNew;
            nTimeLastSuccessfulStep = GetTimeMillis();
            fLastEntryAccepted = true;
            LogPrintf("CPrivatesendPool::CheckPoolStateUpdate -- new entry accepted!\n");
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
bool CPrivatesendPool::SignFinalTransaction(const CTransaction& finalTransactionNew, CNode* pnode)
{
    if(fDyNode || pnode == NULL) return false;

    finalMutableTransaction = finalTransactionNew;
    LogPrintf("CPrivatesendPool::SignFinalTransaction -- finalMutableTransaction=%s", finalMutableTransaction.ToString());

    std::vector<CTxIn> sigs;

    //make sure my inputs/outputs are present, otherwise refuse to sign
    BOOST_FOREACH(const CPrivateSendEntry entry, vecEntries) {
        BOOST_FOREACH(const CTxPSIn txpsin, entry.vecTxPSIn) {
            /* Sign my transaction and all outputs */
            int nMyInputIndex = -1;
            CScript prevPubKey = CScript();
            CTxIn txin = CTxIn();

            for(unsigned int i = 0; i < finalMutableTransaction.vin.size(); i++) {
                if(finalMutableTransaction.vin[i] == txpsin) {
                    nMyInputIndex = i;
                    prevPubKey = txpsin.prevPubKey;
                    txin = txpsin;
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
                    LogPrintf("CPrivatesendPool::SignFinalTransaction -- My entries are not correct! Refusing to sign: nFoundOutputsCount: %d, nTargetOuputsCount: %d\n", nFoundOutputsCount, nTargetOuputsCount);
                    UnlockCoins();
                    SetNull();

                    return false;
                }

                const CKeyStore& keystore = *pwalletMain;

                LogPrint("privatesend", "CPrivatesendPool::SignFinalTransaction -- Signing my input %i\n", nMyInputIndex);
                if(!SignSignature(keystore, prevPubKey, finalMutableTransaction, nMyInputIndex, int(SIGHASH_ALL|SIGHASH_ANYONECANPAY))) { // changes scriptSig
                    LogPrint("privatesend", "CPrivatesendPool::SignFinalTransaction -- Unable to sign my own transaction!\n");
                    // not sure what to do here, it will timeout...?
                }

                sigs.push_back(finalMutableTransaction.vin[nMyInputIndex]);
                LogPrint("privatesend", "CPrivatesendPool::SignFinalTransaction -- nMyInputIndex: %d, sigs.size(): %d, scriptSig=%s\n", nMyInputIndex, (int)sigs.size(), ScriptToAsmStr(finalMutableTransaction.vin[nMyInputIndex].scriptSig));
            }
        }
    }

    if(sigs.empty()) {
        LogPrintf("CPrivatesendPool::SignFinalTransaction -- can't sign anything!\n");
        UnlockCoins();
        SetNull();

        return false;
    }

    // push all of our signatures to the Dynode
    LogPrintf("CPrivatesendPool::SignFinalTransaction -- pushing sigs to the Dynode, finalMutableTransaction=%s", finalMutableTransaction.ToString());
    pnode->PushMessage(NetMsgType::PSSIGNFINALTX, sigs);
    SetState(POOL_STATE_SIGNING);
    nTimeLastSuccessfulStep = GetTimeMillis();

    return true;
}

void CPrivatesendPool::NewBlock()
{
    static int64_t nTimeNewBlockReceived = 0;

    //we we're processing lots of blocks, we'll just leave
    if(GetTime() - nTimeNewBlockReceived < 10) return;
    nTimeNewBlockReceived = GetTime();
    LogPrint("privatesend", "CPrivatesendPool::NewBlock\n");

    CheckTimeout();
}

// mixing transaction was completed (failed or successful)
void CPrivatesendPool::CompletedTransaction(PoolMessage nMessageID)
{
    if(fDyNode) return;

    if(nMessageID == MSG_SUCCESS) {
        LogPrintf("CompletedTransaction -- success\n");
        nCachedLastSuccessBlock = pCurrentBlockIndex->nHeight;
    } else {
        LogPrintf("CompletedTransaction -- error\n");
    }
    UnlockCoins();
    SetNull();
    strLastMessage = GetMessageByID(nMessageID);
}

//
// Passively run mixing in the background to anonymize funds based on the given configuration.
//
bool CPrivatesendPool::DoAutomaticDenominating(bool fDryRun)
{
    if(!fEnablePrivateSend || fDyNode || !pCurrentBlockIndex) return false;
    if(!pwalletMain || pwalletMain->IsLocked(true)) return false;
    if(nState != POOL_STATE_IDLE) return false;

    if(!dynodeSync.IsDynodeListSynced()) {
        strAutoDenomResult = _("Can't mix while sync in progress.");
        return false;
    }

    switch(nWalletBackups) {
        case 0:
            LogPrint("privatesend", "CPrivatesendPool::DoAutomaticDenominating -- Automatic backups disabled, no mixing available.\n");
            strAutoDenomResult = _("Automatic backups disabled") + ", " + _("no mixing available.");
            fEnablePrivateSend = false; // stop mixing
            pwalletMain->nKeysLeftSinceAutoBackup = 0; // no backup, no "keys since last backup"
            return false;
        case -1:
            // Automatic backup failed, nothing else we can do until user fixes the issue manually.
            // There is no way to bring user attention in daemon mode so we just update status and
            // keep spaming if debug is on.
            LogPrint("privatesend", "CPrivatesendPool::DoAutomaticDenominating -- ERROR! Failed to create automatic backup.\n");
            strAutoDenomResult = _("ERROR! Failed to create automatic backup") + ", " + _("see debug.log for details.");
            return false;
        case -2:
            // We were able to create automatic backup but keypool was not replenished because wallet is locked.
            // There is no way to bring user attention in daemon mode so we just update status and
            // keep spaming if debug is on.
            LogPrint("privatesend", "CPrivatesendPool::DoAutomaticDenominating -- WARNING! Failed to create replenish keypool, please unlock your wallet to do so.\n");
            strAutoDenomResult = _("WARNING! Failed to replenish keypool, please unlock your wallet to do so.") + ", " + _("see debug.log for details.");
            return false;
    }

    if(pwalletMain->nKeysLeftSinceAutoBackup < PRIVATESEND_KEYS_THRESHOLD_STOP) {
        // We should never get here via mixing itself but probably smth else is still actively using keypool
        LogPrint("privatesend", "CPrivatesendPool::DoAutomaticDenominating -- Very low number of keys left: %d, no mixing available.\n", pwalletMain->nKeysLeftSinceAutoBackup);
        strAutoDenomResult = strprintf(_("Very low number of keys left: %d") + ", " + _("no mixing available."), pwalletMain->nKeysLeftSinceAutoBackup);
        // It's getting really dangerous, stop mixing
        fEnablePrivateSend = false;
        return false;
    } else if(pwalletMain->nKeysLeftSinceAutoBackup < PRIVATESEND_KEYS_THRESHOLD_WARNING) {
        // Low number of keys left but it's still more or less safe to continue
        LogPrint("privatesend", "CPrivatesendPool::DoAutomaticDenominating -- Very low number of keys left: %d\n", pwalletMain->nKeysLeftSinceAutoBackup);
        strAutoDenomResult = strprintf(_("Very low number of keys left: %d"), pwalletMain->nKeysLeftSinceAutoBackup);

        if(fCreateAutoBackups) {
            LogPrint("privatesend", "CPrivatesendPool::DoAutomaticDenominating -- Trying to create new backup.\n");
            std::string warningString;
            std::string errorString;

            if(!AutoBackupWallet(pwalletMain, "", warningString, errorString)) {
                if(!warningString.empty()) {
                    // There were some issues saving backup but yet more or less safe to continue
                    LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- WARNING! Something went wrong on automatic backup: %s\n", warningString);
                }
                if(!errorString.empty()) {
                    // Things are really broken
                    LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- ERROR! Failed to create automatic backup: %s\n", errorString);
                    strAutoDenomResult = strprintf(_("ERROR! Failed to create automatic backup") + ": %s", errorString);
                    return false;
                }
            }
        } else {
            // Wait for someone else (e.g. GUI action) to create automatic backup for us
            return false;
        }
    }

    LogPrint("privatesend", "CPrivatesendPool::DoAutomaticDenominating -- Keys left since latest backup: %d\n", pwalletMain->nKeysLeftSinceAutoBackup);

    if(GetEntriesCount() > 0) {
        strAutoDenomResult = _("Mixing in progress...");
        return false;
    }

    TRY_LOCK(cs_privatesend, lockPS);
    if(!lockPS) {
        strAutoDenomResult = _("Lock is already in place.");
        return false;
    }

    if(!fDryRun && pwalletMain->IsLocked(true)) {
        strAutoDenomResult = _("Wallet is locked.");
        return false;
    }

    if(!fPrivateSendMultiSession && pCurrentBlockIndex->nHeight - nCachedLastSuccessBlock < nMinBlockSpacing) {
        LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- Last successful PrivateSend action was too recent\n");
        strAutoDenomResult = _("Last successful PrivateSend action was too recent.");
        return false;
    }

    if(dnodeman.size() == 0) {
        LogPrint("privatesend", "CPrivatesendPool::DoAutomaticDenominating -- No Dynodes detected\n");
        strAutoDenomResult = _("No Dynodes detected.");
        return false;
    }

    CAmount nValueMin = vecPrivateSendDenominations.back();

    // if there are no confirmed PS collateral inputs yet
    if(!pwalletMain->HasCollateralInputs()) {
        // should have some additional amount for them
        nValueMin += PRIVATESEND_COLLATERAL*4;
    }

    // including denoms but applying some restrictions
    CAmount nBalanceNeedsAnonymized = pwalletMain->GetNeedsToBeAnonymizedBalance(nValueMin);

    // anonymizable balance is way too small
    if(nBalanceNeedsAnonymized < nValueMin) {
        LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- Not enough funds to anonymize\n");
        strAutoDenomResult = _("Not enough funds to anonymize.");
        return false;
    }

    // excluding denoms
    CAmount nBalanceAnonimizableNonDenom = pwalletMain->GetAnonymizableBalance(true);
    // denoms
    CAmount nBalanceDenominatedConf = pwalletMain->GetDenominatedBalance();
    CAmount nBalanceDenominatedUnconf = pwalletMain->GetDenominatedBalance(true);
    CAmount nBalanceDenominated = nBalanceDenominatedConf + nBalanceDenominatedUnconf;

    LogPrint("privatesend", "CPrivatesendPool::DoAutomaticDenominating -- nValueMin: %f, nBalanceNeedsAnonymized: %f, nBalanceAnonimizableNonDenom: %f, nBalanceDenominatedConf: %f, nBalanceDenominatedUnconf: %f, nBalanceDenominated: %f\n",
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
    if(nBalanceAnonimizableNonDenom >= nValueMin + PRIVATESEND_COLLATERAL && nBalanceDenominated < nPrivateSendAmount*COIN)
        return CreateDenominated();

    //check if we have the collateral sized inputs
    if(!pwalletMain->HasCollateralInputs())
        return !pwalletMain->HasCollateralInputs(false) && MakeCollateralAmounts();

    if(nSessionID) {
        strAutoDenomResult = _("Mixing in progress...");
        return false;
    }

    // Initial phase, find a Dynode
    // Clean if there is anything left from previous session
    UnlockCoins();
    SetNull();

    // should be no unconfirmed denoms in non-multi-session mode
    if(!fPrivateSendMultiSession && nBalanceDenominatedUnconf > 0) {
        LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- Found unconfirmed denominated outputs, will wait till they confirm to continue.\n");
        strAutoDenomResult = _("Found unconfirmed denominated outputs, will wait till they confirm to continue.");
        return false;
    }

    //check our collateral and create new if needed
    std::string strReason;
    if(txMyCollateral == CMutableTransaction()) {
        if(!pwalletMain->CreateCollateralTransaction(txMyCollateral, strReason)) {
            LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- create collateral error:%s\n", strReason);
            return false;
        }
    } else {
        if(!IsCollateralValid(txMyCollateral)) {
            LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- invalid collateral, recreating...\n");
            if(!pwalletMain->CreateCollateralTransaction(txMyCollateral, strReason)) {
                LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- create collateral error: %s\n", strReason);
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
    if(nLiquidityProvider || fUseQueue) {

        // Look through the queues and see if anything matches
        BOOST_FOREACH(CPrivatesendQueue& psq, vecPrivatesendQueue) {
            // only try each queue once
            if(psq.fTried) continue;
            psq.fTried = true;

            if(psq.IsExpired()) continue;

            CDynode* pdn = dnodeman.Find(psq.vin);
            if(pdn == NULL) {
                LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- psq Dynode is not in Dynode list, Dynode=%s\n", psq.vin.prevout.ToStringShort());
                continue;
            }

            if(pdn->nProtocolVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) continue;

            std::vector<int> vecBits;
            if(!GetDenominationsBits(psq.nDenom, vecBits)) {
                // incompatible denom
                continue;
            }

            // mixing rate limit i.e. nLastSsq check should already pass in PSQUEUE ProcessMessage
            // in order for psq to get into vecPrivatesendQueue, so we should be safe to mix already,
            // no need for additional verification here

            LogPrint("privatesend", "CPrivatesendPool::DoAutomaticDenominating -- found valid queue: %s\n", psq.ToString());

            CAmount nValueInTmp = 0;
            std::vector<CTxIn> vecTxInTmp;
            std::vector<COutput> vCoinsTmp;

            // Try to match their denominations if possible, select at least 1 denominations
            if(!pwalletMain->SelectCoinsByDenominations(psq.nDenom, vecPrivateSendDenominations[vecBits.front()], nBalanceNeedsAnonymized, vecTxInTmp, vCoinsTmp, nValueInTmp, 0, nPrivateSendRounds)) {
                LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- Couldn't match denominations %d %d (%s)\n", vecBits.front(), psq.nDenom, GetDenominationsToString(psq.nDenom));
                continue;
            }

            vecDynodesUsed.push_back(psq.vin);

            CNode* pnodeFound = NULL;
            {
                LOCK(cs_vNodes);
                pnodeFound = FindNode(pdn->addr);
                if(pnodeFound) {
                    if(pnodeFound->fDisconnect) {
                        continue;
                    } else {
                        pnodeFound->AddRef();
                    }
                }
            }

            LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- attempt to connect to Dynode from queue, addr=%s\n", pdn->addr.ToString());
            // connect to Dynode and submit the queue request
            CNode* pnode = (pnodeFound && pnodeFound->fDynode) ? pnodeFound : ConnectNode((CAddress)pdn->addr, NULL, true);
            if(pnode) {
                pSubmittedToDynode = pdn;
                nSessionDenom = psq.nDenom;

                pnode->PushMessage(NetMsgType::PSACCEPT, nSessionDenom, txMyCollateral);
                LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- connected (from queue), sending PSACCEPT: nSessionDenom: %d (%s), addr=%s\n",
                        nSessionDenom, GetDenominationsToString(nSessionDenom), pnode->addr.ToString());
                strAutoDenomResult = _("Mixing in progress...");
                SetState(POOL_STATE_QUEUE);
                nTimeLastSuccessfulStep = GetTimeMillis();
                if(pnodeFound) {
                    pnodeFound->Release();
                }
                return true;
            } else {
                LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- can't connect, addr=%s\n", pdn->addr.ToString());
                strAutoDenomResult = _("Error connecting to Dynode.");
                continue;
            }
        }
    }

    // do not initiate queue if we are a liquidity provider to avoid useless inter-mixing
    if(nLiquidityProvider) return false;

    int nTries = 0;

    // ** find the coins we'll use
    std::vector<CTxIn> vecTxIn;
    CAmount nValueInTmp = 0;
    if(!pwalletMain->SelectCoinsMix(nValueMin, nBalanceNeedsAnonymized, vecTxIn, nValueInTmp, 0, nPrivateSendRounds)) {
        // this should never happen
        LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- Can't mix: no compatible inputs found!\n");
        strAutoDenomResult = _("Can't mix: no compatible inputs found!");
        return false;
    }

    // otherwise, try one randomly
    while(nTries < 10) {
        CDynode* pdn = dnodeman.FindRandomNotInVec(vecDynodesUsed, MIN_PRIVATESEND_PEER_PROTO_VERSION);
        if(pdn == NULL) {
            LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- Can't find random Dynode!\n");
            strAutoDenomResult = _("Can't find random Dynode.");
            return false;
        }
        vecDynodesUsed.push_back(pdn->vin);

        if(pdn->nLastSsq != 0 && pdn->nLastSsq + nDnCountEnabled/5 > dnodeman.nSsqCount) {
            LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- Too early to mix on this Dynode!"
                        " Dynode=%s  addr=%s  nLastSsq=%d  CountEnabled/5=%d  nSsqCount=%d\n",
                        pdn->vin.prevout.ToStringShort(), pdn->addr.ToString(), pdn->nLastSsq,
                        nDnCountEnabled/5, dnodeman.nSsqCount);
            nTries++;
            continue;
        }

        CNode* pnodeFound = NULL;
        {
            LOCK(cs_vNodes);
            pnodeFound = FindNode(pdn->addr);
            if(pnodeFound) {
                if(pnodeFound->fDisconnect) {
                    nTries++;
                    continue;
                } else {
                    pnodeFound->AddRef();
                }
            }
        }

        LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- attempt %d connection to Dynode %s\n", nTries, pdn->addr.ToString());
        CNode* pnode = (pnodeFound && pnodeFound->fDynode) ? pnodeFound : ConnectNode((CAddress)pdn->addr, NULL, true);
        if(pnode) {
            LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- connected, addr=%s\n", pdn->addr.ToString());
            pSubmittedToDynode = pdn;

            std::vector<CAmount> vecAmounts;
            pwalletMain->ConvertList(vecTxIn, vecAmounts);
            // try to get a single random denom out of vecAmounts
            while(nSessionDenom == 0) {
                nSessionDenom = GetDenominationsByAmounts(vecAmounts);
            }

            pnode->PushMessage(NetMsgType::PSACCEPT, nSessionDenom, txMyCollateral);
            LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- connected, sending PSACCEPT, nSessionDenom: %d (%s)\n",
                    nSessionDenom, GetDenominationsToString(nSessionDenom));
            strAutoDenomResult = _("Mixing in progress...");
            SetState(POOL_STATE_QUEUE);
            nTimeLastSuccessfulStep = GetTimeMillis();
            if(pnodeFound) {
                pnodeFound->Release();
            }
            return true;
        } else {
            LogPrintf("CPrivatesendPool::DoAutomaticDenominating -- can't connect, addr=%s\n", pdn->addr.ToString());
            nTries++;
            continue;
        }
    }

    strAutoDenomResult = _("No compatible Dynode found.");
    return false;
}

bool CPrivatesendPool::SubmitDenominate()
{
    std::string strError;
    std::vector<CTxIn> vecTxInRet;
    std::vector<CTxOut> vecTxOutRet;

    // Submit transaction to the pool if we get here
    // Try to use only inputs with the same number of rounds starting from the highest number of rounds possible
    for(int i = nPrivateSendRounds; i > 0; i--) {
        if(PrepareDenominate(i - 1, i, strError, vecTxInRet, vecTxOutRet)) {
            LogPrintf("CPrivatesendPool::SubmitDenominate -- Running PrivateSend denominate for %d rounds, success\n", i);
            return SendDenominate(vecTxInRet, vecTxOutRet);
        }
        LogPrint("privatesend", "CPrivatesendPool::SubmitDenominate -- Running PrivateSend denominate for %d rounds, error: %s\n", i, strError);
    }

    // We failed? That's strange but let's just make final attempt and try to mix everything
    if(PrepareDenominate(0, nPrivateSendRounds, strError, vecTxInRet, vecTxOutRet)) {
        LogPrintf("CPrivatesendPool::SubmitDenominate -- Running PrivateSend denominate for all rounds, success\n");
        return SendDenominate(vecTxInRet, vecTxOutRet);
    }

    // Should never actually get here but just in case
    LogPrintf("CPrivatesendPool::SubmitDenominate -- Running PrivateSend denominate for all rounds, error: %s\n", strError);
    strAutoDenomResult = strError;
    return false;
}

bool CPrivatesendPool::PrepareDenominate(int nMinRounds, int nMaxRounds, std::string& strErrorRet, std::vector<CTxIn>& vecTxInRet, std::vector<CTxOut>& vecTxOutRet)
{
    if(!pwalletMain) return false;

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
    CReserveKey reservekey(pwalletMain);

    /*
        Select the coins we'll use

        if nMinRounds >= 0 it means only denominated inputs are going in and coming out
    */
    std::vector<int> vecBits;
    if (!GetDenominationsBits(nSessionDenom, vecBits)) {
        strErrorRet = "Incorrect session denom";
        return false;
    }

    bool fSelected = pwalletMain->SelectCoinsByDenominations(nSessionDenom, vecPrivateSendDenominations[vecBits.front()], GetMaxPoolAmount(), vecTxIn, vCoins, nValueIn, nMinRounds, nMaxRounds);     
    if (nMinRounds >= 0 && !fSelected) {
        strErrorRet = "Can't select current denominated inputs";
        return false;
    }

    LogPrintf("CPrivatesendPool::PrepareDenominate -- max value: %f\n", (double)nValueIn/COIN);

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
            CAmount nValueDenom = vecPrivateSendDenominations[nBit];
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

                    CScript scriptChange;
                    CPubKey vchPubKey;
                    // use a unique change address
                    assert(reservekey.GetReservedKey(vchPubKey)); // should never fail, as we just unlocked
                    scriptChange = GetScriptForDestination(vchPubKey.GetID());
                    reservekey.KeepKey();

                    // add new output
                    CTxOut txout(nValueDenom, scriptChange);
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

    if (GetDenominations(vecTxOutRet) != nSessionDenom) {
        // unlock used coins on failure
        LOCK(pwalletMain->cs_wallet);
        BOOST_FOREACH(CTxIn txin, vecTxInRet) {
            pwalletMain->UnlockCoin(txin.prevout);
        }
        strErrorRet = "Can't make current denominated outputs";
        return false;
    }

    // We also do not care about full amount as long as we have right denominations
    return true;
}

// Create collaterals by looping through inputs grouped by addresses
bool CPrivatesendPool::MakeCollateralAmounts()
{
    std::vector<CompactTallyItem> vecTally;
    if(!pwalletMain->SelectCoinsGrouppedByAddresses(vecTally, false)) {
        LogPrint("privatesend", "CPrivatesendPool::MakeCollateralAmounts -- SelectCoinsGrouppedByAddresses can't find any inputs!\n");
        return false;
    }

    BOOST_FOREACH(CompactTallyItem& item, vecTally) {
        if(!MakeCollateralAmounts(item)) continue;
        return true;
    }

    LogPrintf("CPrivatesendPool::MakeCollateralAmounts -- failed!\n");
    return false;
}

// Split up large inputs or create fee sized inputs
bool CPrivatesendPool::MakeCollateralAmounts(const CompactTallyItem& tallyItem)
{
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
    assert(reservekeyCollateral.GetReservedKey(vchPubKey)); // should never fail, as we just unlocked
    scriptCollateral = GetScriptForDestination(vchPubKey.GetID());

    vecSend.push_back((CRecipient){scriptCollateral, PRIVATESEND_COLLATERAL*4, false});

    // try to use non-denominated and not DN-like funds first, select them explicitly
    CCoinControl coinControl;
    coinControl.fAllowOtherInputs = false;
    coinControl.fAllowWatchOnly = false;
    // send change to the same address so that we were able create more denoms out of it later
    coinControl.destChange = tallyItem.address.Get();
    BOOST_FOREACH(const CTxIn& txin, tallyItem.vecTxIn)
        coinControl.Select(txin.prevout);

    bool fSuccess = pwalletMain->CreateTransaction(vecSend, wtx, reservekeyChange,
            nFeeRet, nChangePosRet, strFail, &coinControl, true, ONLY_NONDENOMINATED_NOT1000IFDN);
    if(!fSuccess) {
        // if we failed (most likeky not enough funds), try to use all coins instead -
        // DN-like funds should not be touched in any case and we can't mix denominated without collaterals anyway
        LogPrintf("CPrivatesendPool::MakeCollateralAmounts -- ONLY_NONDENOMINATED_NOT1000IFDN Error: %s\n", strFail);
        CCoinControl *coinControlNull = NULL;
        fSuccess = pwalletMain->CreateTransaction(vecSend, wtx, reservekeyChange,
                nFeeRet, nChangePosRet, strFail, coinControlNull, true, ONLY_NOT1000IFDN);
        if(!fSuccess) {
            LogPrintf("CPrivatesendPool::MakeCollateralAmounts -- ONLY_NOT1000IFDN Error: %s\n", strFail);
            reservekeyCollateral.ReturnKey();
            return false;
        }
    }

    reservekeyCollateral.KeepKey();

    LogPrintf("CPrivatesendPool::MakeCollateralAmounts -- txid=%s\n", wtx.GetHash().GetHex());

    // use the same nCachedLastSuccessBlock as for PS mixinx to prevent race
    if(!pwalletMain->CommitTransaction(wtx, reservekeyChange)) {
        LogPrintf("CPrivatesendPool::MakeCollateralAmounts -- CommitTransaction failed!\n");
        return false;
    }

    nCachedLastSuccessBlock = pCurrentBlockIndex->nHeight;

    return true;
}

// Create denominations by looping through inputs grouped by addresses
bool CPrivatesendPool::CreateDenominated()
{
    std::vector<CompactTallyItem> vecTally;
    if(!pwalletMain->SelectCoinsGrouppedByAddresses(vecTally)) {
        LogPrint("privatesend", "CPrivatesendPool::CreateDenominated -- SelectCoinsGrouppedByAddresses can't find any inputs!\n");
        return false;
    }

    bool fCreateMixingCollaterals = !pwalletMain->HasCollateralInputs();

    BOOST_FOREACH(CompactTallyItem& item, vecTally) {
        if(!CreateDenominated(item, fCreateMixingCollaterals)) continue;
        return true;
    }

    LogPrintf("CPrivatesendPool::CreateDenominated -- failed!\n");
    return false;
}

// Create denominations
bool CPrivatesendPool::CreateDenominated(const CompactTallyItem& tallyItem, bool fCreateMixingCollaterals)
{
    std::vector<CRecipient> vecSend;
    CAmount nValueLeft = tallyItem.nAmount;
    nValueLeft -= PRIVATESEND_COLLATERAL; // leave some room for fees

    LogPrintf("CreateDenominated0 nValueLeft: %f\n", (float)nValueLeft/COIN);
    // make our collateral address
    CReserveKey reservekeyCollateral(pwalletMain);

    CScript scriptCollateral;
    CPubKey vchPubKey;
    assert(reservekeyCollateral.GetReservedKey(vchPubKey)); // should never fail, as we just unlocked
    scriptCollateral = GetScriptForDestination(vchPubKey.GetID());

    // ****** Add collateral outputs ************ /

    if(fCreateMixingCollaterals) {
        vecSend.push_back((CRecipient){scriptCollateral, PRIVATESEND_COLLATERAL*4, false});
        nValueLeft -= PRIVATESEND_COLLATERAL*4;
    }

    // ****** Add denoms ************ /

    // make our denom addresses
    CReserveKey reservekeyDenom(pwalletMain);

    // try few times - skipping smallest denoms first if there are too much already, if failed - use them
    int nOutputsTotal = 0;
    bool fSkip = true;
    do {

        BOOST_REVERSE_FOREACH(CAmount nDenomValue, vecPrivateSendDenominations) {

            if(fSkip) {
                // Note: denoms are skipped if there are already DENOMS_COUNT_MAX of them
                // and there are still larger denoms which can be used for mixing

                // check skipped denoms
                if(IsDenomSkipped(nDenomValue)) continue;

                // find new denoms to skip if any (ignore the largest one)
                if(nDenomValue != vecPrivateSendDenominations[0] && pwalletMain->CountInputsWithAmount(nDenomValue) > DENOMS_COUNT_MAX) {
                    strAutoDenomResult = strprintf(_("Too many %f denominations, removing."), (float)nDenomValue/COIN);
                    LogPrintf("CPrivatesendPool::CreateDenominated -- %s\n", strAutoDenomResult);
                    vecDenominationsSkipped.push_back(nDenomValue);
                    continue;
                }
            }

            int nOutputs = 0;

            // add each output up to 10 times until it can't be added again
            while(nValueLeft - nDenomValue >= 0 && nOutputs <= 10) {
                CScript scriptDenom;
                CPubKey vchPubKey;
                //use a unique change address
                assert(reservekeyDenom.GetReservedKey(vchPubKey)); // should never fail, as we just unlocked
                scriptDenom = GetScriptForDestination(vchPubKey.GetID());
                // TODO: do not keep reservekeyDenom here
                reservekeyDenom.KeepKey();

                vecSend.push_back((CRecipient){ scriptDenom, nDenomValue, false });

                //increment outputs and subtract denomination amount
                nOutputs++;
                nValueLeft -= nDenomValue;
                LogPrintf("CreateDenominated1: nOutputsTotal: %d, nOutputs: %d, nValueLeft: %f\n", nOutputsTotal, nOutputs, (float)nValueLeft/COIN);
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
    coinControl.destChange = tallyItem.address.Get();
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
        LogPrintf("CPrivatesendPool::CreateDenominated -- Error: %s\n", strFail);
        // TODO: return reservekeyDenom here
        reservekeyCollateral.ReturnKey();
        return false;
    }

    // TODO: keep reservekeyDenom here
    reservekeyCollateral.KeepKey();

    if(!pwalletMain->CommitTransaction(wtx, reservekeyChange)) {
        LogPrintf("CPrivatesendPool::CreateDenominated -- CommitTransaction failed!\n");
        return false;
    }

    // use the same nCachedLastSuccessBlock as for PS mixing to prevent race
    nCachedLastSuccessBlock = pCurrentBlockIndex->nHeight;
    LogPrintf("CPrivatesendPool::CreateDenominated -- txid=%s\n", wtx.GetHash().GetHex());

    return true;
}

bool CPrivatesendPool::IsOutputsCompatibleWithSessionDenom(const std::vector<CTxPSOut>& vecTxPSOut)
{
    if(GetDenominations(vecTxPSOut) == 0) return false;

    BOOST_FOREACH(const CPrivateSendEntry entry, vecEntries) {
        LogPrintf("CPrivatesendPool::IsOutputsCompatibleWithSessionDenom -- vecTxPSOut denom %d, entry.vecTxPSOut denom %d\n", GetDenominations(vecTxPSOut), GetDenominations(entry.vecTxPSOut));
        if(GetDenominations(vecTxPSOut) != GetDenominations(entry.vecTxPSOut)) return false;
    }

    return true;
}

bool CPrivatesendPool::IsAcceptableDenomAndCollateral(int nDenom, CTransaction txCollateral, PoolMessage& nMessageIDRet)
{
    if(!fDyNode) return false;

    // is denom even smth legit?
    std::vector<int> vecBits;
    if(!GetDenominationsBits(nDenom, vecBits)) {
        LogPrint("privatesend", "CPrivatesendPool::IsAcceptableDenomAndCollateral -- denom not valid!\n");
        nMessageIDRet = ERR_DENOM;
        return false;
    }

    // check collateral
    if(!fUnitTest && !IsCollateralValid(txCollateral)) {
        LogPrint("privatesend", "CPrivatesendPool::IsAcceptableDenomAndCollateral -- collateral not valid!\n");
        nMessageIDRet = ERR_INVALID_COLLATERAL;
        return false;
    }

    return true;
}

bool CPrivatesendPool::CreateNewSession(int nDenom, CTransaction txCollateral, PoolMessage& nMessageIDRet)
{
    if(!fDyNode || nSessionID != 0) return false;

    // new session can only be started in idle mode
    if(nState != POOL_STATE_IDLE) {
        nMessageIDRet = ERR_MODE;
        LogPrintf("CPrivatesendPool::CreateNewSession -- incompatible mode: nState=%d\n", nState);
        return false;
    }

    if(!IsAcceptableDenomAndCollateral(nDenom, txCollateral, nMessageIDRet)) {
        return false;
    }

    // start new session
    nMessageIDRet = MSG_NOERR;
    nSessionID = GetRandInt(999999)+1;
    nSessionDenom = nDenom;

    SetState(POOL_STATE_QUEUE);
    nTimeLastSuccessfulStep = GetTimeMillis();

    if(!fUnitTest) {
        //broadcast that I'm accepting entries, only if it's the first entry through
        CPrivatesendQueue psq(nDenom, activeDynode.vin, GetTime(), false);
        LogPrint("privatesend", "CPrivatesendPool::CreateNewSession -- signing and relaying new queue: %s\n", psq.ToString());
        psq.Sign();
        psq.Relay();
        vecPrivatesendQueue.push_back(psq);
    }

    vecSessionCollaterals.push_back(txCollateral);
    LogPrintf("CPrivatesendPool::CreateNewSession -- new session created, nSessionID: %d  nSessionDenom: %d (%s)  vecSessionCollaterals.size(): %d\n",
            nSessionID, nSessionDenom, GetDenominationsToString(nSessionDenom), vecSessionCollaterals.size());

    return true;
}

bool CPrivatesendPool::AddUserToExistingSession(int nDenom, CTransaction txCollateral, PoolMessage& nMessageIDRet)
{
    if(!fDyNode || nSessionID == 0 || IsSessionReady()) return false;

    if(!IsAcceptableDenomAndCollateral(nDenom, txCollateral, nMessageIDRet)) {
        return false;
    }

    // we only add new users to an existing session when we are in queue mode
    if(nState != POOL_STATE_QUEUE) {
        nMessageIDRet = ERR_MODE;
        LogPrintf("CPrivatesendPool::AddUserToExistingSession -- incompatible mode: nState=%d\n", nState);
        return false;
    }

    if(nDenom != nSessionDenom) {
        LogPrintf("CPrivatesendPool::AddUserToExistingSession -- incompatible denom %d (%s) != nSessionDenom %d (%s)\n",
                    nDenom, GetDenominationsToString(nDenom), nSessionDenom, GetDenominationsToString(nSessionDenom));
        nMessageIDRet = ERR_DENOM;
        return false;
    }

    // count new user as accepted to an existing session

    nMessageIDRet = MSG_NOERR;
    nTimeLastSuccessfulStep = GetTimeMillis();
    vecSessionCollaterals.push_back(txCollateral);

    LogPrintf("CPrivatesendPool::AddUserToExistingSession -- new user accepted, nSessionID: %d  nSessionDenom: %d (%s)  vecSessionCollaterals.size(): %d\n",
            nSessionID, nSessionDenom, GetDenominationsToString(nSessionDenom), vecSessionCollaterals.size());

    return true;
}

/*  Create a nice string to show the denominations
    Function returns as follows (for 4 denominations):
        ( bit on if present )
        bit 0           - 100
        bit 1           - 10
        bit 2           - 1
        bit 3           - .1
        bit 4 and so on - out-of-bounds
        none of above   - non-denom
*/
std::string CPrivatesendPool::GetDenominationsToString(int nDenom)
{
    std::string strDenom = "";
    int nMaxDenoms = vecPrivateSendDenominations.size();

    if(nDenom >= (1 << nMaxDenoms)) {
        return "out-of-bounds";
    }

    for (int i = 0; i < nMaxDenoms; ++i) {
        if(nDenom & (1 << i)) {
            strDenom += (strDenom.empty() ? "" : "+") + FormatMoney(vecPrivateSendDenominations[i]);
        }
    }

    if(strDenom.empty()) {
        return "non-denom";
    }

    return strDenom;
}

int CPrivatesendPool::GetDenominations(const std::vector<CTxPSOut>& vecTxPSOut)
{
    std::vector<CTxOut> vecTxOut;

    BOOST_FOREACH(CTxPSOut out, vecTxPSOut)
        vecTxOut.push_back(out);

    return GetDenominations(vecTxOut);
}

/*  Return a bitshifted integer representing the denominations in this list
    Function returns as follows (for 4 denominations):
        ( bit on if present )
        100       - bit 0
        10        - bit 1
        1         - bit 2
        .1        - bit 3
        non-denom - 0, all bits off
*/
int CPrivatesendPool::GetDenominations(const std::vector<CTxOut>& vecTxOut, bool fSingleRandomDenom)
{
    std::vector<std::pair<CAmount, int> > vecDenomUsed;

    // make a list of denominations, with zero uses
    BOOST_FOREACH(CAmount nDenomValue, vecPrivateSendDenominations)
        vecDenomUsed.push_back(std::make_pair(nDenomValue, 0));

    // look for denominations and update uses to 1
    BOOST_FOREACH(CTxOut txout, vecTxOut) {
        bool found = false;
        BOOST_FOREACH (PAIRTYPE(CAmount, int)& s, vecDenomUsed) {
            if(txout.nValue == s.first) {
                s.second = 1;
                found = true;
            }
        }
        if(!found) return 0;
    }

    int nDenom = 0;
    int c = 0;
    // if the denomination is used, shift the bit on
    BOOST_FOREACH (PAIRTYPE(CAmount, int)& s, vecDenomUsed) {
        int bit = (fSingleRandomDenom ? GetRandInt(2) : 1) & s.second;
        nDenom |= bit << c++;
        if(fSingleRandomDenom && bit) break; // use just one random denomination
    }

    return nDenom;
}

bool CPrivatesendPool::GetDenominationsBits(int nDenom, std::vector<int> &vecBitsRet)
{
    // ( bit on if present, 4 denominations example )
    // bit 0 - 100DYN+1
    // bit 1 - 10DYN+1
    // bit 2 - 1DYN+1
    // bit 3 - .1DYN+1

    int nMaxDenoms = vecPrivateSendDenominations.size();

    if(nDenom >= (1 << nMaxDenoms)) return false;

    vecBitsRet.clear();

    for (int i = 0; i < nMaxDenoms; ++i) {
        if(nDenom & (1 << i)) {
            vecBitsRet.push_back(i);
        }
    }

    return !vecBitsRet.empty();
}

int CPrivatesendPool::GetDenominationsByAmounts(const std::vector<CAmount>& vecAmount)
{
    CScript scriptTmp = CScript();
    std::vector<CTxOut> vecTxOut;

    BOOST_REVERSE_FOREACH(CAmount nAmount, vecAmount) {
        CTxOut txout(nAmount, scriptTmp);
        vecTxOut.push_back(txout);
    }

    return GetDenominations(vecTxOut, true);
}

std::string CPrivatesendPool::GetMessageByID(PoolMessage nMessageID)
{
    switch (nMessageID) {
        case ERR_ALREADY_HAVE:          return _("Already have that input.");
        case ERR_DENOM:                 return _("No matching denominations found for mixing.");
        case ERR_ENTRIES_FULL:          return _("Entries are full.");
        case ERR_EXISTING_TX:           return _("Not compatible with existing transactions.");
        case ERR_FEES:                  return _("Transaction fees are too high.");
        case ERR_INVALID_COLLATERAL:    return _("Collateral not valid.");
        case ERR_INVALID_INPUT:         return _("Input is not valid.");
        case ERR_INVALID_SCRIPT:        return _("Invalid script detected.");
        case ERR_INVALID_TX:            return _("Transaction not valid.");
        case ERR_MAXIMUM:               return _("Entry exceeds maximum size.");
        case ERR_DN_LIST:               return _("Not in the Dynode list.");
        case ERR_MODE:                  return _("Incompatible mode.");
        case ERR_NON_STANDARD_PUBKEY:   return _("Non-standard public key detected.");
        case ERR_NOT_A_DN:              return _("This is not a Dynode.");
        case ERR_QUEUE_FULL:            return _("Dynode queue is full.");
        case ERR_RECENT:                return _("Last PrivateSend was too recent.");
        case ERR_SESSION:               return _("Session not complete!");
        case ERR_MISSING_TX:            return _("Missing input transaction information.");
        case ERR_VERSION:               return _("Incompatible version.");
        case MSG_NOERR:                 return _("No errors detected.");
        case MSG_SUCCESS:               return _("Transaction created successfully.");
        case MSG_ENTRIES_ADDED:         return _("Your entries added successfully.");
        default:                        return _("Unknown response.");
    }
}

bool CPrivateSendEntry::AddScriptSig(const CTxIn& txin)
{
    BOOST_FOREACH(CTxPSIn& txpsin, vecTxPSIn) {
        if(txpsin.prevout == txin.prevout && txpsin.nSequence == txin.nSequence) {
            if(txpsin.fHasSig) return false;

            txpsin.scriptSig = txin.scriptSig;
            txpsin.prevPubKey = txin.prevPubKey;
            txpsin.fHasSig = true;

            return true;
        }
    }

    return false;
}

bool CPrivatesendQueue::Sign()
{
    if(!fDyNode) return false;

    std::string strMessage = vin.ToString() + boost::lexical_cast<std::string>(nDenom) + boost::lexical_cast<std::string>(nTime) + boost::lexical_cast<std::string>(fReady);

    if(!CMessageSigner::SignMessage(strMessage, vchSig, activeDynode.keyDynode)) {
        LogPrintf("CPrivatesendQueue::Sign -- SignMessage() failed, %s\n", ToString());
        return false;
    }

    return CheckSignature(activeDynode.pubKeyDynode);
}

bool CPrivatesendQueue::CheckSignature(const CPubKey& pubKeyDynode)
{
    std::string strMessage = vin.ToString() + boost::lexical_cast<std::string>(nDenom) + boost::lexical_cast<std::string>(nTime) + boost::lexical_cast<std::string>(fReady);
    std::string strError = "";

    if(!CMessageSigner::VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
        LogPrintf("CPrivatesendQueue::CheckSignature -- Got bad Dynode queue signature: %s; error: %s\n", ToString(), strError);
        return false;
    }

    return true;
}

bool CPrivatesendQueue::Relay()
{
    std::vector<CNode*> vNodesCopy = CopyNodeVector();
    BOOST_FOREACH(CNode* pnode, vNodesCopy)
        if(pnode->nVersion >= MIN_PRIVATESEND_PEER_PROTO_VERSION)
            pnode->PushMessage(NetMsgType::PSQUEUE, (*this));

    ReleaseNodeVector(vNodesCopy);
    return true;
}

bool CPrivatesendBroadcastTx::Sign()
{
    if(!fDyNode) return false;

    std::string strMessage = tx.GetHash().ToString() + boost::lexical_cast<std::string>(sigTime);

    if(!CMessageSigner::SignMessage(strMessage, vchSig, activeDynode.keyDynode)) {
        LogPrintf("CPrivatesendBroadcastTx::Sign -- SignMessage() failed\n");
        return false;
    }

    return CheckSignature(activeDynode.pubKeyDynode);
}

bool CPrivatesendBroadcastTx::CheckSignature(const CPubKey& pubKeyDynode)
{
    std::string strMessage = tx.GetHash().ToString() + boost::lexical_cast<std::string>(sigTime);
    std::string strError = "";

    if(!CMessageSigner::VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
        LogPrintf("CPrivatesendBroadcastTx::CheckSignature -- Got bad pstx signature, error: %s\n", strError);
        return false;
    }

    return true;
}

void CPrivatesendPool::RelayFinalTransaction(const CTransaction& txFinal)
{
    LOCK(cs_vNodes);
    BOOST_FOREACH(CNode* pnode, vNodes)
        if(pnode->nVersion >= MIN_PRIVATESEND_PEER_PROTO_VERSION)
            pnode->PushMessage(NetMsgType::PSFINALTX, nSessionID, txFinal);
}

void CPrivatesendPool::RelayIn(const CPrivateSendEntry& entry)
{
    if(!pSubmittedToDynode) return;

    CNode* pnode = FindNode(pSubmittedToDynode->addr);
    if(pnode != NULL) {
        LogPrintf("CPrivatesendPool::RelayIn -- found Dynode, relaying message to %s\n", pnode->addr.ToString());
        pnode->PushMessage(NetMsgType::PSVIN, entry);
    }
}

void CPrivatesendPool::PushStatus(CNode* pnode, PoolStatusUpdate nStatusUpdate, PoolMessage nMessageID)
{
    if(!pnode) return;
    pnode->PushMessage(NetMsgType::PSSTATUSUPDATE, nSessionID, (int)nState, (int)vecEntries.size(), (int)nStatusUpdate, (int)nMessageID);
}

void CPrivatesendPool::RelayStatus(PoolStatusUpdate nStatusUpdate, PoolMessage nMessageID)
{
    LOCK(cs_vNodes);
    BOOST_FOREACH(CNode* pnode, vNodes)
        if(pnode->nVersion >= MIN_PRIVATESEND_PEER_PROTO_VERSION)
            PushStatus(pnode, nStatusUpdate, nMessageID);
}

void CPrivatesendPool::RelayCompletedTransaction(PoolMessage nMessageID)
{
    LOCK(cs_vNodes);
    BOOST_FOREACH(CNode* pnode, vNodes)
        if(pnode->nVersion >= MIN_PRIVATESEND_PEER_PROTO_VERSION)
            pnode->PushMessage(NetMsgType::PSCOMPLETE, nSessionID, (int)nMessageID);
}

void CPrivatesendPool::SetState(PoolState nStateNew)
{
    if(fDyNode && (nStateNew == POOL_STATE_ERROR || nStateNew == POOL_STATE_SUCCESS)) {
        LogPrint("privatesend", "CPrivatesendPool::SetState -- Can't set state to ERROR or SUCCESS as a Dynode. \n");
        return;
    }

    LogPrintf("CPrivatesendPool::SetState -- nState: %d, nStateNew: %d\n", nState, nStateNew);
    nState = nStateNew;
}

void CPrivatesendPool::UpdatedBlockTip(const CBlockIndex *pindex)
{
    pCurrentBlockIndex = pindex;
    LogPrint("privatesend", "CPrivatesendPool::UpdatedBlockTip -- pCurrentBlockIndex->nHeight: %d\n", pCurrentBlockIndex->nHeight);

    if(!fLiteMode && dynodeSync.IsDynodeListSynced()) {
        NewBlock();
    }
}

//TODO: Rename/move to core
void ThreadCheckPrivateSendPool()
{
    if(fLiteMode) return; // disable all Dynamic specific functionality

    static bool fOneThread;
    if(fOneThread) return;
    fOneThread = true;

    // Make this thread recognisable as the PrivateSend thread
    RenameThread("dynamic-privatesend");

    unsigned int nTick = 0;
    unsigned int nDoAutoNextRun = nTick + PRIVATESEND_AUTO_TIMEOUT_MIN;

    while (true)
    {
        MilliSleep(1000);

        // try to sync from all available nodes, one step at a time
        dynodeSync.ProcessTick();

        if(dynodeSync.IsBlockchainSynced() && !ShutdownRequested()) {

            nTick++;

            // make sure to check all Dynodes first
            dnodeman.Check();

            // check if we should activate or ping every few minutes,
            // slightly postpone first run to give net thread a chance to connect to some peers
            if(nTick % DYNODE_MIN_DNP_SECONDS == 15)
                activeDynode.ManageState();

            if(nTick % 60 == 0) {
                dnodeman.ProcessDynodeConnections();
                dnodeman.CheckAndRemove();
                dnpayments.CheckAndRemove();
                instantsend.CheckAndRemove();
            }
            if(fDyNode && (nTick % (60 * 5) == 0)) {
                dnodeman.DoFullVerificationStep();
            }

            if(nTick % (60 * 5) == 0) {
                governance.DoMaintenance();
            }

            privateSendPool.CheckTimeout();
            privateSendPool.CheckForCompleteQueue();

            if(nDoAutoNextRun == nTick) {
                privateSendPool.DoAutomaticDenominating();
                nDoAutoNextRun = nTick + PRIVATESEND_AUTO_TIMEOUT_MIN + GetRandInt(PRIVATESEND_AUTO_TIMEOUT_MAX - PRIVATESEND_AUTO_TIMEOUT_MIN);
            }
        }
    }
}
