// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "activestormnode.h"
#include "coincontrol.h"
#include "consensus/validation.h"
#include "sandstorm.h"
#include "init.h"
#include "instantx.h"
#include "stormnode-payments.h"
#include "stormnode-sync.h"
#include "stormnodeman.h"
#include "script/sign.h"
#include "txmempool.h"
#include "util.h"
#include "utilmoneystr.h"

#include <boost/lexical_cast.hpp>

int nPrivateSendRounds = DEFAULT_PRIVATESEND_ROUNDS;
int nPrivateSendAmount = DEFAULT_PRIVATESEND_AMOUNT;
int nLiquidityProvider = DEFAULT_PRIVATESEND_LIQUIDITY;
bool fEnablePrivateSend = false;
bool fPrivateSendMultiSession = DEFAULT_PRIVATESEND_MULTISESSION;

CSandstormPool sandStormPool;
CSandStormSigner sandStormSigner;
std::map<uint256, CSandstormBroadcastTx> mapSandstormBroadcastTxes;
std::vector<CAmount> vecPrivateSendDenominations;

void CSandstormPool::ProcessMessage(CNode* pfrom, std::string& strCommand, CDataStream& vRecv)
{
    if(fLiteMode) return; // ignore all DarkSilk related functionality
    if(!stormnodeSync.IsBlockchainSynced()) return;

    if(strCommand == NetMsgType::SSACCEPT) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("SSACCEPT -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            PushStatus(pfrom, STATUS_REJECTED, ERR_VERSION);
            return;
        }

        if(!fStormNode) {
            LogPrintf("SSACCEPT -- not a Stormnode!\n");
            PushStatus(pfrom, STATUS_REJECTED, ERR_NOT_A_SN);
            return;
        }

        if(IsSessionReady()) {
            // too many users in this session already, reject new ones
            LogPrintf("SSACCEPT -- queue is already full!\n");
            PushStatus(pfrom, STATUS_ACCEPTED, ERR_QUEUE_FULL);
            return;
        }

        int nDenom;
        CTransaction txCollateral;
        vRecv >> nDenom >> txCollateral;

        LogPrint("privatesend", "SSACCEPT -- nDenom %d (%s)  txCollateral %s", nDenom, GetDenominationsToString(nDenom), txCollateral.ToString());

        CStormnode* psn = snodeman.Find(activeStormnode.vin);
        if(psn == NULL) {
            PushStatus(pfrom, STATUS_REJECTED, ERR_SN_LIST);
            return;
        }

        if(vecSessionCollaterals.size() == 0 && psn->nLastSsq != 0 &&
            psn->nLastSsq + snodeman.CountEnabled(MIN_PRIVATESEND_PEER_PROTO_VERSION)/5 > snodeman.nSsqCount)
        {
            LogPrintf("SSACCEPT -- last ssq too recent, must wait: addr=%s\n", pfrom->addr.ToString());
            PushStatus(pfrom, STATUS_REJECTED, ERR_RECENT);
            return;
        }

        PoolMessage nMessageID = MSG_NOERR;

        bool fResult = nSessionID == 0  ? CreateNewSession(nDenom, txCollateral, nMessageID)
                                        : AddUserToExistingSession(nDenom, txCollateral, nMessageID);
        if(fResult) {
            LogPrintf("SSACCEPT -- is compatible, please submit!\n");
            PushStatus(pfrom, STATUS_ACCEPTED, nMessageID);
            return;
        } else {
            LogPrintf("SSACCEPT -- not compatible with existing transactions!\n");
            PushStatus(pfrom, STATUS_REJECTED, nMessageID);
            return;
        }

    } else if(strCommand == NetMsgType::SSQUEUE) {
        TRY_LOCK(cs_sandstorm, lockRecv);
        if(!lockRecv) return;

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrint("privatesend", "SSQUEUE -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            return;
        }

        CSandstormQueue ssq;
        vRecv >> ssq;

        // process every ssq only once
        BOOST_FOREACH(CSandstormQueue q, vecSandstormQueue) {
            if(q == ssq) {
                // LogPrint("privatesend", "SSQUEUE -- %s seen\n", ssq.ToString());
                return;
            }
        }

        LogPrint("privatesend", "SSQUEUE -- %s new\n", ssq.ToString());

        if(ssq.IsExpired() || ssq.nTime > GetTime() + PRIVATESEND_QUEUE_TIMEOUT) return;

        CStormnode* psn = snodeman.Find(ssq.vin);
        if(psn == NULL) return;

        if(!ssq.CheckSignature(psn->pubKeyStormnode)) {
            // we probably have outdated info
            snodeman.AskForSN(pfrom, ssq.vin);
            return;
        }

        // if the queue is ready, submit if we can
        if(ssq.fReady) {
            if(!pSubmittedToStormnode) return;
            if((CNetAddr)pSubmittedToStormnode->addr != (CNetAddr)psn->addr) {
                LogPrintf("SSQUEUE -- message doesn't match current Stormnode: pSubmittedToStormnode=%s, addr=%s\n", pSubmittedToStormnode->addr.ToString(), psn->addr.ToString());
                return;
            }

            if(nState == POOL_STATE_QUEUE) {
                LogPrint("privatesend", "SSQUEUE -- PrivateSend queue (%s) is ready on stormnode %s\n", ssq.ToString(), psn->addr.ToString());
                SubmitDenominate();
            }
        } else {
            BOOST_FOREACH(CSandstormQueue q, vecSandstormQueue) {
                if(q.vin == ssq.vin) {
                    // no way same sn can send another "not yet ready" ssq this soon
                    LogPrint("privatesend", "SSQUEUE -- Stormnode %s is sending WAY too many ssq messages\n", psn->addr.ToString());
                    return;
                }
            }

            int nThreshold = psn->nLastSsq + snodeman.CountEnabled(MIN_PRIVATESEND_PEER_PROTO_VERSION)/5;
            LogPrint("privatesend", "SSQUEUE -- nLastSsq: %d  threshold: %d  nSsqCount: %d\n", psn->nLastSsq, nThreshold, snodeman.nSsqCount);
            //don't allow a few nodes to dominate the queuing process
            if(psn->nLastSsq != 0 && nThreshold > snodeman.nSsqCount) {
                LogPrint("privatesend", "SSQUEUE -- Stormnode %s is sending too many ssq messages\n", psn->addr.ToString());
                return;
            }
            snodeman.nSsqCount++;
            psn->nLastSsq = snodeman.nSsqCount;
            psn->fAllowMixingTx = true;

            LogPrint("privatesend", "SSQUEUE -- new PrivateSend queue (%s) from stormnode %s\n", ssq.ToString(), psn->addr.ToString());
            if(pSubmittedToStormnode && pSubmittedToStormnode->vin.prevout == ssq.vin.prevout) {
                ssq.fTried = true;
            }
            vecSandstormQueue.push_back(ssq);
            ssq.Relay();
        }

    } else if(strCommand == NetMsgType::SSVIN) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("SSVIN -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            PushStatus(pfrom, STATUS_REJECTED, ERR_VERSION);
            return;
        }

        if(!fStormNode) {
            LogPrintf("SSVIN -- not a Stormnode!\n");
            PushStatus(pfrom, STATUS_REJECTED, ERR_NOT_A_SN);
            return;
        }

        //do we have enough users in the current session?
        if(!IsSessionReady()) {
            LogPrintf("SSVIN -- session not complete!\n");
            PushStatus(pfrom, STATUS_REJECTED, ERR_SESSION);
            return;
        }

        CSandStormEntry entry;
        vRecv >> entry;

        LogPrint("privatesend", "SSVIN -- txCollateral %s", entry.txCollateral.ToString());

        //do we have the same denominations as the current session?
        if(!IsOutputsCompatibleWithSessionDenom(entry.vecTxSSOut)) {
            LogPrintf("SSVIN -- not compatible with existing transactions!\n");
            PushStatus(pfrom, STATUS_REJECTED, ERR_EXISTING_TX);
            return;
        }

        //check it like a transaction
        {
            CAmount nValueIn = 0;
            CAmount nValueOut = 0;

            CMutableTransaction tx;

            BOOST_FOREACH(const CTxOut txout, entry.vecTxSSOut) {
                nValueOut += txout.nValue;
                tx.vout.push_back(txout);

                if(txout.scriptPubKey.size() != 25) {
                    LogPrintf("SSVIN -- non-standard pubkey detected! scriptPubKey=%s\n", ScriptToAsmStr(txout.scriptPubKey));
                    PushStatus(pfrom, STATUS_REJECTED, ERR_NON_STANDARD_PUBKEY);
                    return;
                }
                if(!txout.scriptPubKey.IsNormalPaymentScript()) {
                    LogPrintf("SSVIN -- invalid script! scriptPubKey=%s\n", ScriptToAsmStr(txout.scriptPubKey));
                    PushStatus(pfrom, STATUS_REJECTED, ERR_INVALID_SCRIPT);
                    return;
                }
            }

            BOOST_FOREACH(const CTxIn txin, entry.vecTxSSIn) {
                tx.vin.push_back(txin);

                LogPrint("privatesend", "SSVIN -- txin=%s\n", txin.ToString());

                CTransaction txPrev;
                uint256 hash;
                if(GetTransaction(txin.prevout.hash, txPrev, Params().GetConsensus(), hash, true)) {
                    if(txPrev.vout.size() > txin.prevout.n)
                        nValueIn += txPrev.vout[txin.prevout.n].nValue;
                } else {
                    LogPrintf("SSVIN -- missing input! tx=%s", tx.ToString());
                    PushStatus(pfrom, STATUS_REJECTED, ERR_MISSING_TX);
                    return;
                }
            }

            if(nValueIn > PRIVATESEND_POOL_MAX) {
                LogPrintf("SSVIN -- more than PrivateSend pool max! nValueIn: %lld, tx=%s", nValueIn, tx.ToString());
                PushStatus(pfrom, STATUS_REJECTED, ERR_MAXIMUM);
                return;
            }

            // Allow lowest denom (at max) as a a fee. Normally shouldn't happen though.
            // TODO: Or do not allow fees at all?
            if(nValueIn - nValueOut > vecPrivateSendDenominations.back()) {
                LogPrintf("SSVIN -- fees are too high! fees: %lld, tx=%s", nValueIn - nValueOut, tx.ToString());
                PushStatus(pfrom, STATUS_REJECTED, ERR_FEES);
                return;
            }

            {
                LOCK(cs_main);
                CValidationState validationState;
                mempool.PrioritiseTransaction(tx.GetHash(), tx.GetHash().ToString(), 1000, 0.1*COIN);
                if(!AcceptToMemoryPool(mempool, validationState, CTransaction(tx), false, NULL, false, true, true)) {
                    LogPrintf("SSVIN -- transaction not valid! tx=%s", tx.ToString());
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

    } else if(strCommand == NetMsgType::SSSTATUSUPDATE) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("SSSTATUSUPDATE -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            return;
        }

        if(fStormNode) {
            // LogPrintf("SSSTATUSUPDATE -- Can't run on a Stormnode!\n");
            return;
        }

        if(!pSubmittedToStormnode) return;
        if((CNetAddr)pSubmittedToStormnode->addr != (CNetAddr)pfrom->addr) {
            //LogPrintf("SSSTATUSUPDATE -- message doesn't match current Stormnode: pSubmittedToStormnode %s addr %s\n", pSubmittedToStormnode->addr.ToString(), pfrom->addr.ToString());
            return;
        }

        int nMsgSessionID;
        int nMsgState;
        int nMsgEntriesCount;
        int nMsgStatusUpdate;
        int nMsgMessageID;
        vRecv >> nMsgSessionID >> nMsgState >> nMsgEntriesCount >> nMsgStatusUpdate >> nMsgMessageID;

        LogPrint("privatesend", "SSSTATUSUPDATE -- nMsgSessionID %d  nMsgState: %d  nEntriesCount: %d  nMsgStatusUpdate: %d  nMsgMessageID %d\n",
                nMsgSessionID, nMsgState, nEntriesCount, nMsgStatusUpdate, nMsgMessageID);

        if(nMsgState < POOL_STATE_MIN || nMsgState > POOL_STATE_MAX) {
            LogPrint("privatesend", "SSSTATUSUPDATE -- nMsgState is out of bounds: %d\n", nMsgState);
            return;
        }

        if(nMsgStatusUpdate < STATUS_REJECTED || nMsgStatusUpdate > STATUS_ACCEPTED) {
            LogPrint("privatesend", "SSSTATUSUPDATE -- nMsgStatusUpdate is out of bounds: %d\n", nMsgStatusUpdate);
            return;
        }

        if(nMsgMessageID < MSG_POOL_MIN || nMsgMessageID > MSG_POOL_MAX) {
            LogPrint("privatesend", "SSSTATUSUPDATE -- nMsgMessageID is out of bounds: %d\n", nMsgMessageID);
            return;
        }

        LogPrint("privatesend", "SSSTATUSUPDATE -- GetMessageByID: %s\n", GetMessageByID(PoolMessage(nMsgMessageID)));

        if(!CheckPoolStateUpdate(PoolState(nMsgState), nMsgEntriesCount, PoolStatusUpdate(nMsgStatusUpdate), PoolMessage(nMsgMessageID), nMsgSessionID)) {
            LogPrint("privatesend", "SSSTATUSUPDATE -- CheckPoolStateUpdate failed\n");
        }

    } else if(strCommand == NetMsgType::SSSIGNFINALTX) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("SSSIGNFINALTX -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            return;
        }

        if(!fStormNode) {
            LogPrintf("SSSIGNFINALTX -- not a Stormnode!\n");
            return;
        }

        std::vector<CTxIn> vecTxIn;
        vRecv >> vecTxIn;

        LogPrint("privatesend", "SSSIGNFINALTX -- vecTxIn.size() %s\n", vecTxIn.size());

        int nTxInIndex = 0;
        int nTxInsCount = (int)vecTxIn.size();

        BOOST_FOREACH(const CTxIn txin, vecTxIn) {
            nTxInIndex++;
            if(!AddScriptSig(txin)) {
                LogPrint("privatesend", "SSSIGNFINALTX -- AddScriptSig() failed at %d/%d, session: %d\n", nTxInIndex, nTxInsCount, nSessionID);
                RelayStatus(STATUS_REJECTED);
                return;
            }
            LogPrint("privatesend", "SSSIGNFINALTX -- AddScriptSig() %d/%d success\n", nTxInIndex, nTxInsCount);
        }
        // all is good
        CheckPool();

    } else if(strCommand == NetMsgType::SSFINALTX) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("SSFINALTX -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            return;
        }

        if(fStormNode) {
            // LogPrintf("SSFINALTX -- Can't run on a Stormnode!\n");
            return;
        }

        if(!pSubmittedToStormnode) return;
        if((CNetAddr)pSubmittedToStormnode->addr != (CNetAddr)pfrom->addr) {
            //LogPrintf("SSFINALTX -- message doesn't match current Stormnode: pSubmittedToStormnode %s addr %s\n", pSubmittedToStormnode->addr.ToString(), pfrom->addr.ToString());
            return;
        }

        int nMsgSessionID;
        CTransaction txNew;
        vRecv >> nMsgSessionID >> txNew;

        if(nSessionID != nMsgSessionID) {
            LogPrint("privatesend", "SSFINALTX -- message doesn't match current PrivateSend session: nSessionID: %d  nMsgSessionID: %d\n", nSessionID, nMsgSessionID);
            return;
        }

        LogPrint("privatesend", "SSFINALTX -- txNew %s", txNew.ToString());

        //check to see if input is spent already? (and probably not confirmed)
        SignFinalTransaction(txNew, pfrom);

    } else if(strCommand == NetMsgType::SSCOMPLETE) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("SSCOMPLETE -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            return;
        }

        if(fStormNode) {
            // LogPrintf("SSCOMPLETE -- Can't run on a Stormnode!\n");
            return;
        }

        if(!pSubmittedToStormnode) return;
        if((CNetAddr)pSubmittedToStormnode->addr != (CNetAddr)pfrom->addr) {
            LogPrint("privatesend", "SSCOMPLETE -- message doesn't match current Stormnode: pSubmittedToStormnode=%s  addr=%s\n", pSubmittedToStormnode->addr.ToString(), pfrom->addr.ToString());
            return;
        }

        int nMsgSessionID;
        int nMsgMessageID;
        vRecv >> nMsgSessionID >> nMsgMessageID;

        if(nMsgMessageID < MSG_POOL_MIN || nMsgMessageID > MSG_POOL_MAX) {
            LogPrint("privatesend", "SSCOMPLETE -- nMsgMessageID is out of bounds: %d\n", nMsgMessageID);
            return;
        }

        if(nSessionID != nMsgSessionID) {
            LogPrint("privatesend", "SSCOMPLETE -- message doesn't match current PrivateSend session: nSessionID: %d  nMsgSessionID: %d\n", sandStormPool.nSessionID, nMsgSessionID);
            return;
        }

        LogPrint("privatesend", "SSCOMPLETE -- nMsgSessionID %d  nMsgMessageID %d (%s)\n", nMsgSessionID, nMsgMessageID, GetMessageByID(PoolMessage(nMsgMessageID)));

        CompletedTransaction(PoolMessage(nMsgMessageID));
    }
}

void CSandstormPool::InitDenominations()
{
    vecPrivateSendDenominations.clear();
    /* Denominations

        A note about convertability. Within mixing pools, each denomination
        is convertable to another.

        For example:
        1DSLK+1000 == (.1DSLK+100)*10
        10DSLK+10000 == (1DSLK+1000)*10
    */
    vecPrivateSendDenominations.push_back( (100      * COIN)+100000 );
    vecPrivateSendDenominations.push_back( (10       * COIN)+10000 );
    vecPrivateSendDenominations.push_back( (1        * COIN)+1000 );
    vecPrivateSendDenominations.push_back( (.1       * COIN)+100 );
    /* Disabled till we need them
    vecPrivateSendDenominations.push_back( (.01      * COIN)+10 );
    vecPrivateSendDenominations.push_back( (.001     * COIN)+1 );
    */
}

void CSandstormPool::ResetPool()
{
    nCachedLastSuccessBlock = 0;
    txMyCollateral = CMutableTransaction();
    vecStormnodesUsed.clear();
    UnlockCoins();
    SetNull();
}

void CSandstormPool::SetNull()
{
    // SN side
    vecSessionCollaterals.clear();

    // Client side
    nEntriesCount = 0;
    fLastEntryAccepted = false;
    pSubmittedToStormnode = NULL;

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
void CSandstormPool::UnlockCoins()
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

std::string CSandstormPool::GetStateString() const
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

std::string CSandstormPool::GetStatus()
{
    static int nStatusMessageProgress = 0;
    nStatusMessageProgress += 10;
    std::string strSuffix = "";

    if((pCurrentBlockIndex && pCurrentBlockIndex->nHeight - nCachedLastSuccessBlock < nMinBlockSpacing) || !stormnodeSync.IsBlockchainSynced())
        return strAutoDenomResult;

    switch(nState) {
        case POOL_STATE_IDLE:
            return _("PrivateSend is idle.");
        case POOL_STATE_QUEUE:
            if(     nStatusMessageProgress % 70 <= 30) strSuffix = ".";
            else if(nStatusMessageProgress % 70 <= 50) strSuffix = "..";
            else if(nStatusMessageProgress % 70 <= 70) strSuffix = "...";
            return strprintf(_("Submitted to stormnode, waiting in queue %s"), strSuffix);;
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
                if(     nStatusMessageProgress % 70 <= 40) return strprintf(_("Submitted following entries to stormnode: %u / %d"), nEntriesCount, GetMaxPoolTransactions());
                else if(nStatusMessageProgress % 70 <= 50) strSuffix = ".";
                else if(nStatusMessageProgress % 70 <= 60) strSuffix = "..";
                else if(nStatusMessageProgress % 70 <= 70) strSuffix = "...";
                return strprintf(_("Submitted to stormnode, waiting for more entries ( %u / %d ) %s"), nEntriesCount, GetMaxPoolTransactions(), strSuffix);
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
// Check the mixing progress and send client updates if a Stormnode
//
void CSandstormPool::CheckPool()
{
    if(fStormNode) {
        LogPrint("privatesend", "CSandstormPool::CheckPool -- entries count %lu\n", GetEntriesCount());

        // If entries are full, create finalized transaction
        if(nState == POOL_STATE_ACCEPTING_ENTRIES && GetEntriesCount() >= GetMaxPoolTransactions()) {
            LogPrint("privatesend", "CSandstormPool::CheckPool -- FINALIZE TRANSACTIONS\n");
            CreateFinalTransaction();
            return;
        }

        // If we have all of the signatures, try to compile the transaction
        if(nState == POOL_STATE_SIGNING && IsSignaturesComplete()) {
            LogPrint("privatesend", "CSandstormPool::CheckPool -- SIGNING\n");
            CommitFinalTransaction();
            return;
        }
    }

    // reset if we're here for 10 seconds
    if((nState == POOL_STATE_ERROR || nState == POOL_STATE_SUCCESS) && GetTimeMillis() - nTimeLastSuccessfulStep >= 10000) {
        LogPrint("privatesend", "CSandstormPool::CheckPool -- timeout, RESETTING\n");
        UnlockCoins();
        SetNull();
    }
}

void CSandstormPool::CreateFinalTransaction()
{
    LogPrint("privatesend", "CSandstormPool::CreateFinalTransaction -- FINALIZE TRANSACTIONS\n");

    CMutableTransaction txNew;

    // make our new transaction
    for(int i = 0; i < GetEntriesCount(); i++) {
        BOOST_FOREACH(const CTxSSOut& txssout, vecEntries[i].vecTxSSOut)
            txNew.vout.push_back(txssout);

        BOOST_FOREACH(const CTxSSIn& txssin, vecEntries[i].vecTxSSIn)
            txNew.vin.push_back(txssin);
    }

    // BIP69 https://github.com/kristovatlas/bips/blob/master/bip-0069.mediawiki
    sort(txNew.vin.begin(), txNew.vin.end());
    sort(txNew.vout.begin(), txNew.vout.end());

    finalMutableTransaction = txNew;
    LogPrint("privatesend", "CSandstormPool::CreateFinalTransaction -- finalMutableTransaction=%s", txNew.ToString());

    // request signatures from clients
    RelayFinalTransaction(finalMutableTransaction);
    SetState(POOL_STATE_SIGNING);
}

void CSandstormPool::CommitFinalTransaction()
{
    if(!fStormNode) return; // check and relay final tx only on stormnode

    CTransaction finalTransaction = CTransaction(finalMutableTransaction);
    uint256 hashTx = finalTransaction.GetHash();

    LogPrint("privatesend", "CSandstormPool::CommitFinalTransaction -- finalTransaction=%s", finalTransaction.ToString());

    {
        // See if the transaction is valid
        TRY_LOCK(cs_main, lockMain);
        CValidationState validationState;
        mempool.PrioritiseTransaction(hashTx, hashTx.ToString(), 1000, 0.1*COIN);
        if(!lockMain || !AcceptToMemoryPool(mempool, validationState, finalTransaction, false, NULL, false, true, true))
        {
            LogPrintf("CSandstormPool::CommitFinalTransaction -- AcceptToMemoryPool() error: Transaction not valid\n");
            SetNull();
            // not much we can do in this case, just notify clients
            RelayCompletedTransaction(ERR_INVALID_TX);
            return;
        }
    }

    LogPrintf("CSandstormPool::CommitFinalTransaction -- CREATING SSTX\n");

    // create and sign stormnode sstx transaction
    if(!mapSandstormBroadcastTxes.count(hashTx)) {
        CSandstormBroadcastTx sstx(finalTransaction, activeStormnode.vin, GetAdjustedTime());
        sstx.Sign();
        mapSandstormBroadcastTxes.insert(std::make_pair(hashTx, sstx));
    }

    LogPrintf("CSandstormPool::CommitFinalTransaction -- TRANSMITTING SSTX\n");

    CInv inv(MSG_SSTX, hashTx);
    RelayInv(inv);

    // Tell the clients it was successful
    RelayCompletedTransaction(MSG_SUCCESS);

    // Randomly charge clients
    ChargeRandomFees();

    // Reset
    LogPrint("privatesend", "CSandstormPool::CommitFinalTransaction -- COMPLETED -- RESETTING\n");
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
// How does this work? Messages to Stormnodes come in via NetMsgType::SSVIN, these require a valid collateral
// transaction for the client to be able to enter the pool. This transaction is kept by the Stormnodes
// until the transaction is either complete or fails.
//
void CSandstormPool::ChargeFees()
{
    if(!fStormNode) return;

    //we don't need to charge collateral for every offence.
    if(GetRandInt(100) > 33) return;

    std::vector<CTransaction> vecOffendersCollaterals;

    if(nState == POOL_STATE_ACCEPTING_ENTRIES) {
        BOOST_FOREACH(const CTransaction& txCollateral, vecSessionCollaterals) {
            bool fFound = false;
            BOOST_FOREACH(const CSandStormEntry& entry, vecEntries)
                if(entry.txCollateral == txCollateral)
                    fFound = true;

            // This queue entry didn't send us the promised transaction
            if(!fFound) {
                LogPrintf("CSandstormPool::ChargeFees -- found uncooperative node (didn't send transaction), found offence\n");
                vecOffendersCollaterals.push_back(txCollateral);
            }
        }
    }

    if(nState == POOL_STATE_SIGNING) {
        // who didn't sign?
        BOOST_FOREACH(const CSandStormEntry entry, vecEntries) {
            BOOST_FOREACH(const CTxSSIn txssin, entry.vecTxSSIn) {
                if(!txssin.fHasSig) {
                    LogPrintf("CSandstormPool::ChargeFees -- found uncooperative node (didn't sign), found offence\n");
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
        LogPrintf("CSandstormPool::ChargeFees -- found uncooperative node (didn't %s transaction), charging fees: %s\n",
                (nState == POOL_STATE_SIGNING) ? "sign" : "send", vecOffendersCollaterals[0].ToString());

        LOCK(cs_main);

        CValidationState state;
        bool fMissingInputs;
        if(!AcceptToMemoryPool(mempool, state, vecOffendersCollaterals[0], false, &fMissingInputs, false, true)) {
            // should never really happen
            LogPrintf("CSandstormPool::ChargeFees -- ERROR: AcceptToMemoryPool failed!\n");
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
    allow endless transaction that would bloat DarkSilk and make it unusable. To
    stop these kinds of attacks 1 in 10 successful transactions are charged. This
    adds up to a cost of 0.001DSLK per transaction on average.
*/
void CSandstormPool::ChargeRandomFees()
{
    if(!fStormNode) return;

    LOCK(cs_main);

    BOOST_FOREACH(const CTransaction& txCollateral, vecSessionCollaterals) {

        if(GetRandInt(100) > 10) return;

        LogPrintf("CSandstormPool::ChargeRandomFees -- charging random fees, txCollateral=%s", txCollateral.ToString());

        CValidationState state;
        bool fMissingInputs;
        if(!AcceptToMemoryPool(mempool, state, txCollateral, false, &fMissingInputs, false, true)) {
            // should never really happen
            LogPrintf("CSandstormPool::ChargeRandomFees -- ERROR: AcceptToMemoryPool failed!\n");
        } else {
            RelayTransaction(txCollateral);
        }
    }
}

//
// Check for various timeouts (queue objects, mixing, etc)
//
void CSandstormPool::CheckTimeout()
{
    {
        TRY_LOCK(cs_sandstorm, lockSS);
        if(!lockSS) return; // it's ok to fail here, we run this quite frequently

       int c = 0;
       vector<CSandstormQueue>::iterator it = vecSandstormQueue.begin();
       while(it != vecSandstormQueue.end()){
           if((*it).IsExpired()){
               LogPrint("privatesend", "CSandstormPool::CheckTimeout() : Removing expired queue entry - %d\n", c);
               it = vecSandstormQueue.erase(it);
           } else ++it;
           c++;
       }
    }

    if(!fEnablePrivateSend && !fStormNode) return;

    // catching hanging sessions
    if(!fStormNode) {
        switch(nState) {
            case POOL_STATE_ERROR:
                LogPrint("privatesend", "CSandstormPool::CheckTimeout -- Pool error -- Running CheckPool\n");
                CheckPool();
                break;
            case POOL_STATE_SUCCESS:
                LogPrint("privatesend", "CSandstormPool::CheckTimeout -- Pool success -- Running CheckPool\n");
                CheckPool();
                break;
            default:
                break;
        }
    }

    int nLagTime = fStormNode ? 0 : 10000; // if we're the client, give the server a few extra seconds before resetting.
    int nTimeout = (nState == POOL_STATE_SIGNING) ? PRIVATESEND_SIGNING_TIMEOUT : PRIVATESEND_QUEUE_TIMEOUT;
    bool fTimeout = GetTimeMillis() - nTimeLastSuccessfulStep >= nTimeout*1000 + nLagTime;

    if(nState != POOL_STATE_IDLE && fTimeout) {
        LogPrint("privatesend", "CSandstormPool::CheckTimeout -- %s timed out (%ds) -- restting\n",
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
    After receiving multiple ssa messages, the queue will switch to "accepting entries"
    which is the active state right before merging the transaction
*/
void CSandstormPool::CheckForCompleteQueue()
{
    if(!fEnablePrivateSend && !fStormNode) return;

    if(nState == POOL_STATE_QUEUE && IsSessionReady()) {
        SetState(POOL_STATE_ACCEPTING_ENTRIES);

        CSandstormQueue ssq(nSessionDenom, activeStormnode.vin, GetTime(), true);
        LogPrint("privatesend", "CSandstormPool::CheckForCompleteQueue -- queue is ready, signing and relaying (%s)\n", ssq.ToString());
        ssq.Sign();
        ssq.Relay();
    }
}

// Check to make sure a given input matches an input in the pool and its scriptSig is valid
bool CSandstormPool::IsInputScriptSigValid(const CTxIn& txin)
{
    CMutableTransaction txNew;
    txNew.vin.clear();
    txNew.vout.clear();

    int i = 0;
    int nTxInIndex = -1;
    CScript sigPubKey = CScript();

    BOOST_FOREACH(CSandStormEntry& entry, vecEntries) {

        BOOST_FOREACH(const CTxSSOut& txssout, entry.vecTxSSOut)
            txNew.vout.push_back(txssout);

        BOOST_FOREACH(const CTxSSIn& txssin, entry.vecTxSSIn) {
            txNew.vin.push_back(txssin);

            if(txssin.prevout == txin.prevout) {
                nTxInIndex = i;
                sigPubKey = txssin.prevPubKey;
            }
            i++;
        }
    }

    if(nTxInIndex >= 0) { //might have to do this one input at a time?
        txNew.vin[nTxInIndex].scriptSig = txin.scriptSig;
        LogPrint("privatesend", "CSandstormPool::IsInputScriptSigValid -- verifying scriptSig %s\n", ScriptToAsmStr(txin.scriptSig).substr(0,24));
        if(!VerifyScript(txNew.vin[nTxInIndex].scriptSig, sigPubKey, SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_STRICTENC, MutableTransactionSignatureChecker(&txNew, nTxInIndex))) {
            LogPrint("privatesend", "CSandstormPool::IsInputScriptSigValid -- VerifyScript() failed on input %d\n", nTxInIndex);
            return false;
        }
    } else {
        LogPrint("privatesend", "CSandstormPool::IsInputScriptSigValid -- Failed to find matching input in pool, %s\n", txin.ToString());
        return false;
    }

    LogPrint("privatesend", "CSandstormPool::IsInputScriptSigValid -- Successfully validated input and scriptSig\n");
    return true;
}

// check to make sure the collateral provided by the client is valid
bool CSandstormPool::IsCollateralValid(const CTransaction& txCollateral)
{
    if(txCollateral.vout.empty()) return false;
    if(txCollateral.nLockTime != 0) return false;

    CAmount nValueIn = 0;
    CAmount nValueOut = 0;
    bool fMissingTx = false;

    BOOST_FOREACH(const CTxOut txout, txCollateral.vout) {
        nValueOut += txout.nValue;

        if(!txout.scriptPubKey.IsNormalPaymentScript()) {
            LogPrintf ("CSandstormPool::IsCollateralValid -- Invalid Script, txCollateral=%s", txCollateral.ToString());
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
        LogPrint("privatesend", "CSandstormPool::IsCollateralValid -- Unknown inputs in collateral transaction, txCollateral=%s", txCollateral.ToString());
        return false;
    }

    //collateral transactions are required to pay out PRIVATESEND_COLLATERAL as a fee to the miners
    if(nValueIn - nValueOut < PRIVATESEND_COLLATERAL) {
        LogPrint("privatesend", "CSandstormPool::IsCollateralValid -- did not include enough fees in transaction: fees: %d, txCollateral=%s", nValueOut - nValueIn, txCollateral.ToString());
        return false;
    }

    LogPrint("privatesend", "CSandstormPool::IsCollateralValid -- %s", txCollateral.ToString());

    {
        LOCK(cs_main);
        CValidationState validationState;
        if(!AcceptToMemoryPool(mempool, validationState, txCollateral, false, NULL, false, true, true)) {
            LogPrint("privatesend", "CSandstormPool::IsCollateralValid -- didn't pass AcceptToMemoryPool()\n");
            return false;
        }
    }

    return true;
}


//
// Add a clients transaction to the pool
//
bool CSandstormPool::AddEntry(const CSandStormEntry& entryNew, PoolMessage& nMessageIDRet)
{
    if(!fStormNode) return false;

    BOOST_FOREACH(CTxIn txin, entryNew.vecTxSSIn) {
        if(txin.prevout.IsNull()) {
            LogPrint("privatesend", "CSandstormPool::AddEntry -- input not valid!\n");
            nMessageIDRet = ERR_INVALID_INPUT;
            return false;
        }
    }

    if(!IsCollateralValid(entryNew.txCollateral)) {
        LogPrint("privatesend", "CSandstormPool::AddEntry -- collateral not valid!\n");
        nMessageIDRet = ERR_INVALID_COLLATERAL;
        return false;
    }

    if(GetEntriesCount() >= GetMaxPoolTransactions()) {
        LogPrint("privatesend", "CSandstormPool::AddEntry -- entries is full!\n");
        nMessageIDRet = ERR_ENTRIES_FULL;
        return false;
    }

    BOOST_FOREACH(CTxIn txin, entryNew.vecTxSSIn) {
        LogPrint("privatesend", "looking for txin -- %s\n", txin.ToString());
        BOOST_FOREACH(const CSandStormEntry& entry, vecEntries) {
            BOOST_FOREACH(const CTxSSIn& txssin, entry.vecTxSSIn) {
                if(txssin.prevout == txin.prevout) {
                    LogPrint("privatesend", "CSandstormPool::AddEntry -- found in txin\n");
                    nMessageIDRet = ERR_ALREADY_HAVE;
                    return false;
                }
            }
        }
    }

    vecEntries.push_back(entryNew);

    LogPrint("privatesend", "CSandstormPool::AddEntry -- adding entry\n");
    nMessageIDRet = MSG_ENTRIES_ADDED;
    nTimeLastSuccessfulStep = GetTimeMillis();

    return true;
}

bool CSandstormPool::AddScriptSig(const CTxIn& txinNew)
{
    LogPrint("privatesend", "CSandstormPool::AddScriptSig -- scriptSig=%s\n", ScriptToAsmStr(txinNew.scriptSig).substr(0,24));

    BOOST_FOREACH(const CSandStormEntry& entry, vecEntries) {
        BOOST_FOREACH(const CTxSSIn& txssin, entry.vecTxSSIn) {
            if(txssin.scriptSig == txinNew.scriptSig) {
                LogPrint("privatesend", "CSandstormPool::AddScriptSig -- already exists\n");
                return false;
            }
        }
    }

    if(!IsInputScriptSigValid(txinNew)) {
        LogPrint("privatesend", "CSandstormPool::AddScriptSig -- Invalid scriptSig\n");
        return false;
    }

    LogPrint("privatesend", "CSandstormPool::AddScriptSig -- scriptSig=%s new\n", ScriptToAsmStr(txinNew.scriptSig).substr(0,24));

    BOOST_FOREACH(CTxIn& txin, finalMutableTransaction.vin) {
        if(txinNew.prevout == txin.prevout && txin.nSequence == txinNew.nSequence) {
            txin.scriptSig = txinNew.scriptSig;
            txin.prevPubKey = txinNew.prevPubKey;
            LogPrint("privatesend", "CSandstormPool::AddScriptSig -- adding to finalMutableTransaction, scriptSig=%s\n", ScriptToAsmStr(txinNew.scriptSig).substr(0,24));
        }
    }
    for(int i = 0; i < GetEntriesCount(); i++) {
        if(vecEntries[i].AddScriptSig(txinNew)) {
            LogPrint("privatesend", "CSandstormPool::AddScriptSig -- adding to entries, scriptSig=%s\n", ScriptToAsmStr(txinNew.scriptSig).substr(0,24));
            return true;
        }
    }

    LogPrintf("CSandstormPool::AddScriptSig -- Couldn't set sig!\n" );
    return false;
}

// Check to make sure everything is signed
bool CSandstormPool::IsSignaturesComplete()
{
    BOOST_FOREACH(const CSandStormEntry& entry, vecEntries)
        BOOST_FOREACH(const CTxSSIn& txssin, entry.vecTxSSIn)
            if(!txssin.fHasSig) return false;

    return true;
}

//
// Execute a mixing denomination via a Stormnode.
// This is only ran from clients
//
bool CSandstormPool::SendDenominate(const std::vector<CTxIn>& vecTxIn, const std::vector<CTxOut>& vecTxOut)
{
    if(fStormNode) {
        LogPrintf("CSandstormPool::SendDenominate -- PrivateSend from a Stormnode is not supported currently.\n");
        return false;
    }

    if(txMyCollateral == CMutableTransaction()) {
        LogPrintf("CSandstormPool:SendDenominate -- PrivateSend collateral not set\n");
        return false;
    }

    // lock the funds we're going to use
    BOOST_FOREACH(CTxIn txin, txMyCollateral.vin)
        vecOutPointLocked.push_back(txin.prevout);

    BOOST_FOREACH(CTxIn txin, vecTxIn)
        vecOutPointLocked.push_back(txin.prevout);

    // we should already be connected to a Stormnode
    if(!nSessionID) {
        LogPrintf("CSandstormPool::SendDenominate -- No Stormnode has been selected yet.\n");
        UnlockCoins();
        SetNull();
        return false;
    }

    if(!CheckDiskSpace()) {
        UnlockCoins();
        SetNull();
        fEnablePrivateSend = false;
        LogPrintf("CSandstormPool::SendDenominate -- Not enough disk space, disabling PrivateSend.\n");
        return false;
    }

    SetState(POOL_STATE_ACCEPTING_ENTRIES);
    strLastMessage = "";

    LogPrintf("CSandstormPool::SendDenominate -- Added transaction to pool.\n");

    //check it against the memory pool to make sure it's valid
    {
        CValidationState validationState;
        CMutableTransaction tx;

        BOOST_FOREACH(const CTxIn& txin, vecTxIn) {
            LogPrint("privatesend", "CSandstormPool::SendDenominate -- txin=%s\n", txin.ToString());
            tx.vin.push_back(txin);
        }

        BOOST_FOREACH(const CTxOut& txout, vecTxOut) {
            LogPrint("privatesend", "CSandstormPool::SendDenominate -- txout=%s\n", txout.ToString());
            tx.vout.push_back(txout);
        }

        LogPrintf("CSandstormPool::SendDenominate -- Submitting partial tx %s", tx.ToString());

        mempool.PrioritiseTransaction(tx.GetHash(), tx.GetHash().ToString(), 1000, 0.1*COIN);
        TRY_LOCK(cs_main, lockMain);
        if(!lockMain || !AcceptToMemoryPool(mempool, validationState, CTransaction(tx), false, NULL, false, true, true)) {
            LogPrintf("CSandstormPool::SendDenominate -- AcceptToMemoryPool() failed! tx=%s", tx.ToString());
            UnlockCoins();
            SetNull();
            return false;
        }
    }

    // store our entry for later use
    CSandStormEntry entry(vecTxIn, vecTxOut, txMyCollateral);
    vecEntries.push_back(entry);
    RelayIn(entry);
    nTimeLastSuccessfulStep = GetTimeMillis();

    return true;
}

// Incoming message from Stormnode updating the progress of mixing
bool CSandstormPool::CheckPoolStateUpdate(PoolState nStateNew, int nEntriesCountNew, PoolStatusUpdate nStatusUpdate, PoolMessage nMessageID, int nSessionIDNew)
{
    if(fStormNode) return false;

    // do not update state when mixing client state is one of these
    if(nState == POOL_STATE_IDLE || nState == POOL_STATE_ERROR || nState == POOL_STATE_SUCCESS) return false;

    strAutoDenomResult = _("Stormnode:") + " " + GetMessageByID(nMessageID);

    // if rejected at any state
    if(nStatusUpdate == STATUS_REJECTED) {
        LogPrintf("CSandstormPool::CheckPoolStateUpdate -- entry is rejected by Stormnode\n");
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
            LogPrintf("CSandstormPool::CheckPoolStateUpdate -- set nSessionID to %d\n", nSessionID);
            return true;
        }
        else if(nStateNew == POOL_STATE_ACCEPTING_ENTRIES && nEntriesCount != nEntriesCountNew) {
            nEntriesCount = nEntriesCountNew;
            nTimeLastSuccessfulStep = GetTimeMillis();
            fLastEntryAccepted = true;
            LogPrintf("CSandstormPool::CheckPoolStateUpdate -- new entry accepted!\n");
            return true;
        }
    }

    // only situations above are allowed, fail in any other case
    return false;
}

//
// After we receive the finalized transaction from the Stormnode, we must
// check it to make sure it's what we want, then sign it if we agree.
// If we refuse to sign, it's possible we'll be charged collateral
//
bool CSandstormPool::SignFinalTransaction(const CTransaction& finalTransactionNew, CNode* pnode)
{
    if(fStormNode || pnode == NULL) return false;

    finalMutableTransaction = finalTransactionNew;
    LogPrintf("CSandstormPool::SignFinalTransaction -- finalMutableTransaction=%s", finalMutableTransaction.ToString());

    std::vector<CTxIn> sigs;

    //make sure my inputs/outputs are present, otherwise refuse to sign
    BOOST_FOREACH(const CSandStormEntry entry, vecEntries) {
        BOOST_FOREACH(const CTxSSIn txssin, entry.vecTxSSIn) {
            /* Sign my transaction and all outputs */
            int nMyInputIndex = -1;
            CScript prevPubKey = CScript();
            CTxIn txin = CTxIn();

            for(unsigned int i = 0; i < finalMutableTransaction.vin.size(); i++) {
                if(finalMutableTransaction.vin[i] == txssin) {
                    nMyInputIndex = i;
                    prevPubKey = txssin.prevPubKey;
                    txin = txssin;
                }
            }

            if(nMyInputIndex >= 0) { //might have to do this one input at a time?
                int nFoundOutputsCount = 0;
                CAmount nValue1 = 0;
                CAmount nValue2 = 0;

                for(unsigned int i = 0; i < finalMutableTransaction.vout.size(); i++) {
                    BOOST_FOREACH(const CTxOut& txout, entry.vecTxSSOut) {
                        if(finalMutableTransaction.vout[i] == txout) {
                            nFoundOutputsCount++;
                            nValue1 += finalMutableTransaction.vout[i].nValue;
                        }
                    }
                }

                BOOST_FOREACH(const CTxOut txout, entry.vecTxSSOut)
                    nValue2 += txout.nValue;

                int nTargetOuputsCount = entry.vecTxSSOut.size();
                if(nFoundOutputsCount < nTargetOuputsCount || nValue1 != nValue2) {
                    // in this case, something went wrong and we'll refuse to sign. It's possible we'll be charged collateral. But that's
                    // better then signing if the transaction doesn't look like what we wanted.
                    LogPrintf("CSandstormPool::SignFinalTransaction -- My entries are not correct! Refusing to sign: nFoundOutputsCount: %d, nTargetOuputsCount: %d\n", nFoundOutputsCount, nTargetOuputsCount);
                    UnlockCoins();
                    SetNull();

                    return false;
                }

                const CKeyStore& keystore = *pwalletMain;

                LogPrint("privatesend", "CSandstormPool::SignFinalTransaction -- Signing my input %i\n", nMyInputIndex);
                if(!SignSignature(keystore, prevPubKey, finalMutableTransaction, nMyInputIndex, int(SIGHASH_ALL|SIGHASH_ANYONECANPAY))) { // changes scriptSig
                    LogPrint("privatesend", "CSandstormPool::SignFinalTransaction -- Unable to sign my own transaction!\n");
                    // not sure what to do here, it will timeout...?
                }

                sigs.push_back(finalMutableTransaction.vin[nMyInputIndex]);
                LogPrint("privatesend", "CSandstormPool::SignFinalTransaction -- nMyInputIndex: %d, sigs.size(): %d, scriptSig=%s\n", nMyInputIndex, (int)sigs.size(), ScriptToAsmStr(finalMutableTransaction.vin[nMyInputIndex].scriptSig));
            }
        }
    }

    if(sigs.empty()) {
        LogPrintf("CSandstormPool::SignFinalTransaction -- can't sign anything!\n");
        UnlockCoins();
        SetNull();

        return false;
    }

    // push all of our signatures to the Stormnode
    LogPrintf("CSandstormPool::SignFinalTransaction -- pushing sigs to the stormnode, finalMutableTransaction=%s", finalMutableTransaction.ToString());
    pnode->PushMessage(NetMsgType::SSSIGNFINALTX, sigs);
    SetState(POOL_STATE_SIGNING);
    nTimeLastSuccessfulStep = GetTimeMillis();

    return true;
}

void CSandstormPool::NewBlock()
{
    static int64_t nTimeNewBlockReceived = 0;

    //we we're processing lots of blocks, we'll just leave
    if(GetTime() - nTimeNewBlockReceived < 10) return;
    nTimeNewBlockReceived = GetTime();
    LogPrint("privatesend", "CSandstormPool::NewBlock\n");

    CheckTimeout();
}

// mixing transaction was completed (failed or successful)
void CSandstormPool::CompletedTransaction(PoolMessage nMessageID)
{
    if(fStormNode) return;

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
bool CSandstormPool::DoAutomaticDenominating(bool fDryRun)
{
    if(!fEnablePrivateSend || fStormNode || !pCurrentBlockIndex) return false;
    if(!pwalletMain || pwalletMain->IsLocked(true)) return false;
    if(nState != POOL_STATE_IDLE) return false;

    if(!stormnodeSync.IsStormnodeListSynced()) {
        strAutoDenomResult = _("Can't mix while sync in progress.");
        return false;
    }

    switch(nWalletBackups) {
        case 0:
            LogPrint("privatesend", "CSandstormPool::DoAutomaticDenominating -- Automatic backups disabled, no mixing available.\n");
            strAutoDenomResult = _("Automatic backups disabled") + ", " + _("no mixing available.");
            fEnablePrivateSend = false; // stop mixing
            pwalletMain->nKeysLeftSinceAutoBackup = 0; // no backup, no "keys since last backup"
            return false;
        case -1:
            // Automatic backup failed, nothing else we can do until user fixes the issue manually.
            // There is no way to bring user attention in daemon mode so we just update status and
            // keep spaming if debug is on.
            LogPrint("privatesend", "CSandstormPool::DoAutomaticDenominating -- ERROR! Failed to create automatic backup.\n");
            strAutoDenomResult = _("ERROR! Failed to create automatic backup") + ", " + _("see debug.log for details.");
            return false;
        case -2:
            // We were able to create automatic backup but keypool was not replenished because wallet is locked.
            // There is no way to bring user attention in daemon mode so we just update status and
            // keep spaming if debug is on.
            LogPrint("privatesend", "CSandstormPool::DoAutomaticDenominating -- WARNING! Failed to create replenish keypool, please unlock your wallet to do so.\n");
            strAutoDenomResult = _("WARNING! Failed to replenish keypool, please unlock your wallet to do so.") + ", " + _("see debug.log for details.");
            return false;
    }

    if(pwalletMain->nKeysLeftSinceAutoBackup < PRIVATESEND_KEYS_THRESHOLD_STOP) {
        // We should never get here via mixing itself but probably smth else is still actively using keypool
        LogPrint("privatesend", "CSandstormPool::DoAutomaticDenominating -- Very low number of keys left: %d, no mixing available.\n", pwalletMain->nKeysLeftSinceAutoBackup);
        strAutoDenomResult = strprintf(_("Very low number of keys left: %d") + ", " + _("no mixing available."), pwalletMain->nKeysLeftSinceAutoBackup);
        // It's getting really dangerous, stop mixing
        fEnablePrivateSend = false;
        return false;
    } else if(pwalletMain->nKeysLeftSinceAutoBackup < PRIVATESEND_KEYS_THRESHOLD_WARNING) {
        // Low number of keys left but it's still more or less safe to continue
        LogPrint("privatesend", "CSandstormPool::DoAutomaticDenominating -- Very low number of keys left: %d\n", pwalletMain->nKeysLeftSinceAutoBackup);
        strAutoDenomResult = strprintf(_("Very low number of keys left: %d"), pwalletMain->nKeysLeftSinceAutoBackup);

        if(fCreateAutoBackups) {
            LogPrint("privatesend", "CSandstormPool::DoAutomaticDenominating -- Trying to create new backup.\n");
            std::string warningString;
            std::string errorString;

            if(!AutoBackupWallet(pwalletMain, "", warningString, errorString)) {
                if(!warningString.empty()) {
                    // There were some issues saving backup but yet more or less safe to continue
                    LogPrintf("CSandstormPool::DoAutomaticDenominating -- WARNING! Something went wrong on automatic backup: %s\n", warningString);
                }
                if(!errorString.empty()) {
                    // Things are really broken
                    LogPrintf("CSandstormPool::DoAutomaticDenominating -- ERROR! Failed to create automatic backup: %s\n", errorString);
                    strAutoDenomResult = strprintf(_("ERROR! Failed to create automatic backup") + ": %s", errorString);
                    return false;
                }
            }
        } else {
            // Wait for someone else (e.g. GUI action) to create automatic backup for us
            return false;
        }
    }

    LogPrint("privatesend", "CSandstormPool::DoAutomaticDenominating -- Keys left since latest backup: %d\n", pwalletMain->nKeysLeftSinceAutoBackup);

    if(GetEntriesCount() > 0) {
        strAutoDenomResult = _("Mixing in progress...");
        return false;
    }

    TRY_LOCK(cs_sandstorm, lockSS);
    if(!lockSS) {
        strAutoDenomResult = _("Lock is already in place.");
        return false;
    }

    if(!fDryRun && pwalletMain->IsLocked(true)) {
        strAutoDenomResult = _("Wallet is locked.");
        return false;
    }

    if(!fPrivateSendMultiSession && pCurrentBlockIndex->nHeight - nCachedLastSuccessBlock < nMinBlockSpacing) {
        LogPrintf("CSandstormPool::DoAutomaticDenominating -- Last successful PrivateSend action was too recent\n");
        strAutoDenomResult = _("Last successful PrivateSend action was too recent.");
        return false;
    }

    if(snodeman.size() == 0) {
        LogPrint("privatesend", "CSandstormPool::DoAutomaticDenominating -- No Stormnodes detected\n");
        strAutoDenomResult = _("No Stormnodes detected.");
        return false;
    }

    // ** find the coins we'll use
    std::vector<CTxIn> vecTxIn;
    CAmount nValueMin = CENT;
    CAmount nValueIn = 0;

    CAmount nOnlyDenominatedBalance;
    CAmount nBalanceNeedsDenominated;

    CAmount nLowestDenom = vecPrivateSendDenominations.back();
    // if there are no confirmed DS collateral inputs yet
    if(!pwalletMain->HasCollateralInputs()) {
        // should have some additional amount for them
        nLowestDenom += PRIVATESEND_COLLATERAL*4;
    }

    CAmount nBalanceNeedsAnonymized = pwalletMain->GetNeedsToBeAnonymizedBalance(nLowestDenom);

    // anonymizable balance is way too small
    if(nBalanceNeedsAnonymized < nLowestDenom) {
        LogPrintf("CSandstormPool::DoAutomaticDenominating -- Not enough funds to anonymize\n");
        strAutoDenomResult = _("Not enough funds to anonymize.");
        return false;
    }

    LogPrint("privatesend", "CSandstormPool::DoAutomaticDenominating -- nLowestDenom: %f, nBalanceNeedsAnonymized: %f\n", (float)nLowestDenom/COIN, (float)nBalanceNeedsAnonymized/COIN);

    // select coins that should be given to the pool
    if(!pwalletMain->SelectCoinsDark(nValueMin, nBalanceNeedsAnonymized, vecTxIn, nValueIn, 0, nPrivateSendRounds))
    {
        if(pwalletMain->SelectCoinsDark(nValueMin, 9999999*COIN, vecTxIn, nValueIn, -2, 0))
        {
            nOnlyDenominatedBalance = pwalletMain->GetDenominatedBalance(true) + pwalletMain->GetDenominatedBalance() - pwalletMain->GetAnonymizedBalance();
            nBalanceNeedsDenominated = nBalanceNeedsAnonymized - nOnlyDenominatedBalance;

            if(nBalanceNeedsDenominated > nValueIn) nBalanceNeedsDenominated = nValueIn;

            LogPrint("privatesend", "CSandstormPool::DoAutomaticDenominating -- `SelectCoinsDark` (%f - (%f + %f - %f = %f) ) = %f\n",
                            (float)nBalanceNeedsAnonymized/COIN,
                            (float)pwalletMain->GetDenominatedBalance(true)/COIN,
                            (float)pwalletMain->GetDenominatedBalance()/COIN,
                            (float)pwalletMain->GetAnonymizedBalance()/COIN,
                            (float)nOnlyDenominatedBalance/COIN,
                            (float)nBalanceNeedsDenominated/COIN);

            if(nBalanceNeedsDenominated < nLowestDenom) { // most likely we are just waiting for denoms to confirm
                LogPrintf("CSandstormPool::DoAutomaticDenominating -- No funds detected in need of denominating\n");
                strAutoDenomResult = _("No funds detected in need of denominating.");
                return false;
            }
            if(!fDryRun) return CreateDenominated();

            return true;
        } else {
            LogPrintf("CSandstormPool::DoAutomaticDenominating -- Can't denominate (no compatible inputs left)\n");
            strAutoDenomResult = _("Can't denominate: no compatible inputs left.");
            return false;
        }
    }

    if(fDryRun) return true;

    nOnlyDenominatedBalance = pwalletMain->GetDenominatedBalance(true) + pwalletMain->GetDenominatedBalance() - pwalletMain->GetAnonymizedBalance();
    nBalanceNeedsDenominated = nBalanceNeedsAnonymized - nOnlyDenominatedBalance;
    LogPrint("privatesend", "CSandstormPool::DoAutomaticDenominating -- 'nBalanceNeedsDenominated > 0' (%f - (%f + %f - %f = %f) ) = %f\n",
                    (float)nBalanceNeedsAnonymized/COIN,
                    (float)pwalletMain->GetDenominatedBalance(true)/COIN,
                    (float)pwalletMain->GetDenominatedBalance()/COIN,
                    (float)pwalletMain->GetAnonymizedBalance()/COIN,
                    (float)nOnlyDenominatedBalance/COIN,
                    (float)nBalanceNeedsDenominated/COIN);

    //check if we have should create more denominated inputs
    if(nBalanceNeedsDenominated > 0) return CreateDenominated();

    //check if we have the collateral sized inputs
    if(!pwalletMain->HasCollateralInputs())
        return !pwalletMain->HasCollateralInputs(false) && MakeCollateralAmounts();

    if(nSessionID) {
        strAutoDenomResult = _("Mixing in progress...");
        return false;
    }

    // Initial phase, find a Stormnode
    // Clean if there is anything left from previous session
    UnlockCoins();
    SetNull();

    if(!fPrivateSendMultiSession && pwalletMain->GetDenominatedBalance(true) > 0) { //get denominated unconfirmed inputs
        LogPrintf("CSandstormPool::DoAutomaticDenominating -- Found unconfirmed denominated outputs, will wait till they confirm to continue.\n");
        strAutoDenomResult = _("Found unconfirmed denominated outputs, will wait till they confirm to continue.");
        return false;
    }

    //check our collateral and create new if needed
    std::string strReason;
    if(txMyCollateral == CMutableTransaction()) {
        if(!pwalletMain->CreateCollateralTransaction(txMyCollateral, strReason)) {
            LogPrintf("CSandstormPool::DoAutomaticDenominating -- create collateral error:%s\n", strReason);
            return false;
        }
    } else {
        if(!IsCollateralValid(txMyCollateral)) {
            LogPrintf("CSandstormPool::DoAutomaticDenominating -- invalid collateral, recreating...\n");
            if(!pwalletMain->CreateCollateralTransaction(txMyCollateral, strReason)) {
                LogPrintf("CSandstormPool::DoAutomaticDenominating -- create collateral error: %s\n", strReason);
                return false;
            }
        }
    }

    int nSnCountEnabled = snodeman.CountEnabled(MIN_PRIVATESEND_PEER_PROTO_VERSION);

    // If we've used 90% of the Stormnode list then drop the oldest first ~30%
    int nThreshold_high = nSnCountEnabled * 0.9;
    int nThreshold_low = nThreshold_high * 0.7;
    LogPrint("privatesend", "Checking vecStormnodesUsed: size: %d, threshold: %d\n", (int)vecStormnodesUsed.size(), nThreshold_high);

    if((int)vecStormnodesUsed.size() > nThreshold_high) {
        vecStormnodesUsed.erase(vecStormnodesUsed.begin(), vecStormnodesUsed.begin() + vecStormnodesUsed.size() - nThreshold_low);
        LogPrint("privatesend", "  vecStormnodesUsed: new size: %d, threshold: %d\n", (int)vecStormnodesUsed.size(), nThreshold_high);
    }

    bool fUseQueue = GetRandInt(100) > 33;
    // don't use the queues all of the time for mixing unless we are a liquidity provider
    if(nLiquidityProvider || fUseQueue) {

        // Look through the queues and see if anything matches
        BOOST_FOREACH(CSandstormQueue& ssq, vecSandstormQueue) {
            // only try each queue once
            if(ssq.fTried) continue;
            ssq.fTried = true;

            if(ssq.IsExpired()) continue;

            CStormnode* psn = snodeman.Find(ssq.vin);
            if(psn == NULL) {
                LogPrintf("CSandstormPool::DoAutomaticDenominating -- ssq stormnode is not in stormnode list, stormnode=%s\n", ssq.vin.prevout.ToStringShort());
                continue;
            }

            if(psn->nProtocolVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) continue;

            // incompatible denom
            if(ssq.nDenom >= (1 << vecPrivateSendDenominations.size())) continue;

            // mixing rate limit i.e. nLastSsq check should already pass in SSQUEUE ProcessMessage
            // in order for ssq to get into vecSandstormQueue, so we should be safe to mix already,
            // no need for additional verification here

            LogPrint("privatesend", "CSandstormPool::DoAutomaticDenominating -- found valid queue: %s\n", ssq.ToString());

            std::vector<CTxIn> vecTxInTmp;
            std::vector<COutput> vCoinsTmp;
            // Try to match their denominations if possible
            if(!pwalletMain->SelectCoinsByDenominations(ssq.nDenom, nValueMin, nBalanceNeedsAnonymized, vecTxInTmp, vCoinsTmp, nValueIn, 0, nPrivateSendRounds)) {
                LogPrintf("CSandstormPool::DoAutomaticDenominating -- Couldn't match denominations %d (%s)\n", ssq.nDenom, GetDenominationsToString(ssq.nDenom));
                continue;
            }

            vecStormnodesUsed.push_back(ssq.vin);

            LogPrintf("CSandstormPool::DoAutomaticDenominating -- attempt to connect to stormnode from queue, addr=%s\n", psn->addr.ToString());
            // connect to Stormnode and submit the queue request
            CNode* pnode = ConnectNode((CAddress)psn->addr, NULL, true);
            if(pnode) {
                pSubmittedToStormnode = psn;
                nSessionDenom = ssq.nDenom;

                pnode->PushMessage(NetMsgType::SSACCEPT, nSessionDenom, txMyCollateral);
                LogPrintf("CSandstormPool::DoAutomaticDenominating -- connected (from queue), sending SSACCEPT: nSessionDenom: %d (%s), addr=%s\n",
                        nSessionDenom, GetDenominationsToString(nSessionDenom), pnode->addr.ToString());
                strAutoDenomResult = _("Mixing in progress...");
                SetState(POOL_STATE_QUEUE);
                nTimeLastSuccessfulStep = GetTimeMillis();
                return true;
            } else {
                LogPrintf("CSandstormPool::DoAutomaticDenominating -- can't connect, addr=%s\n", psn->addr.ToString());
                strAutoDenomResult = _("Error connecting to Stormnode.");
                continue;
            }
        }
    }

    // do not initiate queue if we are a liquidity provider to avoid useless inter-mixing
    if(nLiquidityProvider) return false;

    int nTries = 0;

    // otherwise, try one randomly
    while(nTries < 10) {
        CStormnode* psn = snodeman.FindRandomNotInVec(vecStormnodesUsed, MIN_PRIVATESEND_PEER_PROTO_VERSION);
        if(psn == NULL) {
            LogPrintf("CSandstormPool::DoAutomaticDenominating -- Can't find random stormnode!\n");
            strAutoDenomResult = _("Can't find random Stormnode.");
            return false;
        }
        vecStormnodesUsed.push_back(psn->vin);

        if(psn->nLastSsq != 0 && psn->nLastSsq + nSnCountEnabled/5 > snodeman.nSsqCount) {
            LogPrintf("CSandstormPool::DoAutomaticDenominating -- Too early to mix on this stormnode!"
                        " stormnode=%s  addr=%s  nLastSsq=%d  CountEnabled/5=%d  nSsqCount=%d\n",
                        psn->vin.prevout.ToStringShort(), psn->addr.ToString(), psn->nLastSsq,
                        nSnCountEnabled/5, snodeman.nSsqCount);
            nTries++;
            continue;
        }

        LogPrintf("CSandstormPool::DoAutomaticDenominating -- attempt %d connection to Stormnode %s\n", nTries, psn->addr.ToString());
        CNode* pnode = ConnectNode((CAddress)psn->addr, NULL, true);
        if(pnode) {
            LogPrintf("CSandstormPool::DoAutomaticDenominating -- connected, addr=%s\n", psn->addr.ToString());
            pSubmittedToStormnode = psn;

            std::vector<CAmount> vecAmounts;
            pwalletMain->ConvertList(vecTxIn, vecAmounts);
            // try to get a single random denom out of vecAmounts
            while(nSessionDenom == 0) {
                nSessionDenom = GetDenominationsByAmounts(vecAmounts);
            }

            pnode->PushMessage(NetMsgType::SSACCEPT, nSessionDenom, txMyCollateral);
            LogPrintf("CSandstormPool::DoAutomaticDenominating -- connected, sending SSACCEPT, nSessionDenom: %d (%s)\n",
                    nSessionDenom, GetDenominationsToString(nSessionDenom));
            strAutoDenomResult = _("Mixing in progress...");
            SetState(POOL_STATE_QUEUE);
            nTimeLastSuccessfulStep = GetTimeMillis();
            return true;
        } else {
            LogPrintf("CSandstormPool::DoAutomaticDenominating -- can't connect, addr=%s\n", psn->addr.ToString());
            nTries++;
            continue;
        }
    }

    strAutoDenomResult = _("No compatible Stormnode found.");
    return false;
}

bool CSandstormPool::SubmitDenominate()
{
    std::string strError;
    std::vector<CTxIn> vecTxInRet;
    std::vector<CTxOut> vecTxOutRet;

    // Submit transaction to the pool if we get here
    // Try to use only inputs with the same number of rounds starting from lowest number of rounds possible
    for(int i = 0; i < nPrivateSendRounds; i++) {
        if(PrepareDenominate(i, i+1, strError, vecTxInRet, vecTxOutRet)) {
            LogPrintf("CSandstormPool::SubmitDenominate -- Running PrivateSend denominate for %d rounds, success\n", i);
            return SendDenominate(vecTxInRet, vecTxOutRet);
        }
        LogPrintf("CSandstormPool::SubmitDenominate -- Running PrivateSend denominate for %d rounds, error: %s\n", i, strError);
    }

    // We failed? That's strange but let's just make final attempt and try to mix everything
    if(PrepareDenominate(0, nPrivateSendRounds, strError, vecTxInRet, vecTxOutRet)) {
        LogPrintf("CSandstormPool::SubmitDenominate -- Running PrivateSend denominate for all rounds, success\n");
        return SendDenominate(vecTxInRet, vecTxOutRet);
    }

    // Should never actually get here but just in case
    LogPrintf("CSandstormPool::SubmitDenominate -- Running PrivateSend denominate for all rounds, error: %s\n", strError);
    strAutoDenomResult = strError;
    return false;
}

bool CSandstormPool::PrepareDenominate(int nMinRounds, int nMaxRounds, std::string& strErrorRet, std::vector<CTxIn>& vecTxInRet, std::vector<CTxOut>& vecTxOutRet)
{
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
    bool fSelected = pwalletMain->SelectCoinsByDenominations(nSessionDenom, vecPrivateSendDenominations.back(), PRIVATESEND_POOL_MAX, vecTxIn, vCoins, nValueIn, nMinRounds, nMaxRounds);
    if (nMinRounds >= 0 && !fSelected) {
        strErrorRet = "Can't select current denominated inputs";
        return false;
    }

    LogPrintf("CSandstormPool::PrepareDenominate -- max value: %f\n", (double)nValueIn/COIN);

    {
        LOCK(pwalletMain->cs_wallet);
        BOOST_FOREACH(CTxIn txin, vecTxIn) {
            pwalletMain->LockCoin(txin.prevout);
        }
    }

    CAmount nValueLeft = nValueIn;

    // Try to add every needed denomination, repeat up to 5-9 times.
    // NOTE: No need to randomize order of inputs because they were
    // initially shuffled in CWallet::SelectCoinsByDenominations already.
    int nStep = 0;
    int nStepsMax = 5 + GetRandInt(5);
    std::vector<int> vecBits;
    if (!GetDenominationsBits(nSessionDenom, vecBits)) {
        strErrorRet = "Incorrect session denom";
        return false;
    }

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
bool CSandstormPool::MakeCollateralAmounts()
{
    std::vector<CompactTallyItem> vecTally;
    if(!pwalletMain->SelectCoinsGrouppedByAddresses(vecTally, false)) {
        LogPrint("privatesend", "CSandstormPool::MakeCollateralAmounts -- SelectCoinsGrouppedByAddresses can't find any inputs!\n");
        return false;
    }

    BOOST_FOREACH(CompactTallyItem& item, vecTally) {
        if(!MakeCollateralAmounts(item)) continue;
        return true;
    }

    LogPrintf("CSandstormPool::MakeCollateralAmounts -- failed!\n");
    return false;
}

// Split up large inputs or create fee sized inputs
bool CSandstormPool::MakeCollateralAmounts(const CompactTallyItem& tallyItem)
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

    // try to use non-denominated and not sn-like funds first, select them explicitly
    CCoinControl coinControl;
    coinControl.fAllowOtherInputs = false;
    coinControl.fAllowWatchOnly = false;
    // send change to the same address so that we were able create more denoms out of it later
    coinControl.destChange = tallyItem.address.Get();
    BOOST_FOREACH(const CTxIn& txin, tallyItem.vecTxIn)
        coinControl.Select(txin.prevout);

    bool fSuccess = pwalletMain->CreateTransaction(vecSend, wtx, reservekeyChange,
            nFeeRet, nChangePosRet, strFail, &coinControl, true, ONLY_NONDENOMINATED_NOT1000IFSN);
    if(!fSuccess) {
        // if we failed (most likeky not enough funds), try to use all coins instead -
        // SN-like funds should not be touched in any case and we can't mix denominated without collaterals anyway
        LogPrintf("CSandstormPool::MakeCollateralAmounts -- ONLY_NONDENOMINATED_NOT1000IFSN Error: %s\n", strFail);
        CCoinControl *coinControlNull = NULL;
        fSuccess = pwalletMain->CreateTransaction(vecSend, wtx, reservekeyChange,
                nFeeRet, nChangePosRet, strFail, coinControlNull, true, ONLY_NOT1000IFSN);
        if(!fSuccess) {
            LogPrintf("CSandstormPool::MakeCollateralAmounts -- ONLY_NOT1000IFSN Error: %s\n", strFail);
            reservekeyCollateral.ReturnKey();
            return false;
        }
    }

    reservekeyCollateral.KeepKey();

    LogPrintf("CSandstormPool::MakeCollateralAmounts -- txid=%s\n", wtx.GetHash().GetHex());

    // use the same nCachedLastSuccessBlock as for DS mixinx to prevent race
    if(!pwalletMain->CommitTransaction(wtx, reservekeyChange)) {
        LogPrintf("CSandstormPool::MakeCollateralAmounts -- CommitTransaction failed!\n");
        return false;
    }

    nCachedLastSuccessBlock = pCurrentBlockIndex->nHeight;

    return true;
}

// Create denominations by looping through inputs grouped by addresses
bool CSandstormPool::CreateDenominated()
{
    std::vector<CompactTallyItem> vecTally;
    if(!pwalletMain->SelectCoinsGrouppedByAddresses(vecTally)) {
        LogPrint("privatesend", "CSandstormPool::CreateDenominated -- SelectCoinsGrouppedByAddresses can't find any inputs!\n");
        return false;
    }

    bool fCreateMixingCollaterals = !pwalletMain->HasCollateralInputs();

    BOOST_FOREACH(CompactTallyItem& item, vecTally) {
        if(!CreateDenominated(item, fCreateMixingCollaterals)) continue;
        return true;
    }

    LogPrintf("CSandstormPool::CreateDenominated -- failed!\n");
    return false;
}

// Create denominations
bool CSandstormPool::CreateDenominated(const CompactTallyItem& tallyItem, bool fCreateMixingCollaterals)
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
                    LogPrintf("CSandstormPool::CreateDenominated -- %s\n", strAutoDenomResult);
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
            nFeeRet, nChangePosRet, strFail, &coinControl, true, ONLY_NONDENOMINATED_NOT1000IFSN);
    if(!fSuccess) {
        LogPrintf("CSandstormPool::CreateDenominated -- Error: %s\n", strFail);
        // TODO: return reservekeyDenom here
        reservekeyCollateral.ReturnKey();
        return false;
    }

    // TODO: keep reservekeyDenom here
    reservekeyCollateral.KeepKey();

    if(!pwalletMain->CommitTransaction(wtx, reservekeyChange)) {
        LogPrintf("CSandstormPool::CreateDenominated -- CommitTransaction failed!\n");
        return false;
    }

    // use the same nCachedLastSuccessBlock as for SS mixing to prevent race
    nCachedLastSuccessBlock = pCurrentBlockIndex->nHeight;
    LogPrintf("CSandstormPool::CreateDenominated -- txid=%s\n", wtx.GetHash().GetHex());

    return true;
}

bool CSandstormPool::IsOutputsCompatibleWithSessionDenom(const std::vector<CTxSSOut>& vecTxSSOut)
{
    if(GetDenominations(vecTxSSOut) == 0) return false;

    BOOST_FOREACH(const CSandStormEntry entry, vecEntries) {
        LogPrintf("CSandstormPool::IsOutputsCompatibleWithSessionDenom -- vecTxSSOut denom %d, entry.vecTxSSOut denom %d\n", GetDenominations(vecTxSSOut), GetDenominations(entry.vecTxSSOut));
        if(GetDenominations(vecTxSSOut) != GetDenominations(entry.vecTxSSOut)) return false;
    }

    return true;
}

bool CSandstormPool::IsAcceptableDenomAndCollateral(int nDenom, CTransaction txCollateral, PoolMessage& nMessageIDRet)
{
    if(!fStormNode) return false;

    // is denom even smth legit?
    std::vector<int> vecBits;
    if(!GetDenominationsBits(nDenom, vecBits)) {
        LogPrint("privatesend", "CSandstormPool::IsAcceptableDenomAndCollateral -- denom not valid!\n");
        nMessageIDRet = ERR_DENOM;
        return false;
    }

    // check collateral
    if(!fUnitTest && !IsCollateralValid(txCollateral)) {
        LogPrint("privatesend", "CSandstormPool::IsAcceptableDenomAndCollateral -- collateral not valid!\n");
        nMessageIDRet = ERR_INVALID_COLLATERAL;
        return false;
    }

    return true;
}

bool CSandstormPool::CreateNewSession(int nDenom, CTransaction txCollateral, PoolMessage& nMessageIDRet)
{
    if(!fStormNode || nSessionID != 0) return false;

    // new session can only be started in idle mode
    if(nState != POOL_STATE_IDLE) {
        nMessageIDRet = ERR_MODE;
        LogPrintf("CSandstormPool::CreateNewSession -- incompatible mode: nState=%d\n", nState);
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
        CSandstormQueue ssq(nDenom, activeStormnode.vin, GetTime(), false);
        LogPrint("privatesend", "CSandstormPool::CreateNewSession -- signing and relaying new queue: %s\n", ssq.ToString());
        ssq.Sign();
        ssq.Relay();
        vecSandstormQueue.push_back(ssq);
    }

    vecSessionCollaterals.push_back(txCollateral);
    LogPrintf("CSandstormPool::CreateNewSession -- new session created, nSessionID: %d  nSessionDenom: %d (%s)  vecSessionCollaterals.size(): %d\n",
            nSessionID, nSessionDenom, GetDenominationsToString(nSessionDenom), vecSessionCollaterals.size());

    return true;
}

bool CSandstormPool::AddUserToExistingSession(int nDenom, CTransaction txCollateral, PoolMessage& nMessageIDRet)
{
    if(!fStormNode || nSessionID == 0 || IsSessionReady()) return false;

    if(!IsAcceptableDenomAndCollateral(nDenom, txCollateral, nMessageIDRet)) {
        return false;
    }

    // we only add new users to an existing session when we are in queue mode
    if(nState != POOL_STATE_QUEUE) {
        nMessageIDRet = ERR_MODE;
        LogPrintf("CSandstormPool::AddUserToExistingSession -- incompatible mode: nState=%d\n", nState);
        return false;
    }

    if(nDenom != nSessionDenom) {
        LogPrintf("CSandstormPool::AddUserToExistingSession -- incompatible denom %d (%s) != nSessionDenom %d (%s)\n",
                    nDenom, GetDenominationsToString(nDenom), nSessionDenom, GetDenominationsToString(nSessionDenom));
        nMessageIDRet = ERR_DENOM;
        return false;
    }

    // count new user as accepted to an existing session

    nMessageIDRet = MSG_NOERR;
    nTimeLastSuccessfulStep = GetTimeMillis();
    vecSessionCollaterals.push_back(txCollateral);

    LogPrintf("CSandstormPool::AddUserToExistingSession -- new user accepted, nSessionID: %d  nSessionDenom: %d (%s)  vecSessionCollaterals.size(): %d\n",
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
std::string CSandstormPool::GetDenominationsToString(int nDenom)
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

int CSandstormPool::GetDenominations(const std::vector<CTxSSOut>& vecTxSSOut)
{
    std::vector<CTxOut> vecTxOut;

    BOOST_FOREACH(CTxSSOut out, vecTxSSOut)
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
int CSandstormPool::GetDenominations(const std::vector<CTxOut>& vecTxOut, bool fSingleRandomDenom)
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

bool CSandstormPool::GetDenominationsBits(int nDenom, std::vector<int> &vecBitsRet)
{
    // ( bit on if present, 4 denominations example )
    // bit 0 - 100DSLK+1
    // bit 1 - 10DSLK+1
    // bit 2 - 1DSLK+1
    // bit 3 - .1DSLK+1

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

int CSandstormPool::GetDenominationsByAmounts(const std::vector<CAmount>& vecAmount)
{
    CScript scriptTmp = CScript();
    std::vector<CTxOut> vecTxOut;

    BOOST_REVERSE_FOREACH(CAmount nAmount, vecAmount) {
        CTxOut txout(nAmount, scriptTmp);
        vecTxOut.push_back(txout);
    }

    return GetDenominations(vecTxOut, true);
}

std::string CSandstormPool::GetMessageByID(PoolMessage nMessageID)
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
        case ERR_MAXIMUM:               return _("Value more than PrivateSend pool maximum allows.");
        case ERR_SN_LIST:               return _("Not in the Stormnode list.");
        case ERR_MODE:                  return _("Incompatible mode.");
        case ERR_NON_STANDARD_PUBKEY:   return _("Non-standard public key detected.");
        case ERR_NOT_A_SN:              return _("This is not a Stormnode.");
        case ERR_QUEUE_FULL:            return _("Stormnode queue is full.");
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

bool CSandStormSigner::IsVinAssociatedWithPubkey(const CTxIn& txin, const CPubKey& pubkey)
{
    CScript payee;
    payee = GetScriptForDestination(pubkey.GetID());

    CTransaction tx;
    uint256 hash;
    if(GetTransaction(txin.prevout.hash, tx, Params().GetConsensus(), hash, true)) {
        BOOST_FOREACH(CTxOut out, tx.vout)
            if(out.nValue == 1000*COIN && out.scriptPubKey == payee) return true;
    }

    return false;
}

bool CSandStormSigner::GetKeysFromSecret(std::string strSecret, CKey& keyRet, CPubKey& pubkeyRet)
{
    CDarkSilkSecret vchSecret;

    if(!vchSecret.SetString(strSecret)) return false;

    keyRet = vchSecret.GetKey();
    pubkeyRet = keyRet.GetPubKey();

    return true;
}

bool CSandStormSigner::SignMessage(std::string strMessage, std::vector<unsigned char>& vchSigRet, CKey key)
{
    CHashWriter ss(SER_GETHASH, 0);
    ss << strMessageMagic;
    ss << strMessage;

    return key.SignCompact(ss.GetHash(), vchSigRet);
}

bool CSandStormSigner::VerifyMessage(CPubKey pubkey, const std::vector<unsigned char>& vchSig, std::string strMessage, std::string& strErrorRet)
{
    CHashWriter ss(SER_GETHASH, 0);
    ss << strMessageMagic;
    ss << strMessage;

    CPubKey pubkeyFromSig;
    if(!pubkeyFromSig.RecoverCompact(ss.GetHash(), vchSig)) {
        strErrorRet = "Error recovering public key.";
        return false;
    }

    if(pubkeyFromSig.GetID() != pubkey.GetID()) {
        strErrorRet = strprintf("Keys don't match: pubkey=%s, pubkeyFromSig=%s, strMessage=%s, vchSig=%s",
                    pubkey.GetID().ToString(), pubkeyFromSig.GetID().ToString(), strMessage,
                    EncodeBase64(&vchSig[0], vchSig.size()));
        return false;
    }

    return true;
}

bool CSandStormEntry::AddScriptSig(const CTxIn& txin)
{
    BOOST_FOREACH(CTxSSIn& txssin, vecTxSSIn) {
        if(txssin.prevout == txin.prevout && txssin.nSequence == txin.nSequence) {
            if(txssin.fHasSig) return false;

            txssin.scriptSig = txin.scriptSig;
            txssin.prevPubKey = txin.prevPubKey;
            txssin.fHasSig = true;

            return true;
        }
    }

    return false;
}

bool CSandstormQueue::Sign()
{
    if(!fStormNode) return false;

    std::string strMessage = vin.ToString() + boost::lexical_cast<std::string>(nDenom) + boost::lexical_cast<std::string>(nTime) + boost::lexical_cast<std::string>(fReady);

    if(!sandStormSigner.SignMessage(strMessage, vchSig, activeStormnode.keyStormnode)) {
        LogPrintf("CSandstormQueue::Sign -- SignMessage() failed, %s\n", ToString());
        return false;
    }

    return CheckSignature(activeStormnode.pubKeyStormnode);
}

bool CSandstormQueue::CheckSignature(const CPubKey& pubKeyStormnode)
{
    std::string strMessage = vin.ToString() + boost::lexical_cast<std::string>(nDenom) + boost::lexical_cast<std::string>(nTime) + boost::lexical_cast<std::string>(fReady);
    std::string strError = "";

    if(!sandStormSigner.VerifyMessage(pubKeyStormnode, vchSig, strMessage, strError)) {
        LogPrintf("CSandstormQueue::CheckSignature -- Got bad Stormnode queue signature: %s; error: %s\n", ToString(), strError);
        return false;
    }

    return true;
}

bool CSandstormQueue::Relay()
{
    std::vector<CNode*> vNodesCopy;
    {
        LOCK(cs_vNodes);
        vNodesCopy = vNodes;
        BOOST_FOREACH(CNode* pnode, vNodesCopy)
            pnode->AddRef();
    }
    BOOST_FOREACH(CNode* pnode, vNodesCopy)
        if(pnode->nVersion >= MIN_PRIVATESEND_PEER_PROTO_VERSION)
            pnode->PushMessage(NetMsgType::SSQUEUE, (*this));
    {
        LOCK(cs_vNodes);
        BOOST_FOREACH(CNode* pnode, vNodesCopy)
            pnode->Release();
    }
    return true;
}

bool CSandstormBroadcastTx::Sign()
{
    if(!fStormNode) return false;

    std::string strMessage = tx.GetHash().ToString() + boost::lexical_cast<std::string>(sigTime);

    if(!sandStormSigner.SignMessage(strMessage, vchSig, activeStormnode.keyStormnode)) {
        LogPrintf("CSandstormBroadcastTx::Sign -- SignMessage() failed\n");
        return false;
    }

    return CheckSignature(activeStormnode.pubKeyStormnode);
}

bool CSandstormBroadcastTx::CheckSignature(const CPubKey& pubKeyStormnode)
{
    std::string strMessage = tx.GetHash().ToString() + boost::lexical_cast<std::string>(sigTime);
    std::string strError = "";

    if(!sandStormSigner.VerifyMessage(pubKeyStormnode, vchSig, strMessage, strError)) {
        LogPrintf("CSandstormBroadcastTx::CheckSignature -- Got bad sstx signature, error: %s\n", strError);
        return false;
    }

    return true;
}

void CSandstormPool::RelayFinalTransaction(const CTransaction& txFinal)
{
    LOCK(cs_vNodes);
    BOOST_FOREACH(CNode* pnode, vNodes)
        if(pnode->nVersion >= MIN_PRIVATESEND_PEER_PROTO_VERSION)
            pnode->PushMessage(NetMsgType::SSFINALTX, nSessionID, txFinal);
}

void CSandstormPool::RelayIn(const CSandStormEntry& entry)
{
    if(!pSubmittedToStormnode) return;

    CNode* pnode = FindNode(pSubmittedToStormnode->addr);
    if(pnode != NULL) {
        LogPrintf("CSandstormPool::RelayIn -- found stormnode, relaying message to %s\n", pnode->addr.ToString());
        pnode->PushMessage(NetMsgType::SSVIN, entry);
    }
}

void CSandstormPool::PushStatus(CNode* pnode, PoolStatusUpdate nStatusUpdate, PoolMessage nMessageID)
{
    if(!pnode) return;
    pnode->PushMessage(NetMsgType::SSSTATUSUPDATE, nSessionID, (int)nState, (int)vecEntries.size(), (int)nStatusUpdate, (int)nMessageID);
}

void CSandstormPool::RelayStatus(PoolStatusUpdate nStatusUpdate, PoolMessage nMessageID)
{
    LOCK(cs_vNodes);
    BOOST_FOREACH(CNode* pnode, vNodes)
        if(pnode->nVersion >= MIN_PRIVATESEND_PEER_PROTO_VERSION)
            PushStatus(pnode, nStatusUpdate, nMessageID);
}

void CSandstormPool::RelayCompletedTransaction(PoolMessage nMessageID)
{
    LOCK(cs_vNodes);
    BOOST_FOREACH(CNode* pnode, vNodes)
        if(pnode->nVersion >= MIN_PRIVATESEND_PEER_PROTO_VERSION)
            pnode->PushMessage(NetMsgType::SSCOMPLETE, nSessionID, (int)nMessageID);
}

void CSandstormPool::SetState(PoolState nStateNew)
{
    if(fStormNode && (nStateNew == POOL_STATE_ERROR || nStateNew == POOL_STATE_SUCCESS)) {
        LogPrint("privatesend", "CSandstormPool::SetState -- Can't set state to ERROR or SUCCESS as a Stormnode. \n");
        return;
    }

    LogPrintf("CSandstormPool::SetState -- nState: %d, nStateNew: %d\n", nState, nStateNew);
    nState = nStateNew;
}

void CSandstormPool::UpdatedBlockTip(const CBlockIndex *pindex)
{
    pCurrentBlockIndex = pindex;
    LogPrint("privatesend", "CSandstormPool::UpdatedBlockTip -- pCurrentBlockIndex->nHeight: %d\n", pCurrentBlockIndex->nHeight);

    if(!fLiteMode && stormnodeSync.IsStormnodeListSynced()) {
        NewBlock();
    }
}

//TODO: Rename/move to core
void ThreadCheckSandStormPool()
{
    if(fLiteMode) return; // disable all DarkSilk specific functionality

    static bool fOneThread;
    if(fOneThread) return;
    fOneThread = true;

    // Make this thread recognisable as the PrivateSend thread
    RenameThread("darksilk-privatesend");

    unsigned int nTick = 0;
    unsigned int nDoAutoNextRun = nTick + PRIVATESEND_AUTO_TIMEOUT_MIN;

    while (true)
    {
        MilliSleep(1000);

        // try to sync from all available nodes, one step at a time
        stormnodeSync.ProcessTick();

        if(stormnodeSync.IsBlockchainSynced() && !ShutdownRequested()) {

            nTick++;

            // check if we should activate or ping every few minutes,
            // start right after sync is considered to be done
            if(nTick % STORMNODE_MIN_SNP_SECONDS == 1)
                activeStormnode.ManageState();

            snodeman.Check();

            if(nTick % 60 == 0) {
                snodeman.CheckAndRemove();
                snodeman.ProcessStormnodeConnections();
                snpayments.CheckAndRemove();
                CleanTxLockCandidates();
            }

            sandStormPool.CheckTimeout();
            sandStormPool.CheckForCompleteQueue();

            if(nDoAutoNextRun == nTick) {
                sandStormPool.DoAutomaticDenominating();
                nDoAutoNextRun = nTick + PRIVATESEND_AUTO_TIMEOUT_MIN + GetRandInt(PRIVATESEND_AUTO_TIMEOUT_MAX - PRIVATESEND_AUTO_TIMEOUT_MIN);
            }
        }
    }
}
