// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "privatesend-server.h"

#include "activedynode.h"
#include "consensus/validation.h"
#include "core_io.h"
#include "init.h"
#include "dynode-sync.h"
#include "dynodeman.h"
#include "script/interpreter.h"
#include "txmempool.h"
#include "util.h"
#include "utilmoneystr.h"

CPrivateSendServer privateSendServer;

void CPrivateSendServer::ProcessMessage(CNode* pfrom, std::string& strCommand, CDataStream& vRecv, CConnman& connman)
{
    if(!fDynodeMode) return;
    if(fLiteMode) return; // ignore all Dynamic related functionality
    if(!dynodeSync.IsBlockchainSynced()) return;

    if(strCommand == NetMsgType::PSACCEPT) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("PSACCEPT -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            PushStatus(pfrom, STATUS_REJECTED, ERR_VERSION, connman);
            return;
        }

        if(IsSessionReady()) {
            // too many users in this session already, reject new ones
            LogPrintf("PSACCEPT -- queue is already full!\n");
            PushStatus(pfrom, STATUS_ACCEPTED, ERR_QUEUE_FULL, connman);
            return;
        }

        int nDenom;
        CTransaction txCollateral;
        vRecv >> nDenom >> txCollateral;

        LogPrint("privatesend", "PSACCEPT -- nDenom %d (%s)  txCollateral %s", nDenom, CPrivateSend::GetDenominationsToString(nDenom), txCollateral.ToString());

        dynode_info_t dnInfo;
        if(!dnodeman.GetDynodeInfo(activeDynode.outpoint, dnInfo)) {
            PushStatus(pfrom, STATUS_REJECTED, ERR_DN_LIST, connman);
            return;
        }

        if(vecSessionCollaterals.size() == 0 && dnInfo.nLastPsq != 0 &&
            dnInfo.nLastPsq + dnodeman.CountEnabled(MIN_PRIVATESEND_PEER_PROTO_VERSION)/5 > dnodeman.nPsqCount)
        {
            LogPrintf("PSACCEPT -- last psq too recent, must wait: addr=%s\n", pfrom->addr.ToString());
            PushStatus(pfrom, STATUS_REJECTED, ERR_RECENT, connman);
            return;
        }

        PoolMessage nMessageID = MSG_NOERR;

        bool fResult = nSessionID == 0  ? CreateNewSession(nDenom, txCollateral, nMessageID, connman)
                                        : AddUserToExistingSession(nDenom, txCollateral, nMessageID);
        if(fResult) {
            LogPrintf("PSACCEPT -- is compatible, please submit!\n");
            PushStatus(pfrom, STATUS_ACCEPTED, nMessageID, connman);
            return;
        } else {
            LogPrintf("PSACCEPT -- not compatible with existing transactions!\n");
            PushStatus(pfrom, STATUS_REJECTED, nMessageID, connman);
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

        if(psq.IsExpired()) return;

        dynode_info_t dnInfo;
        if(!dnodeman.GetDynodeInfo(psq.vin.prevout, dnInfo)) return;

        if(!psq.CheckSignature(dnInfo.pubKeyDynode)) {
            // we probably have outdated info
            dnodeman.AskForDN(pfrom, psq.vin.prevout, connman);
            return;
        }

        if(!psq.fReady) {
            BOOST_FOREACH(CPrivatesendQueue q, vecPrivatesendQueue) {
                if(q.vin == psq.vin) {
                    // no way same dn can send another "not yet ready" psq this soon
                    LogPrint("privatesend", "PSQUEUE -- Dynode %s is sending WAY too many psq messages\n", dnInfo.addr.ToString());
                    return;
                }
            }

            int nThreshold = dnInfo.nLastPsq + dnodeman.CountEnabled(MIN_PRIVATESEND_PEER_PROTO_VERSION)/5;
            LogPrint("privatesend", "PSQUEUE -- nLastPsq: %d  threshold: %d  nPsqCount: %d\n", dnInfo.nLastPsq, nThreshold, dnodeman.nPsqCount);
            //don't allow a few nodes to dominate the queuing process
            if(dnInfo.nLastPsq != 0 && nThreshold > dnodeman.nPsqCount) {
                LogPrint("privatesend", "PSQUEUE -- Dynode %s is sending too many psq messages\n", dnInfo.addr.ToString());
                return;
            }
            dnodeman.AllowMixing(psq.vin.prevout);

            LogPrint("privatesend", "PSQUEUE -- new PrivateSend queue (%s) from dynode %s\n", psq.ToString(), dnInfo.addr.ToString());
            vecPrivatesendQueue.push_back(psq);
            psq.Relay(connman);
        }

    } else if(strCommand == NetMsgType::PSVIN) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("PSVIN -- incompatible version! nVersion: %d\n", pfrom->nVersion);
            PushStatus(pfrom, STATUS_REJECTED, ERR_VERSION, connman);
            return;
        }

        //do we have enough users in the current session?
        if(!IsSessionReady()) {
            LogPrintf("PSVIN -- session not complete!\n");
            PushStatus(pfrom, STATUS_REJECTED, ERR_SESSION, connman);
            return;
        }

        CPrivateSendEntry entry;
        vRecv >> entry;

        LogPrint("privatesend", "PSVIN -- txCollateral %s", entry.txCollateral.ToString());

        if(entry.vecTxPSIn.size() > PRIVATESEND_ENTRY_MAX_SIZE) {
            LogPrintf("PSVIN -- ERROR: too many inputs! %d/%d\n", entry.vecTxPSIn.size(), PRIVATESEND_ENTRY_MAX_SIZE);
            PushStatus(pfrom, STATUS_REJECTED, ERR_MAXIMUM, connman);
            return;
        }

        if(entry.vecTxOut.size() > PRIVATESEND_ENTRY_MAX_SIZE) {
            LogPrintf("PSVIN -- ERROR: too many outputs! %d/%d\n", entry.vecTxOut.size(), PRIVATESEND_ENTRY_MAX_SIZE);
            PushStatus(pfrom, STATUS_REJECTED, ERR_MAXIMUM, connman);
            return;
        }

        //do we have the same denominations as the current session?
        if(!IsOutputsCompatibleWithSessionDenom(entry.vecTxOut)) {
            LogPrintf("PSVIN -- not compatible with existing transactions!\n");
            PushStatus(pfrom, STATUS_REJECTED, ERR_EXISTING_TX, connman);
            return;
        }

        //check it like a transaction
        {
            CAmount nValueIn = 0;
            CAmount nValueOut = 0;

            CMutableTransaction tx;

            for (const auto& txout : entry.vecTxOut) {
                nValueOut += txout.nValue;
                tx.vout.push_back(txout);

                if(txout.scriptPubKey.size() != 25) {
                    LogPrintf("PSVIN -- non-standard pubkey detected! scriptPubKey=%s\n", ScriptToAsmStr(txout.scriptPubKey));
                    PushStatus(pfrom, STATUS_REJECTED, ERR_NON_STANDARD_PUBKEY, connman);
                    return;
                }
                if(!txout.scriptPubKey.IsPayToPublicKeyHash()) {
                    LogPrintf("PSVIN -- invalid script! scriptPubKey=%s\n", ScriptToAsmStr(txout.scriptPubKey));
                    PushStatus(pfrom, STATUS_REJECTED, ERR_INVALID_SCRIPT, connman);
                    return;
                }
            }

            BOOST_FOREACH(const CTxIn txin, entry.vecTxPSIn) {
                tx.vin.push_back(txin);

                LogPrint("privatesend", "PSVIN -- txin=%s\n", txin.ToString());

                Coin coin;
                if(GetUTXOCoin(txin.prevout, coin)) {
                    nValueIn += coin.out.nValue;
                } else {
                    LogPrintf("PSVIN -- missing input! tx=%s", tx.ToString());
                    PushStatus(pfrom, STATUS_REJECTED, ERR_MISSING_TX, connman);
                    return;
                }
            }

            // There should be no fee in mixing tx
            CAmount nFee = nValueIn - nValueOut;
            if(nFee != 0) {
                LogPrintf("PSVIN -- there should be no fee in mixing tx! fees: %lld, tx=%s", nFee, tx.ToString());
                PushStatus(pfrom, STATUS_REJECTED, ERR_FEES, connman);
                return;
            }

            {
                LOCK(cs_main);
                CValidationState validationState;
                mempool.PrioritiseTransaction(tx.GetHash(), tx.GetHash().ToString(), 1000, 0.1*COIN);
                if(!AcceptToMemoryPool(mempool, validationState, CTransaction(tx), false, NULL, false, maxTxFee, true)) {
                    LogPrintf("PSVIN -- transaction not valid! tx=%s", tx.ToString());
                    PushStatus(pfrom, STATUS_REJECTED, ERR_INVALID_TX, connman);
                    return;
                }
            }
        }

        PoolMessage nMessageID = MSG_NOERR;

        entry.addr = pfrom->addr;
        if(AddEntry(entry, nMessageID)) {
            PushStatus(pfrom, STATUS_ACCEPTED, nMessageID, connman);
            CheckPool(connman);
            RelayStatus(STATUS_ACCEPTED, connman);
        } else {
            PushStatus(pfrom, STATUS_REJECTED, nMessageID, connman);
            SetNull();
        }

    } else if(strCommand == NetMsgType::PSSIGNFINALTX) {

        if(pfrom->nVersion < MIN_PRIVATESEND_PEER_PROTO_VERSION) {
            LogPrintf("PSSIGNFINALTX -- incompatible version! nVersion: %d\n", pfrom->nVersion);
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
                RelayStatus(STATUS_REJECTED, connman);
                return;
            }
            LogPrint("privatesend", "PSSIGNFINALTX -- AddScriptSig() %d/%d success\n", nTxInIndex, nTxInsCount);
        }
        // all is good
        CheckPool(connman);
    }
}

void CPrivateSendServer::SetNull()
{
    // DN side
    vecSessionCollaterals.clear();

    CPrivateSendBase::SetNull();
}

//
// Check the mixing progress and send client updates if a Dynode
//
void CPrivateSendServer::CheckPool(CConnman& connman)
{
    if(fDynodeMode) {
        LogPrint("privatesend", "CPrivateSendServer::CheckPool -- entries count %lu\n", GetEntriesCount());

        // If entries are full, create finalized transaction
        if(nState == POOL_STATE_ACCEPTING_ENTRIES && GetEntriesCount() >= CPrivateSend::GetMaxPoolTransactions()) {
            LogPrint("privatesend", "CPrivateSendServer::CheckPool -- FINALIZE TRANSACTIONS\n");
            CreateFinalTransaction(connman);
            return;
        }

        // If we have all of the signatures, try to compile the transaction
        if(nState == POOL_STATE_SIGNING && IsSignaturesComplete()) {
            LogPrint("privatesend", "CPrivateSendServer::CheckPool -- SIGNING\n");
            CommitFinalTransaction(connman);
            return;
        }
    }

    // reset if we're here for 10 seconds
    if((nState == POOL_STATE_ERROR || nState == POOL_STATE_SUCCESS) && GetTimeMillis() - nTimeLastSuccessfulStep >= 10000) {
        LogPrint("privatesend", "CPrivateSendServer::CheckPool -- timeout, RESETTING\n");
        SetNull();
    }
}

void CPrivateSendServer::CreateFinalTransaction(CConnman& connman)
{
    LogPrint("privatesend", "CPrivateSendServer::CreateFinalTransaction -- FINALIZE TRANSACTIONS\n");

    CMutableTransaction txNew;

    // make our new transaction
    for(int i = 0; i < GetEntriesCount(); i++) {
        for (const auto& txout : vecEntries[i].vecTxOut)
            txNew.vout.push_back(txout);

        BOOST_FOREACH(const CTxPSIn& txpsin, vecEntries[i].vecTxPSIn)
            txNew.vin.push_back(txpsin);
    }

    sort(txNew.vin.begin(), txNew.vin.end(), CompareInputBIP69());
    sort(txNew.vout.begin(), txNew.vout.end(), CompareOutputBIP69());

    finalMutableTransaction = txNew;
    LogPrint("privatesend", "CPrivateSendServer::CreateFinalTransaction -- finalMutableTransaction=%s", txNew.ToString());

    // request signatures from clients
    RelayFinalTransaction(finalMutableTransaction, connman);
    SetState(POOL_STATE_SIGNING);
}

void CPrivateSendServer::CommitFinalTransaction(CConnman& connman)
{
    if(!fDynodeMode) return; // check and relay final tx only on dynode

    CTransaction finalTransaction = CTransaction(finalMutableTransaction);
    uint256 hashTx = finalTransaction.GetHash();

    LogPrint("privatesend", "CPrivateSendServer::CommitFinalTransaction -- finalTransaction=%s", finalTransaction.ToString());

    {
        // See if the transaction is valid
        TRY_LOCK(cs_main, lockMain);
        CValidationState validationState;
        mempool.PrioritiseTransaction(hashTx, hashTx.ToString(), 1000, 0.1*COIN);
        if(!lockMain || !AcceptToMemoryPool(mempool, validationState, finalTransaction, false, NULL, false, maxTxFee, true))
        {
            LogPrintf("CPrivateSendServer::CommitFinalTransaction -- AcceptToMemoryPool() error: Transaction not valid\n");
            SetNull();
            // not much we can do in this case, just notify clients
            RelayCompletedTransaction(ERR_INVALID_TX, connman);
            return;
        }
    }

    LogPrintf("CPrivateSendServer::CommitFinalTransaction -- CREATING PSTX\n");

    // create and sign dynode pstx transaction
    if(!CPrivateSend::GetPSTX(hashTx)) {
        CPrivatesendBroadcastTx pstxNew(finalTransaction, activeDynode.outpoint, GetAdjustedTime());
        pstxNew.Sign();
        CPrivateSend::AddPSTX(pstxNew);
    }

    LogPrintf("CPrivateSendServer::CommitFinalTransaction -- TRANSMITTING PSTX\n");

    CInv inv(MSG_PSTX, hashTx);
    connman.RelayInv(inv);

    // Tell the clients it was successful
    RelayCompletedTransaction(MSG_SUCCESS, connman);

    // Randomly charge clients
    ChargeRandomFees(connman);

    // Reset
    LogPrint("privatesend", "CPrivateSendServer::CommitFinalTransaction -- COMPLETED -- RESETTING\n");
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
// transaction for the client to be able to enter the pool. This transaction is kept by the Dynode
// until the transaction is either complete or fails.
//
void CPrivateSendServer::ChargeFees(CConnman& connman)
{
    if(!fDynodeMode) return;

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
                LogPrintf("CPrivateSendServer::ChargeFees -- found uncooperative node (didn't send transaction), found offence\n");
                vecOffendersCollaterals.push_back(txCollateral);
            }
        }
    }

    if(nState == POOL_STATE_SIGNING) {
        // who didn't sign?
        BOOST_FOREACH(const CPrivateSendEntry entry, vecEntries) {
            BOOST_FOREACH(const CTxPSIn txpsin, entry.vecTxPSIn) {
                if(!txpsin.fHasSig) {
                    LogPrintf("CPrivateSendServer::ChargeFees -- found uncooperative node (didn't sign), found offence\n");
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
        LogPrintf("CPrivateSendServer::ChargeFees -- found uncooperative node (didn't %s transaction), charging fees: %s\n",
                (nState == POOL_STATE_SIGNING) ? "sign" : "send", vecOffendersCollaterals[0].ToString());

        LOCK(cs_main);

        CValidationState state;
        bool fMissingInputs;
        if(!AcceptToMemoryPool(mempool, state, vecOffendersCollaterals[0], false, &fMissingInputs, false, maxTxFee)) {
            // should never really happen
            LogPrintf("CPrivateSendServer::ChargeFees -- ERROR: AcceptToMemoryPool failed!\n");
        } else {
            connman.RelayTransaction(vecOffendersCollaterals[0]);
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
    adds up to a cost of 0.001DRK per transaction on average.
*/
void CPrivateSendServer::ChargeRandomFees(CConnman& connman)
{
    if(!fDynodeMode) return;

    LOCK(cs_main);

    BOOST_FOREACH(const CTransaction& txCollateral, vecSessionCollaterals) {

        if(GetRandInt(100) > 10) return;

        LogPrintf("CPrivateSendServer::ChargeRandomFees -- charging random fees, txCollateral=%s", txCollateral.ToString());

        CValidationState state;
        bool fMissingInputs;
        if(!AcceptToMemoryPool(mempool, state, txCollateral, false, &fMissingInputs, false, maxTxFee)) {
            // should never really happen
            LogPrintf("CPrivateSendServer::ChargeRandomFees -- ERROR: AcceptToMemoryPool failed!\n");
        } else {
            connman.RelayTransaction(txCollateral);
        }
    }
}

//
// Check for various timeouts (queue objects, mixing, etc)
//
void CPrivateSendServer::CheckTimeout(CConnman& connman)
{
    CheckQueue();

    if(!fDynodeMode) return;

    int nLagTime = fDynodeMode ? 0 : 10000; // if we're the client, give the server a few extra seconds before resetting.
    int nTimeout = (nState == POOL_STATE_SIGNING) ? PRIVATESEND_SIGNING_TIMEOUT : PRIVATESEND_QUEUE_TIMEOUT;
    bool fTimeout = GetTimeMillis() - nTimeLastSuccessfulStep >= nTimeout*1000 + nLagTime;

    if(nState != POOL_STATE_IDLE && fTimeout) {
        LogPrint("privatesend", "CPrivateSendServer::CheckTimeout -- %s timed out (%ds) -- restting\n",
                (nState == POOL_STATE_SIGNING) ? "Signing" : "Session", nTimeout);
        ChargeFees(connman);
        SetNull();
        SetState(POOL_STATE_ERROR);
    }
}

/*
    Check to see if we're ready for submissions from clients
    After receiving multiple psa messages, the queue will switch to "accepting entries"
    which is the active state right before merging the transaction
*/
void CPrivateSendServer::CheckForCompleteQueue(CConnman& connman)
{
    if(!fDynodeMode) return;

    if(nState == POOL_STATE_QUEUE && IsSessionReady()) {
        SetState(POOL_STATE_ACCEPTING_ENTRIES);

        CPrivatesendQueue psq(nSessionDenom, activeDynode.outpoint, GetAdjustedTime(), true);
        LogPrint("privatesend", "CPrivateSendServer::CheckForCompleteQueue -- queue is ready, signing and relaying (%s)\n", psq.ToString());
        psq.Sign();
        psq.Relay(connman);
    }
}

// Check to make sure a given input matches an input in the pool and its scriptSig is valid
bool CPrivateSendServer::IsInputScriptSigValid(const CTxIn& txin)
{
    CMutableTransaction txNew;
    txNew.vin.clear();
    txNew.vout.clear();

    int i = 0;
    int nTxInIndex = -1;
    CScript sigPubKey = CScript();

    BOOST_FOREACH(CPrivateSendEntry& entry, vecEntries) {

        for (const auto& txout : entry.vecTxOut)
            txNew.vout.push_back(txout);

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
        LogPrint("privatesend", "CPrivateSendServer::IsInputScriptSigValid -- verifying scriptSig %s\n", ScriptToAsmStr(txin.scriptSig).substr(0,24));
        if(!VerifyScript(txNew.vin[nTxInIndex].scriptSig, sigPubKey, SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_STRICTENC, MutableTransactionSignatureChecker(&txNew, nTxInIndex))) {
            LogPrint("privatesend", "CPrivateSendServer::IsInputScriptSigValid -- VerifyScript() failed on input %d\n", nTxInIndex);
            return false;
        }
    } else {
        LogPrint("privatesend", "CPrivateSendServer::IsInputScriptSigValid -- Failed to find matching input in pool, %s\n", txin.ToString());
        return false;
    }

    LogPrint("privatesend", "CPrivateSendServer::IsInputScriptSigValid -- Successfully validated input and scriptSig\n");
    return true;
}

//
// Add a clients transaction to the pool
//
bool CPrivateSendServer::AddEntry(const CPrivateSendEntry& entryNew, PoolMessage& nMessageIDRet)
{
    if(!fDynodeMode) return false;

    BOOST_FOREACH(CTxIn txin, entryNew.vecTxPSIn) {
        if(txin.prevout.IsNull()) {
            LogPrint("privatesend", "CPrivateSendServer::AddEntry -- input not valid!\n");
            nMessageIDRet = ERR_INVALID_INPUT;
            return false;
        }
    }

    if(!CPrivateSend::IsCollateralValid(entryNew.txCollateral)) {
        LogPrint("privatesend", "CPrivateSendServer::AddEntry -- collateral not valid!\n");
        nMessageIDRet = ERR_INVALID_COLLATERAL;
        return false;
    }

    if(GetEntriesCount() >= CPrivateSend::GetMaxPoolTransactions()) {
        LogPrint("privatesend", "CPrivateSendServer::AddEntry -- entries is full!\n");
        nMessageIDRet = ERR_ENTRIES_FULL;
        return false;
    }

    BOOST_FOREACH(CTxIn txin, entryNew.vecTxPSIn) {
        LogPrint("privatesend", "looking for txin -- %s\n", txin.ToString());
        BOOST_FOREACH(const CPrivateSendEntry& entry, vecEntries) {
            BOOST_FOREACH(const CTxPSIn& txpsin, entry.vecTxPSIn) {
                if(txpsin.prevout == txin.prevout) {
                    LogPrint("privatesend", "CPrivateSendServer::AddEntry -- found in txin\n");
                    nMessageIDRet = ERR_ALREADY_HAVE;
                    return false;
                }
            }
        }
    }

    vecEntries.push_back(entryNew);

    LogPrint("privatesend", "CPrivateSendServer::AddEntry -- adding entry\n");
    nMessageIDRet = MSG_ENTRIES_ADDED;
    nTimeLastSuccessfulStep = GetTimeMillis();

    return true;
}

bool CPrivateSendServer::AddScriptSig(const CTxIn& txinNew)
{
    LogPrint("privatesend", "CPrivateSendServer::AddScriptSig -- scriptSig=%s\n", ScriptToAsmStr(txinNew.scriptSig).substr(0,24));

    BOOST_FOREACH(const CPrivateSendEntry& entry, vecEntries) {
        BOOST_FOREACH(const CTxPSIn& txpsin, entry.vecTxPSIn) {
            if(txpsin.scriptSig == txinNew.scriptSig) {
                LogPrint("privatesend", "CPrivateSendServer::AddScriptSig -- already exists\n");
                return false;
            }
        }
    }

    if(!IsInputScriptSigValid(txinNew)) {
        LogPrint("privatesend", "CPrivateSendServer::AddScriptSig -- Invalid scriptSig\n");
        return false;
    }

    LogPrint("privatesend", "CPrivateSendServer::AddScriptSig -- scriptSig=%s new\n", ScriptToAsmStr(txinNew.scriptSig).substr(0,24));

    BOOST_FOREACH(CTxIn& txin, finalMutableTransaction.vin) {
        if(txinNew.prevout == txin.prevout && txin.nSequence == txinNew.nSequence) {
            txin.scriptSig = txinNew.scriptSig;
            LogPrint("privatesend", "CPrivateSendServer::AddScriptSig -- adding to finalMutableTransaction, scriptSig=%s\n", ScriptToAsmStr(txinNew.scriptSig).substr(0,24));
        }
    }
    for(int i = 0; i < GetEntriesCount(); i++) {
        if(vecEntries[i].AddScriptSig(txinNew)) {
            LogPrint("privatesend", "CPrivateSendServer::AddScriptSig -- adding to entries, scriptSig=%s\n", ScriptToAsmStr(txinNew.scriptSig).substr(0,24));
            return true;
        }
    }

    LogPrintf("CPrivateSendServer::AddScriptSig -- Couldn't set sig!\n" );
    return false;
}

// Check to make sure everything is signed
bool CPrivateSendServer::IsSignaturesComplete()
{
    BOOST_FOREACH(const CPrivateSendEntry& entry, vecEntries)
        BOOST_FOREACH(const CTxPSIn& txpsin, entry.vecTxPSIn)
            if(!txpsin.fHasSig) return false;

    return true;
}

bool CPrivateSendServer::IsOutputsCompatibleWithSessionDenom(const std::vector<CTxOut>& vecTxOut)
{
    if(CPrivateSend::GetDenominations(vecTxOut) == 0) return false;

    BOOST_FOREACH(const CPrivateSendEntry entry, vecEntries) {
        LogPrintf("CPrivateSendServer::IsOutputsCompatibleWithSessionDenom -- vecTxOut denom %d, entry.vecTxOut denom %d\n",
                CPrivateSend::GetDenominations(vecTxOut), CPrivateSend::GetDenominations(entry.vecTxOut));
        if(CPrivateSend::GetDenominations(vecTxOut) != CPrivateSend::GetDenominations(entry.vecTxOut)) return false;
    }

    return true;
}

bool CPrivateSendServer::IsAcceptableDenomAndCollateral(int nDenom, CTransaction txCollateral, PoolMessage& nMessageIDRet)
{
    if(!fDynodeMode) return false;

    // is denom even smth legit?
    std::vector<int> vecBits;
    if(!CPrivateSend::GetDenominationsBits(nDenom, vecBits)) {
        LogPrint("privatesend", "CPrivateSendServer::IsAcceptableDenomAndCollateral -- denom not valid!\n");
        nMessageIDRet = ERR_DENOM;
        return false;
    }

    // check collateral
    if(!fUnitTest && !CPrivateSend::IsCollateralValid(txCollateral)) {
        LogPrint("privatesend", "CPrivateSendServer::IsAcceptableDenomAndCollateral -- collateral not valid!\n");
        nMessageIDRet = ERR_INVALID_COLLATERAL;
        return false;
    }

    return true;
}

bool CPrivateSendServer::CreateNewSession(int nDenom, CTransaction txCollateral, PoolMessage& nMessageIDRet, CConnman& connman)
{
    if(!fDynodeMode || nSessionID != 0) return false;

    // new session can only be started in idle mode
    if(nState != POOL_STATE_IDLE) {
        nMessageIDRet = ERR_MODE;
        LogPrintf("CPrivateSendServer::CreateNewSession -- incompatible mode: nState=%d\n", nState);
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
        CPrivatesendQueue psq(nDenom, activeDynode.outpoint, GetAdjustedTime(), false);
        LogPrint("privatesend", "CPrivateSendServer::CreateNewSession -- signing and relaying new queue: %s\n", psq.ToString());
        psq.Sign();
        psq.Relay(connman);
        vecPrivatesendQueue.push_back(psq);
    }

    vecSessionCollaterals.push_back(txCollateral);
    LogPrintf("CPrivateSendServer::CreateNewSession -- new session created, nSessionID: %d  nSessionDenom: %d (%s)  vecSessionCollaterals.size(): %d\n",
            nSessionID, nSessionDenom, CPrivateSend::GetDenominationsToString(nSessionDenom), vecSessionCollaterals.size());

    return true;
}

bool CPrivateSendServer::AddUserToExistingSession(int nDenom, CTransaction txCollateral, PoolMessage& nMessageIDRet)
{
    if(!fDynodeMode || nSessionID == 0 || IsSessionReady()) return false;

    if(!IsAcceptableDenomAndCollateral(nDenom, txCollateral, nMessageIDRet)) {
        return false;
    }

    // we only add new users to an existing session when we are in queue mode
    if(nState != POOL_STATE_QUEUE) {
        nMessageIDRet = ERR_MODE;
        LogPrintf("CPrivateSendServer::AddUserToExistingSession -- incompatible mode: nState=%d\n", nState);
        return false;
    }

    if(nDenom != nSessionDenom) {
        LogPrintf("CPrivateSendServer::AddUserToExistingSession -- incompatible denom %d (%s) != nSessionDenom %d (%s)\n",
                    nDenom, CPrivateSend::GetDenominationsToString(nDenom), nSessionDenom, CPrivateSend::GetDenominationsToString(nSessionDenom));
        nMessageIDRet = ERR_DENOM;
        return false;
    }

    // count new user as accepted to an existing session

    nMessageIDRet = MSG_NOERR;
    nTimeLastSuccessfulStep = GetTimeMillis();
    vecSessionCollaterals.push_back(txCollateral);

    LogPrintf("CPrivateSendServer::AddUserToExistingSession -- new user accepted, nSessionID: %d  nSessionDenom: %d (%s)  vecSessionCollaterals.size(): %d\n",
            nSessionID, nSessionDenom, CPrivateSend::GetDenominationsToString(nSessionDenom), vecSessionCollaterals.size());

    return true;
}

void CPrivateSendServer::RelayFinalTransaction(const CTransaction& txFinal, CConnman& connman)
{
    LogPrint("privatesend", "CPrivateSendServer::%s -- nSessionID: %d  nSessionDenom: %d (%s)\n",
            __func__, nSessionID, nSessionDenom, CPrivateSend::GetDenominationsToString(nSessionDenom));

    // final mixing tx with empty signatures should be relayed to mixing participants only
    for (const auto entry : vecEntries) {
        bool fOk = connman.ForNode(entry.addr, [&txFinal, &connman, this](CNode* pnode) {
            connman.PushMessage(pnode, NetMsgType::PSFINALTX, nSessionID, txFinal);
            return true;
        });
        if(!fOk) {
            // no such node? maybe this client disconnected or our own connection went down
            RelayStatus(STATUS_REJECTED, connman);
            break;
        }
    }
}

void CPrivateSendServer::PushStatus(CNode* pnode, PoolStatusUpdate nStatusUpdate, PoolMessage nMessageID, CConnman& connman)
{
    if(!pnode) return;
    connman.PushMessage(pnode, NetMsgType::PSSTATUSUPDATE, nSessionID, (int)nState, (int)vecEntries.size(), (int)nStatusUpdate, (int)nMessageID);
}

void CPrivateSendServer::RelayStatus(PoolStatusUpdate nStatusUpdate, CConnman& connman, PoolMessage nMessageID)
{
    unsigned int nDisconnected{};
    // status updates should be relayed to mixing participants only
    for (const auto entry : vecEntries) {
        // make sure everyone is still connected
        bool fOk = connman.ForNode(entry.addr, [&nStatusUpdate, &nMessageID, &connman, this](CNode* pnode) {
            PushStatus(pnode, nStatusUpdate, nMessageID, connman);
            return true;
        });
        if(!fOk) {
            // no such node? maybe this client disconnected or our own connection went down
            ++nDisconnected;
        }
    }
    if (nDisconnected == 0) return; // all is clear

    // smth went wrong
    LogPrintf("CPrivateSendServer::%s -- can't continue, %llu client(s) disconnected, nSessionID: %d  nSessionDenom: %d (%s)\n",
            __func__, nDisconnected, nSessionID, nSessionDenom, CPrivateSend::GetDenominationsToString(nSessionDenom));

    // notify everyone else that this session should be terminated
    for (const auto entry : vecEntries) {
        connman.ForNode(entry.addr, [&connman, this](CNode* pnode) {
            PushStatus(pnode, STATUS_REJECTED, MSG_NOERR, connman);
            return true;
        });
    }

    if(nDisconnected == vecEntries.size()) {
        // all clients disconnected, there is probably some issues with our own connection
        // do not charge any fees, just reset the pool
        SetNull();
    }
}

void CPrivateSendServer::RelayCompletedTransaction(PoolMessage nMessageID, CConnman& connman)
{
    LogPrint("privatesend", "CPrivateSendServer::%s -- nSessionID: %d  nSessionDenom: %d (%s)\n",
            __func__, nSessionID, nSessionDenom, CPrivateSend::GetDenominationsToString(nSessionDenom));

    // final mixing tx with empty signatures should be relayed to mixing participants only
    for (const auto entry : vecEntries) {
        bool fOk = connman.ForNode(entry.addr, [&nMessageID, &connman, this](CNode* pnode) {
            connman.PushMessage(pnode, NetMsgType::PSCOMPLETE, nSessionID, (int)nMessageID);
            return true;
        });
        if(!fOk) {
            // no such node? maybe client disconnected or our own connection went down
            RelayStatus(STATUS_REJECTED, connman);
            break;
        }
    }
}

void CPrivateSendServer::SetState(PoolState nStateNew)
{
    if(fDynodeMode && (nStateNew == POOL_STATE_ERROR || nStateNew == POOL_STATE_SUCCESS)) {
        LogPrint("privatesend", "CPrivateSendServer::SetState -- Can't set state to ERROR or SUCCESS as a Dynode. \n");
        return;
    }

    LogPrintf("CPrivateSendServer::SetState -- nState: %d, nStateNew: %d\n", nState, nStateNew);
    nState = nStateNew;
}

//TODO: Rename/move to core
void ThreadCheckPrivateSendServer(CConnman& connman)
{
    if(fLiteMode) return; // disable all Dynamic specific functionality

    static bool fOneThread;
    if(fOneThread) return;
    fOneThread = true;

    // Make this thread recognisable as the PrivateSend thread
    RenameThread("dynamic-ps-server");

    unsigned int nTick = 0;

    while (true)
    {
        MilliSleep(1000);

        if(dynodeSync.IsBlockchainSynced() && !ShutdownRequested()) {
            nTick++;
            privateSendServer.CheckTimeout(connman);
            privateSendServer.CheckForCompleteQueue(connman);
        }
    }
}
