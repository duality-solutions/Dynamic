// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "consensus/validation.h"
#include "sync.h"
#include "net.h"
#include "key.h"
#include "util.h"
#include "base58.h"
#include "protocol.h"
#include "instantx.h"
#include "activestormnode.h"
#include "sandstorm.h"
#include "stormnode-sync.h"
#include "stormnodeman.h"
#include "spork.h"

#include <boost/algorithm/string/replace.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>

extern CWallet* pwalletMain;

bool fEnableInstantSend = true;
int nInstantSendDepth = DEFAULT_INSTANTSEND_DEPTH;
int nCompleteTXLocks;

std::map<uint256, CTransaction> mapLockRequestAccepted;
std::map<uint256, CTransaction> mapLockRequestRejected;
std::map<uint256, CTxLockVote> mapTxLockVotes;
std::map<uint256, CTxLockVote> mapTxLockVotesOrphan;
std::map<COutPoint, uint256> mapLockedInputs;

std::map<uint256, CTxLockCandidate> mapTxLockCandidates;
std::map<COutPoint, int64_t> mapStormnodeOrphanVotes; //track stormnodes who voted with no txreq (for DOS protection)

CCriticalSection cs_instantsend;

// Transaction Locks
//
// step 2) Top INSTANTSEND_SIGNATURES_TOTAL stormnodes push "txvote" message
// step 2) Top INSTANTSEND_SIGNATURES_TOTAL stormnodes push "txvote" message
// step 3) Once there are INSTANTSEND_SIGNATURES_REQUIRED valid "txvote" messages
//         for a corresponding "txlreg" message, all inputs from that tx are treated as locked

void ProcessMessageInstantSend(CNode* pfrom, std::string& strCommand, CDataStream& vRecv)
{
    if(fLiteMode) return; // disable all DarkSilk specific functionality
    if(!sporkManager.IsSporkActive(SPORK_2_INSTANTSEND_ENABLED)) return;

    // Ignore any InstantSend messages until stormnode list is synced
    if(!stormnodeSync.IsStormnodeListSynced()) return;

    // NOTE: NetMsgType::TXLOCKREQUEST is handled via ProcessMessage() in main.cpp
    if (strCommand == NetMsgType::TXLOCKVOTE) // InstantSend Transaction Lock Consensus Votes
    {
        CTxLockVote vote;
        vRecv >> vote;

        if(mapTxLockVotes.count(vote.GetHash())) return;
        mapTxLockVotes.insert(std::make_pair(vote.GetHash(), vote));

        ProcessTxLockVote(pfrom, vote);

        return;
    }
}

bool IsInstantSendTxValid(const CTransaction& txCandidate)
{
    if(txCandidate.vout.size() < 1) return false;

    {
        LOCK(cs_main);
        if(!CheckFinalTx(txCandidate)) {
            LogPrint("instantsend", "IsInstantSendTxValid -- Transaction is not final: txCandidate=%s", txCandidate.ToString());
            return false;
        }
    }

    int64_t nValueIn = 0;
    int64_t nValueOut = 0;
    bool fMissingInputs = false;

    BOOST_FOREACH(const CTxOut& txout, txCandidate.vout) {
        nValueOut += txout.nValue;
    }

    BOOST_FOREACH(const CTxIn& txin, txCandidate.vin) {
        CTransaction tx2;
        uint256 hash;
        if(GetTransaction(txin.prevout.hash, tx2, Params().GetConsensus(), hash, true)) {
            if(tx2.vout.size() > txin.prevout.n)
                nValueIn += tx2.vout[txin.prevout.n].nValue;
        } else {
            fMissingInputs = true;
        }
    }

    if(nValueOut > sporkManager.GetSporkValue(SPORK_5_INSTANTSEND_MAX_VALUE)*COIN) {
        LogPrint("instantsend", "IsInstantSendTxValid -- Transaction value too high: nValueOut=%d, txCandidate=%s", nValueOut, txCandidate.ToString());
        return false;
    }

    if(fMissingInputs) {
        LogPrint("instantsend", "IsInstantSendTxValid -- Unknown inputs in transaction: txCandidate=%s", txCandidate.ToString());
        /*
            This happens sometimes for an unknown reason, so we'll return that it's a valid transaction.
            If someone submits an invalid transaction it will be rejected by the network anyway and this isn't
            very common, but we don't want to block IX just because the client can't figure out the fee.
        */
        return true;
    }

    if(nValueIn - nValueOut < INSTANTSEND_MIN_FEE) {
        LogPrint("instantsend", "IsInstantSendTxValid -- did not include enough fees in transaction: fees=%d, txCandidate=%s", nValueOut - nValueIn, txCandidate.ToString());
        return false;
    }

    return true;
}

bool ProcessTxLockRequest(CNode* pfrom, const CTransaction &tx)
{
 if(!IsInstantSendTxValid(tx)) return false;

 BOOST_FOREACH(const CTxOut o, tx.vout) {
     // InstandSend supports normal scripts and unspendable scripts (used in PrivateSend collateral and Governance collateral).
     // TODO: Look into other script types that are normal and can be included
     if(!o.scriptPubKey.IsNormalPaymentScript() && !o.scriptPubKey.IsUnspendable()) {
         LogPrintf("TXLOCKREQUEST -- Invalid Script %s", tx.ToString());
         return false;
     }
 }

 int nBlockHeight = CreateTxLockCandidate(tx);
 if(!nBlockHeight) {
     // smth is not right
     return false;
 }

 uint256 txHash = tx.GetHash();
 mapLockRequestAccepted.insert(std::make_pair(txHash, tx));

 LogPrintf("TXLOCKREQUEST -- Transaction Lock Request: %s %s : accepted %s\n",
         pfrom ? pfrom->addr.ToString() : "", pfrom ? pfrom->cleanSubVer : "", txHash.ToString());

 CreateTxLockVote(tx, nBlockHeight);
 ProcessOrphanTxLockVotes();

 // Stormnodes will sometimes propagate votes before the transaction is known to the client.
 // If this just happened - update transaction status, try forcing external script notification,
 // lock inputs and resolve conflicting locks
 if(IsLockedInstandSendTransaction(txHash)) {
     UpdateLockedTransaction(tx, true);
     LockTransactionInputs(tx);
     ResolveConflicts(tx);
 }

 return true;
}

int64_t CreateTxLockCandidate(const CTransaction& tx)
{
    // Find the age of the first input but all inputs must be old enough too
    int64_t nTxAge = 0;
    BOOST_REVERSE_FOREACH(const CTxIn& txin, tx.vin) {
        nTxAge = GetInputAge(txin);
        if(nTxAge < 9) { //1 less than the "send IX" gui requires, incase of a block propagating the network at the time
            LogPrintf("CreateTxLockCandidate -- Transaction not found / too new: nTxAge=%d, txid=%s\n", nTxAge, tx.GetHash().ToString());
            return 0;
        }
    }

    /*
        Use a blockheight newer than the input.
        This prevents attackers from using transaction mallibility to predict which stormnodes
        they'll use.
    */
    int nCurrentHeight = 0;
    int nLockInputHeight = 0;
    {
        LOCK(cs_main);
        if(!chainActive.Tip()) return 0;
        nCurrentHeight = chainActive.Height();
        nLockInputHeight = nCurrentHeight - nTxAge + 4;
    }

    uint256 txHash = tx.GetHash();

    if(!mapTxLockCandidates.count(txHash)) {
        LogPrintf("CreateTxLockCandidate -- New Transaction Lock Candidate! txid=%s\n", txHash.ToString());

        CTxLockCandidate txLockCandidate;
        txLockCandidate.nBlockHeight = nLockInputHeight;
        //locks expire after nInstantSendKeepLock confirmations
        txLockCandidate.nExpirationBlock = nCurrentHeight + Params().GetConsensus().nInstantSendKeepLock;
        txLockCandidate.nTimeout = GetTime()+(60*5);
        txLockCandidate.txHash = txHash;
        mapTxLockCandidates.insert(std::make_pair(txHash, txLockCandidate));
    } else {
        mapTxLockCandidates[txHash].nBlockHeight = nLockInputHeight;
        LogPrint("instantsend", "CreateTxLockCandidate -- Transaction Lock Candidate exists! txid=%s\n", txHash.ToString());
    }
    return nLockInputHeight;

}

// check if we need to vote on this transaction
void CreateTxLockVote(const CTransaction& tx, int nBlockHeight)
{
    if(!fStormNode) return;

    int n = snodeman.GetStormnodeRank(activeStormnode.vin, nBlockHeight, MIN_INSTANTSEND_PROTO_VERSION);

    if(n == -1) {
        LogPrint("instantsend", "CreateTxLockVote -- Unknown Stormnode %s\n", activeStormnode.vin.prevout.ToStringShort());
        return;
    }

    if(n > INSTANTSEND_SIGNATURES_TOTAL) {
        LogPrint("instantsend", "CreateTxLockVote -- Stormnode not in the top %d (%d)\n", INSTANTSEND_SIGNATURES_TOTAL, n);
        return;
    }
    /*
        nBlockHeight calculated from the transaction is the authoritive source
    */

    LogPrint("instantsend", "CreateTxLockVote -- In the top %d (%d)\n", INSTANTSEND_SIGNATURES_TOTAL, n);

    CTxLockVote vote;
    vote.vinStormnode = activeStormnode.vin;
    vote.txHash = tx.GetHash();
    vote.nBlockHeight = nBlockHeight;
    if(!vote.Sign()) {
        LogPrintf("CreateTxLockVote -- Failed to sign consensus vote\n");
        return;
    }
    if(!vote.CheckSignature()) {
        LogPrintf("CreateTxLockVote -- Signature invalid\n");
        return;
    }

    {
        LOCK(cs_instantsend);
        mapTxLockVotes[vote.GetHash()] = vote;
    }

    CInv inv(MSG_TXLOCK_VOTE, vote.GetHash());
    RelayInv(inv);
}

//received a consensus vote
bool ProcessTxLockVote(CNode* pnode, CTxLockVote& vote)
{
    // Stormnodes will sometimes propagate votes before the transaction is known to the client,
    // will actually process only after the lock request itself has arrived
    if(!mapLockRequestAccepted.count(vote.txHash)) {
        if(!mapTxLockVotesOrphan.count(vote.GetHash())) {
            LogPrint("instantsend", "ProcessTxLockVote -- Orphan vote: txid=%s  stormnode=%s new\n", vote.txHash.ToString(), vote.vinStormnode.prevout.ToStringShort());
            vote.nOrphanExpireTime = GetTime() + 60; // keep orphan votes for 1 minute
            mapTxLockVotesOrphan[vote.GetHash()] = vote;
        } else {
            LogPrint("instantsend", "ProcessTxLockVote -- Orphan vote: txid=%s  stormnode=%s seen\n", vote.txHash.ToString(), vote.vinStormnode.prevout.ToStringShort());
        }

        // This tracks those messages and allows only the same rate as of the rest of the network

        int nStormnodeOrphanExpireTime = GetTime() + 60*10; // keep time data for 10 minutes
        if(!mapStormnodeOrphanVotes.count(vote.vinStormnode.prevout)) {
            mapStormnodeOrphanVotes[vote.vinStormnode.prevout] = nStormnodeOrphanExpireTime;
        } else {
            int64_t nPrevOrphanVote = mapStormnodeOrphanVotes[vote.vinStormnode.prevout];
            if(nPrevOrphanVote > GetTime() && nPrevOrphanVote > GetAverageStormnodeOrphanVoteTime()) {
                LogPrint("instantsend", "ProcessTxLockVote -- stormnode is spamming orphan Transaction Lock Votes: txid=%s  stormnode=%s\n",
                        vote.vinStormnode.prevout.ToStringShort(), vote.txHash.ToString());
                // Misbehaving(pfrom->id, 1);
                return false;
            }
            // not spamming, refresh
            mapStormnodeOrphanVotes[vote.vinStormnode.prevout] = nStormnodeOrphanExpireTime;
        }

        return true;
    }

    LogPrint("instantsend", "ProcessTxLockVote -- Transaction Lock Vote, txid=%s\n", vote.txHash.ToString());
    if(!snodeman.Has(vote.vinStormnode)) {
        LogPrint("instantsend", "ProcessTxLockVote -- Unknown Stormnode %s\n", vote.vinStormnode.prevout.ToStringShort());
        return false;
    }

    int n = snodeman.GetStormnodeRank(vote.vinStormnode, vote.nBlockHeight, MIN_INSTANTSEND_PROTO_VERSION);

    if(n == -1) {
        //can be caused by past versions trying to vote with an invalid protocol
        LogPrint("instantsend", "ProcessTxLockVote -- Outdated Stormnodes %s\n", vote.vinStormnode.prevout.ToStringShort());
        if(pnode) {
            snodeman.AskForSN(pnode, vote.vinStormnode);
        }        return false;
    }
    LogPrint("instantsend", "ProcessTxLockVote -- Stormnode %s, rank=%d\n", vote.vinStormnode.prevout.ToStringShort(), n);

    if(n > INSTANTSEND_SIGNATURES_TOTAL) {
        LogPrint("instantsend", "ProcessTxLockVote -- Stormnode %s is not in the top %d (%d), vote hash=%s\n",
                vote.vinStormnode.prevout.ToStringShort(), INSTANTSEND_SIGNATURES_TOTAL, n, vote.GetHash().ToString());
        return false;
    }

    if(!vote.CheckSignature()) {
        LogPrintf("ProcessTxLockVote -- Signature invalid\n");
        // don't ban, it could just be a non-synced stormnode
        if(pnode) {
            snodeman.AskForSN(pnode, vote.vinStormnode);
        }        return false;
    }

    if(!mapTxLockCandidates.count(vote.txHash)) {
        // this should never happen

        return false;
    }

    //compile consessus vote
    mapTxLockCandidates[vote.txHash].AddVote(vote);

    int nSignatures = mapTxLockCandidates[vote.txHash].CountVotes();
    LogPrint("instantsend", "ProcessTxLockVote -- Transaction Lock signatures count: %d, vote hash=%s\n", nSignatures, vote.GetHash().ToString());

    if(nSignatures >= INSTANTSEND_SIGNATURES_REQUIRED) {
        LogPrint("instantsend", "ProcessTxLockVote -- Transaction Lock Is Complete! txid=%s\n", vote.txHash.ToString());

        if(!FindConflictingLocks(mapLockRequestAccepted[vote.txHash])) { //?????
            if(mapLockRequestAccepted.count(vote.txHash)) {

                UpdateLockedTransaction(mapLockRequestAccepted[vote.txHash]);
                LockTransactionInputs(mapLockRequestAccepted[vote.txHash]);
            } else if(mapLockRequestRejected.count(vote.txHash)) {
                ResolveConflicts(mapLockRequestRejected[vote.txHash]); ///?????
            } else {
                LogPrint("instantsend", "ProcessTxLockVote -- Transaction Lock is missing! nSignatures=%d, vote hash=%s\n", nSignatures, vote.GetHash().ToString());
            }
        }
    }
    CInv inv(MSG_TXLOCK_VOTE, vote.GetHash());
    RelayInv(inv);


    return true;
}

void ProcessOrphanTxLockVotes()
{
    std::map<uint256, CTxLockVote>::iterator it = mapTxLockVotesOrphan.begin();
    while(it != mapTxLockVotesOrphan.end()) {
        if(ProcessTxLockVote(NULL, it->second)) {
            mapTxLockVotesOrphan.erase(it++);
        } else {
            ++it;
        }
    }
}

void UpdateLockedTransaction(const CTransaction& tx, bool fForceNotification)
{
    // there should be no conflicting locks
    if(FindConflictingLocks(tx)) return;
    uint256 txHash = tx.GetHash();
    // there must be a successfully verified lock request
    if(!mapLockRequestAccepted.count(txHash)) return;

    int nSignatures = GetTransactionLockSignatures(txHash);

#ifdef ENABLE_WALLET
    if(pwalletMain && pwalletMain->UpdatedTransaction(txHash)) {
        // bumping this to update UI
        nCompleteTXLocks++;
        // a transaction lock must have enough signatures to trigger this notification
        if(nSignatures == INSTANTSEND_SIGNATURES_REQUIRED || (fForceNotification && nSignatures > INSTANTSEND_SIGNATURES_REQUIRED)) {
            // notify an external script once threshold is reached
            std::string strCmd = GetArg("-instantsendnotify", "");
            if(!strCmd.empty()) {
                boost::replace_all(strCmd, "%s", txHash.GetHex());
                boost::thread t(runCommand, strCmd); // thread runs free
            }
        }
    }
#endif

    if(nSignatures == INSTANTSEND_SIGNATURES_REQUIRED || (fForceNotification && nSignatures > INSTANTSEND_SIGNATURES_REQUIRED))
        GetMainSignals().NotifyTransactionLock(tx);
}

void LockTransactionInputs(const CTransaction& tx) {
    if(!mapLockRequestAccepted.count(tx.GetHash())) return;

    BOOST_FOREACH(const CTxIn& txin, tx.vin)
        if(!mapLockedInputs.count(txin.prevout))
            mapLockedInputs.insert(std::make_pair(txin.prevout, tx.GetHash()));
}

bool FindConflictingLocks(const CTransaction& tx)
{
    /*
        It's possible (very unlikely though) to get 2 conflicting transaction locks approved by the network.
        In that case, they will cancel each other out.

        Blocks could have been rejected during this time, which is OK. After they cancel out, the client will
        rescan the blocks and find they're acceptable and then take the chain with the most work.
    */
    uint256 txHash = tx.GetHash();
    BOOST_FOREACH(const CTxIn& txin, tx.vin) {
        if(mapLockedInputs.count(txin.prevout)) {
            if(mapLockedInputs[txin.prevout] != txHash) {
                LogPrintf("FindConflictingLocks -- found two complete conflicting Transaction Locks, removing both: txid=%s, txin=%s", txHash.ToString(), mapLockedInputs[txin.prevout].ToString());

                if(mapTxLockCandidates.count(txHash))
                    mapTxLockCandidates[txHash].nExpirationBlock = -1;

                if(mapTxLockCandidates.count(mapLockedInputs[txin.prevout]))
                    mapTxLockCandidates[mapLockedInputs[txin.prevout]].nExpirationBlock = -1;

                return true;
            }
        }
    }

    return false;
}

void ResolveConflicts(const CTransaction& tx)
{
    uint256 txHash = tx.GetHash();
    // resolve conflicts
    if (IsLockedInstandSendTransaction(txHash) && !FindConflictingLocks(tx)) { //?????
        LogPrintf("ResolveConflicts -- Found existing complete Transaction Lock, resolving...\n");

        //reprocess the last nInstantSendReprocessBlocks blocks
        ReprocessBlocks(Params().GetConsensus().nInstantSendReprocessBlocks);
        if(!mapLockRequestAccepted.count(txHash))
            mapLockRequestAccepted.insert(std::make_pair(txHash, tx)); //?????
    }
}

int64_t GetAverageStormnodeOrphanVoteTime()
{
    // NOTE: should never actually call this function when mapStormnodeOrphanVotes is empty
    if(mapStormnodeOrphanVotes.empty()) return 0;

    std::map<COutPoint, int64_t>::iterator it = mapStormnodeOrphanVotes.begin();
    int64_t total = 0;

    while(it != mapStormnodeOrphanVotes.end()) {
        total+= it->second;
        ++it;
    }

    return total / mapStormnodeOrphanVotes.size();
}

void CleanTxLockCandidates()
{
    int nHeight;
    {
        LOCK(cs_main);
        nHeight = chainActive.Height();
    }

    LOCK(cs_instantsend);

    std::map<uint256, CTxLockCandidate>::iterator it = mapTxLockCandidates.begin();

    while(it != mapTxLockCandidates.end()) {
        CTxLockCandidate &txLockCandidate = it->second;
        if(nHeight > txLockCandidate.nExpirationBlock) {
            LogPrintf("CleanTxLockCandidates -- Removing expired Transaction Lock Candidate: txid=%s\n", txLockCandidate.txHash.ToString());

            if(mapLockRequestAccepted.count(txLockCandidate.txHash)){
                CTransaction& tx = mapLockRequestAccepted[txLockCandidate.txHash];

                BOOST_FOREACH(const CTxIn& txin, tx.vin)
                    mapLockedInputs.erase(txin.prevout);

                mapLockRequestAccepted.erase(txLockCandidate.txHash);
                mapLockRequestRejected.erase(txLockCandidate.txHash);

                BOOST_FOREACH(const CTxLockVote& vote, txLockCandidate.vecTxLockVotes)
                    if(mapTxLockVotes.count(vote.GetHash()))
                        mapTxLockVotes.erase(vote.GetHash());
            }

            mapTxLockCandidates.erase(it++);
        } else {
            ++it;
        }
    }

    // clean expired orphan votes
    std::map<uint256, CTxLockVote>::iterator it1 = mapTxLockVotesOrphan.begin();
    while(it1 != mapTxLockVotesOrphan.end()) {
        if(it1->second.nOrphanExpireTime < GetTime()) {
            LogPrint("instantsend", "CleanTxLockCandidates -- Removing expired orphan vote: txid=%s  stormnode=%s\n", it1->second.txHash.ToString(), it1->second.vinStormnode.prevout.ToStringShort());
            mapTxLockVotesOrphan.erase(it1++);
        } else {
            ++it1;
        }
    }

    // clean expired stormnode orphan vote times
    std::map<COutPoint, int64_t>::iterator it2 = mapStormnodeOrphanVotes.begin();
    while(it2 != mapStormnodeOrphanVotes.end()) {
        if(it2->second < GetTime()) {
            LogPrint("instantsend", "CleanTxLockCandidates -- Removing expired orphan stormnode vote time: stormnode=%s\n", it2->first.ToStringShort());
            mapStormnodeOrphanVotes.erase(it2++);
        } else {
            ++it2;
        }
    }
}

bool IsLockedInstandSendTransaction(const uint256 &txHash)
{
    // there must be a successfully verified lock request...
    if (!mapLockRequestAccepted.count(txHash)) return false;
    // ...and corresponding lock must have enough signatures
    return GetTransactionLockSignatures(txHash) >= INSTANTSEND_SIGNATURES_REQUIRED;
}

int GetTransactionLockSignatures(const uint256 &txHash)
{
    if(!fEnableInstantSend) return -1;
    if(fLargeWorkForkFound || fLargeWorkInvalidChainFound) return -2;
    if(!sporkManager.IsSporkActive(SPORK_2_INSTANTSEND_ENABLED)) return -3;

    std::map<uint256, CTxLockCandidate>::iterator it = mapTxLockCandidates.find(txHash);
    if(it != mapTxLockCandidates.end()) return it->second.CountVotes();

    return -1;
}

bool IsTransactionLockTimedOut(const uint256 &txHash)
{
    if(!fEnableInstantSend) return 0;

    std::map<uint256, CTxLockCandidate>::iterator i = mapTxLockCandidates.find(txHash);
    if (i != mapTxLockCandidates.end()) return GetTime() > (*i).second.nTimeout;

    return false;
}

uint256 CTxLockVote::GetHash() const
{
    return ArithToUint256(UintToArith256(vinStormnode.prevout.hash) + vinStormnode.prevout.n + UintToArith256(txHash));
}


bool CTxLockVote::CheckSignature() const
{
    std::string strError;
    std::string strMessage = txHash.ToString().c_str() + boost::lexical_cast<std::string>(nBlockHeight);

    stormnode_info_t infoSn = snodeman.GetStormnodeInfo(vinStormnode);

    if(!infoSn.fInfoValid) {
        LogPrintf("CTxLockVote::CheckSignature -- Unknown Stormnode: txin=%s\n", vinStormnode.ToString());
        return false;
    }

    if(!sandStormSigner.VerifyMessage(infoSn.pubKeyStormnode, vchStormNodeSignature, strMessage, strError)) {
        LogPrintf("CTxLockVote::CheckSignature -- VerifyMessage() failed, error: %s\n", strError);
        return false;
    }

    return true;
}

bool CTxLockVote::Sign()
{
    std::string strError;

    std::string strMessage = txHash.ToString().c_str() + boost::lexical_cast<std::string>(nBlockHeight);

    if(!sandStormSigner.SignMessage(strMessage, vchStormNodeSignature, activeStormnode.keyStormnode)) {
        LogPrintf("CTxLockVote::Sign -- SignMessage() failed\n");
        return false;
    }

    if(!sandStormSigner.VerifyMessage(activeStormnode.pubKeyStormnode, vchStormNodeSignature, strMessage, strError)) {
        LogPrintf("CTxLockVote::Sign -- VerifyMessage() failed, error: %s\n", strError);
        return false;
    }

    return true;
}


bool CTxLockCandidate::IsAllVotesValid()
{

    BOOST_FOREACH(const CTxLockVote& vote, vecTxLockVotes)
    {
        int n = snodeman.GetStormnodeRank(vote.vinStormnode, vote.nBlockHeight, MIN_INSTANTSEND_PROTO_VERSION);

        if(n == -1) {
            LogPrintf("CTxLockCandidate::IsAllVotesValid -- Unknown Stormnode, txin=%s\n", vote.vinStormnode.ToString());
            return false;
        }

        if(n > INSTANTSEND_SIGNATURES_TOTAL) {
            LogPrintf("CTxLockCandidate::IsAllVotesValid -- Stormnode not in the top %s\n", INSTANTSEND_SIGNATURES_TOTAL);
            return false;
        }

        if(!vote.CheckSignature()) {
            LogPrintf("CTxLockCandidate::IsAllVotesValid -- Signature not valid\n");
            return false;
        }
    }

    return true;
}

void CTxLockCandidate::AddVote(const CTxLockVote& vote)
{
    vecTxLockVotes.push_back(vote);
}

int CTxLockCandidate::CountVotes()
{
    /*
        Only count signatures where the BlockHeight matches the transaction's blockheight.
        The votes have no proof it's the correct blockheight
    */

    if(nBlockHeight == 0) return -1;

    int nCount = 0;
    BOOST_FOREACH(CTxLockVote vote, vecTxLockVotes)
        if(vote.nBlockHeight == nBlockHeight)
            nCount++;

    return nCount;
}
