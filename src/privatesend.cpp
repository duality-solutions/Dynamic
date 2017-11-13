// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "privatesend.h"

#include "activedynode.h"
#include "consensus/validation.h"
#include "governance.h"
#include "init.h"
#include "instantsend.h"
#include "dynode-payments.h"
#include "dynode-sync.h"
#include "dynodeman.h"
#include "messagesigner.h"
#include "script/sign.h"
#include "txmempool.h"
#include "util.h"
#include "utilmoneystr.h"

#include <boost/lexical_cast.hpp>

std::map<uint256, CPrivatesendBroadcastTx> mapPrivatesendBroadcastTxes;
std::vector<CAmount> vecPrivateSendDenominations;

CPrivateSendEntry::CPrivateSendEntry(const std::vector<CTxIn>& vecTxIn, const std::vector<CTxOut>& vecTxOut, const CTransaction& txCollateral) :
    txCollateral(txCollateral)
{
    BOOST_FOREACH(CTxIn txin, vecTxIn)
        vecTxPSIn.push_back(txin);
    BOOST_FOREACH(CTxOut txout, vecTxOut)
        vecTxPSOut.push_back(txout);
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

void CPrivateSend::InitDenominations()
{
    vecPrivateSendDenominations.clear();
    /* Denominations

        A note about convertability. Within mixing pools, each denomination
        is convertable to another.

        For example:
        1DRK+1000 == (.1DRK+100)*10
        10DRK+10000 == (1DRK+1000)*10
    */
    /* Disabled
    vecPrivateSendDenominations.push_back( (100      * COIN)+100000 );
    */
    vecPrivateSendDenominations.push_back( (10       * COIN)+10000 );
    vecPrivateSendDenominations.push_back( (1        * COIN)+1000 );
    vecPrivateSendDenominations.push_back( (.1       * COIN)+100 );
    vecPrivateSendDenominations.push_back( (.01      * COIN)+10 );
    /* Disabled till we need them
    vecPrivateSendDenominations.push_back( (.001     * COIN)+1 );
    */
}

void CPrivateSend::SetNull()
{
    // Both sides
    nState = POOL_STATE_IDLE;
    nSessionID = 0;
    nSessionDenom = 0;
    vecEntries.clear();
    finalMutableTransaction.vin.clear();
    finalMutableTransaction.vout.clear();
    nTimeLastSuccessfulStep = GetTimeMillis();
}

std::string CPrivateSend::GetStateString() const
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

// check to make sure the collateral provided by the client is valid
bool CPrivateSend::IsCollateralValid(const CTransaction& txCollateral)
{
    if(txCollateral.vout.empty()) return false;
    if(txCollateral.nLockTime != 0) return false;

    CAmount nValueIn = 0;
    CAmount nValueOut = 0;
    bool fMissingTx = false;

    BOOST_FOREACH(const CTxOut txout, txCollateral.vout) {
        nValueOut += txout.nValue;

        if(!txout.scriptPubKey.IsNormalPaymentScript()) {
            LogPrintf ("CPrivateSend::IsCollateralValid -- Invalid Script, txCollateral=%s", txCollateral.ToString());
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
        LogPrint("privatesend", "CPrivateSend::IsCollateralValid -- Unknown inputs in collateral transaction, txCollateral=%s", txCollateral.ToString());
        return false;
    }

    //collateral transactions are required to pay out PRIVATESEND_COLLATERAL as a fee to the miners
    if(nValueIn - nValueOut < PRIVATESEND_COLLATERAL) {
        LogPrint("privatesend", "CPrivateSend::IsCollateralValid -- did not include enough fees in transaction: fees: %d, txCollateral=%s", nValueOut - nValueIn, txCollateral.ToString());
        return false;
    }

    LogPrint("privatesend", "CPrivateSend::IsCollateralValid -- %s", txCollateral.ToString());

    {
        LOCK(cs_main);
        CValidationState validationState;
        if(!AcceptToMemoryPool(mempool, validationState, txCollateral, false, NULL, false, true, true)) {
            LogPrint("privatesend", "CPrivateSend::IsCollateralValid -- didn't pass AcceptToMemoryPool()\n");
            return false;
        }
    }

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
std::string CPrivateSend::GetDenominationsToString(int nDenom)
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

int CPrivateSend::GetDenominations(const std::vector<CTxPSOut>& vecTxPSOut)
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
int CPrivateSend::GetDenominations(const std::vector<CTxOut>& vecTxOut, bool fSingleRandomDenom)
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

bool CPrivateSend::GetDenominationsBits(int nDenom, std::vector<int> &vecBitsRet)
{
    // ( bit on if present, 4 denominations example )
    // bit 0 - 100DASH+1
    // bit 1 - 10DASH+1
    // bit 2 - 1DASH+1
    // bit 3 - .1DASH+1

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

int CPrivateSend::GetDenominationsByAmounts(const std::vector<CAmount>& vecAmount)
{
    CScript scriptTmp = CScript();
    std::vector<CTxOut> vecTxOut;

    BOOST_REVERSE_FOREACH(CAmount nAmount, vecAmount) {
        CTxOut txout(nAmount, scriptTmp);
        vecTxOut.push_back(txout);
    }

    return GetDenominations(vecTxOut, true);
}

std::string CPrivateSend::GetMessageByID(PoolMessage nMessageID)
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
        case ERR_MN_LIST:               return _("Not in the Dynode list.");
        case ERR_MODE:                  return _("Incompatible mode.");
        case ERR_NON_STANDARD_PUBKEY:   return _("Non-standard public key detected.");
        case ERR_NOT_A_MN:              return _("This is not a Dynode."); // not used
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
        LogPrintf("CPrivatesendBroadcastTx::CheckSignature -- Got bad dstx signature, error: %s\n", strError);
        return false;
    }

    return true;
}

//TODO: Rename/move to core
void ThreadCheckPrivateSend()
{
    if(fLiteMode) return; // disable all Dynamic specific functionality

    static bool fOneThread;
    if(fOneThread) return;
    fOneThread = true;

    // Make this thread recognisable as the PrivateSend thread
    RenameThread("dynamic-ps");

    unsigned int nTick = 0;

    while (true)
    {
        MilliSleep(1000);

        // try to sync from all available nodes, one step at a time
        dynodeSync.ProcessTick();

        if(dynodeSync.IsBlockchainSynced() && !ShutdownRequested()) {

            nTick++;

            // make sure to check all dynodes first
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
        }
    }
}