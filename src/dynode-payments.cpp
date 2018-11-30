// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dynode-payments.h"

#include "activedynode.h"
#include "chain.h"
#include "consensus/validation.h"
#include "dynode-sync.h"
#include "dynodeman.h"
#include "fluid/fluiddb.h"
#include "governance-classes.h"
#include "init.h"
#include "key_io.h"
#include "messagesigner.h"
#include "netfulfilledman.h"
#include "netmessagemaker.h"
#include "policy/fees.h"
#include "spork.h"
#include "util.h"
#include "utilmoneystr.h"

#include <boost/lexical_cast.hpp>

/** Object for who's going to get paid on which blocks */
CDynodePayments dnpayments;

CCriticalSection cs_vecPayees;
CCriticalSection cs_mapDynodeBlocks;
CCriticalSection cs_mapDynodePaymentVotes;

/**
* IsBlockValueValid
*
*   Determine if coinbase outgoing created money is the correct value
*
*   Why is this needed?
*   - In Dynamic some blocks are superblocks, which output much higher amounts of coins
*   - Otherblocks are 10% lower in outgoing value, so in total, no extra coins are created
*   - When non-superblocks are detected, the normal schedule should be maintained
*/

bool IsBlockValueValid(const CBlock& block, int nBlockHeight, CAmount blockReward, std::string& strErrorRet)
{
    strErrorRet = "";

    bool isBlockRewardValueMet = (block.vtx[0]->GetValueOut() <= blockReward);
    if (fDebug)
        LogPrintf("block.vtx[0].GetValueOut() %lld <= blockReward %lld\n", block.vtx[0]->GetValueOut(), blockReward);

    // we are still using budgets, but we have no data about them anymore,
    // all we know is predefined budget cycle and window

    const Consensus::Params& consensusParams = Params().GetConsensus();
    if (nBlockHeight < consensusParams.nSuperblockStartBlock) {
        int nOffset = nBlockHeight % consensusParams.nBudgetPaymentsCycleBlocks;
        if (nBlockHeight >= consensusParams.nBudgetPaymentsStartBlock &&
            nOffset < consensusParams.nBudgetPaymentsWindowBlocks) {
            if (dynodeSync.IsSynced() && !sporkManager.IsSporkActive(SPORK_13_OLD_SUPERBLOCK_FLAG)) {
                LogPrint("gobject", "IsBlockValueValid -- Client synced but budget spork is disabled, checking block value against block reward\n");
                if (!isBlockRewardValueMet) {
                    strErrorRet = strprintf("coinbase pays too much at height %d (actual=%d vs limit=%d), exceeded block reward, budgets are disabled",
                        nBlockHeight, block.vtx[0]->GetValueOut(), blockReward);
                }
                return isBlockRewardValueMet;
            }
            LogPrint("gobject", "IsBlockValueValid -- WARNING: Skipping budget block value checks, accepting block\n");
            // TODO: reprocess blocks to make sure they are legit?
            return true;
        }
        // LogPrint("gobject", "IsBlockValueValid -- Block is not in budget cycle window, checking block value against block reward\n");
        if (!isBlockRewardValueMet) {
            strErrorRet = strprintf("coinbase pays too much at height %d (actual=%d vs limit=%d), exceeded block reward, block is not in budget cycle window",
                nBlockHeight, block.vtx[0]->GetValueOut(), blockReward);
        }
        return isBlockRewardValueMet;
    }

    // superblocks started

    CAmount nSuperblockMaxValue = blockReward + CSuperblock::GetPaymentsLimit(nBlockHeight);
    bool isSuperblockMaxValueMet = (block.vtx[0]->GetValueOut() <= nSuperblockMaxValue);

    LogPrint("gobject", "block.vtx[0].GetValueOut() %lld <= nSuperblockMaxValue %lld\n", block.vtx[0]->GetValueOut(), nSuperblockMaxValue);

    if (!dynodeSync.IsSynced()) {
        // not enough data but at least it must NOT exceed superblock max value
        if (CSuperblock::IsValidBlockHeight(nBlockHeight)) {
            if (fDebug)
                LogPrintf("IsBlockPayeeValid -- WARNING: Client not synced, checking superblock max bounds only\n");
            if (!isSuperblockMaxValueMet) {
                strErrorRet = strprintf("coinbase pays too much at height %d (actual=%d vs limit=%d), exceeded superblock max value",
                    nBlockHeight, block.vtx[0]->GetValueOut(), nSuperblockMaxValue);
            }
            return isSuperblockMaxValueMet;
        }
        if (!isBlockRewardValueMet) {
            strErrorRet = strprintf("coinbase pays too much at height %d (actual=%d vs limit=%d), exceeded block reward, only regular blocks are allowed at this height",
                nBlockHeight, block.vtx[0]->GetValueOut(), blockReward);
        }
        // it MUST be a regular block otherwise
        return isBlockRewardValueMet;
    }

    // we are synced, let's try to check as much data as we can

    if (sporkManager.IsSporkActive(SPORK_9_SUPERBLOCKS_ENABLED)) {
        if (CSuperblockManager::IsSuperblockTriggered(nBlockHeight)) {
            if (CSuperblockManager::IsValid(*block.vtx[0], nBlockHeight, blockReward)) {
                LogPrint("gobject", "IsBlockValueValid -- Valid superblock at height %d: %s", nBlockHeight, block.vtx[0]->ToString());
                // all checks are done in CSuperblock::IsValid, nothing to do here
                return true;
            }

            // triggered but invalid? that's weird
            LogPrintf("IsBlockValueValid -- ERROR: Invalid superblock detected at height %d: %s", nBlockHeight, block.vtx[0]->ToString());
            // should NOT allow invalid superblocks, when superblocks are enabled
            strErrorRet = strprintf("invalid superblock detected at height %d", nBlockHeight);
            return false;
        }
        LogPrint("gobject", "IsBlockValueValid -- No triggered superblock detected at height %d\n", nBlockHeight);
        if (!isBlockRewardValueMet) {
            strErrorRet = strprintf("coinbase pays too much at height %d (actual=%d vs limit=%d), exceeded block reward, no triggered superblock detected",
                nBlockHeight, block.vtx[0]->GetValueOut(), blockReward);
        }
    } else {
        // should NOT allow superblocks at all, when superblocks are disabled
        LogPrint("gobject", "IsBlockValueValid -- Superblocks are disabled, no superblocks allowed\n");
        if (!isBlockRewardValueMet) {
            strErrorRet = strprintf("coinbase pays too much at height %d (actual=%d vs limit=%d), exceeded block reward, superblocks are disabled",
                nBlockHeight, block.vtx[0]->GetValueOut(), blockReward);
        }
    }

    // it MUST be a regular block
    return isBlockRewardValueMet;
}

bool IsBlockPayeeValid(const CTransaction& txNew, int nBlockHeight, CAmount blockReward)
{
    if (!dynodeSync.IsSynced()) {
        //there is no budget data to use to check anything, let's just accept the longest chain
        if (fDebug)
            LogPrintf("IsBlockPayeeValid -- WARNING: Client not synced, skipping block payee checks\n");
        return true;
    }

    // we are still using budgets, but we have no data about them anymore,
    // we can only check dynode payments

    const Consensus::Params& consensusParams = Params().GetConsensus();

    if (nBlockHeight < consensusParams.nSuperblockStartBlock) {
        LogPrint("gobject", "IsBlockPayeeValid -- WARNING: Client synced but old budget system is disabled, accepting any payee\n");
        return true;
    }

    // superblocks started
    // SEE IF THIS IS A VALID SUPERBLOCK

    if (sporkManager.IsSporkActive(SPORK_9_SUPERBLOCKS_ENABLED)) {
        if (CSuperblockManager::IsSuperblockTriggered(nBlockHeight)) {
            if (CSuperblockManager::IsValid(txNew, nBlockHeight, blockReward)) {
                LogPrint("gobject", "IsBlockPayeeValid -- Valid superblock at height %d: %s", nBlockHeight, txNew.ToString());
                return true;
            }

            LogPrintf("IsBlockPayeeValid -- ERROR: Invalid superblock detected at height %d: %s", nBlockHeight, txNew.ToString());
            // should NOT allow such superblocks, when superblocks are enabled
            return false;
        }
        // continue validation, should pay DN
        LogPrint("gobject", "IsBlockPayeeValid -- No triggered superblock detected at height %d\n", nBlockHeight);
    } else {
        // should NOT allow superblocks at all, when superblocks are disabled
        LogPrint("gobject", "IsBlockPayeeValid -- Superblocks are disabled, no superblocks allowed\n");
    }

    // IF THIS ISN'T A SUPERBLOCK OR SUPERBLOCK IS INVALID, IT SHOULD PAY A DYNODE DIRECTLY
    if (dnpayments.IsTransactionValid(txNew, nBlockHeight)) {
        LogPrint("dnpayments", "IsBlockPayeeValid -- Valid dynode payment at height %d: %s", nBlockHeight, txNew.ToString());
        return true;
    }

    if (sporkManager.IsSporkActive(SPORK_8_DYNODE_PAYMENT_ENFORCEMENT)) {
        LogPrintf("IsBlockPayeeValid -- ERROR: Invalid dynode payment detected at height %d: %s", nBlockHeight, txNew.ToString());
        return false;
    }

    LogPrintf("IsBlockPayeeValid -- WARNING: Dynode payment enforcement is disabled, accepting any payee\n");
    return true;
}

void FillBlockPayments(CMutableTransaction& txNew, int nBlockHeight, CAmount blockReward, CTxOut& txoutDynodeRet, std::vector<CTxOut>& voutSuperblockRet)
{
    // only create superblocks if spork is enabled AND if superblock is actually triggered
    // (height should be validated inside)
    if (sporkManager.IsSporkActive(SPORK_9_SUPERBLOCKS_ENABLED) &&
        CSuperblockManager::IsSuperblockTriggered(nBlockHeight)) {
        LogPrint("gobject", "FillBlockPayments -- triggered superblock creation at height %d\n", nBlockHeight);
        CSuperblockManager::CreateSuperblock(txNew, nBlockHeight, voutSuperblockRet);
        return;
    }


    // FILL BLOCK PAYEE WITH DYNODE PAYMENT OTHERWISE
    dnpayments.FillBlockPayee(txNew, nBlockHeight, blockReward, txoutDynodeRet);
    LogPrint("dnpayments", "FillBlockPayments -- nBlockHeight %d blockReward %lld txoutDynodeRet %s txNew %s",
        nBlockHeight, blockReward, txoutDynodeRet.ToString(), txNew.ToString());
}

std::string GetRequiredPaymentsString(int nBlockHeight)
{
    // IF WE HAVE A ACTIVATED TRIGGER FOR THIS HEIGHT - IT IS A SUPERBLOCK, GET THE REQUIRED PAYEES
    if (CSuperblockManager::IsSuperblockTriggered(nBlockHeight)) {
        return CSuperblockManager::GetRequiredPaymentsString(nBlockHeight);
    }

    // OTHERWISE, PAY DYNODE
    return dnpayments.GetRequiredPaymentsString(nBlockHeight);
}

void CDynodePayments::Clear()
{
    LOCK2(cs_mapDynodeBlocks, cs_mapDynodePaymentVotes);
    mapDynodeBlocks.clear();
    mapDynodePaymentVotes.clear();
}

bool CDynodePayments::UpdateLastVote(const CDynodePaymentVote& vote)
{
    LOCK(cs_mapDynodePaymentVotes);

    const auto it = mapDynodesLastVote.find(vote.dynodeOutpoint);
    if (it != mapDynodesLastVote.end()) {
        if (it->second == vote.nBlockHeight)
            return false;
        it->second = vote.nBlockHeight;
        return true;
    }

    //record this dynode voted
    mapDynodesLastVote.emplace(vote.dynodeOutpoint, vote.nBlockHeight);
    return true;
}

/**
*   FillBlockPayee
*
*   Fill Dynode ONLY payment block
*/

void CDynodePayments::FillBlockPayee(CMutableTransaction& txNew, int nBlockHeight, CAmount blockReward, CTxOut& txoutDynodeRet) const
{
    // make sure it's not filled yet
    txoutDynodeRet = CTxOut();

    CScript payee;

    bool hasPayment = true;

    if (hasPayment && !dnpayments.GetBlockPayee(nBlockHeight, payee)) {
        // no dynode detected...
        int nCount = 0;
        dynode_info_t dnInfo;
        if (!dnodeman.GetNextDynodeInQueueForPayment(nBlockHeight, true, nCount, dnInfo)) {
            hasPayment = false;
            LogPrintf("CDynodePayments::FillBlockPayee: Failed to detect Dynode to pay\n");
            return;
        }
        // fill payee with locally calculated winner and hope for the best
        payee = GetScriptForDestination(dnInfo.pubKeyCollateralAddress.GetID());
    }

    // make sure it's not filled yet
    txoutDynodeRet = CTxOut();
    CAmount dynodePayment = GetFluidDynodeReward(nBlockHeight);

    // split reward between miner ...
    txoutDynodeRet = CTxOut(dynodePayment, payee);
    txNew.vout.push_back(txoutDynodeRet);
    // ... and dynode
    CTxDestination address1;
    ExtractDestination(payee, address1);
    CDynamicAddress address2(address1);

    LogPrintf("CDynodePayments::FillBlockPayee -- Dynode payment %lld to %s\n", dynodePayment, address2.ToString());
}

int CDynodePayments::GetMinDynodePaymentsProto() const
{
    return sporkManager.IsSporkActive(SPORK_10_DYNODE_PAY_UPDATED_NODES) ? MIN_DYNODE_PAYMENT_PROTO_VERSION_2 : MIN_DYNODE_PAYMENT_PROTO_VERSION_1;
}

void CDynodePayments::ProcessMessage(CNode* pfrom, const std::string& strCommand, CDataStream& vRecv, CConnman& connman)
{
    if (fLiteMode)
        return; // disable all Dynamic specific functionality

    if (strCommand == NetMsgType::DYNODEPAYMENTSYNC) { //Dynode Payments Request Sync

        if (pfrom->nVersion < GetMinDynodePaymentsProto()) {
            LogPrint("dnpayments", "DYNODEPAYMENTSYNC -- peer=%d using obsolete version %i\n", pfrom->id, pfrom->nVersion);
            connman.PushMessage(pfrom, CNetMsgMaker(pfrom->GetSendVersion()).Make(NetMsgType::REJECT, strCommand, REJECT_OBSOLETE, strprintf("Version must be %d or greater", GetMinDynodePaymentsProto())));
            return;
        }

        // Ignore such requests until we are fully synced.
        // We could start processing this after Dynode list is synced
        // but this is a heavy one so it's better to finish sync first.
        if (!dynodeSync.IsSynced())
            return;

        // DEPRECATED, should be removed on next protocol bump
        if (pfrom->nVersion == 70900) {
            int nCountNeeded;
            vRecv >> nCountNeeded;
        }


        if (netfulfilledman.HasFulfilledRequest(pfrom->addr, NetMsgType::DYNODEPAYMENTSYNC)) {
            // Asking for the payments list multiple times in a short period of time is no good
            LogPrintf("DYNODEPAYMENTSYNC -- peer already asked me for the list, peer=%d\n", pfrom->id);
            Misbehaving(pfrom->GetId(), 20);
            return;
        }

        netfulfilledman.AddFulfilledRequest(pfrom->addr, NetMsgType::DYNODEPAYMENTSYNC);

        Sync(pfrom, connman);
        LogPrintf("DYNODEPAYMENTSYNC -- Sent Dynode payment votes to peer %d\n", pfrom->id);

    } else if (strCommand == NetMsgType::DYNODEPAYMENTVOTE) { // Dynode Payments Vote for the Winner

        CDynodePaymentVote vote;
        vRecv >> vote;

        if (pfrom->nVersion < GetMinDynodePaymentsProto()) {
            LogPrint("dnpayments", "DYNODEPAYMENTVOTE -- peer=%d using obsolete version %i\n", pfrom->id, pfrom->nVersion);
            connman.PushMessage(pfrom, CNetMsgMaker(pfrom->GetSendVersion()).Make(NetMsgType::REJECT, strCommand, REJECT_OBSOLETE, strprintf("Version must be %d or greater", GetMinDynodePaymentsProto())));
            return;
        }

        uint256 nHash = vote.GetHash();

        pfrom->setAskFor.erase(nHash);

        // TODO: clear setAskFor for MSG_DYNODE_PAYMENT_BLOCK too

        // Ignore any payments messages until dynode list is synced
        if (!dynodeSync.IsDynodeListSynced())
            return;

        {
            LOCK(cs_mapDynodePaymentVotes);

            auto res = mapDynodePaymentVotes.emplace(nHash, vote);

            // Avoid processing same vote multiple times if it was already verified earlier
            if (!res.second && res.first->second.IsVerified()) {
                LogPrint("dnpayments", "DYNODEPAYMENTVOTE -- hash=%s, nBlockHeight=%d/%d seen\n",
                    nHash.ToString(), vote.nBlockHeight, nCachedBlockHeight);
                return;
            }

            // Mark vote as non-verified when it's seen for the first time,
            // AddOrUpdatePaymentVote() below should take care of it if vote is actually ok
            res.first->second.MarkAsNotVerified();
        }

        int nFirstBlock = nCachedBlockHeight - GetStorageLimit();
        if (vote.nBlockHeight < nFirstBlock || vote.nBlockHeight > nCachedBlockHeight + 20) {
            LogPrint("dnpayments", "DYNODEPAYMENTVOTE -- vote out of range: nFirstBlock=%d, nBlockHeight=%d, nHeight=%d\n", nFirstBlock, vote.nBlockHeight, nCachedBlockHeight);
            return;
        }

        std::string strError = "";
        if (!vote.IsValid(pfrom, nCachedBlockHeight, strError, connman)) {
            LogPrint("dnpayments", "DYNODEPAYMENTVOTE -- invalid message, error: %s\n", strError);
            return;
        }

        dynode_info_t dnInfo;
        if (!dnodeman.GetDynodeInfo(vote.dynodeOutpoint, dnInfo)) {
            // dn was not found, so we can't check vote, some info is probably missing
            LogPrintf("DYNODEPAYMENTVOTE -- Dynode is missing %s\n", vote.dynodeOutpoint.ToStringShort());
            dnodeman.AskForDN(pfrom, vote.dynodeOutpoint, connman);
            return;
        }

        int nDos = 0;
        if (!vote.CheckSignature(dnInfo.pubKeyDynode, nCachedBlockHeight, nDos)) {
            if (nDos) {
                LOCK(cs_main);
                LogPrintf("DYNODEPAYMENTVOTE -- ERROR: invalid signature\n");
                Misbehaving(pfrom->GetId(), nDos);
            } else {
                // only warn about anything non-critical (i.e. nDos == 0) in debug mode
                LogPrint("dnpayments", "DYNODEPAYMENTVOTE -- WARNING: invalid signature\n");
            }
            // Either our info or vote info could be outdated.
            // In case our info is outdated, ask for an update,
            dnodeman.AskForDN(pfrom, vote.dynodeOutpoint, connman);
            // but there is nothing we can do if vote info itself is outdated
            // (i.e. it was signed by a DN which changed its key),
            // so just quit here.
            return;
        }

        if (!UpdateLastVote(vote)) {
            LogPrintf("DYNODEPAYMENTVOTE -- dynode already voted, dynode=%s\n", vote.dynodeOutpoint.ToStringShort());
            return;
        }

        CTxDestination address1;
        ExtractDestination(vote.payee, address1);
        CDynamicAddress address2(address1);

        LogPrint("dnpayments", "DYNODEPAYMENTVOTE -- vote: address=%s, nBlockHeight=%d, nHeight=%d, prevout=%s, hash=%s new\n",
            address2.ToString(), vote.nBlockHeight, nCachedBlockHeight, vote.dynodeOutpoint.ToStringShort(), nHash.ToString());

        if (AddOrUpdatePaymentVote(vote)) {
            vote.Relay(connman);
            dynodeSync.BumpAssetLastTime("DYNODEPAYMENTVOTE");
        }
    }
}

uint256 CDynodePaymentVote::GetHash() const
{
    // Note: doesn't match serialization

    CHashWriter ss(SER_GETHASH, PROTOCOL_VERSION);
    ss << *(CScriptBase*)(&payee);
    ss << nBlockHeight;
    ss << dynodeOutpoint;
    return ss.GetHash();
}

uint256 CDynodePaymentVote::GetSignatureHash() const
{
    return SerializeHash(*this);
}

bool CDynodePaymentVote::Sign()
{
    std::string strError;

    if (sporkManager.IsSporkActive(SPORK_6_NEW_SIGS)) {
        uint256 hash = GetSignatureHash();

        if (!CHashSigner::SignHash(hash, activeDynode.keyDynode, vchSig)) {
            LogPrintf("CDynodePaymentVote::Sign -- SignHash() failed\n");
            return false;
        }

        if (!CHashSigner::VerifyHash(hash, activeDynode.pubKeyDynode, vchSig, strError)) {
            LogPrintf("CDynodePaymentVote::Sign -- VerifyHash() failed, error: %s\n", strError);
            return false;
        }
    } else {
        std::string strMessage = dynodeOutpoint.ToStringShort() +
                                 std::to_string(nBlockHeight) +
                                 ScriptToAsmStr(payee);

        if (!CMessageSigner::SignMessage(strMessage, vchSig, activeDynode.keyDynode)) {
            LogPrintf("CDynodePaymentVote::Sign -- SignMessage() failed\n");
            return false;
        }

        if (!CMessageSigner::VerifyMessage(activeDynode.pubKeyDynode, vchSig, strMessage, strError)) {
            LogPrintf("CDynodePaymentVote::Sign -- VerifyMessage() failed, error: %s\n", strError);
            return false;
        }
    }

    return true;
}

bool CDynodePayments::GetBlockPayee(int nBlockHeight, CScript& payeeRet) const
{
    LOCK(cs_mapDynodeBlocks);

    auto it = mapDynodeBlocks.find(nBlockHeight);
    return it != mapDynodeBlocks.end() && it->second.GetBestPayee(payeeRet);
}

// Is this Dynode scheduled to get paid soon?
// -- Only look ahead up to 8 blocks to allow for propagation of the latest 2 blocks of votes
bool CDynodePayments::IsScheduled(const dynode_info_t& dnInfo, int nNotBlockHeight) const
{
    LOCK(cs_mapDynodeBlocks);

    if (!dynodeSync.IsDynodeListSynced())
        return false;

    CScript dnpayee;
    dnpayee = GetScriptForDestination(dnInfo.pubKeyCollateralAddress.GetID());

    CScript payee;
    for (int64_t h = nCachedBlockHeight; h <= nCachedBlockHeight + 8; h++) {
        if (h == nNotBlockHeight)
            continue;
        if (GetBlockPayee(h, payee) && dnpayee == payee) {
            return true;
        }
    }

    return false;
}

bool CDynodePayments::AddOrUpdatePaymentVote(const CDynodePaymentVote& vote)
{
    uint256 blockHash = uint256();
    if (!GetBlockHash(blockHash, vote.nBlockHeight - 101))
        return false;

    uint256 nVoteHash = vote.GetHash();

    if (HasVerifiedPaymentVote(nVoteHash))
        return false;

    LOCK2(cs_mapDynodeBlocks, cs_mapDynodePaymentVotes);

    mapDynodePaymentVotes[nVoteHash] = vote;

    auto it = mapDynodeBlocks.emplace(vote.nBlockHeight, CDynodeBlockPayees(vote.nBlockHeight)).first;
    it->second.AddPayee(vote);

    LogPrint("dnpayments", "CDynodePayments::AddOrUpdatePaymentVote -- added, hash=%s\n", nVoteHash.ToString());

    return true;
}

bool CDynodePayments::HasVerifiedPaymentVote(const uint256& hashIn) const
{
    LOCK(cs_mapDynodePaymentVotes);
    const auto it = mapDynodePaymentVotes.find(hashIn);
    return it != mapDynodePaymentVotes.end() && it->second.IsVerified();
}

void CDynodeBlockPayees::AddPayee(const CDynodePaymentVote& vote)
{
    LOCK(cs_vecPayees);

    uint256 nVoteHash = vote.GetHash();

    for (auto& payee : vecPayees) {
        if (payee.GetPayee() == vote.payee) {
            payee.AddVoteHash(nVoteHash);
            return;
        }
    }
    CDynodePayee payeeNew(vote.payee, nVoteHash);
    vecPayees.push_back(payeeNew);
}

bool CDynodeBlockPayees::GetBestPayee(CScript& payeeRet) const
{
    LOCK(cs_vecPayees);

    if (!vecPayees.size()) {
        LogPrint("dnpayments", "CDynodeBlockPayees::GetBestPayee -- ERROR: couldn't find any payee\n");
        return false;
    }

    int nVotes = -1;
    for (const auto& payee : vecPayees) {
        if (payee.GetVoteCount() > nVotes) {
            payeeRet = payee.GetPayee();
            nVotes = payee.GetVoteCount();
        }
    }

    return (nVotes > -1);
}

bool CDynodeBlockPayees::HasPayeeWithVotes(const CScript& payeeIn, int nVotesReq) const
{
    LOCK(cs_vecPayees);

    for (const auto& payee : vecPayees) {
        if (payee.GetVoteCount() >= nVotesReq && payee.GetPayee() == payeeIn) {
            return true;
        }
    }

    LogPrint("dnpayments", "CDynodeBlockPayees::HasPayeeWithVotes -- ERROR: couldn't find any payee with %d+ votes\n", nVotesReq);
    return false;
}

bool CDynodeBlockPayees::IsTransactionValid(const CTransaction& txNew, const int nHeight) const
{
    LOCK(cs_vecPayees);

    int nMaxSignatures = 0;
    std::string strPayeesPossible = "";
    CAmount nDynodePayment = GetFluidDynodeReward(nHeight);

    //require at least DNPAYMENTS_SIGNATURES_REQUIRED signatures

    for (const auto& payee : vecPayees) {
        if (payee.GetVoteCount() >= nMaxSignatures) {
            nMaxSignatures = payee.GetVoteCount();
        }
    }

    // if we don't have at least DNPAYMENTS_SIGNATURES_REQUIRED signatures on a payee, approve whichever is the longest chain
    if (nMaxSignatures < DNPAYMENTS_SIGNATURES_REQUIRED)
        return true;

    for (const auto& payee : vecPayees) {
        if (payee.GetVoteCount() >= DNPAYMENTS_SIGNATURES_REQUIRED) {
            for (const auto& txout : txNew.vout) {
                if (payee.GetPayee() == txout.scriptPubKey && nDynodePayment == txout.nValue) {
                    LogPrint("dnpayments", "CDynodeBlockPayees::IsTransactionValid -- Found required payment\n");
                    return true;
                }
            }

            CTxDestination address1;
            ExtractDestination(payee.GetPayee(), address1);
            CDynamicAddress address2(address1);

            if (strPayeesPossible == "") {
                strPayeesPossible = address2.ToString();
            } else {
                strPayeesPossible += "," + address2.ToString();
            }
        }
    }

    LogPrintf("CDynodeBlockPayees::IsTransactionValid -- ERROR: Missing required payment, possible payees: '%s', amount: %f DYN\n", strPayeesPossible, (float)nDynodePayment / COIN);
    return false;
}

std::string CDynodeBlockPayees::GetRequiredPaymentsString() const
{
    LOCK(cs_vecPayees);

    std::string strRequiredPayments = "";

    for (const auto& payee : vecPayees) {
        CTxDestination address1;
        ExtractDestination(payee.GetPayee(), address1);
        CDynamicAddress address2(address1);

        if (!strRequiredPayments.empty())
            strRequiredPayments += ", ";

        strRequiredPayments += strprintf("%s:%d", address2.ToString(), payee.GetVoteCount());
    }

    if (strRequiredPayments.empty())
        return "Unknown";

    return strRequiredPayments;
}

std::string CDynodePayments::GetRequiredPaymentsString(int nBlockHeight) const
{
    LOCK(cs_mapDynodeBlocks);

    const auto it = mapDynodeBlocks.find(nBlockHeight);
    return it == mapDynodeBlocks.end() ? "Unknown" : it->second.GetRequiredPaymentsString();
}

bool CDynodePayments::IsTransactionValid(const CTransaction& txNew, int nBlockHeight) const
{
    LOCK(cs_mapDynodeBlocks);

    const auto it = mapDynodeBlocks.find(nBlockHeight);
    return it == mapDynodeBlocks.end() ? true : it->second.IsTransactionValid(txNew, nBlockHeight);
}

void CDynodePayments::CheckAndRemove()
{
    if (!dynodeSync.IsBlockchainSynced())
        return;

    LOCK2(cs_mapDynodeBlocks, cs_mapDynodePaymentVotes);

    int nLimit = GetStorageLimit();

    std::map<uint256, CDynodePaymentVote>::iterator it = mapDynodePaymentVotes.begin();
    while (it != mapDynodePaymentVotes.end()) {
        CDynodePaymentVote vote = (*it).second;

        if (nCachedBlockHeight - vote.nBlockHeight > nLimit) {
            LogPrint("dnpayments", "CDynodePayments::CheckAndRemove -- Removing old Dynode payment: nBlockHeight=%d\n", vote.nBlockHeight);
            mapDynodePaymentVotes.erase(it++);
            mapDynodeBlocks.erase(vote.nBlockHeight);
        } else {
            ++it;
        }
    }
    LogPrint("dnpayments", "CDynodePayments::CheckAndRemove -- %s\n", ToString());
}

bool CDynodePaymentVote::IsValid(CNode* pnode, int nValidationHeight, std::string& strError, CConnman& connman) const
{
    dynode_info_t dnInfo;

    if (!dnodeman.GetDynodeInfo(dynodeOutpoint, dnInfo)) {
        strError = strprintf("Unknown dynode=%s", dynodeOutpoint.ToStringShort());
        // Only ask if we are already synced and still have no idea about that Dynode
        if (dynodeSync.IsDynodeListSynced()) {
            dnodeman.AskForDN(pnode, dynodeOutpoint, connman);
        }

        return false;
    }

    int nMinRequiredProtocol;
    if (nBlockHeight >= nValidationHeight) {
        // new votes must comply SPORK_10_DYNODE_PAY_UPDATED_NODES rules
        nMinRequiredProtocol = dnpayments.GetMinDynodePaymentsProto();
    } else {
        // allow non-updated dynodes for old blocks
        nMinRequiredProtocol = MIN_DYNODE_PAYMENT_PROTO_VERSION_1;
    }

    if (dnInfo.nProtocolVersion < nMinRequiredProtocol) {
        strError = strprintf("Dynode protocol is too old: nProtocolVersion=%d, nMinRequiredProtocol=%d", dnInfo.nProtocolVersion, nMinRequiredProtocol);
        return false;
    }

    // Only dynodes should try to check dynode rank for old votes - they need to pick the right winner for future blocks.
    // Regular clients (miners included) need to verify dynode rank for future block votes only.
    if (!fDynodeMode && nBlockHeight < nValidationHeight)
        return true;

    int nRank;

    if (!dnodeman.GetDynodeRank(dynodeOutpoint, nRank, nBlockHeight - 101, nMinRequiredProtocol)) {
        LogPrint("dnpayments", "CDynodePaymentVote::IsValid -- Can't calculate rank for dynode %s\n",
            dynodeOutpoint.ToStringShort());
        return false;
    }

    if (nRank > DNPAYMENTS_SIGNATURES_TOTAL) {
        // It's common to have dynodes mistakenly think they are in the top 10
        // We don't want to print all of these messages in normal mode, debug mode should print though
        strError = strprintf("Dynode %s is not in the top %d (%d)", dynodeOutpoint.ToStringShort(), DNPAYMENTS_SIGNATURES_TOTAL, nRank);
        // Only ban for new dnw which is out of bounds, for old dnw DN list itself might be way too much off
        if (nRank > DNPAYMENTS_SIGNATURES_TOTAL * 2 && nBlockHeight > nValidationHeight) {
            LOCK(cs_main);
            strError = strprintf("Dynode %s is not in the top %d (%d)", dynodeOutpoint.ToStringShort(), DNPAYMENTS_SIGNATURES_TOTAL * 2, nRank);
            LogPrintf("CDynodePaymentVote::IsValid -- Error: %s\n", strError);
            Misbehaving(pnode->GetId(), 20);
        }
        // Still invalid however
        return false;
    }

    return true;
}

bool CDynodePayments::ProcessBlock(int nBlockHeight, CConnman& connman)
{
    // DETERMINE IF WE SHOULD BE VOTING FOR THE NEXT PAYEE

    if (fLiteMode || !fDynodeMode)
        return false;

    // We have little chances to pick the right winner if winners list is out of sync
    // but we have no choice, so we'll try. However it doesn't make sense to even try to do so
    // if we have not enough data about Dynodes.
    if (!dynodeSync.IsDynodeListSynced())
        return false;

    int nRank;

    if (!dnodeman.GetDynodeRank(activeDynode.outpoint, nRank, nBlockHeight - 101, GetMinDynodePaymentsProto())) {
        LogPrint("dnpayments", "CDynodePayments::ProcessBlock -- Unknown Dynode\n");
        return false;
    }

    if (nRank > DNPAYMENTS_SIGNATURES_TOTAL) {
        LogPrint("dnpayments", "CDynodePayments::ProcessBlock -- Dynode not in the top %d (%d)\n", DNPAYMENTS_SIGNATURES_TOTAL, nRank);
        return false;
    }


    // LOCATE THE NEXT DYNODE WHICH SHOULD BE PAID

    LogPrint("dnpayments", "CDynodePayments::ProcessBlock -- Start: nBlockHeight=%d, dynode=%s\n", nBlockHeight, activeDynode.outpoint.ToStringShort());

    // pay to the oldest DN that still had no payment but its input is old enough and it was active long enough
    int nCount = 0;
    dynode_info_t dnInfo;

    if (!dnodeman.GetNextDynodeInQueueForPayment(nBlockHeight, true, nCount, dnInfo)) {
        LogPrintf("CDynodePayments::ProcessBlock -- ERROR: Failed to find Dynode to pay\n");
        return false;
    }

    LogPrint("dnpayments", "CDynodePayments::ProcessBlock -- Dynode found by GetNextDynodeInQueueForPayment(): %s\n", dnInfo.outpoint.ToStringShort());

    CScript payee = GetScriptForDestination(dnInfo.pubKeyCollateralAddress.GetID());

    CDynodePaymentVote voteNew(activeDynode.outpoint, nBlockHeight, payee);

    CTxDestination address1;
    ExtractDestination(payee, address1);
    CDynamicAddress address2(address1);

    LogPrint("dnpayments", "CDynodePayments::ProcessBlock -- vote: payee=%s, nBlockHeight=%d\n", address2.ToString(), nBlockHeight);

    // SIGN MESSAGE TO NETWORK WITH OUR DYNODE KEYS

    LogPrint("dnpayments", "CDynodePayments::ProcessBlock -- Signing vote\n");
    if (voteNew.Sign()) {
        LogPrint("dnpayments", "CDynodePayments::ProcessBlock -- AddPaymentVote()\n");

        if (AddOrUpdatePaymentVote(voteNew)) {
            voteNew.Relay(connman);
            return true;
        }
    }

    return false;
}

void CDynodePayments::CheckBlockVotes(int nBlockHeight)
{
    if (!dynodeSync.IsWinnersListSynced())
        return;

    CDynodeMan::rank_pair_vec_t dns;
    if (!dnodeman.GetDynodeRanks(dns, nBlockHeight - 101, GetMinDynodePaymentsProto())) {
        LogPrintf("CDynodePayments::CheckBlockVotes -- nBlockHeight=%d, GetDynodeRanks failed\n", nBlockHeight);
        return;
    }

    std::string debugStr;

    debugStr += strprintf("CDynodePayments::CheckBlockVotes -- nBlockHeight=%d,\n  Expected voting DNs:\n", nBlockHeight);

    LOCK2(cs_mapDynodeBlocks, cs_mapDynodePaymentVotes);

    int i{0};
    for (const auto& dn : dns) {
        CScript payee;
        bool found = false;

        const auto it = mapDynodeBlocks.find(nBlockHeight);
        if (it != mapDynodeBlocks.end()) {
            for (const auto& p : it->second.vecPayees) {
                for (const auto& voteHash : p.GetVoteHashes()) {
                    const auto itVote = mapDynodePaymentVotes.find(voteHash);
                    if (itVote == mapDynodePaymentVotes.end()) {
                        debugStr += strprintf("    - could not find vote %s\n",
                            voteHash.ToString());
                        continue;
                    }
                    if (itVote->second.dynodeOutpoint == dn.second.outpoint) {
                        payee = itVote->second.payee;
                        found = true;
                        break;
                    }
                }
            }
        }

        if (found) {
            CTxDestination address1;
            ExtractDestination(payee, address1);
            CDynamicAddress address2(address1);

            debugStr += strprintf("    - %s - voted for %s\n",
                dn.second.outpoint.ToStringShort(), address2.ToString());
        } else {
            mapDynodesDidNotVote.emplace(dn.second.outpoint, 0).first->second++;

            debugStr += strprintf("    - %s - no vote received\n",
                dn.second.outpoint.ToStringShort());
        }

        if (++i >= DNPAYMENTS_SIGNATURES_TOTAL)
            break;
    }

    if (mapDynodesDidNotVote.empty()) {
        LogPrint("dnpayments", "%s", debugStr);
        return;
    }

    debugStr += "  Dynodes which missed a vote in the past:\n";
    for (const auto& item : mapDynodesDidNotVote) {
        debugStr += strprintf("    - %s: %d\n", item.first.ToStringShort(), item.second);
    }

    LogPrint("dnpayments", "%s", debugStr);
}

void CDynodePaymentVote::Relay(CConnman& connman) const
{
    // Do not relay until fully synced
    if (!dynodeSync.IsSynced()) {
        LogPrint("dnpayments", "CDynodePayments::Relay -- won't relay until fully synced\n");
        return;
    }

    CInv inv(MSG_DYNODE_PAYMENT_VOTE, GetHash());
    connman.RelayInv(inv);
}

bool CDynodePaymentVote::CheckSignature(const CPubKey& pubKeyDynode, int nValidationHeight, int& nDos) const
{
    // do not ban by default
    nDos = 0;
    std::string strError = "";

    if (sporkManager.IsSporkActive(SPORK_6_NEW_SIGS)) {
        uint256 hash = GetSignatureHash();

        if (!CHashSigner::VerifyHash(hash, pubKeyDynode, vchSig, strError)) {
            // could be a signature in old format
            std::string strMessage = dynodeOutpoint.ToStringShort() +
                                     boost::lexical_cast<std::string>(nBlockHeight) +
                                     ScriptToAsmStr(payee);
            if (!CMessageSigner::VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
                // nope, not in old format either
                // Only ban for future block vote when we are already synced.
                // Otherwise it could be the case when DN which signed this vote is using another key now
                // and we have no idea about the old one.
                if (dynodeSync.IsDynodeListSynced() && nBlockHeight > nValidationHeight) {
                    nDos = 20;
                }
                return error("CDynodePaymentVote::CheckSignature -- Got bad Dynode payment signature, dynode=%s, error: %s",
                    dynodeOutpoint.ToStringShort(), strError);
            }
        }
    } else {
        std::string strMessage = dynodeOutpoint.ToStringShort() +
                                 boost::lexical_cast<std::string>(nBlockHeight) +
                                 ScriptToAsmStr(payee);

        if (!CMessageSigner::VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
            // Only ban for future block vote when we are already synced.
            // Otherwise it could be the case when DN which signed this vote is using another key now
            // and we have no idea about the old one.
            if (dynodeSync.IsDynodeListSynced() && nBlockHeight > nValidationHeight) {
                nDos = 20;
            }
            return error("CDynodePaymentVote::CheckSignature -- Got bad Dynode payment signature, dynode=%s, error: %s",
                dynodeOutpoint.ToStringShort(), strError);
        }
    }

    return true;
}

std::string CDynodePaymentVote::ToString() const
{
    std::ostringstream info;

    info << dynodeOutpoint.ToStringShort() << ", " << nBlockHeight << ", " << ScriptToAsmStr(payee) << ", " << (int)vchSig.size();

    return info.str();
}

// Send all votes up to nCountNeeded blocks (but not more than GetStorageLimit)
void CDynodePayments::Sync(CNode* pnode, CConnman& connman) const
{
    LOCK(cs_mapDynodeBlocks);

    if (!dynodeSync.IsWinnersListSynced())
        return;

    int nInvCount = 0;

    for (int h = nCachedBlockHeight; h < nCachedBlockHeight + 20; h++) {
        const auto it = mapDynodeBlocks.find(h);
        if (it != mapDynodeBlocks.end()) {
            for (const auto& payee : it->second.vecPayees) {
                std::vector<uint256> vecVoteHashes = payee.GetVoteHashes();
                for (const auto& hash : vecVoteHashes) {
                    if (!HasVerifiedPaymentVote(hash))
                        continue;
                    pnode->PushInventory(CInv(MSG_DYNODE_PAYMENT_VOTE, hash));
                    nInvCount++;
                }
            }
        }
    }

    LogPrintf("CDynodePayments::Sync -- Sent %d votes to peer=%d\n", nInvCount, pnode->id);
    CNetMsgMaker msgMaker(pnode->GetSendVersion());
    connman.PushMessage(pnode, msgMaker.Make(NetMsgType::SYNCSTATUSCOUNT, DYNODE_SYNC_DNW, nInvCount));
}
// Request low data/unknown payment blocks in batches directly from some node instead of/after preliminary Sync.
void CDynodePayments::RequestLowDataPaymentBlocks(CNode* pnode, CConnman& connman) const
{
    if (!dynodeSync.IsDynodeListSynced())
        return;

    CNetMsgMaker msgMaker(pnode->GetSendVersion());
    LOCK2(cs_main, cs_mapDynodeBlocks);

    std::vector<CInv> vToFetch;
    int nLimit = GetStorageLimit();

    const CBlockIndex* pindex = chainActive.Tip();

    while (nCachedBlockHeight - pindex->nHeight < nLimit) {
        if (!mapDynodeBlocks.count(pindex->nHeight)) {
            // We have no idea about this block height, let's ask
            vToFetch.push_back(CInv(MSG_DYNODE_PAYMENT_BLOCK, pindex->GetBlockHash()));
            // We should not violate GETDATA rules
            if (vToFetch.size() == MAX_INV_SZ) {
                LogPrint("dnpayments", "CDynodePayments::SyncLowDataPaymentBlocks -- asking peer %d for %d blocks\n", pnode->id, MAX_INV_SZ);
                connman.PushMessage(pnode, msgMaker.Make(NetMsgType::GETDATA, vToFetch));
                // Start filling new batch
                vToFetch.clear();
            }
        }
        if (!pindex->pprev)
            break;
        pindex = pindex->pprev;
    }

    for (auto& dnBlockPayees : mapDynodeBlocks) {
        int nBlockHeight = dnBlockPayees.first;
        int nTotalVotes = 0;
        bool fFound = false;
        for (const auto& payee : dnBlockPayees.second.vecPayees) {
            if (payee.GetVoteCount() >= DNPAYMENTS_SIGNATURES_REQUIRED) {
                fFound = true;
                break;
            }
            nTotalVotes += payee.GetVoteCount();
        }
        // A clear winner (DNPAYMENTS_SIGNATURES_REQUIRED+ votes) was found
        // or no clear winner was found but there are at least avg number of votes
        if (fFound || nTotalVotes >= (DNPAYMENTS_SIGNATURES_TOTAL + DNPAYMENTS_SIGNATURES_REQUIRED) / 2) {
            // so just move to the next block
            continue;
        }
        // DEBUG
        DBG(
            // Let's see why this failed
            for (const auto& payee
                 : dnBlockPayees.second.vecPayees) {
                CTxDestination address1;
                ExtractDestination(payee.GetPayee(), address1);
                CDynamicAddress address2(address1);
                printf("payee %s votes %d\n", address2.ToString().c_str(), payee.GetVoteCount());
            } printf("block %d votes total %d\n", it->first, nTotalVotes);)
        // END DEBUG
        // Low data block found, let's try to sync it
        uint256 hash;
        if (GetBlockHash(hash, nBlockHeight)) {
            vToFetch.push_back(CInv(MSG_DYNODE_PAYMENT_BLOCK, hash));
        }
        // We should not violate GETDATA rules
        if (vToFetch.size() == MAX_INV_SZ) {
            LogPrint("dnpayments", "CDynodePayments::SyncLowDataPaymentBlocks -- asking peer %d for %d payment blocks\n", pnode->id, MAX_INV_SZ);
            connman.PushMessage(pnode, msgMaker.Make(NetMsgType::GETDATA, vToFetch));
            // Start filling new batch
            vToFetch.clear();
        }
    }
    // Ask for the rest of it
    if (!vToFetch.empty()) {
        LogPrint("dnpayments", "CDynodePayments::SyncLowDataPaymentBlocks -- asking peer %d for %d payment blocks\n", pnode->id, vToFetch.size());
        connman.PushMessage(pnode, msgMaker.Make(NetMsgType::GETDATA, vToFetch));
    }
}

std::string CDynodePayments::ToString() const
{
    std::ostringstream info;

    info << "Votes: " << (int)mapDynodePaymentVotes.size() << ", Blocks: " << (int)mapDynodeBlocks.size();

    return info.str();
}

bool CDynodePayments::IsEnoughData() const
{
    float nAverageVotes = (DNPAYMENTS_SIGNATURES_TOTAL + DNPAYMENTS_SIGNATURES_REQUIRED) / 2;
    int nStorageLimit = GetStorageLimit();
    return GetBlockCount() > nStorageLimit && GetVoteCount() > nStorageLimit * nAverageVotes;
}

int CDynodePayments::GetStorageLimit() const
{
    return std::max(int(dnodeman.size() * nStorageCoeff), nMinBlocksToStore);
}

void CDynodePayments::UpdatedBlockTip(const CBlockIndex* pindex, CConnman& connman)
{
    if (!pindex)
        return;

    nCachedBlockHeight = pindex->nHeight;
    LogPrint("dnpayments", "CDynodePayments::UpdatedBlockTip -- nCachedBlockHeight=%d\n", nCachedBlockHeight);

    int nFutureBlock = nCachedBlockHeight + 10;

    CheckBlockVotes(nFutureBlock - 1);
    ProcessBlock(nFutureBlock, connman);
}

void CDynodePayments::DoMaintenance()
{
    if (ShutdownRequested()) return;
     CheckAndRemove();
}
