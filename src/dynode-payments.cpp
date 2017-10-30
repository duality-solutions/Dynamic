// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dynode-payments.h"

#include "activedynode.h"
#include "policy/fees.h"
#include "governance-classes.h"
#include "dynode-sync.h"
#include "dynodeman.h"
#include "messagesigner.h"
#include "netfulfilledman.h"
#include "spork.h"
#include "util.h"

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

bool IsBlockValueValid(const CBlock& block, int nBlockHeight, CAmount blockReward,std::string &strErrorRet)
{
    strErrorRet = "";

    bool isBlockRewardValueMet = (block.vtx[0].GetValueOut() <= blockReward);
    if(fDebug) LogPrintf("block.vtx[0].GetValueOut() %lld <= blockReward %lld\n", block.vtx[0].GetValueOut(), blockReward);

    // we are still using budgets, but we have no data about them anymore,
    // all we know is predefined budget cycle and window

    const Consensus::Params& consensusParams = Params().GetConsensus();
     if(nBlockHeight < consensusParams.nSuperblockStartBlock) {
        int nOffset = nBlockHeight % consensusParams.nBudgetPaymentsCycleBlocks;
        if(nBlockHeight >= consensusParams.nBudgetPaymentsStartBlock &&
            nOffset < consensusParams.nBudgetPaymentsWindowBlocks) {
            if(dynodeSync.IsSynced() && !sporkManager.IsSporkActive(SPORK_13_OLD_SUPERBLOCK_FLAG)) {
                LogPrint("gobject", "IsBlockValueValid -- Client synced but budget spork is disabled, checking block value against block reward\n");
                if(!isBlockRewardValueMet) {
                    strErrorRet = strprintf("coinbase pays too much at height %d (actual=%d vs limit=%d), exceeded block reward, budgets are disabled",
                                            nBlockHeight, block.vtx[0].GetValueOut(), blockReward);
                }
                return isBlockRewardValueMet;
            }
            LogPrint("gobject", "IsBlockValueValid -- WARNING: Skipping budget block value checks, accepting block\n");
            // TODO: reprocess blocks to make sure they are legit?
            return true;
        }
        // LogPrint("gobject", "IsBlockValueValid -- Block is not in budget cycle window, checking block value against block reward\n");
        if(!isBlockRewardValueMet) {
            strErrorRet = strprintf("coinbase pays too much at height %d (actual=%d vs limit=%d), exceeded block reward, block is not in budget cycle window",
                                    nBlockHeight, block.vtx[0].GetValueOut(), blockReward);
        }
        return isBlockRewardValueMet;
    }

    // superblocks started

    CAmount nSuperblockMaxValue =  blockReward + CSuperblock::GetPaymentsLimit(nBlockHeight);
    bool isSuperblockMaxValueMet = (block.vtx[0].GetValueOut() <= nSuperblockMaxValue);

    LogPrint("gobject", "block.vtx[0].GetValueOut() %lld <= nSuperblockMaxValue %lld\n", block.vtx[0].GetValueOut(), nSuperblockMaxValue);

    if(!dynodeSync.IsSynced()) {
        // not enough data but at least it must NOT exceed superblock max value
        if(CSuperblock::IsValidBlockHeight(nBlockHeight)) {
            if(fDebug) LogPrintf("IsBlockPayeeValid -- WARNING: Client not synced, checking superblock max bounds only\n");
            if(!isSuperblockMaxValueMet) {
                strErrorRet = strprintf("coinbase pays too much at height %d (actual=%d vs limit=%d), exceeded superblock max value",
                                        nBlockHeight, block.vtx[0].GetValueOut(), nSuperblockMaxValue);
            }
            return isSuperblockMaxValueMet;
        }
        if(!isBlockRewardValueMet) {
            strErrorRet = strprintf("coinbase pays too much at height %d (actual=%d vs limit=%d), exceeded block reward, only regular blocks are allowed at this height",
                                    nBlockHeight, block.vtx[0].GetValueOut(), blockReward);
        }
        // it MUST be a regular block otherwise
        return isBlockRewardValueMet;
    }

    // we are synced, let's try to check as much data as we can

    if(sporkManager.IsSporkActive(SPORK_9_SUPERBLOCKS_ENABLED)) {
        // ONLY CHECK SUPERBLOCKS WHEN INITIALLY SYNCED AND CHECKING NEW BLOCK
        {
            // UP TO ONE HOUR OLD, OTHERWISE LONGEST CHAIN
            if(block.nTime + 60*60 < GetTime())
                return true;
        }

        if(CSuperblockManager::IsSuperblockTriggered(nBlockHeight)) {
            if(CSuperblockManager::IsValid(block.vtx[0], nBlockHeight, blockReward)) {
                LogPrint("gobject", "IsBlockValueValid -- Valid superblock at height %d: %s", nBlockHeight, block.vtx[0].ToString());
                // all checks are done in CSuperblock::IsValid, nothing to do here
                return true;
            }

            // triggered but invalid? that's weird
            LogPrintf("IsBlockValueValid -- ERROR: Invalid superblock detected at height %d: %s", nBlockHeight, block.vtx[0].ToString());
            // should NOT allow invalid superblocks, when superblocks are enabled
            strErrorRet = strprintf("invalid superblock detected at height %d", nBlockHeight);
            return false;
        }
        LogPrint("gobject", "IsBlockValueValid -- No triggered superblock detected at height %d\n", nBlockHeight);
        if(!isBlockRewardValueMet) {
            strErrorRet = strprintf("coinbase pays too much at height %d (actual=%d vs limit=%d), exceeded block reward, no triggered superblock detected",
                                    nBlockHeight, block.vtx[0].GetValueOut(), blockReward);
        }
    } else {
        // should NOT allow superblocks at all, when superblocks are disabled
        LogPrint("gobject", "IsBlockValueValid -- Superblocks are disabled, no superblocks allowed\n");
        if(!isBlockRewardValueMet) {
            strErrorRet = strprintf("coinbase pays too much at height %d (actual=%d vs limit=%d), exceeded block reward, superblocks are disabled",
                                    nBlockHeight, block.vtx[0].GetValueOut(), blockReward);
        }
    }

    // it MUST be a regular block
    return isBlockRewardValueMet;
}

bool IsBlockPayeeValid(const CTransaction& txNew, int nBlockHeight, CAmount blockReward)
{
    if(!dynodeSync.IsSynced()) {
        //there is no budget data to use to check anything, let's just accept the longest chain
        if(fDebug) LogPrintf("IsBlockPayeeValid -- WARNING: Client not synced, skipping block payee checks\n");
        return true;
    }

    // we are still using budgets, but we have no data about them anymore,
    // we can only check Dynode payments

    const Consensus::Params& consensusParams = Params().GetConsensus();

    if(nBlockHeight < consensusParams.nSuperblockStartBlock) {
        if(dnpayments.IsTransactionValid(txNew, nBlockHeight)) {
            LogPrint("dnpayments", "IsBlockPayeeValid -- Valid Dynode payment at height %d: %s", nBlockHeight, txNew.ToString());
            return true;
        }

        int nOffset = nBlockHeight % consensusParams.nBudgetPaymentsCycleBlocks;
        if(nBlockHeight >= consensusParams.nBudgetPaymentsStartBlock &&
            nOffset < consensusParams.nBudgetPaymentsWindowBlocks) {
            if(!sporkManager.IsSporkActive(SPORK_13_OLD_SUPERBLOCK_FLAG)) {
                // no budget blocks should be accepted here, if SPORK_13_OLD_SUPERBLOCK_FLAG is disabled
                LogPrint("gobject", "IsBlockPayeeValid -- ERROR: Client synced but budget spork is disabled and Dynode payment is invalid\n");
                return false;
            }
            // NOTE: this should never happen in real, SPORK_13_OLD_SUPERBLOCK_FLAG MUST be disabled when 12.1 starts to go live
            LogPrint("gobject", "IsBlockPayeeValid -- WARNING: Probably valid budget block, have no data, accepting\n");
            // TODO: reprocess blocks to make sure they are legit?
            return true;
        }

        if(sporkManager.IsSporkActive(SPORK_8_DYNODE_PAYMENT_ENFORCEMENT)) {
            LogPrintf("IsBlockPayeeValid -- ERROR: Invalid Dynode payment detected at height %d: %s", nBlockHeight, txNew.ToString());
            return false;
        }

        LogPrintf("IsBlockPayeeValid -- WARNING: Dynode payment enforcement is disabled, accepting any payee\n");
        return true;
    }

    // superblocks started
    // SEE IF THIS IS A VALID SUPERBLOCK

    if(!sporkManager.IsSporkActive(SPORK_9_SUPERBLOCKS_ENABLED)) {
        if(CSuperblockManager::IsSuperblockTriggered(nBlockHeight)) {
            if(CSuperblockManager::IsValid(txNew, nBlockHeight, blockReward)) {
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
    if(dnpayments.IsTransactionValid(txNew, nBlockHeight)) {
        LogPrint("dnpayments", "IsBlockPayeeValid -- Valid Dynode payment at height %d: %s", nBlockHeight, txNew.ToString());
        return true;
    }

    if(sporkManager.IsSporkActive(SPORK_8_DYNODE_PAYMENT_ENFORCEMENT)) {
        LogPrintf("IsBlockPayeeValid -- ERROR: Invalid Dynode payment detected at height %d: %s", nBlockHeight, txNew.ToString());
        return false;
    }

    LogPrintf("IsBlockPayeeValid -- WARNING: Dynode payment enforcement is disabled, accepting any payee\n");
    return true;
}

void FillBlockPayments(CMutableTransaction& txNew, int nBlockHeight, CAmount blockReward, CTxOut& txoutDynodeRet, std::vector<CTxOut>& voutSuperblockRet)
{
    // only create superblocks if spork is enabled AND if superblock is actually triggered
    // (height should be validated inside)
    if(sporkManager.IsSporkActive(SPORK_9_SUPERBLOCKS_ENABLED) &&
        CSuperblockManager::IsSuperblockTriggered(nBlockHeight)) {
            LogPrint("gobject", "FillBlockPayments -- triggered superblock creation at height %d\n", nBlockHeight);
            CSuperblockManager::CreateSuperblock(txNew, nBlockHeight, voutSuperblockRet);
            return;
    }

    if (chainActive.Height() > Params().GetConsensus().nDynodePaymentsStartBlock) {
        // FILL BLOCK PAYEE WITH DYNODE PAYMENT OTHERWISE
        dnpayments.FillBlockPayee(txNew);
        LogPrint("dnpayments", "FillBlockPayments -- nBlockHeight %d blockReward %lld txoutDynodeRet %s txNew %s",
                                nBlockHeight, blockReward, txoutDynodeRet.ToString(), txNew.ToString());
    }
}

std::string GetRequiredPaymentsString(int nBlockHeight)
{
    // IF WE HAVE A ACTIVATED TRIGGER FOR THIS HEIGHT - IT IS A SUPERBLOCK, GET THE REQUIRED PAYEES
    if(CSuperblockManager::IsSuperblockTriggered(nBlockHeight)) {
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

bool CDynodePayments::CanVote(COutPoint outDynode, int nBlockHeight)
{
    LOCK(cs_mapDynodePaymentVotes);

    if (mapDynodesLastVote.count(outDynode) && mapDynodesLastVote[outDynode] == nBlockHeight) {
        return false;
    }

    //record this Dynode voted
    mapDynodesLastVote[outDynode] = nBlockHeight;
    return true;
}

/**
*   FillBlockPayee
*
*   Fill Dynode ONLY payment block
*/

void CDynodePayments::FillBlockPayee(CMutableTransaction& txNew)
{
    CBlockIndex* pindexPrev = chainActive.Tip();       
    if(!pindexPrev) return;        

    bool hasPayment = true;
    CScript payee;

    if (chainActive.Height() <= Params().GetConsensus().nDynodePaymentsStartBlock){
            if (fDebug)
                LogPrintf("CreateNewBlock: No Dynode payments prior to block 20,546\n");
            hasPayment = false;
    }

    //spork
    if(!dnpayments.GetBlockPayee(pindexPrev->nHeight+1, payee)){       
        //no Dynode detected
        CDynode* winningNode = dnodeman.Find(payee);
        if(winningNode){
            payee = GetScriptForDestination(winningNode->pubKeyCollateralAddress.GetID());
        } else {
            if (fDebug)
                LogPrintf("CreateNewBlock: Failed to detect Dynode to pay\n");
            hasPayment = false;
        }
    }

    CAmount blockValue;
    CAmount dynodePayment;

    if (chainActive.Height() == 0) { blockValue = INITIAL_SUPERBLOCK_PAYMENT; }
    else if (chainActive.Height() >= 1 && chainActive.Height() <= Params().GetConsensus().nRewardsStart) { blockValue = BLOCKCHAIN_INIT_REWARD; }
    else if (chainActive.Height() > Params().GetConsensus().nRewardsStart) { blockValue = PHASE_1_POW_REWARD; }
    else { blockValue = BLOCKCHAIN_INIT_REWARD; }

    if (!hasPayment && hasPayment && chainActive.Height() <= Params().GetConsensus().nDynodePaymentsStartBlock) { dynodePayment = BLOCKCHAIN_INIT_REWARD; }
    else if (hasPayment && chainActive.Height() > Params().GetConsensus().nDynodePaymentsStartBlock && chainActive.Height() < Params().GetConsensus().nUpdateDiffAlgoHeight) {dynodePayment = PHASE_1_DYNODE_PAYMENT; }
    else if (hasPayment && chainActive.Height() > Params().GetConsensus().nDynodePaymentsStartBlock && chainActive.Height() >= Params().GetConsensus().nUpdateDiffAlgoHeight) {dynodePayment = PHASE_2_DYNODE_PAYMENT; }
    else { dynodePayment = BLOCKCHAIN_INIT_REWARD; }

    txNew.vout[0].nValue = blockValue;

    if(hasPayment){
        txNew.vout.resize(2);

        txNew.vout[1].scriptPubKey = payee;
        txNew.vout[1].nValue = dynodePayment;

        if (chainActive.Height() == 0) { txNew.vout[0].nValue = INITIAL_SUPERBLOCK_PAYMENT; }
        else if (chainActive.Height() >= 1 && chainActive.Height() <= Params().GetConsensus().nRewardsStart) { txNew.vout[0].nValue = BLOCKCHAIN_INIT_REWARD; }
        else if (chainActive.Height() > Params().GetConsensus().nRewardsStart) { txNew.vout[0].nValue = PHASE_1_POW_REWARD; }
        else { txNew.vout[0].nValue = BLOCKCHAIN_INIT_REWARD; }

        CTxDestination address1;
        ExtractDestination(payee, address1);
        CDynamicAddress address2(address1);

        LogPrintf("CDynodePayments::FillBlockPayee -- Dynode payment %lld to %s\n", dynodePayment, address2.ToString());
    }
}

int CDynodePayments::GetMinDynodePaymentsProto() {
    return sporkManager.IsSporkActive(SPORK_10_DYNODE_PAY_UPDATED_NODES);
}

void CDynodePayments::ProcessMessage(CNode* pfrom, std::string& strCommand, CDataStream& vRecv)
{
    // Ignore any payments messages until Dynode list is synced
    if(!dynodeSync.IsDynodeListSynced()) return;

    if(fLiteMode) return; // disable all Dynamic specific functionality

    if (strCommand == NetMsgType::DYNODEPAYMENTSYNC) { //Dynode Payments Request Sync

        // Ignore such requests until we are fully synced.
        // We could start processing this after Dynode list is synced
        // but this is a heavy one so it's better to finish sync first.
        if (!dynodeSync.IsSynced()) return;

        int nCountNeeded;
        vRecv >> nCountNeeded;
        
        int nDnCount = dnodeman.CountDynodes();

        if (nDnCount > 200) {
            if(netfulfilledman.HasFulfilledRequest(pfrom->addr, NetMsgType::DYNODEPAYMENTSYNC)) {
                // Asking for the payments list multiple times in a short period of time is no good
                LogPrintf("DYNODEPAYMENTSYNC -- peer already asked me for the list, peer=%d\n", pfrom->id);
                Misbehaving(pfrom->GetId(), 20);
                return;
            }
        }

        netfulfilledman.AddFulfilledRequest(pfrom->addr, NetMsgType::DYNODEPAYMENTSYNC);

        Sync(pfrom);
        LogPrintf("DYNODEPAYMENTSYNC -- Sent Dynode payment votes to peer %d\n", pfrom->id);

    } else if (strCommand == NetMsgType::DYNODEPAYMENTVOTE) { // Dynode Payments Vote for the Winner

        CDynodePaymentVote vote;
        vRecv >> vote;

        if(pfrom->nVersion < GetMinDynodePaymentsProto()) return;

        if(!pCurrentBlockIndex) return;

        uint256 nHash = vote.GetHash();

        pfrom->setAskFor.erase(nHash);

        {
            LOCK(cs_mapDynodePaymentVotes);
            if(mapDynodePaymentVotes.count(nHash)) {
                LogPrint("dnpayments", "DYNODEPAYMENTVOTE -- hash=%s, nHeight=%d seen\n", nHash.ToString(), pCurrentBlockIndex->nHeight);
                return;
            }

            // Avoid processing same vote multiple times
            mapDynodePaymentVotes[nHash] = vote;
            // but first mark vote as non-verified,
            // AddPaymentVote() below should take care of it if vote is actually ok
            mapDynodePaymentVotes[nHash].MarkAsNotVerified();
        }

        int nFirstBlock = pCurrentBlockIndex->nHeight - GetStorageLimit();
        if(vote.nBlockHeight < nFirstBlock || vote.nBlockHeight > pCurrentBlockIndex->nHeight+20) {
            LogPrint("dnpayments", "DYNODEPAYMENTVOTE -- vote out of range: nFirstBlock=%d, nBlockHeight=%d, nHeight=%d\n", nFirstBlock, vote.nBlockHeight, pCurrentBlockIndex->nHeight);
            return;
        }

        std::string strError = "";
        if(!vote.IsValid(pfrom, pCurrentBlockIndex->nHeight, strError)) {
            LogPrint("dnpayments", "DYNODEPAYMENTVOTE -- invalid message, error: %s\n", strError);
            return;
        }

        if(!CanVote(vote.vinDynode.prevout, vote.nBlockHeight)) {
            LogPrintf("DYNODEPAYMENTVOTE -- Dynode already voted, Dynode=%s\n", vote.vinDynode.prevout.ToStringShort());
            return;
        }

        dynode_info_t dnInfo = dnodeman.GetDynodeInfo(vote.vinDynode);
        if(!dnInfo.fInfoValid) {
            // dn was not found, so we can't check vote, some info is probably missing
            LogPrintf("DYNODEPAYMENTVOTE -- Dynode is missing %s\n", vote.vinDynode.prevout.ToStringShort());
            dnodeman.AskForDN(pfrom, vote.vinDynode);
            return;
        }

        int nDos = 0;
        if(!vote.CheckSignature(dnInfo.pubKeyDynode, pCurrentBlockIndex->nHeight, nDos)) {
            if(nDos) {
                LogPrintf("DYNODEPAYMENTVOTE -- ERROR: invalid signature\n");
                Misbehaving(pfrom->GetId(), nDos);
            } else {
                // only warn about anything non-critical (i.e. nDos == 0) in debug mode
                LogPrint("dnpayments", "DYNODEPAYMENTVOTE -- WARNING: invalid signature\n");
            }
            // Either our info or vote info could be outdated.
            // In case our info is outdated, ask for an update,
            dnodeman.AskForDN(pfrom, vote.vinDynode);
            // but there is nothing we can do if vote info itself is outdated
            // (i.e. it was signed by a DN which changed its key),
            // so just quit here.
            return;
        }

        CTxDestination address1;
        ExtractDestination(vote.payee, address1);
        CDynamicAddress address2(address1);

        LogPrint("dnpayments", "DYNODEPAYMENTVOTE -- vote: address=%s, nBlockHeight=%d, nHeight=%d, prevout=%s\n", address2.ToString(), vote.nBlockHeight, pCurrentBlockIndex->nHeight, vote.vinDynode.prevout.ToStringShort());

        if(AddPaymentVote(vote)){
            vote.Relay();
            dynodeSync.AddedPaymentVote();
        }
    }
}

bool CDynodePaymentVote::Sign()
{
    std::string strError;
    std::string strMessage = vinDynode.prevout.ToStringShort() +
                boost::lexical_cast<std::string>(nBlockHeight) +
                ScriptToAsmStr(payee);

    if(!CMessageSigner::SignMessage(strMessage, vchSig, activeDynode.keyDynode)) {
        LogPrintf("CDynodePaymentVote::Sign -- SignMessage() failed\n");
        return false;
    }

    if(!CMessageSigner::VerifyMessage(activeDynode.pubKeyDynode, vchSig, strMessage, strError)) {
        LogPrintf("CDynodePaymentVote::Sign -- VerifyMessage() failed, error: %s\n", strError);
        return false;
    }

    return true;
}

bool CDynodePayments::GetBlockPayee(int nBlockHeight, CScript& payee)
{
    if(mapDynodeBlocks.count(nBlockHeight)){
        return mapDynodeBlocks[nBlockHeight].GetBestPayee(payee);
    }

    return false;
}

// Is this Dynode scheduled to get paid soon?
// -- Only look ahead up to 8 blocks to allow for propagation of the latest 2 blocks of votes
bool CDynodePayments::IsScheduled(CDynode& dn, int nNotBlockHeight)
{
    LOCK(cs_mapDynodeBlocks);

    if(!pCurrentBlockIndex) return false;

    CScript dnpayee;
    dnpayee = GetScriptForDestination(dn.pubKeyCollateralAddress.GetID());

    CScript payee;
    for(int64_t h = pCurrentBlockIndex->nHeight; h <= pCurrentBlockIndex->nHeight + 8; h++){
        if(h == nNotBlockHeight) continue;
        if(mapDynodeBlocks.count(h) && mapDynodeBlocks[h].GetBestPayee(payee) && dnpayee == payee) {
            return true;
        }
    }

    return false;
}

bool CDynodePayments::AddPaymentVote(const CDynodePaymentVote& vote)
{
    uint256 blockHash = uint256();
    if(!GetBlockHash(blockHash, vote.nBlockHeight - 101)) return false;

    if(HasVerifiedPaymentVote(vote.GetHash())) return false;

    LOCK2(cs_mapDynodeBlocks, cs_mapDynodePaymentVotes);

    mapDynodePaymentVotes[vote.GetHash()] = vote;

    if(!mapDynodeBlocks.count(vote.nBlockHeight)) {
       CDynodeBlockPayees blockPayees(vote.nBlockHeight);
       mapDynodeBlocks[vote.nBlockHeight] = blockPayees;
    }

    mapDynodeBlocks[vote.nBlockHeight].AddPayee(vote);

    return true;
}

bool CDynodePayments::HasVerifiedPaymentVote(uint256 hashIn)
{
    LOCK(cs_mapDynodePaymentVotes);
    std::map<uint256, CDynodePaymentVote>::iterator it = mapDynodePaymentVotes.find(hashIn);
    return it != mapDynodePaymentVotes.end() && it->second.IsVerified();
}

void CDynodeBlockPayees::AddPayee(const CDynodePaymentVote& vote)
{
    LOCK(cs_vecPayees);

    BOOST_FOREACH(CDynodePayee& payee, vecPayees) {
        if (payee.GetPayee() == vote.payee) {
            payee.AddVoteHash(vote.GetHash());
            return;
        }
    }
    CDynodePayee payeeNew(vote.payee, vote.GetHash());
    vecPayees.push_back(payeeNew);
}

bool CDynodeBlockPayees::GetBestPayee(CScript& payeeRet)
{
    LOCK(cs_vecPayees);

    if(!vecPayees.size()) {
        LogPrint("dnpayments", "CDynodeBlockPayees::GetBestPayee -- ERROR: couldn't find any payee\n");
        return false;
    }

    int nVotes = -1;
    BOOST_FOREACH(CDynodePayee& payee, vecPayees) {
        if (payee.GetVoteCount() > nVotes) {
            payeeRet = payee.GetPayee();
            nVotes = payee.GetVoteCount();
        }
    }

    return (nVotes > -1);
}

bool CDynodeBlockPayees::HasPayeeWithVotes(CScript payeeIn, int nVotesReq)
{
    LOCK(cs_vecPayees);

    BOOST_FOREACH(CDynodePayee& payee, vecPayees) {
        if (payee.GetVoteCount() >= nVotesReq && payee.GetPayee() == payeeIn) {
            return true;
        }
    }

    LogPrint("dnpayments", "CDynodeBlockPayees::HasPayeeWithVotes -- ERROR: couldn't find any payee with %d+ votes\n", nVotesReq);
    return false;
}

bool CDynodeBlockPayees::IsTransactionValid(const CTransaction& txNew)
{
    LOCK(cs_vecPayees);

    int nMaxSignatures = 0;
    std::string strPayeesPossible = "";

    CAmount nDynodePayment = GetDynodePayment();

    //require at least DNPAYMENTS_SIGNATURES_REQUIRED signatures

    BOOST_FOREACH(CDynodePayee& payee, vecPayees) {
        if (payee.GetVoteCount() >= nMaxSignatures) {
            nMaxSignatures = payee.GetVoteCount();
        }
    }

    // if we don't have at least DNPAYMENTS_SIGNATURES_REQUIRED signatures on a payee, approve whichever is the longest chain
    if(nMaxSignatures < DNPAYMENTS_SIGNATURES_REQUIRED) return true;

    BOOST_FOREACH(CDynodePayee& payee, vecPayees) {
        if (payee.GetVoteCount() >= DNPAYMENTS_SIGNATURES_REQUIRED) {
            BOOST_FOREACH(CTxOut txout, txNew.vout) {
                if (payee.GetPayee() == txout.scriptPubKey && nDynodePayment == txout.nValue) {
                    LogPrint("dnpayments", "CDynodeBlockPayees::IsTransactionValid -- Found required payment\n");
                    return true;
                }
            }

            CTxDestination address1;
            ExtractDestination(payee.GetPayee(), address1);
            CDynamicAddress address2(address1);

            if(strPayeesPossible == "") {
                strPayeesPossible = address2.ToString();
            } else {
                strPayeesPossible += "," + address2.ToString();
            }
        }
    }

    LogPrintf("CDynodeBlockPayees::IsTransactionValid -- ERROR: Missing required payment, possible payees: '%s', amount: %f DYN\n", strPayeesPossible, (float)nDynodePayment/COIN);
    return false;
}

std::string CDynodeBlockPayees::GetRequiredPaymentsString()
{
    LOCK(cs_vecPayees);

    std::string strRequiredPayments = "Unknown";

    BOOST_FOREACH(CDynodePayee& payee, vecPayees)
    {
        CTxDestination address1;
        ExtractDestination(payee.GetPayee(), address1);
        CDynamicAddress address2(address1);

        if (strRequiredPayments != "Unknown") {
            strRequiredPayments += ", " + address2.ToString() + ":" + boost::lexical_cast<std::string>(payee.GetVoteCount());
        } else {
            strRequiredPayments = address2.ToString() + ":" + boost::lexical_cast<std::string>(payee.GetVoteCount());
        }
    }

    return strRequiredPayments;
}

std::string CDynodePayments::GetRequiredPaymentsString(int nBlockHeight)
{
    LOCK(cs_mapDynodeBlocks);

    if(mapDynodeBlocks.count(nBlockHeight)){
        return mapDynodeBlocks[nBlockHeight].GetRequiredPaymentsString();
    }

    return "Unknown";
}

bool CDynodePayments::IsTransactionValid(const CTransaction& txNew, int nBlockHeight)
{
    LOCK(cs_mapDynodeBlocks);

    if(mapDynodeBlocks.count(nBlockHeight)){
        return mapDynodeBlocks[nBlockHeight].IsTransactionValid(txNew);
    }

    return true;
}

void CDynodePayments::CheckAndRemove()
{
    if(!pCurrentBlockIndex) return;

    LOCK2(cs_mapDynodeBlocks, cs_mapDynodePaymentVotes);

    int nLimit = GetStorageLimit();

    std::map<uint256, CDynodePaymentVote>::iterator it = mapDynodePaymentVotes.begin();
    while(it != mapDynodePaymentVotes.end()) {
        CDynodePaymentVote vote = (*it).second;

        if(pCurrentBlockIndex->nHeight - vote.nBlockHeight > nLimit) {
            LogPrint("dnpayments", "CDynodePayments::CheckAndRemove -- Removing old Dynode payment: nBlockHeight=%d\n", vote.nBlockHeight);
            mapDynodePaymentVotes.erase(it++);
            mapDynodeBlocks.erase(vote.nBlockHeight);
        } else {
            ++it;
        }
    }
    LogPrintf("CDynodePayments::CheckAndRemove -- %s\n", ToString());
}

bool CDynodePaymentVote::IsValid(CNode* pnode, int nValidationHeight, std::string& strError)
{
    CDynode* pdn = dnodeman.Find(vinDynode);

    if(!pdn) {
        strError = strprintf("Unknown Dynode: prevout=%s", vinDynode.prevout.ToStringShort());
        // Only ask if we are already synced and still have no idea about that Dynode
        if(dynodeSync.IsDynodeListSynced()) {
            dnodeman.AskForDN(pnode, vinDynode);
        }

        return false;
    }

    int nMinRequiredProtocol;
    if(nBlockHeight >= nValidationHeight) {
        // new votes must comply SPORK_10_DYNODE_PAY_UPDATED_NODES rules
        nMinRequiredProtocol = dnpayments.GetMinDynodePaymentsProto();
    } else {
        // allow non-updated Dynodes for old blocks
        nMinRequiredProtocol = MIN_DYNODE_PAYMENT_PROTO_VERSION;
    }

    if(pdn->nProtocolVersion < nMinRequiredProtocol) {
        strError = strprintf("Dynode protocol is too old: nProtocolVersion=%d, nMinRequiredProtocol=%d", pdn->nProtocolVersion, nMinRequiredProtocol);
        return false;
    }

    // Only Dynodes should try to check Dynode rank for old votes - they need to pick the right winner for future blocks.
    // Regular clients (miners included) need to verify Dynode rank for future block votes only.
    if(!fDyNode && nBlockHeight < nValidationHeight) return true;

    int nRank = dnodeman.GetDynodeRank(vinDynode, nBlockHeight - 101, nMinRequiredProtocol, false);

    if(nRank > DNPAYMENTS_SIGNATURES_TOTAL) {
        // It's common to have Dynodes mistakenly think they are in the top 10
        // We don't want to print all of these messages in normal mode, debug mode should print though
        strError = strprintf("Dynode is not in the top %d (%d)", DNPAYMENTS_SIGNATURES_TOTAL, nRank);
        // Only ban for new dnw which is out of bounds, for old dnw DN list itself might be way too much off
        if(nRank > DNPAYMENTS_SIGNATURES_TOTAL*2 && nBlockHeight > nValidationHeight) {
            strError = strprintf("Dynode is not in the top %d (%d)", DNPAYMENTS_SIGNATURES_TOTAL*2, nRank);
            LogPrintf("CDynodePaymentVote::IsValid -- Error: %s\n", strError);
            Misbehaving(pnode->GetId(), 20);
        }
        // Still invalid however
        return false;
    }

    return true;
}

bool CDynodePayments::ProcessBlock(int nBlockHeight)
{
    // DETERMINE IF WE SHOULD BE VOTING FOR THE NEXT PAYEE

    if(fLiteMode || !fDyNode) return false;

    // We have little chances to pick the right winner if winners list is out of sync
    // but we have no choice, so we'll try. However it doesn't make sense to even try to do so
    // if we have not enough data about Dynodes.
    if(!dynodeSync.IsDynodeListSynced()) return false;

    int nRank = dnodeman.GetDynodeRank(activeDynode.vin, nBlockHeight - 101, GetMinDynodePaymentsProto(), false);

    if (nRank == -1) {
        LogPrint("dnpayments", "CDynodePayments::ProcessBlock -- Unknown Dynode\n");
        return false;
    }

    if (nRank > DNPAYMENTS_SIGNATURES_TOTAL) {
        LogPrint("dnpayments", "CDynodePayments::ProcessBlock -- Dynode not in the top %d (%d)\n", DNPAYMENTS_SIGNATURES_TOTAL, nRank);
        return false;
    }


    // LOCATE THE NEXT DYNODE WHICH SHOULD BE PAID

    LogPrintf("CDynodePayments::ProcessBlock -- Start: nBlockHeight=%d, Dynode=%s\n", nBlockHeight, activeDynode.vin.prevout.ToStringShort());

    // pay to the oldest DN that still had no payment but its input is old enough and it was active long enough
    int nCount = 0;
    CDynode *pdn = dnodeman.GetNextDynodeInQueueForPayment(nBlockHeight, true, nCount);

    if (pdn == NULL) {
        LogPrintf("CDynodePayments::ProcessBlock -- ERROR: Failed to find Dynode to pay\n");
        return false;
    }

    LogPrintf("CDynodePayments::ProcessBlock -- Dynode found by GetNextDynodeInQueueForPayment(): %s\n", pdn->vin.prevout.ToStringShort());


    CScript payee = GetScriptForDestination(pdn->pubKeyCollateralAddress.GetID());

    CDynodePaymentVote voteNew(activeDynode.vin, nBlockHeight, payee);

    CTxDestination address1;
    ExtractDestination(payee, address1);
    CDynamicAddress address2(address1);

    LogPrintf("CDynodePayments::ProcessBlock -- vote: payee=%s, nBlockHeight=%d\n", address2.ToString(), nBlockHeight);

    // SIGN MESSAGE TO NETWORK WITH OUR DYNODE KEYS

    LogPrintf("CDynodePayments::ProcessBlock -- Signing vote\n");
    if (voteNew.Sign()) {
        LogPrintf("CDynodePayments::ProcessBlock -- AddPaymentVote()\n");

        if (AddPaymentVote(voteNew)) {
            voteNew.Relay();
            return true;
        }
    }

    return false;
}

void CDynodePaymentVote::Relay()
{
    // do not relay until synced
    if (!dynodeSync.IsWinnersListSynced()) return;
    CInv inv(MSG_DYNODE_PAYMENT_VOTE, GetHash());
    RelayInv(inv);
}

bool CDynodePaymentVote::CheckSignature(const CPubKey& pubKeyDynode, int nValidationHeight, int &nDos)
{
    // do not ban by default
    nDos = 0;

    std::string strMessage = vinDynode.prevout.ToStringShort() +
                boost::lexical_cast<std::string>(nBlockHeight) +
                ScriptToAsmStr(payee);

    std::string strError = "";
    if (!CMessageSigner::VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
        // Only ban for future block vote when we are already synced.
        // Otherwise it could be the case when DN which signed this vote is using another key now
        // and we have no idea about the old one.
        if(dynodeSync.IsDynodeListSynced() && nBlockHeight > nValidationHeight) {
            nDos = 20;
        }
        return error("CDynodePaymentVote::CheckSignature -- Got bad Dynode payment signature, Dynode=%s, error: %s", vinDynode.prevout.ToStringShort().c_str(), strError);
    }

    return true;
}

std::string CDynodePaymentVote::ToString() const
{
    std::ostringstream info;

    info << vinDynode.prevout.ToStringShort() <<
            ", " << nBlockHeight <<
            ", " << ScriptToAsmStr(payee) <<
            ", " << (int)vchSig.size();

    return info.str();
}

// Send all votes up to nCountNeeded blocks (but not more than GetStorageLimit)        
void CDynodePayments::Sync(CNode* pnode)
{
    LOCK(cs_mapDynodeBlocks);

    if(!pCurrentBlockIndex) return;

    int nInvCount = 0;

    for(int h = pCurrentBlockIndex->nHeight; h < pCurrentBlockIndex->nHeight + 20; h++) {
        if(mapDynodeBlocks.count(h)) {
            BOOST_FOREACH(CDynodePayee& payee, mapDynodeBlocks[h].vecPayees) {
                std::vector<uint256> vecVoteHashes = payee.GetVoteHashes();
                BOOST_FOREACH(uint256& hash, vecVoteHashes) {
                    if(!HasVerifiedPaymentVote(hash)) continue;
                    pnode->PushInventory(CInv(MSG_DYNODE_PAYMENT_VOTE, hash));
                    nInvCount++;
                }
            }
        }
    }

    LogPrintf("CDynodePayments::Sync -- Sent %d votes to peer %d\n", nInvCount, pnode->id);
    pnode->PushMessage(NetMsgType::SYNCSTATUSCOUNT, DYNODE_SYNC_DNW, nInvCount);
}

// Request low data/unknown payment blocks in batches directly from some node instead of/after preliminary Sync.
void CDynodePayments::RequestLowDataPaymentBlocks(CNode* pnode)
{
    if(!pCurrentBlockIndex) return;

    LOCK2(cs_main, cs_mapDynodeBlocks);

    std::vector<CInv> vToFetch;
    int nLimit = GetStorageLimit();

    const CBlockIndex *pindex = pCurrentBlockIndex;

    while(pCurrentBlockIndex->nHeight - pindex->nHeight < nLimit) {
        if(!mapDynodeBlocks.count(pindex->nHeight)) {
            // We have no idea about this block height, let's ask
            vToFetch.push_back(CInv(MSG_DYNODE_PAYMENT_BLOCK, pindex->GetBlockHash()));
            // We should not violate GETDATA rules
            if(vToFetch.size() == MAX_INV_SZ) {
                LogPrintf("CDynodePayments::SyncLowDataPaymentBlocks -- asking peer %d for %d blocks\n", pnode->id, MAX_INV_SZ);
                pnode->PushMessage(NetMsgType::GETDATA, vToFetch);
                // Start filling new batch
                vToFetch.clear();
            }
        }
        if(!pindex->pprev) break;
        pindex = pindex->pprev;
    }

    std::map<int, CDynodeBlockPayees>::iterator it = mapDynodeBlocks.begin();

    while(it != mapDynodeBlocks.end()) {
        int nTotalVotes = 0;
        bool fFound = false;
        BOOST_FOREACH(CDynodePayee& payee, it->second.vecPayees) {
            if(payee.GetVoteCount() >= DNPAYMENTS_SIGNATURES_REQUIRED) {
                fFound = true;
                break;
            }
            nTotalVotes += payee.GetVoteCount();
        }
        // A clear winner (DNPAYMENTS_SIGNATURES_REQUIRED+ votes) was found
        // or no clear winner was found but there are at least avg number of votes
        if(fFound || nTotalVotes >= (DNPAYMENTS_SIGNATURES_TOTAL + DNPAYMENTS_SIGNATURES_REQUIRED)/2) {
            // so just move to the next block
            ++it;
            continue;
        }
        // DEBUG
        DBG (
            // Let's see why this failed
            BOOST_FOREACH(CDynodePayee& payee, it->second.vecPayees) {
                CTxDestination address1;
                ExtractDestination(payee.GetPayee(), address1);
                CDynamicAddress address2(address1);
                printf("payee %s votes %d\n", address2.ToString().c_str(), payee.GetVoteCount());
            }
            printf("block %d votes total %d\n", it->first, nTotalVotes);
        )
        // END DEBUG
        // Low data block found, let's try to sync it
        uint256 hash;
        if(GetBlockHash(hash, it->first)) {
            vToFetch.push_back(CInv(MSG_DYNODE_PAYMENT_BLOCK, hash));
        }
        // We should not violate GETDATA rules
        if(vToFetch.size() == MAX_INV_SZ) {
            LogPrintf("CDynodePayments::SyncLowDataPaymentBlocks -- asking peer %d for %d payment blocks\n", pnode->id, MAX_INV_SZ);
            pnode->PushMessage(NetMsgType::GETDATA, vToFetch);
            // Start filling new batch
            vToFetch.clear();
        }
        ++it;
    }
    // Ask for the rest of it
    if(!vToFetch.empty()) {
        LogPrintf("CDynodePayments::SyncLowDataPaymentBlocks -- asking peer %d for %d payment blocks\n", pnode->id, vToFetch.size());
        pnode->PushMessage(NetMsgType::GETDATA, vToFetch);
    }
}

std::string CDynodePayments::ToString() const
{
    std::ostringstream info;

    info << "Votes: " << (int)mapDynodePaymentVotes.size() <<
            ", Blocks: " << (int)mapDynodeBlocks.size();

    return info.str();
}

bool CDynodePayments::IsEnoughData()
{
    float nAverageVotes = (DNPAYMENTS_SIGNATURES_TOTAL + DNPAYMENTS_SIGNATURES_REQUIRED) / 2;
    int nStorageLimit = GetStorageLimit();
    return GetBlockCount() > nStorageLimit && GetVoteCount() > nStorageLimit * nAverageVotes;
}

int CDynodePayments::GetStorageLimit()
{
    return std::max(int(dnodeman.size() * nStorageCoeff), nMinBlocksToStore);
}

void CDynodePayments::UpdatedBlockTip(const CBlockIndex *pindex)
{
    pCurrentBlockIndex = pindex;
    LogPrint("dnpayments", "CDynodePayments::UpdatedBlockTip -- pCurrentBlockIndex->nHeight=%d\n", pCurrentBlockIndex->nHeight);

    ProcessBlock(pindex->nHeight + 10);
}
