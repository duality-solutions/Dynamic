// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "activestormnode.h"
#include "privatesend.h"
#include "governance-classes.h"
#include "policy/fees.h"
#include "stormnode-payments.h"
#include "stormnode-sync.h"
#include "stormnodeman.h"
#include "netfulfilledman.h"
#include "spork.h"
#include "util.h"

#include <boost/lexical_cast.hpp>

/** Object for who's going to get paid on which blocks */
CStormnodePayments snpayments;

CCriticalSection cs_vecPayees;
CCriticalSection cs_mapStormnodeBlocks;
CCriticalSection cs_mapStormnodePaymentVotes;

/**
* IsBlockValueValid
*
*   Determine if coinbase outgoing created money is the correct value
*
*   Why is this needed?
*   - In DarkSilk some blocks are superblocks, which output much higher amounts of coins
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
            if(stormnodeSync.IsSynced() && !sporkManager.IsSporkActive(SPORK_13_OLD_SUPERBLOCK_FLAG)) {
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

    if(!stormnodeSync.IsSynced()) {
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
    if(!stormnodeSync.IsSynced()) {
        //there is no budget data to use to check anything, let's just accept the longest chain
        if(fDebug) LogPrintf("IsBlockPayeeValid -- WARNING: Client not synced, skipping block payee checks\n");
        return true;
    }

    // we are still using budgets, but we have no data about them anymore,
    // we can only check Stormnode payments

    const Consensus::Params& consensusParams = Params().GetConsensus();

    if(nBlockHeight < consensusParams.nSuperblockStartBlock) {
        if(snpayments.IsTransactionValid(txNew, nBlockHeight)) {
            LogPrint("snpayments", "IsBlockPayeeValid -- Valid Stormnode payment at height %d: %s", nBlockHeight, txNew.ToString());
            return true;
        }

        int nOffset = nBlockHeight % consensusParams.nBudgetPaymentsCycleBlocks;
        if(nBlockHeight >= consensusParams.nBudgetPaymentsStartBlock &&
            nOffset < consensusParams.nBudgetPaymentsWindowBlocks) {
            if(!sporkManager.IsSporkActive(SPORK_13_OLD_SUPERBLOCK_FLAG)) {
                // no budget blocks should be accepted here, if SPORK_13_OLD_SUPERBLOCK_FLAG is disabled
                LogPrint("gobject", "IsBlockPayeeValid -- ERROR: Client synced but budget spork is disabled and Stormnode payment is invalid\n");
                return false;
            }
            // NOTE: this should never happen in real, SPORK_13_OLD_SUPERBLOCK_FLAG MUST be disabled when 12.1 starts to go live
            LogPrint("gobject", "IsBlockPayeeValid -- WARNING: Probably valid budget block, have no data, accepting\n");
            // TODO: reprocess blocks to make sure they are legit?
            return true;
        }

        if(sporkManager.IsSporkActive(SPORK_8_STORMNODE_PAYMENT_ENFORCEMENT)) {
            LogPrintf("IsBlockPayeeValid -- ERROR: Invalid Stormnode payment detected at height %d: %s", nBlockHeight, txNew.ToString());
            return false;
        }

        LogPrintf("IsBlockPayeeValid -- WARNING: Stormnode payment enforcement is disabled, accepting any payee\n");
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
        // continue validation, should pay SN
        LogPrint("gobject", "IsBlockPayeeValid -- No triggered superblock detected at height %d\n", nBlockHeight);
    } else {
        // should NOT allow superblocks at all, when superblocks are disabled
        LogPrint("gobject", "IsBlockPayeeValid -- Superblocks are disabled, no superblocks allowed\n");
    }

    // IF THIS ISN'T A SUPERBLOCK OR SUPERBLOCK IS INVALID, IT SHOULD PAY A STORMNODE DIRECTLY
    if(snpayments.IsTransactionValid(txNew, nBlockHeight)) {
        LogPrint("snpayments", "IsBlockPayeeValid -- Valid Stormnode payment at height %d: %s", nBlockHeight, txNew.ToString());
        return true;
    }

    if(sporkManager.IsSporkActive(SPORK_8_STORMNODE_PAYMENT_ENFORCEMENT)) {
        LogPrintf("IsBlockPayeeValid -- ERROR: Invalid Stormnode payment detected at height %d: %s", nBlockHeight, txNew.ToString());
        return false;
    }

    LogPrintf("IsBlockPayeeValid -- WARNING: Stormnode payment enforcement is disabled, accepting any payee\n");
    return true;
}

void FillBlockPayments(CMutableTransaction& txNew, int nBlockHeight, CAmount blockReward, CTxOut& txoutStormnodeRet, std::vector<CTxOut>& voutSuperblockRet)
{
    // only create superblocks if spork is enabled AND if superblock is actually triggered
    // (height should be validated inside)
    if(sporkManager.IsSporkActive(SPORK_9_SUPERBLOCKS_ENABLED) &&
        CSuperblockManager::IsSuperblockTriggered(nBlockHeight)) {
            LogPrint("gobject", "FillBlockPayments -- triggered superblock creation at height %d\n", nBlockHeight);
            CSuperblockManager::CreateSuperblock(txNew, nBlockHeight, voutSuperblockRet);
            return;
    }

    // FILL BLOCK PAYEE WITH STORMNODE PAYMENT OTHERWISE
    snpayments.FillBlockPayee(txNew);
    LogPrint("snpayments", "FillBlockPayments -- nBlockHeight %d blockReward %lld txoutStormnodeRet %s txNew %s",
                            nBlockHeight, blockReward, txoutStormnodeRet.ToString(), txNew.ToString());
}

std::string GetRequiredPaymentsString(int nBlockHeight)
{
    // IF WE HAVE A ACTIVATED TRIGGER FOR THIS HEIGHT - IT IS A SUPERBLOCK, GET THE REQUIRED PAYEES
    if(CSuperblockManager::IsSuperblockTriggered(nBlockHeight)) {
        return CSuperblockManager::GetRequiredPaymentsString(nBlockHeight);
    }

    // OTHERWISE, PAY STORMNODE
    return snpayments.GetRequiredPaymentsString(nBlockHeight);
}

void CStormnodePayments::Clear()
{
    LOCK2(cs_mapStormnodeBlocks, cs_mapStormnodePaymentVotes);
    mapStormnodeBlocks.clear();
    mapStormnodePaymentVotes.clear();
}

bool CStormnodePayments::CanVote(COutPoint outStormnode, int nBlockHeight)
{
    LOCK(cs_mapStormnodePaymentVotes);

    if (mapStormnodesLastVote.count(outStormnode) && mapStormnodesLastVote[outStormnode] == nBlockHeight) {
        return false;
    }

    //record this Stormnode voted
    mapStormnodesLastVote[outStormnode] = nBlockHeight;
    return true;
}

/**
*   FillBlockPayee
*
*   Fill Stormnode ONLY payment block
*/

void CStormnodePayments::FillBlockPayee(CMutableTransaction& txNew /*CAmount nFees*/)  // TODO GB : Add fees
{
    CBlockIndex* pindexPrev = chainActive.Tip();       
    if(!pindexPrev) return;        

    bool hasPayment = true;
    CScript payee;

    //spork
    if(!snpayments.GetBlockPayee(pindexPrev->nHeight+1, payee)){       
        //no Stormnode detected
        CStormnode* winningNode = snodeman.Find(payee);
        if(winningNode){
            payee = GetScriptForDestination(winningNode->pubKeyCollateralAddress.GetID());
        } else {
            if (fDebug)
                LogPrintf("CreateNewBlock: Failed to detect Stormnode to pay\n");
            hasPayment = false;
        }
    }

    CAmount blockValue;
    CAmount stormnodePayment;

    if (chainActive.Height() == 0) { blockValue = 4000000 * COIN; }
    else if (chainActive.Height() >= 1 && chainActive.Height() <= Params().StartStormnodePayments()) { blockValue = BLOCKCHAIN_INIT_REWARD; }
    else { blockValue = STATIC_POW_REWARD; }

    if (!hasPayment && chainActive.Height() < Params().StartStormnodePayments()) { stormnodePayment = BLOCKCHAIN_INIT_REWARD; }
    else { stormnodePayment = STATIC_STORMNODE_PAYMENT; }

    txNew.vout[0].nValue = blockValue;

    if(hasPayment){
        txNew.vout.resize(2);

        txNew.vout[1].scriptPubKey = payee;
        txNew.vout[1].nValue = stormnodePayment;

        txNew.vout[0].nValue = STATIC_POW_REWARD;

        CTxDestination address1;
        ExtractDestination(payee, address1);
        CDarkSilkAddress address2(address1);

        LogPrintf("CStormnodePayments::FillBlockPayee -- Stormnode payment %lld to %s\n", stormnodePayment, address2.ToString());
    }
}

int CStormnodePayments::GetMinStormnodePaymentsProto() {
    return sporkManager.IsSporkActive(SPORK_10_STORMNODE_PAY_UPDATED_NODES);
}

void CStormnodePayments::ProcessMessage(CNode* pfrom, std::string& strCommand, CDataStream& vRecv)
{
    // Ignore any payments messages until Stormnode list is synced
    if(!stormnodeSync.IsStormnodeListSynced()) return;

    if(fLiteMode) return; // disable all DarkSilk specific functionality

    if (strCommand == NetMsgType::STORMNODEPAYMENTSYNC) { //Stormnode Payments Request Sync

        // Ignore such requests until we are fully synced.
        // We could start processing this after Stormnode list is synced
        // but this is a heavy one so it's better to finish sync first.
        if (!stormnodeSync.IsSynced()) return;

        int nCountNeeded;
        vRecv >> nCountNeeded;

        if(netfulfilledman.HasFulfilledRequest(pfrom->addr, NetMsgType::STORMNODEPAYMENTSYNC)) {
            // Asking for the payments list multiple times in a short period of time is no good
            LogPrintf("STORMNODEPAYMENTSYNC -- peer already asked me for the list, peer=%d\n", pfrom->id);
            Misbehaving(pfrom->GetId(), 20);
            return;
        }
        netfulfilledman.AddFulfilledRequest(pfrom->addr, NetMsgType::STORMNODEPAYMENTSYNC);

        Sync(pfrom, nCountNeeded);
        LogPrintf("STORMNODEPAYMENTSYNC -- Sent Stormnode payment votes to peer %d\n", pfrom->id);

    } else if (strCommand == NetMsgType::STORMNODEPAYMENTVOTE) { // Stormnode Payments Vote for the Winner

        CStormnodePaymentVote vote;
        vRecv >> vote;

        if(pfrom->nVersion < GetMinStormnodePaymentsProto()) return;

        if(!pCurrentBlockIndex) return;

        uint256 nHash = vote.GetHash();

        pfrom->setAskFor.erase(nHash);

        {
            LOCK(cs_mapStormnodePaymentVotes);
            if(mapStormnodePaymentVotes.count(nHash)) {
                LogPrint("snpayments", "STORMNODEPAYMENTVOTE -- hash=%s, nHeight=%d seen\n", nHash.ToString(), pCurrentBlockIndex->nHeight);
                return;
            }

            // Avoid processing same vote multiple times
            mapStormnodePaymentVotes[nHash] = vote;
            // but first mark vote as non-verified,
            // AddPaymentVote() below should take care of it if vote is actually ok
            mapStormnodePaymentVotes[nHash].MarkAsNotVerified();
        }

        int nFirstBlock = pCurrentBlockIndex->nHeight - GetStorageLimit();
        if(vote.nBlockHeight < nFirstBlock || vote.nBlockHeight > pCurrentBlockIndex->nHeight+20) {
            LogPrint("snpayments", "STORMNODEPAYMENTVOTE -- vote out of range: nFirstBlock=%d, nBlockHeight=%d, nHeight=%d\n", nFirstBlock, vote.nBlockHeight, pCurrentBlockIndex->nHeight);
            return;
        }

        std::string strError = "";
        if(!vote.IsValid(pfrom, pCurrentBlockIndex->nHeight, strError)) {
            LogPrint("snpayments", "STORMNODEPAYMENTVOTE -- invalid message, error: %s\n", strError);
            return;
        }

        if(!CanVote(vote.vinStormnode.prevout, vote.nBlockHeight)) {
            LogPrintf("STORMNODEPAYMENTVOTE -- Stormnode already voted, Stormnode=%s\n", vote.vinStormnode.prevout.ToStringShort());
            return;
        }

        stormnode_info_t snInfo = snodeman.GetStormnodeInfo(vote.vinStormnode);
        if(!snInfo.fInfoValid) {
            // sn was not found, so we can't check vote, some info is probably missing
            LogPrintf("STORMNODEPAYMENTVOTE -- Stormnode is missing %s\n", vote.vinStormnode.prevout.ToStringShort());
            snodeman.AskForSN(pfrom, vote.vinStormnode);
            return;
        }

        int nDos = 0;
        if(!vote.CheckSignature(snInfo.pubKeyStormnode, pCurrentBlockIndex->nHeight, nDos)) {
            if(nDos) {
                LogPrintf("STORMNODEPAYMENTVOTE -- ERROR: invalid signature\n");
                Misbehaving(pfrom->GetId(), nDos);
            } else {
                // only warn about anything non-critical (i.e. nDos == 0) in debug mode
                LogPrint("snpayments", "STORMNODEPAYMENTVOTE -- WARNING: invalid signature\n");
            }
            // Either our info or vote info could be outdated.
            // In case our info is outdated, ask for an update,
            snodeman.AskForSN(pfrom, vote.vinStormnode);
            // but there is nothing we can do if vote info itself is outdated
            // (i.e. it was signed by a sn which changed its key),
            // so just quit here.
            return;
        }

        CTxDestination address1;
        ExtractDestination(vote.payee, address1);
        CDarkSilkAddress address2(address1);

        LogPrint("snpayments", "STORMNODEPAYMENTVOTE -- vote: address=%s, nBlockHeight=%d, nHeight=%d, prevout=%s\n", address2.ToString(), vote.nBlockHeight, pCurrentBlockIndex->nHeight, vote.vinStormnode.prevout.ToStringShort());

        if(AddPaymentVote(vote)){
            vote.Relay();
            stormnodeSync.AddedPaymentVote();
        }
    }
}

bool CStormnodePaymentVote::Sign()
{
    std::string strError;
    std::string strMessage = vinStormnode.prevout.ToStringShort() +
                boost::lexical_cast<std::string>(nBlockHeight) +
                ScriptToAsmStr(payee);

    if(!privateSendSigner.SignMessage(strMessage, vchSig, activeStormnode.keyStormnode)) {
        LogPrintf("CStormnodePaymentVote::Sign -- SignMessage() failed\n");
        return false;
    }

    if(!privateSendSigner.VerifyMessage(activeStormnode.pubKeyStormnode, vchSig, strMessage, strError)) {
        LogPrintf("CStormnodePaymentVote::Sign -- VerifyMessage() failed, error: %s\n", strError);
        return false;
    }

    return true;
}

bool CStormnodePayments::GetBlockPayee(int nBlockHeight, CScript& payee)
{
    if(mapStormnodeBlocks.count(nBlockHeight)){
        return mapStormnodeBlocks[nBlockHeight].GetBestPayee(payee);
    }

    return false;
}

// Is this Stormnode scheduled to get paid soon?
// -- Only look ahead up to 8 blocks to allow for propagation of the latest 2 blocks of votes
bool CStormnodePayments::IsScheduled(CStormnode& sn, int nNotBlockHeight)
{
    LOCK(cs_mapStormnodeBlocks);

    if(!pCurrentBlockIndex) return false;

    CScript snpayee;
    snpayee = GetScriptForDestination(sn.pubKeyCollateralAddress.GetID());

    CScript payee;
    for(int64_t h = pCurrentBlockIndex->nHeight; h <= pCurrentBlockIndex->nHeight + 8; h++){
        if(h == nNotBlockHeight) continue;
        if(mapStormnodeBlocks.count(h) && mapStormnodeBlocks[h].GetBestPayee(payee) && snpayee == payee) {
            return true;
        }
    }

    return false;
}

bool CStormnodePayments::AddPaymentVote(const CStormnodePaymentVote& vote)
{
    uint256 blockHash = uint256();
    if(!GetBlockHash(blockHash, vote.nBlockHeight - 101)) return false;

    if(HasVerifiedPaymentVote(vote.GetHash())) return false;

    LOCK2(cs_mapStormnodeBlocks, cs_mapStormnodePaymentVotes);

    mapStormnodePaymentVotes[vote.GetHash()] = vote;

    if(!mapStormnodeBlocks.count(vote.nBlockHeight)) {
       CStormnodeBlockPayees blockPayees(vote.nBlockHeight);
       mapStormnodeBlocks[vote.nBlockHeight] = blockPayees;
    }

    mapStormnodeBlocks[vote.nBlockHeight].AddPayee(vote);

    return true;
}

bool CStormnodePayments::HasVerifiedPaymentVote(uint256 hashIn)
{
    LOCK(cs_mapStormnodePaymentVotes);
    std::map<uint256, CStormnodePaymentVote>::iterator it = mapStormnodePaymentVotes.find(hashIn);
    return it != mapStormnodePaymentVotes.end() && it->second.IsVerified();
}

void CStormnodeBlockPayees::AddPayee(const CStormnodePaymentVote& vote)
{
    LOCK(cs_vecPayees);

    BOOST_FOREACH(CStormnodePayee& payee, vecPayees) {
        if (payee.GetPayee() == vote.payee) {
            payee.AddVoteHash(vote.GetHash());
            return;
        }
    }
    CStormnodePayee payeeNew(vote.payee, vote.GetHash());
    vecPayees.push_back(payeeNew);
}

bool CStormnodeBlockPayees::GetBestPayee(CScript& payeeRet)
{
    LOCK(cs_vecPayees);

    if(!vecPayees.size()) {
        LogPrint("snpayments", "CStormnodeBlockPayees::GetBestPayee -- ERROR: couldn't find any payee\n");
        return false;
    }

    int nVotes = -1;
    BOOST_FOREACH(CStormnodePayee& payee, vecPayees) {
        if (payee.GetVoteCount() > nVotes) {
            payeeRet = payee.GetPayee();
            nVotes = payee.GetVoteCount();
        }
    }

    return (nVotes > -1);
}

bool CStormnodeBlockPayees::HasPayeeWithVotes(CScript payeeIn, int nVotesReq)
{
    LOCK(cs_vecPayees);

    BOOST_FOREACH(CStormnodePayee& payee, vecPayees) {
        if (payee.GetVoteCount() >= nVotesReq && payee.GetPayee() == payeeIn) {
            return true;
        }
    }

    LogPrint("snpayments", "CStormnodeBlockPayees::HasPayeeWithVotes -- ERROR: couldn't find any payee with %d+ votes\n", nVotesReq);
    return false;
}

bool CStormnodeBlockPayees::IsTransactionValid(const CTransaction& txNew)
{
    LOCK(cs_vecPayees);

    int nMaxSignatures = 0;
    std::string strPayeesPossible = "";

    CAmount nStormnodePayment = STATIC_STORMNODE_PAYMENT;

    //require at least SNPAYMENTS_SIGNATURES_REQUIRED signatures

    BOOST_FOREACH(CStormnodePayee& payee, vecPayees) {
        if (payee.GetVoteCount() >= nMaxSignatures) {
            nMaxSignatures = payee.GetVoteCount();
        }
    }

    // if we don't have at least SNPAYMENTS_SIGNATURES_REQUIRED signatures on a payee, approve whichever is the longest chain
    if(nMaxSignatures < SNPAYMENTS_SIGNATURES_REQUIRED) return true;

    BOOST_FOREACH(CStormnodePayee& payee, vecPayees) {
        if (payee.GetVoteCount() >= SNPAYMENTS_SIGNATURES_REQUIRED) {
            BOOST_FOREACH(CTxOut txout, txNew.vout) {
                if (payee.GetPayee() == txout.scriptPubKey && nStormnodePayment == txout.nValue) {
                    LogPrint("snpayments", "CStormnodeBlockPayees::IsTransactionValid -- Found required payment\n");
                    return true;
                }
            }

            CTxDestination address1;
            ExtractDestination(payee.GetPayee(), address1);
            CDarkSilkAddress address2(address1);

            if(strPayeesPossible == "") {
                strPayeesPossible = address2.ToString();
            } else {
                strPayeesPossible += "," + address2.ToString();
            }
        }
    }

    LogPrintf("CStormnodeBlockPayees::IsTransactionValid -- ERROR: Missing required payment, possible payees: '%s', amount: %f DSLK\n", strPayeesPossible, (float)nStormnodePayment/COIN);
    return false;
}

std::string CStormnodeBlockPayees::GetRequiredPaymentsString()
{
    LOCK(cs_vecPayees);

    std::string strRequiredPayments = "Unknown";

    BOOST_FOREACH(CStormnodePayee& payee, vecPayees)
    {
        CTxDestination address1;
        ExtractDestination(payee.GetPayee(), address1);
        CDarkSilkAddress address2(address1);

        if (strRequiredPayments != "Unknown") {
            strRequiredPayments += ", " + address2.ToString() + ":" + boost::lexical_cast<std::string>(payee.GetVoteCount());
        } else {
            strRequiredPayments = address2.ToString() + ":" + boost::lexical_cast<std::string>(payee.GetVoteCount());
        }
    }

    return strRequiredPayments;
}

std::string CStormnodePayments::GetRequiredPaymentsString(int nBlockHeight)
{
    LOCK(cs_mapStormnodeBlocks);

    if(mapStormnodeBlocks.count(nBlockHeight)){
        return mapStormnodeBlocks[nBlockHeight].GetRequiredPaymentsString();
    }

    return "Unknown";
}

bool CStormnodePayments::IsTransactionValid(const CTransaction& txNew, int nBlockHeight)
{
    LOCK(cs_mapStormnodeBlocks);

    if(mapStormnodeBlocks.count(nBlockHeight)){
        return mapStormnodeBlocks[nBlockHeight].IsTransactionValid(txNew);
    }

    return true;
}

void CStormnodePayments::CheckAndRemove()
{
    if(!pCurrentBlockIndex) return;

    LOCK2(cs_mapStormnodeBlocks, cs_mapStormnodePaymentVotes);

    int nLimit = GetStorageLimit();

    std::map<uint256, CStormnodePaymentVote>::iterator it = mapStormnodePaymentVotes.begin();
    while(it != mapStormnodePaymentVotes.end()) {
        CStormnodePaymentVote vote = (*it).second;

        if(pCurrentBlockIndex->nHeight - vote.nBlockHeight > nLimit) {
            LogPrint("snpayments", "CStormnodePayments::CheckAndRemove -- Removing old Stormnode payment: nBlockHeight=%d\n", vote.nBlockHeight);
            mapStormnodePaymentVotes.erase(it++);
            mapStormnodeBlocks.erase(vote.nBlockHeight);
        } else {
            ++it;
        }
    }
    LogPrintf("CStormnodePayments::CheckAndRemove -- %s\n", ToString());
}

bool CStormnodePaymentVote::IsValid(CNode* pnode, int nValidationHeight, std::string& strError)
{
    CStormnode* psn = snodeman.Find(vinStormnode);

    if(!psn) {
        strError = strprintf("Unknown Stormnode: prevout=%s", vinStormnode.prevout.ToStringShort());
        // Only ask if we are already synced and still have no idea about that Stormnode
        if(stormnodeSync.IsStormnodeListSynced()) {
            snodeman.AskForSN(pnode, vinStormnode);
        }

        return false;
    }

    int nMinRequiredProtocol;
    if(nBlockHeight >= nValidationHeight) {
        // new votes must comply SPORK_10_STORMNODE_PAY_UPDATED_NODES rules
        nMinRequiredProtocol = snpayments.GetMinStormnodePaymentsProto();
    } else {
        // allow non-updated Stormnodes for old blocks
        nMinRequiredProtocol = MIN_STORMNODE_PAYMENT_PROTO_VERSION;
    }

    if(psn->nProtocolVersion < nMinRequiredProtocol) {
        strError = strprintf("Stormnode protocol is too old: nProtocolVersion=%d, nMinRequiredProtocol=%d", psn->nProtocolVersion, nMinRequiredProtocol);
        return false;
    }

    // Only Stormnodes should try to check Stormnode rank for old votes - they need to pick the right winner for future blocks.
    // Regular clients (miners included) need to verify Stormnode rank for future block votes only.
    if(!fStormNode && nBlockHeight < nValidationHeight) return true;

    int nRank = snodeman.GetStormnodeRank(vinStormnode, nBlockHeight - 101, nMinRequiredProtocol, false);

    if(nRank > SNPAYMENTS_SIGNATURES_TOTAL) {
        // It's common to have Stormnodes mistakenly think they are in the top 10
        // We don't want to print all of these messages in normal mode, debug mode should print though
        strError = strprintf("Stormnode is not in the top %d (%d)", SNPAYMENTS_SIGNATURES_TOTAL, nRank);
        // Only ban for new snw which is out of bounds, for old snw SN list itself might be way too much off
        if(nRank > SNPAYMENTS_SIGNATURES_TOTAL*2 && nBlockHeight > nValidationHeight) {
            strError = strprintf("Stormnode is not in the top %d (%d)", SNPAYMENTS_SIGNATURES_TOTAL*2, nRank);
            LogPrintf("CStormnodePaymentVote::IsValid -- Error: %s\n", strError);
            Misbehaving(pnode->GetId(), 20);
        }
        // Still invalid however
        return false;
    }

    return true;
}

bool CStormnodePayments::ProcessBlock(int nBlockHeight)
{
    // DETERMINE IF WE SHOULD BE VOTING FOR THE NEXT PAYEE

    if(fLiteMode || !fStormNode) return false;

    // We have little chances to pick the right winner if winners list is out of sync
    // but we have no choice, so we'll try. However it doesn't make sense to even try to do so
    // if we have not enough data about Stormnodes.
    if(!stormnodeSync.IsStormnodeListSynced()) return false;

    int nRank = snodeman.GetStormnodeRank(activeStormnode.vin, nBlockHeight - 101, GetMinStormnodePaymentsProto(), false);

    if (nRank == -1) {
        LogPrint("snpayments", "CStormnodePayments::ProcessBlock -- Unknown Stormnode\n");
        return false;
    }

    if (nRank > SNPAYMENTS_SIGNATURES_TOTAL) {
        LogPrint("snpayments", "CStormnodePayments::ProcessBlock -- Stormnode not in the top %d (%d)\n", SNPAYMENTS_SIGNATURES_TOTAL, nRank);
        return false;
    }


    // LOCATE THE NEXT STORMNODE WHICH SHOULD BE PAID

    LogPrintf("CStormnodePayments::ProcessBlock -- Start: nBlockHeight=%d, Stormnode=%s\n", nBlockHeight, activeStormnode.vin.prevout.ToStringShort());

    // pay to the oldest SN that still had no payment but its input is old enough and it was active long enough
    int nCount = 0;
    CStormnode *psn = snodeman.GetNextStormnodeInQueueForPayment(nBlockHeight, true, nCount);

    if (psn == NULL) {
        LogPrintf("CStormnodePayments::ProcessBlock -- ERROR: Failed to find Stormnode to pay\n");
        return false;
    }

    LogPrintf("CStormnodePayments::ProcessBlock -- Stormnode found by GetNextStormnodeInQueueForPayment(): %s\n", psn->vin.prevout.ToStringShort());


    CScript payee = GetScriptForDestination(psn->pubKeyCollateralAddress.GetID());

    CStormnodePaymentVote voteNew(activeStormnode.vin, nBlockHeight, payee);

    CTxDestination address1;
    ExtractDestination(payee, address1);
    CDarkSilkAddress address2(address1);

    LogPrintf("CStormnodePayments::ProcessBlock -- vote: payee=%s, nBlockHeight=%d\n", address2.ToString(), nBlockHeight);

    // SIGN MESSAGE TO NETWORK WITH OUR STORMNODE KEYS

    LogPrintf("CStormnodePayments::ProcessBlock -- Signing vote\n");
    if (voteNew.Sign()) {
        LogPrintf("CStormnodePayments::ProcessBlock -- AddPaymentVote()\n");

        if (AddPaymentVote(voteNew)) {
            voteNew.Relay();
            return true;
        }
    }

    return false;
}

void CStormnodePaymentVote::Relay()
{
    // do not relay until synced
    if (!stormnodeSync.IsWinnersListSynced()) return;
    CInv inv(MSG_STORMNODE_PAYMENT_VOTE, GetHash());
    RelayInv(inv);
}

bool CStormnodePaymentVote::CheckSignature(const CPubKey& pubKeyStormnode, int nValidationHeight, int &nDos)
{
    // do not ban by default
    nDos = 0;

    std::string strMessage = vinStormnode.prevout.ToStringShort() +
                boost::lexical_cast<std::string>(nBlockHeight) +
                ScriptToAsmStr(payee);

    std::string strError = "";
    if (!privateSendSigner.VerifyMessage(pubKeyStormnode, vchSig, strMessage, strError)) {
        // Only ban for future block vote when we are already synced.
        // Otherwise it could be the case when SN which signed this vote is using another key now
        // and we have no idea about the old one.
        if(stormnodeSync.IsStormnodeListSynced() && nBlockHeight > nValidationHeight) {
            nDos = 20;
        }
        return error("CStormnodePaymentVote::CheckSignature -- Got bad Stormnode payment signature, Stormnode=%s, error: %s", vinStormnode.prevout.ToStringShort().c_str(), strError);
    }

    return true;
}

std::string CStormnodePaymentVote::ToString() const
{
    std::ostringstream info;

    info << vinStormnode.prevout.ToStringShort() <<
            ", " << nBlockHeight <<
            ", " << ScriptToAsmStr(payee) <<
            ", " << (int)vchSig.size();

    return info.str();
}

// Send all votes up to nCountNeeded blocks (but not more than GetStorageLimit)
void CStormnodePayments::Sync(CNode* pnode, int nCountNeeded)
{
    LOCK(cs_mapStormnodeBlocks);

    if(!pCurrentBlockIndex) return;

    nCountNeeded = 0;

    int nInvCount = 0;

    for(int h = pCurrentBlockIndex->nHeight - nCountNeeded; h < pCurrentBlockIndex->nHeight + 20; h++) {
        if(mapStormnodeBlocks.count(h)) {
            BOOST_FOREACH(CStormnodePayee& payee, mapStormnodeBlocks[h].vecPayees) {
                std::vector<uint256> vecVoteHashes = payee.GetVoteHashes();
                BOOST_FOREACH(uint256& hash, vecVoteHashes) {
                    if(!HasVerifiedPaymentVote(hash)) continue;
                    pnode->PushInventory(CInv(MSG_STORMNODE_PAYMENT_VOTE, hash));
                    nInvCount++;
                }
            }
        }
    }

    LogPrintf("CStormnodePayments::Sync -- Sent %d votes to peer %d\n", nInvCount, pnode->id);
    pnode->PushMessage(NetMsgType::SYNCSTATUSCOUNT, STORMNODE_SYNC_SNW, nInvCount);
}

// Request low data/unknown payment blocks in batches directly from some node instead of/after preliminary Sync.
void CStormnodePayments::RequestLowDataPaymentBlocks(CNode* pnode)
{
    if(!pCurrentBlockIndex) return;

    LOCK2(cs_main, cs_mapStormnodeBlocks);

    std::vector<CInv> vToFetch;
    int nLimit = GetStorageLimit();

    const CBlockIndex *pindex = pCurrentBlockIndex;

    while(pCurrentBlockIndex->nHeight - pindex->nHeight < nLimit) {
        if(!mapStormnodeBlocks.count(pindex->nHeight)) {
            // We have no idea about this block height, let's ask
            vToFetch.push_back(CInv(MSG_STORMNODE_PAYMENT_BLOCK, pindex->GetBlockHash()));
            // We should not violate GETDATA rules
            if(vToFetch.size() == MAX_INV_SZ) {
                LogPrintf("CStormnodePayments::SyncLowDataPaymentBlocks -- asking peer %d for %d blocks\n", pnode->id, MAX_INV_SZ);
                pnode->PushMessage(NetMsgType::GETDATA, vToFetch);
                // Start filling new batch
                vToFetch.clear();
            }
        }
        if(!pindex->pprev) break;
        pindex = pindex->pprev;
    }

    std::map<int, CStormnodeBlockPayees>::iterator it = mapStormnodeBlocks.begin();

    while(it != mapStormnodeBlocks.end()) {
        int nTotalVotes = 0;
        bool fFound = false;
        BOOST_FOREACH(CStormnodePayee& payee, it->second.vecPayees) {
            if(payee.GetVoteCount() >= SNPAYMENTS_SIGNATURES_REQUIRED) {
                fFound = true;
                break;
            }
            nTotalVotes += payee.GetVoteCount();
        }
        // A clear winner (SNPAYMENTS_SIGNATURES_REQUIRED+ votes) was found
        // or no clear winner was found but there are at least avg number of votes
        if(fFound || nTotalVotes >= (SNPAYMENTS_SIGNATURES_TOTAL + SNPAYMENTS_SIGNATURES_REQUIRED)/2) {
            // so just move to the next block
            ++it;
            continue;
        }
        // DEBUG
        DBG (
            // Let's see why this failed
            BOOST_FOREACH(CStormnodePayee& payee, it->second.vecPayees) {
                CTxDestination address1;
                ExtractDestination(payee.GetPayee(), address1);
                CDarkSilkAddress address2(address1);
                printf("payee %s votes %d\n", address2.ToString().c_str(), payee.GetVoteCount());
            }
            printf("block %d votes total %d\n", it->first, nTotalVotes);
        )
        // END DEBUG
        // Low data block found, let's try to sync it
        uint256 hash;
        if(GetBlockHash(hash, it->first)) {
            vToFetch.push_back(CInv(MSG_STORMNODE_PAYMENT_BLOCK, hash));
        }
        // We should not violate GETDATA rules
        if(vToFetch.size() == MAX_INV_SZ) {
            LogPrintf("CStormnodePayments::SyncLowDataPaymentBlocks -- asking peer %d for %d payment blocks\n", pnode->id, MAX_INV_SZ);
            pnode->PushMessage(NetMsgType::GETDATA, vToFetch);
            // Start filling new batch
            vToFetch.clear();
        }
        ++it;
    }
    // Ask for the rest of it
    if(!vToFetch.empty()) {
        LogPrintf("CStormnodePayments::SyncLowDataPaymentBlocks -- asking peer %d for %d payment blocks\n", pnode->id, vToFetch.size());
        pnode->PushMessage(NetMsgType::GETDATA, vToFetch);
    }
}

std::string CStormnodePayments::ToString() const
{
    std::ostringstream info;

    info << "Votes: " << (int)mapStormnodePaymentVotes.size() <<
            ", Blocks: " << (int)mapStormnodeBlocks.size();

    return info.str();
}

bool CStormnodePayments::IsEnoughData()
{
    float nAverageVotes = (SNPAYMENTS_SIGNATURES_TOTAL + SNPAYMENTS_SIGNATURES_REQUIRED) / 2;
    int nStorageLimit = GetStorageLimit();
    return GetBlockCount() > nStorageLimit && GetVoteCount() > nStorageLimit * nAverageVotes;
}

int CStormnodePayments::GetStorageLimit()
{
    return std::max(int(snodeman.size() * nStorageCoeff), nMinBlocksToStore);
}

void CStormnodePayments::UpdatedBlockTip(const CBlockIndex *pindex)
{
    pCurrentBlockIndex = pindex;
    LogPrint("snpayments", "CStormnodePayments::UpdatedBlockTip -- pCurrentBlockIndex->nHeight=%d\n", pCurrentBlockIndex->nHeight);

    ProcessBlock(pindex->nHeight + 10);
}
