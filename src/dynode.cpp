// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dynode.h"

#include "activedynode.h"
#include "chain.h"
#include "dynode-payments.h"
#include "dynode-sync.h"
#include "dynodeman.h"
#include "fluid.h"
#include "init.h"
#include "messagesigner.h"
#include "util.h"
#include "validation.h"

#include <boost/lexical_cast.hpp>

CDynode::CDynode() :
    dynode_info_t{ DYNODE_ENABLED, PROTOCOL_VERSION, GetAdjustedTime()},
    fAllowMixingTx(true)
{}

CDynode::CDynode(CService addr, COutPoint outpoint, CPubKey pubKeyCollateralAddress, CPubKey pubKeyDynode, int nProtocolVersionIn) :
    dynode_info_t{ DYNODE_ENABLED, nProtocolVersionIn, GetAdjustedTime(),
                       outpoint, addr, pubKeyCollateralAddress, pubKeyDynode},
    fAllowMixingTx(true)
{}

CDynode::CDynode(const CDynode& other) :
    dynode_info_t{other},
    lastPing(other.lastPing),
    vchSig(other.vchSig),
    nCollateralMinConfBlockHash(other.nCollateralMinConfBlockHash),
    nBlockLastPaid(other.nBlockLastPaid),
    nPoSeBanScore(other.nPoSeBanScore),
    nPoSeBanHeight(other.nPoSeBanHeight),
    fAllowMixingTx(other.fAllowMixingTx),
    fUnitTest(other.fUnitTest)
{}

CDynode::CDynode(const CDynodeBroadcast& dnb) :
    dynode_info_t{ dnb.nActiveState, dnb.nProtocolVersion, dnb.sigTime,
                       dnb.vin.prevout, dnb.addr, dnb.pubKeyCollateralAddress, dnb.pubKeyDynode,
                       dnb.sigTime /*nTimeLastWatchdogVote*/},
    lastPing(dnb.lastPing),
    vchSig(dnb.vchSig),
    fAllowMixingTx(true)
{}

//
// When a new Dynode broadcast is sent, update our information
//
bool CDynode::UpdateFromNewBroadcast(CDynodeBroadcast& dnb, CConnman& connman)
{
    if(dnb.sigTime <= sigTime && !dnb.fRecovery) return false;

    pubKeyDynode = dnb.pubKeyDynode;
    sigTime = dnb.sigTime;
    vchSig = dnb.vchSig;
    nProtocolVersion = dnb.nProtocolVersion;
    addr = dnb.addr;
    nPoSeBanScore = 0;
    nPoSeBanHeight = 0;
    nTimeLastChecked = 0;
    int nDos = 0;
    if(dnb.lastPing == CDynodePing() || (dnb.lastPing != CDynodePing() && dnb.lastPing.CheckAndUpdate(this, true, nDos, connman))) {
        lastPing = dnb.lastPing;
        dnodeman.mapSeenDynodePing.insert(std::make_pair(lastPing.GetHash(), lastPing));
    }
    // if it matches our Dynode privkey...
    if(fDyNode && pubKeyDynode == activeDynode.pubKeyDynode) {
        nPoSeBanScore = -DYNODE_POSE_BAN_MAX_SCORE;
        if(nProtocolVersion == PROTOCOL_VERSION) {
            // ... and PROTOCOL_VERSION, then we've been remotely activated ...
            activeDynode.ManageState(connman);
        } else {
            // ... otherwise we need to reactivate our node, do not add it to the list and do not relay
            // but also do not ban the node we get this message from
            LogPrintf("CDynode::UpdateFromNewBroadcast -- wrong PROTOCOL_VERSION, re-activate your DN: message nProtocolVersion=%d  PROTOCOL_VERSION=%d\n", nProtocolVersion, PROTOCOL_VERSION);
            return false;
        }
    }
    return true;
}

//
// Deterministically calculate a given "score" for a Dynode depending on how close it's hash is to
// the proof of work for that block. The further away they are the better, the furthest will win the election
// and get paid this block
//
arith_uint256 CDynode::CalculateScore(const uint256& blockHash)
{
        // Deterministically calculate a "score" for a Dynode based on any given (block)hash
        CHashWriter ss(SER_GETHASH, PROTOCOL_VERSION);
        ss << vin.prevout << nCollateralMinConfBlockHash << blockHash;
        return UintToArith256(ss.GetHash());
}

CDynode::CollateralStatus CDynode::CheckCollateral(const COutPoint& outpoint)
{
    int nHeight;
    return CheckCollateral(outpoint, nHeight);
}

CDynode::CollateralStatus CDynode::CheckCollateral(const COutPoint& outpoint, int& nHeightRet)
{
    AssertLockHeld(cs_main);

    CCoins coins;
    if(!GetUTXOCoins(outpoint, coins)) {
        return COLLATERAL_UTXO_NOT_FOUND;
    }

    if(coins.vout[outpoint.n].nValue != 1000 * COIN) {
        return COLLATERAL_INVALID_AMOUNT;
    }

    nHeightRet = coins.nHeight;
    return COLLATERAL_OK;
}

void CDynode::Check(bool fForce)
{
    LOCK(cs);

    if(ShutdownRequested()) return;

    if(!fForce && (GetTime() - nTimeLastChecked < DYNODE_CHECK_SECONDS)) return;
    nTimeLastChecked = GetTime();

    LogPrint("Dynode", "CDynode::Check -- Dynode %s is in %s state\n", vin.prevout.ToStringShort(), GetStateString());

    //once spent, stop doing the checks
    if(IsOutpointSpent()) return;

    int nHeight = 0;
    if(!fUnitTest) {
        TRY_LOCK(cs_main, lockMain);
        if(!lockMain) return;

        CollateralStatus err = CheckCollateral(vin.prevout);
        if (err == COLLATERAL_UTXO_NOT_FOUND) {
            nActiveState = DYNODE_OUTPOINT_SPENT;
            LogPrint("Dynode", "CDynode::Check -- Failed to find Dynode UTXO, Dynode=%s\n", vin.prevout.ToStringShort());
            return;
        }

        nHeight = chainActive.Height();
    }

    if(IsPoSeBanned()) {
        if(nHeight < nPoSeBanHeight) return; // too early?
        // Otherwise give it a chance to proceed further to do all the usual checks and to change its state.
        // Dynode still will be on the edge and can be banned back easily if it keeps ignoring dnverify
        // or connect attempts. Will require few dnverify messages to strengthen its position in dn list.
        LogPrintf("CDynode::Check -- Dynode %s is unbanned and back in list now\n", vin.prevout.ToStringShort());
        DecreasePoSeBanScore();
    } else if(nPoSeBanScore >= DYNODE_POSE_BAN_MAX_SCORE) {
        nActiveState = DYNODE_POSE_BAN;
        // ban for the whole payment cycle
        nPoSeBanHeight = nHeight + dnodeman.size();
        LogPrintf("CDynode::Check -- Dynode %s is banned till block %d now\n", vin.prevout.ToStringShort(), nPoSeBanHeight);
        return;
    }

    int nActiveStatePrev = nActiveState;
    bool fOurDynode = fDyNode && activeDynode.pubKeyDynode == pubKeyDynode;
                   // Dynode doesn't meet payment protocol requirements ...
    bool fRequireUpdate = nProtocolVersion < dnpayments.GetMinDynodePaymentsProto() ||
                   // or it's our own node and we just updated it to the new protocol but we are still waiting for activation ...
                   (fOurDynode && nProtocolVersion < PROTOCOL_VERSION);

    if(fRequireUpdate) {
        nActiveState = DYNODE_UPDATE_REQUIRED;
        if(nActiveStatePrev != nActiveState) {
            LogPrint("Dynode", "CDynode::Check -- Dynode %s is in %s state now\n", vin.prevout.ToStringShort(), GetStateString());
        }
        return;
    }

    // keep old Dynodes on start, give them a chance to receive updates...
    bool fWaitForPing = !dynodeSync.IsDynodeListSynced() && !IsPingedWithin(DYNODE_MIN_DNP_SECONDS);

    if(fWaitForPing && !fOurDynode) {
        // ...but if it was already expired before the initial check - return right away
        if(IsExpired() || IsWatchdogExpired() || IsNewStartRequired()) {
            LogPrint("Dynode", "CDynode::Check -- Dynode %s is in %s state, waiting for ping\n", vin.prevout.ToStringShort(), GetStateString());
            return;
        }
    }

    // don't expire if we are still in "waiting for ping" mode unless it's our own Dynode
    if(!fWaitForPing || fOurDynode) {

        if(!IsPingedWithin(DYNODE_NEW_START_REQUIRED_SECONDS)) {
            nActiveState = DYNODE_NEW_START_REQUIRED;
            if(nActiveStatePrev != nActiveState) {
                LogPrint("Dynode", "CDynode::Check -- Dynode %s is in %s state now\n", vin.prevout.ToStringShort(), GetStateString());
            }
            return;
        }

        /*
        bool fWatchdogActive = dynodeSync.IsSynced() && dnodeman.IsWatchdogActive();
        bool fWatchdogExpired = (fWatchdogActive && ((GetAdjustedTime() - nTimeLastWatchdogVote) > DYNODE_WATCHDOG_MAX_SECONDS));

        LogPrint("Dynode", "CDynode::Check -- outpoint=%s, nTimeLastWatchdogVote=%d, GetAdjustedTime()=%d, fWatchdogExpired=%d\n",
                vin.prevout.ToStringShort(), nTimeLastWatchdogVote, GetAdjustedTime(), fWatchdogExpired);


        if(fWatchdogExpired) {
            nActiveState = DYNODE_WATCHDOG_EXPIRED;
            if(nActiveStatePrev != nActiveState) {
                LogPrint("Dynode", "CDynode::Check -- Dynode %s is in %s state now\n", vin.prevout.ToStringShort(), GetStateString());
            }
            return;
        }
        */

        if(!IsPingedWithin(DYNODE_EXPIRATION_SECONDS)) {
            nActiveState = DYNODE_EXPIRED;
            if(nActiveStatePrev != nActiveState) {
                LogPrint("Dynode", "CDynode::Check -- Dynode %s is in %s state now\n", vin.prevout.ToStringShort(), GetStateString());
            }
            return;
        }
    }

    if(lastPing.sigTime - sigTime < DYNODE_MIN_DNP_SECONDS) {
        nActiveState = DYNODE_PRE_ENABLED;
        if(nActiveStatePrev != nActiveState) {
            LogPrint("Dynode", "CDynode::Check -- Dynode %s is in %s state now\n", vin.prevout.ToStringShort(), GetStateString());
        }
        return;
    }

    nActiveState = DYNODE_ENABLED; // OK
    if(nActiveStatePrev != nActiveState) {
        LogPrint("Dynode", "CDynode::Check -- Dynode %s is in %s state now\n", vin.prevout.ToStringShort(), GetStateString());
    }
}

bool CDynode::IsValidNetAddr()
{
    return IsValidNetAddr(addr);
}

bool CDynode::IsValidNetAddr(CService addrIn)
{
    // TODO: regtest is fine with any addresses for now,
    // should probably be a bit smarter if one day we start to implement tests for this
    return Params().NetworkIDString() == CBaseChainParams::REGTEST ||
            (addrIn.IsIPv4() && !addrIn.IsIPv6() && IsReachable(addrIn) && addrIn.IsRoutable());
}

dynode_info_t CDynode::GetInfo()
{
    dynode_info_t info{*this};
    info.nTimeLastPing = lastPing.sigTime;
    info.fInfoValid = true;
    return info;
}

std::string CDynode::StateToString(int nStateIn)
{
    switch(nStateIn) {
        case DYNODE_PRE_ENABLED:            return "PRE_ENABLED";
        case DYNODE_ENABLED:                return "ENABLED";
        case DYNODE_EXPIRED:                return "EXPIRED";
        case DYNODE_OUTPOINT_SPENT:         return "OUTPOINT_SPENT";
        case DYNODE_UPDATE_REQUIRED:        return "UPDATE_REQUIRED";
        case DYNODE_WATCHDOG_EXPIRED:       return "WATCHDOG_EXPIRED";
        case DYNODE_NEW_START_REQUIRED:     return "NEW_START_REQUIRED";
        case DYNODE_POSE_BAN:               return "POSE_BAN";
        default:                               return "UNKNOWN";
    }
}

std::string CDynode::GetStateString() const
{
    return StateToString(nActiveState);
}

std::string CDynode::GetStatus() const
{
    // TODO: return smth a bit more human readable here
    return GetStateString();
}

void CDynode::UpdateLastPaid(const CBlockIndex *pindex, int nMaxBlocksToScanBack)
{
    if(!pindex) return;

    const CBlockIndex *BlockReading = pindex;

    CScript dnpayee = GetScriptForDestination(pubKeyCollateralAddress.GetID());
    // LogPrint("Dynode", "CDynode::UpdateLastPaidBlock -- searching for block with payment to %s\n", vin.prevout.ToStringShort());

    LOCK(cs_mapDynodeBlocks);

    for (int i = 0; BlockReading && BlockReading->nHeight > nBlockLastPaid && i < nMaxBlocksToScanBack; i++) {
        if(dnpayments.mapDynodeBlocks.count(BlockReading->nHeight) &&
            dnpayments.mapDynodeBlocks[BlockReading->nHeight].HasPayeeWithVotes(dnpayee, 2))
        {
            CBlock block;
            if(!ReadBlockFromDisk(block, BlockReading, Params().GetConsensus())) // shouldn't really happen
                continue;

            CAmount nDynodePayment = getDynodeSubsidyWithOverride(BlockReading->fluidParams.dynodeReward);

            BOOST_FOREACH(CTxOut txout, block.vtx[0].vout)
                if(dnpayee == txout.scriptPubKey && nDynodePayment == txout.nValue) {
                    nBlockLastPaid = BlockReading->nHeight;
                    nTimeLastPaid = BlockReading->nTime;
                    LogPrint("Dynode", "CDynode::UpdateLastPaidBlock -- searching for block with payment to %s -- found new %d\n", vin.prevout.ToStringShort(), nBlockLastPaid);
                    return;
                }
        }

        if (BlockReading->pprev == NULL) { assert(BlockReading); break; }
        BlockReading = BlockReading->pprev;
    }

    // Last payment for this Dynode wasn't found in latest dnpayments blocks
    // or it was found in dnpayments blocks but wasn't found in the blockchain.
    // LogPrint("Dynode", "CDynode::UpdateLastPaidBlock -- searching for block with payment to %s -- keeping old %d\n", vin.prevout.ToStringShort(), nBlockLastPaid);
}

bool CDynodeBroadcast::Create(std::string strService, std::string strKeyDynode, std::string strTxHash, std::string strOutputIndex, std::string& strErrorRet, CDynodeBroadcast &dnbRet, bool fOffline)
{
    COutPoint outpoint;
    CPubKey pubKeyCollateralAddressNew;
    CKey keyCollateralAddressNew;
    CPubKey pubKeyDynodeNew;
    CKey keyDynodeNew;

    auto Log = [&strErrorRet](std::string sErr)->bool
    {
        strErrorRet = sErr;
        LogPrintf("CDynodeBroadcast::Create -- %s\n", strErrorRet);
        return false;
    };

    //need correct blocks to send ping
    if (!fOffline && !dynodeSync.IsBlockchainSynced())
        return Log("Sync in progress. Must wait until sync is complete to start Dynode");

    if (!CMessageSigner::GetKeysFromSecret(strKeyDynode, keyDynodeNew, pubKeyDynodeNew))
        return Log(strprintf("Invalid Dynode key %s", strKeyDynode));

    if (!pwalletMain->GetDynodeOutpointAndKeys(outpoint, pubKeyCollateralAddressNew, keyCollateralAddressNew, strTxHash, strOutputIndex))
        return Log(strprintf("Could not allocate outpoint %s:%s for dynode %s", strTxHash, strOutputIndex, strService));

    CService service = CService(strService);
    int mainnetDefaultPort = DEFAULT_P2P_PORT;
    if (Params().NetworkIDString() == CBaseChainParams::MAIN) {
        if (service.GetPort() != mainnetDefaultPort)
            return Log(strprintf("Invalid port %u for Dynode %s, only %d is supported on mainnet.", service.GetPort(), strService, mainnetDefaultPort));
    } else if (service.GetPort() == mainnetDefaultPort)
        return Log(strprintf("Invalid port %u for Dynode %s, %d is the only supported on mainnet.", service.GetPort(), strService, mainnetDefaultPort));

    return Create(outpoint, CService(strService), keyCollateralAddressNew, pubKeyCollateralAddressNew, keyDynodeNew, pubKeyDynodeNew, strErrorRet, dnbRet);
}

bool CDynodeBroadcast::Create(const COutPoint& outpoint, const CService& service, const CKey& keyCollateralAddressNew, const CPubKey& pubKeyCollateralAddressNew, const CKey& keyDynodeNew, const CPubKey& pubKeyDynodeNew, std::string &strErrorRet, CDynodeBroadcast &dnbRet)
{
    // wait for reindex and/or import to finish
    if (fImporting || fReindex) return false;

    LogPrint("Dynode", "CDynodeBroadcast::Create -- pubKeyCollateralAddressNew = %s, pubKeyDynodeNew.GetID() = %s\n",
             CDynamicAddress(pubKeyCollateralAddressNew.GetID()).ToString(),
             pubKeyDynodeNew.GetID().ToString());

    auto Log = [&strErrorRet,&dnbRet](std::string sErr)->bool
    {
        strErrorRet = sErr;
        LogPrintf("CDynodeBroadcast::Create -- %s\n", strErrorRet);
        dnbRet = CDynodeBroadcast();
        return false;
    };

    CDynodePing dnp(outpoint);
    if (!dnp.Sign(keyDynodeNew, pubKeyDynodeNew))
        return Log(strprintf("Failed to sign ping, dynode=%s", outpoint.ToStringShort()));

    dnbRet = CDynodeBroadcast(service, outpoint, pubKeyCollateralAddressNew, pubKeyDynodeNew, PROTOCOL_VERSION);

    if (!dnbRet.IsValidNetAddr())
        return Log(strprintf("Invalid IP address, dynode=%s", outpoint.ToStringShort()));

    dnbRet.lastPing = dnp;
    if(!dnbRet.Sign(keyCollateralAddressNew)) {
        return Log(strprintf("Failed to sign broadcast, dynode=%s", outpoint.ToStringShort()));
        LogPrintf("CDynodeBroadcast::Create -- %s\n", strErrorRet);
        dnbRet = CDynodeBroadcast();
        return false;
    }

    return true;
}

bool CDynodeBroadcast::SimpleCheck(int& nDos)
{
    nDos = 0;

    // make sure addr is valid
    if(!IsValidNetAddr()) {
        LogPrintf("CDynodeBroadcast::SimpleCheck -- Invalid addr, rejected: Dynode=%s  addr=%s\n",
                    vin.prevout.ToStringShort(), addr.ToString());
        return false;
    }

    // make sure signature isn't in the future (past is OK)
    if (sigTime > GetAdjustedTime() + 60 * 60) {
        LogPrintf("CDynodeBroadcast::SimpleCheck -- Signature rejected, too far into the future: Dynode=%s\n", vin.prevout.ToStringShort());
        nDos = 1;
        return false;
    }

    // empty ping or incorrect sigTime/unknown blockhash
    if(lastPing == CDynodePing() || !lastPing.SimpleCheck(nDos)) {
        // one of us is probably forked or smth, just mark it as expired and check the rest of the rules
        nActiveState = DYNODE_EXPIRED;
    }

    if(nProtocolVersion < dnpayments.GetMinDynodePaymentsProto()) {
        LogPrintf("CDynodeBroadcast::SimpleCheck -- ignoring outdated Dynode: Dynode=%s  nProtocolVersion=%d\n", vin.prevout.ToStringShort(), nProtocolVersion);
        return false;
    }

    CScript pubkeyScript;
    pubkeyScript = GetScriptForDestination(pubKeyCollateralAddress.GetID());

    if(pubkeyScript.size() != 25) {
        LogPrintf("CDynodeBroadcast::SimpleCheck -- pubKeyCollateralAddress has the wrong size\n");
        nDos = 100;
        return false;
    }

    CScript pubkeyScript2;
    pubkeyScript2 = GetScriptForDestination(pubKeyDynode.GetID());

    if(pubkeyScript2.size() != 25) {
        LogPrintf("CDynodeBroadcast::SimpleCheck -- pubKeyDynode has the wrong size\n");
        nDos = 100;
        return false;
    }

    if(!vin.scriptSig.empty()) {
        LogPrintf("CDynodeBroadcast::SimpleCheck -- Ignore Not Empty ScriptSig %s\n",vin.ToString());
        nDos = 100;
        return false;
    }

    int mainnetDefaultPort = DEFAULT_P2P_PORT;
    if(Params().NetworkIDString() == CBaseChainParams::MAIN) {
        if(addr.GetPort() != mainnetDefaultPort) return false;
    } else if(addr.GetPort() == mainnetDefaultPort) return false;

    return true;
}

bool CDynodeBroadcast::Update(CDynode* pdn, int& nDos, CConnman& connman)
{
    nDos = 0;

    if(pdn->sigTime == sigTime && !fRecovery) {
        // mapSeenDynodeBroadcast in CDynodeMan::CheckDnbAndUpdateDynodeList should filter legit duplicates
        // but this still can happen if we just started, which is ok, just do nothing here.
        return false;
    }

    // this broadcast is older than the one that we already have - it's bad and should never happen
    // unless someone is doing something fishy
    if(pdn->sigTime > sigTime) {
        LogPrintf("CDynodeBroadcast::Update -- Bad sigTime %d (existing broadcast is at %d) for Dynode %s %s\n",
                      sigTime, pdn->sigTime, vin.prevout.ToStringShort(), addr.ToString());
        return false;
    }

    pdn->Check();

    // Dynode is banned by PoSe
    if(pdn->IsPoSeBanned()) {
        LogPrintf("CDynodeBroadcast::Update -- Banned by PoSe, Dynode=%s\n", vin.prevout.ToStringShort());
        return false;
    }

    // IsVnAssociatedWithPubkey is validated once in CheckOutpoint, after that they just need to match
    if(pdn->pubKeyCollateralAddress != pubKeyCollateralAddress) {
        LogPrintf("CDynodeBroadcast::Update -- Got mismatched pubKeyCollateralAddress and vin\n");
        nDos = 33;
        return false;
    }

    if (!CheckSignature(nDos)) {
        LogPrintf("CDynodeBroadcast::Update -- CheckSignature() failed, Dynode=%s\n", vin.prevout.ToStringShort());
        return false;
    }

    // if there was no Dynode broadcast recently or if it matches our Dynode privkey...
    if(!pdn->IsBroadcastedWithin(DYNODE_MIN_DNB_SECONDS) || (fDyNode && pubKeyDynode == activeDynode.pubKeyDynode)) {
        // take the newest entry
        LogPrintf("CDynodeBroadcast::Update -- Got UPDATED Dynode entry: addr=%s\n", addr.ToString());
        if(pdn->UpdateFromNewBroadcast(*this, connman)) {
            pdn->Check();
            Relay(connman);
        }
        dynodeSync.BumpAssetLastTime("CDynodeBroadcast::Update");
    }

    return true;
}

bool CDynodeBroadcast::CheckOutpoint(int& nDos)
{
    // we are a Dynode with the same vin (i.e. already activated) and this dnb is ours (matches our Dynodes privkey)
    // so nothing to do here for us
    if(fDyNode && vin.prevout == activeDynode.outpoint && pubKeyDynode == activeDynode.pubKeyDynode) {
        return false;
    }

    if (!CheckSignature(nDos)) {
        LogPrintf("CDynodeBroadcast::CheckOutpoint -- CheckSignature() failed, Dynode=%s\n", vin.prevout.ToStringShort());
        return false;
    }

    {
        TRY_LOCK(cs_main, lockMain);
        if(!lockMain) {
            // not dnb fault, let it to be checked again later
            LogPrint("Dynode", "CDynodeBroadcast::CheckOutpoint -- Failed to aquire lock, addr=%s", addr.ToString());
            dnodeman.mapSeenDynodeBroadcast.erase(GetHash());
            return false;
        }

        int nHeight;
        CollateralStatus err = CheckCollateral(vin.prevout, nHeight);
        if (err == COLLATERAL_UTXO_NOT_FOUND) {
            LogPrint("Dynode", "CDynodeBroadcast::CheckOutpoint -- Failed to find Dynode UTXO, Dynode=%s\n", vin.prevout.ToStringShort());
            return false;
        }
        if (err == COLLATERAL_INVALID_AMOUNT) {
            LogPrint("Dynode", "CDynodeBroadcast::CheckOutpoint -- Dynode UTXO should have 1000 DYN, Dynode=%s\n", vin.prevout.ToStringShort());
            return false;
        }

        if(chainActive.Height() - nHeight + 1 < Params().GetConsensus().nDynodeMinimumConfirmations) {
            LogPrintf("CDynodeBroadcast::CheckOutpoint -- Dynode UTXO must have at least %d confirmations, Dynode=%s\n",
                    Params().GetConsensus().nDynodeMinimumConfirmations, vin.prevout.ToStringShort());
            // maybe we miss few blocks, let this dnb to be checked again later
            dnodeman.mapSeenDynodeBroadcast.erase(GetHash());
            return false;
        }
        // remember the hash of the block where dynode collateral had minimum required confirmations
        nCollateralMinConfBlockHash = chainActive[nHeight + Params().GetConsensus().nDynodeMinimumConfirmations - 1]->GetBlockHash();
    }

    LogPrint("Dynode", "CDynodeBroadcast::CheckOutpoint -- Dynode UTXO verified\n");

    // make sure the vout that was signed is related to the transaction that spawned the Dynode
    //  - this is expensive, so it's only done once per Dynode
    if(!IsVinAssociatedWithPubkey(vin, pubKeyCollateralAddress)) {
        LogPrintf("CDynodeMan::CheckOutpoint -- Got mismatched pubKeyCollateralAddress and vin\n");
        nDos = 33;
        return false;
    }

    // verify that sig time is legit in past
    // should be at least not earlier than block when 1000 DYN tx got nDynodeMinimumConfirmations
    uint256 hashBlock = uint256();
    CTransaction tx2;
    GetTransaction(vin.prevout.hash, tx2, Params().GetConsensus(), hashBlock, true);
    {
        LOCK(cs_main);
        BlockMap::iterator mi = mapBlockIndex.find(hashBlock);
        if (mi != mapBlockIndex.end() && (*mi).second) {
            CBlockIndex* pDNIndex = (*mi).second; // block for 1000 DYN tx -> 1 confirmation
            CBlockIndex* pConfIndex = chainActive[pDNIndex->nHeight + Params().GetConsensus().nDynodeMinimumConfirmations - 1]; // block where tx got nDynodeMinimumConfirmations
            if(pConfIndex->GetBlockTime() > sigTime) {
                LogPrintf("CDynodeBroadcast::CheckOutpoint -- Bad sigTime %d (%d conf block is at %d) for Dynode %s %s\n",
                          sigTime, Params().GetConsensus().nDynodeMinimumConfirmations, pConfIndex->GetBlockTime(), vin.prevout.ToStringShort(), addr.ToString());
                return false;
            }
        }
    }

    return true;
}

bool CDynodeBroadcast::IsVinAssociatedWithPubkey(const CTxIn& txin, const CPubKey& pubkey)
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

bool CDynodeBroadcast::Sign(const CKey& keyCollateralAddress)
{
    std::string strError;
    std::string strMessage;

    sigTime = GetAdjustedTime();

    strMessage = addr.ToString(false) + boost::lexical_cast<std::string>(sigTime) +
                    pubKeyCollateralAddress.GetID().ToString() + pubKeyDynode.GetID().ToString() +
                    boost::lexical_cast<std::string>(nProtocolVersion);

    if(!CMessageSigner::SignMessage(strMessage, vchSig, keyCollateralAddress)) {
        LogPrintf("CDynodeBroadcast::Sign -- SignMessage() failed\n");
        return false;
    }

    if(!CMessageSigner::VerifyMessage(pubKeyCollateralAddress, vchSig, strMessage, strError)) {
        LogPrintf("CDynodeBroadcast::Sign -- VerifyMessage() failed, error: %s\n", strError);
        return false;
    }

    return true;
}

bool CDynodeBroadcast::CheckSignature(int& nDos)
{
    std::string strMessage;
    std::string strError = "";
    nDos = 0;

    strMessage = addr.ToString(false) + boost::lexical_cast<std::string>(sigTime) +
                    pubKeyCollateralAddress.GetID().ToString() + pubKeyDynode.GetID().ToString() +
                    boost::lexical_cast<std::string>(nProtocolVersion);

    LogPrint("Dynode", "CDynodeBroadcast::CheckSignature -- strMessage: %s  pubKeyCollateralAddress address: %s  sig: %s\n", strMessage, CDynamicAddress(pubKeyCollateralAddress.GetID()).ToString(), EncodeBase64(&vchSig[0], vchSig.size()));

    if(!CMessageSigner::VerifyMessage(pubKeyCollateralAddress, vchSig, strMessage, strError)){
        LogPrintf("CDynodeBroadcast::CheckSignature -- Got bad Dynode announce signature, error: %s\n", strError);
        nDos = 100;
        return false;
    }

    return true;
}

void CDynodeBroadcast::Relay(CConnman& connman)
{
    CInv inv(MSG_DYNODE_ANNOUNCE, GetHash());
    connman.RelayInv(inv);
}

CDynodePing::CDynodePing(const COutPoint& outpoint)
{
    LOCK(cs_main);
    if (!chainActive.Tip() || chainActive.Height() < 12) return;

    vin = CTxIn(outpoint);
    blockHash = chainActive[chainActive.Height() - 12]->GetBlockHash();
    sigTime = GetAdjustedTime();
}

bool CDynodePing::Sign(const CKey& keyDynode, const CPubKey& pubKeyDynode)
{
    std::string strError;
    std::string strDyNodeSignMessage;

    // TODO: add sentinel data
    sigTime = GetAdjustedTime();
    std::string strMessage = vin.ToString() + blockHash.ToString() + boost::lexical_cast<std::string>(sigTime);

    if(!CMessageSigner::SignMessage(strMessage, vchSig, keyDynode)) {
        LogPrintf("CDynodePing::Sign -- SignMessage() failed\n");
        return false;
    }

    if(!CMessageSigner::VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
        LogPrintf("CDynodePing::Sign -- VerifyMessage() failed, error: %s\n", strError);
        return false;
    }

    return true;
}

bool CDynodePing::CheckSignature(CPubKey& pubKeyDynode, int &nDos)
{
    // TODO: add sentinel data
    std::string strMessage = vin.ToString() + blockHash.ToString() + boost::lexical_cast<std::string>(sigTime);
    std::string strError = "";
    nDos = 0;

    if(!CMessageSigner::VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
        LogPrintf("CDynodePing::CheckSignature -- Got bad Dynode ping signature, Dynode=%s, error: %s\n", vin.prevout.ToStringShort(), strError);
        nDos = 33;
        return false;
    }
    return true;
}

bool CDynodePing::SimpleCheck(int& nDos)
{

    // don't ban by default
    nDos = 0;

    if (sigTime > GetAdjustedTime() + 60 * 60) {
        LogPrintf("CDynodePing::SimpleCheck -- Signature rejected, too far into the future, Dynode=%s\n", vin.prevout.ToStringShort());
        nDos = 1;
        return false;
    }

    {
        AssertLockHeld(cs_main);
        BlockMap::iterator mi = mapBlockIndex.find(blockHash);
        if (mi == mapBlockIndex.end()) {
            LogPrint("Dynode", "DynodePing::SimpleCheck -- Dynode ping is invalid, unknown block hash: Dynode=%s blockHash=%s\n", vin.prevout.ToStringShort(), blockHash.ToString());
            // maybe we stuck or forked so we shouldn't ban this node, just fail to accept this ping
            // TODO: or should we also request this block?
            return false;
        }
    }
     LogPrint("Dynode", "CDynodePing::SimpleCheck -- Dynode ping verified: Dynode=%s  blockHash=%s  sigTime=%d\n", vin.prevout.ToStringShort(), blockHash.ToString(), sigTime);
     return true;
 }
 
 bool CDynodePing::CheckAndUpdate(CDynode* pdn, bool fFromNewBroadcast, int& nDos, CConnman& connman)
 {
     // don't ban by default
     nDos = 0;
 
     if (!SimpleCheck(nDos)) {
        return false;
    }

    if (pdn == NULL) {
        LogPrint("Dynode", "CDynodePing::CheckAndUpdate -- Couldn't find Dynode entry, Dynode=%s\n", vin.prevout.ToStringShort());
        return false;
    }

    if(!fFromNewBroadcast) {
        if (pdn->IsUpdateRequired()) {
            LogPrint("Dynode", "CDynodePing::CheckAndUpdate -- Dynode protocol is outdated, Dynode=%s\n", vin.prevout.ToStringShort());
            return false;
        }

        if (pdn->IsNewStartRequired()) {
            LogPrint("Dynode", "CDynodePing::CheckAndUpdate -- Dynode is completely expired, new start is required, Dynode=%s\n", vin.prevout.ToStringShort());
            return false;
       }
    }

    LogPrint("Dynode", "CDynodePing::CheckAndUpdate -- New ping: Dynode=%s  blockHash=%s  sigTime=%d\n", vin.prevout.ToStringShort(), blockHash.ToString(), sigTime);

    // LogPrintf("dnping - Found corresponding dn for vin: %s\n", vin.prevout.ToStringShort());
    // update only if there is no known ping for this Dynode or
    // last ping was more then DYNODE_MIN_DNP_SECONDS-60 ago comparing to this one
    if (pdn->IsPingedWithin(DYNODE_MIN_DNP_SECONDS - 60, sigTime)) {
        LogPrint("Dynode", "CDynodePing::CheckAndUpdate -- Dynode ping arrived too early, Dynode=%s\n", vin.prevout.ToStringShort());
        //nDos = 1; //disable, this is happening frequently and causing banned peers
        return false;
    }

    if (!CheckSignature(pdn->pubKeyDynode, nDos)) return false;

    // so, ping seems to be ok

    // if we are still syncing and there was no known ping for this dn for quite a while
    // (NOTE: assuming that DYNODE_EXPIRATION_SECONDS/2 should be enough to finish dn list sync)
    if(!dynodeSync.IsDynodeListSynced() && !pdn->IsPingedWithin(DYNODE_EXPIRATION_SECONDS/2)) {
        // let's bump sync timeout
        LogPrint("Dynode", "CDynodePing::CheckAndUpdate -- bumping sync timeout, dynode=%s\n", vin.prevout.ToStringShort());
        dynodeSync.BumpAssetLastTime("CDynodePing::CheckAndUpdate");
    }

    // let's store this ping as the last one
    LogPrint("Dynode", "CDynodePing::CheckAndUpdate -- Dynode ping accepted, Dynode=%s\n", vin.prevout.ToStringShort());
    pdn->lastPing = *this;

    // and update dnodeman.mapSeenDynodeBroadcast.lastPing which is probably outdated
    CDynodeBroadcast dnb(*pdn);
    uint256 hash = dnb.GetHash();
    if (dnodeman.mapSeenDynodeBroadcast.count(hash)) {
        dnodeman.mapSeenDynodeBroadcast[hash].second.lastPing = *this;
    }

    // force update, ignoring cache
    pdn->Check(true);
    // relay ping for nodes in ENABLED/EXPIRED/WATCHDOG_EXPIRED state only, skip everyone else
    if (!pdn->IsEnabled() && !pdn->IsExpired() && !pdn->IsWatchdogExpired()) return false;

    LogPrint("Dynode", "CDynodePing::CheckAndUpdate -- Dynode ping acceepted and relayed, Dynode=%s\n", vin.prevout.ToStringShort());
    Relay(connman);

    return true;
}

void CDynodePing::Relay(CConnman& connman)
{
    CInv inv(MSG_DYNODE_PING, GetHash());
    connman.RelayInv(inv);
}

void CDynode::AddGovernanceVote(uint256 nGovernanceObjectHash)
{
    if(mapGovernanceObjectsVotedOn.count(nGovernanceObjectHash)) {
        mapGovernanceObjectsVotedOn[nGovernanceObjectHash]++;
    } else {
        mapGovernanceObjectsVotedOn.insert(std::make_pair(nGovernanceObjectHash, 1));
    }
}

void CDynode::RemoveGovernanceObject(uint256 nGovernanceObjectHash)
{
    std::map<uint256, int>::iterator it = mapGovernanceObjectsVotedOn.find(nGovernanceObjectHash);
    if(it == mapGovernanceObjectsVotedOn.end()) {
        return;
    }
    mapGovernanceObjectsVotedOn.erase(it);
}

void CDynode::UpdateWatchdogVoteTime(uint64_t nVoteTime)
{
    LOCK(cs);
    nTimeLastWatchdogVote = (nVoteTime == 0) ? GetAdjustedTime() : nVoteTime;
}

/**
*   FLAG GOVERNANCE ITEMS AS DIRTY
*
*   - When Dynode come and go on the network, we must flag the items they voted on to recalc it's cached flags
*
*/
void CDynode::FlagGovernanceItemsAsDirty()
{
    std::vector<uint256> vecDirty;
    {
        std::map<uint256, int>::iterator it = mapGovernanceObjectsVotedOn.begin();
        while(it != mapGovernanceObjectsVotedOn.end()) {
            vecDirty.push_back(it->first);
            ++it;
        }
    }
    for(size_t i = 0; i < vecDirty.size(); ++i) {
        dnodeman.AddDirtyGovernanceObjectHash(vecDirty[i]);
    }
}
