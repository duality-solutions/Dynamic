// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dynode.h"

#include "init.h"
#include "util.h"
#include "consensus/validation.h"

#include "activedynode.h"
#include "governance.h"
#include "dynode-payments.h"
#include "dynode-sync.h"
#include "dynodeman.h"
#include "privatesend.h"

#include <boost/lexical_cast.hpp>

CDynode::CDynode() :
    vin(),
    addr(),
    pubKeyCollateralAddress(),
    pubKeyDynode(),
    lastPing(),
    vchSig(),
    sigTime(GetAdjustedTime()),
    nLastSsq(0),
    nTimeLastChecked(0),
    nTimeLastPaid(0),
    nTimeLastWatchdogVote(0),
    nActiveState(DYNODE_ENABLED),
    nCacheCollateralBlock(0),
    nBlockLastPaid(0),
    nProtocolVersion(PROTOCOL_VERSION),
    nPoSeBanScore(0),
    nPoSeBanHeight(0),
    fAllowMixingTx(true),
    fUnitTest(false)
{}

CDynode::CDynode(CService addrNew, CTxIn vinNew, CPubKey pubKeyCollateralAddressNew, CPubKey pubKeyDynodeNew, int nProtocolVersionIn) :
    vin(vinNew),
    addr(addrNew),
    pubKeyCollateralAddress(pubKeyCollateralAddressNew),
    pubKeyDynode(pubKeyDynodeNew),
    lastPing(),
    vchSig(),
    sigTime(GetAdjustedTime()),
    nLastSsq(0),
    nTimeLastChecked(0),
    nTimeLastPaid(0),
    nTimeLastWatchdogVote(0),
    nActiveState(DYNODE_ENABLED),
    nCacheCollateralBlock(0),
    nBlockLastPaid(0),
    nProtocolVersion(nProtocolVersionIn),
    nPoSeBanScore(0),
    nPoSeBanHeight(0),
    fAllowMixingTx(true),
    fUnitTest(false)
{}

CDynode::CDynode(const CDynode& other) :
    vin(other.vin),
    addr(other.addr),
    pubKeyCollateralAddress(other.pubKeyCollateralAddress),
    pubKeyDynode(other.pubKeyDynode),
    lastPing(other.lastPing),
    vchSig(other.vchSig),
    sigTime(other.sigTime),
    nLastSsq(other.nLastSsq),
    nTimeLastChecked(other.nTimeLastChecked),
    nTimeLastPaid(other.nTimeLastPaid),
    nTimeLastWatchdogVote(other.nTimeLastWatchdogVote),
    nActiveState(other.nActiveState),
    nCacheCollateralBlock(other.nCacheCollateralBlock),
    nBlockLastPaid(other.nBlockLastPaid),
    nProtocolVersion(other.nProtocolVersion),
    nPoSeBanScore(other.nPoSeBanScore),
    nPoSeBanHeight(other.nPoSeBanHeight),
    fAllowMixingTx(other.fAllowMixingTx),
    fUnitTest(other.fUnitTest)
{}

CDynode::CDynode(const CDynodeBroadcast& dnb) :
    vin(dnb.vin),
    addr(dnb.addr),
    pubKeyCollateralAddress(dnb.pubKeyCollateralAddress),
    pubKeyDynode(dnb.pubKeyDynode),
    lastPing(dnb.lastPing),
    vchSig(dnb.vchSig),
    sigTime(dnb.sigTime),
    nLastSsq(0),
    nTimeLastChecked(0),
    nTimeLastPaid(0),
    nTimeLastWatchdogVote(dnb.sigTime),
    nActiveState(dnb.nActiveState),
    nCacheCollateralBlock(0),
    nBlockLastPaid(0),
    nProtocolVersion(dnb.nProtocolVersion),
    nPoSeBanScore(0),
    nPoSeBanHeight(0),
    fAllowMixingTx(true),
    fUnitTest(false)
{}

//
// When a new Dynode broadcast is sent, update our information
//
bool CDynode::UpdateFromNewBroadcast(CDynodeBroadcast& dnb)
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
    if(dnb.lastPing == CDynodePing() || (dnb.lastPing != CDynodePing() && dnb.lastPing.CheckAndUpdate(this, true, nDos))) {
        lastPing = dnb.lastPing;
        dnodeman.mapSeenDynodePing.insert(std::make_pair(lastPing.GetHash(), lastPing));
    }
    // if it matches our Dynode privkey...
    if(fDyNode && pubKeyDynode == activeDynode.pubKeyDynode) {
        nPoSeBanScore = -DYNODE_POSE_BAN_MAX_SCORE;
        if(nProtocolVersion == PROTOCOL_VERSION) {
            // ... and PROTOCOL_VERSION, then we've been remotely activated ...
            activeDynode.ManageState();
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
    uint256 aux = ArithToUint256(UintToArith256(vin.prevout.hash) + vin.prevout.n);

    CHashWriter ss(SER_GETHASH, PROTOCOL_VERSION);
    ss << blockHash;
    arith_uint256 hash2 = UintToArith256(ss.GetHash());

    CHashWriter ss2(SER_GETHASH, PROTOCOL_VERSION);
    ss2 << blockHash;
    ss2 << aux;
    arith_uint256 hash3 = UintToArith256(ss2.GetHash());

    return (hash3 > hash2 ? hash3 - hash2 : hash2 - hash3);
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

        CCoins coins;
        if(!pcoinsTip->GetCoins(vin.prevout.hash, coins) ||
           (unsigned int)vin.prevout.n>=coins.vout.size() ||
           coins.vout[vin.prevout.n].IsNull()) {
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

        bool fWatchdogActive = dynodeSync.IsSynced() && dnodeman.IsWatchdogActive();
        bool fWatchdogExpired = (fWatchdogActive && ((GetTime() - nTimeLastWatchdogVote) > DYNODE_WATCHDOG_MAX_SECONDS));

        LogPrint("Dynode", "CDynode::Check -- outpoint=%s, nTimeLastWatchdogVote=%d, GetTime()=%d, fWatchdogExpired=%d\n",
                vin.prevout.ToStringShort(), nTimeLastWatchdogVote, GetTime(), fWatchdogExpired);

        // (TODO):: Check to see if WATCHDOG_EXPIRED is fixed or enable once Dynode network has grown       
        if(fWatchdogExpired) {
            nActiveState = DYNODE_WATCHDOG_EXPIRED;
            if(nActiveStatePrev != nActiveState) {
                LogPrint("Dynode", "CDynode::Check -- Dynode %s is in %s state now\n", vin.prevout.ToStringShort(), GetStateString());
            }
            return;
        }

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
    dynode_info_t info;
    info.vin = vin;
    info.addr = addr;
    info.pubKeyCollateralAddress = pubKeyCollateralAddress;
    info.pubKeyDynode = pubKeyDynode;
    info.sigTime = sigTime;
    info.nLastSsq = nLastSsq;
    info.nTimeLastChecked = nTimeLastChecked;
    info.nTimeLastPaid = nTimeLastPaid;
    info.nTimeLastWatchdogVote = nTimeLastWatchdogVote;
    info.nTimeLastPing = lastPing.sigTime;
    info.nActiveState = nActiveState;
    info.nProtocolVersion = nProtocolVersion;
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

int CDynode::GetCollateralAge()
{
    int nHeight;
    {
        TRY_LOCK(cs_main, lockMain);
        if(!lockMain || !chainActive.Tip()) return -1;
        nHeight = chainActive.Height();
    }

    if (nCacheCollateralBlock == 0) {
        int nInputAge = GetInputAge(vin);
        if(nInputAge > 0) {
            nCacheCollateralBlock = nHeight - nInputAge;
        } else {
            return nInputAge;
        }
    }

    return nHeight - nCacheCollateralBlock;
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

            CAmount nDynodePayment = STATIC_DYNODE_PAYMENT;

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
    CTxIn txin;
    CPubKey pubKeyCollateralAddressNew;
    CKey keyCollateralAddressNew;
    CPubKey pubKeyDynodeNew;
    CKey keyDynodeNew;

    //need correct blocks to send ping
    if(!fOffline && !dynodeSync.IsBlockchainSynced()) {
        strErrorRet = "Sync in progress. Must wait until sync is complete to start Dynode";
        LogPrintf("CDynodeBroadcast::Create -- %s\n", strErrorRet);
        return false;
    }

    if(!privateSendSigner.GetKeysFromSecret(strKeyDynode, keyDynodeNew, pubKeyDynodeNew)) {
        strErrorRet = strprintf("Invalid Dynode key %s", strKeyDynode);
        LogPrintf("CDynodeBroadcast::Create -- %s\n", strErrorRet);
        return false;
    }

    if(!pwalletMain->GetDynodeVinAndKeys(txin, pubKeyCollateralAddressNew, keyCollateralAddressNew, strTxHash, strOutputIndex)) {
        strErrorRet = strprintf("Could not allocate txin %s:%s for Dynode %s", strTxHash, strOutputIndex, strService);
        LogPrintf("CDynodeBroadcast::Create -- %s\n", strErrorRet);
        return false;
    }

    CService service = CService(strService);
    int mainnetDefaultPort = Params(CBaseChainParams::MAIN).GetDefaultPort();
    if(Params().NetworkIDString() == CBaseChainParams::MAIN) {
        if(service.GetPort() != mainnetDefaultPort) {
            strErrorRet = strprintf("Invalid port %u for Dynode %s, only %d is supported on mainnet.", service.GetPort(), strService, mainnetDefaultPort);
            LogPrintf("CDynodeBroadcast::Create -- %s\n", strErrorRet);
            return false;
        }
    } else if (service.GetPort() == mainnetDefaultPort) {
        strErrorRet = strprintf("Invalid port %u for Dynode %s, %d is the only supported on mainnet.", service.GetPort(), strService, mainnetDefaultPort);
        LogPrintf("CDynodeBroadcast::Create -- %s\n", strErrorRet);
        return false;
    }

    BOOST_FOREACH(CNode* pnode, vNodes) {
    if (pnode->addr.IsIPv6()) {
        LogPrintf("Invalid protocol for Dynode, only IPv4 is supported.");
        return false;
        }
    }

    return Create(txin, CService(strService), keyCollateralAddressNew, pubKeyCollateralAddressNew, keyDynodeNew, pubKeyDynodeNew, strErrorRet, dnbRet);
}

bool CDynodeBroadcast::Create(CTxIn txin, CService service, CKey keyCollateralAddressNew, CPubKey pubKeyCollateralAddressNew, CKey keyDynodeNew, CPubKey pubKeyDynodeNew, std::string &strErrorRet, CDynodeBroadcast &dnbRet)
{
    // wait for reindex and/or import to finish
    if (fImporting || fReindex) return false;

    LogPrint("Dynode", "CDynodeBroadcast::Create -- pubKeyCollateralAddressNew = %s, pubKeyDynodeNew.GetID() = %s\n",
             CDynamicAddress(pubKeyCollateralAddressNew.GetID()).ToString(),
             pubKeyDynodeNew.GetID().ToString());


    CDynodePing dnp(txin);
    if(!dnp.Sign(keyDynodeNew, pubKeyDynodeNew)) {
        strErrorRet = strprintf("Failed to sign ping, Dynode=%s", txin.prevout.ToStringShort());
        LogPrintf("CDynodeBroadcast::Create -- %s\n", strErrorRet);
        dnbRet = CDynodeBroadcast();
        return false;
    }

    dnbRet = CDynodeBroadcast(service, txin, pubKeyCollateralAddressNew, pubKeyDynodeNew, PROTOCOL_VERSION);

    if(!dnbRet.IsValidNetAddr()) {
        strErrorRet = strprintf("Invalid IP address, Dynode=%s", txin.prevout.ToStringShort());
        LogPrintf("CDynodeBroadcast::Create -- %s\n", strErrorRet);
        dnbRet = CDynodeBroadcast();
        return false;
    }

    dnbRet.lastPing = dnp;
    if(!dnbRet.Sign(keyCollateralAddressNew)) {
        strErrorRet = strprintf("Failed to sign broadcast, Dynode=%s", txin.prevout.ToStringShort());
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

    int mainnetDefaultPort = Params(CBaseChainParams::MAIN).GetDefaultPort();
    if(Params().NetworkIDString() == CBaseChainParams::MAIN) {
        if(addr.GetPort() != mainnetDefaultPort) return false;
    } else if(addr.GetPort() == mainnetDefaultPort) return false;

    return true;
}

bool CDynodeBroadcast::Update(CDynode* pdn, int& nDos)
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
        if(pdn->UpdateFromNewBroadcast((*this))) {
            pdn->Check();
            Relay();
        }
        dynodeSync.AddedDynodeList();
    }

    return true;
}

bool CDynodeBroadcast::CheckOutpoint(int& nDos)
{
    // we are a Dynode with the same vin (i.e. already activated) and this dnb is ours (matches our Dynodes privkey)
    // so nothing to do here for us
    if(fDyNode && vin.prevout == activeDynode.vin.prevout && pubKeyDynode == activeDynode.pubKeyDynode) {
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

        CCoins coins;
        if(!pcoinsTip->GetCoins(vin.prevout.hash, coins) ||
           (unsigned int)vin.prevout.n>=coins.vout.size() ||
           coins.vout[vin.prevout.n].IsNull()) {
            LogPrint("Dynode", "CDynodeBroadcast::CheckOutpoint -- Failed to find Dynode UTXO, Dynode=%s\n", vin.prevout.ToStringShort());
            return false;
        }
        if(coins.vout[vin.prevout.n].nValue != 1000 * COIN) {
            LogPrint("Dynode", "CDynodeBroadcast::CheckOutpoint -- Dynode UTXO should have 1000 DYN, Dynode=%s\n", vin.prevout.ToStringShort());
            return false;
        }
        if(chainActive.Height() - coins.nHeight + 1 < Params().GetConsensus().nDynodeMinimumConfirmations) {
            LogPrintf("CDynodeBroadcast::CheckOutpoint -- Dynode UTXO must have at least %d confirmations, Dynode=%s\n",
                    Params().GetConsensus().nDynodeMinimumConfirmations, vin.prevout.ToStringShort());
            // maybe we miss few blocks, let this dnb to be checked again later
            dnodeman.mapSeenDynodeBroadcast.erase(GetHash());
            return false;
        }
    }

    LogPrint("Dynode", "CDynodeBroadcast::CheckOutpoint -- Dynode UTXO verified\n");

    // make sure the vout that was signed is related to the transaction that spawned the Dynode
    //  - this is expensive, so it's only done once per Dynode
    if(!privateSendSigner.IsVinAssociatedWithPubkey(vin, pubKeyCollateralAddress)) {
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

bool CDynodeBroadcast::Sign(CKey& keyCollateralAddress)
{
    std::string strError;
    std::string strMessage;

    sigTime = GetAdjustedTime();

    strMessage = addr.ToString(false) + boost::lexical_cast<std::string>(sigTime) +
                    pubKeyCollateralAddress.GetID().ToString() + pubKeyDynode.GetID().ToString() +
                    boost::lexical_cast<std::string>(nProtocolVersion);

    if(!privateSendSigner.SignMessage(strMessage, vchSig, keyCollateralAddress)) {
        LogPrintf("CDynodeBroadcast::Sign -- SignMessage() failed\n");
        return false;
    }

    if(!privateSendSigner.VerifyMessage(pubKeyCollateralAddress, vchSig, strMessage, strError)) {
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

    if(!privateSendSigner.VerifyMessage(pubKeyCollateralAddress, vchSig, strMessage, strError)){
        LogPrintf("CDynodeBroadcast::CheckSignature -- Got bad Dynode announce signature, error: %s\n", strError);
        nDos = 100;
        return false;
    }

    return true;
}

void CDynodeBroadcast::Relay()
{
    CInv inv(MSG_DYNODE_ANNOUNCE, GetHash());
    RelayInv(inv);
}

CDynodePing::CDynodePing(CTxIn& vinNew)
{
    LOCK(cs_main);
    if (!chainActive.Tip() || chainActive.Height() < 12) return;

    vin = vinNew;
    blockHash = chainActive[chainActive.Height() - 12]->GetBlockHash();
    sigTime = GetAdjustedTime();
    vchSig = std::vector<unsigned char>();
}

bool CDynodePing::Sign(CKey& keyDynode, CPubKey& pubKeyDynode)
{
    std::string strError;
    std::string strDyNodeSignMessage;

    sigTime = GetAdjustedTime();
    std::string strMessage = vin.ToString() + blockHash.ToString() + boost::lexical_cast<std::string>(sigTime);

    if(!privateSendSigner.SignMessage(strMessage, vchSig, keyDynode)) {
        LogPrintf("CDynodePing::Sign -- SignMessage() failed\n");
        return false;
    }

    if(!privateSendSigner.VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
        LogPrintf("CDynodePing::Sign -- VerifyMessage() failed, error: %s\n", strError);
        return false;
    }

    return true;
}

bool CDynodePing::CheckSignature(CPubKey& pubKeyDynode, int &nDos)
{
    std::string strMessage = vin.ToString() + blockHash.ToString() + boost::lexical_cast<std::string>(sigTime);
    std::string strError = "";
    nDos = 0;

    if(!privateSendSigner.VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
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
        LOCK(cs_main);
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
 
 bool CDynodePing::CheckAndUpdate(CDynode* pdn, bool fFromNewBroadcast, int& nDos)
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

    // so, ping seems to be ok, let's store it
    LogPrint("Dynode", "CDynodePing::CheckAndUpdate -- Dynode ping accepted, Dynode=%s\n", vin.prevout.ToStringShort());
    pdn->lastPing = *this;

    // and update dnodeman.mapSeenDynodeBroadcast.lastPing which is probably outdated
    CDynodeBroadcast dnb(*pdn);
    uint256 hash = dnb.GetHash();
    if (dnodeman.mapSeenDynodeBroadcast.count(hash)) {
        dnodeman.mapSeenDynodeBroadcast[hash].second.lastPing = *this;
    }

    pdn->Check(true); // force update, ignoring cache
    if (!pdn->IsEnabled()) return false;

    LogPrint("Dynode", "CDynodePing::CheckAndUpdate -- Dynode ping acceepted and relayed, Dynode=%s\n", vin.prevout.ToStringShort());
    Relay();

    return true;
}

void CDynodePing::Relay()
{
    CInv inv(MSG_DYNODE_PING, GetHash());
    RelayInv(inv);
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

void CDynode::UpdateWatchdogVoteTime()
{
    LOCK(cs);
    nTimeLastWatchdogVote = GetTime();
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
