// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "activestormnode.h"
#include "consensus/validation.h"
#include "sandstorm.h"
#include "init.h"
#include "governance.h"
#include "stormnode.h"
#include "stormnode-payments.h"
#include "stormnode-sync.h"
#include "stormnodeman.h"
#include "util.h"

#include <boost/lexical_cast.hpp>


CStormnode::CStormnode() :
    vin(),
    addr(),
    pubKeyCollateralAddress(),
    pubKeyStormnode(),
    lastPing(),
    vchSig(),
    sigTime(GetAdjustedTime()),
    nLastSsq(0),
    nTimeLastChecked(0),
    nTimeLastPaid(0),
    nTimeLastWatchdogVote(0),
    nActiveState(STORMNODE_ENABLED),
    nCacheCollateralBlock(0),
    nBlockLastPaid(0),
    nProtocolVersion(PROTOCOL_VERSION),
    nPoSeBanScore(0),
    nPoSeBanHeight(0),
    fAllowMixingTx(true),
    fUnitTest(false)
{}

CStormnode::CStormnode(CService addrNew, CTxIn vinNew, CPubKey pubKeyCollateralAddressNew, CPubKey pubKeyStormnodeNew, int nProtocolVersionIn) :
    vin(vinNew),
    addr(addrNew),
    pubKeyCollateralAddress(pubKeyCollateralAddressNew),
    pubKeyStormnode(pubKeyStormnodeNew),
    lastPing(),
    vchSig(),
    sigTime(GetAdjustedTime()),
    nLastSsq(0),
    nTimeLastChecked(0),
    nTimeLastPaid(0),
    nTimeLastWatchdogVote(0),
    nActiveState(STORMNODE_ENABLED),
    nCacheCollateralBlock(0),
    nBlockLastPaid(0),
    nProtocolVersion(nProtocolVersionIn),
    nPoSeBanScore(0),
    nPoSeBanHeight(0),
    fAllowMixingTx(true),
    fUnitTest(false)
{}

CStormnode::CStormnode(const CStormnode& other) :
    vin(other.vin),
    addr(other.addr),
    pubKeyCollateralAddress(other.pubKeyCollateralAddress),
    pubKeyStormnode(other.pubKeyStormnode),
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

CStormnode::CStormnode(const CStormnodeBroadcast& snb) :
    vin(snb.vin),
    addr(snb.addr),
    pubKeyCollateralAddress(snb.pubKeyCollateralAddress),
    pubKeyStormnode(snb.pubKeyStormnode),
    lastPing(snb.lastPing),
    vchSig(snb.vchSig),
    sigTime(snb.sigTime),
    nLastSsq(0),
    nTimeLastChecked(0),
    nTimeLastPaid(0),
    nTimeLastWatchdogVote(snb.sigTime),
    nActiveState(snb.nActiveState),
    nCacheCollateralBlock(0),
    nBlockLastPaid(0),
    nProtocolVersion(snb.nProtocolVersion),
    nPoSeBanScore(0),
    nPoSeBanHeight(0),
    fAllowMixingTx(true),
    fUnitTest(false)
{}

//
// When a new Stormnode broadcast is sent, update our information
//
bool CStormnode::UpdateFromNewBroadcast(CStormnodeBroadcast& snb)
{
    if(snb.sigTime <= sigTime && !snb.fRecovery) return false;

    pubKeyStormnode = snb.pubKeyStormnode;
    sigTime = snb.sigTime;
    vchSig = snb.vchSig;
    nProtocolVersion = snb.nProtocolVersion;
    addr = snb.addr;
    nPoSeBanScore = 0;
    nPoSeBanHeight = 0;
    nTimeLastChecked = 0;
    int nDos = 0;
    if(snb.lastPing == CStormnodePing() || (snb.lastPing != CStormnodePing() && snb.lastPing.CheckAndUpdate(this, true, nDos))) {
        lastPing = snb.lastPing;
        snodeman.mapSeenStormnodePing.insert(std::make_pair(lastPing.GetHash(), lastPing));
    }
    // if it matches our Stormnode privkey...
    if(fStormNode && pubKeyStormnode == activeStormnode.pubKeyStormnode) {
        nPoSeBanScore = -STORMNODE_POSE_BAN_MAX_SCORE;
        if(nProtocolVersion == PROTOCOL_VERSION) {
            // ... and PROTOCOL_VERSION, then we've been remotely activated ...
            activeStormnode.ManageState();
        } else {
            // ... otherwise we need to reactivate our node, do not add it to the list and do not relay
            // but also do not ban the node we get this message from
            LogPrintf("CStormnode::UpdateFromNewBroadcast -- wrong PROTOCOL_VERSION, re-activate your SN: message nProtocolVersion=%d  PROTOCOL_VERSION=%d\n", nProtocolVersion, PROTOCOL_VERSION);
            return false;
        }
    }
    return true;
}

//
// Deterministically calculate a given "score" for a Stormnode depending on how close it's hash is to
// the proof of work for that block. The further away they are the better, the furthest will win the election
// and get paid this block
//
arith_uint256 CStormnode::CalculateScore(const uint256& blockHash)
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

void CStormnode::Check(bool fForce)
{
    LOCK(cs);

    if(ShutdownRequested()) return;

    if(!fForce && (GetTime() - nTimeLastChecked < STORMNODE_CHECK_SECONDS)) return;
    nTimeLastChecked = GetTime();

   LogPrint("Stormnode", "CStormnode::Check -- Stormnode %s is in %s state\n", vin.prevout.ToStringShort(), GetStateString());

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
            nActiveState = STORMNODE_OUTPOINT_SPENT;
            LogPrint("Stormnode", "CStormnode::Check -- Failed to find Stormnode UTXO, Stormnode=%s\n", vin.prevout.ToStringShort());
            return;
        }

        nHeight = chainActive.Height();
    }

    if(IsPoSeBanned()) {
        if(nHeight < nPoSeBanHeight) return; // too early?
        // Otherwise give it a chance to proceed further to do all the usual checks and to change its state.
        // Stormnode still will be on the edge and can be banned back easily if it keeps ignoring snverify
        // or connect attempts. Will require few snverify messages to strengthen its position in sn list.
        LogPrintf("CStormnode::Check -- Stormnode %s is unbanned and back in list now\n", vin.prevout.ToStringShort());
        DecreasePoSeBanScore();
    } else if(nPoSeBanScore >= STORMNODE_POSE_BAN_MAX_SCORE) {
        nActiveState = STORMNODE_POSE_BAN;
        // ban for the whole payment cycle
        nPoSeBanHeight = nHeight + snodeman.size();
        LogPrintf("CStormnode::Check -- Stormnode %s is banned till block %d now\n", vin.prevout.ToStringShort(), nPoSeBanHeight);
        return;
    }

    int nActiveStatePrev = nActiveState;
    bool fOurStormnode = fStormNode && activeStormnode.pubKeyStormnode == pubKeyStormnode;
                   // Stormnode doesn't meet payment protocol requirements ...
    bool fRequireUpdate = nProtocolVersion < snpayments.GetMinStormnodePaymentsProto() ||
                   // or it's our own node and we just updated it to the new protocol but we are still waiting for activation ...
                   (fOurStormnode && nProtocolVersion < PROTOCOL_VERSION);

    if(fRequireUpdate) {
        nActiveState = STORMNODE_UPDATE_REQUIRED;
        if(nActiveStatePrev != nActiveState) {
            LogPrint("Stormnode", "CStormnode::Check -- Stormnode %s is in %s state now\n", vin.prevout.ToStringShort(), GetStateString());
        }
        return;
    }

    // keep old Stormnodes on start, give them a chance to receive updates...
    bool fWaitForPing = !stormnodeSync.IsStormnodeListSynced() && !IsPingedWithin(STORMNODE_MIN_SNP_SECONDS);

    if(fWaitForPing && !fOurStormnode) {
        // ...but if it was already expired before the initial check - return right away
        if(IsExpired() || IsWatchdogExpired() || IsNewStartRequired()) {
            LogPrint("Stormnode", "CStormnode::Check -- Stormnode %s is in %s state, waiting for ping\n", vin.prevout.ToStringShort(), GetStateString());
            return;
        }
    }

    // don't expire if we are still in "waiting for ping" mode unless it's our own Stormnode
    if(!fWaitForPing || fOurStormnode) {

        if(!IsPingedWithin(STORMNODE_NEW_START_REQUIRED_SECONDS)) {
            nActiveState = STORMNODE_NEW_START_REQUIRED;
            if(nActiveStatePrev != nActiveState) {
                LogPrint("Stormnode", "CStormnode::Check -- Stormnode %s is in %s state now\n", vin.prevout.ToStringShort(), GetStateString());
            }
            return;
        }

        bool fWatchdogActive = stormnodeSync.IsSynced() && snodeman.IsWatchdogActive();
        bool fWatchdogExpired = (fWatchdogActive && ((GetTime() - nTimeLastWatchdogVote) > STORMNODE_WATCHDOG_MAX_SECONDS));

        LogPrint("Stormnode", "CStormnode::Check -- outpoint=%s, nTimeLastWatchdogVote=%d, GetTime()=%d, fWatchdogExpired=%d\n",
                vin.prevout.ToStringShort(), nTimeLastWatchdogVote, GetTime(), fWatchdogExpired);

        if(fWatchdogExpired) {
            nActiveState = STORMNODE_WATCHDOG_EXPIRED;
            if(nActiveStatePrev != nActiveState) {
                LogPrint("Stormnode", "CStormnode::Check -- Stormnode %s is in %s state now\n", vin.prevout.ToStringShort(), GetStateString());
            }
            return;
        }

        if(!IsPingedWithin(STORMNODE_EXPIRATION_SECONDS)) {
            nActiveState = STORMNODE_EXPIRED;
            if(nActiveStatePrev != nActiveState) {
                LogPrint("Stormnode", "CStormnode::Check -- Stormnode %s is in %s state now\n", vin.prevout.ToStringShort(), GetStateString());
            }
            return;
        }
    }

    if(lastPing.sigTime - sigTime < STORMNODE_MIN_SNP_SECONDS) {
        nActiveState = STORMNODE_PRE_ENABLED;
        if(nActiveStatePrev != nActiveState) {
            LogPrint("Stormnode", "CStormnode::Check -- Stormnode %s is in %s state now\n", vin.prevout.ToStringShort(), GetStateString());
        }
        return;
    }

    nActiveState = STORMNODE_ENABLED; // OK
    if(nActiveStatePrev != nActiveState) {
        LogPrint("Stormnode", "CStormnode::Check -- Stormnode %s is in %s state now\n", vin.prevout.ToStringShort(), GetStateString());
    }
}

bool CStormnode::IsValidNetAddr()
{
    return IsValidNetAddr(addr);
}

bool CStormnode::IsValidNetAddr(CService addrIn)
{
    // TODO: regtest is fine with any addresses for now,
    // should probably be a bit smarter if one day we start to implement tests for this
    return Params().NetworkIDString() == CBaseChainParams::REGTEST ||
            (addrIn.IsIPv4() && IsReachable(addrIn) && addrIn.IsRoutable());
}

stormnode_info_t CStormnode::GetInfo()
{
    stormnode_info_t info;
    info.vin = vin;
    info.addr = addr;
    info.pubKeyCollateralAddress = pubKeyCollateralAddress;
    info.pubKeyStormnode = pubKeyStormnode;
    info.sigTime = sigTime;
    info.nLastSsq = nLastSsq;
    info.nTimeLastChecked = nTimeLastChecked;
    info.nTimeLastPaid = nTimeLastPaid;
    info.nTimeLastWatchdogVote = nTimeLastWatchdogVote;
    info.nActiveState = nActiveState;
    info.nProtocolVersion = nProtocolVersion;
    info.fInfoValid = true;
    return info;
}

std::string CStormnode::StateToString(int nStateIn)
{
    switch(nStateIn) {
        case STORMNODE_PRE_ENABLED:            return "PRE_ENABLED";
        case STORMNODE_ENABLED:                return "ENABLED";
        case STORMNODE_EXPIRED:                return "EXPIRED";
        case STORMNODE_OUTPOINT_SPENT:         return "OUTPOINT_SPENT";
        case STORMNODE_UPDATE_REQUIRED:        return "UPDATE_REQUIRED";
        case STORMNODE_WATCHDOG_EXPIRED:       return "WATCHDOG_EXPIRED";
        case STORMNODE_NEW_START_REQUIRED:     return "NEW_START_REQUIRED";
        case STORMNODE_POSE_BAN:               return "POSE_BAN";
        default:                               return "UNKNOWN";
    }
}

std::string CStormnode::GetStateString() const
{
    return StateToString(nActiveState);
}

std::string CStormnode::GetStatus() const
{
    // TODO: return smth a bit more human readable here
    return GetStateString();
}

int CStormnode::GetCollateralAge()
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

void CStormnode::UpdateLastPaid(const CBlockIndex *pindex, int nMaxBlocksToScanBack)
{
    if(!pindex) return;

    const CBlockIndex *BlockReading = pindex;

    CScript snpayee = GetScriptForDestination(pubKeyCollateralAddress.GetID());
    // LogPrint("Stormnode", "CStormnode::UpdateLastPaidBlock -- searching for block with payment to %s\n", vin.prevout.ToStringShort());

    LOCK(cs_mapStormnodeBlocks);

    for (int i = 0; BlockReading && BlockReading->nHeight > nBlockLastPaid && i < nMaxBlocksToScanBack; i++) {
        if(snpayments.mapStormnodeBlocks.count(BlockReading->nHeight) &&
            snpayments.mapStormnodeBlocks[BlockReading->nHeight].HasPayeeWithVotes(snpayee, 2))
        {
            CBlock block;
            if(!ReadBlockFromDisk(block, BlockReading, Params().GetConsensus())) // shouldn't really happen
                continue;

            CAmount nStormnodePayment = STATIC_STORMNODE_PAYMENT;

            BOOST_FOREACH(CTxOut txout, block.vtx[0].vout)
                if(snpayee == txout.scriptPubKey && nStormnodePayment == txout.nValue) {
                    nBlockLastPaid = BlockReading->nHeight;
                    nTimeLastPaid = BlockReading->nTime;
                    LogPrint("Stormnode", "CStormnode::UpdateLastPaidBlock -- searching for block with payment to %s -- found new %d\n", vin.prevout.ToStringShort(), nBlockLastPaid);
                    return;
                }
        }

        if (BlockReading->pprev == NULL) { assert(BlockReading); break; }
        BlockReading = BlockReading->pprev;
    }

    // Last payment for this Stormnode wasn't found in latest snpayments blocks
    // or it was found in snpayments blocks but wasn't found in the blockchain.
    // LogPrint("Stormnode", "CStormnode::UpdateLastPaidBlock -- searching for block with payment to %s -- keeping old %d\n", vin.prevout.ToStringShort(), nBlockLastPaid);
}

bool CStormnodeBroadcast::Create(std::string strService, std::string strKeyStormnode, std::string strTxHash, std::string strOutputIndex, std::string& strErrorRet, CStormnodeBroadcast &snbRet, bool fOffline)
{
    CTxIn txin;
    CPubKey pubKeyCollateralAddressNew;
    CKey keyCollateralAddressNew;
    CPubKey pubKeyStormnodeNew;
    CKey keyStormnodeNew;

    //need correct blocks to send ping
    if(!fOffline && !stormnodeSync.IsBlockchainSynced()) {
        strErrorRet = "Sync in progress. Must wait until sync is complete to start Stormnode";
        LogPrintf("CStormnodeBroadcast::Create -- %s\n", strErrorRet);
        return false;
    }

    if(!sandStormSigner.GetKeysFromSecret(strKeyStormnode, keyStormnodeNew, pubKeyStormnodeNew)) {
        strErrorRet = strprintf("Invalid Stormnode key %s", strKeyStormnode);
        LogPrintf("CStormnodeBroadcast::Create -- %s\n", strErrorRet);
        return false;
    }

    if(!pwalletMain->GetStormnodeVinAndKeys(txin, pubKeyCollateralAddressNew, keyCollateralAddressNew, strTxHash, strOutputIndex)) {
        strErrorRet = strprintf("Could not allocate txin %s:%s for Stormnode %s", strTxHash, strOutputIndex, strService);
        LogPrintf("CStormnodeBroadcast::Create -- %s\n", strErrorRet);
        return false;
    }

    /*
    CService service = CService(strService);
    int mainnetDefaultPort = Params(CBaseChainParams::MAIN).GetDefaultPort();
    if(Params().NetworkIDString() == CBaseChainParams::MAIN) {
        if(service.GetPort() != mainnetDefaultPort) {
            strErrorRet = strprintf("Invalid port %u for Stormnode %s, only %d is supported on mainnet.", service.GetPort(), strService, mainnetDefaultPort);
            LogPrintf("CStormnodeBroadcast::Create -- %s\n", strErrorRet);
            return false;
        }
    } else if (service.GetPort() == mainnetDefaultPort) {
        strErrorRet = strprintf("Invalid port %u for Stormnode %s, %d is the only supported on mainnet.", service.GetPort(), strService, mainnetDefaultPort);
        LogPrintf("CStormnodeBroadcast::Create -- %s\n", strErrorRet);
        return false;
    }
    */

    return Create(txin, CService(strService), keyCollateralAddressNew, pubKeyCollateralAddressNew, keyStormnodeNew, pubKeyStormnodeNew, strErrorRet, snbRet);
}

bool CStormnodeBroadcast::Create(CTxIn txin, CService service, CKey keyCollateralAddressNew, CPubKey pubKeyCollateralAddressNew, CKey keyStormnodeNew, CPubKey pubKeyStormnodeNew, std::string &strErrorRet, CStormnodeBroadcast &snbRet)
{
    // wait for reindex and/or import to finish
    if (fImporting || fReindex) return false;

    LogPrint("Stormnode", "CStormnodeBroadcast::Create -- pubKeyCollateralAddressNew = %s, pubKeyStormnodeNew.GetID() = %s\n",
             CDarkSilkAddress(pubKeyCollateralAddressNew.GetID()).ToString(),
             pubKeyStormnodeNew.GetID().ToString());


    CStormnodePing snp(txin);
    if(!snp.Sign(keyStormnodeNew, pubKeyStormnodeNew)) {
        strErrorRet = strprintf("Failed to sign ping, Stormnode=%s", txin.prevout.ToStringShort());
        LogPrintf("CStormnodeBroadcast::Create -- %s\n", strErrorRet);
        snbRet = CStormnodeBroadcast();
        return false;
    }

    snbRet = CStormnodeBroadcast(service, txin, pubKeyCollateralAddressNew, pubKeyStormnodeNew, PROTOCOL_VERSION);

    if(!snbRet.IsValidNetAddr()) {
        strErrorRet = strprintf("Invalid IP address, Stormnode=%s", txin.prevout.ToStringShort());
        LogPrintf("CStormnodeBroadcast::Create -- %s\n", strErrorRet);
        snbRet = CStormnodeBroadcast();
        return false;
    }

    snbRet.lastPing = snp;
    if(!snbRet.Sign(keyCollateralAddressNew)) {
        strErrorRet = strprintf("Failed to sign broadcast, Stormnode=%s", txin.prevout.ToStringShort());
        LogPrintf("CStormnodeBroadcast::Create -- %s\n", strErrorRet);
        snbRet = CStormnodeBroadcast();
        return false;
    }

    return true;
}

bool CStormnodeBroadcast::SimpleCheck(int& nDos)
{
    nDos = 0;

    // make sure addr is valid
    if(!IsValidNetAddr()) {
        LogPrintf("CStormnodeBroadcast::SimpleCheck -- Invalid addr, rejected: Stormnode=%s  addr=%s\n",
                    vin.prevout.ToStringShort(), addr.ToString());
        return false;
    }

    // make sure signature isn't in the future (past is OK)
    if (sigTime > GetAdjustedTime() + 60 * 60) {
        LogPrintf("CStormnodeBroadcast::SimpleCheck -- Signature rejected, too far into the future: Stormnode=%s\n", vin.prevout.ToStringShort());
        nDos = 1;
        return false;
    }

    // empty ping or incorrect sigTime/unknown blockhash
    if(lastPing == CStormnodePing() || !lastPing.SimpleCheck(nDos)) {
        // one of us is probably forked or smth, just mark it as expired and check the rest of the rules
        nActiveState = STORMNODE_EXPIRED;
    }

    if(nProtocolVersion < snpayments.GetMinStormnodePaymentsProto()) {
        LogPrintf("CStormnodeBroadcast::SimpleCheck -- ignoring outdated Stormnode: Stormnode=%s  nProtocolVersion=%d\n", vin.prevout.ToStringShort(), nProtocolVersion);
        return false;
    }

    CScript pubkeyScript;
    pubkeyScript = GetScriptForDestination(pubKeyCollateralAddress.GetID());

    if(pubkeyScript.size() != 25) {
        LogPrintf("CStormnodeBroadcast::SimpleCheck -- pubKeyCollateralAddress has the wrong size\n");
        nDos = 100;
        return false;
    }

    CScript pubkeyScript2;
    pubkeyScript2 = GetScriptForDestination(pubKeyStormnode.GetID());

    if(pubkeyScript2.size() != 25) {
        LogPrintf("CStormnodeBroadcast::SimpleCheck -- pubKeyStormnode has the wrong size\n");
        nDos = 100;
        return false;
    }

    if(!vin.scriptSig.empty()) {
        LogPrintf("CStormnodeBroadcast::SimpleCheck -- Ignore Not Empty ScriptSig %s\n",vin.ToString());
        nDos = 100;
        return false;
    }

    int mainnetDefaultPort = Params(CBaseChainParams::MAIN).GetDefaultPort();
    if(Params().NetworkIDString() == CBaseChainParams::MAIN) {
        if(addr.GetPort() != mainnetDefaultPort) return false;
    } else if(addr.GetPort() == mainnetDefaultPort) return false;

    return true;
}

bool CStormnodeBroadcast::Update(CStormnode* psn, int& nDos)
{
    nDos = 0;

    if(psn->sigTime == sigTime && !fRecovery) {
        // mapSeenStormnodeBroadcast in CStormnodeMan::CheckSnbAndUpdateStormnodeList should filter legit duplicates
        // but this still can happen if we just started, which is ok, just do nothing here.
        return false;
    }

    // this broadcast is older than the one that we already have - it's bad and should never happen
    // unless someone is doing something fishy
    if(psn->sigTime > sigTime) {
        LogPrintf("CStormnodeBroadcast::Update -- Bad sigTime %d (existing broadcast is at %d) for Stormnode %s %s\n",
                      sigTime, psn->sigTime, vin.prevout.ToStringShort(), addr.ToString());
        return false;
    }

    psn->Check();

    // Stormnode is banned by PoSe
    if(psn->IsPoSeBanned()) {
        LogPrintf("CStormnodeBroadcast::Update -- Banned by PoSe, Stormnode=%s\n", vin.prevout.ToStringShort());
        return false;
    }

    // IsVnAssociatedWithPubkey is validated once in CheckOutpoint, after that they just need to match
    if(psn->pubKeyCollateralAddress != pubKeyCollateralAddress) {
        LogPrintf("CStormnodeBroadcast::Update -- Got mismatched pubKeyCollateralAddress and vin\n");
        nDos = 33;
        return false;
    }

    if (!CheckSignature(nDos)) {
        LogPrintf("CStormnodeBroadcast::Update -- CheckSignature() failed, Stormnode=%s\n", vin.prevout.ToStringShort());
        return false;
    }

    // if there was no Stormnode broadcast recently or if it matches our Stormnode privkey...
    if(!psn->IsBroadcastedWithin(STORMNODE_MIN_SNB_SECONDS) || (fStormNode && pubKeyStormnode == activeStormnode.pubKeyStormnode)) {
        // take the newest entry
        LogPrintf("CStormnodeBroadcast::Update -- Got UPDATED Stormnode entry: addr=%s\n", addr.ToString());
        if(psn->UpdateFromNewBroadcast((*this))) {
            psn->Check();
            Relay();
        }
        stormnodeSync.AddedStormnodeList();
    }

    return true;
}

bool CStormnodeBroadcast::CheckOutpoint(int& nDos)
{
    // we are a Stormnode with the same vin (i.e. already activated) and this snb is ours (matches our Stormnodes privkey)
    // so nothing to do here for us
    if(fStormNode && vin.prevout == activeStormnode.vin.prevout && pubKeyStormnode == activeStormnode.pubKeyStormnode) {
        return false;
    }

    if (!CheckSignature(nDos)) {
        LogPrintf("CStormnodeBroadcast::CheckOutpoint -- CheckSignature() failed, Stormnode=%s\n", vin.prevout.ToStringShort());
        return false;
    }

    {
        TRY_LOCK(cs_main, lockMain);
        if(!lockMain) {
            // not snb fault, let it to be checked again later
            LogPrint("Stormnode", "CStormnodeBroadcast::CheckOutpoint -- Failed to aquire lock, addr=%s", addr.ToString());
            snodeman.mapSeenStormnodeBroadcast.erase(GetHash());
            return false;
        }

        CCoins coins;
        if(!pcoinsTip->GetCoins(vin.prevout.hash, coins) ||
           (unsigned int)vin.prevout.n>=coins.vout.size() ||
           coins.vout[vin.prevout.n].IsNull()) {
            LogPrint("Stormnode", "CStormnodeBroadcast::CheckOutpoint -- Failed to find Stormnode UTXO, Stormnode=%s\n", vin.prevout.ToStringShort());
            return false;
        }
        if(coins.vout[vin.prevout.n].nValue != 1000 * COIN) {
            LogPrint("Stormnode", "CStormnodeBroadcast::CheckOutpoint -- Stormnode UTXO should have 1000 DSLK, Stormnode=%s\n", vin.prevout.ToStringShort());
            return false;
        }
        if(chainActive.Height() - coins.nHeight + 1 < Params().GetConsensus().nStormnodeMinimumConfirmations) {
            LogPrintf("CStormnodeBroadcast::CheckOutpoint -- Stormnode UTXO must have at least %d confirmations, Stormnode=%s\n",
                    Params().GetConsensus().nStormnodeMinimumConfirmations, vin.prevout.ToStringShort());
            // maybe we miss few blocks, let this snb to be checked again later
            snodeman.mapSeenStormnodeBroadcast.erase(GetHash());
            return false;
        }
    }

    LogPrint("Stormnode", "CStormnodeBroadcast::CheckOutpoint -- Stormnode UTXO verified\n");

    // make sure the vout that was signed is related to the transaction that spawned the Stormnode
    //  - this is expensive, so it's only done once per Stormnode
    if(!sandStormSigner.IsVinAssociatedWithPubkey(vin, pubKeyCollateralAddress)) {
        LogPrintf("CStormnodeMan::CheckOutpoint -- Got mismatched pubKeyCollateralAddress and vin\n");
        nDos = 33;
        return false;
    }

    // verify that sig time is legit in past
    // should be at least not earlier than block when 1000 DSLK tx got nStormnodeMinimumConfirmations
    uint256 hashBlock = uint256();
    CTransaction tx2;
    GetTransaction(vin.prevout.hash, tx2, Params().GetConsensus(), hashBlock, true);
    {
        LOCK(cs_main);
        BlockMap::iterator mi = mapBlockIndex.find(hashBlock);
        if (mi != mapBlockIndex.end() && (*mi).second) {
            CBlockIndex* pSNIndex = (*mi).second; // block for 1000 DSLK tx -> 1 confirmation
            CBlockIndex* pConfIndex = chainActive[pSNIndex->nHeight + Params().GetConsensus().nStormnodeMinimumConfirmations - 1]; // block where tx got nStormnodeMinimumConfirmations
            if(pConfIndex->GetBlockTime() > sigTime) {
                LogPrintf("CStormnodeBroadcast::CheckOutpoint -- Bad sigTime %d (%d conf block is at %d) for Stormnode %s %s\n",
                          sigTime, Params().GetConsensus().nStormnodeMinimumConfirmations, pConfIndex->GetBlockTime(), vin.prevout.ToStringShort(), addr.ToString());
                return false;
            }
        }
    }

    return true;
}

bool CStormnodeBroadcast::Sign(CKey& keyCollateralAddress)
{
    std::string strError;
    std::string strMessage;

    sigTime = GetAdjustedTime();

    strMessage = addr.ToString(false) + boost::lexical_cast<std::string>(sigTime) +
                    pubKeyCollateralAddress.GetID().ToString() + pubKeyStormnode.GetID().ToString() +
                    boost::lexical_cast<std::string>(nProtocolVersion);

    if(!sandStormSigner.SignMessage(strMessage, vchSig, keyCollateralAddress)) {
        LogPrintf("CStormnodeBroadcast::Sign -- SignMessage() failed\n");
        return false;
    }

    if(!sandStormSigner.VerifyMessage(pubKeyCollateralAddress, vchSig, strMessage, strError)) {
        LogPrintf("CStormnodeBroadcast::Sign -- VerifyMessage() failed, error: %s\n", strError);
        return false;
    }

    return true;
}

bool CStormnodeBroadcast::CheckSignature(int& nDos)
{
    std::string strMessage;
    std::string strError = "";
    nDos = 0;

    strMessage = addr.ToString(false) + boost::lexical_cast<std::string>(sigTime) +
                    pubKeyCollateralAddress.GetID().ToString() + pubKeyStormnode.GetID().ToString() +
                    boost::lexical_cast<std::string>(nProtocolVersion);

    LogPrint("Stormnode", "CStormnodeBroadcast::CheckSignature -- strMessage: %s  pubKeyCollateralAddress address: %s  sig: %s\n", strMessage, CDarkSilkAddress(pubKeyCollateralAddress.GetID()).ToString(), EncodeBase64(&vchSig[0], vchSig.size()));

    if(!sandStormSigner.VerifyMessage(pubKeyCollateralAddress, vchSig, strMessage, strError)){
        LogPrintf("CStormnodeBroadcast::CheckSignature -- Got bad Stormnode announce signature, error: %s\n", strError);
        nDos = 100;
        return false;
    }

    return true;
}

void CStormnodeBroadcast::Relay()
{
    CInv inv(MSG_STORMNODE_ANNOUNCE, GetHash());
    RelayInv(inv);
}

CStormnodePing::CStormnodePing(CTxIn& vinNew)
{
    LOCK(cs_main);
    if (!chainActive.Tip() || chainActive.Height() < 12) return;

    vin = vinNew;
    blockHash = chainActive[chainActive.Height() - 12]->GetBlockHash();
    sigTime = GetAdjustedTime();
    vchSig = std::vector<unsigned char>();
}

bool CStormnodePing::Sign(CKey& keyStormnode, CPubKey& pubKeyStormnode)
{
    std::string strError;
    std::string strStormNodeSignMessage;

    sigTime = GetAdjustedTime();
    std::string strMessage = vin.ToString() + blockHash.ToString() + boost::lexical_cast<std::string>(sigTime);

    if(!sandStormSigner.SignMessage(strMessage, vchSig, keyStormnode)) {
        LogPrintf("CStormnodePing::Sign -- SignMessage() failed\n");
        return false;
    }

    if(!sandStormSigner.VerifyMessage(pubKeyStormnode, vchSig, strMessage, strError)) {
        LogPrintf("CStormnodePing::Sign -- VerifyMessage() failed, error: %s\n", strError);
        return false;
    }

    return true;
}

bool CStormnodePing::CheckSignature(CPubKey& pubKeyStormnode, int &nDos)
{
    std::string strMessage = vin.ToString() + blockHash.ToString() + boost::lexical_cast<std::string>(sigTime);
    std::string strError = "";
    nDos = 0;

    if(!sandStormSigner.VerifyMessage(pubKeyStormnode, vchSig, strMessage, strError)) {
        LogPrintf("CStormnodePing::CheckSignature -- Got bad Stormnode ping signature, Stormnode=%s, error: %s\n", vin.prevout.ToStringShort(), strError);
        nDos = 33;
        return false;
    }
    return true;
}

bool CStormnodePing::SimpleCheck(int& nDos)
{

    // don't ban by default
    nDos = 0;

    if (sigTime > GetAdjustedTime() + 60 * 60) {
        LogPrintf("CStormnodePing::SimpleCheck -- Signature rejected, too far into the future, Stormnode=%s\n", vin.prevout.ToStringShort());
        nDos = 1;
        return false;
    }

    {
        LOCK(cs_main);
        BlockMap::iterator mi = mapBlockIndex.find(blockHash);
        if (mi == mapBlockIndex.end()) {
            LogPrint("Stormnode", "StormnodePing::SimpleCheck -- Stormnode ping is invalid, unknown block hash: Stormnode=%s blockHash=%s\n", vin.prevout.ToStringShort(), blockHash.ToString());
            // maybe we stuck or forked so we shouldn't ban this node, just fail to accept this ping
            // TODO: or should we also request this block?
            return false;
        }
    }
     LogPrint("Stormnode", "CStormnodePing::SimpleCheck -- Stormnode ping verified: Stormnode=%s  blockHash=%s  sigTime=%d\n", vin.prevout.ToStringShort(), blockHash.ToString(), sigTime);
     return true;
 }
 
 bool CStormnodePing::CheckAndUpdate(CStormnode* psn, bool fFromNewBroadcast, int& nDos)
 {
     // don't ban by default
     nDos = 0;
 
     if (!SimpleCheck(nDos)) {
        return false;
    }

    if (psn == NULL) {
        LogPrint("Stormnode", "CStormnodePing::CheckAndUpdate -- Couldn't find Stormnode entry, Stormnode=%s\n", vin.prevout.ToStringShort());
        return false;
    }

    if(!fFromNewBroadcast) {
        if (psn->IsUpdateRequired()) {
            LogPrint("Stormnode", "CStormnodePing::CheckAndUpdate -- Stormnode protocol is outdated, Stormnode=%s\n", vin.prevout.ToStringShort());
            return false;
        }

        if (psn->IsNewStartRequired()) {
            LogPrint("Stormnode", "CStormnodePing::CheckAndUpdate -- Stormnode is completely expired, new start is required, Stormnode=%s\n", vin.prevout.ToStringShort());
            return false;
       }
    }

    {
        LOCK(cs_main);
        BlockMap::iterator mi = mapBlockIndex.find(blockHash);
        if ((*mi).second && (*mi).second->nHeight < chainActive.Height() - 24) {
            LogPrintf("CStormnodePing::CheckAndUpdate -- Stormnode ping is invalid, block hash is too old: Stormnode=%s  blockHash=%s\n", vin.prevout.ToStringShort(), blockHash.ToString());
            //nDos = 1;
            return false;
        }
    }

    LogPrint("Stormnode", "CStormnodePing::CheckAndUpdate -- New ping: Stormnode=%s  blockHash=%s  sigTime=%d\n", vin.prevout.ToStringShort(), blockHash.ToString(), sigTime);

    // LogPrintf("snping - Found corresponding sn for vin: %s\n", vin.prevout.ToStringShort());
    // update only if there is no known ping for this Stormnode or
    // last ping was more then STORMNODE_MIN_SNP_SECONDS-60 ago comparing to this one
    if (psn->IsPingedWithin(STORMNODE_MIN_SNP_SECONDS - 60, sigTime)) {
        LogPrint("Stormnode", "CStormnodePing::CheckAndUpdate -- Stormnode ping arrived too early, Stormnode=%s\n", vin.prevout.ToStringShort());
        //nDos = 1; //disable, this is happening frequently and causing banned peers
        return false;
    }

    if (!CheckSignature(psn->pubKeyStormnode, nDos)) return false;

    // so, ping seems to be ok, let's store it
    LogPrint("Stormnode", "CStormnodePing::CheckAndUpdate -- Stormnode ping accepted, Stormnode=%s\n", vin.prevout.ToStringShort());
    psn->lastPing = *this;

    // and update snodeman.mapSeenStormnodeBroadcast.lastPing which is probably outdated
    CStormnodeBroadcast snb(*psn);
    uint256 hash = snb.GetHash();
    if (snodeman.mapSeenStormnodeBroadcast.count(hash)) {
        snodeman.mapSeenStormnodeBroadcast[hash].second.lastPing = *this;
    }

    psn->Check(true); // force update, ignoring cache
    if (!psn->IsEnabled()) return false;

    LogPrint("Stormnode", "CStormnodePing::CheckAndUpdate -- Stormnode ping acceepted and relayed, Stormnode=%s\n", vin.prevout.ToStringShort());
    Relay();

    return true;
}

void CStormnodePing::Relay()
{
    CInv inv(MSG_STORMNODE_PING, GetHash());
    RelayInv(inv);
}

void CStormnode::AddGovernanceVote(uint256 nGovernanceObjectHash)
{
    if(mapGovernanceObjectsVotedOn.count(nGovernanceObjectHash)) {
        mapGovernanceObjectsVotedOn[nGovernanceObjectHash]++;
    } else {
        mapGovernanceObjectsVotedOn.insert(std::make_pair(nGovernanceObjectHash, 1));
    }
}

void CStormnode::RemoveGovernanceObject(uint256 nGovernanceObjectHash)
{
    std::map<uint256, int>::iterator it = mapGovernanceObjectsVotedOn.find(nGovernanceObjectHash);
    if(it == mapGovernanceObjectsVotedOn.end()) {
        return;
    }
    mapGovernanceObjectsVotedOn.erase(it);
}

void CStormnode::UpdateWatchdogVoteTime()
{
    LOCK(cs);
    nTimeLastWatchdogVote = GetTime();
}

/**
*   FLAG GOVERNANCE ITEMS AS DIRTY
*
*   - When Stormnode come and go on the network, we must flag the items they voted on to recalc it's cached flags
*
*/
void CStormnode::FlagGovernanceItemsAsDirty()
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
        snodeman.AddDirtyGovernanceObjectHash(vecDirty[i]);
    }
}
