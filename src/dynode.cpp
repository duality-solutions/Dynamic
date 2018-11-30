// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dynode.h"

#include "activedynode.h"
#include "chain.h"
#include "clientversion.h"
#include "dynode-payments.h"
#include "dynode-sync.h"
#include "dynodeman.h"
#include "fluid/fluiddb.h"
#include "init.h"
#include "key_io.h"
#include "messagesigner.h"
#include "netbase.h"
#include "script/standard.h"
#include "util.h"
#include "validation.h"
#ifdef ENABLE_WALLET
#include "wallet/wallet.h"
#endif // ENABLE_WALLET

#include <boost/lexical_cast.hpp>

CDynode::CDynode() : dynode_info_t{DYNODE_ENABLED, PROTOCOL_VERSION, GetAdjustedTime()},
                     fAllowMixingTx(true)
{
}

CDynode::CDynode(CService addr, COutPoint outpoint, CPubKey pubKeyCollateralAddress, CPubKey pubKeyDynode, int nProtocolVersionIn) : dynode_info_t{DYNODE_ENABLED, nProtocolVersionIn, GetAdjustedTime(),
                                                                                                                                         outpoint, addr, pubKeyCollateralAddress, pubKeyDynode},
                                                                                                                                     fAllowMixingTx(true)
{
}

CDynode::CDynode(const CDynode& other) : dynode_info_t{other},
                                         lastPing(other.lastPing),
                                         vchSig(other.vchSig),
                                         nCollateralMinConfBlockHash(other.nCollateralMinConfBlockHash),
                                         nBlockLastPaid(other.nBlockLastPaid),
                                         nPoSeBanScore(other.nPoSeBanScore),
                                         nPoSeBanHeight(other.nPoSeBanHeight),
                                         fAllowMixingTx(other.fAllowMixingTx),
                                         fUnitTest(other.fUnitTest)
{
}

CDynode::CDynode(const CDynodeBroadcast& dnb) : dynode_info_t{dnb.nActiveState, dnb.nProtocolVersion, dnb.sigTime,
                                                    dnb.outpoint, dnb.addr, dnb.pubKeyCollateralAddress, dnb.pubKeyDynode},
                                                lastPing(dnb.lastPing),
                                                vchSig(dnb.vchSig),
                                                fAllowMixingTx(true)
{
}

//
// When a new Dynode broadcast is sent, update our information
//
bool CDynode::UpdateFromNewBroadcast(CDynodeBroadcast& dnb, CConnman& connman)
{
    if (dnb.sigTime <= sigTime && !dnb.fRecovery)
        return false;

    pubKeyDynode = dnb.pubKeyDynode;
    sigTime = dnb.sigTime;
    vchSig = dnb.vchSig;
    nProtocolVersion = dnb.nProtocolVersion;
    addr = dnb.addr;
    nPoSeBanScore = 0;
    nPoSeBanHeight = 0;
    nTimeLastChecked = 0;
    int nDos = 0;
    if (!dnb.lastPing || (dnb.lastPing && dnb.lastPing.CheckAndUpdate(this, true, nDos, connman))) {
        lastPing = dnb.lastPing;
        dnodeman.mapSeenDynodePing.insert(std::make_pair(lastPing.GetHash(), lastPing));
    }
    // if it matches our Dynode privkey...
    if (fDynodeMode && pubKeyDynode == activeDynode.pubKeyDynode) {
        nPoSeBanScore = -DYNODE_POSE_BAN_MAX_SCORE;
        if (nProtocolVersion == PROTOCOL_VERSION) {
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
arith_uint256 CDynode::CalculateScore(const uint256& blockHash) const
{
    // Deterministically calculate a "score" for a Dynode based on any given (block)hash
    CHashWriter ss(SER_GETHASH, PROTOCOL_VERSION);
    ss << outpoint << nCollateralMinConfBlockHash << blockHash;
    return UintToArith256(ss.GetHash());
}

CDynode::CollateralStatus CDynode::CheckCollateral(const COutPoint& outpoint, const CPubKey& pubkey)
{
    int nHeight;
    return CheckCollateral(outpoint, pubkey, nHeight);
}

CDynode::CollateralStatus CDynode::CheckCollateral(const COutPoint& outpoint, const CPubKey& pubkey, int& nHeightRet)
{
    AssertLockHeld(cs_main);

    Coin coin;
    if (!GetUTXOCoin(outpoint, coin)) {
        return COLLATERAL_UTXO_NOT_FOUND;
    }

    if (coin.out.nValue != 1000 * COIN) {
        return COLLATERAL_INVALID_AMOUNT;
    }

    if (pubkey == CPubKey() || coin.out.scriptPubKey != GetScriptForDestination(pubkey.GetID())) {
        return COLLATERAL_INVALID_PUBKEY;
    }

    nHeightRet = coin.nHeight;
    return COLLATERAL_OK;
}

void CDynode::Check(bool fForce)
{
    AssertLockHeld(cs_main);
    LOCK(cs);

    if (ShutdownRequested())
        return;

    if (!fForce && (GetTime() - nTimeLastChecked < DYNODE_CHECK_SECONDS))
        return;
    nTimeLastChecked = GetTime();

    LogPrint("dynode", "CDynode::Check -- Dynode %s is in %s state\n", outpoint.ToStringShort(), GetStateString());

    //once spent, stop doing the checks
    if (IsOutpointSpent())
        return;

    int nHeight = 0;
    if (!fUnitTest) {
        Coin coin;
        if (!GetUTXOCoin(outpoint, coin)) {
            nActiveState = DYNODE_OUTPOINT_SPENT;
            LogPrint("dynode", "CDynode::Check -- Failed to find Dynode UTXO, dynode=%s\n", outpoint.ToStringShort());
            return;
        }

        nHeight = chainActive.Height();
    }

    if (IsPoSeBanned()) {
        if (nHeight < nPoSeBanHeight)
            return; // too early?
        // Otherwise give it a chance to proceed further to do all the usual checks and to change its state.
        // Dynode still will be on the edge and can be banned back easily if it keeps ignoring dnverify
        // or connect attempts. Will require few dnverify messages to strengthen its position in dn list.
        LogPrintf("CDynode::Check -- Dynode %s is unbanned and back in list now\n", outpoint.ToStringShort());
        DecreasePoSeBanScore();
    } else if (nPoSeBanScore >= DYNODE_POSE_BAN_MAX_SCORE) {
        nActiveState = DYNODE_POSE_BAN;
        // ban for the whole payment cycle
        nPoSeBanHeight = nHeight + dnodeman.size();
        LogPrintf("CDynode::Check -- Dynode %s is banned till block %d now\n", outpoint.ToStringShort(), nPoSeBanHeight);
        return;
    }

    int nActiveStatePrev = nActiveState;
    bool fOurDynode = fDynodeMode && activeDynode.pubKeyDynode == pubKeyDynode;
    // Dynode doesn't meet payment protocol requirements ...
    bool fRequireUpdate = nProtocolVersion < dnpayments.GetMinDynodePaymentsProto() ||
                          // or it's our own node and we just updated it to the new protocol but we are still waiting for activation ...
                          (fOurDynode && nProtocolVersion < PROTOCOL_VERSION);

    if (fRequireUpdate) {
        nActiveState = DYNODE_UPDATE_REQUIRED;
        if (nActiveStatePrev != nActiveState) {
            LogPrint("dynode", "CDynode::Check -- Dynode %s is in %s state now\n", outpoint.ToStringShort(), GetStateString());
        }
        return;
    }

    // keep old Dynodes on start, give them a chance to receive updates...
    bool fWaitForPing = !dynodeSync.IsDynodeListSynced() && !IsPingedWithin(DYNODE_MIN_DNP_SECONDS);

    if (fWaitForPing && !fOurDynode) {
        // ...but if it was already expired before the initial check - return right away
        if (IsExpired() || IsSentinelPingExpired() || IsNewStartRequired()) {
            LogPrint("dynode", "CDynode::Check -- Dynode %s is in %s state, waiting for ping\n", outpoint.ToStringShort(), GetStateString());
            return;
        }
    }

    // don't expire if we are still in "waiting for ping" mode unless it's our own dynode
    if (!fWaitForPing || fOurDynode) {
        if (!IsPingedWithin(DYNODE_NEW_START_REQUIRED_SECONDS)) {
            nActiveState = DYNODE_NEW_START_REQUIRED;
            if (nActiveStatePrev != nActiveState) {
                LogPrint("dynode", "CDynode::Check -- Dynode %s is in %s state now\n", outpoint.ToStringShort(), GetStateString());
            }
            return;
        }

        if (!IsPingedWithin(DYNODE_EXPIRATION_SECONDS)) {
            nActiveState = DYNODE_EXPIRED;
            if (nActiveStatePrev != nActiveState) {
                LogPrint("dynode", "CDynode::Check -- Dynode %s is in %s state now\n", outpoint.ToStringShort(), GetStateString());
            }
            return;
        }

        // part 1: expire based on dynamicd ping
        bool fSentinelPingActive = dynodeSync.IsSynced() && dnodeman.IsSentinelPingActive();
        bool fSentinelPingExpired = fSentinelPingActive && !IsPingedWithin(DYNODE_SENTINEL_PING_MAX_SECONDS);
        LogPrint("dynode", "CDynode::Check -- outpoint=%s, GetAdjustedTime()=%d, fSentinelPingExpired=%d\n",
            outpoint.ToStringShort(), GetAdjustedTime(), fSentinelPingExpired);

        if (fSentinelPingExpired) {
            nActiveState = DYNODE_SENTINEL_PING_EXPIRED;
            if (nActiveStatePrev != nActiveState) {
                LogPrint("dynode", "CDynode::Check -- Dynode %s is in %s state now\n", outpoint.ToStringShort(), GetStateString());
            }
            return;
        }
    }

    // We require DNs to be in PRE_ENABLED until they either start to expire or receive a ping and go into ENABLED state
    // Works on mainnet/testnet only and not the case on regtest/devnet.
    if (Params().NetworkIDString() != CBaseChainParams::REGTEST) {
        if (lastPing.sigTime - sigTime < DYNODE_MIN_DNP_SECONDS) {
            nActiveState = DYNODE_PRE_ENABLED;
            if (nActiveStatePrev != nActiveState) {
                LogPrint("dynode", "CDynode::Check -- Dynode %s is in %s state now\n", outpoint.ToStringShort(), GetStateString());
            }
            return;
        }
    }

    if (!fWaitForPing || fOurDynode) {
        // part 2: expire based on sentinel info
        bool fSentinelPingActive = dynodeSync.IsSynced() && dnodeman.IsSentinelPingActive();
        bool fSentinelPingExpired = fSentinelPingActive && !lastPing.fSentinelIsCurrent;

        LogPrint("dynode", "CDynode::Check -- outpoint=%s, GetAdjustedTime()=%d, fSentinelPingExpired=%d\n",
            outpoint.ToStringShort(), GetAdjustedTime(), fSentinelPingExpired);

        if (fSentinelPingExpired) {
            nActiveState = DYNODE_SENTINEL_PING_EXPIRED;
            if (nActiveStatePrev != nActiveState) {
                LogPrint("dynode", "CDynode::Check -- Dynode %s is in %s state now\n", outpoint.ToStringShort(), GetStateString());
            }
            return;
        }
    }

    nActiveState = DYNODE_ENABLED; // OK
    if (nActiveStatePrev != nActiveState) {
        LogPrint("dynode", "CDynode::Check -- Dynode %s is in %s state now\n", outpoint.ToStringShort(), GetStateString());
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

dynode_info_t CDynode::GetInfo() const
{
    dynode_info_t info{*this};
    info.nTimeLastPing = lastPing.sigTime;
    info.fInfoValid = true;
    return info;
}

std::string CDynode::StateToString(int nStateIn)
{
    switch (nStateIn) {
    case DYNODE_PRE_ENABLED:
        return "PRE_ENABLED";
    case DYNODE_ENABLED:
        return "ENABLED";
    case DYNODE_EXPIRED:
        return "EXPIRED";
    case DYNODE_OUTPOINT_SPENT:
        return "OUTPOINT_SPENT";
    case DYNODE_UPDATE_REQUIRED:
        return "UPDATE_REQUIRED";
    case DYNODE_SENTINEL_PING_EXPIRED:
        return "SENTINEL_PING_EXPIRED";
    case DYNODE_NEW_START_REQUIRED:
        return "NEW_START_REQUIRED";
    case DYNODE_POSE_BAN:
        return "POSE_BAN";
    default:
        return "UNKNOWN";
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

void CDynode::UpdateLastPaid(const CBlockIndex* pindex, int nMaxBlocksToScanBack)
{
    if (!pindex)
        return;

    const CBlockIndex* BlockReading = pindex;

    CScript dnpayee = GetScriptForDestination(pubKeyCollateralAddress.GetID());
    // LogPrint("dynode", "CDynode::UpdateLastPaidBlock -- searching for block with payment to %s\n", vin.prevout.ToStringShort());

    LOCK(cs_mapDynodeBlocks);

    for (int i = 0; BlockReading && BlockReading->nHeight > nBlockLastPaid && i < nMaxBlocksToScanBack; i++) {
        if (dnpayments.mapDynodeBlocks.count(BlockReading->nHeight) &&
            dnpayments.mapDynodeBlocks[BlockReading->nHeight].HasPayeeWithVotes(dnpayee, 2)) {
            CBlock block;
            if (!ReadBlockFromDisk(block, BlockReading, Params().GetConsensus())) // shouldn't really happen
                continue;

            CAmount nDynodePayment = GetFluidDynodeReward(BlockReading->nHeight);

            for (const auto& txout : block.vtx[0]->vout)
                if (dnpayee == txout.scriptPubKey && nDynodePayment == txout.nValue) {
                    nBlockLastPaid = BlockReading->nHeight;
                    nTimeLastPaid = BlockReading->nTime;
                    LogPrint("dynode", "CDynode::UpdateLastPaidBlock -- searching for block with payment to %s -- found new %d\n", outpoint.ToStringShort(), nBlockLastPaid);
                    return;
                }
        }

        if (BlockReading->pprev == nullptr) {
            assert(BlockReading);
            break;
        }
        BlockReading = BlockReading->pprev;
    }

    // Last payment for this Dynode wasn't found in latest dnpayments blocks
    // or it was found in dnpayments blocks but wasn't found in the blockchain.
    // LogPrint("dynode", "CDynode::UpdateLastPaidBlock -- searching for block with payment to %s -- keeping old %d\n", vin.prevout.ToStringShort(), nBlockLastPaid);
}

#ifdef ENABLE_WALLET
bool CDynodeBroadcast::Create(const std::string strService, const std::string strKeyDynode, const std::string strTxHash, const std::string strOutputIndex, std::string& strErrorRet, CDynodeBroadcast& dnbRet, bool fOffline)
{
    COutPoint outpoint;
    CPubKey pubKeyCollateralAddressNew;
    CKey keyCollateralAddressNew;
    CPubKey pubKeyDynodeNew;
    CKey keyDynodeNew;

    auto Log = [&strErrorRet](std::string sErr) -> bool {
        strErrorRet = sErr;
        LogPrintf("CDynodeBroadcast::Create -- %s\n", strErrorRet);
        return false;
    };

    // Wait for sync to finish because dnb simply won't be relayed otherwise
    if (!fOffline && !dynodeSync.IsSynced())
        return Log("Sync in progress. Must wait until sync is complete to start Dynode");

    if (!CMessageSigner::GetKeysFromSecret(strKeyDynode, keyDynodeNew, pubKeyDynodeNew))
        return Log(strprintf("Invalid Dynode key %s", strKeyDynode));

    if (!pwalletMain->GetDynodeOutpointAndKeys(outpoint, pubKeyCollateralAddressNew, keyCollateralAddressNew, strTxHash, strOutputIndex))
        return Log(strprintf("Could not allocate outpoint %s:%s for dynode %s", strTxHash, strOutputIndex, strService));

    CService service;
    if (!Lookup(strService.c_str(), service, 0, false))
        return Log(strprintf("Invalid address %s for dynode.", strService));
    int mainnetDefaultPort = Params(CBaseChainParams::MAIN).GetDefaultPort();
    if (Params().NetworkIDString() == CBaseChainParams::MAIN) {
        if (service.GetPort() != mainnetDefaultPort)
            return Log(strprintf("Invalid port %u for dynode %s, only %d is supported on mainnet.", service.GetPort(), strService, mainnetDefaultPort));
    } else if (service.GetPort() == mainnetDefaultPort)
        return Log(strprintf("Invalid port %u for dynode %s, %d is the only supported on mainnet.", service.GetPort(), strService, mainnetDefaultPort));

    return Create(outpoint, service, keyCollateralAddressNew, pubKeyCollateralAddressNew, keyDynodeNew, pubKeyDynodeNew, strErrorRet, dnbRet);
}

bool CDynodeBroadcast::Create(const COutPoint& outpoint, const CService& service, const CKey& keyCollateralAddressNew, const CPubKey& pubKeyCollateralAddressNew, const CKey& keyDynodeNew, const CPubKey& pubKeyDynodeNew, std::string& strErrorRet, CDynodeBroadcast& dnbRet)
{
    // wait for reindex and/or import to finish
    if (fImporting || fReindex)
        return false;

    LogPrint("dynode", "CDynodeBroadcast::Create -- pubKeyCollateralAddressNew = %s, pubKeyDynodeNew.GetID() = %s\n",
        CDynamicAddress(pubKeyCollateralAddressNew.GetID()).ToString(),
        pubKeyDynodeNew.GetID().ToString());

    auto Log = [&strErrorRet, &dnbRet](std::string sErr) -> bool {
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
    if (!dnbRet.Sign(keyCollateralAddressNew))
        return Log(strprintf("Failed to sign broadcast, dynode=%s", outpoint.ToStringShort()));


    return true;
}
#endif // ENABLE_WALLET

bool CDynodeBroadcast::SimpleCheck(int& nDos)
{
    nDos = 0;

    AssertLockHeld(cs_main);

    // make sure addr is valid
    if (!IsValidNetAddr()) {
        LogPrintf("CDynodeBroadcast::SimpleCheck -- Invalid addr, rejected: Dynode=%s  addr=%s\n",
            outpoint.ToStringShort(), addr.ToString());
        return false;
    }

    // make sure signature isn't in the future (past is OK)
    if (sigTime > GetAdjustedTime() + 60 * 60) {
        LogPrintf("CDynodeBroadcast::SimpleCheck -- Signature rejected, too far into the future: Dynode=%s\n", outpoint.ToStringShort());
        nDos = 1;
        return false;
    }

    // empty ping or incorrect sigTime/unknown blockhash
    if (!lastPing || !lastPing.SimpleCheck(nDos)) {
        // one of us is probably forked or smth, just mark it as expired and check the rest of the rules
        nActiveState = DYNODE_EXPIRED;
    }

    if (nProtocolVersion < dnpayments.GetMinDynodePaymentsProto()) {
        LogPrintf("CDynodeBroadcast::SimpleCheck -- outdated Dynode: Dynode=%s  nProtocolVersion=%d\n", outpoint.ToStringShort(), nProtocolVersion);
        nActiveState = DYNODE_UPDATE_REQUIRED;
    }

    CScript pubkeyScript;
    pubkeyScript = GetScriptForDestination(pubKeyCollateralAddress.GetID());

    if (pubkeyScript.size() != 25) {
        LogPrintf("CDynodeBroadcast::SimpleCheck -- pubKeyCollateralAddress has the wrong size\n");
        nDos = 100;
        return false;
    }

    CScript pubkeyScript2;
    pubkeyScript2 = GetScriptForDestination(pubKeyDynode.GetID());

    if (pubkeyScript2.size() != 25) {
        LogPrintf("CDynodeBroadcast::SimpleCheck -- pubKeyDynode has the wrong size\n");
        nDos = 100;
        return false;
    }

    int mainnetDefaultPort = Params(CBaseChainParams::MAIN).GetDefaultPort();
    if (Params().NetworkIDString() == CBaseChainParams::MAIN) {
        if (addr.GetPort() != mainnetDefaultPort)
            return false;
    } else if (addr.GetPort() == mainnetDefaultPort)
        return false;

    return true;
}

bool CDynodeBroadcast::Update(CDynode* pdn, int& nDos, CConnman& connman)
{
    nDos = 0;

    AssertLockHeld(cs_main);

    if (pdn->sigTime == sigTime && !fRecovery) {
        // mapSeenDynodeBroadcast in CDynodeMan::CheckDnbAndUpdateDynodeList should filter legit duplicates
        // but this still can happen if we just started, which is ok, just do nothing here.
        return false;
    }

    // this broadcast is older than the one that we already have - it's bad and should never happen
    // unless someone is doing something fishy
    if (pdn->sigTime > sigTime) {
        LogPrintf("CDynodeBroadcast::Update -- Bad sigTime %d (existing broadcast is at %d) for Dynode %s %s\n",
            sigTime, pdn->sigTime, outpoint.ToStringShort(), addr.ToString());
        return false;
    }

    pdn->Check();

    // Dynode is banned by PoSe
    if (pdn->IsPoSeBanned()) {
        LogPrintf("CDynodeBroadcast::Update -- Banned by PoSe, Dynode=%s\n", outpoint.ToStringShort());
        return false;
    }

    // IsVnAssociatedWithPubkey is validated once in CheckOutpoint, after that they just need to match
    if (pdn->pubKeyCollateralAddress != pubKeyCollateralAddress) {
        LogPrintf("CDynodeBroadcast::Update -- Got mismatched pubKeyCollateralAddress and vin\n");
        nDos = 33;
        return false;
    }

    if (!CheckSignature(nDos)) {
        LogPrintf("CDynodeBroadcast::Update -- CheckSignature() failed, dynode=%s\n", outpoint.ToStringShort());
        return false;
    }

    // if there was no Dynode broadcast recently or if it matches our Dynode privkey...
    if (!pdn->IsBroadcastedWithin(DYNODE_MIN_DNB_SECONDS) || (fDynodeMode && pubKeyDynode == activeDynode.pubKeyDynode)) {
        // take the newest entry
        LogPrintf("CDynodeBroadcast::Update -- Got UPDATED Dynode entry: addr=%s\n", addr.ToString());
        if (pdn->UpdateFromNewBroadcast(*this, connman)) {
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
    if (fDynodeMode && outpoint == activeDynode.outpoint && pubKeyDynode == activeDynode.pubKeyDynode) {
        return false;
    }

    AssertLockHeld(cs_main);

    int nHeight;
    CollateralStatus err = CheckCollateral(outpoint, pubKeyCollateralAddress, nHeight);
    if (err == COLLATERAL_UTXO_NOT_FOUND) {
        LogPrint("dynode", "CDynodeBroadcast::CheckOutpoint -- Failed to find Dynode UTXO, dynode=%s\n", outpoint.ToStringShort());
        return false;
    }

    if (err == COLLATERAL_INVALID_AMOUNT) {
        LogPrint("dynode", "CDynodeBroadcast::CheckOutpoint -- Dynode UTXO should have 1000 DYN, dynode=%s\n", outpoint.ToStringShort());
        nDos = 33;
        return false;
    }

    if (err == COLLATERAL_INVALID_PUBKEY) {
        LogPrint("dynode", "CDynodeBroadcast::CheckOutpoint -- Dynode UTXO should match pubKeyCollateralAddress, dynode=%s\n", outpoint.ToStringShort());
        nDos = 33;
        return false;
    }

    if (chainActive.Height() - nHeight + 1 < Params().GetConsensus().nDynodeMinimumConfirmations) {
        LogPrintf("CDynodeBroadcast::CheckOutpoint -- Dynode UTXO must have at least %d confirmations, dynode=%s\n",
            Params().GetConsensus().nDynodeMinimumConfirmations, outpoint.ToStringShort());
        // UTXO is legit but has not enough confirmations.
        // Maybe we miss few blocks, let this dnb be checked again later.
        dnodeman.mapSeenDynodeBroadcast.erase(GetHash());
        return false;
    }

    LogPrint("dynode", "CDynodeBroadcast::CheckOutpoint -- Dynode UTXO verified\n");

    // Verify that sig time is legit, should be at least not earlier than the timestamp of the block
    // at which collateral became nDynodeMinimumConfirmations blocks deep.
    // NOTE: this is not accurate because block timestamp is NOT guaranteed to be 100% correct one.
    CBlockIndex* pRequredConfIndex = chainActive[nHeight + Params().GetConsensus().nDynodeMinimumConfirmations - 1]; // block where tx got nDynodeMinimumConfirmations
    if (pRequredConfIndex->GetBlockTime() > sigTime) {
        LogPrintf("CDynodeBroadcast::CheckOutpoint -- Bad sigTime %d (%d conf block is at %d) for Dynode %s %s\n",
            sigTime, Params().GetConsensus().nDynodeMinimumConfirmations, pRequredConfIndex->GetBlockTime(), outpoint.ToStringShort(), addr.ToString());
        return false;
    }

    if (!CheckSignature(nDos)) {
        LogPrintf("CDynodeBroadcast::CheckOutpoint -- CheckSignature() failed, dynode=%s\n", outpoint.ToStringShort());
        return false;
    }

    // remember the block hash when collateral for this dynode had minimum required confirmations
    nCollateralMinConfBlockHash = pRequredConfIndex->GetBlockHash();

    return true;
}

uint256 CDynodeBroadcast::GetHash() const
{
    // Note: doesn't match serialization

    CHashWriter ss(SER_GETHASH, PROTOCOL_VERSION);
    ss << outpoint << uint8_t{} << 0xffffffff; // adding dummy values here to match old hashing format
    ss << pubKeyCollateralAddress;
    ss << sigTime;
    return ss.GetHash();
}

uint256 CDynodeBroadcast::GetSignatureHash() const
{
    // TODO: replace with "return SerializeHash(*this);" after migration to 70100
    CHashWriter ss(SER_GETHASH, PROTOCOL_VERSION);
    ss << outpoint;
    ss << addr;
    ss << pubKeyCollateralAddress;
    ss << pubKeyDynode;
    ss << sigTime;
    ss << nProtocolVersion;
    return ss.GetHash();
}

bool CDynodeBroadcast::Sign(const CKey& keyCollateralAddress)
{
    std::string strError;

    sigTime = GetAdjustedTime();

    if (sporkManager.IsSporkActive(SPORK_6_NEW_SIGS)) {
        uint256 hash = GetSignatureHash();

        if (!CHashSigner::SignHash(hash, keyCollateralAddress, vchSig)) {
            LogPrintf("CDynodeBroadcast::Sign -- SignHash() failed\n");
            return false;
        }

        if (!CHashSigner::VerifyHash(hash, pubKeyCollateralAddress, vchSig, strError)) {
            LogPrintf("CDynodeBroadcast::Sign -- VerifyMessage() failed, error: %s\n", strError);
            return false;
        }
    } else {
        std::string strMessage = addr.ToString(false) + std::to_string(sigTime) +
                                 pubKeyCollateralAddress.GetID().ToString() + pubKeyDynode.GetID().ToString() +
                                 std::to_string(nProtocolVersion);

        if (!CMessageSigner::SignMessage(strMessage, vchSig, keyCollateralAddress)) {
            LogPrintf("CDynodeBroadcast::Sign -- SignMessage() failed\n");
            return false;
        }

        if (!CMessageSigner::VerifyMessage(pubKeyCollateralAddress, vchSig, strMessage, strError)) {
            LogPrintf("CDynodeBroadcast::Sign -- VerifyMessage() failed, error: %s\n", strError);
            return false;
        }
    }

    return true;
}

bool CDynodeBroadcast::CheckSignature(int& nDos) const
{
    std::string strError = "";
    nDos = 0;

    if (sporkManager.IsSporkActive(SPORK_6_NEW_SIGS)) {
        uint256 hash = GetSignatureHash();

        if (!CHashSigner::VerifyHash(hash, pubKeyCollateralAddress, vchSig, strError)) {
            // maybe it's in old format
            std::string strMessage = addr.ToString(false) + std::to_string(sigTime) +
                                     pubKeyCollateralAddress.GetID().ToString() + pubKeyDynode.GetID().ToString() +
                                     std::to_string(nProtocolVersion);

            if (!CMessageSigner::VerifyMessage(pubKeyCollateralAddress, vchSig, strMessage, strError)) {
                // nope, not in old format either
                LogPrintf("CDynodeBroadcast::CheckSignature -- Got bad Dynode announce signature, error: %s\n", strError);
                nDos = 100;
                return false;
            }
        }
    } else {
        std::string strMessage = addr.ToString(false) + std::to_string(sigTime) +
                                 pubKeyCollateralAddress.GetID().ToString() + pubKeyDynode.GetID().ToString() +
                                 std::to_string(nProtocolVersion);

        if (!CMessageSigner::VerifyMessage(pubKeyCollateralAddress, vchSig, strMessage, strError)) {
            LogPrintf("CDynodeBroadcast::CheckSignature -- Got bad Dynode announce signature, error: %s\n", strError);
            nDos = 100;
            return false;
        }
    }

    return true;
}

void CDynodeBroadcast::Relay(CConnman& connman) const
{
    // Do not relay until fully synced
    if (!dynodeSync.IsSynced()) {
        LogPrint("dynode", "CDynodeBroadcast::Relay -- won't relay until fully synced\n");
        return;
    }

    CInv inv(MSG_DYNODE_ANNOUNCE, GetHash());
    connman.RelayInv(inv);
}

uint256 CDynodePing::GetHash() const
{
    CHashWriter ss(SER_GETHASH, PROTOCOL_VERSION);
    if (sporkManager.IsSporkActive(SPORK_6_NEW_SIGS)) {
        // TODO: replace with "return SerializeHash(*this);" after migration to 70100
        ss << dynodeOutpoint;
        ss << blockHash;
        ss << sigTime;
        ss << fSentinelIsCurrent;
        ss << nSentinelVersion;
        ss << nDaemonVersion;
    } else {
        // Note: doesn't match serialization

        ss << dynodeOutpoint << uint8_t{} << 0xffffffff; // adding dummy values here to match old hashing format
        ss << sigTime;
    }
    return ss.GetHash();
}

uint256 CDynodePing::GetSignatureHash() const
{
    return GetHash();
}

CDynodePing::CDynodePing(const COutPoint& outpoint)
{
    LOCK(cs_main);
    if (!chainActive.Tip() || chainActive.Height() < 12)
        return;

    dynodeOutpoint = outpoint;
    blockHash = chainActive[chainActive.Height() - 12]->GetBlockHash();
    sigTime = GetAdjustedTime();
    nDaemonVersion = CLIENT_VERSION;
}

bool CDynodePing::Sign(const CKey& keyDynode, const CPubKey& pubKeyDynode)
{
    std::string strError;

    sigTime = GetAdjustedTime();

    if (sporkManager.IsSporkActive(SPORK_6_NEW_SIGS)) {
        uint256 hash = GetSignatureHash();

        if (!CHashSigner::SignHash(hash, keyDynode, vchSig)) {
            LogPrintf("CDynodePing::Sign -- SignHash() failed\n");
            return false;
        }

        if (!CHashSigner::VerifyHash(hash, pubKeyDynode, vchSig, strError)) {
            LogPrintf("CDynodePing::Sign -- VerifyHash() failed, error: %s\n", strError);
            return false;
        }
    } else {
        std::string strMessage = CTxIn(dynodeOutpoint).ToString() + blockHash.ToString() +
                                 std::to_string(sigTime);

        if (!CMessageSigner::SignMessage(strMessage, vchSig, keyDynode)) {
            LogPrintf("CDynodePing::Sign -- SignMessage() failed\n");
            return false;
        }

        if (!CMessageSigner::VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
            LogPrintf("CDynodePing::Sign -- VerifyMessage() failed, error: %s\n", strError);
            return false;
        }
    }

    return true;
}

bool CDynodePing::CheckSignature(const CPubKey& pubKeyDynode, int& nDos) const
{
    std::string strError = "";
    nDos = 0;

    if (sporkManager.IsSporkActive(SPORK_6_NEW_SIGS)) {
        uint256 hash = GetSignatureHash();

        if (!CHashSigner::VerifyHash(hash, pubKeyDynode, vchSig, strError)) {
            std::string strMessage = CTxIn(dynodeOutpoint).ToString() + blockHash.ToString() +
                                     std::to_string(sigTime);

            if (!CMessageSigner::VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
                LogPrintf("CDynodePing::CheckSignature -- Got bad Dynode ping signature, dynode=%s, error: %s\n", dynodeOutpoint.ToStringShort(), strError);
                nDos = 33;
                return false;
            }
        }
    } else {
        std::string strMessage = CTxIn(dynodeOutpoint).ToString() + blockHash.ToString() +
                                 std::to_string(sigTime);

        if (!CMessageSigner::VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
            LogPrintf("CDynodePing::CheckSignature -- Got bad Dynode ping signature, dynode=%s, error: %s\n", dynodeOutpoint.ToStringShort(), strError);
            nDos = 33;
            return false;
        }
    }

    return true;
}

bool CDynodePing::SimpleCheck(int& nDos)
{
    // don't ban by default
    nDos = 0;

    if (sigTime > GetAdjustedTime() + 60 * 60) {
        LogPrintf("CDynodePing::SimpleCheck -- Signature rejected, too far into the future, Dynode=%s\n", dynodeOutpoint.ToStringShort());
        nDos = 1;
        return false;
    }

    {
        AssertLockHeld(cs_main);
        BlockMap::iterator mi = mapBlockIndex.find(blockHash);
        if (mi == mapBlockIndex.end()) {
            LogPrint("dynode", "DynodePing::SimpleCheck -- Dynode ping is invalid, unknown block hash: Dynode=%s blockHash=%s\n", dynodeOutpoint.ToStringShort(), blockHash.ToString());
            // maybe we stuck or forked so we shouldn't ban this node, just fail to accept this ping
            // TODO: or should we also request this block?
            return false;
        }
    }
    LogPrint("dynode", "CDynodePing::SimpleCheck -- Dynode ping verified: Dynode=%s  blockHash=%s  sigTime=%d\n", dynodeOutpoint.ToStringShort(), blockHash.ToString(), sigTime);
    return true;
}

bool CDynodePing::CheckAndUpdate(CDynode* pdn, bool fFromNewBroadcast, int& nDos, CConnman& connman)
{
    AssertLockHeld(cs_main);

    // don't ban by default
    nDos = 0;

    if (!SimpleCheck(nDos)) {
        return false;
    }

    if (pdn == nullptr) {
        LogPrint("dynode", "CDynodePing::CheckAndUpdate -- Couldn't find Dynode entry, Dynode=%s\n", dynodeOutpoint.ToStringShort());
        return false;
    }

    if (!fFromNewBroadcast) {
        if (pdn->IsUpdateRequired()) {
            LogPrint("dynode", "CDynodePing::CheckAndUpdate -- Dynode protocol is outdated, Dynode=%s\n", dynodeOutpoint.ToStringShort());
            return false;
        }

        if (pdn->IsNewStartRequired()) {
            LogPrint("dynode", "CDynodePing::CheckAndUpdate -- Dynode is completely expired, new start is required, Dynode=%s\n", dynodeOutpoint.ToStringShort());
            return false;
        }
    }

    {
        BlockMap::iterator mi = mapBlockIndex.find(blockHash);
        if ((*mi).second && (*mi).second->nHeight < chainActive.Height() - 24) {
            LogPrintf("CDynodePing::CheckAndUpdate -- Dynode ping is invalid, block hash is too old: dynode=%s  blockHash=%s\n", dynodeOutpoint.ToStringShort(), blockHash.ToString());
            // nDos = 1;
            return false;
        }
    }

    LogPrint("dynode", "CDynodePing::CheckAndUpdate -- New ping: Dynode=%s  blockHash=%s  sigTime=%d\n", dynodeOutpoint.ToStringShort(), blockHash.ToString(), sigTime);

    // LogPrintf("dnping - Found corresponding dn for vin: %s\n", vin.prevout.ToStringShort());
    // update only if there is no known ping for this Dynode or
    // last ping was more then DYNODE_MIN_DNP_SECONDS-60 ago comparing to this one
    if (pdn->IsPingedWithin(DYNODE_MIN_DNP_SECONDS - 60, sigTime)) {
        LogPrint("dynode", "CDynodePing::CheckAndUpdate -- Dynode ping arrived too early, Dynode=%s\n", dynodeOutpoint.ToStringShort());
        //nDos = 1; //disable, this is happening frequently and causing banned peers
        return false;
    }

    if (!CheckSignature(pdn->pubKeyDynode, nDos))
        return false;

    // so, ping seems to be ok

    // if we are still syncing and there was no known ping for this dn for quite a while
    // (NOTE: assuming that DYNODE_EXPIRATION_SECONDS/2 should be enough to finish dn list sync)
    if (!dynodeSync.IsDynodeListSynced() && !pdn->IsPingedWithin(DYNODE_EXPIRATION_SECONDS / 2)) {
        // let's bump sync timeout
        LogPrint("dynode", "CDynodePing::CheckAndUpdate -- bumping sync timeout, dynode=%s\n", dynodeOutpoint.ToStringShort());
        dynodeSync.BumpAssetLastTime("CDynodePing::CheckAndUpdate");
    }

    // let's store this ping as the last one
    LogPrint("dynode", "CDynodePing::CheckAndUpdate -- Dynode ping accepted, Dynode=%s\n", dynodeOutpoint.ToStringShort());
    pdn->lastPing = *this;

    // and update dnodeman.mapSeenDynodeBroadcast.lastPing which is probably outdated
    CDynodeBroadcast dnb(*pdn);
    uint256 hash = dnb.GetHash();
    if (dnodeman.mapSeenDynodeBroadcast.count(hash)) {
        dnodeman.mapSeenDynodeBroadcast[hash].second.lastPing = *this;
    }

    // force update, ignoring cache
    pdn->Check(true);
    // relay ping for nodes in ENABLED/EXPIRED/SENTINEL_PING_EXPIRED state only, skip everyone else
    if (!pdn->IsEnabled() && !pdn->IsExpired() && !pdn->IsSentinelPingExpired())
        return false;

    LogPrint("dynode", "CDynodePing::CheckAndUpdate -- Dynode ping acceepted and relayed, Dynode=%s\n", dynodeOutpoint.ToStringShort());
    Relay(connman);

    return true;
}

void CDynodePing::Relay(CConnman& connman)
{
    // Do not relay until fully synced
    if (!dynodeSync.IsSynced()) {
        LogPrint("dynode", "CDynodePing::Relay -- won't relay until fully synced\n");
        return;
    }

    CInv inv(MSG_DYNODE_PING, GetHash());
    connman.RelayInv(inv);
}

std::string CDynodePing::GetSentinelString() const
{
    return nSentinelVersion > DEFAULT_SENTINEL_VERSION ? SafeIntVersionToString(nSentinelVersion) : "Unknown";
}
std::string CDynodePing::GetDaemonString() const
{
    return nDaemonVersion > DEFAULT_DAEMON_VERSION ? FormatVersion(nDaemonVersion) : "Unknown";
}

void CDynode::AddGovernanceVote(uint256 nGovernanceObjectHash)
{
    if (mapGovernanceObjectsVotedOn.count(nGovernanceObjectHash)) {
        mapGovernanceObjectsVotedOn[nGovernanceObjectHash]++;
    } else {
        mapGovernanceObjectsVotedOn.insert(std::make_pair(nGovernanceObjectHash, 1));
    }
}

void CDynode::RemoveGovernanceObject(uint256 nGovernanceObjectHash)
{
    std::map<uint256, int>::iterator it = mapGovernanceObjectsVotedOn.find(nGovernanceObjectHash);
    if (it == mapGovernanceObjectsVotedOn.end()) {
        return;
    }
    mapGovernanceObjectsVotedOn.erase(it);
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
        while (it != mapGovernanceObjectsVotedOn.end()) {
            vecDirty.push_back(it->first);
            ++it;
        }
    }
    for (size_t i = 0; i < vecDirty.size(); ++i) {
        dnodeman.AddDirtyGovernanceObjectHash(vecDirty[i]);
    }
}
