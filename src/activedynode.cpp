// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "activedynode.h"

#include "dynode-sync.h"
#include "dynode.h"
#include "dynodeman.h"
#include "init.h"
#include "netbase.h"
#include "protocol.h"

#ifdef ENABLE_WALLET
extern CWallet* pwalletMain;
#endif //ENABLE_WALLET

// Keep track of the active Dynode
CActiveDynode activeDynode;

void CActiveDynode::DoMaintenance(CConnman &connman)
{
    if (ShutdownRequested()) return;
     ManageState(connman);
}

void CActiveDynode::ManageState(CConnman& connman)
{
    LogPrint("dynode", "CActiveDynode::ManageState -- Start\n");
    if (!fDynodeMode) {
        LogPrint("dynode", "CActiveDynode::ManageState -- Not a Dynode, returning\n");
        return;
    }

    if (Params().NetworkIDString() != CBaseChainParams::REGTEST && !dynodeSync.IsBlockchainSynced()) {
        nState = ACTIVE_DYNODE_SYNC_IN_PROCESS;
        LogPrintf("CActiveDynode::ManageState -- %s: %s\n", GetStateString(), GetStatus());
        return;
    }

    if (nState == ACTIVE_DYNODE_SYNC_IN_PROCESS) {
        nState = ACTIVE_DYNODE_INITIAL;
    }

    LogPrint("dynode", "CActiveDynode::ManageState -- status = %s, type = %s, pinger enabled = %d\n", GetStatus(), GetTypeString(), fPingerEnabled);

    if (eType == DYNODE_UNKNOWN) {
        ManageStateInitial(connman);
    }

    if (eType == DYNODE_REMOTE) {
        ManageStateRemote();
    }

    SendDynodePing(connman);
}

std::string CActiveDynode::GetStateString() const
{
    switch (nState) {
    case ACTIVE_DYNODE_INITIAL:
        return "INITIAL";
    case ACTIVE_DYNODE_SYNC_IN_PROCESS:
        return "SYNC_IN_PROCESS";
    case ACTIVE_DYNODE_INPUT_TOO_NEW:
        return "INPUT_TOO_NEW";
    case ACTIVE_DYNODE_NOT_CAPABLE:
        return "NOT_CAPABLE";
    case ACTIVE_DYNODE_STARTED:
        return "STARTED";
    default:
        return "UNKNOWN";
    }
}

std::string CActiveDynode::GetStatus() const
{
    switch (nState) {
    case ACTIVE_DYNODE_INITIAL:
        return "Node just started, not yet activated";
    case ACTIVE_DYNODE_SYNC_IN_PROCESS:
        return "Sync in progress. Must wait until sync is complete to start Dynode";
    case ACTIVE_DYNODE_INPUT_TOO_NEW:
        return strprintf("Dynode input must have at least %d confirmations", Params().GetConsensus().nDynodeMinimumConfirmations);
    case ACTIVE_DYNODE_NOT_CAPABLE:
        return "Not capable Dynode: " + strNotCapableReason;
    case ACTIVE_DYNODE_STARTED:
        return "Dynode successfully started";
    default:
        return "Unknown";
    }
}

std::string CActiveDynode::GetTypeString() const
{
    std::string strType;
    switch (eType) {
    case DYNODE_REMOTE:
        strType = "REMOTE";
        break;
    default:
        strType = "UNKNOWN";
        break;
    }
    return strType;
}

bool CActiveDynode::SendDynodePing(CConnman& connman)
{
    if (!fPingerEnabled) {
        LogPrint("dynode", "CActiveDynode::SendDynodePing -- %s: Dynode ping service is disabled, skipping...\n", GetStateString());
        return false;
    }

    if (!dnodeman.Has(outpoint)) {
        strNotCapableReason = "Dynode not in Dynode list";
        nState = ACTIVE_DYNODE_NOT_CAPABLE;
        LogPrintf("CActiveDynode::SendDynodePing -- %s: %s\n", GetStateString(), strNotCapableReason);
        return false;
    }

    CDynodePing dnp(outpoint);
    dnp.nSentinelVersion = nSentinelVersion;
    dnp.fSentinelIsCurrent =
        (llabs(GetAdjustedTime() - nSentinelPingTime) < DYNODE_SENTINEL_PING_MAX_SECONDS);
    if (!dnp.Sign(keyDynode, pubKeyDynode)) {
        LogPrintf("CActiveDynode::SendDynodePing -- ERROR: Couldn't sign Dynode Ping\n");
        return false;
    }

    // Update lastPing for our Dynode in Dynode list
    if (dnodeman.IsDynodePingedWithin(outpoint, DYNODE_MIN_DNP_SECONDS, dnp.sigTime)) {
        LogPrintf("CActiveDynode::SendDynodePing -- Too early to send Dynode Ping\n");
        return false;
    }

    dnodeman.SetDynodeLastPing(outpoint, dnp);

    LogPrintf("CActiveDynode::SendDynodePing -- Relaying ping, collateral=%s\n", outpoint.ToStringShort());
    dnp.Relay(connman);

    return true;
}

bool CActiveDynode::UpdateSentinelPing(int version)
{
    nSentinelVersion = version;
    nSentinelPingTime = GetAdjustedTime();

    return true;
}

void CActiveDynode::ManageStateInitial(CConnman& connman)
{
    LogPrint("dynode", "CActiveDynode::ManageStateInitial -- status = %s, type = %s, pinger enabled = %d\n", GetStatus(), GetTypeString(), fPingerEnabled);
    // Check that our local network configuration is correct
    if (!fListen) {
        // listen option is probably overwritten by smth else, no good
        nState = ACTIVE_DYNODE_NOT_CAPABLE;
        strNotCapableReason = "Dynode must accept connections from outside. Make sure listen configuration option is not overwritten by some another parameter.";
        LogPrintf("CActiveDynode::ManageStateInitial -- %s: %s\n", GetStateString(), strNotCapableReason);
        return;
    }

    // First try to find whatever local address is specified by externalip option
    bool fFoundLocal = GetLocal(service) && CDynode::IsValidNetAddr(service);
    if (!fFoundLocal) {
        bool empty = true;
        // If we have some peers, let's try to find our local address from one of them
        connman.ForEachNodeContinueIf(CConnman::AllNodes, [&fFoundLocal, &empty, this](CNode* pnode) {
            empty = false;
            if (pnode->addr.IsIPv4())
                fFoundLocal = GetLocal(service, &pnode->addr) && CDynode::IsValidNetAddr(service);
            return !fFoundLocal;
        });
        // nothing and no live connections, can't do anything for now
        if (empty) {
            nState = ACTIVE_DYNODE_NOT_CAPABLE;
            strNotCapableReason = "Can't detect valid external address. Will retry when there are some connections available.";
            LogPrintf("CActiveDynode::ManageStateInitial -- %s: %s\n", GetStateString(), strNotCapableReason);
            return;
        }
    }

    if (!fFoundLocal) {
        nState = ACTIVE_DYNODE_NOT_CAPABLE;
        strNotCapableReason = "Can't detect valid external address. Please consider using the externalip configuration option if problem persists. Make sure to use IPv4 address only.";
        LogPrintf("CActiveDynode::ManageStateInitial -- %s: %s\n", GetStateString(), strNotCapableReason);
        return;
    }

    int mainnetDefaultPort = Params(CBaseChainParams::MAIN).GetDefaultPort();
    
    if (Params().NetworkIDString() == CBaseChainParams::MAIN) { 
        if (service.GetPort() != mainnetDefaultPort) {
            nState = ACTIVE_DYNODE_NOT_CAPABLE;
            strNotCapableReason = strprintf("Invalid port: %u - only %d is supported on mainnet.", service.GetPort(), mainnetDefaultPort);
            LogPrintf("CActiveDynode::ManageStateInitial -- %s: %s\n", GetStateString(), strNotCapableReason);
            return;
        }
    } else if (Params().NetworkIDString() != CBaseChainParams::MAIN && service.GetPort() == mainnetDefaultPort) {
        nState = ACTIVE_DYNODE_NOT_CAPABLE;
        strNotCapableReason = strprintf("Invalid port: %u - %d is only supported on mainnet.", service.GetPort(), mainnetDefaultPort);
        LogPrintf("CActiveDynode::ManageStateInitial -- %s: %s\n", GetStateString(), strNotCapableReason);
        return;
    }

    // Check socket connectivity
    LogPrintf("CActiveDynode::ManageStateInitial -- Checking inbound connection to '%s'\n", service.ToString());
    SOCKET hSocket;
    bool fConnected = ConnectSocket(service, hSocket, nConnectTimeout) && IsSelectableSocket(hSocket);
    CloseSocket(hSocket);

    if (!fConnected) {
        nState = ACTIVE_DYNODE_NOT_CAPABLE;
        strNotCapableReason = "Could not connect to " + service.ToString();
        LogPrintf("CActiveDynode::ManageStateInitial -- %s: %s\n", GetStateString(), strNotCapableReason);
        return;
    }

    // Default to REMOTE
    eType = DYNODE_REMOTE;

    LogPrint("dynode", "CActiveDynode::ManageStateInitial -- End status = %s, type = %s, pinger enabled = %d\n", GetStatus(), GetTypeString(), fPingerEnabled);
}

void CActiveDynode::ManageStateRemote()
{
    LogPrint("dynode", "CActiveDynode::ManageStateRemote -- Start status = %s, type = %s, pinger enabled = %d, pubKeyDynode.GetID() = %s\n",
        GetStatus(), fPingerEnabled, GetTypeString(), pubKeyDynode.GetID().ToString());

    dnodeman.CheckDynode(pubKeyDynode, true);
    dynode_info_t infoDn;
    if (dnodeman.GetDynodeInfo(pubKeyDynode, infoDn)) {
        if (infoDn.nProtocolVersion != PROTOCOL_VERSION) {
            nState = ACTIVE_DYNODE_NOT_CAPABLE;
            strNotCapableReason = "Invalid protocol version";
            LogPrintf("CActiveDynode::ManageStateRemote -- %s: %s\n", GetStateString(), strNotCapableReason);
            return;
        }
        if (service != infoDn.addr) {
            nState = ACTIVE_DYNODE_NOT_CAPABLE;
            strNotCapableReason = "Broadcasted IP doesn't match our external address. Make sure you issued a new broadcast if IP of this Dynode changed recently.";
            LogPrintf("CActiveDynode::ManageStateRemote -- %s: %s\n", GetStateString(), strNotCapableReason);
            return;
        }
        if (!CDynode::IsValidStateForAutoStart(infoDn.nActiveState)) {
            nState = ACTIVE_DYNODE_NOT_CAPABLE;
            strNotCapableReason = strprintf("Dynode in %s state", CDynode::StateToString(infoDn.nActiveState));
            LogPrintf("CActiveDynode::ManageStateRemote -- %s: %s\n", GetStateString(), strNotCapableReason);
            return;
        }
        if (nState != ACTIVE_DYNODE_STARTED) {
            LogPrintf("CActiveDynode::ManageStateRemote -- STARTED!\n");
            outpoint = infoDn.outpoint;
            service = infoDn.addr;
            fPingerEnabled = true;
            nState = ACTIVE_DYNODE_STARTED;
        }
    } else {
        nState = ACTIVE_DYNODE_NOT_CAPABLE;
        strNotCapableReason = "Dynode not in Dynode list";
        LogPrintf("CActiveDynode::ManageStateRemote -- %s: %s\n", GetStateString(), strNotCapableReason);
    }
}
