// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2014-2017 The Dash CoreDevelopers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "activestormnode.h"
#include "stormnode.h"
#include "stormnode-sync.h"
#include "stormnodeman.h"
#include "protocol.h"

extern CWallet* pwalletMain;

// Keep track of the active Stormnode
CActiveStormnode activeStormnode;

void CActiveStormnode::ManageState()
{
    LogPrint("stormnode", "CActiveStormnode::ManageState -- Start\n");
    if(!fStormNode) {
        LogPrint("stormnode", "CActiveStormnode::ManageState -- Not a stormnode, returning\n");
        return;
    }

    if(Params().NetworkIDString() != CBaseChainParams::REGTEST && !stormnodeSync.IsBlockchainSynced()) {
        nState = ACTIVE_STORMNODE_SYNC_IN_PROCESS;
        LogPrintf("CActiveStormnode::ManageState -- %s: %s\n", GetStateString(), GetStatus());
        return;
    }

    if(nState == ACTIVE_STORMNODE_SYNC_IN_PROCESS) {
        nState = ACTIVE_STORMNODE_INITIAL;
    }

    LogPrint("stormnode", "CActiveStormnode::ManageState -- status = %s, type = %s, pinger enabled = %d\n", GetStatus(), GetTypeString(), fPingerEnabled);

    if(eType == STORMNODE_UNKNOWN) {
        ManageStateInitial();
    }

    if(eType == STORMNODE_REMOTE) {
        ManageStateRemote();
    } else if(eType == STORMNODE_LOCAL) {
        ManageStateLocal();
    }

    SendStormnodePing();
}

std::string CActiveStormnode::GetStateString() const
{
    switch (nState) {
        case ACTIVE_STORMNODE_INITIAL:         return "INITIAL";
        case ACTIVE_STORMNODE_SYNC_IN_PROCESS: return "SYNC_IN_PROCESS";
        case ACTIVE_STORMNODE_INPUT_TOO_NEW:   return "INPUT_TOO_NEW";
        case ACTIVE_STORMNODE_NOT_CAPABLE:     return "NOT_CAPABLE";
        case ACTIVE_STORMNODE_STARTED:         return "STARTED";
        default:                                return "UNKNOWN";
    }
}

std::string CActiveStormnode::GetStatus() const
{
    switch (nState) {
        case ACTIVE_STORMNODE_INITIAL:         return "Node just started, not yet activated";
        case ACTIVE_STORMNODE_SYNC_IN_PROCESS: return "Sync in progress. Must wait until sync is complete to start Stormnode";
        case ACTIVE_STORMNODE_INPUT_TOO_NEW:   return strprintf("Stormnode input must have at least %d confirmations", Params().GetConsensus().nStormnodeMinimumConfirmations);
        case ACTIVE_STORMNODE_NOT_CAPABLE:     return "Not capable stormnode: " + strNotCapableReason;
        case ACTIVE_STORMNODE_STARTED:         return "Stormnode successfully started";
        default:                                return "Unknown";
    }
}

std::string CActiveStormnode::GetTypeString() const
{
    std::string strType;
    switch(eType) {
    case STORMNODE_UNKNOWN:
        strType = "UNKNOWN";
        break;
    case STORMNODE_REMOTE:
        strType = "REMOTE";
        break;
    case STORMNODE_LOCAL:
        strType = "LOCAL";
        break;
    default:
        strType = "UNKNOWN";
        break;
    }
    return strType;
}

bool CActiveStormnode::SendMasternodePing()
{
    if(!fPingerEnabled) {
        LogPrint("stormnode", "CActiveStormnode::SendStormnodePing -- %s: stormnode ping service is disabled, skipping...\n", GetStateString());
        return false;
    }

    if(!snodeman.Has(vin)) {
        strNotCapableReason = "Stormnode not in stormnode list";
        nState = ACTIVE_STORMNODE_NOT_CAPABLE;
        LogPrintf("CActiveStormnode::SendStormnodePing -- %s: %s\n", GetStateString(), strNotCapableReason);
        return false;
    }

    CStormnodePing snp(vin);
    if(!snp.Sign(keyStormnode, pubKeyStormnode)) {
        LogPrintf("CActiveStormnode::SendStormnodePing -- ERROR: Couldn't sign Stormnode Ping\n");
        return false;
    }

    // Update lastPing for our masternode in Stormnode list
    if(snodeman.IsStormnodePingedWithin(vin, STORMNODE_MIN_SNP_SECONDS, snp.sigTime)) {
        LogPrintf("CActiveStormnode::SendStormnodePing -- Too early to send Stormnode Ping\n");
        return false;
    }

    snodeman.SetStormnodeLastPing(vin, snp);

    LogPrintf("CActiveStormnode::SendStormnodePing -- Relaying ping, collateral=%s\n", vin.ToString());
    snp.Relay();

    return true;
}

void CActiveStormnode::ManageStateInitial()
{
    LogPrint("stormnode", "CActiveStormnode::ManageStateInitial -- status = %s, type = %s, pinger enabled = %d\n", GetStatus(), GetTypeString(), fPingerEnabled);
    // Check that our local network configuration is correct
    if(!GetLocal(service)) {
        nState = ACTIVE_STORMNODE_NOT_CAPABLE;
        strNotCapableReason = "Can't detect external address. Please consider using the externalip configuration option if problem persists.";
        LogPrintf("CActiveStormnode::ManageStateInitial -- %s: %s\n", GetStateString(), strNotCapableReason);
        return;
    }

    int mainnetDefaultPort = Params(CBaseChainParams::MAIN).GetDefaultPort();
    if(Params().NetworkIDString() == CBaseChainParams::MAIN) {
        if(service.GetPort() != mainnetDefaultPort) {
            nState = ACTIVE_STORMNODE_NOT_CAPABLE;
            strNotCapableReason = strprintf("Invalid port: %u - only %d is supported on mainnet.", service.GetPort(), mainnetDefaultPort);
            LogPrintf("CActiveStormnode::ManageStateInitial -- %s: %s\n", GetStateString(), strNotCapableReason);
            return;
        }
    } else if(service.GetPort() == mainnetDefaultPort) {
        nState = ACTIVE_STORMNODE_NOT_CAPABLE;
        strNotCapableReason = strprintf("Invalid port: %u - %d is only supported on mainnet.", service.GetPort(), mainnetDefaultPort);
        LogPrintf("CActiveStormnode::ManageStateInitial -- %s: %s\n", GetStateString(), strNotCapableReason);
        return;
    }

    LogPrintf("CActiveStormnode::ManageState -- Checking inbound connection to '%s'\n", service.ToString());

    if(!ConnectNode((CAddress)service, NULL, true)) {
        nState = ACTIVE_STORMNODE_NOT_CAPABLE;
        strNotCapableReason = "Could not connect to " + service.ToString();
        LogPrintf("CActiveStormnode::ManageStateInitial -- %s: %s\n", GetStateString(), strNotCapableReason);
        return;
    }

    // Default to REMOTE
    eType = STORMNODE_REMOTE;

    // Check if wallet funds are available
    if(!pwalletMain) {
        LogPrintf("CActiveStormnode::ManageStateInitial -- %s: Wallet not available\n", GetStateString());
        return;
    }

    if(pwalletMain->IsLocked()) {
        LogPrintf("CActiveStormnode::ManageStateInitial -- %s: Wallet is locked\n", GetStateString());
        return;
    }

    if(pwalletMain->GetBalance() < 1000*COIN) {
        LogPrintf("CActiveStormnode::ManageStateInitial -- %s: Wallet balance is < 1000 DSLK", GetStateString());
        return;
    }

    // Choose coins to use
    CPubKey pubKeyCollateral;
    CKey keyCollateral;

    // If collateral is found switch to LOCAL mode
    if(pwalletMain->GetStormnodeVinAndKeys(vin, pubKeyCollateral, keyCollateral)) {
        eType = STORMNODE_LOCAL;
    }

    LogPrint("stormnode", "CActiveStormnode::ManageStateInitial -- End status = %s, type = %s, pinger enabled = %d\n", GetStatus(), GetTypeString(), fPingerEnabled);
}

void CActiveStormnode::ManageStateRemote()
{
    LogPrint("stormnode", "CActiveStormnode::ManageStateRemote -- Start status = %s, type = %s, pinger enabled = %d, pubKeyStormnode.GetID() = %s\n", 
             GetStatus(), fPingerEnabled, GetTypeString(), pubKeyStormnode.GetID().ToString());

    snodeman.CheckStormnode(pubKeyStormnode);
    stormnode_info_t infoSn = snodeman.GetStormnodeInfo(pubKeyStormnode);
    if(infoSn.fInfoValid) {
        if(infoSn.nProtocolVersion != PROTOCOL_VERSION) {
            nState = ACTIVE_STORMNODE_NOT_CAPABLE;
            strNotCapableReason = "Invalid protocol version";
            LogPrintf("CActiveStormnode::ManageStateRemote -- %s: %s\n", GetStateString(), strNotCapableReason);
            return;
        }
        if(service != infoSn.addr) {
            nState = ACTIVE_STORMNODE_NOT_CAPABLE;
            strNotCapableReason = "Specified IP doesn't match our external address.";
            LogPrintf("CActiveStormnode::ManageStateRemote -- %s: %s\n", GetStateString(), strNotCapableReason);
            return;
        }
        vin = infoSn.vin;
        service = infoSn.addr;
        fPingerEnabled = true;
        if(((infoSn.nActiveState == CStormnode::STORMNODE_ENABLED) ||
            (infoSn.nActiveState == CStormnode::STORMNODE_PRE_ENABLED) ||
            (infoSn.nActiveState == CStormnode::STORMNODE_WATCHDOG_EXPIRED))) {
            if(nState != ACTIVE_STORMNODE_STARTED) {
                LogPrintf("CActiveStormnode::ManageStateRemote -- STARTED!\n");
            }
            nState = ACTIVE_STORMNODE_STARTED;
        }
        else {
            nState = ACTIVE_STORMNODE_NOT_CAPABLE;
            strNotCapableReason = strprintf("Stormnode in %s state", CStormnode::StateToString(infoSn.nActiveState));
            LogPrintf("CActiveStormnode::ManageStateRemote -- %s: %s\n", GetStateString(), strNotCapableReason);
        }
    }
    else {
        nState = ACTIVE_STORMNODE_NOT_CAPABLE;
        strNotCapableReason = "Stormnode not in stormnode list";
        LogPrintf("CActiveStormnode::ManageStateRemote -- %s: %s\n", GetStateString(), strNotCapableReason);
    }
}

void CActiveStormnode::ManageStateLocal()
{
    LogPrint("stormnode", "CActiveStormnode::ManageStateLocal -- status = %s, type = %s, pinger enabled = %d\n", GetStatus(), GetTypeString(), fPingerEnabled);
    if(nState == ACTIVE_STORMNODE_STARTED) {
        return;
    }

    // Choose coins to use
    CPubKey pubKeyCollateral;
    CKey keyCollateral;

    if(pwalletMain->GetStormnodeVinAndKeys(vin, pubKeyCollateral, keyCollateral)) {
        int nInputAge = GetInputAge(vin);
        if(nInputAge < Params().GetConsensus().nStormnodeMinimumConfirmations){
            nState = ACTIVE_STORMNODE_INPUT_TOO_NEW;
            strNotCapableReason = strprintf(_("%s - %d confirmations"), GetStatus(), nInputAge);
            LogPrintf("CActiveMasternode::ManageStateLocal -- %s: %s\n", GetStateString(), strNotCapableReason);
            return;
        }

        {
            LOCK(pwalletMain->cs_wallet);
            pwalletMain->LockCoin(vin.prevout);
        }

        CStormnodeBroadcast snb;
        std::string strError;
        if(!CStormnodeBroadcast::Create(vin, service, keyCollateral, pubKeyCollateral, keyStormnode, pubKeyStormnode, strError, snb)) {
            nState = ACTIVE_STORMNODE_NOT_CAPABLE;
            strNotCapableReason = "Error creating stormnode broadcast: " + strError;
            LogPrintf("CActiveStormnode::ManageStateLocal -- %s: %s\n", GetStateString(), strNotCapableReason);
            return;
        }

        //update to masternode list
        LogPrintf("CActiveStormnode::ManageStateLocal -- Update Stormnode List\n");
        snodeman.UpdateStormnodeList(snb);
        snodeman.NotifyStormnodeUpdates();

        //send to all peers
        LogPrintf("CActiveStormnode::ManageStateLocal -- Relay broadcast, vin=%s\n", vin.ToString());
        snb.Relay();
        fPingerEnabled = true;
        nState = ACTIVE_STORMNODE_STARTED;
    }
}
