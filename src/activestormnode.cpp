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
    LogPrint("Stormnode", "CActiveStormnode::ManageState -- Start\n");
    if(!fStormNode) {
        LogPrint("Stormnode", "CActiveStormnode::ManageState -- Not a Stormnode, returning\n");
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

    LogPrint("Stormnode", "CActiveStormnode::ManageState -- status = %s, type = %s, pinger enabled = %d\n", GetStatus(), GetTypeString(), fPingerEnabled);

    if(eType == STORMNODE_UNKNOWN) {
        ManageStateInitial();
    }

    if(eType == STORMNODE_REMOTE) {
        ManageStateRemote();
    } else if(eType == STORMNODE_LOCAL) {
        // Try Remote Start first so the started local Stormnode can be restarted without recreate Stormnode broadcast.
        ManageStateRemote();
        if(nState != ACTIVE_STORMNODE_STARTED)
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
        case ACTIVE_STORMNODE_NOT_CAPABLE:     return "Not capable Stormnode: " + strNotCapableReason;
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

bool CActiveStormnode::SendStormnodePing()
{
    if(!fPingerEnabled) {
        LogPrint("Stormnode", "CActiveStormnode::SendStormnodePing -- %s: Stormnode ping service is disabled, skipping...\n", GetStateString());
        return false;
    }

    if(!snodeman.Has(vin)) {
        strNotCapableReason = "Stormnode not in Stormnode list";
        nState = ACTIVE_STORMNODE_NOT_CAPABLE;
        LogPrintf("CActiveStormnode::SendStormnodePing -- %s: %s\n", GetStateString(), strNotCapableReason);
        return false;
    }

    CStormnodePing snp(vin);
    if(!snp.Sign(keyStormnode, pubKeyStormnode)) {
        LogPrintf("CActiveStormnode::SendStormnodePing -- ERROR: Couldn't sign Stormnode Ping\n");
        return false;
    }

    // Update lastPing for our Stormnode in Stormnode list
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
    LogPrint("Stormnode", "CActiveStormnode::ManageStateInitial -- status = %s, type = %s, pinger enabled = %d\n", GetStatus(), GetTypeString(), fPingerEnabled);
    // Check that our local network configuration is correct
    if (!fListen) {
        // listen option is probably overwritten by smth else, no good
        nState = ACTIVE_STORMNODE_NOT_CAPABLE;
        strNotCapableReason = "Stormnode must accept connections from outside. Make sure listen configuration option is not overwritten by some another parameter.";
        LogPrintf("CActiveStormnode::ManageStateInitial -- %s: %s\n", GetStateString(), strNotCapableReason);
        return;
    }

    bool fFoundLocal = false;
    {
        LOCK(cs_vNodes);

        // First try to find whatever local address is specified by externalip option
        fFoundLocal = GetLocal(service) && CStormnode::IsValidNetAddr(service);
        if(!fFoundLocal) {
            // nothing and no live connections, can't do anything for now
            if (vNodes.empty()) {
                nState = ACTIVE_STORMNODE_NOT_CAPABLE;
                strNotCapableReason = "Can't detect valid external address. Will retry when there are some connections available.";
                LogPrintf("CActiveStormnode::ManageStateInitial -- %s: %s\n", GetStateString(), strNotCapableReason);
                return;
            }
            // We have some peers, let's try to find our local address from one of them
            BOOST_FOREACH(CNode* pnode, vNodes) {
                if (pnode->fSuccessfullyConnected && pnode->addr.IsIPv4()) {
                    fFoundLocal = GetLocal(service, &pnode->addr) && CStormnode::IsValidNetAddr(service);
                    if(fFoundLocal) break;
                }
            }
        }
    }

    if(!fFoundLocal) {
        nState = ACTIVE_STORMNODE_NOT_CAPABLE;
        strNotCapableReason = "Can't detect valid external address. Please consider using the externalip configuration option if problem persists. Make sure to use IPv4 address only.";
        LogPrintf("CActiveStormnode::ManageStateInitial -- %s: %s\n", GetStateString(), strNotCapableReason);
        return;
    }
    
    int mainnetDefaultPort = Params(CBaseChainParams::MAIN).GetDefaultPort();
    
    if(Params().NetworkIDString() == CBaseChainParams::MAIN) {
        if(service.GetPort() != mainnetDefaultPort) {
            nState = ACTIVE_STORMNODE_NOT_CAPABLE;
            strNotCapableReason = strprintf("Invalid port: %u - only 31600 is supported on mainnet.", service.GetPort());
            LogPrintf("CActiveStormnode::ManageStatus() - not capable: %s\n", strNotCapableReason);
            return;
        }
    }

    if(Params().NetworkIDString() != CBaseChainParams::MAIN) {
        if(service.GetPort() == mainnetDefaultPort) {
            nState = ACTIVE_STORMNODE_NOT_CAPABLE;
            strNotCapableReason = strprintf("Invalid port: %u - 31600 is only supported on mainnet.", service.GetPort());
            LogPrintf("CActiveStormnode::ManageStatus() - not capable: %s\n", strNotCapableReason);
            return;
        }
    }

    LogPrintf("CActiveStormnode::ManageStateInitial -- Checking inbound connection to '%s'\n", service.ToString());

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
        LogPrintf("CActiveStormnode::ManageStateInitial -- %s: Wallet balance is < 1000 DSLK\n", GetStateString());
        return;
    }

    // Choose coins to use
    CPubKey pubKeyCollateral;
    CKey keyCollateral;

    // If collateral is found switch to LOCAL mode
    if(pwalletMain->GetStormnodeVinAndKeys(vin, pubKeyCollateral, keyCollateral)) {
        eType = STORMNODE_LOCAL;
    }

    LogPrint("Stormnode", "CActiveStormnode::ManageStateInitial -- End status = %s, type = %s, pinger enabled = %d\n", GetStatus(), GetTypeString(), fPingerEnabled);
}

void CActiveStormnode::ManageStateRemote()
{
    LogPrint("Stormnode", "CActiveStormnode::ManageStateRemote -- Start status = %s, type = %s, pinger enabled = %d, pubKeyStormnode.GetID() = %s\n", 
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
        if(!CStormnode::IsValidStateForAutoStart(infoSn.nActiveState)) {
            nState = ACTIVE_STORMNODE_NOT_CAPABLE;
            strNotCapableReason = strprintf("Stormnode in %s state", CStormnode::StateToString(infoSn.nActiveState));
            LogPrintf("CActiveStormnode::ManageStateRemote -- %s: %s\n", GetStateString(), strNotCapableReason);
            return;
        }
        if(nState != ACTIVE_STORMNODE_STARTED) {
            LogPrintf("CActiveStormnode::ManageStateRemote -- STARTED!\n");
            vin = infoSn.vin;
            service = infoSn.addr;
            fPingerEnabled = true;
            nState = ACTIVE_STORMNODE_STARTED;
        }
    }
    else {
        nState = ACTIVE_STORMNODE_NOT_CAPABLE;
        strNotCapableReason = "Stormnode not in Stormnode list";
        LogPrintf("CActiveStormnode::ManageStateRemote -- %s: %s\n", GetStateString(), strNotCapableReason);
    }
}

void CActiveStormnode::ManageStateLocal()
{
    LogPrint("Stormnode", "CActiveStormnode::ManageStateLocal -- status = %s, type = %s, pinger enabled = %d\n", GetStatus(), GetTypeString(), fPingerEnabled);
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
            LogPrintf("CActiveStormnode::ManageStateLocal -- %s: %s\n", GetStateString(), strNotCapableReason);
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
            strNotCapableReason = "Error creating Stormnode broadcast: " + strError;
            LogPrintf("CActiveStormnode::ManageStateLocal -- %s: %s\n", GetStateString(), strNotCapableReason);
            return;
        }

        fPingerEnabled = true;
        nState = ACTIVE_STORMNODE_STARTED;

        //update to Stormnode list
        LogPrintf("CActiveStormnode::ManageStateLocal -- Update Stormnode List\n");
        snodeman.UpdateStormnodeList(snb);
        snodeman.NotifyStormnodeUpdates();

        //send to all peers
        LogPrintf("CActiveStormnode::ManageStateLocal -- Relay broadcast, vin=%s\n", vin.ToString());
        snb.Relay();
    }
}
