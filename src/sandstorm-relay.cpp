// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "sandstorm.h"
#include "sandstorm-relay.h"


CSandStormRelay::CSandStormRelay()
{
    vinStormnode = CTxIn();
    nBlockHeight = 0;
    nRelayType = 0;
    in = CTxIn();
    out = CTxOut();
}

CSandStormRelay::CSandStormRelay(CTxIn& vinStormnodeIn, vector<unsigned char>& vchSigIn, int nBlockHeightIn, int nRelayTypeIn, CTxIn& in2, CTxOut& out2)
{
    vinStormnode = vinStormnodeIn;
    vchSig = vchSigIn;
    nBlockHeight = nBlockHeightIn;
    nRelayType = nRelayTypeIn;
    in = in2;
    out = out2;
}

std::string CSandStormRelay::ToString()
{
    std::ostringstream info;

    info << "vin: " << vinStormnode.ToString() <<
        " nBlockHeight: " << (int)nBlockHeight <<
        " nRelayType: "  << (int)nRelayType <<
        " in " << in.ToString() <<
        " out " << out.ToString();
        
    return info.str();   
}

bool CSandStormRelay::Sign(std::string strSharedKey)
{
    std::string strError = "";
    std::string strMessage = in.ToString() + out.ToString();

    CKey key2;
    CPubKey pubkey2;

    if(!sandStormSigner.GetKeysFromSecret(strSharedKey, key2, pubkey2)) {
        LogPrintf("CSandStormRelay::Sign -- GetKeysFromSecret() failed, invalid shared key %s\n", strSharedKey);
        return false;
    }

    if(!sandStormSigner.SignMessage(strMessage, vchSig2, key2)) {
        LogPrintf("CSandStormRelay::Sign -- SignMessage() failed\n");
        return false;
    }

    if(!sandStormSigner.VerifyMessage(pubkey2, vchSig2, strMessage, strError)) {
        LogPrintf("CSandStormRelay::Sign -- VerifyMessage() failed, error: %s\n", strError);
        return false;
    }

    return true;
}

bool CSandStormRelay::VerifyMessage(std::string strSharedKey)
{
    std::string strError = "";
    std::string strMessage = in.ToString() + out.ToString();

    CKey key2;
    CPubKey pubkey2;

    if(!sandStormSigner.GetKeysFromSecret(strSharedKey, key2, pubkey2)) {
        LogPrintf("CSandStormRelay::VerifyMessage -- GetKeysFromSecret() failed, invalid shared key %s\n", strSharedKey);
        return false;
    }

    if(!sandStormSigner.VerifyMessage(pubkey2, vchSig2, strMessage, strError)) {
        LogPrintf("CSandStormRelay::VerifyMessage -- VerifyMessage() failed, error: %s\n", strError);
        return false;
    }

    return true;
}

void CSandStormRelay::Relay()
{
    int nCount = std::min(snodeman.CountEnabled(MIN_PRIVATESEND_PEER_PROTO_VERSION), 20);
    int nRank1 = (rand() % nCount)+1; 
    int nRank2 = (rand() % nCount)+1; 

    //keep picking another second number till we get one that doesn't match
    while(nRank1 == nRank2) nRank2 = (rand() % nCount)+1;

    //printf("rank 1 - rank2 %d %d \n", nRank1, nRank2);

    //relay this message through 2 separate nodes for redundancy
    RelayThroughNode(nRank1);
    RelayThroughNode(nRank2);
}

void CSandStormRelay::RelayThroughNode(int nRank)
{
    CStormnode* psn = snodeman.GetStormnodeByRank(nRank, nBlockHeight, MIN_PRIVATESEND_PEER_PROTO_VERSION);

    if(psn != NULL){
        //printf("RelayThroughNode %s\n", psn->addr.ToString().c_str());
        CNode* pnode = ConnectNode((CAddress)psn->addr, NULL);
        if(pnode) {
            //printf("Connected\n");
            pnode->PushMessage("ssr", (*this));
            return;
        }
    } else {
        //printf("RelayThroughNode NULL\n");
    }
}
