// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "keystore.h"

#include "key.h"
#include "pubkey.h"
#include "util.h"

#include <boost/foreach.hpp>

bool CKeyStore::AddKey(const CKey& key)
{
    return AddKeyPubKey(key, key.GetPubKey());
}

bool CBasicKeyStore::GetPubKey(const CKeyID& address, CPubKey& vchPubKeyOut) const
{
    CKey key;
    if (!GetKey(address, key)) {
        LOCK(cs_KeyStore);
        WatchKeyMap::const_iterator it = mapWatchKeys.find(address);
        if (it != mapWatchKeys.end()) {
            vchPubKeyOut = it->second;
            return true;
        }
        return false;
    }
    vchPubKeyOut = key.GetPubKey();
    return true;
}

isminetype CBasicKeyStore::IsMine(const CKeyID &address) const
{
    LOCK(cs_KeyStore);
    if (mapKeys.count(address) > 0)
        return ISMINE_SPENDABLE;
     if (mapDHTKeys.count(address) > 0)
        return ISMINE_BDAP;
    if (mapWatchKeys.count(address) > 0)
        return ISMINE_WATCH_ONLY;
    return ISMINE_NO;
};

bool CBasicKeyStore::AddKeyPubKey(const CKey& key, const CPubKey& pubkey)
{
    LOCK(cs_KeyStore);
    mapKeys[pubkey.GetID()] = key;
    return true;
}

bool CBasicKeyStore::AddDHTKey(const CKeyEd25519& key, const std::vector<unsigned char>& vchPubKey)
{
    LOCK(cs_KeyStore);

    CKeyID keyID(Hash160(vchPubKey.begin(), vchPubKey.end()));
    LogPrint("dht", "CBasicKeyStore::AddDHTKey, \nkeyID = %s, \npubkey = %s, \nprivkey = %s, \nprivseed = %s\n", 
                                                keyID.ToString(), key.GetPubKeyString(), 
                                                key.GetPrivKeyString(), key.GetPrivSeedString());
    if (keyID != key.GetID()) {
        LogPrint("dht", "CBasicKeyStore::AddDHTKey GetID does't match \nvchPubKey.GetID() = %s, \nkey.GetID() = %s\n", 
                                                                keyID.ToString(), key.GetID().ToString());
        return false;
    }
    mapDHTKeys[keyID] = key;
    return true;
}

bool CBasicKeyStore::GetDHTKey(const CKeyID& address, CKeyEd25519& keyOut) const
{
    {
        LOCK(cs_KeyStore);
        DHTKeyMap::const_iterator mi = mapDHTKeys.find(address);
        if (mi != mapDHTKeys.end())
        {
            keyOut = mi->second;
            return true;
        }
    }
    return false;
}

bool CBasicKeyStore::AddCScript(const CScript& redeemScript)
{
    if (redeemScript.size() > MAX_SCRIPT_ELEMENT_SIZE)
        return error("CBasicKeyStore::AddCScript(): redeemScripts > %i bytes are invalid", MAX_SCRIPT_ELEMENT_SIZE);

    LOCK(cs_KeyStore);
    mapScripts[CScriptID(redeemScript)] = redeemScript;
    return true;
}

bool CBasicKeyStore::HaveCScript(const CScriptID& hash) const
{
    LOCK(cs_KeyStore);
    return mapScripts.count(hash) > 0;
}

bool CBasicKeyStore::GetCScript(const CScriptID& hash, CScript& redeemScriptOut) const
{
    LOCK(cs_KeyStore);
    ScriptMap::const_iterator mi = mapScripts.find(hash);
    if (mi != mapScripts.end()) {
        redeemScriptOut = (*mi).second;
        return true;
    }
    return false;
}

static bool ExtractPubKey(const CScript& dest, CPubKey& pubKeyOut)
{
    //TODO: Use Solver to extract this?
    CScript::const_iterator pc = dest.begin();
    opcodetype opcode;
    std::vector<unsigned char> vch;
    if (!dest.GetOp(pc, opcode, vch) || vch.size() < 33 || vch.size() > 65)
        return false;
    pubKeyOut = CPubKey(vch);
    if (!pubKeyOut.IsFullyValid())
        return false;
    if (!dest.GetOp(pc, opcode, vch) || opcode != OP_CHECKSIG || dest.GetOp(pc, opcode, vch))
        return false;
    return true;
}

bool CBasicKeyStore::AddWatchOnly(const CScript& dest)
{
    LOCK(cs_KeyStore);
    setWatchOnly.insert(dest);
    CPubKey pubKey;
    if (ExtractPubKey(dest, pubKey))
        mapWatchKeys[pubKey.GetID()] = pubKey;
    return true;
}

bool CBasicKeyStore::RemoveWatchOnly(const CScript& dest)
{
    LOCK(cs_KeyStore);
    setWatchOnly.erase(dest);
    CPubKey pubKey;
    if (ExtractPubKey(dest, pubKey))
        mapWatchKeys.erase(pubKey.GetID());
    return true;
}

bool CBasicKeyStore::HaveWatchOnly(const CScript& dest) const
{
    LOCK(cs_KeyStore);
    return setWatchOnly.count(dest) > 0;
}

bool CBasicKeyStore::HaveWatchOnly() const
{
    LOCK(cs_KeyStore);
    return (!setWatchOnly.empty());
}

bool CBasicKeyStore::GetHDChain(CHDChain& hdChainRet) const
{
    hdChainRet = hdChain;
    return !hdChain.IsNull();
}

bool CBasicKeyStore::GetDHTPubKeys(std::vector<std::vector<unsigned char>>& vvchDHTPubKeys) const
{
    for (const std::pair<CKeyID, CKeyEd25519>& key : mapDHTKeys) {
        vvchDHTPubKeys.push_back(key.second.GetPubKey());
        LogPrint("dht", "CBasicKeyStore::GetDHTPubKeys -- pubkey = %s\n", key.second.GetPubKeyString());
    }
    return (vvchDHTPubKeys.size() > 0);
}
