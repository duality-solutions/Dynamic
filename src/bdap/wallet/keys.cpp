// Copyright (c) 2016-2019 Duality Blockchain Solutions
// Copyright (c) 2009-2019 The Bitcoin Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/wallet/keys.h"

#include "arith_uint256.h"
#include "base58.h"
#include "key.h"
#include "pubkey.h"
#include "keystore.h"
#include "util.h"
#include "crypto/sha512.h"
#include "wallet/crypter.h"

#include <string>
#include <vector>
#include <boost/foreach.hpp>

extern bool EncryptSecret(const CKeyingMaterial& vMasterKey, const CKeyingMaterial &vchPlaintext, const uint256& nIV, std::vector<unsigned char> &vchCiphertext);
extern bool DecryptKey(const CKeyingMaterial& vMasterKey, const std::vector<unsigned char>& vchCryptedSecret, const std::vector<unsigned char>& vchPubKey, CKeyEd25519& key);

bool CBasicKeyStore::AddDHTKey(const CKeyEd25519& key, const std::vector<unsigned char>& vchPubKey)
{
    LOCK(cs_KeyStore);

    CKeyID keyID(Hash160(vchPubKey.begin(), vchPubKey.end()));
#ifdef ENABLE_SENSITIVE_DEBUG
    LogPrint("dht", "CBasicKeyStore::AddDHTKey, \nkeyID = %s, \npubkey = %s, \nprivkey = %s, \nprivseed = %s\n",
                                                keyID.ToString(), key.GetPubKeyString(),
                                                key.GetPrivKeyString(), key.GetPrivSeedString());
#endif // ENABLE_SENSITIVE_DEBUG
    if (keyID != key.GetID()) {
#ifdef ENABLE_SENSITIVE_DEBUG
        LogPrint("dht", "CBasicKeyStore::AddDHTKey GetID does't match \nvchPubKey.GetID() = %s, \nkey.GetID() = %s\n",
                                                                keyID.ToString(), key.GetID().ToString());
#endif // ENABLE_SENSITIVE_DEBUG
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

bool CBasicKeyStore::GetDHTPubKeys(std::vector<std::vector<unsigned char>>& vvchDHTPubKeys) const
{
    for (const std::pair<CKeyID, CKeyEd25519>& key : mapDHTKeys) {
        vvchDHTPubKeys.push_back(key.second.GetPubKey());
#ifdef ENABLE_SENSITIVE_DEBUG
        LogPrint("dht", "CBasicKeyStore::GetDHTPubKeys -- pubkey = %s\n", key.second.GetPubKeyString());
#endif // ENABLE_SENSITIVE_DEBUG
    }
    return (vvchDHTPubKeys.size() > 0);
}

bool CCryptoKeyStore::AddDHTKey(const CKeyEd25519& key, const std::vector<unsigned char>& pubkey)
{
    {
        LOCK(cs_KeyStore);
        if (!IsCrypted()) {
            return CBasicKeyStore::AddDHTKey(key, pubkey);
        }

        if (IsLocked(true))
            return false;

#ifdef ENABLE_SENSITIVE_DEBUG
        LogPrintf("dht: CCryptoKeyStore::AddDHTKey \npubkey = %s, \nprivkey = %s, \nprivseed = %s\n",
                    key.GetPubKeyString(), key.GetPrivKeyString(), key.GetPrivSeedString());
#endif // ENABLE_SENSITIVE_DEBUG

        std::vector<unsigned char> vchDHTPrivSeed = key.GetPrivSeed();
        std::vector<unsigned char> vchCryptedSecret;
        CKeyingMaterial vchSecret(vchDHTPrivSeed.begin(), vchDHTPrivSeed.end());
        if (!EncryptSecret(vMasterKey, vchSecret, key.GetHash(), vchCryptedSecret)) {
            LogPrintf("dht: CCryptoKeyStore::AddDHTKey -- Error after EncryptSecret\n");
            return false;
        }

        if (!AddCryptedDHTKey(key.GetPubKey(), vchCryptedSecret)) {
            LogPrintf("dht: CCryptoKeyStore::AddDHTKey -- Error after AddCryptedDHTKey\n");
            return false;
        }
    }
    return true;
}

bool CCryptoKeyStore::AddCryptedDHTKey(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchCryptedSecret)
{
    {
        LOCK(cs_KeyStore);
        if (!SetCrypted())
            return false;

        CKeyID keyID(Hash160(vchPubKey.begin(), vchPubKey.end()));
        mapCryptedDHTKeys[keyID] = make_pair(vchPubKey, vchCryptedSecret);
    }
    return true;
}

bool CCryptoKeyStore::GetDHTKey(const CKeyID& address, CKeyEd25519& keyOut) const
{
    {
        LOCK(cs_KeyStore);
        if (!IsCrypted())
            return CBasicKeyStore::GetDHTKey(address, keyOut);

        CryptedDHTKeyMap::const_iterator mi = mapCryptedDHTKeys.find(address);
        if (mi != mapCryptedDHTKeys.end())
        {
            const std::vector<unsigned char>& vchPubKey = (*mi).second.first;
            const std::vector<unsigned char>& vchCryptedSecret = (*mi).second.second;
            return DecryptKey(vMasterKey, vchCryptedSecret, vchPubKey, keyOut);
        }
    }
    return false;
}
