// Copyright (c) 2016-2019 Duality Blockchain Solutions
// Copyright (c) 2009-2019 The Bitcoin Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/wallet/wallet.h"

#include "protocol.h"
#include "serialize.h"
#include "util.h"
#include "utiltime.h"
#include "validation.h" // For CheckTransaction
#include "wallet/db.h"
#include "wallet/wallet.h"
#include "wallet/walletdb.h"

#include "dht/ed25519.h"
#include "bdap/bdap.h"
#include "bdap/utils.h"
#include "bdap/linkstorage.h"

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/thread.hpp>

extern std::atomic<unsigned int> nWalletDBUpdateCounter;
/* extern */ unsigned int nWalletDBUpdated;

bool CWallet::NewEdKeyPool()
{
    {
        LOCK(cs_wallet);
        CWalletDB walletdb(strWalletFile);
        BOOST_FOREACH (int64_t nIndex, setInternalEdKeyPool) {
            walletdb.EraseEdPool(nIndex);
        }
        setInternalEdKeyPool.clear();
        BOOST_FOREACH (int64_t nIndex, setExternalEdKeyPool) {
            walletdb.EraseEdPool(nIndex);
        }
        setExternalEdKeyPool.clear();

        if (!TopUpKeyPoolCombo()) //was topupedkeypool
            return false;

        LogPrintf("CWallet::NewEdKeyPool rewrote edkeypool\n");
    }
    return true;
}

bool CWallet::SyncEdKeyPool()
{
    if (pwalletMain->IsLocked())
        return false;

    CWalletDB walletdb(strWalletFile);

    CKeyPool keypool;
    CPubKey retrievedPubKey;
    CKey retrievedKey;

    CKeyPool keypool2;
    CPubKey retrievedPubKey2;
    CKey retrievedKey2;

    for (int64_t nIndex : setInternalKeyPool) {
        if (!walletdb.ReadPool(nIndex, keypool)) {
            throw std::runtime_error(std::string(__func__) + ": read failed");
        }
        retrievedPubKey = keypool.vchPubKey;
        GetKey(retrievedPubKey.GetID(), retrievedKey);

        std::array<char, 32> edSeed = ConvertSecureVector32ToArray(retrievedKey.getKeyData());
        CKeyEd25519 childKey(edSeed);
        CharString vchDHTPubKey = childKey.GetPubKey();

        CKeyID vchDHTPubKeyID = GetIdFromCharVector(vchDHTPubKey);

        //only add if doesn't currently exist
        if (!pwalletMain->HaveDHTKey(vchDHTPubKeyID)) {
            if (!walletdb.WriteEdPool(nIndex, CEdKeyPool(GenerateNewEdKey(0, true, retrievedKey), true)))
                throw std::runtime_error("SyncEdKeyPool(): writing generated key failed");
            else {
                setInternalEdKeyPool.insert(nIndex);
            }
        }
    }

    for (int64_t nIndex : setExternalKeyPool) {
        if (!walletdb.ReadPool(nIndex, keypool2)) {
            throw std::runtime_error(std::string(__func__) + ": read failed");
        }
        retrievedPubKey2 = keypool2.vchPubKey;
        GetKey(retrievedPubKey2.GetID(), retrievedKey2);

        std::array<char, 32> edSeed2 = ConvertSecureVector32ToArray(retrievedKey2.getKeyData());
        CKeyEd25519 childKey2(edSeed2);
        CharString vchDHTPubKey2 = childKey2.GetPubKey();

        CKeyID vchDHTPubKeyID2 = GetIdFromCharVector(vchDHTPubKey2);

        // only add if doesn't currently exist
        if (!pwalletMain->HaveDHTKey(vchDHTPubKeyID2)) {
            if (!walletdb.WriteEdPool(nIndex, CEdKeyPool(GenerateNewEdKey(0, false, retrievedKey2), false)))
                throw std::runtime_error("SyncEdKeyPool(): writing generated key failed");
            else {
                setExternalEdKeyPool.insert(nIndex);
            }
        }
    }

    return true;
}



bool CWallet::TopUpKeyPoolCombo(unsigned int kpSize, bool fIncreaseSize)
{
    {
        LOCK(cs_wallet);

        if (IsLocked(true))
            return false;

        int64_t amountExternal = setExternalKeyPool.size();
        int64_t amountInternal = setInternalKeyPool.size();

        // Top up key pool
        unsigned int nTargetSize;
        unsigned int defaultKeyPoolSize = std::max(GetArg("-keypool", DEFAULT_KEYPOOL_SIZE), (int64_t)0);
        if (kpSize > 0)
            nTargetSize = kpSize;
        else {
            if (defaultKeyPoolSize >= DynamicKeyPoolSize) {
                DynamicKeyPoolSize = defaultKeyPoolSize;
            }

            if (fIncreaseSize) {
                DynamicKeyPoolSize = DynamicKeyPoolSize + 2;
            } //if fIncreaseSize

            nTargetSize = DynamicKeyPoolSize;
        }

        // count amount of available keys (internal, external)
        // make sure the keypool of external and internal keys fits the user selected target (-keypool)

        int64_t missingExternal = std::max(std::max((int64_t)nTargetSize, (int64_t)1) - amountExternal, (int64_t)0);
        int64_t missingInternal = std::max(std::max((int64_t)nTargetSize, (int64_t)1) - amountInternal, (int64_t)0);

        if (!IsHDEnabled()) {
            // don't create extra internal keys
            missingInternal = 0;
        } else {
            nTargetSize *= 2;
        }
        bool fInternal = false;
        CWalletDB walletdb(strWalletFile);
        for (int64_t i = missingInternal + missingExternal; i--;) {
            int64_t nEnd = 1;
            //int64_t nEdEnd = 1;
            if (i < missingInternal) {
                fInternal = true;
            }
            if (!setInternalKeyPool.empty()) {
                nEnd = *(--setInternalKeyPool.end()) + 1;
            }
            if (!setExternalKeyPool.empty()) {
                nEnd = std::max(nEnd, *(--setExternalKeyPool.end()) + 1);
            }
            // TODO: implement keypools for all accounts?

            //get seed for ed29915 keys
            CPubKey retrievedPubKey;
            CKey retrievedKey;

            retrievedPubKey = GenerateNewKey(0, fInternal);
            GetKey(retrievedPubKey.GetID(), retrievedKey);

            if (!walletdb.WritePool(nEnd, CKeyPool(retrievedPubKey, fInternal)))
                throw std::runtime_error("TopUpKeyPoolCombo(): writing generated key failed");

            if (fInternal) {
                setInternalKeyPool.insert(nEnd);
            } else {
                setExternalKeyPool.insert(nEnd);
            }

            // TODO: implement keypools for all accounts?
            if (!walletdb.WriteEdPool(nEnd, CEdKeyPool(GenerateNewEdKey(0, fInternal, retrievedKey), fInternal)))
                throw std::runtime_error("TopUpKeyPoolCombo(): writing generated key failed");

            if (fInternal) {
                setInternalEdKeyPool.insert(nEnd);
            } else {
                setExternalEdKeyPool.insert(nEnd);
            }

            double dProgress = 100.f * nEnd / (nTargetSize + 1);
            std::string strMsg = "";
            if (dProgress <= 100)
                strMsg = strprintf(_("Loading wallet... (%3.2f %%)"), dProgress);
            else
                strMsg = strprintf(_("Increasing keypool... (%d)"),amountExternal);
            uiInterface.InitMessage(strMsg);
        }
    }
    return true;
}

void CWallet::ReserveEdKeyForTransactions(const std::vector<unsigned char>& pubKeyToReserve)
{
    CWalletDB walletdb(strWalletFile);
    CEdKeyPool edkeypool;
    std::vector<unsigned char> edPubKey;
    std::vector<int64_t> keypoolIndexes;
    bool EraseIndex = false;
    int64_t IndexToErase = 0;
    int64_t nIndex = 0;
    std::set<std::int64_t>::iterator it = setInternalEdKeyPool.begin();

    while ((it != setInternalEdKeyPool.end()) && (!EraseIndex)) {
        nIndex = *it;
        if (!walletdb.ReadEdPool(nIndex, edkeypool)) {
            throw std::runtime_error(std::string(__func__) + ": read failed");
        }
        edPubKey = edkeypool.edPubKey;

        if(pubKeyToReserve == edPubKey) {
            KeepKey(nIndex);
            fNeedToUpdateKeyPools = true;
            EraseIndex = true;
            IndexToErase = nIndex;
            ReserveKeyCount++;
        }
        it++;
    }

    if (EraseIndex) {
        std::set<int64_t>::iterator eraseIndexEd = setInternalEdKeyPool.find(IndexToErase);
        std::set<int64_t>::iterator eraseIndex = setInternalKeyPool.find(IndexToErase);
        if (eraseIndexEd != setInternalEdKeyPool.end())
            setInternalEdKeyPool.erase(eraseIndexEd);
        if (eraseIndex != setInternalKeyPool.end())
            setInternalKeyPool.erase(eraseIndex);
    }
}

size_t CWallet::EdKeypoolCountExternalKeys()
{
    AssertLockHeld(cs_wallet); // setExternalEdKeyPool
    return setExternalEdKeyPool.size();
}

size_t CWallet::EdKeypoolCountInternalKeys()
{
    AssertLockHeld(cs_wallet); // setInternalEdKeyPool
    return setInternalEdKeyPool.size();
}

std::array<char, 32> CWallet::ConvertSecureVector32ToArray(const std::vector<unsigned char, secure_allocator<unsigned char> >& vIn)
{
    std::array<char, 32> arrReturn;
    for(unsigned int i = 0; i < 32; i++) {
         arrReturn[i] = (char)vIn[i];
    }
    return arrReturn;
}


std::vector<unsigned char> CWallet::GenerateNewEdKey(uint32_t nAccountIndex, bool fInternal, const CKey& seedIn)
{
    bool fCompressed = CanSupportFeature(FEATURE_COMPRPUBKEY); // default to compressed public keys if we want 0.6.0 wallets
    CKey secretRet;
    secretRet.MakeNewKey(fCompressed);

    // Create new metadata
    int64_t nCreationTime = GetTime();
    CKeyMetadata metadata(nCreationTime);

    CKeyEd25519 secretEdRet;

    if (!seedIn.IsValid()){
        DeriveEd25519ChildKey(secretRet,secretEdRet);
    } else {
        DeriveEd25519ChildKey(seedIn,secretEdRet);
    };

    mapKeyMetadata[secretEdRet.GetID()] = metadata;
    UpdateTimeFirstKey(nCreationTime);

    return secretEdRet.GetPubKey();
}

void CWallet::DeriveEd25519ChildKey(const CKey& seed, CKeyEd25519& secretEdRet)
{
    std::array<char, 32> edSeed = ConvertSecureVector32ToArray(seed.getKeyData());
    CKeyEd25519 childKey(edSeed);
    AddDHTKey(childKey, childKey.GetPubKey()); //TODO
    secretEdRet = childKey;

}

bool CWallet::AddDHTKey(const CKeyEd25519& key, const std::vector<unsigned char>& pubkey)
{
    AssertLockHeld(cs_wallet); // mapKeyMetadata
    if (!CCryptoKeyStore::AddDHTKey(key, pubkey)) {
        return false;
    }

    if (!fFileBacked)
        return true;

    if (!IsCrypted()) {
        CKeyID keyID(Hash160(pubkey.begin(), pubkey.end()));
        return CWalletDB(strWalletFile).WriteDHTKey(key, pubkey, mapKeyMetadata[keyID]);
    }
    LogPrint("dht", "CWallet::AddDHTKey \npubkey = %s, \nprivkey = %s, \nprivseed = %s\n",
                    key.GetPubKeyString(), key.GetPrivKeyString(), key.GetPrivSeedString());
    return true;
}

bool CWallet::AddCryptedDHTKey(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchCryptedSecret)
{
    if (!CCryptoKeyStore::AddCryptedDHTKey(vchPubKey, vchCryptedSecret)) {
        LogPrint("dht", "CWallet::AddCryptedDHTKey AddCryptedDHTKey failed.\n");
        return false;
    }
    if (!fFileBacked)
        return true;
    {
        LOCK(cs_wallet);
        CKeyID keyID(Hash160(vchPubKey.begin(), vchPubKey.end()));
        if (pwalletdbEncryption) {
            return pwalletdbEncryption->WriteCryptedDHTKey(vchPubKey, vchCryptedSecret, mapKeyMetadata[keyID]);
        }
        else {
            return CWalletDB(strWalletFile).WriteCryptedDHTKey(vchPubKey, vchCryptedSecret, mapKeyMetadata[keyID]);
        }
    }
    return false;
}

bool CWallet::GetDHTKey(const CKeyID& address, CKeyEd25519& keyOut) const
{
    LOCK(cs_wallet);
    return CCryptoKeyStore::GetDHTKey(address, keyOut);
}

bool CWallet::HaveDHTKey(const CKeyID &address) const
{
    LOCK(cs_wallet);
    if (mapHdPubKeys.count(address) > 0)
        return true;
    return CCryptoKeyStore::HaveDHTKey(address);
}

bool CWallet::GetDHTPubKeys(std::vector<std::vector<unsigned char>>& vvchDHTPubKeys) const
{
    if (IsCrypted())
        return CCryptoKeyStore::GetDHTPubKeys(vvchDHTPubKeys);

    return CBasicKeyStore::GetDHTPubKeys(vvchDHTPubKeys);
}

bool CWallet::LoadCryptedDHTKey(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchCryptedSecret)
{
    return CCryptoKeyStore::AddCryptedDHTKey(vchPubKey, vchCryptedSecret);
}

bool CWallet::WriteLinkMessageInfo(const uint256& subjectID, const std::vector<unsigned char>& vchPubKey)
{
    CWalletDB walletdb(strWalletFile);
    return walletdb.WriteLinkMessageInfo(subjectID, vchPubKey);
}

bool CWallet::EraseLinkMessageInfo(const uint256& subjectID)
{
    CWalletDB walletdb(strWalletFile);
    return walletdb.EraseLinkMessageInfo(subjectID);
}

bool CWalletDB::ReadEdPool(int64_t nPool, CEdKeyPool& edkeypool)
{
    return Read(std::make_pair(std::string("edpool"), nPool), edkeypool);
}

bool CWalletDB::WriteEdPool(int64_t nPool, const CEdKeyPool& edkeypool)
{
    nWalletDBUpdateCounter++;
    return Write(std::make_pair(std::string("edpool"), nPool), edkeypool);
}

bool CWalletDB::EraseEdPool(int64_t nPool)
{
    nWalletDBUpdateCounter++;
    return Erase(std::make_pair(std::string("edpool"), nPool));
}

bool CWalletDB::WriteDHTKey(const CKeyEd25519& key, const std::vector<unsigned char>& vchPubKey, const CKeyMetadata& keyMeta)
{
    CKeyID keyID(Hash160(vchPubKey.begin(), vchPubKey.end()));
    if (!Write(std::make_pair(std::string("dhtkeymeta"), keyID), keyMeta, false))
        return false;

    nWalletDBUpdated++;
    std::vector<unsigned char> vchPrivKeySeed = key.GetPrivSeed();
    // hash pubkey/privkey to accelerate wallet load
    std::vector<unsigned char> vchKey;
    vchKey.reserve(vchPubKey.size() + vchPrivKeySeed.size());
    vchKey.insert(vchKey.end(), vchPubKey.begin(), vchPubKey.end());
    vchKey.insert(vchKey.end(), vchPrivKeySeed.begin(), vchPrivKeySeed.end());

    LogPrint("dht", "CWalletDB::WriteDHTKey \nvchKey = %s, \nkeyID = %s, \npubkey = %s, \nprivkey = %s, \nprivseed = %s\n",
                    stringFromVch(vchKey), keyID.ToString(),
                    key.GetPubKeyString(), key.GetPrivKeyString(), key.GetPrivSeedString());

    return Write(std::make_pair(std::string("dhtkey"), vchPubKey), std::make_pair(vchPrivKeySeed, Hash(vchKey.begin(), vchKey.end())), false);
}

bool CWalletDB::WriteCryptedDHTKey(const std::vector<unsigned char>& vchPubKey,
    const std::vector<unsigned char>& vchCryptedSecret,
    const CKeyMetadata& keyMeta)
{
    const bool fEraseUnencryptedKey = true;
    nWalletDBUpdateCounter++;

    CKeyID keyID(Hash160(vchPubKey.begin(), vchPubKey.end()));
    if (!Write(std::make_pair(std::string("keymeta"), keyID), keyMeta)) {
        LogPrint("dht", "CWalletDB::WriteCryptedDHTKey Write keymeta failed.\n");
        return false;
    }

    // hash pubkey/privkey to accelerate wallet load
    std::vector<unsigned char> vchKey;
    vchKey.reserve(vchPubKey.size() + vchCryptedSecret.size());
    vchKey.insert(vchKey.end(), vchPubKey.begin(), vchPubKey.end());
    vchKey.insert(vchKey.end(), vchCryptedSecret.begin(), vchCryptedSecret.end());

    if (!Write(std::make_pair(std::string("cdhtkey"), vchPubKey), vchCryptedSecret, false)) {
        LogPrint("dht", "CWalletDB::WriteCryptedDHTKey Write cdhtkey failed.\n");
        return false;
    }

    if (fEraseUnencryptedKey) {
        Erase(std::make_pair(std::string("dhtkey"), vchPubKey));
    }
    return true;
}

// Stores the raw link in the wallet database
bool CWalletDB::WriteLink(const CLinkStorage& link)
{
    ProcessLink(link);
    std::vector<unsigned char> vchPubKeys = link.vchLinkPubKey;
    vchPubKeys.insert(vchPubKeys.end(), link.vchSharedPubKey.begin(), link.vchSharedPubKey.end());
    uint256 linkID = Hash(vchPubKeys.begin(), vchPubKeys.end());
    LogPrint("bdap", "%s -- linkID = %s\n", __func__, linkID.ToString());
    return Write(std::make_pair(std::string("link"), linkID), link);
}

bool CWalletDB::EraseLink(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchSharedKey)
{
    std::vector<unsigned char> vchPubKeys = vchPubKey;
    vchPubKeys.insert(vchPubKeys.end(), vchSharedKey.begin(), vchSharedKey.end());
    uint256 linkID = Hash(vchPubKeys.begin(), vchPubKeys.end());
    LogPrint("bdap", "%s -- linkID = %s\n", __func__, linkID.ToString());
    return Erase(std::make_pair(std::string("link"), linkID));
}

bool CWalletDB::WriteLinkMessageInfo(const uint256& subjectID, const std::vector<unsigned char>& vchPubKey)
{
    LogPrint("bdap", "%s -- subjectID = %s\n", __func__, subjectID.ToString());
    return Write(std::make_pair(std::string("linkid"), subjectID), vchPubKey);
}

bool CWalletDB::EraseLinkMessageInfo(const uint256& subjectID)
{
    LogPrint("bdap", "%s -- subjectID = %s\n", __func__, subjectID.ToString());
    return Erase(std::make_pair(std::string("linkid"), subjectID));
}
