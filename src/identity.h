// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef IDENTITY_H
#define IDENTITY_H

#include "consensus/params.h"
#include "dbwrapper.h"
#include "script/script.h"
#include "serialize.h"
#include "sync.h"

class CTransaction;
class CDynamicAddress;

static const unsigned int ACTIVATE_IDENTITY_HEIGHT    = 10;
static const unsigned int MAX_IDENTITY_NAME_SIZE      = 64;
static const unsigned int MAX_IDENTITY_DOMAIN_SIZE    = 24;
static const unsigned int MAX_IDENTITY_UNIQUE_ID_SIZE = 72;
static const unsigned int MAX_PUBLIC_VALUE_LENGTH     = 512;
static const unsigned int MAX_SECRET_VALUE_LENGTH     = 512;
static const std::string  DEFAULT_IDENTITY_DOMAIN     = "core";

/** Configure Identity Framework */
class CIdentityParameters {
public:
    std::vector<std::pair<std::string, std::vector<unsigned char>>> InitialiseCoreIdentities();
};

class CIdentity {
public:
    std::vector<unsigned char> vchIdentityName;
    std::vector<unsigned char> vchIdentityDomain;
    std::vector<unsigned char> vchUniqueID;

    std::vector<unsigned char> vchWalletAddress;
    std::vector<unsigned char> vchEncryptPublicKey;
    std::vector<unsigned char> vchEncryptPrivateKey;
    std::vector<unsigned char> vchSignPublicKey;
    std::vector<unsigned char> vchSignPrivateKey;

    uint256 txHash;

    unsigned int nHeight;
    uint64_t nExpireTime;

    std::vector<unsigned char> vchPublicData;
    std::vector<unsigned char> vchSecretData;

    CIdentity() { 
        SetNull();
    }

    CIdentity(const CTransaction &tx) {
        SetNull();
        UnserializeFromTx(tx);
    }

    inline void SetNull()
    {
        vchIdentityName.clear();
        vchIdentityDomain.clear();
        vchUniqueID.clear();
        vchWalletAddress.clear();
        vchEncryptPublicKey.clear();
        vchEncryptPrivateKey.clear();
        vchSignPublicKey.clear();
        vchSignPrivateKey.clear();
        txHash.SetNull();
        nHeight = 0;
        nExpireTime = 0;
        vchPublicData.clear();
        vchSecretData.clear();
    }

    ADD_SERIALIZE_METHODS;
    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(vchIdentityName);
        READWRITE(vchIdentityDomain);      
        READWRITE(vchUniqueID);
        READWRITE(vchWalletAddress);
        READWRITE(vchEncryptPublicKey);
        READWRITE(vchEncryptPrivateKey);
        READWRITE(vchSignPublicKey);
        READWRITE(vchSignPrivateKey);
        READWRITE(txHash);
        READWRITE(VARINT(nHeight));
        READWRITE(VARINT(nExpireTime));
        READWRITE(vchPublicData);    
        READWRITE(vchSecretData);
    }

    inline friend bool operator==(const CIdentity &a, const CIdentity &b) {
        return (a.vchIdentityName == b.vchIdentityName && a.vchIdentityDomain == b.vchIdentityDomain && a.vchUniqueID == b.vchUniqueID);
    }

    inline friend bool operator!=(const CIdentity &a, const CIdentity &b) {
        return !(a == b);
    }

    inline CIdentity operator=(const CIdentity &b) {
        vchIdentityName = b.vchIdentityName;
        vchIdentityDomain = b.vchIdentityDomain;
        vchUniqueID = b.vchUniqueID;
        vchWalletAddress = b.vchWalletAddress;
        vchEncryptPublicKey = b.vchEncryptPublicKey;
        vchEncryptPrivateKey = b.vchEncryptPrivateKey;
        vchSignPublicKey = b.vchSignPublicKey;
        vchSignPrivateKey = b.vchSignPrivateKey;
        txHash = b.txHash;
        nHeight = b.nHeight;
        nExpireTime = b.nExpireTime;
        vchPublicData = b.vchPublicData;
        vchSecretData = b.vchSecretData;
        return *this;
    }
 
    inline bool IsNull() const { return (vchIdentityName.empty()); }
    bool UnserializeFromTx(const CTransaction &tx);
    bool UnserializeFromData(const std::vector<unsigned char> &vchData, const std::vector<unsigned char> &vchHash);
    void Serialize(std::vector<unsigned char>& vchData);

    CDynamicAddress WalletAddress();
};

class CIdentityDB : public CDBWrapper {
public:
    CIdentityDB(size_t nCacheSize, bool fMemory, bool fWipe) : CDBWrapper(GetDataDir() / "identities", nCacheSize, fMemory, fWipe) {
    }

    bool ReadIdentity();

    bool AddIdentity();

    bool UpdateIdentity();

    bool ExpireIdentity();

    bool IdentityExists();

    bool CleanupDatabase();

    void WriteIdentityIndex(const CIdentity& identity, const int &op);
    void WriteIdentityIndexHistory(const CIdentity& identity, const int &op);
};

extern CIdentityDB *pIdentityDB;


#endif // IDENTITY_H
