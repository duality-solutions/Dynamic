// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DIRECTORY_H
#define DIRECTORY_H

#include "consensus/params.h"
#include "dbwrapper.h"
#include "script/script.h"
#include "serialize.h"
#include "sync.h"

class CTransaction;
class CDynamicAddress;

static const unsigned int ACTIVATE_BDAP_HEIGHT        = 10; // TODO: Change for mainnet or spork activate (???)
static const unsigned int MAX_DOMAIN_NAME_SIZE        = 32;
static const unsigned int MAX_ACCOUNT_NAME_SIZE       = 32;
static const unsigned int MAX_TORRENT_NETWORK_LENGTH  = 128;
static const unsigned int MAX_KEY_LENGTH              = 156;
static const unsigned int MAX_PUBLIC_VALUE_LENGTH     = 512;
static const unsigned int MAX_SECRET_VALUE_LENGTH     = 49152; // Pay per byte for hosting on chain
static const std::string  INTERNAL_DOMAIN_PREFIX      = "#";
static const std::string  DEFAULT_ADMIN_DOMAIN        = INTERNAL_DOMAIN_PREFIX + "admin";
static const std::string  DEFAULT_PUBLIC_DOMAIN       = INTERNAL_DOMAIN_PREFIX + "public";

typedef std::vector<unsigned char> CharString;

/* Blockchain Directory Access Framework

  ***** Design Notes *****
- Modeling after X.500 Directory and LDAP
- Top level domain objects do not have an ObjectName
- Sub level directory objects need permission from parent domain.
- Top level domain objects can run side chains, sub level can with permission from parent
- Torrent network link used to store and transmit sharded and encrypted data.
- Side chains secured by Proof of Stake tokens issued by directory domain owner/creator
- Top level domains require collateral and more expensive than 
*/

class CDirectoryDefaultParameters {
public:
    std::vector<std::pair<std::string, std::vector<unsigned char>>> InitialiseAdminOwners();
};

class CDirectory {
public:
    std::vector<CharString> DomainName; // required. controls child objects
    CharString ObjectName; // blank for top level domain directories
    unsigned int intObjectType; // 0 = root domain account, 1 = user account, 2 = device or computer account
    CharString WalletAddress; // used to send collateral funds for this directory record.
    unsigned int fPublicObject; // public and private visibility is relative to other objects in its domain directory
    CharString EncryptPublicKey; // used to encrypt data to send to this directory record.
    CharString EncryptPrivateKey; // used to decrypt messages and data for this directory record
    CharString SignPublicKey; // used to verify authorized update transaction 
    CharString SignPrivateKey;  // used for updating the Directory record.
    CharString TorrentNetworkAddress; // used for temp storage and transmision of sharded and encrypted data.

    uint256 txHash;

    unsigned int nHeight;
    uint64_t nExpireTime;

    CharString PublicCertificate;
    CharString PrivateData;

    CDirectory() { 
        SetNull();
    }

    CDirectory(const CTransaction &tx) {
        SetNull();
        UnserializeFromTx(tx);
    }

    inline void SetNull()
    {
        DomainName.clear();
        ObjectName.clear();
        intObjectType = 0;
        WalletAddress.clear();
        fPublicObject = 0; // by default set to private.
        EncryptPublicKey.clear();
        EncryptPrivateKey.clear();
        SignPublicKey.clear();
        SignPrivateKey.clear();
        TorrentNetworkAddress.clear();
        txHash.SetNull();
        nHeight = 0;
        nExpireTime = 0;
        PublicCertificate.clear();
        PrivateData.clear();
    }

    ADD_SERIALIZE_METHODS;
    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(DomainName);
        READWRITE(ObjectName);
        READWRITE(VARINT(intObjectType));   
        READWRITE(WalletAddress);
        READWRITE(VARINT(fPublicObject));
        READWRITE(EncryptPublicKey);
        READWRITE(EncryptPrivateKey);
        READWRITE(SignPublicKey);
        READWRITE(SignPrivateKey);
        READWRITE(TorrentNetworkAddress);
        READWRITE(txHash);
        READWRITE(VARINT(nHeight));
        READWRITE(VARINT(nExpireTime));
        READWRITE(PublicCertificate);    
        READWRITE(PrivateData);
    }

    inline friend bool operator==(const CDirectory &a, const CDirectory &b) {
        return (a.DomainName == b.DomainName && a.ObjectName == b.ObjectName);
    }

    inline friend bool operator!=(const CDirectory &a, const CDirectory &b) {
        return !(a == b);
    }

    inline CDirectory operator=(const CDirectory &b) {
        DomainName = b.DomainName;
        ObjectName = b.ObjectName;
        intObjectType = b.intObjectType;
        WalletAddress = b.WalletAddress;
        fPublicObject = b.fPublicObject;
        EncryptPublicKey = b.EncryptPublicKey;
        EncryptPrivateKey = b.EncryptPrivateKey;
        SignPublicKey = b.SignPublicKey;
        SignPrivateKey = b.SignPrivateKey;
        TorrentNetworkAddress = b.TorrentNetworkAddress;
        txHash = b.txHash;
        nHeight = b.nHeight;
        nExpireTime = b.nExpireTime;
        PublicCertificate = b.PublicCertificate;
        PrivateData = b.PrivateData;
        return *this;
    }
 
    inline bool IsNull() const { return (DomainName.empty()); }
    bool UnserializeFromTx(const CTransaction &tx);
    bool UnserializeFromData(const std::vector<unsigned char> &vchData, const std::vector<unsigned char> &vchHash);
    void Serialize(std::vector<unsigned char>& vchData);

    CDynamicAddress GetWalletAddress();
};

class CDirectoryDB : public CDBWrapper {
public:
    CDirectoryDB(size_t nCacheSize, bool fMemory, bool fWipe) : CDBWrapper(GetDataDir() / "directories", nCacheSize, fMemory, fWipe) {
    }

    bool ReadDirectory();

    bool AddDirectory();

    bool UpdateDirectory();

    bool ExpireDirectory();

    bool DirectoryExists();

    bool CleanupDirectoryDatabase();

    void WriteDirectoryIndex(const CDirectory& directory, const int &op);
    void WriteDirectoryIndexHistory(const CDirectory& directory, const int &op);
};

extern CDirectoryDB *pDirectoryDB;


#endif // DIRECTORY_H
