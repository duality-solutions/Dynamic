// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DIRECTORY_H
#define DIRECTORY_H

#include "amount.h"
#include "consensus/params.h"
#include "dbwrapper.h"
#include "script/script.h"
#include "serialize.h"
#include "sync.h"

#include <univalue.h>

class CDynamicAddress;
class CRecipient;
class CTransaction;
class CTxOut;

typedef std::vector<unsigned char> CharString;
typedef std::vector<CharString> vchCharString;
typedef std::vector<std::pair<uint32_t, CharString> > vCheckPoints; // << height, block hash >>

static const unsigned int ACTIVATE_BDAP_HEIGHT        = 10; // TODO: Change for mainnet or spork activate (???)
static const unsigned int MAX_DOMAIN_NAME_LENGTH      = 128;
static const unsigned int MAX_OBJECT_NAME_LENGTH      = 128;
static const unsigned int MAX_RESOURCE_POINTER_LENGTH = 128;
static const unsigned int MAX_KEY_LENGTH              = 156;
static const unsigned int MAX_PUBLIC_VALUE_LENGTH     = 512;
static const unsigned int MAX_SECRET_VALUE_LENGTH     = 512; // Pay per byte for hosting on chain
static const unsigned int MAX_NUMBER_CHECKPOINTS      = 1000; // Pay per byte for hosting on chain
static const std::string DEFAULT_ADMIN_DOMAIN         = "admin.bdap.io";
static const std::string DEFAULT_PUBLIC_DOMAIN        = "public.bdap.io";
static const int BDAP_TX_VERSION = 0x3500;

enum DirectoryObjectType {
    USER_ACCOUNT = 0,
    DEVICE_ACCOUNT = 1,
    GROUP = 2,
    DOMAIN_ACCOUNT = 3,
    ORGANIZATIONAL_UNIT = 4,
    CERTIFICATE = 5,
    CODE = 6,
    BINDING = 7
};

/* Blockchain Directory Access Framework

  ***** Design Notes *****
- BDAP root DNS entry is bdap.io.  It hosts the root public and admin domains for the BDAP system.
- Modeling after X.500 Directory and LDAP (RFC 4512): https://docs.ldap.com/specs/rfc4512.txt
- Top level domain objects do not have an ObjectName and are considered controlers.
- OID canonical identifiers like 1.23.456.7.89
- Sub level directory objects need permission from parent domain object.
- Top level domain objects can run side chains, sub level can with permission from parent
- Torrent network link used to store and transmit sharded and encrypted data.
- Side chains secured by Proof of Stake tokens issued by directory domain owner/creator
- Top level domains require collateral and more expensive than
- Recommended to store lower level domain object data as encrypted shards on Torrent network and not on the chain.
- The basic operations of DAP: Bind, Read, List, Search, Compare, Modify, Add, Delete and ModifyRDN
- Implement file sharing using IPFS
*/

class CDirectoryDefaultParameters {
public:
    void InitialiseAdminOwners(); //DEFAULT_ADMIN_DOMAIN
    void InitialisePublicDomain(); //DEFAULT_PUBLIC_DOMAIN
};

class CDirectory {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    CharString OID; // Canonical Object ID
    CharString DomainName; // required. controls child objects
    CharString ObjectName; // blank for top level domain directories
    DirectoryObjectType ObjectType; // see enum above
    CharString WalletAddress; // used to send collateral funds for this directory record.
    int8_t fPublicObject; // public and private visibility is relative to other objects in its domain directory
    CharString EncryptPublicKey; // used to encrypt data to send to this directory record.
    CharString EncryptPrivateKey; // used to decrypt messages and data for this directory record
    CharString SignWalletAddress; // used to verify authorized update transaction
    unsigned int nSigaturesRequired; // number of signatures needed to approve a transaction.  Default = 1
    CharString ResourcePointer; // used to point to a domain shared resource like a stream (video, audio, file sharing), P2P storage (BitTorrent or IPFS network), or private cloud storage
    
    uint256 txHash;

    unsigned int nHeight;
    uint64_t nExpireTime;

    CharString Certificate;
    CharString PrivateData;
    CAmount transactionFee;
    CAmount registrationFeePerDay;
    vCheckPoints CheckpointHashes; // used to store main chain hash checkpoints for added security

    CDirectory() { 
        SetNull();
    }

    CDirectory(const CTransaction &tx) {
        SetNull();
        UnserializeFromTx(tx);
    }

    inline void SetNull()
    {
        nVersion = CDirectory::CURRENT_VERSION;
        OID.clear();
        DomainName.clear();
        ObjectName.clear();
        ObjectType = DirectoryObjectType::DOMAIN_ACCOUNT;
        WalletAddress.clear();
        fPublicObject = 0; // by default set to private visibility.
        EncryptPublicKey.clear();
        EncryptPrivateKey.clear();
        SignWalletAddress.clear();
        nSigaturesRequired = 1;
        ResourcePointer.clear();
        txHash.SetNull();
        nHeight = 0;
        nExpireTime = 0;
        Certificate.clear();
        PrivateData.clear();
        transactionFee = 0;
        registrationFeePerDay = 0;
        CheckpointHashes.clear();
    }

    ADD_SERIALIZE_METHODS;
    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(this->nVersion);
        nVersion = this->nVersion;
        READWRITE(OID);
        READWRITE(DomainName);
        READWRITE(ObjectName);
        //READWRITE(static_cast<int>(ObjectType));
        READWRITE(WalletAddress);
        READWRITE(VARINT(fPublicObject));
        READWRITE(EncryptPublicKey);
        READWRITE(EncryptPrivateKey);
        READWRITE(SignWalletAddress);
        READWRITE(VARINT(nSigaturesRequired));
        READWRITE(ResourcePointer);
        READWRITE(VARINT(nHeight));
        READWRITE(txHash);
        READWRITE(VARINT(nExpireTime));
        READWRITE(Certificate);
        READWRITE(PrivateData);
        READWRITE(transactionFee);
        READWRITE(registrationFeePerDay);
        READWRITE(CheckpointHashes);
    }

    inline friend bool operator==(const CDirectory &a, const CDirectory &b) {
        return (a.OID == b.OID && a.DomainName == b.DomainName && a.ObjectName == b.ObjectName);
    }

    inline friend bool operator!=(const CDirectory &a, const CDirectory &b) {
        return !(a == b);
    }

    inline CDirectory operator=(const CDirectory &b) {
        OID = b.OID;
        DomainName = b.DomainName;
        ObjectName = b.ObjectName;
        ObjectType = b.ObjectType;
        WalletAddress = b.WalletAddress;
        fPublicObject = b.fPublicObject;
        EncryptPublicKey = b.EncryptPublicKey;
        EncryptPrivateKey = b.EncryptPrivateKey;
        SignWalletAddress = b.SignWalletAddress;
        nSigaturesRequired = b.nSigaturesRequired;
        ResourcePointer = b.ResourcePointer;
        txHash = b.txHash;
        nHeight = b.nHeight;
        nExpireTime = b.nExpireTime;
        Certificate = b.Certificate;
        PrivateData = b.PrivateData;
        CheckpointHashes = b.CheckpointHashes;
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
    CDirectoryDB(size_t nCacheSize, bool fMemory, bool fWipe) : CDBWrapper(GetDataDir() / "bdap", nCacheSize, fMemory, fWipe) {
    }

    // Add, Read, Modify, ModifyRDN, Delete, List, Search, Bind, and Compare

    bool AddDirectory(const CDirectory& directory, const int& op) { 
        bool writeState = Write(make_pair(std::string("domain_name"), directory.DomainName), directory) 
                             && Write(make_pair(std::string("domain_wallet_address"), directory.WalletAddress), directory.DomainName);

        AddDirectoryIndex(directory, op);
        return writeState;
    }

    void AddDirectoryIndex(const CDirectory& directory, const int& op);

    bool ReadDirectory();

    bool UpdateDirectory();

    bool ExpireDirectory();

    bool DirectoryExists();

    bool CleanupDirectoryDatabase();
    

    void WriteDirectoryIndex(const CDirectory& directory, const int& op);
    void WriteDirectoryIndexHistory(const CDirectory& directory, const int& op);
};

bool IsDirectoryDataOutput(const CTxOut& out);
int GetDirectoryDataOutput(const CTransaction& tx);
bool GetDirectoryData(const CTransaction& tx, std::vector<unsigned char>& vchData, std::vector<unsigned char>& vchHash, int& nOut);
bool GetDirectoryData(const CScript& scriptPubKey, std::vector<unsigned char>& vchData, std::vector<unsigned char>& vchHash);
bool BuildBDAPJson(const CDirectory& directory, UniValue& oName);

std::string stringFromVch(const CharString& vch);
std::vector<unsigned char> vchFromValue(const UniValue& value);
void CreateRecipient(const CScript& scriptPubKey, CRecipient& recipient);
void ToLowerCase(CharString& vchValue);
void ToLowerCase(std::string& strValue);
bool CheckIfNameExists(const CharString& vchObjectName, const CharString& vchDomainName);

extern CDirectoryDB *pDirectoryDB;

#endif // DIRECTORY_H