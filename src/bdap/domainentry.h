// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_DOMAINENTRY_H
#define DYNAMIC_BDAP_DOMAINENTRY_H

#include "bdap.h"
#include "amount.h"
#include "consensus/params.h"
#include "primitives/transaction.h"
#include "script/script.h"
#include "serialize.h"
#include "sync.h"

#include <univalue.h>

class CDynamicAddress;
struct CRecipient;
class CTxMemPool;

/* Blockchain Directory Access Framework

  ***** Design Notes *****
- BDAP root DNS entry is bdap.io.  It hosts the root public and admin domains for the BDAP system.
- Modeling after X.500 Directory and LDAP (RFC 4512): https://docs.ldap.com/specs/rfc4512.txt
- Top level domain objects do not have an ObjectID and are considered controlers.
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

using namespace BDAP;

namespace BDAP {
    std::string GetObjectTypeString(unsigned int nObjectType);
    unsigned int GetObjectTypeInt(BDAP::ObjectType ObjectType);
    BDAP::ObjectType GetObjectTypeEnum(unsigned int nObjectType);
}

class CDomainEntryDefaultParameters {
public:
    void InitialiseAdminOwners(); //DEFAULT_ADMIN_DOMAIN
    void InitialisePublicDomain(); //DEFAULT_PUBLIC_DOMAIN
};

// See LDAP Distinguished Name
class CDomainEntry {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    CharString OID; // Canonical Object ID
    //CN=John Smith,OU=Public,DC=BDAP,DC=IO, O=Duality Blockchain Solutions, UID=johnsmith21
    CharString DomainComponent; // DC. Like DC=bdap.io. required. controls child objects
    CharString CommonName; // CN. Like CN=John Smith
    CharString OrganizationalUnit; // OU. Like OU=sales. blank for top level domain directories
    CharString OrganizationName; // O. Like Duality Blockchain Solutions
    CharString ObjectID; // UID. Like johnsmith21.  blank for top level domain directories
    unsigned int nObjectType; // see enum above
    CharString WalletAddress; // used to send collateral funds for this directory record.
    int8_t fPublicObject; // public and private visibility is relative to other objects in its domain directory
    CharString DHTPublicKey; // used to save data in the DHT and encrypt data for this entry.
    CharString LinkAddress; // used to send link requests.  should use a stealth address.

    uint256 txHash;

    unsigned int nHeight;
    uint64_t nExpireTime;

    CDomainEntry() { 
        SetNull();
    }

    CDomainEntry(const CTransactionRef& tx) {
        SetNull();
        UnserializeFromTx(tx);
    }

    inline void SetNull()
    {
        nVersion = CDomainEntry::CURRENT_VERSION;
        OID.clear();
        DomainComponent.clear();
        CommonName.clear();
        OrganizationalUnit.clear();
        OrganizationName.clear();
        ObjectID.clear();
        nObjectType = 0;
        WalletAddress.clear();
        fPublicObject = 0; // by default set to private visibility.
        DHTPublicKey.clear();
        LinkAddress.clear();
        txHash.SetNull();
        nHeight = 0;
        nExpireTime = 0;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(OID);
        READWRITE(DomainComponent);
        READWRITE(CommonName);
        READWRITE(OrganizationalUnit);
        READWRITE(OrganizationName);
        READWRITE(ObjectID);
        READWRITE(VARINT(nObjectType));
        READWRITE(WalletAddress);
        READWRITE(VARINT(fPublicObject));
        READWRITE(DHTPublicKey);
        READWRITE(LinkAddress);
        READWRITE(VARINT(nHeight));
        READWRITE(txHash);
        READWRITE(VARINT(nExpireTime));
    }

    inline friend bool operator==(const CDomainEntry &a, const CDomainEntry &b) {
        return (a.OID == b.OID && a.DomainComponent == b.DomainComponent && a.OrganizationalUnit == b.OrganizationalUnit && a.nObjectType == b.nObjectType);
    }

    inline friend bool operator!=(const CDomainEntry &a, const CDomainEntry &b) {
        return !(a == b);
    }

    inline CDomainEntry operator=(const CDomainEntry &b) {
        OID = b.OID;
        DomainComponent = b.DomainComponent;
        CommonName = b.CommonName;
        OrganizationalUnit = b.OrganizationalUnit;
        OrganizationName = b.OrganizationName;
        ObjectID = b.ObjectID;
        nObjectType = b.nObjectType;
        WalletAddress = b.WalletAddress;
        fPublicObject = b.fPublicObject;
        DHTPublicKey = b.DHTPublicKey;
        LinkAddress = b.LinkAddress;
        txHash = b.txHash;
        nHeight = b.nHeight;
        nExpireTime = b.nExpireTime;
        txHash = b.txHash;
        return *this;
    }
 
    inline bool IsNull() const { return (DomainComponent.empty()); }
    bool UnserializeFromTx(const CTransactionRef &tx);
    bool UnserializeFromData(const std::vector<unsigned char> &vchData, const std::vector<unsigned char> &vchHash);
    void Serialize(std::vector<unsigned char>& vchData);

    CDynamicAddress GetWalletAddress() const;
    CDynamicAddress GetLinkAddress() const;
    std::string DHTPubKeyString() const;
    std::string GetFullObjectPath() const;
    std::string GetObjectLocation() const;
    std::vector<unsigned char> vchFullObjectPath() const;
    std::vector<unsigned char> vchObjectLocation() const; // OU . Domain Name
    bool ValidateValues(std::string& errorMessage);
    bool CheckIfExistsInMemPool(const CTxMemPool& pool, std::string& errorMessage);
    bool TxUsesPreviousUTXO(const CTransactionRef& tx);
    BDAP::ObjectType ObjectType() const { return (BDAP::ObjectType)nObjectType; }
    std::string ObjectTypeString() const { return BDAP::GetObjectTypeString(nObjectType); };
};

bool BuildBDAPJson(const CDomainEntry& entry, UniValue& oName, bool fAbridged = false);

#endif // DYNAMIC_BDAP_DOMAINENTRY_H