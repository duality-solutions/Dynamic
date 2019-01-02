// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_IDENTITY_H
#define DYNAMIC_BDAP_IDENTITY_H

#include "bdap.h"
#include "domainentry.h"
#include "serialize.h"
#include "uint256.h"

using namespace BDAP;

namespace BDAP {
    enum AuditType {
        UNKNOWN = 0,
        HASH_POINTER_AUDIT = 1
    };
    std::string GetAuditTypeString(unsigned int nAuditType);
}

class CAuditData {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    CharString OwnerFullPath;  // name of the owner's full domain entry path
    CharString AuditData; // usually just a hash that points to the document being audited
    unsigned int nAuditType;
    unsigned int nHeight;
    uint64_t nExpireTime;

    uint256 txHash;

    CDomainEntry* OwnerDomainEntry;

    CAuditData() {
        SetNull();
    }

    CAuditData(const CTransactionRef& tx) {
        SetNull();
        UnserializeFromTx(tx);
    }

    inline void SetNull()
    {
        nVersion = CAuditData::CURRENT_VERSION;
        OwnerFullPath.clear();
        AuditData.clear();
        nAuditType = 0; 
        nHeight = 0;
        nExpireTime = 0;
        txHash.SetNull();
        OwnerDomainEntry = nullptr;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(OwnerFullPath);
        READWRITE(AuditData);
        READWRITE(VARINT(nAuditType));
        READWRITE(VARINT(nHeight));
        READWRITE(VARINT(nExpireTime));
        READWRITE(txHash);
    }

    inline friend bool operator==(const CAuditData& a, const CAuditData& b) {
        return (a.OwnerFullPath == b.OwnerFullPath && a.AuditData == b.AuditData && a.nExpireTime == b.nExpireTime);
    }

    inline friend bool operator!=(const CAuditData& a, const CAuditData& b) {
        return !(a == b);
    }

    inline CAuditData operator=(const CAuditData& b) {
        OwnerFullPath = b.OwnerFullPath;
        AuditData = b.AuditData;
        nAuditType = b.nAuditType;
        nHeight = b.nHeight;
        nExpireTime = b.nExpireTime;
        txHash = b.txHash;
        return *this;
    }
 
    inline bool IsNull() const { return (OwnerFullPath.empty()); }
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash);
    bool UnserializeFromTx(const CTransactionRef& tx);

    BDAP::AuditType AuditType() { return (BDAP::AuditType)nAuditType; }
    std::string AuditTypeString() { return BDAP::GetAuditTypeString(nAuditType); }
};

#endif // DYNAMIC_BDAP_DOMAINENTRY_H