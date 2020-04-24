// Copyright (c) 2019-2020 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_AUDITDATA_H
#define DYNAMIC_BDAP_AUDITDATA_H

#include "bdap.h"
#include "primitives/transaction.h"
#include "serialize.h"
#include "uint256.h"

class CAuditData {
public:
    static const int CURRENT_VERSION = 1;
    int nVersion;
    CharString OwnerFullObjectPath;  // name of the owner's full domain entry path
    std::vector<CharString> vAuditData; // vector of hashes that points to the document being audited
    unsigned int nHeight;
    uint64_t nExpireTime;

    uint256 txHash;

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
        OwnerFullObjectPath.clear();
        vAuditData.clear();
        nHeight = 0;
        nExpireTime = 0;
        txHash.SetNull();
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(OwnerFullObjectPath);
        READWRITE(vAuditData);
        READWRITE(VARINT(nHeight));
        READWRITE(VARINT(nExpireTime));
        READWRITE(txHash);
    }

    inline friend bool operator==(const CAuditData& a, const CAuditData& b) {
        return (a.OwnerFullObjectPath == b.OwnerFullObjectPath && a.vAuditData == b.vAuditData);
    }

    inline friend bool operator!=(const CAuditData& a, const CAuditData& b) {
        return !(a == b);
    }

    inline CAuditData operator=(const CAuditData& b) {
        nVersion = b.nVersion;
        OwnerFullObjectPath = b.OwnerFullObjectPath;
        vAuditData = b.vAuditData;
        nHeight = b.nHeight;
        nExpireTime = b.nExpireTime;
        txHash = b.txHash;
        return *this;
    }
 
    inline bool IsNull() const { return (vAuditData.size() == 0); }
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash);
    bool UnserializeFromTx(const CTransactionRef& tx);
    bool ValidateValues(std::string& strErrorMessage);
};

#endif // DYNAMIC_BDAP_AUDITDATA_H