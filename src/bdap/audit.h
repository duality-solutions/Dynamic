// Copyright (c) 2019-2020 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_AUDITDATA_H
#define DYNAMIC_BDAP_AUDITDATA_H

#include "bdap/bdap.h"
#include "primitives/transaction.h"
#include "serialize.h"
#include "uint256.h"

class CKey;
class UniValue;

class CAuditData {
public:
    static const int CURRENT_VERSION = 1;
    int nVersion;
    std::vector<CharString> vAuditData; // vector of hashes that points to the document being audited
    int64_t nTimeStamp;

    CAuditData() {
        SetNull();
    }

    CAuditData(const std::vector<unsigned char>& vchData) {
        UnserializeFromData(vchData);
    }

    inline void SetNull()
    {
        nVersion = CAuditData::CURRENT_VERSION;
        vAuditData.clear();
        nTimeStamp = 0;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(vAuditData);
        READWRITE(VARINT(nTimeStamp));
    }

    inline friend bool operator==(const CAuditData& a, const CAuditData& b) {
        return (a.vAuditData == b.vAuditData && a.nTimeStamp == b.nTimeStamp);
    }

    inline friend bool operator!=(const CAuditData& a, const CAuditData& b) {
        return !(a == b);
    }

    inline CAuditData operator=(const CAuditData& b) {
        nVersion = b.nVersion;
        vAuditData = b.vAuditData;
        nTimeStamp = b.nTimeStamp;
        return *this;
    }
 
    inline bool IsNull() const { return (vAuditData.size() == 0); }
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData);
    bool ValidateValues(std::string& strErrorMessage);
};

/** A CAudit is a combination of a serialized CAuditData class and BDAP account with a signature. */
class CAudit
{
public:
    std::vector<unsigned char> vchAuditData;
    std::vector<unsigned char> vchOwnerFullObjectPath;  // name of the owner's full domain entry path
    std::vector<unsigned char> vchSignature; // signature using the owners wallet public key
    unsigned int nHeight;
    uint64_t nExpireTime;
    uint256 txHash;

    CAudit() {
        SetNull();
    }

    CAudit(const CTransactionRef& tx) {
        SetNull();
        UnserializeFromTx(tx);
    }

    CAudit(CAuditData& auditData);

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(vchAuditData);
        READWRITE(vchOwnerFullObjectPath);
        READWRITE(vchSignature);
        READWRITE(VARINT(nHeight));
        READWRITE(VARINT(nExpireTime));
        READWRITE(txHash);
    }

    inline CAudit operator=(const CAudit& b) {
        vchAuditData = b.vchAuditData;
        vchOwnerFullObjectPath = b.vchOwnerFullObjectPath;
        vchSignature = b.vchSignature;
        nHeight = b.nHeight;
        nExpireTime = b.nExpireTime;
        txHash = b.txHash;
        return *this;
    }

    inline void SetNull()
    {
        vchAuditData.clear();
        vchOwnerFullObjectPath.clear();
        vchSignature.clear();
        nHeight = 0;
        nExpireTime = 0;
        txHash.SetNull();
    }

    bool SignatureRequired() const {
        return (vchOwnerFullObjectPath.size() > 0);
    }

    bool IsSigned() const {
        return (vchSignature.size() > 0);
    }

    bool IsNull() const {
        return (vchAuditData.size() == 0);
    }

    CAuditData GetAuditData() const {
        return CAuditData(vchAuditData);
    }

    std::vector<CharString> GetAudits() const {
        return GetAuditData().vAuditData;
    }

    int64_t GetTimeStamp() const {
        return GetAuditData().nTimeStamp;
    }

    uint256 GetHash() const;
    bool Sign(const CKey& key);
    bool CheckSignature(const std::vector<unsigned char>& vchPubKey) const;
    int Version() const;
    bool ValidateValues(std::string& strErrorMessage) const;
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash);
    bool UnserializeFromTx(const CTransactionRef& tx);
    std::string ToString() const;
};

bool BuildAuditJson(const CAudit& audit, UniValue& oAudit);

#endif // DYNAMIC_BDAP_AUDITDATA_H