// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_ENTRYCERTIFICATE_H
#define DYNAMIC_BDAP_ENTRYCERTIFICATE_H

#include "bdap.h"
#include "bdap/domainentry.h"
#include "serialize.h"
#include "uint256.h"

class CTransaction;

class CEntryCertificate {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    CharString OwnerFullPath;  // name of the owner's full domain entry path
    CharString Name; // Certificate name
    CharString CertificateData;
    CharString AuthorityFullPath;
    CharString AuthoritySignature;
    unsigned int nHeight;
    uint64_t nExpireTime;
    uint256 txHash;
    CDomainEntry* OwnerDomainEntry;
    CDomainEntry* AuthorityDomainEntry;

    CEntryCertificate() {
        SetNull();
    }

    CEntryCertificate(const CTransactionRef& tx) {
        SetNull();
        UnserializeFromTx(tx);
    }

    inline void SetNull()
    {
        nVersion = CEntryCertificate::CURRENT_VERSION;
        OwnerFullPath.clear();
        Name.clear();
        CertificateData.clear();
        AuthorityFullPath.clear();
        AuthoritySignature.clear();
        nHeight = 0;
        nExpireTime = 0;
        txHash.SetNull();
        OwnerDomainEntry = nullptr;
        AuthorityDomainEntry = nullptr;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(OwnerFullPath);
        READWRITE(Name);
        READWRITE(CertificateData);
        READWRITE(AuthorityFullPath);
        READWRITE(AuthoritySignature);
        READWRITE(VARINT(nHeight));
        READWRITE(VARINT(nExpireTime));
        READWRITE(txHash);
    }

    inline friend bool operator==(const CEntryCertificate &a, const CEntryCertificate &b) {
        return (a.OwnerFullPath == b.OwnerFullPath && a.Name == b.Name && a.CertificateData == b.CertificateData && a.nExpireTime == b.nExpireTime);
    }

    inline friend bool operator!=(const CEntryCertificate &a, const CEntryCertificate &b) {
        return !(a == b);
    }

    inline CEntryCertificate operator=(const CEntryCertificate &b) {
        OwnerFullPath = b.OwnerFullPath;
        Name = b.Name;
        CertificateData = b.CertificateData;
        AuthorityFullPath = b.AuthorityFullPath;
        AuthoritySignature = b.AuthoritySignature;
        nHeight = b.nHeight;
        nExpireTime = b.nExpireTime;
        txHash = b.txHash;
        return *this;
    }
 
    inline bool IsNull() const { return (OwnerFullPath.empty()); }
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char> &vchData, const std::vector<unsigned char> &vchHash);
    bool UnserializeFromTx(const CTransactionRef& tx);

    bool SelfSignedCertificate() const {
        if (OwnerDomainEntry == nullptr || AuthorityDomainEntry == nullptr)
            return false;
        
        if (OwnerDomainEntry == AuthorityDomainEntry)
            return true;

        return false;
    }

    bool ValidateValues(std::string& errorMessage);
};

#endif // DYNAMIC_BDAP_ENTRYCERTIFICATE_H