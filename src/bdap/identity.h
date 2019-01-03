// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_IDENTITY_H
#define DYNAMIC_BDAP_IDENTITY_H

#include "bdap.h"
#include "domainentry.h"
#include "primitives/transaction.h"
#include "serialize.h"
#include "uint256.h"

class CIdentity {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    CharString OwnerFullPath;  // name of the owner's full domain entry path
    CharString VerificationData;
    unsigned int nHeight;

    uint256 txHash;

    CDomainEntry* OwnerDomainEntry;

    CIdentity() {
        SetNull();
    }

    CIdentity(const CTransactionRef& tx) {
        SetNull();
        UnserializeFromTx(tx);
    }

    inline void SetNull()
    {
        nVersion = CIdentity::CURRENT_VERSION;
        OwnerFullPath.clear();
        VerificationData.clear();
        nHeight = 0;
        txHash.SetNull();
        OwnerDomainEntry = nullptr;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(OwnerFullPath);
        READWRITE(VerificationData);
        READWRITE(VARINT(nHeight));
        READWRITE(txHash);
    }

    inline friend bool operator==(const CIdentity& a, const CIdentity& b) {
        return (a.OwnerFullPath == b.OwnerFullPath && a.VerificationData == b.VerificationData);
    }

    inline friend bool operator!=(const CIdentity& a, const CIdentity& b) {
        return !(a == b);
    }

    inline CIdentity operator=(const CIdentity& b) {
        OwnerFullPath = b.OwnerFullPath;
        VerificationData = b.VerificationData;
        nHeight = b.nHeight;
        txHash = b.txHash;
        return *this;
    }
 
    inline bool IsNull() const { return (OwnerFullPath.empty()); }
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash);
    bool UnserializeFromTx(const CTransactionRef& tx);
};

class CIdentityVerification {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    CharString VerifierFullPath;  // name of the verifier's full domain entry path
    CIdentity Identity;
    CharString VerificationData;
    unsigned int nHeight;
    uint64_t nExpireTime;

    uint256 txHash;

    CDomainEntry* VerifierDomainEntry;

    CIdentityVerification() {
        SetNull();
    }

    CIdentityVerification(const CTransactionRef& tx) {
        SetNull();
        UnserializeFromTx(tx);
    }

    inline void SetNull()
    {
        nVersion = CIdentityVerification::CURRENT_VERSION;
        VerifierFullPath.clear();
        Identity.SetNull();
        VerificationData.clear();
        nHeight = 0;
        nExpireTime = 0;
        txHash.SetNull();
        VerifierDomainEntry = nullptr;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(VerifierFullPath);
        READWRITE(Identity);
        READWRITE(VerificationData);
        READWRITE(VARINT(nHeight));
        READWRITE(VARINT(nExpireTime));
        READWRITE(txHash);
    }

    inline friend bool operator==(const CIdentityVerification& a, const CIdentityVerification& b) {
        return (a.VerifierFullPath == b.VerifierFullPath && a.Identity == b.Identity &&  a.VerificationData == b.VerificationData && a.nExpireTime == b.nExpireTime);
    }

    inline friend bool operator!=(const CIdentityVerification& a, const CIdentityVerification& b) {
        return !(a == b);
    }

    inline CIdentityVerification operator=(const CIdentityVerification& b) {
        VerifierFullPath = b.VerifierFullPath;
        Identity = b.Identity;
        VerificationData = b.VerificationData;
        nHeight = b.nHeight;
        nExpireTime = b.nExpireTime;
        txHash = b.txHash;
        return *this;
    }
 
    inline bool IsNull() const { return (VerifierFullPath.empty()); }
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash);
    bool UnserializeFromTx(const CTransactionRef& tx);

};

#endif // DYNAMIC_BDAP_DOMAINENTRY_H