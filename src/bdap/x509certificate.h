// Copyright (c) 2020 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_X509CERTIFICATE_H
#define DYNAMIC_BDAP_X509CERTIFICATE_H

#include "bdap.h"
#include "bdap/domainentry.h"
#include "hash.h"
#include "primitives/transaction.h"
#include "pubkey.h"
#include "serialize.h"
#include "uint256.h"

class CKey;
class CTransaction;
class UniValue;
class CKeyEd25519;

//Implementing X.509 X509Certificates
class CX509Certificate {
public:
    static const int CURRENT_VERSION = 1;
    int nVersion;

    uint16_t MonthsValid;
    CharString Subject; //owner full path
    CharString SubjectSignature;
    CharString Issuer; //authority full path
    CharString IssuerSignature;
    CharString PublicKey; //new PublicKey for the x509certificate owned by the subject
    uint64_t SerialNumber; //must be unique
    CharString PEM;
    CharString ExternalVerificationFile;

    //Needed for blockchain
    unsigned int nHeightRequest; 
    unsigned int nHeightApprove; 
    uint256 txHashRequest;
    
    uint256 txHashApprove;

    CX509Certificate() {
        SetNull();
    }

    CX509Certificate(const CTransactionRef& tx) {
        SetNull();
        UnserializeFromTx(tx);
    }

    inline void SetNull()
    {
        nVersion = CX509Certificate::CURRENT_VERSION;
        MonthsValid = 0;
        Subject.clear();
        SubjectSignature.clear();
        Issuer.clear();
        IssuerSignature.clear();
        PublicKey.clear();
        SerialNumber = 0;
        PEM.clear();
        ExternalVerificationFile.clear();
        nHeightRequest = 0;
        nHeightApprove = 0;
        txHashRequest.SetNull();
        txHashApprove.SetNull();
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(VARINT(MonthsValid));
        READWRITE(Subject);
        READWRITE(SubjectSignature);
        READWRITE(Issuer);
        READWRITE(IssuerSignature);
        READWRITE(PublicKey);
        READWRITE(VARINT(SerialNumber));
        READWRITE(PEM);
        READWRITE(ExternalVerificationFile);
        READWRITE(VARINT(nHeightRequest));
        READWRITE(VARINT(nHeightApprove));
        READWRITE(txHashRequest);
        READWRITE(txHashApprove);
    }

    inline friend bool operator==(const CX509Certificate &a, const CX509Certificate &b) {
        return (a.MonthsValid == b.MonthsValid &&
        a.Subject == b.Subject &&
        a.SubjectSignature == b.SubjectSignature &&
        a.Issuer == b.Issuer &&
        a.IssuerSignature == b.IssuerSignature &&
        a.PublicKey == b.PublicKey &&
        a.SerialNumber == b.SerialNumber &&
        a.PEM == b.PEM &&
        a.ExternalVerificationFile == b.ExternalVerificationFile);
    }

    inline friend bool operator!=(const CX509Certificate &a, const CX509Certificate &b) {
        return !(a == b);
    }

    inline CX509Certificate operator=(const CX509Certificate &b) {
        MonthsValid = b.MonthsValid;
        Subject = b.Subject;
        SubjectSignature = b.SubjectSignature;
        Issuer = b.Issuer;
        IssuerSignature = b.IssuerSignature;
        PublicKey = b.PublicKey;
        SerialNumber = b.SerialNumber;
        PEM = b.PEM;
        ExternalVerificationFile = b.ExternalVerificationFile;
        nHeightRequest = b.nHeightRequest;
        nHeightApprove = b.nHeightApprove;
        txHashRequest = b.txHashRequest;
        txHashApprove = b.txHashApprove;
        return *this;
    }
 
    bool IsApproved() const {
        return (txHashApprove != 0);
    }

    bool SelfSignedX509Certificate() const {
        
        if (Subject == Issuer)
            return true;

        return false;
    }

    CKeyID GetX509CertificateKeyID() const {
        return CKeyID(Hash160(PublicKey.begin(), PublicKey.end()));
    }

    std::string GetFingerPrint() const {
        return GetHash().ToString();
    }

    inline bool IsNull() const { return (Subject.empty()); }
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char> &vchData, const std::vector<unsigned char> &vchHash);
    bool UnserializeFromTx(const CTransactionRef& tx, const unsigned int& height=0);

    uint256 GetHash() const;
    uint256 GetSubjectHash() const;
    uint256 GetIssuerHash() const;
    std::string GetPubKeyHex() const;
    std::string GetSubjectSignature() const;
    std::string GetIssuerSignature() const;
    bool SignSubject(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchPrivKey);
    bool SignIssuer(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchPrivKey);
    bool CheckSubjectSignature(const std::vector<unsigned char>& vchPubKey) const;
    bool CheckIssuerSignature(const std::vector<unsigned char>& vchPubKey) const;
    bool ValidateValues(std::string& errorMessage) const;
    bool SelfSign(const std::vector<unsigned char>& vchSubjectPrivKey); //Pass PrivKeyBytes

    std::string ToString() const;
};

bool BuildX509CertificateJson(const CX509Certificate& x509certificate, UniValue& oX509Certificate);


#endif // DYNAMIC_BDAP_X509CERTIFICATE_H