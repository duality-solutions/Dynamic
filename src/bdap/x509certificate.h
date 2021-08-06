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

//Implementing X.509 Certificates
class CX509Certificate {
public:
    static const int CURRENT_VERSION = 1;
    int nVersion;

    uint16_t MonthsValid;
    CharString Subject; //owner full path
    CharString SubjectSignature; //BDAP ed25519 signature (PubKeyBytes)
    CharString Issuer; //authority full path
    CharString IssuerSignature; //BDAP ed25519 signature (PubKeyBytes)
    CharString SubjectPublicKey; //new ed25519 PublicKey for the x509certificate owned by the subject (PubKeyBytes)
    CharString IssuerPublicKey; //new ed25519 PublicKey for the x509certificate owned by the issuer [rootCA] (PubKeyBytes)
    uint64_t SerialNumber; //must be unique
    CharString PEM;
    CharString ExternalVerificationFile;
    bool IsRootCA;

    //Needed for blockchain
    unsigned int nHeightRequest; 
    unsigned int nHeightSigned; 
    uint256 txHashRequest;
    uint256 txHashSigned;

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
        SubjectPublicKey.clear();
        IssuerPublicKey.clear();
        SerialNumber = 0;
        PEM.clear();
        ExternalVerificationFile.clear();
        IsRootCA = false;
        nHeightRequest = 0;
        nHeightSigned = 0;
        txHashRequest.SetNull();
        txHashSigned.SetNull();
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
        READWRITE(SubjectPublicKey);
        READWRITE(IssuerPublicKey);
        READWRITE(VARINT(SerialNumber));
        READWRITE(PEM);
        READWRITE(IsRootCA);
        READWRITE(ExternalVerificationFile);
        READWRITE(VARINT(nHeightRequest));
        READWRITE(VARINT(nHeightSigned));
        READWRITE(txHashRequest);
        READWRITE(txHashSigned);
    }

    inline friend bool operator==(const CX509Certificate &a, const CX509Certificate &b) {
        return (a.MonthsValid == b.MonthsValid &&
        a.Subject == b.Subject &&
        a.SubjectSignature == b.SubjectSignature &&
        a.Issuer == b.Issuer &&
        a.IssuerSignature == b.IssuerSignature &&
        a.SubjectPublicKey == b.SubjectPublicKey &&
        a.IssuerPublicKey == b.IssuerPublicKey &&
        a.SerialNumber == b.SerialNumber &&
        a.PEM == b.PEM &&
        a.IsRootCA == b.IsRootCA &&
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
        SubjectPublicKey = b.SubjectPublicKey;
        IssuerPublicKey = b.IssuerPublicKey;
        SerialNumber = b.SerialNumber;
        PEM = b.PEM;
        IsRootCA = b.IsRootCA;
        ExternalVerificationFile = b.ExternalVerificationFile;
        nHeightRequest = b.nHeightRequest;
        nHeightSigned = b.nHeightSigned;
        txHashRequest = b.txHashRequest;
        txHashSigned = b.txHashSigned;
        return *this;
    }
 
    //Note: if root certificate, consider approved (self-signed)
    bool IsApproved() const {
        return ((IssuerSignature.size() > 0) || (txHashSigned != 0) || (IsRootCA));
    }

    bool SelfSignedX509Certificate() const {
        
        if (Subject == Issuer)
            return true;

        return false;
    }

    //needs review
    CKeyID GetX509CertificateKeyID() const {
        return CKeyID(Hash160(SubjectPublicKey.begin(), SubjectPublicKey.end()));
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
    //bool SetSerialNumber();
    std::string GetPubKeyHex() const;
    std::string GetIssuerPubKeyHex() const;
    std::string GetPubKeyBase64() const;
    std::string GetSubjectSignature() const;
    std::string GetIssuerSignature() const;
    bool SignSubject(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchPrivKey);
    bool SignIssuer(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchPrivKey);
    bool CheckSubjectSignature(const std::vector<unsigned char>& vchPubKey) const;
    bool CheckIssuerSignature(const std::vector<unsigned char>& vchPubKey) const;
    unsigned char* TestSign(const std::vector<unsigned char>& vchPrivSeedBytes, const std::vector<unsigned char>& vchData) const;
    bool VerifySignature(const std::vector<unsigned char>& vchSignature, const std::vector<unsigned char>& vchData) const;

    bool ValidateValues(std::string& errorMessage) const;
    bool ValidatePEM(std::string& errorMessage) const;
    bool ValidatePEMSignature(std::string& errorMessage) const;
    bool X509SelfSign(const std::vector<unsigned char>& vchSubjectPrivKey); //Pass PrivKeyBytes
    bool X509RequestSign(const std::vector<unsigned char>& vchSubjectPrivSeedBytes); //Pass PrivSeedBytes
    bool X509ApproveSign(const std::vector<unsigned char>& pemCA, const std::vector<unsigned char>& vchIssuerPrivSeedBytes);
    bool X509TestApproveSign(const std::vector<unsigned char>& vchSubjectPrivSeedBytes, const std::vector<unsigned char>& vchIssuerPrivSeedBytes);
    bool X509RootCASign(const std::vector<unsigned char>& vchIssuerPrivSeedBytes);  //Pass PrivSeedBytes
    bool X509Export(const std::vector<unsigned char>& vchSubjectPrivSeedBytes, std::string filename = "");  //Pass PrivSeedBytes
    bool X509ExportRoot(std::string filename = "");

    bool CheckIfExistsInMemPool(const CTxMemPool& pool, std::string& errorMessage);

    std::string GetPEMSubject() const;
    std::string GetReqPEMSubject() const;
    std::string GetPEMIssuer() const;
    std::string GetPEMPubKey() const;
    std::string GetReqPEMPubKey() const;
    std::string GetPEMSerialNumber() const;

    std::string ToString() const;
};

bool BuildX509CertificateJson(const CX509Certificate& x509certificate, UniValue& oX509Certificate);


#endif // DYNAMIC_BDAP_X509CERTIFICATE_H