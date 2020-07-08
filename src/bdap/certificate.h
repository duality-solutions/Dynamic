// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_CERTIFICATE_H
#define DYNAMIC_BDAP_CERTIFICATE_H

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

//Implementing X.509 Certificates
class CCertificate {
public:
    static const int CURRENT_VERSION = 1;
    int nVersion;

    CharString SignatureAlgorithm; //only support ed25519 and secp256k1 (future)
    CharString SignatureHashAlgorithm; //sha256
    CharString FingerPrint; //hash of the publickey (possible method)
    uint16_t MonthsValid;
    CharString Subject; //owner full path
    CharString SubjectSignature; //BDAP account (in lieu of proof of domain) (take this class serialized and sign it)
    CharString Issuer; //authority full path
    CharString PublicKey; //new PublicKey for the certificate owned by the issuer
    CharString SignatureValue; //Issuer Signature
    uint64_t SerialNumber; //must be unique

    std::vector<CharString> KeyUsage; 
    std::vector<CharString> ExtendedKeyUsage; 

    std::vector<CharString> AuthorityInformationAccess;
    std::vector<CharString> SubjectAlternativeName;
    std::vector<CharString> Policies;
    std::vector<CharString> CRLDistributionPoints;
    std::vector<CharString> SCTList;

    //Needed for blockchain
    unsigned int nHeightRequest; 
    unsigned int nHeightApprove; 
    uint256 txHashRequest;
    uint256 txHashApprove;

    CCertificate() {
        SetNull();
    }

    CCertificate(const CTransactionRef& tx) {
        SetNull();
        UnserializeFromTx(tx);
    }

    inline void SetNull()
    {
        nVersion = CCertificate::CURRENT_VERSION;
        SignatureAlgorithm.clear();
        SignatureHashAlgorithm.clear();
        FingerPrint.clear();
        MonthsValid = 0;
        Subject.clear();
        SubjectSignature.clear();
        Issuer.clear();
        PublicKey.clear();
        SignatureValue.clear();
        SerialNumber = 0;
        KeyUsage.clear();
        ExtendedKeyUsage.clear();
        AuthorityInformationAccess.clear();
        SubjectAlternativeName.clear();
        Policies.clear();
        CRLDistributionPoints.clear();
        SCTList.clear();
        nHeightRequest = 0;
        nHeightApprove = 0;
        txHashRequest.SetNull();
        txHashApprove.SetNull();
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(SignatureAlgorithm);
        READWRITE(SignatureHashAlgorithm);
        READWRITE(FingerPrint);
        READWRITE(VARINT(MonthsValid));
        READWRITE(Subject);
        READWRITE(SubjectSignature);
        READWRITE(Issuer);
        READWRITE(PublicKey);
        READWRITE(SignatureValue);
        READWRITE(VARINT(SerialNumber));
        READWRITE(KeyUsage);
        READWRITE(ExtendedKeyUsage);
        READWRITE(AuthorityInformationAccess);
        READWRITE(SubjectAlternativeName);
        READWRITE(Policies);
        READWRITE(CRLDistributionPoints);
        READWRITE(SCTList);
        READWRITE(VARINT(nHeightRequest));
        READWRITE(VARINT(nHeightApprove));
        READWRITE(txHashRequest);
        READWRITE(txHashApprove);
    }

    inline friend bool operator==(const CCertificate &a, const CCertificate &b) {
        return (a.SignatureAlgorithm == b.SignatureAlgorithm &&
        a.SignatureHashAlgorithm == b.SignatureHashAlgorithm &&
        a.FingerPrint == b.FingerPrint &&
        a.MonthsValid == b.MonthsValid &&
        a.Subject == b.Subject &&
        a.SubjectSignature == b.SubjectSignature &&
        a.Issuer == b.Issuer &&
        a.PublicKey == b.PublicKey &&
        a.SignatureValue == b.SignatureValue &&
        a.SerialNumber == b.SerialNumber &&
        a.KeyUsage == b.KeyUsage &&
        a.ExtendedKeyUsage == b.ExtendedKeyUsage &&
        a.AuthorityInformationAccess == b.AuthorityInformationAccess &&
        a.SubjectAlternativeName == b.SubjectAlternativeName &&
        a.Policies == b.Policies &&
        a.CRLDistributionPoints == b.CRLDistributionPoints &&
        a.SCTList == b.SCTList);
    }

    inline friend bool operator!=(const CCertificate &a, const CCertificate &b) {
        return !(a == b);
    }

    inline CCertificate operator=(const CCertificate &b) {
        SignatureAlgorithm = b.SignatureAlgorithm;
        SignatureHashAlgorithm = b.SignatureHashAlgorithm;
        FingerPrint = b.FingerPrint;
        MonthsValid = b.MonthsValid;
        Subject = b.Subject;
        SubjectSignature = b.SubjectSignature;
        Issuer = b.Issuer;
        PublicKey = b.PublicKey;
        SignatureValue = b.SignatureValue;
        SerialNumber = b.SerialNumber;
        KeyUsage = b.KeyUsage;
        ExtendedKeyUsage = b.ExtendedKeyUsage;
        AuthorityInformationAccess = b.AuthorityInformationAccess;
        SubjectAlternativeName = b.SubjectAlternativeName;
        Policies = b.Policies;
        CRLDistributionPoints = b.CRLDistributionPoints;
        SCTList = b.SCTList;
        nHeightRequest = b.nHeightRequest;
        nHeightApprove = b.nHeightApprove;
        txHashRequest = b.txHashRequest;
        txHashApprove = b.txHashApprove;
        return *this;
    }
 
    inline bool IsNull() const { return (Subject.empty()); }
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char> &vchData, const std::vector<unsigned char> &vchHash);
    bool UnserializeFromTx(const CTransactionRef& tx, const unsigned int& height=0);

    bool SelfSignedCertificate() const {
        
        if (Subject == Issuer)
            return true;

        return false;
    }

    uint256 GetHash() const;
    uint256 GetSubjectHash() const;
    uint256 GetIssuerHash() const;
    bool SignSubject(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchPrivKey);
    bool SignIssuer(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchPrivKey);
    bool CheckSubjectSignature(const std::vector<unsigned char>& vchPubKey);
    bool CheckIssuerSignature(const std::vector<unsigned char>& vchPubKey);
    bool ValidateValues(std::string& errorMessage);
};

bool BuildCertificateJson(const CCertificate& certificate, UniValue& oCertificate);


#endif // DYNAMIC_BDAP_CERTIFICATE_H