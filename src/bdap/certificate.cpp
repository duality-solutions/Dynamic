
// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/certificate.h"

#include "bdap/utils.h"
#include "hash.h"
#include "script/script.h"
#include "streams.h"
#include "validation.h"

#include <libtorrent/ed25519.hpp>
#include "uint256.h"


#include <univalue.h>

void CCertificate::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsEntryCertificate(SER_NETWORK, PROTOCOL_VERSION);
    dsEntryCertificate << *this;
    vchData = std::vector<unsigned char>(dsEntryCertificate.begin(), dsEntryCertificate.end());
}

bool CCertificate::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsEntryCertificate(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsEntryCertificate >> *this;

        std::vector<unsigned char> vchEntryLinkData;
        Serialize(vchEntryLinkData);
        const uint256 &calculatedHash = Hash(vchEntryLinkData.begin(), vchEntryLinkData.end());
        const std::vector<unsigned char> &vchRandEntryLink = vchFromValue(calculatedHash.GetHex());
        if(vchRandEntryLink != vchHash)
        {
            SetNull();
            return false;
        }
    } catch (std::exception &e) {
        SetNull();
        return false;
    }
    return true;
}

bool CCertificate::UnserializeFromTx(const CTransactionRef& tx, const unsigned int& height) 
{
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nOut;
    if(!GetBDAPData(tx, vchData, vchHash, nOut))
    {
        SetNull();
        return false;
    }
    if(!UnserializeFromData(vchData, vchHash))
    {
        return false;
    }
    //TODO: Distringuish between Request and Approve
    //if issuer = subject 
    txHashRequest = tx->GetHash();
    nHeightRequest = height;
    return true;
}

uint256 CCertificate::GetHash() const
{
    CDataStream dsCertificate(SER_NETWORK, PROTOCOL_VERSION);
    dsCertificate << *this;
    return Hash(dsCertificate.begin(), dsCertificate.end());
}

uint256 CCertificate::GetSubjectHash() const
{
    CDataStream dsCertificate(SER_NETWORK, PROTOCOL_VERSION);
    //dsCertificate << Subject << SignatureAlgorithm << SignatureHashAlgorithm << SerialNumber;
    dsCertificate << SignatureAlgorithm << SignatureHashAlgorithm << Subject << SerialNumber << KeyUsage << ExtendedKeyUsage << AuthorityInformationAccess << SubjectAlternativeName << Policies << CRLDistributionPoints << SCTList;
    return Hash(dsCertificate.begin(), dsCertificate.end());
}

uint256 CCertificate::GetIssuerHash() const
{
    CDataStream dsCertificate(SER_NETWORK, PROTOCOL_VERSION);
    //dsCertificate << Issuer << Subject << SignatureAlgorithm << SignatureHashAlgorithm << SerialNumber;
    dsCertificate << SignatureAlgorithm << SignatureHashAlgorithm << MonthsValid << Subject << SubjectSignature << Issuer << PublicKey << SerialNumber << KeyUsage << ExtendedKeyUsage << AuthorityInformationAccess << SubjectAlternativeName << Policies << CRLDistributionPoints << SCTList;
    return Hash(dsCertificate.begin(), dsCertificate.end());
}

bool CCertificate::SignSubject(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchPrivKey)
{
    std::vector<unsigned char> msg = vchFromString(GetSubjectHash().ToString());
    std::vector<unsigned char> sig(64);

    libtorrent::ed25519_sign(&sig[0], &msg[0], msg.size(), &vchPubKey[0], &vchPrivKey[0]);
    SubjectSignature = sig;

    return true;
}

bool CCertificate::SignIssuer(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchPrivKey)
{
    std::vector<unsigned char> msg = vchFromString(GetIssuerHash().ToString());
    std::vector<unsigned char> sig(64);

    libtorrent::ed25519_sign(&sig[0], &msg[0], msg.size(), &vchPubKey[0], &vchPrivKey[0]);
    SignatureValue = sig;

    return true;
}

bool CCertificate::CheckSubjectSignature(const std::vector<unsigned char>& vchPubKey)
{
    std::vector<unsigned char> msg = vchFromString(GetSubjectHash().ToString());

    if (!libtorrent::ed25519_verify(&SubjectSignature[0], &msg[0], msg.size(), &vchPubKey[0])) {
        return false;
    }

    return true;
}

bool CCertificate::CheckIssuerSignature(const std::vector<unsigned char>& vchPubKey)
{
    std::vector<unsigned char> msg = vchFromString(GetIssuerHash().ToString());

    if (!libtorrent::ed25519_verify(&SignatureValue[0], &msg[0], msg.size(), &vchPubKey[0])) {
        return false;
    }

    return true;
}

bool CCertificate::ValidateValues(std::string& errorMessage)
{
    // check Signature Algorithm
    std::string strSignatureAlgorithm = stringFromVch(SignatureAlgorithm);
    if (strSignatureAlgorithm.length() > MAX_ALGORITHM_TYPE_LENGTH) 
    {
        errorMessage = "Invalid Signature Algorithm. Can not have more than " + std::to_string(MAX_ALGORITHM_TYPE_LENGTH) + " characters.";
        return false;
    }

    // check SignatureHashAlgorithm
    std::string strSignatureHashAlgorithm = stringFromVch(SignatureHashAlgorithm);
    if (strSignatureHashAlgorithm.length() > MAX_ALGORITHM_TYPE_LENGTH) 
    {
        errorMessage = "Invalid Signature Hash Algorithm. Can not have more than " + std::to_string(MAX_ALGORITHM_TYPE_LENGTH) + " characters.";
        return false;
    }

    // check FingerPrint
    std::string strFingerPrint = stringFromVch(FingerPrint);
    if (strFingerPrint.length() > MAX_CERTIFICATE_FINGERPRINT) 
    {
        errorMessage = "Invalid Finger Print. Can not have more than " + std::to_string(MAX_CERTIFICATE_FINGERPRINT) + " characters.";
        return false;
    }

    //TODO: ValidFrom and ValidTo checks?

    // check subject owner path
    std::string strSubject = stringFromVch(Subject);
    if (strSubject.length() > MAX_OBJECT_FULL_PATH_LENGTH) 
    {
        errorMessage = "Invalid Subject full path name. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }

    // check SubjectSignature
    std::string strSubjectSignature = stringFromVch(SubjectSignature);
    if (strSubjectSignature.length() > MAX_SIGNATURE_LENGTH) 
    {
        errorMessage = "Invalid SubjectSignature. Can not have more than " + std::to_string(MAX_SIGNATURE_LENGTH) + " characters.";
        return false;
    }

    // check issuer owner path
    std::string strIssuer = stringFromVch(Issuer);
    if (strIssuer.length() > MAX_OBJECT_FULL_PATH_LENGTH) 
    {
        errorMessage = "Invalid Issuer full path name. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }

    // check PublicKey
    std::string strPublicKey = stringFromVch(PublicKey);
    if (strPublicKey.length() > MAX_KEY_LENGTH) 
    {
        errorMessage = "Invalid PublicKey. Can not have more than " + std::to_string(MAX_KEY_LENGTH) + " characters.";
        return false;
    }

    // check SignatureValue
    std::string strSignatureValue = stringFromVch(SignatureValue);
    if (strSignatureValue.length() > MAX_SIGNATURE_LENGTH) 
    {
        errorMessage = "Invalid SubjectSiSignatureValuegnature. Can not have more than " + std::to_string(MAX_SIGNATURE_LENGTH) + " characters.";
        return false;
    }

    //check KeyUsage (amount of records, and length of each record)
    if (KeyUsage.size() > MAX_CERTIFICATE_EXTENSION_RECORDS)
    {
        errorMessage = "Invalid KeyUsage size. Can not have more than " + std::to_string(MAX_CERTIFICATE_EXTENSION_RECORDS) + " records.";
        return false;
    }
    for (const CharString& KeyUsageValue : KeyUsage) {
        std::string strKeyUsageValue = stringFromVch(KeyUsageValue);
        if (strKeyUsageValue.length() > MAX_CERTIFICATE_EXTENSION_LENGTH) 
        {
            errorMessage = "Invalid KeyUsage. Can not have more than " + std::to_string(MAX_CERTIFICATE_EXTENSION_LENGTH) + " characters.";
            return false;
        }
    }

    //check ExtendedKeyUsage (amount of records, and length of each record)
    if (ExtendedKeyUsage.size() > MAX_CERTIFICATE_EXTENSION_RECORDS)
    {
        errorMessage = "Invalid ExtendedKeyUsage size. Can not have more than " + std::to_string(MAX_CERTIFICATE_EXTENSION_RECORDS) + " records.";
        return false;
    }
    for (const CharString& ExtendedKeyUsageValue : ExtendedKeyUsage) {
        std::string strExtendedKeyUsageValue = stringFromVch(ExtendedKeyUsageValue);
        if (strExtendedKeyUsageValue.length() > MAX_CERTIFICATE_EXTENSION_LENGTH) 
        {
            errorMessage = "Invalid ExtendedKeyUsage. Can not have more than " + std::to_string(MAX_CERTIFICATE_EXTENSION_LENGTH) + " characters.";
            return false;
        }
    }

    //check AuthorityInformationAccess (amount of records, and length of each record)
    if (AuthorityInformationAccess.size() > MAX_CERTIFICATE_EXTENSION_RECORDS)
    {
        errorMessage = "Invalid AuthorityInformationAccess size. Can not have more than " + std::to_string(MAX_CERTIFICATE_EXTENSION_RECORDS) + " records.";
        return false;
    }
    for (const CharString& AuthorityInformationAccessValue : AuthorityInformationAccess) {
        std::string strAuthorityInformationAccessValue = stringFromVch(AuthorityInformationAccessValue);
        if (strAuthorityInformationAccessValue.length() > MAX_CERTIFICATE_EXTENSION_LENGTH) 
        {
            errorMessage = "Invalid AuthorityInformationAccess. Can not have more than " + std::to_string(MAX_CERTIFICATE_EXTENSION_LENGTH) + " characters.";
            return false;
        }
    }

    //check SubjectAlternativeName (amount of records, and length of each record)
    if (SubjectAlternativeName.size() > MAX_CERTIFICATE_EXTENSION_RECORDS)
    {
        errorMessage = "Invalid SubjectAlternativeName size. Can not have more than " + std::to_string(MAX_CERTIFICATE_EXTENSION_RECORDS) + " records.";
        return false;
    }
    for (const CharString& SubjectAlternativeNameValue : SubjectAlternativeName) {
        std::string strSubjectAlternativeNameValue = stringFromVch(SubjectAlternativeNameValue);
        if (strSubjectAlternativeNameValue.length() > MAX_CERTIFICATE_EXTENSION_LENGTH) 
        {
            errorMessage = "Invalid SubjectAlternativeName. Can not have more than " + std::to_string(MAX_CERTIFICATE_EXTENSION_LENGTH) + " characters.";
            return false;
        }
    }

    //check Policies (amount of records, and length of each record)
    if (Policies.size() > MAX_CERTIFICATE_EXTENSION_RECORDS)
    {
        errorMessage = "Invalid Policies size. Can not have more than " + std::to_string(MAX_CERTIFICATE_EXTENSION_RECORDS) + " records.";
        return false;
    }
    for (const CharString& PoliciesValue : Policies) {
        std::string strPoliciesValue = stringFromVch(PoliciesValue);
        if (strPoliciesValue.length() > MAX_CERTIFICATE_EXTENSION_LENGTH) 
        {
            errorMessage = "Invalid Policies. Can not have more than " + std::to_string(MAX_CERTIFICATE_EXTENSION_LENGTH) + " characters.";
            return false;
        }
    }

    //check CRLDistributionPoints (amount of records, and length of each record)
    if (CRLDistributionPoints.size() > MAX_CERTIFICATE_EXTENSION_RECORDS)
    {
        errorMessage = "Invalid CRLDistributionPoints size. Can not have more than " + std::to_string(MAX_CERTIFICATE_EXTENSION_RECORDS) + " records.";
        return false;
    }
    for (const CharString& CRLDistributionPointsValue : CRLDistributionPoints) {
        std::string strCRLDistributionPointsValue = stringFromVch(CRLDistributionPointsValue);
        if (strCRLDistributionPointsValue.length() > MAX_CERTIFICATE_EXTENSION_LENGTH) 
        {
            errorMessage = "Invalid CRLDistributionPoints. Can not have more than " + std::to_string(MAX_CERTIFICATE_EXTENSION_LENGTH) + " characters.";
            return false;
        }
    }

    //check SCTList (amount of records, and length of each record)
    if (SCTList.size() > MAX_CERTIFICATE_EXTENSION_RECORDS)
    {
        errorMessage = "Invalid SCTList size. Can not have more than " + std::to_string(MAX_CERTIFICATE_EXTENSION_RECORDS) + " records.";
        return false;
    }
    for (const CharString& SCTListValue : SCTList) {
        std::string strSCTListValue = stringFromVch(SCTListValue);
        if (strSCTListValue.length() > MAX_CERTIFICATE_EXTENSION_LENGTH) 
        {
            errorMessage = "Invalid SCTList. Can not have more than " + std::to_string(MAX_CERTIFICATE_EXTENSION_LENGTH) + " characters.";
            return false;
        }
    }

    return true;
}

bool BuildCertificateJson(const CCertificate& certificate, UniValue& oCertificate)
{
    int64_t nTime = 0;

    std::vector<unsigned char> subjectSig = certificate.SubjectSignature;
    std::vector<unsigned char> issuerSig = certificate.SignatureValue;

    oCertificate.push_back(Pair("version", std::to_string(certificate.nVersion)));

    oCertificate.push_back(Pair("signature_algorithm", stringFromVch(certificate.SignatureAlgorithm)));
    oCertificate.push_back(Pair("signature_hash_algorithm", stringFromVch(certificate.SignatureHashAlgorithm)));
    oCertificate.push_back(Pair("fingerprint", stringFromVch(certificate.FingerPrint)));
    oCertificate.push_back(Pair("months_valid", std::to_string(certificate.MonthsValid)));
    oCertificate.push_back(Pair("subject", stringFromVch(certificate.Subject)));
    oCertificate.push_back(Pair("subject_signature", EncodeBase64(&subjectSig[0], subjectSig.size())));
    oCertificate.push_back(Pair("issuer", stringFromVch(certificate.Issuer)));
    oCertificate.push_back(Pair("public_key", stringFromVch(certificate.PublicKey)));
    oCertificate.push_back(Pair("signature_value", EncodeBase64(&issuerSig[0], issuerSig.size())));
    oCertificate.push_back(Pair("serial_number", std::to_string(certificate.SerialNumber)));


    oCertificate.push_back(Pair("txid_request", certificate.txHashRequest.GetHex()));
    oCertificate.push_back(Pair("txid_approve", certificate.txHashApprove.GetHex()));
    //TODO: need to change block_height calculation
    if ((unsigned int)chainActive.Height() >= certificate.nHeightRequest) {
        CBlockIndex *pindex = chainActive[certificate.nHeightRequest];
        if (pindex) {
            nTime = pindex->GetBlockTime();
        }
    }
    oCertificate.push_back(Pair("block_time", nTime));
    oCertificate.push_back(Pair("block_height", std::to_string(certificate.nHeightRequest)));
    
    return true;
}
