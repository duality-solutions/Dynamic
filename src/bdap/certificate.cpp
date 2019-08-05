
// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/certificate.h"

#include "bdap/utils.h"
#include "hash.h"
#include "script/script.h"
#include "streams.h"

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

bool CCertificate::UnserializeFromTx(const CTransactionRef& tx) 
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
    return true;
}

bool CCertificate::ValidateValues(std::string& errorMessage)
{
    // check certificate owner path
    std::string strOwnerFullPath = stringFromVch(OwnerFullPath);
    if (strOwnerFullPath.length() > MAX_OBJECT_FULL_PATH_LENGTH) // object
    {
        errorMessage = "Invalid BDAP owner full path name. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }

    // check certificate name length
    std::string strName = stringFromVch(Name);
    if (strName.length() > MAX_CERTIFICATE_NAME) 
    {
        errorMessage = "Invalid BDAP Certificate name. Can not have more than " + std::to_string(MAX_CERTIFICATE_NAME) + " characters.";
        return false;
    }

    // check category length
    std::string strCategory = stringFromVch(Category);
    if (strCategory.length() > MAX_CERTIFICATE_CATEGORY) 
    {
        errorMessage = "Invalid BDAP Certificate name. Can not have more than " + std::to_string(MAX_CERTIFICATE_CATEGORY) + " characters.";
        return false;
    }

    // check certificate data length
    std::string strCertificate = stringFromVch(CertificateData);
    if (strCertificate.length() > MAX_CERTIFICATE_LENGTH) 
    {
        errorMessage = "Invalid BDAP Certificate data. Can not have more than " + std::to_string(MAX_CERTIFICATE_LENGTH) + " characters.";
        return false;
    }
    
    // check authority full path
    std::string strAuthorityFullPath = stringFromVch(AuthorityFullPath);
    if (strAuthorityFullPath.length() > MAX_OBJECT_FULL_PATH_LENGTH) 
    {
        errorMessage = "Invalid BDAP certificate authority full path. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }

    // check authority full path
    std::string strAuthoritySignature = stringFromVch(AuthoritySignature);
    if (strAuthoritySignature.length() > MAX_SIGNATURE_LENGTH) 
    {
        errorMessage = "Invalid BDAP certificate authority signature. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }

    return true;
}