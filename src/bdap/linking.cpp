// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: license

#include "bdap/linking.h"

#include "bdap/domainentry.h"
#include "hash.h"
#include "script/script.h"
#include "streams.h"

bool CRequestLink::UnserializeFromTx(const CTransactionRef& tx) 
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

void CRequestLink::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsRequestLink(SER_NETWORK, PROTOCOL_VERSION);
    dsRequestLink << *this;
    vchData = std::vector<unsigned char>(dsRequestLink.begin(), dsRequestLink.end());
}

bool CRequestLink::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsRequestLink(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsRequestLink >> *this;

        std::vector<unsigned char> vchRequestLinkData;
        Serialize(vchRequestLinkData);
        const uint256 &calculatedHash = Hash(vchRequestLinkData.begin(), vchRequestLinkData.end());
        const std::vector<unsigned char> &vchRandRequestLink = vchFromValue(calculatedHash.GetHex());
        if(vchRandRequestLink != vchHash)
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

bool CRequestLink::ValidateValues(std::string& errorMessage)
{
    return true;
}


bool CRequestLink::IsMyLinkRequest(const CTransactionRef& tx)
{
    return false;
}

void CAcceptLink::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsAcceptLink(SER_NETWORK, PROTOCOL_VERSION);
    dsAcceptLink << *this;
    vchData = std::vector<unsigned char>(dsAcceptLink.begin(), dsAcceptLink.end());
}

bool CAcceptLink::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsAcceptLink(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsAcceptLink >> *this;

        std::vector<unsigned char> vchAcceptLinkData;
        Serialize(vchAcceptLinkData);
        const uint256 &calculatedHash = Hash(vchAcceptLinkData.begin(), vchAcceptLinkData.end());
        const std::vector<unsigned char> &vchRandAcceptLink = vchFromValue(calculatedHash.GetHex());
        if(vchRandAcceptLink != vchHash)
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

bool CAcceptLink::ValidateValues(std::string& errorMessage)
{
    // TODO: (BDAP) implement validation
    return true;
}

// TODO: (BDAP) implement BDAP encryption by converting ed25519 private key to curve25519
/*
CharString GetEncryptedLinkMessage(const CRequestLink& requestLink)
{

}

CharString GetEncryptedLinkMessage(const CRequestLink& requestLink, const CAcceptLink& acceptLink)
{


}
*/