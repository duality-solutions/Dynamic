// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/identity.h"

#include "bdap/utils.h"
#include "hash.h"
#include "streams.h"

void CIdentity::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsIdentity(SER_NETWORK, PROTOCOL_VERSION);
    dsIdentity << *this;
    vchData = std::vector<unsigned char>(dsIdentity.begin(), dsIdentity.end());
}

bool CIdentity::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsIdentity(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsIdentity >> *this;

        std::vector<unsigned char> vchIdentityData;
        Serialize(vchIdentityData);
        const uint256 &calculatedHash = Hash(vchIdentityData.begin(), vchIdentityData.end());
        const std::vector<unsigned char> &vchRandIdentity = vchFromValue(calculatedHash.GetHex());
        if(vchRandIdentity != vchHash)
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

bool CIdentity::UnserializeFromTx(const CTransactionRef& tx) 
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

void CIdentityVerification::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsIdentityVerification(SER_NETWORK, PROTOCOL_VERSION);
    dsIdentityVerification << *this;
    vchData = std::vector<unsigned char>(dsIdentityVerification.begin(), dsIdentityVerification.end());
}

bool CIdentityVerification::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsIdentityVerification(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsIdentityVerification >> *this;

        std::vector<unsigned char> vchIdentityVerification;
        Serialize(vchIdentityVerification);
        const uint256 &calculatedHash = Hash(vchIdentityVerification.begin(), vchIdentityVerification.end());
        const std::vector<unsigned char> &vchRandIdentityVerification = vchFromValue(calculatedHash.GetHex());
        if(vchRandIdentityVerification != vchHash)
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

bool CIdentityVerification::UnserializeFromTx(const CTransactionRef& tx) 
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