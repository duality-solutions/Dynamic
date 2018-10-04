// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#include "dht/mutable.h"

#include "bdap/domainentry.h"
#include "hash.h"
#include "streams.h"

void CMutableData::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsMutableData(SER_NETWORK, PROTOCOL_VERSION);
    dsMutableData << *this;
    vchData = std::vector<unsigned char>(dsMutableData.begin(), dsMutableData.end());
}

bool CMutableData::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsMutableData(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsMutableData >> *this;

        std::vector<unsigned char> vchMutableData;
        Serialize(vchMutableData);
        const uint256 &calculatedHash = Hash(vchMutableData.begin(), vchMutableData.end());
        const std::vector<unsigned char> &vchRandMutableData = vchFromValue(calculatedHash.GetHex());
        if(vchRandMutableData != vchHash)
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

std::string CMutableData::InfoHash() const
{
    return stringFromVch(vchInfoHash);
}

std::string CMutableData::PublicKey() const
{
    return stringFromVch(vchPublicKey);
}

std::string CMutableData::Signature() const
{
    return stringFromVch(vchSignature);
}

std::string CMutableData::Salt() const
{
    return stringFromVch(vchSalt);
}

std::string CMutableData::Value() const
{
    return stringFromVch(vchValue);
}