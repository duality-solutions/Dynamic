
// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/entrychannel.h"

#include "hash.h"
#include "script/script.h"
#include "streams.h"

void CEntryChannel::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsEntryChannel(SER_NETWORK, PROTOCOL_VERSION);
    dsEntryChannel << *this;
    vchData = std::vector<unsigned char>(dsEntryChannel.begin(), dsEntryChannel.end());
}

bool CEntryChannel::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsEntryChannel(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsEntryChannel >> *this;

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

bool CEntryChannel::UnserializeFromTx(const CTransactionRef& tx) 
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

bool CEntryChannel::ValidateValues(std::string& errorMessage)
{
    // check channel owner path
    std::string strOwnerFullPath = stringFromVch(OwnerFullPath);
    if (strOwnerFullPath.length() > MAX_OBJECT_FULL_PATH_LENGTH) // object
    {
        errorMessage = "Invalid BDAP owner full path name. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }
    
    // check channel description
    std::string strDescription = stringFromVch(Description);
    if (strDescription.length() > MAX_DESCRIPTION_LENGTH)
    {
        errorMessage = "Invalid channel description. Can not have more than " + std::to_string(MAX_DESCRIPTION_LENGTH) + " characters.";
        return false;
    }

    // check channel resource pointer
    std::string strResourcePointer = stringFromVch(ResourcePointer);
    if (strResourcePointer.length() > MAX_RESOURCE_POINTER_LENGTH) // object
    {
        errorMessage = "Invalid BDAP resource pointer. Can not have more than " + std::to_string(MAX_RESOURCE_POINTER_LENGTH) + " characters.";
        return false;
    }

    return true;
}