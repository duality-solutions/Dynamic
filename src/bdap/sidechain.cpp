
// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/sidechain.h"

#include "bdap/utils.h"
#include "hash.h"
#include "script/script.h"
#include "streams.h"

void CSideChain::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsSideChain(SER_NETWORK, PROTOCOL_VERSION);
    dsSideChain << *this;
    vchData = std::vector<unsigned char>(dsSideChain.begin(), dsSideChain.end());
}

bool CSideChain::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsSideChain(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsSideChain >> *this;

        std::vector<unsigned char> vchSideChainData;
        Serialize(vchSideChainData);
        const uint256 &calculatedHash = Hash(vchSideChainData.begin(), vchSideChainData.end());
        const std::vector<unsigned char> &vchRandSideChain = vchFromValue(calculatedHash.GetHex());
        if(vchRandSideChain != vchHash)
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

bool CSideChain::UnserializeFromTx(const CTransactionRef& tx) 
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

bool CSideChain::ValidateValues(std::string& errorMessage)
{
    // check sidechain owner path
    std::string strOwnerFullPath = stringFromVch(OwnerFullPath);
    if (strOwnerFullPath.length() > MAX_OBJECT_FULL_PATH_LENGTH) // object
    {
        errorMessage = "Invalid BDAP owner full path name. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }
    
    // check sidechain description
    std::string strDescription = stringFromVch(Description);
    if (strDescription.length() > MAX_DESCRIPTION_LENGTH)
    {
        errorMessage = "Invalid sidechain description. Can not have more than " + std::to_string(MAX_DESCRIPTION_LENGTH) + " characters.";
        return false;
    }

    // check sidechain resource pointer
    std::string strResourcePointer = stringFromVch(ResourcePointer);
    if (strResourcePointer.length() > MAX_RESOURCE_POINTER_LENGTH) // object
    {
        errorMessage = "Invalid BDAP resource pointer. Can not have more than " + std::to_string(MAX_RESOURCE_POINTER_LENGTH) + " characters.";
        return false;
    }

    return true;
}