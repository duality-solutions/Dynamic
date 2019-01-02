
// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/entrycheckpoints.h"

#include "bdap/utils.h"
#include "hash.h"
#include "script/script.h"
#include "streams.h"

void CEntryCheckpoints::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsEntryCheckpoints(SER_NETWORK, PROTOCOL_VERSION);
    dsEntryCheckpoints << *this;
    vchData = std::vector<unsigned char>(dsEntryCheckpoints.begin(), dsEntryCheckpoints.end());
}

bool CEntryCheckpoints::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsEntryCheckpoints(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsEntryCheckpoints >> *this;

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

bool CEntryCheckpoints::UnserializeFromTx(const CTransactionRef& tx) 
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

bool CEntryCheckpoints::ValidateValues(std::string& errorMessage)
{
    // check certificate owner path
    std::string strOwnerFullPath = stringFromVch(OwnerFullPath);
    if (strOwnerFullPath.length() > MAX_OBJECT_FULL_PATH_LENGTH) // object
    {
        errorMessage = "Invalid BDAP owner full path name. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }

    // check max checkpoints per transaction
    if (CheckPointHashes.size() > MAX_NUMBER_CHECKPOINTS)
    {
        errorMessage = "Invalid number of BDAP checkpoints. Can not have more than " + std::to_string(MAX_NUMBER_CHECKPOINTS) + " checkpoints per transaction.";
        return false;
    }

    // check length of checkpoint hashes
    for (unsigned int i = 0; i < CheckPointHashes.size(); i++) {
        std::string strCheckpointHash = stringFromVch(CheckPointHashes[i].second);
        if (strCheckpointHash.length() > MAX_CHECKPOINT_HASH_LENGTH) // object
        {
            errorMessage = "Invalid BDAP checkpoint hash length. Can not have more than " + std::to_string(MAX_CHECKPOINT_HASH_LENGTH) + " characters.";
            return false;
        }
    }

    return true;
}

void CEntryCheckpoints::AddCheckpoint(const uint32_t& height, const CharString& vchHash) 
{
    CheckPoint pairNewCheckpoint;
    pairNewCheckpoint.first = height;
    pairNewCheckpoint.second = vchHash;
    CheckPointHashes.push_back(pairNewCheckpoint);
}