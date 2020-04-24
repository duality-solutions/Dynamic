// Copyright (c) 2019-2020 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/auditdata.h"

#include "bdap/bdap.h"
#include "bdap/utils.h"
#include "hash.h"
#include "serialize.h"
#include "streams.h"

#include <univalue.h>

void CAuditData::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsAuditData(SER_NETWORK, PROTOCOL_VERSION);
    dsAuditData << *this;
    vchData = std::vector<unsigned char>(dsAuditData.begin(), dsAuditData.end());
}

bool CAuditData::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsAuditData(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsAuditData >> *this;

        std::vector<unsigned char> vchAuditData;
        Serialize(vchAuditData);
        const uint256 &calculatedHash = Hash(vchAuditData.begin(), vchAuditData.end());
        const std::vector<unsigned char> &vchRandAuditData = vchFromValue(calculatedHash.GetHex());
        if(vchRandAuditData != vchHash)
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

bool CAuditData::UnserializeFromTx(const CTransactionRef& tx) 
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

bool CAuditData::ValidateValues(std::string& strErrorMessage)
{
    if (OwnerFullObjectPath.size() > MAX_OBJECT_FULL_PATH_LENGTH) {
        strErrorMessage = "Invalid BDAP audit owner FQDN. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }
    for(const CharString& vchHash : vAuditData) {
        if (vchHash.size() > MAX_BDAP_AUDIT_HASH_SIZE) {
            strErrorMessage = "Invalid audit length. Can not have more than " + std::to_string(MAX_BDAP_AUDIT_HASH_SIZE) + " characters.";
            return false;
        }
    }
    return true;
}