// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/auditdata.h"

#include "bdap/utils.h"
#include "hash.h"
#include "streams.h"


namespace BDAP {
    std::string GetAuditTypeString(unsigned int nAuditType)
    {
        switch ((BDAP::AuditType)nAuditType) {

            case BDAP::AuditType::UNKNOWN:
                return "Unknown";
             case BDAP::AuditType::HASH_POINTER_AUDIT:
                return "User Entry";
            default:
                return "Unknown";
        }
    }
}

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