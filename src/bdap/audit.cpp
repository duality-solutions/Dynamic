// Copyright (c) 2019-2021 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/audit.h"

#include "bdap/bdap.h"
#include "bdap/utils.h"
#include "key.h"
#include "hash.h"
#include "messagesigner.h"
#include "pubkey.h"
#include "serialize.h"
#include "streams.h"
#include "validation.h"

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
        const uint256& calculatedHash = Hash(vchAuditData.begin(), vchAuditData.end());
        const std::vector<unsigned char>& vchRandAuditData = vchFromValue(calculatedHash.GetHex());
        if(vchRandAuditData != vchHash) {
            SetNull();
            return false;
        }
    } catch (std::exception& e) {
        SetNull();
        return false;
    }
    return true;
}

bool CAuditData::UnserializeFromData(const std::vector<unsigned char>& vchData) 
{
    try {
        CDataStream dsAuditData(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsAuditData >> *this;

        std::vector<unsigned char> vchAuditData;
        Serialize(vchAuditData);
    } catch (std::exception& e) {
        SetNull();
        return false;
    }
    return true;
}

bool CAuditData::ValidateValues(std::string& strErrorMessage)
{
    for(const CharString& vchHash : vAuditData) {
        if (vchHash.size() > MAX_BDAP_AUDIT_HASH_SIZE) {
            strErrorMessage = "Invalid audit length. Can not have more than " + std::to_string(MAX_BDAP_AUDIT_HASH_SIZE) + " characters.";
            return false;
        }
    }
    return true;
}

CAudit::CAudit(CAuditData& auditData) {
    auditData.Serialize(vchAuditData);
}

uint256 CAudit::GetHash() const
{
    CDataStream dsAudit(SER_NETWORK, PROTOCOL_VERSION);
    dsAudit << *this;
    return Hash(dsAudit.begin(), dsAudit.end());
}

uint256 CAudit::GetAuditHash() const
{
    CDataStream dsAudit(SER_NETWORK, PROTOCOL_VERSION);
    dsAudit << vchAuditData << vchOwnerFullObjectPath << vchAlgorithmType << vchDescription;
    return Hash(dsAudit.begin(), dsAudit.end());
}

bool CAudit::Sign(const CKey& key)
{
    if (!CHashSigner::SignHash(GetAuditHash(), key, vchSignature)) {
        return error("CAudit::%s() -- Failed to sign audit data.\n", __func__);
    }

    return true;
}

bool CAudit::CheckSignature(const std::vector<unsigned char>& vchPubKey) const
{
    std::string strError = "";
    CPubKey pubkey(vchPubKey);
    if (!CHashSigner::VerifyHash(GetAuditHash(), pubkey, vchSignature, strError)) {
        return error("CAudit::%s() -- Failed to verify signature - %s\n", __func__, strError);
    }

    return true;
}

int CAudit::Version() const
{
    if (vchAuditData.size() == 0)
        return -1;

    return CAuditData(vchAuditData).nVersion;
}

bool CAudit::ValidateValues(std::string& strErrorMessage) const
{
    CAuditData auditData(vchAuditData);
    if (!auditData.ValidateValues(strErrorMessage))
        return false;

    if (auditData.vAuditData.size() == 0)
        return false;

    //Check for duplicates
    if (auditData.vAuditData.size() >> 1) {
        auto it = std::unique(auditData.vAuditData.begin(), auditData.vAuditData.end());
        if (!(it == auditData.vAuditData.end())) {
            strErrorMessage = "Invalid Audit data. Can not have duplicates.";
            return false;
        }
    }

    if (vchOwnerFullObjectPath.size() > MAX_OBJECT_FULL_PATH_LENGTH) {
        strErrorMessage = "Invalid BDAP audit owner FQDN length. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }

    if (vchSignature.size() > MAX_SIGNATURE_LENGTH) {
        strErrorMessage = "Invalid BDAP audit signature length. Can not have more than " + std::to_string(MAX_SIGNATURE_LENGTH) + " characters.";
        return false;
    }

    if (vchAlgorithmType.size() > MAX_ALGORITHM_TYPE_LENGTH) {
        strErrorMessage = "Invalid Algorithm Type length. Can not have more than " + std::to_string(MAX_ALGORITHM_TYPE_LENGTH) + " characters.";
        return false;
    }

    if (vchDescription.size() > MAX_DATA_DESCRIPTION_LENGTH) {
        strErrorMessage = "Invalid Data Description length. Can not have more than " + std::to_string(MAX_DATA_DESCRIPTION_LENGTH) + " characters.";
        return false;
    }

    return CAuditData(vchAuditData).ValidateValues(strErrorMessage);
}

void CAudit::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsAudit(SER_NETWORK, PROTOCOL_VERSION);
    dsAudit << *this;
    vchData = std::vector<unsigned char>(dsAudit.begin(), dsAudit.end());
}

bool CAudit::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsAudit(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsAudit >> *this;

        std::vector<unsigned char> vchAudit;
        Serialize(vchAudit);
        const uint256& calculatedHash = Hash(vchAudit.begin(), vchAudit.end());
        const std::vector<unsigned char>& vchRandAudit = vchFromValue(calculatedHash.GetHex());
        if(vchRandAudit != vchHash) {
            SetNull();
            return false;
        }
    } catch (std::exception& e) {
        SetNull();
        return false;
    }
    return true;
}

bool CAudit::UnserializeFromTx(const CTransactionRef& tx, const unsigned int& height) 
{
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nOut;
    if(!GetBDAPData(tx, vchData, vchHash, nOut)) {
        SetNull();
        return false;
    }
    if(!UnserializeFromData(vchData, vchHash)) {
        return false;
    }
    txHash = tx->GetHash();
    nHeight = height;
    return true;
}

std::string CAudit::ToString() const
{
    CAuditData auditData = GetAuditData();
    std::string strAuditData;
    for(const std::vector<unsigned char>& vchAudit : auditData.vAuditData)
        strAuditData += "\n                           " + stringFromVch(vchAudit);

    return strprintf(
        "CAudit(\n"
        "    nVersion             = %d\n"
        "    Audit Count          = %d\n"
        "    Audit Data           = %s\n"
        "    Algorithm Type       = %s\n"
        "    Description          = %s\n"
        "    nTimeStamp           = %d\n"
        "    Owner                = %s\n"
        "    Signed               = %s\n"
        ")\n",
        auditData.nVersion,
        auditData.vAuditData.size(),
        strAuditData,
        stringFromVch(vchAlgorithmType),
        stringFromVch(vchDescription),
        auditData.nTimeStamp,
        stringFromVch(vchOwnerFullObjectPath),
        IsSigned() ? "True" : "False"
        );
}

bool BuildAuditJson(const CAudit& audit, UniValue& oAudit)
{
    int64_t nTime = 0;
    CAuditData auditData = audit.GetAuditData();

    UniValue oAuditHashes(UniValue::VOBJ);
    int counter = 0;
    for(const std::vector<unsigned char>& vchAudit : auditData.vAuditData) {
        counter++;
        oAuditHashes.push_back(Pair("audit_hash" + std::to_string(counter), stringFromVch(vchAudit)));
    }
    oAudit.push_back(Pair("version", std::to_string(audit.Version())));
    oAudit.push_back(Pair("audit_count", auditData.vAuditData.size()));
    oAudit.push_back(Pair("audit_hashes", oAuditHashes));
    oAudit.push_back(Pair("timestamp", std::to_string(auditData.nTimeStamp)));
    oAudit.push_back(Pair("owner", stringFromVch(audit.vchOwnerFullObjectPath)));
    oAudit.push_back(Pair("signed", audit.IsSigned() ? "True" : "False"));
    oAudit.push_back(Pair("algorithm_type", stringFromVch(audit.vchAlgorithmType)));
    oAudit.push_back(Pair("description", stringFromVch(audit.vchDescription)));
    oAudit.push_back(Pair("txid", audit.txHash.GetHex()));
    if ((unsigned int)chainActive.Height() >= audit.nHeight) {
        CBlockIndex *pindex = chainActive[audit.nHeight];
        if (pindex) {
            nTime = pindex->GetBlockTime();
        }
    }
    oAudit.push_back(Pair("block_time", nTime));
    oAudit.push_back(Pair("block_height", std::to_string(audit.nHeight)));
    return true;
}

bool BuildVerifyAuditJson(const CAudit& audit, UniValue& oAudit)
{
    int64_t nTime = 0;
    CAuditData auditData = audit.GetAuditData();

    oAudit.push_back(Pair("version", std::to_string(audit.Version())));
    oAudit.push_back(Pair("audit_count", auditData.vAuditData.size()));
    oAudit.push_back(Pair("timestamp", std::to_string(auditData.nTimeStamp)));
    oAudit.push_back(Pair("owner", stringFromVch(audit.vchOwnerFullObjectPath)));
    oAudit.push_back(Pair("signed", audit.IsSigned() ? "True" : "False"));
    oAudit.push_back(Pair("algorithm_type", stringFromVch(audit.vchAlgorithmType)));
    oAudit.push_back(Pair("description", stringFromVch(audit.vchDescription)));
    oAudit.push_back(Pair("txid", audit.txHash.GetHex()));
    if ((unsigned int)chainActive.Height() >= audit.nHeight) {
        CBlockIndex *pindex = chainActive[audit.nHeight];
        if (pindex) {
            nTime = pindex->GetBlockTime();
        }
    }
    oAudit.push_back(Pair("block_time", nTime));
    oAudit.push_back(Pair("block_height", std::to_string(audit.nHeight)));
    return true;
}

