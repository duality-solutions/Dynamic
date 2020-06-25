// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/auditdb.h"

#include "amount.h"
#include "base58.h"
#include "bdap/domainentrydb.h"
#include "bdap/fees.h"
#include "bdap/utils.h"
#include "coins.h"
#include "utilmoneystr.h"
#include "utiltime.h"
#include "validation.h"
#include "validationinterface.h"

#include <univalue.h>

#include <boost/thread.hpp>

CAuditDB *pAuditDB = nullptr;

bool GetAudit(const std::vector<unsigned char>& vchAudit, CAudit& audit)
{
    if (!pAuditDB || !pAuditDB->ReadAudit(vchAudit, audit))
        return false;

    return !audit.IsNull();
}

bool GetAudit(const std::string& strAudit, CAudit& audit)
{
    if (!pAuditDB || !pAuditDB->ReadAudit(vchFromString(strAudit), audit))
        return false;

    return !audit.IsNull();
}

bool GetAuditTxId(const std::string& strTxId, CAudit& audit)
{
    if (!pAuditDB || !pAuditDB->ReadAuditTxId(vchFromString(strTxId), audit))
        return false;

    return !audit.IsNull();
}

bool AuditExists(const std::vector<unsigned char>& vchAudit)
{
    if (!pAuditDB)
        return false;

    return pAuditDB->AuditExists(vchAudit);
}

bool UndoAddAudit(const CAudit& audit)
{
    if (!pAuditDB)
        return false;

    return pAuditDB->EraseAuditTxId(vchFromString(audit.GetHash().ToString()));
}

bool CAuditDB::AddAudit(const CAudit& audit, const int op) 
{ 
    bool writeState = false;
    bool auditHashWriteState = true;
    {
        LOCK(cs_bdap_audit);
        CAuditData auditData = audit.GetAuditData();
        for(const std::vector<unsigned char>& vchAuditHash : auditData.vAuditData) {
            // each hash points to a txid. The txid record stores the audit record.
            if (!Write(make_pair(std::string("audit"), vchAuditHash), vchFromString(audit.txHash.ToString()))) {
                auditHashWriteState = false;
            }
        }
        writeState = Write(make_pair(std::string("txid"), vchFromString(audit.txHash.ToString())), audit);
    }

    return writeState && auditHashWriteState;
}

bool CAuditDB::ReadAuditTxId(const std::vector<unsigned char>& vchTxId, CAudit& audit) 
{
    LOCK(cs_bdap_audit);
    return CDBWrapper::Read(make_pair(std::string("txid"), vchTxId), audit);
}

bool CAuditDB::ReadAudit(const std::vector<unsigned char>& vchAudit, CAudit& audit) 
{
    std::vector<unsigned char> vchTxId;
    LOCK(cs_bdap_audit);
    if (CDBWrapper::Read(make_pair(std::string("audit"), vchAudit), vchTxId)) {
        if (!ReadAuditTxId(vchTxId, audit))
            return false;
    } else {
        return false;
    }
    return true;
}

bool CAuditDB::AuditExists(const std::vector<unsigned char>& vchAudit)
{
    LOCK(cs_bdap_audit);
    return CDBWrapper::Exists(make_pair(std::string("audit"), vchAudit));
}

bool CAuditDB::EraseAuditTxId(const std::vector<unsigned char>& vchTxId)
{
    LOCK(cs_bdap_audit);
    CAudit audit;
    if (ReadAuditTxId(vchTxId, audit)) {
        for(const std::vector<unsigned char>& vchAudit : audit.GetAudits())
            CDBWrapper::Erase(make_pair(std::string("audit"), vchAudit));
    }
    return CDBWrapper::Erase(make_pair(std::string("txid"), vchTxId));
}

bool CAuditDB::EraseAudit(const std::vector<unsigned char>& vchAudit)
{
    LOCK(cs_bdap_audit);
    return CDBWrapper::Erase(make_pair(std::string("audit"), vchAudit));
}

// Removes expired records from databases.
bool CAuditDB::RemoveExpired(int& entriesRemoved, int& auditsRemoved)
{
    boost::scoped_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    CAudit audit;
    std::pair<std::string, std::vector<unsigned char> > key;
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        try {
            if (pcursor->GetKey(key) && key.first == "txid") {
                std::vector<unsigned char> value;
                CAudit audit;
                pcursor->GetValue(value);
                if (GetAudit(value, audit) && audit.nExpireTime > 0 && (unsigned int)chainActive.Tip()->GetMedianTimePast() >= audit.nExpireTime) {
                    for(const std::vector<unsigned char>& vchAuditHash : audit.GetAudits()) {
                        EraseAudit(vchAuditHash);
                        auditsRemoved++;
                    }
                    EraseAuditTxId(key.second);
                    entriesRemoved++;
                }
            }
            pcursor->Next();
        } catch (std::exception& e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

bool CAuditDB::GetAuditInfo(const std::vector<unsigned char>& vchAudit, UniValue& oAuditInfo)
{
    CAudit audit;
    if (!ReadAudit(vchAudit, audit))
        return false;

    if (!BuildAuditJson(audit, oAuditInfo))
        return false;  

    return true;
}

bool CAuditDB::GetAuditInfo(const std::vector<unsigned char>& vchAudit, CAudit& audit)
{
    if (!ReadAudit(vchAudit, audit))
        return false;

    return true;
}

bool CheckAuditDB()
{
    if (!pAuditDB)
        return false;

    return true;
}

bool FlushAuditLevelDB() 
{
    {
        LOCK(cs_bdap_audit);
        if (pAuditDB != nullptr)
        {
            if (!pAuditDB->Flush()) {
                LogPrintf("Failed to flush Audit BDAP database!");
                return false;
            }
        }
    }
    return true;
}

void RemoveExpired(int& entriesRemoved, int& auditsRemoved)
{
    if(pAuditDB != nullptr)
        pAuditDB->RemoveExpired(entriesRemoved, auditsRemoved);
    FlushAuditLevelDB();
}

static bool CommonDataCheck(const CAudit& audit, const vchCharString& vvchOpParameters, std::string& errorMessage)
{
    if (audit.IsNull()) {
        errorMessage = "CommonDataCheck failed! Audit is null.";
        return false;
    }

    if (!audit.ValidateValues(errorMessage)) {
        errorMessage = "CommonDataCheck failed! Invalid audit value. " + errorMessage;
        return false;
    }

    if (vvchOpParameters.size() > 3) {
        errorMessage = "CommonDataCheck failed! Too many parameters.";
        return false;
    }

    if (vvchOpParameters.size() < 1) {
        errorMessage = "CommonDataCheck failed! Not enough parameters.";
        return false;
    }
    // check count parameter is not longer than 
    if (vvchOpParameters[0].size() > 10) {
        errorMessage = "CommonDataCheck failed! Not enough parameters.";
        return false;
    }
    // check if count equals audit count.
    uint32_t nCount;
    ParseUInt32(stringFromVch(vvchOpParameters[0]), &nCount);
    if (nCount != audit.GetAudits().size()) {
        errorMessage = "CommonDataCheck failed! Parameter count does not match audits in data.";
        return false;
    }

    // check if signed. 
    if (vvchOpParameters.size() == 2) {
        errorMessage = "CommonDataCheck failed! Only two parameters. Signed audits require 3 parameters.";
        return false;
    }

    if (vvchOpParameters.size() > 1 && audit.vchOwnerFullObjectPath != vvchOpParameters[1]) {
        errorMessage = "CommonDataCheck failed! Script operation account parameter does not match account in audit object.";
        return false;
    }
    // check FQDN size
    if (vvchOpParameters.size() > 1 && vvchOpParameters[1].size() > MAX_OBJECT_FULL_PATH_LENGTH) {
        errorMessage = "CommonDataCheck failed! Count parameter value too large.";
        return false;
    }
    // check pubkey size
    if (vvchOpParameters.size() > 1 && vvchOpParameters[2].size() > MAX_KEY_LENGTH) {
        errorMessage = "CommonDataCheck failed! Count parameter value too large.";
        return false;
    }
    return true;
}

static bool CheckNewAuditTxInputs(const CAudit& audit, const CScript& scriptOp, const vchCharString& vvchOpParameters, const uint256& txHash,
                               std::string& errorMessage, bool fJustCheck)
{
    if (!CommonDataCheck(audit, vvchOpParameters, errorMessage))
        return error(errorMessage.c_str());

    if (fJustCheck)
        return true;

    // check audit signature if required.
    if (audit.SignatureRequired()) {
        if (!audit.IsSigned()) {
            errorMessage = "CheckNewAuditTxInputs: - The audit requires a signature.  Add new audit failed!";
            return error(errorMessage.c_str());
        }
        CDomainEntry entry;
        if (!GetDomainEntry(audit.vchOwnerFullObjectPath, entry)) {
            errorMessage = "CheckNewAuditTxInputs: - Could not find specified audit account owner! " + stringFromVch(audit.vchOwnerFullObjectPath);
            return error(errorMessage.c_str());
        }
        CDynamicAddress address = entry.GetWalletAddress();
        CKeyID keyID;
        if (!address.GetKeyID(keyID)) {
            errorMessage = "CheckNewAuditTxInputs: - Could not get key id. " + address.ToString();
            return error(errorMessage.c_str());
        }
        // check signature and pubkey belongs to bdap account.
    }

    CAudit getAudit;
    if (GetAudit(vchFromString(audit.txHash.ToString()), getAudit)) {
        if (audit.txHash != txHash) {
            errorMessage = "CheckNewAuditTxInputs: - The audit " + audit.txHash.ToString() + " already exists.  Add new audit failed!";
            return error(errorMessage.c_str());
        } else {
            LogPrintf("%s -- Already have audit %s in local database. Skipping add audit step.\n", __func__, audit.txHash.ToString());
            return true;
        }
    }

    if (!pAuditDB) {
        errorMessage = "CheckNewAuditTxInputs failed! Can not open LevelDB BDAP audit database.";
        return error(errorMessage.c_str());
    }
    int op = OP_BDAP_NEW;
    if (!pAuditDB->AddAudit(audit, op)) {
        errorMessage = "CheckNewAuditTxInputs failed! Error adding new audit record to LevelDB.";
        return error(errorMessage.c_str());
    }

    return FlushAuditLevelDB();
}

bool CheckAuditTx(const CTransactionRef& tx, const CScript& scriptOp, const int& op1, const int& op2, const std::vector<std::vector<unsigned char> >& vvchArgs, 
                                const bool fJustCheck, const int& nHeight, const uint32_t& nBlockTime, const bool bSanityCheck, std::string& errorMessage) 
{
    if (tx->IsCoinBase() && !fJustCheck && !bSanityCheck) {
        LogPrintf("*Trying to add BDAP audit in coinbase transaction, skipping...");
        return true;
    }

    LogPrint("bdap", "%s -- BDAP nHeight=%d, chainActive.Tip()=%d, op1=%s, op2=%s, hash=%s justcheck=%s\n", __func__, nHeight, chainActive.Tip()->nHeight, BDAPFromOp(op1).c_str(), BDAPFromOp(op2).c_str(), tx->GetHash().ToString().c_str(), fJustCheck ? "JUSTCHECK" : "BLOCK");

    // unserialize BDAP from txn, check if the audit is valid and does not conflict with a previous audit
    CAudit audit;
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nDataOut;
    bool bData = GetBDAPData(tx, vchData, vchHash, nDataOut);
    if(bData && !audit.UnserializeFromData(vchData, vchHash))
    {
        errorMessage = ("UnserializeFromData data in tx failed!");
        LogPrintf("%s -- %s \n", __func__, errorMessage);
        return error(errorMessage.c_str());
    }
    const std::string strOperationType = GetBDAPOpTypeString(op1, op2);
    CAmount monthlyFee, oneTimeFee, depositFee;
    if (strOperationType == "bdap_new_audit") {
        if (vvchArgs.size() > 3) {
            errorMessage = "Failed to get fees to add a new BDAP account";
            return false;
        }
        std::string strCount = stringFromVch(vvchArgs[0]);
        uint32_t nCount;
        ParseUInt32(stringFromVch(vvchArgs[0]), &nCount);
        if (!GetBDAPFees(OP_BDAP_NEW, OP_BDAP_AUDIT, BDAP::ObjectType::BDAP_AUDIT, (uint16_t)nCount, monthlyFee, oneTimeFee, depositFee)) {
            errorMessage = "Failed to get fees to add a new BDAP account";
            return false;
        }
        LogPrint("bdap", "%s -- nCount %d, oneTimeFee %d\n", __func__, nCount, FormatMoney(oneTimeFee));
        // extract amounts from tx.
        CAmount dataAmount, opAmount;
        if (!ExtractAmountsFromTx(tx, dataAmount, opAmount)) {
            errorMessage = "Unable to extract BDAP amounts from transaction";
            return false;
        }
        LogPrint("bdap", "%s -- dataAmount %d, opAmount %d\n", __func__, FormatMoney(dataAmount), FormatMoney(opAmount));
        if (monthlyFee > dataAmount) {
            LogPrintf("%s -- Invalid BDAP monthly registration fee amount for new BDAP account. Monthly paid %d but should be %d\n", __func__, 
                                    FormatMoney(dataAmount), FormatMoney(monthlyFee));
            errorMessage = "Invalid BDAP monthly registration fee amount for new BDAP account";
            return false;
        }
        else {
            LogPrint("bdap", "%s -- Valid BDAP monthly registration fee amount for new BDAP account. Monthly paid %d, should be %d.\n", __func__, 
                                    FormatMoney(dataAmount), FormatMoney(monthlyFee));
        }
        if (depositFee > opAmount) {
            LogPrintf("%s -- Invalid BDAP deposit fee amount for new BDAP account. Deposit paid %d but should be %d\n", __func__, 
                                    FormatMoney(opAmount), FormatMoney(depositFee));
            errorMessage = "Invalid BDAP deposit fee amount for new BDAP account";
            return false;
        }
        else {
            LogPrint("bdap", "%s -- Valid BDAP deposit fee amount for new BDAP account. Deposit paid %d, should be %d\n", __func__, 
                                    FormatMoney(opAmount), FormatMoney(depositFee));
        }

        return CheckNewAuditTxInputs(audit, scriptOp, vvchArgs, tx->GetHash(), errorMessage, fJustCheck);
    }

    return false;
}
