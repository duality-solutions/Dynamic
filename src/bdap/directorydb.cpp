// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/directorydb.h"

#include "base58.h"
#include "validation.h"
#include "validationinterface.h"

#include <univalue.h>

CDirectoryDB *pDirectoryDB = NULL;

bool GetDirectory(const std::vector<unsigned char>& vchObjectPath, CDirectory& directory)
{
    if (!pDirectoryDB || !pDirectoryDB->ReadDirectory(vchObjectPath, directory)) {
        return false;
    }
    
    //TODO: (bdap) calculate directory.nExpireTime
    /*
    if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= directory.nExpireTime) {
        directory.SetNull();
        return false;
    }
    */
    return !directory.IsNull();
}

bool CDirectoryDB::AddDirectory(const CDirectory& directory, const int op) 
{ 
    bool writeState = false;
    {
        LOCK(cs_bdap_directory);
        writeState = Write(make_pair(std::string("domain_component"), directory.GetFullObjectPath()), directory) 
                         && Write(make_pair(std::string("domain_wallet_address"), directory.WalletAddress), directory.GetFullObjectPath());
    }
    if (writeState)
        AddDirectoryIndex(directory, op);

    return writeState;
}

void CDirectoryDB::AddDirectoryIndex(const CDirectory& directory, const int op) 
{
    UniValue oName(UniValue::VOBJ);
    if (BuildBDAPJson(directory, oName)) {
        GetMainSignals().NotifyBDAPUpdate(oName.write().c_str(), "bdap_new");
        WriteDirectoryIndexHistory(directory, op);
    }
}

bool CDirectoryDB::ReadDirectory(const std::vector<unsigned char>& vchObjectPath, CDirectory& directory) 
{
    LOCK(cs_bdap_directory);
    return CDBWrapper::Read(make_pair(std::string("domain_component"), vchObjectPath), directory);
}

bool CDirectoryDB::ReadDirectoryAddress(const std::vector<unsigned char>& vchAddress, std::vector<unsigned char>& vchObjectPath) 
{
    LOCK(cs_bdap_directory);
    return CDBWrapper::Read(make_pair(std::string("domain_wallet_address"), vchAddress), vchObjectPath);
}

bool CDirectoryDB::EraseDirectory(const std::vector<unsigned char>& vchObjectPath) 
{
    LOCK(cs_bdap_directory);
    return CDBWrapper::Erase(make_pair(std::string("domain_component"), vchObjectPath));
}

bool CDirectoryDB::EraseDirectoryAddress(const std::vector<unsigned char>& vchAddress) 
{
    LOCK(cs_bdap_directory);
    return CDBWrapper::Erase(make_pair(std::string("domain_wallet_address"), vchAddress));
}

bool CDirectoryDB::DirectoryExists(const std::vector<unsigned char>& vchObjectPath)
{
    LOCK(cs_bdap_directory);
    return CDBWrapper::Exists(make_pair(std::string("domain_component"), vchObjectPath));
}

bool CDirectoryDB::DirectoryExistsAddress(const std::vector<unsigned char>& vchAddress) 
{
    LOCK(cs_bdap_directory);
    return CDBWrapper::Exists(make_pair(std::string("domain_wallet_address"), vchAddress));
}

bool CDirectoryDB::RemoveExpired(int& entriesRemoved)
{
    boost::scoped_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    CDirectory directory;
    std::pair<std::string, std::vector<unsigned char> > key;
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        try {
            if (pcursor->GetKey(key) && key.first == "domain_component") {
                pcursor->GetValue(directory);
                  if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= directory.nExpireTime)
                {
                    entriesRemoved++;
                    EraseDirectory(key.second);
                } 
                
            }
            else if (pcursor->GetKey(key) && key.first == "domain_wallet_address") {
                std::vector<unsigned char> value;
                CDirectory directory;
                pcursor->GetValue(value);
                if (GetDirectory(value, directory) && (unsigned int)chainActive.Tip()->GetMedianTimePast() >= directory.nExpireTime)
                {
                    entriesRemoved++;
                    EraseDirectoryAddress(directory.WalletAddress);
                }

            }
            pcursor->Next();
        } catch (std::exception &e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

void CDirectoryDB::WriteDirectoryIndexHistory(const CDirectory& directory, const int op) 
{
    if (IsArgSet("-zmqpubbdaphistory")) {
        UniValue oName(UniValue::VOBJ);
        BuildBDAPJson(directory, oName);
        oName.push_back(Pair("op", directoryFromOp(op)));
        GetMainSignals().NotifyBDAPUpdate(oName.write().c_str(), "bdap_history");
    }
}

void CDirectoryDB::WriteDirectoryIndex(const CDirectory& directory, const int op) 
{
    if (IsArgSet("-zmqpubbdaprecord")) {
        UniValue oName(UniValue::VOBJ);
        oName.push_back(Pair("_id", stringFromVch(directory.OID)));
        CDynamicAddress address(EncodeBase58(directory.WalletAddress));
        oName.push_back(Pair("address", address.ToString()));
        oName.push_back(Pair("expires_on", directory.nExpireTime));
        oName.push_back(Pair("encryption_publickey", HexStr(directory.EncryptPublicKey)));
        GetMainSignals().NotifyBDAPUpdate(oName.write().c_str(), "bdap_record");
    }
    WriteDirectoryIndexHistory(directory, op);
}

bool CDirectoryDB::UpdateDirectory(const std::vector<unsigned char>& vchObjectPath, CDirectory& directory)
{
    return false; //TODO (bdap): add update impl
}

bool CDirectoryDB::UpdateDirectoryAddress(const std::vector<unsigned char>& vchAddress, CDirectory& directory)
{
    return false; //TODO (bdap): add update impl
}

// Removes expired records from databases.
bool CDirectoryDB::CleanupLevelDB(int& nRemoved)
{
    boost::scoped_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    CDirectory dirEntry;
    std::pair<std::string, std::vector<unsigned char> > key;
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        try {
            if (pcursor->GetKey(key) && key.first == "domain_component") 
            {
                pcursor->GetValue(dirEntry);
                if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= dirEntry.nExpireTime)
                {
                    nRemoved++;
                    EraseDirectory(key.second);
                }
            }
            else if (pcursor->GetKey(key) && key.first == "domain_wallet_address") 
            {
                std::vector<unsigned char> value;
                CDirectory directory;
                pcursor->GetValue(value);
                if (GetDirectory(value, directory) && (unsigned int)chainActive.Tip()->GetMedianTimePast() >= directory.nExpireTime)
                {
                    nRemoved++;
                    EraseDirectoryAddress(directory.vchFullObjectPath());
                }
            }
            pcursor->Next();
        } catch (std::exception& e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

bool CheckDirectoryDB()
{
    if (!pDirectoryDB)
        return false;

    return true;
}

bool FlushLevelDB() 
{
    {
        LOCK(cs_bdap_directory);
        if (pDirectoryDB != NULL)
        {
            if (!pDirectoryDB->Flush()) {
                LogPrintf("Failed to write to BDAP database!");
                return false;
            }
        }
    }
    return true;
}

void CleanupLevelDB(int& nRemoved)
{
    if(pDirectoryDB != NULL)
        pDirectoryDB->CleanupLevelDB(nRemoved);
    FlushLevelDB();
}


static bool CommonDataCheck(const CDirectory& directory, const vchCharString& vvchOpParameters, std::string& errorMessage)
{
    if (directory.IsNull() == true)
    {
        errorMessage = "CommonDataCheck failed! Directory is null.";
        return false;
    }

    if (vvchOpParameters.size() == 0)
    {
        errorMessage = "CommonDataCheck failed! Invalid parameters.";
        return false;
    }

    if (directory.GetFullObjectPath() != stringFromVch(vvchOpParameters[0]))
    {
        errorMessage = "CommonDataCheck failed! Script operation parameter does not match directory entry object.";
        return false;
    }
    
    if (directory.DomainComponent != vchDefaultDomainName)
    {
        errorMessage = "CommonDataCheck failed! Must use default domain.";
        return false;
    }

    if (directory.OrganizationalUnit != vchDefaultPublicOU && directory.OrganizationalUnit != vchDefaultAdminOU)
    {
        errorMessage = "CommonDataCheck failed! Must use default public organizational unit.";
        return false;
    }

    if (directory.OrganizationalUnit == vchDefaultAdminOU)
    {
        errorMessage = "CommonDataCheck failed! Can not use default admin domain.";
        return false;
    }

    return true;
}

bool CheckNewDirectoryTxInputs(const CTransaction& tx, const CDirectory& directory, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                               const int op, std::string& errorMessage, bool fJustCheck)
{
    if (!CommonDataCheck(directory, vvchOpParameters, errorMessage))
        return false;

    if (fJustCheck)
        return true;

    CDirectory getDirectory;
    if (GetDirectory(directory.vchFullObjectPath(), getDirectory))
    {
        errorMessage = "CheckNewDirectoryTxInputs failed! " + getDirectory.GetFullObjectPath() + " already exists.";
        return false;
    }

    if (!pDirectoryDB)
    {
        errorMessage = "CheckNewDirectoryTxInputs failed! Can not open LevelDB BDAP entry database.";
        return false;
    }

    if (!pDirectoryDB->AddDirectory(directory, op))
    {
        errorMessage = "CheckNewDirectoryTxInputs failed! Error adding new directory entry request to LevelDB.";
        return false;
    }

    return FlushLevelDB();
}

bool CheckDeleteDirectoryTxInputs(const CTransaction& tx, const CDirectory& directory, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                  const int op, std::string& errorMessage, bool fJustCheck)
{
    //check name in operation matches directory data in leveldb
    //check if exists already
    //if exists, check for owner's signature
    return false;
}

bool CheckActivateDirectoryTxInputs(const CTransaction& tx, const CDirectory& directory, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                    const int op, std::string& errorMessage, bool fJustCheck)
{
    //check name in operation matches directory data in leveldb as a new request
    //check if new request exists and is not expired
    return false;
}

bool CheckUpdateDirectoryTxInputs(const CTransaction& tx, const CDirectory& directory, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                  const int op, std::string& errorMessage, bool fJustCheck)
{
    //check name in operation matches directory data in leveldb
    //check if exists already
    //if exists, check for owner's signature
    return false;
}

bool CheckMoveDirectoryTxInputs(const CTransaction& tx, const CDirectory& directory, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                const int op, std::string& errorMessage, bool fJustCheck)
{
    //check name in operation matches directory data in leveldb
    //check if exists already
    //if exists, check for owner's signature
    return false;
}

bool CheckExecuteDirectoryTxInputs(const CTransaction& tx, const CDirectory& directory, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                   const int op, std::string& errorMessage, bool fJustCheck)
{
    //check name in operation matches directory data in leveldb
    //check if exists already
    //if exists, check for owner's signature
    return false;
}

bool CheckBindDirectoryTxInputs(const CTransaction& tx, const CDirectory& directory, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                const int op, std::string& errorMessage, bool fJustCheck)
{
    //check names in operation matches directory data in leveldb
    //check if request or accept response
    //check if names exists already
    //if exists, check for owner's signature
    return false;
}

bool CheckRevokeDirectoryTxInputs(const CTransaction& tx, const CDirectory& directory, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                  const int op, std::string& errorMessage, bool fJustCheck)
{
    //check name in operation matches directory data in leveldb
    //check if names exists already
    //if exists, check for fluid signature
    return false;
}

bool CheckDirectoryTxInputs(const CCoinsViewCache& inputs, const CTransaction& tx, 
                            int op, const std::vector<std::vector<unsigned char> >& vvchArgs, bool fJustCheck, int nHeight, std::string& errorMessage, bool bSanityCheck) 
{
    if (tx.IsCoinBase() && !fJustCheck && !bSanityCheck)
    {
        LogPrintf("*Trying to add BDAP entry in coinbase transaction, skipping...");
        return true;
    }

    //if (fDebug && !bSanityCheck)
        LogPrintf("*** BDAP nHeight=%d, chainActive.Tip()=%d, op=%s, hash=%s justcheck=%s\n", nHeight, chainActive.Tip()->nHeight, directoryFromOp(op).c_str(), tx.GetHash().ToString().c_str(), fJustCheck ? "JUSTCHECK" : "BLOCK");
    
    CScript scriptOp;
    vchCharString vvchOpParameters;
    if (!GetDirectoryOpScript(tx, scriptOp, vvchOpParameters, op))
    {
        errorMessage = "BDAP_CONSENSUS_ERROR: ERRCODE: 3600 - " + _("Transaction does not contain BDAP operation script!");
        return error(errorMessage.c_str());
    }
    const std::string strOperationType = GetDirectoryOpTypeString(scriptOp);
    if (fDebug)
        LogPrintf("CheckDirectoryTxInputs, strOperationType= %s \n", strOperationType);
    
    // unserialize BDAP from txn, check if the entry is valid and does not conflict with a previous entry
    CDirectory directory;
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nDataOut;
    
    bool bData = GetDirectoryData(tx, vchData, vchHash, nDataOut);
    if(bData && !directory.UnserializeFromData(vchData, vchHash))
    {
        errorMessage = "BDAP_CONSENSUS_ERROR: ERRCODE: 3601 - " + _("UnserializeFromData data in tx failed!");
        return error(errorMessage.c_str());
    }

    if(!directory.ValidateValues(errorMessage))
    {
        errorMessage = "BDAP_CONSENSUS_ERROR: ERRCODE: 3602 - " + errorMessage;
        return error(errorMessage.c_str());
    }

    if (strOperationType == "bdap_new")
        return CheckNewDirectoryTxInputs(tx, directory, scriptOp, vvchOpParameters, op, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_delete")
        return CheckDeleteDirectoryTxInputs(tx, directory, scriptOp, vvchOpParameters, op, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_activate")
        return CheckActivateDirectoryTxInputs(tx, directory, scriptOp, vvchOpParameters, op, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_update")
        return CheckUpdateDirectoryTxInputs(tx, directory, scriptOp, vvchOpParameters, op, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_move")
        return CheckMoveDirectoryTxInputs(tx, directory, scriptOp, vvchOpParameters, op, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_execute")
        return CheckExecuteDirectoryTxInputs(tx, directory, scriptOp, vvchOpParameters, op, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_bind")
        return CheckBindDirectoryTxInputs(tx, directory, scriptOp, vvchOpParameters, op, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_revoke")
        return CheckRevokeDirectoryTxInputs(tx, directory, scriptOp, vvchOpParameters, op, errorMessage, fJustCheck);

    return false;
}