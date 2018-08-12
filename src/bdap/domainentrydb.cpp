// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/domainentrydb.h"

#include "base58.h"
#include "validation.h"
#include "validationinterface.h"

#include <univalue.h>

CDomainEntryDB *pDomainEntryDB = NULL;

bool GetDomainEntry(const std::vector<unsigned char>& vchObjectPath, CDomainEntry& entry)
{
    if (!pDomainEntryDB || !pDomainEntryDB->ReadDomainEntry(vchObjectPath, entry)) {
        return false;
    }
    
    //TODO: (bdap) calculate entry.nExpireTime
    /*
    if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= entry.nExpireTime) {
        entry.SetNull();
        return false;
    }
    */
    return !entry.IsNull();
}

bool CDomainEntryDB::AddDomainEntry(const CDomainEntry& entry, const int op) 
{ 
    bool writeState = false;
    {
        LOCK(cs_bdap_entry);
        writeState = Write(make_pair(std::string("domain_component"), entry.GetFullObjectPath()), entry) 
                         && Write(make_pair(std::string("domain_wallet_address"), entry.WalletAddress), entry.GetFullObjectPath());
    }
    if (writeState)
        AddDomainEntryIndex(entry, op);

    return writeState;
}

void CDomainEntryDB::AddDomainEntryIndex(const CDomainEntry& entry, const int op) 
{
    UniValue oName(UniValue::VOBJ);
    if (BuildBDAPJson(entry, oName)) {
        CharString vchOperationType = vchFromString(DomainEntryFromOp(op));
        GetMainSignals().NotifyBDAPUpdate(oName.write().c_str(), reinterpret_cast<char*>(vchOperationType.data()));
        WriteDomainEntryIndexHistory(entry, op);
    }
}

bool CDomainEntryDB::ReadDomainEntry(const std::vector<unsigned char>& vchObjectPath, CDomainEntry& entry) 
{
    LOCK(cs_bdap_entry);
    return CDBWrapper::Read(make_pair(std::string("domain_component"), vchObjectPath), entry);
}

bool CDomainEntryDB::ReadDomainEntryAddress(const std::vector<unsigned char>& vchAddress, std::vector<unsigned char>& vchObjectPath) 
{
    LOCK(cs_bdap_entry);
    return CDBWrapper::Read(make_pair(std::string("domain_wallet_address"), vchAddress), vchObjectPath);
}

bool CDomainEntryDB::EraseDomainEntry(const std::vector<unsigned char>& vchObjectPath) 
{
    LOCK(cs_bdap_entry);
    return CDBWrapper::Erase(make_pair(std::string("domain_component"), vchObjectPath));
}

bool CDomainEntryDB::EraseDomainEntryAddress(const std::vector<unsigned char>& vchAddress) 
{
    LOCK(cs_bdap_entry);
    return CDBWrapper::Erase(make_pair(std::string("domain_wallet_address"), vchAddress));
}

bool CDomainEntryDB::DomainEntryExists(const std::vector<unsigned char>& vchObjectPath)
{
    LOCK(cs_bdap_entry);
    return CDBWrapper::Exists(make_pair(std::string("domain_component"), vchObjectPath));
}

bool CDomainEntryDB::DomainEntryExistsAddress(const std::vector<unsigned char>& vchAddress) 
{
    LOCK(cs_bdap_entry);
    return CDBWrapper::Exists(make_pair(std::string("domain_wallet_address"), vchAddress));
}

bool CDomainEntryDB::RemoveExpired(int& entriesRemoved)
{
    boost::scoped_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    CDomainEntry entry;
    std::pair<std::string, std::vector<unsigned char> > key;
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        try {
            if (pcursor->GetKey(key) && key.first == "domain_component") {
                pcursor->GetValue(entry);
                  if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= entry.nExpireTime)
                {
                    entriesRemoved++;
                    EraseDomainEntry(key.second);
                } 
                
            }
            else if (pcursor->GetKey(key) && key.first == "domain_wallet_address") {
                std::vector<unsigned char> value;
                CDomainEntry entry;
                pcursor->GetValue(value);
                if (GetDomainEntry(value, entry) && (unsigned int)chainActive.Tip()->GetMedianTimePast() >= entry.nExpireTime)
                {
                    entriesRemoved++;
                    EraseDomainEntryAddress(entry.WalletAddress);
                }

            }
            pcursor->Next();
        } catch (std::exception &e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

void CDomainEntryDB::WriteDomainEntryIndexHistory(const CDomainEntry& entry, const int op) 
{
    if (IsArgSet("-zmqpubbdaphistory")) {
        UniValue oName(UniValue::VOBJ);
        BuildBDAPJson(entry, oName);
        oName.push_back(Pair("op", DomainEntryFromOp(op)));
        GetMainSignals().NotifyBDAPUpdate(oName.write().c_str(), "bdap_history");
    }
}

void CDomainEntryDB::WriteDomainEntryIndex(const CDomainEntry& entry, const int op) 
{
    if (IsArgSet("-zmqpubbdaprecord")) {
        UniValue oName(UniValue::VOBJ);
        oName.push_back(Pair("_id", stringFromVch(entry.OID)));
        CDynamicAddress address(EncodeBase58(entry.WalletAddress));
        oName.push_back(Pair("address", address.ToString()));
        oName.push_back(Pair("expires_on", entry.nExpireTime));
        oName.push_back(Pair("encryption_publickey", HexStr(entry.EncryptPublicKey)));
        GetMainSignals().NotifyBDAPUpdate(oName.write().c_str(), "bdap_record");
    }
    WriteDomainEntryIndexHistory(entry, op);
}

bool CDomainEntryDB::UpdateDomainEntry(const std::vector<unsigned char>& vchObjectPath, const CDomainEntry& entry)
{
    LOCK(cs_bdap_entry);

    if (!EraseDomainEntryAddress(entry.WalletAddress))
        return false;

    bool writeState = false;
    writeState = Update(make_pair(std::string("domain_component"), entry.GetFullObjectPath()), entry) 
                    && Write(make_pair(std::string("domain_wallet_address"), entry.WalletAddress), entry.GetFullObjectPath());
    if (writeState)
        AddDomainEntryIndex(entry, OP_BDAP_MODIFY);

    return writeState;
}

// Removes expired records from databases.
bool CDomainEntryDB::CleanupLevelDB(int& nRemoved)
{
    boost::scoped_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    CDomainEntry dirEntry;
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
                    EraseDomainEntry(key.second);
                }
            }
            else if (pcursor->GetKey(key) && key.first == "domain_wallet_address") 
            {
                std::vector<unsigned char> value;
                CDomainEntry entry;
                pcursor->GetValue(value);
                if (GetDomainEntry(value, entry) && (unsigned int)chainActive.Tip()->GetMedianTimePast() >= entry.nExpireTime)
                {
                    nRemoved++;
                    EraseDomainEntryAddress(entry.vchFullObjectPath());
                }
            }
            pcursor->Next();
        } catch (std::exception& e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

// Lists active entries by domain name with paging support
bool CDomainEntryDB::ListDirectories(const std::vector<unsigned char>& vchObjectLocation, const unsigned int nResultsPerPage, const unsigned int nPage, UniValue& oDomainEntryList)
{
    // TODO: (bdap) implement paging
    // if vchObjectLocation is empty, list entries from all domains
    int index = 0;
    std::pair<std::string, CharString> key;
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        CDomainEntry entry;
        try {
            if (pcursor->GetKey(key) && key.first == "domain_component") {
                pcursor->GetValue(entry);
                if (vchObjectLocation.empty() || entry.vchObjectLocation() == vchObjectLocation)
                {
                    UniValue oDomainEntryEntry(UniValue::VOBJ);
                    BuildBDAPJson(entry, oDomainEntryEntry, true);
                    oDomainEntryList.push_back(oDomainEntryEntry);
                    index++;
                }
            }
            pcursor->Next();
        }
        catch (std::exception& e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

bool CDomainEntryDB::GetDomainEntryInfo(const std::vector<unsigned char>& vchFullObjectPath, UniValue& oDomainEntryInfo)
{
    CDomainEntry entry;
    if (!ReadDomainEntry(vchFullObjectPath, entry)) {
        return false;
    }
    
    if (!BuildBDAPJson(entry, oDomainEntryInfo, false)) {
        return false;  
    }

    return true;
}

bool CheckDomainEntryDB()
{
    if (!pDomainEntryDB)
        return false;

    return true;
}

bool FlushLevelDB() 
{
    {
        LOCK(cs_bdap_entry);
        if (pDomainEntryDB != NULL)
        {
            if (!pDomainEntryDB->Flush()) {
                LogPrintf("Failed to write to BDAP database!");
                return false;
            }
        }
    }
    return true;
}

void CleanupLevelDB(int& nRemoved)
{
    if(pDomainEntryDB != NULL)
        pDomainEntryDB->CleanupLevelDB(nRemoved);
    FlushLevelDB();
}


static bool CommonDataCheck(const CDomainEntry& entry, const vchCharString& vvchOpParameters, std::string& errorMessage)
{
    if (entry.IsNull() == true)
    {
        errorMessage = "CommonDataCheck failed! DomainEntry is null.";
        return false;
    }

    if (vvchOpParameters.size() == 0)
    {
        errorMessage = "CommonDataCheck failed! Invalid parameters.";
        return false;
    }

    if (entry.GetFullObjectPath() != stringFromVch(vvchOpParameters[0]))
    {
        errorMessage = "CommonDataCheck failed! Script operation parameter does not match entry entry object.";
        return false;
    }
    
    if (entry.DomainComponent != vchDefaultDomainName)
    {
        errorMessage = "CommonDataCheck failed! Must use default domain.";
        return false;
    }

    if (entry.OrganizationalUnit != vchDefaultPublicOU && entry.OrganizationalUnit != vchDefaultAdminOU)
    {
        errorMessage = "CommonDataCheck failed! Must use default public organizational unit.";
        return false;
    }

    if (entry.OrganizationalUnit == vchDefaultAdminOU)
    {
        errorMessage = "CommonDataCheck failed! Can not use default admin domain.";
        return false;
    }

    return true;
}

bool CheckNewDomainEntryTxInputs(const CTransaction& tx, const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                               const int op, std::string& errorMessage, bool fJustCheck)
{
    if (!CommonDataCheck(entry, vvchOpParameters, errorMessage))
        return error(errorMessage.c_str());

    if (fJustCheck)
        return true;

    CDomainEntry getDomainEntry;
    if (GetDomainEntry(entry.vchFullObjectPath(), getDomainEntry))
    {
        errorMessage = "CheckNewDomainEntryTxInputs: - The entry " + getDomainEntry.GetFullObjectPath() + " already exists.  Add new entry failed!";
        return error(errorMessage.c_str());
    }

    if (!pDomainEntryDB)
    {
        errorMessage = "CheckNewDomainEntryTxInputs failed! Can not open LevelDB BDAP entry database.";
        return error(errorMessage.c_str());
    }

    if (!pDomainEntryDB->AddDomainEntry(entry, op))
    {
        errorMessage = "CheckNewDomainEntryTxInputs failed! Error adding new entry entry request to LevelDB.";
        return error(errorMessage.c_str());
    }

    return FlushLevelDB();
}

bool CheckDeleteDomainEntryTxInputs(const CTransaction& tx, const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                  const int op, std::string& errorMessage, bool fJustCheck)
{
    //if exists, check for owner's signature
    if (!CommonDataCheck(entry, vvchOpParameters, errorMessage))
        return error(errorMessage.c_str());

    if (fJustCheck)
        return true;

    CDomainEntry prevDomainEntry;
    if (!GetDomainEntry(entry.vchFullObjectPath(), prevDomainEntry))
    {
        errorMessage = "CheckDeleteDomainEntryTxInputs: - Can not find " + prevDomainEntry.GetFullObjectPath() + " entry; this delete operation failed!";
        return error(errorMessage.c_str());
    }

    CTxDestination bdapDest;
    if (!ExtractDestination(scriptOp, bdapDest))
    {
        errorMessage = "CheckDeleteDomainEntryTxInputs: - " + _("Cannot extract destination of BDAP input; this delete operation failed!");
        return error(errorMessage.c_str());
    }
    else
    {
        CDynamicAddress prevAddress(bdapDest);
        {
        if (EncodeBase58(entry.WalletAddress) != prevAddress.ToString())
            errorMessage = "CheckDeleteDomainEntryTxInputs: - " + _("You are not the owner of this BDAP entry; this delete operation failed!");
            return error(errorMessage.c_str());
        }
    }

    if (!pDomainEntryDB->EraseDomainEntry(entry.vchFullObjectPath()))
    {
        errorMessage = "CheckDeleteDomainEntryTxInputs: - Error deleting entry entry in LevelDB; this delete operation failed!";
        return error(errorMessage.c_str());
    }

    return FlushLevelDB();
}

bool CheckActivateDomainEntryTxInputs(const CTransaction& tx, const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                    const int op, std::string& errorMessage, bool fJustCheck)
{
    //check name in operation matches entry data in leveldb as a new request
    //check if new request exists and is not expired
    return false;
}

bool CheckUpdateDomainEntryTxInputs(const CTransaction& tx, const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                  const int op, std::string& errorMessage, bool fJustCheck)
{
    //if exists, check for owner's signature
    if (!CommonDataCheck(entry, vvchOpParameters, errorMessage))
        return error(errorMessage.c_str());

    if (fJustCheck)
        return true;

    CDomainEntry prevDomainEntry;
    if (!GetDomainEntry(entry.vchFullObjectPath(), prevDomainEntry))
    {
        errorMessage = "CheckUpdateDomainEntryTxInputs: - Can not find " + prevDomainEntry.GetFullObjectPath() + " entry; this update operation failed!";
        return error(errorMessage.c_str());
    }

    CTxDestination bdapDest;
    if (!ExtractDestination(scriptOp, bdapDest))
    {
        errorMessage = "CheckUpdateDomainEntryTxInputs: - " + _("Cannot extract destination of BDAP input; this update operation failed!");
        return error(errorMessage.c_str());
    }
    else
    {
        CDynamicAddress prevAddress(bdapDest);
        {
        if (EncodeBase58(entry.WalletAddress) != prevAddress.ToString())
            errorMessage = "CheckUpdateDomainEntryTxInputs: - " + _("You are not the owner of this BDAP entry; this update operation failed!");
            return error(errorMessage.c_str());
        }
    }

    if (!pDomainEntryDB->UpdateDomainEntry(entry.vchFullObjectPath(), entry))
    {
        errorMessage = "CheckUpdateDomainEntryTxInputs: - Error updating entry entry in LevelDB; this update operation failed!";
        return error(errorMessage.c_str());
    }

    return FlushLevelDB();
}

bool CheckMoveDomainEntryTxInputs(const CTransaction& tx, const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                const int op, std::string& errorMessage, bool fJustCheck)
{
    //check name in operation matches entry data in leveldb
    //check if exists already
    //if exists, check for owner's signature
    return false;
}

bool CheckExecuteDomainEntryTxInputs(const CTransaction& tx, const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                   const int op, std::string& errorMessage, bool fJustCheck)
{
    //check name in operation matches entry data in leveldb
    //check if exists already
    //if exists, check for owner's signature
    return false;
}

bool CheckBindDomainEntryTxInputs(const CTransaction& tx, const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                const int op, std::string& errorMessage, bool fJustCheck)
{
    //check names in operation matches entry data in leveldb
    //check if request or accept response
    //check if names exists already
    //if exists, check for owner's signature
    return false;
}

bool CheckRevokeDomainEntryTxInputs(const CTransaction& tx, const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                  const int op, std::string& errorMessage, bool fJustCheck)
{
    //check name in operation matches entry data in leveldb
    //check if names exists already
    //if exists, check for fluid signature
    return false;
}

bool CheckDomainEntryTxInputs(const CCoinsViewCache& inputs, const CTransaction& tx, 
                            int op, const std::vector<std::vector<unsigned char> >& vvchArgs, bool fJustCheck, int nHeight, std::string& errorMessage, bool bSanityCheck) 
{
    if (tx.IsCoinBase() && !fJustCheck && !bSanityCheck)
    {
        LogPrintf("*Trying to add BDAP entry in coinbase transaction, skipping...");
        return true;
    }

    //if (fDebug && !bSanityCheck)
        LogPrintf("*** BDAP nHeight=%d, chainActive.Tip()=%d, op=%s, hash=%s justcheck=%s\n", nHeight, chainActive.Tip()->nHeight, DomainEntryFromOp(op).c_str(), tx.GetHash().ToString().c_str(), fJustCheck ? "JUSTCHECK" : "BLOCK");
    
    CScript scriptOp;
    vchCharString vvchOpParameters;
    if (!GetDomainEntryOpScript(tx, scriptOp, vvchOpParameters, op))
    {
        errorMessage = "BDAP_CONSENSUS_ERROR: ERRCODE: 3600 - " + _("Transaction does not contain BDAP operation script!");
        return error(errorMessage.c_str());
    }
    const std::string strOperationType = GetDomainEntryOpTypeString(scriptOp);
    if (fDebug)
        LogPrintf("CheckDomainEntryTxInputs, strOperationType= %s \n", strOperationType);
    
    // unserialize BDAP from txn, check if the entry is valid and does not conflict with a previous entry
    CDomainEntry entry;
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nDataOut;
    
    bool bData = GetDomainEntryData(tx, vchData, vchHash, nDataOut);
    if(bData && !entry.UnserializeFromData(vchData, vchHash))
    {
        errorMessage = "BDAP_CONSENSUS_ERROR: ERRCODE: 3601 - " + _("UnserializeFromData data in tx failed!");
        return error(errorMessage.c_str());
    }

    if(!entry.ValidateValues(errorMessage))
    {
        errorMessage = "BDAP_CONSENSUS_ERROR: ERRCODE: 3602 - " + errorMessage;
        return error(errorMessage.c_str());
    }

    entry.txHash = tx.GetHash();

    if (strOperationType == "bdap_new")
        return CheckNewDomainEntryTxInputs(tx, entry, scriptOp, vvchOpParameters, op, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_delete")
        return CheckDeleteDomainEntryTxInputs(tx, entry, scriptOp, vvchOpParameters, op, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_activate")
        return CheckActivateDomainEntryTxInputs(tx, entry, scriptOp, vvchOpParameters, op, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_update")
        return CheckUpdateDomainEntryTxInputs(tx, entry, scriptOp, vvchOpParameters, op, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_move")
        return CheckMoveDomainEntryTxInputs(tx, entry, scriptOp, vvchOpParameters, op, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_execute")
        return CheckExecuteDomainEntryTxInputs(tx, entry, scriptOp, vvchOpParameters, op, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_bind")
        return CheckBindDomainEntryTxInputs(tx, entry, scriptOp, vvchOpParameters, op, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_revoke")
        return CheckRevokeDomainEntryTxInputs(tx, entry, scriptOp, vvchOpParameters, op, errorMessage, fJustCheck);

    return false;
}