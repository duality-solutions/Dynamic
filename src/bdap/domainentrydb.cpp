// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/domainentrydb.h"

#include "base58.h"
#include "coins.h"
#include "bdap/utils.h"
#include "validation.h"
#include "validationinterface.h"

#include <univalue.h>

#include <boost/thread.hpp>


CDomainEntryDB *pDomainEntryDB = NULL;

bool GetDomainEntry(const std::vector<unsigned char>& vchObjectPath, CDomainEntry& entry)
{
    if (!pDomainEntryDB || !pDomainEntryDB->ReadDomainEntry(vchObjectPath, entry)) {
        return false;
    }
    
    if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= entry.nExpireTime) {
        entry.SetNull();
        return false;
    }
    return !entry.IsNull();
}

bool GetDomainEntryPubKey(const std::vector<unsigned char>& vchPubKey, CDomainEntry& entry)
{
    if (!pDomainEntryDB || !pDomainEntryDB->ReadDomainEntryPubKey(vchPubKey, entry)) {
        return false;
    }
    
    if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= entry.nExpireTime) {
        entry.SetNull();
        return false;
    }
    return !entry.IsNull();
}

bool CDomainEntryDB::AddDomainEntry(const CDomainEntry& entry, const int op) 
{ 
    bool writeState = false;
    {
        LOCK(cs_bdap_entry);
        writeState = Write(make_pair(std::string("dc"), entry.vchFullObjectPath()), entry) 
                         && Write(make_pair(std::string("pk"), entry.DHTPublicKey), entry);
    }
    if (writeState)
        AddDomainEntryIndex(entry, op);

    return writeState;
}

void CDomainEntryDB::AddDomainEntryIndex(const CDomainEntry& entry, const int op) 
{
    UniValue oName(UniValue::VOBJ);
    if (BuildBDAPJson(entry, oName)) {
        CharString vchOperationType = vchFromString(BDAPFromOp(op));
        GetMainSignals().NotifyBDAPUpdate(oName.write().c_str(), reinterpret_cast<char*>(vchOperationType.data()));
        WriteDomainEntryIndexHistory(entry, op);
    }
}

bool CDomainEntryDB::ReadDomainEntry(const std::vector<unsigned char>& vchObjectPath, CDomainEntry& entry) 
{
    LOCK(cs_bdap_entry);
    return CDBWrapper::Read(make_pair(std::string("dc"), vchObjectPath), entry);
}

bool CDomainEntryDB::ReadDomainEntryPubKey(const std::vector<unsigned char>& vchPubKey, CDomainEntry& entry) 
{
    LOCK(cs_bdap_entry);
    return CDBWrapper::Read(make_pair(std::string("pk"), vchPubKey), entry);
}

bool CDomainEntryDB::EraseDomainEntry(const std::vector<unsigned char>& vchObjectPath) 
{
    LOCK(cs_bdap_entry);
    CDomainEntry entry;
    if (!ReadDomainEntry(vchObjectPath, entry)) {
        LogPrintf("CDomainEntryDB::%s -- ReadDomainEntry failed. vchObjectPath = %s\n", __func__, stringFromVch(vchObjectPath));
        return false;
    }

    return CDBWrapper::Erase(make_pair(std::string("dc"), vchObjectPath));
}

bool CDomainEntryDB::EraseDomainEntryPubKey(const std::vector<unsigned char>& vchPubKey) 
{
    LOCK(cs_bdap_entry);
    CDomainEntry entry;
    if (!ReadDomainEntryPubKey(vchPubKey, entry)) 
        return false;

    return CDBWrapper::Erase(make_pair(std::string("pk"), vchPubKey));
}

bool CDomainEntryDB::DomainEntryExists(const std::vector<unsigned char>& vchObjectPath)
{
    LOCK(cs_bdap_entry);
    return CDBWrapper::Exists(make_pair(std::string("dc"), vchObjectPath));
}

bool CDomainEntryDB::DomainEntryExistsPubKey(const std::vector<unsigned char>& vchPubKey) 
{
    LOCK(cs_bdap_entry);
    return CDBWrapper::Exists(make_pair(std::string("pk"), vchPubKey));
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
            if (pcursor->GetKey(key) && key.first == "dc") {
                pcursor->GetValue(entry);
                  if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= entry.nExpireTime)
                {
                    entriesRemoved++;
                    EraseDomainEntry(key.second);
                }
            }
            else if (pcursor->GetKey(key) && key.first == "txid") {
                std::vector<unsigned char> value;
                CDomainEntry entry;
                pcursor->GetValue(value);
                if (GetDomainEntry(value, entry) && (unsigned int)chainActive.Tip()->GetMedianTimePast() >= entry.nExpireTime)
                {
                    entriesRemoved++;
                    EraseDomainEntryPubKey(entry.DHTPublicKey);
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
        oName.push_back(Pair("op", BDAPFromOp(op)));
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
        oName.push_back(Pair("dht_publickey", HexStr(entry.DHTPublicKey)));
        GetMainSignals().NotifyBDAPUpdate(oName.write().c_str(), "bdap_record");
    }
    WriteDomainEntryIndexHistory(entry, op);
}

bool CDomainEntryDB::UpdateDomainEntry(const std::vector<unsigned char>& vchObjectPath, const CDomainEntry& entry)
{
    LOCK(cs_bdap_entry);

    if (!EraseDomainEntry(vchObjectPath)) {
        LogPrintf("CDomainEntryDB::%s -- EraseDomainEntry failed. vchObjectPath = %s\n", __func__, stringFromVch(vchObjectPath));
        return false;
    }
    if (!EraseDomainEntryPubKey(entry.DHTPublicKey)) {
        LogPrintf("CDomainEntryDB::%s -- EraseDomainEntryPubKey failed. vchObjectPath = %s\n", __func__, stringFromVch(entry.DHTPublicKey));
        return false;
    }

    bool writeState = false;
    writeState = Update(make_pair(std::string("dc"), entry.vchFullObjectPath()), entry) 
                    && Update(make_pair(std::string("pk"), entry.DHTPublicKey), entry);
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
            if (pcursor->GetKey(key) && key.first == "dc") 
            {
                pcursor->GetValue(dirEntry);
                if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= dirEntry.nExpireTime)
                {
                    nRemoved++;
                    EraseDomainEntry(key.second);
                }
            }
            else if (pcursor->GetKey(key) && key.first == "pk") 
            {
                pcursor->GetValue(dirEntry);
                if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= dirEntry.nExpireTime)
                {
                    nRemoved++;
                    EraseDomainEntryPubKey(dirEntry.DHTPublicKey);
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
bool CDomainEntryDB::ListDirectories(const std::vector<unsigned char>& vchObjectLocation, const unsigned int& nResultsPerPage, const unsigned int& nPage, UniValue& oDomainEntryList, const BDAP::ObjectType& accountType)
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
            if (pcursor->GetKey(key) && key.first == "dc") {
                pcursor->GetValue(entry);
                //filter by accountType, unless DEFAULT
                if ((entry.nObjectType == GetObjectTypeInt(accountType)) ||  (accountType == DEFAULT_ACCOUNT_TYPE)) {
                    if (vchObjectLocation.empty() || entry.vchObjectLocation() == vchObjectLocation)
                    {
                        UniValue oDomainEntryEntry(UniValue::VOBJ);
                        BuildBDAPJson(entry, oDomainEntryEntry, false);
                        oDomainEntryList.push_back(oDomainEntryEntry);
                        index++;
                    }
                } //if entry.nObjectType
            }
            pcursor->Next();
        }
        catch (std::exception& e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        } //try-catch


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

bool CDomainEntryDB::GetDomainEntryInfo(const std::vector<unsigned char>& vchFullObjectPath, CDomainEntry& entry)
{
    if (!ReadDomainEntry(vchFullObjectPath, entry)) {
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

    if (entry.OrganizationalUnit == vchDefaultAdminOU)
    {
        errorMessage = "CommonDataCheck failed! Can not use default admin domain.";
        return false;
    }

    if (entry.OrganizationalUnit != vchDefaultPublicOU)
    {
        errorMessage = "CommonDataCheck failed! Must use default public organizational unit.";
        return false;
    }

    return true;
}

bool CheckNewDomainEntryTxInputs(const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                               std::string& errorMessage, bool fJustCheck)
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
    int op = OP_BDAP_NEW;
    if (!pDomainEntryDB->AddDomainEntry(entry, op))
    {
        errorMessage = "CheckNewDomainEntryTxInputs failed! Error adding new entry entry request to LevelDB.";
        return error(errorMessage.c_str());
    }

    return FlushLevelDB();
}

bool CheckDeleteDomainEntryTxInputs(const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                  std::string& errorMessage, bool fJustCheck)
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
        CTransactionRef prevTx = MakeTransactionRef();
        uint256 hashBlock;
        if (!GetTransaction(prevDomainEntry.txHash, prevTx, Params().GetConsensus(), hashBlock, true)) {
            errorMessage = "CheckDeleteDomainEntryTxInputs: - " + _("Cannot extract previous transaction from BDAP output; this delete operation failed!");
            return error(errorMessage.c_str());
        }
        // Get current wallet address used for BDAP tx
        CDynamicAddress txAddress = GetScriptAddress(scriptOp);
        // Get previous wallet address used for BDAP tx
        CScript prevScriptPubKey;
        GetBDAPOpScript(prevTx, prevScriptPubKey);
        CDynamicAddress prevAddress = GetScriptAddress(prevScriptPubKey);
        if (txAddress.ToString() != prevAddress.ToString())
        {
            //check if previous wallet address is used for update and delete txs
            errorMessage = "CheckDeleteDomainEntryTxInputs: - " + _("Delete must use the previous wallet address; this delete operation failed!");
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

bool CheckUpdateDomainEntryTxInputs(const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                  std::string& errorMessage, bool fJustCheck)
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
        CTransactionRef prevTx = MakeTransactionRef();
        uint256 hashBlock;
        if (!GetTransaction(prevDomainEntry.txHash, prevTx, Params().GetConsensus(), hashBlock, true)) {
            errorMessage = "CheckUpdateDomainEntryTxInputs: - " + _("Cannot extract previous transaction from BDAP output; this update operation failed!");
            return error(errorMessage.c_str());
        }
        // Get current wallet address used for BDAP tx
        CDynamicAddress txAddress = GetScriptAddress(scriptOp);
        // Get previous wallet address used for BDAP tx
        CScript prevScriptPubKey;
        GetBDAPOpScript(prevTx, prevScriptPubKey);
        CDynamicAddress prevAddress = GetScriptAddress(prevScriptPubKey);
        if (txAddress.ToString() != prevAddress.ToString())
        {
            //check if previous wallet address is used for update and delete txs
            errorMessage = "CheckUpdateDomainEntryTxInputs: - " + _("Update must use the previous wallet address; this update operation failed!");
            return error(errorMessage.c_str());
        }
    }

    if (!pDomainEntryDB->UpdateDomainEntry(entry.vchFullObjectPath(), entry))
    {
        errorMessage = "CheckUpdateDomainEntryTxInputs: - Error updating entry in LevelDB; this update operation failed!";
        return error(errorMessage.c_str());
    }

    return FlushLevelDB();
}

bool CheckMoveDomainEntryTxInputs(const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                std::string& errorMessage, bool fJustCheck)
{
    //check name in operation matches entry data in leveldb
    //check if exists already
    //if exists, check for owner's signature
    return false;
}

bool CheckExecuteDomainEntryTxInputs(const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                   std::string& errorMessage, bool fJustCheck)
{
    //check name in operation matches entry data in leveldb
    //check if exists already
    //if exists, check for owner's signature
    return false;
}

bool CheckBindDomainEntryTxInputs(const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                std::string& errorMessage, bool fJustCheck)
{
    //check names in operation matches entry data in leveldb
    //check if request or accept response
    //check if names exists already
    //if exists, check for owner's signature
    return false;
}

bool CheckRevokeDomainEntryTxInputs(const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                  std::string& errorMessage, bool fJustCheck)
{
    //check name in operation matches entry data in leveldb
    //check if names exists already
    //if exists, check for fluid signature
    return false;
}

bool CheckDomainEntryTx(const CTransactionRef& tx, const CScript& scriptOp, const int& op1, const int& op2, const std::vector<std::vector<unsigned char> >& vvchArgs, 
                                bool fJustCheck, int nHeight, std::string& errorMessage, bool bSanityCheck) 
{
    if (tx->IsCoinBase() && !fJustCheck && !bSanityCheck)
    {
        LogPrintf("*Trying to add BDAP entry in coinbase transaction, skipping...");
        return true;
    }

    LogPrintf("%s -- *** BDAP nHeight=%d, chainActive.Tip()=%d, op1=%s, op2=%s, hash=%s justcheck=%s\n", __func__, nHeight, chainActive.Tip()->nHeight, BDAPFromOp(op1).c_str(), BDAPFromOp(op2).c_str(), tx->GetHash().ToString().c_str(), fJustCheck ? "JUSTCHECK" : "BLOCK");
    
    // unserialize BDAP from txn, check if the entry is valid and does not conflict with a previous entry
    CDomainEntry entry;
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nDataOut;
    
    bool bData = GetBDAPData(tx, vchData, vchHash, nDataOut);
    if(bData && !entry.UnserializeFromData(vchData, vchHash))
    {
        errorMessage = "BDAP_CONSENSUS_ERROR: ERRCODE: 3601 - " + _("UnserializeFromData data in tx failed!");
        LogPrintf("%s -- %s \n", __func__, errorMessage);
        return error(errorMessage.c_str());
    }

    if(!entry.ValidateValues(errorMessage))
    {
        errorMessage = "BDAP_CONSENSUS_ERROR: ERRCODE: 3602 - " + errorMessage;
        LogPrintf("%s -- %s \n", __func__, errorMessage);
        return error(errorMessage.c_str());
    }

    entry.txHash = tx->GetHash();
    entry.nHeight = nHeight;

    const std::string strOperationType = GetBDAPOpTypeString(op1, op2);
    if (strOperationType == "bdap_new_account")
        return CheckNewDomainEntryTxInputs(entry, scriptOp, vvchArgs, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_delete_account")
        return CheckDeleteDomainEntryTxInputs(entry, scriptOp, vvchArgs, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_update_account")
        return CheckUpdateDomainEntryTxInputs(entry, scriptOp, vvchArgs, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_move_account")
        return CheckMoveDomainEntryTxInputs(entry, scriptOp, vvchArgs, errorMessage, fJustCheck);
    else if (strOperationType == "bdap_revoke_account")
        return CheckRevokeDomainEntryTxInputs(entry, scriptOp, vvchArgs, errorMessage, fJustCheck);

    return false;
}
