// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/linkingdb.h"

#include "bdap/utils.h"
#include "base58.h"
#include "validation.h"
#include "validationinterface.h"

#include <boost/thread.hpp>

CLinkRequestDB *pLinkRequestDB = NULL;
CLinkAcceptDB *pLinkAcceptDB = NULL;

bool CLinkRequestDB::AddMyLinkRecipient(const CLinkRequest& link)
{
    LOCK(cs_link_request);
    return Write(make_pair(std::string("recipient"), link.RecipientFullObjectPath), link.RequestorPubKey);
}

bool CLinkRequestDB::AddMyLinkRequest(const CLinkRequest& link)
{
    LOCK(cs_link_request);
    return Write(make_pair(std::string("mylink"), link.RequestorPubKey), link);
}

bool CLinkRequestDB::ReadMyLinkRequest(const std::vector<unsigned char>& vchPubKey, CLinkRequest& link)
{
    LOCK(cs_link_request);
    return CDBWrapper::Read(make_pair(std::string("mylink"), vchPubKey), link);
}

bool CLinkRequestDB::EraseMyLinkRequest(const std::vector<unsigned char>& vchPubKey)
{
    if (!MyLinkRequestExists(vchPubKey))
        return false;

    CLinkRequest link;
    if (!ReadMyLinkRequest(vchPubKey, link))
        return false;

    LOCK(cs_link_request);
    return CDBWrapper::Erase(make_pair(std::string("mylink"), vchPubKey)) &&
           CDBWrapper::Erase(make_pair(std::string("recipient"), link.RecipientFullObjectPath));
}

bool CLinkRequestDB::MyLinkRequestExists(const std::vector<unsigned char>& vchPubKey)
{
    LOCK(cs_link_request);
    return CDBWrapper::Exists(make_pair(std::string("mylink"), vchPubKey));
}

// Removes expired records from databases.
bool CLinkRequestDB::CleanupMyLinkRequestDB(int& nRemoved)
{
    LOCK(cs_link_request);
    boost::scoped_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    CLinkRequest link;
    std::pair<std::string, std::vector<unsigned char> > key;
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        try {
            if (pcursor->GetKey(key) && key.first == "mylink") {
                pcursor->GetValue(link);
                if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= link.nExpireTime)
                {
                    nRemoved++;
                    EraseMyLinkRequest(link.RequestorPubKey);
                }
            }
            pcursor->Next();
        } catch (std::exception &e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

bool CLinkRequestDB::AddLinkRequestIndex(const vchCharString& vvchOpParameters, const uint256& txid)
{ 
    bool writeState = false;
    {
        LOCK(cs_link_request);
        writeState = Write(make_pair(std::string("pubkey"), stringFromVch(vvchOpParameters[0])), txid) && 
                     Write(make_pair(std::string("pubkey"), stringFromVch(vvchOpParameters[1])), txid);
    }

    return writeState;
}

bool CLinkRequestDB::ReadLinkRequestIndex(const std::vector<unsigned char>& vchPubKey, uint256& txid)
{
    LOCK(cs_link_request);
    return CDBWrapper::Read(make_pair(std::string("pubkey"), vchPubKey), txid);
}

bool CLinkRequestDB::EraseLinkRequestIndex(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchSharedPubKey)
{
    if (!LinkRequestExists(vchPubKey)) 
        return false;
    if (!LinkRequestExists(vchSharedPubKey)) 
        return false;

    bool result = false;
    LOCK(cs_link_request);
    result = CDBWrapper::Erase(make_pair(std::string("pubkey"), vchPubKey));
    if (!result)
        return false;

    return CDBWrapper::Erase(make_pair(std::string("pubkey"), vchSharedPubKey));
}

bool CLinkRequestDB::LinkRequestExists(const std::vector<unsigned char>& vchPubKey)
{
    LOCK(cs_link_request);
    return CDBWrapper::Exists(make_pair(std::string("pubkey"), vchPubKey));
}

bool CLinkAcceptDB::AddMyLinkSender(const CLinkAccept& link)
{
    LOCK(cs_link_accept);
    return Write(make_pair(std::string("sender"), link.RequestorFullObjectPath), link.RecipientPubKey);
}

bool CLinkAcceptDB::AddMyLinkAccept(const CLinkAccept& link)
{
    LOCK(cs_link_accept);
    return Write(make_pair(std::string("mylink"), link.RecipientPubKey), link);
}

bool CLinkAcceptDB::ReadMyLinkAccept(const std::vector<unsigned char>& vchPubKey, CLinkAccept& link)
{
    LOCK(cs_link_accept);
    return CDBWrapper::Read(make_pair(std::string("mylink"), vchPubKey), link);
}

bool CLinkAcceptDB::EraseMyLinkAccept(const std::vector<unsigned char>& vchPubKey)
{
    if (!MyLinkAcceptExists(vchPubKey))
        return false;

    CLinkAccept link;
    if (!ReadMyLinkAccept(vchPubKey, link))
        return false;

    LOCK(cs_link_accept);
    return CDBWrapper::Erase(make_pair(std::string("mylink"), vchPubKey)) &&
           CDBWrapper::Erase(make_pair(std::string("sender"), link.RequestorFullObjectPath));
}

bool CLinkAcceptDB::MyLinkAcceptExists(const std::vector<unsigned char>& vchPubKey)
{
    LOCK(cs_link_accept);
    return CDBWrapper::Exists(make_pair(std::string("mylink"), vchPubKey));
}

// Removes expired records from databases.
bool CLinkAcceptDB::CleanupMyLinkAcceptDB(int& nRemoved)
{
    LOCK(cs_link_accept);
    boost::scoped_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    CLinkRequest link;
    std::pair<std::string, std::vector<unsigned char> > key;
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        try {
            if (pcursor->GetKey(key) && key.first == "mylink") {
                pcursor->GetValue(link);
                if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= link.nExpireTime)
                {
                    nRemoved++;
                    EraseMyLinkAccept(link.RequestorPubKey);
                }
            }
            pcursor->Next();
        } catch (std::exception &e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

bool CLinkAcceptDB::AddLinkAcceptIndex(const vchCharString& vvchOpParameters, const uint256& txid)
{ 
    bool writeState = false;
    {
        LOCK(cs_link_accept);
        writeState = Write(make_pair(std::string("pubkey"), stringFromVch(vvchOpParameters[0])), txid) && 
                     Write(make_pair(std::string("pubkey"), stringFromVch(vvchOpParameters[1])), txid);
    }

    return writeState;
}

bool CLinkAcceptDB::ReadLinkAcceptIndex(const std::vector<unsigned char>& vchPubKey, uint256& txid)
{
    LOCK(cs_link_accept);
    return CDBWrapper::Read(make_pair(std::string("pubkey"), vchPubKey), txid);
}

bool CLinkAcceptDB::EraseLinkAcceptIndex(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchSharedPubKey)
{
    if (!LinkAcceptExists(vchPubKey)) 
        return false;
    if (!LinkAcceptExists(vchSharedPubKey)) 
        return false;

    bool result = false;
    LOCK(cs_link_accept);
    result = CDBWrapper::Erase(make_pair(std::string("pubkey"), vchPubKey));
    if (!result)
        return false;

    return CDBWrapper::Erase(make_pair(std::string("pubkey"), vchSharedPubKey));
}

bool CLinkAcceptDB::LinkAcceptExists(const std::vector<unsigned char>& vchPubKey)
{
    LOCK(cs_link_accept);
    return CDBWrapper::Exists(make_pair(std::string("pubkey"), vchPubKey));
}

bool GetLinkRequestIndex(const std::vector<unsigned char>& vchPubKey, uint256& txid)
{
    if (!pLinkRequestDB || !pLinkRequestDB->ReadLinkRequestIndex(vchPubKey, txid)) {
        return false;
    }
    
    return !txid.IsNull();
}

bool GetLinkAcceptIndex(const std::vector<unsigned char>& vchPubKey, uint256& txid)
{
    if (!pLinkAcceptDB || !pLinkAcceptDB->ReadLinkAcceptIndex(vchPubKey, txid)) {
        return false;
    }
    
    return !txid.IsNull();
}

bool CheckLinkRequestDB()
{
    if (!pLinkRequestDB)
        return false;

    return true;
}

bool CheckLinkAcceptDB()
{
    if (!pLinkAcceptDB)
        return false;

    return true;
}

bool CheckLinkDBs()
{
    return CheckLinkAcceptDB() && CheckLinkRequestDB();
}

bool FlushLinkRequestDB() 
{
    {
        LOCK(cs_link_request);
        if (pLinkRequestDB != NULL)
        {
            if (!pLinkRequestDB->Flush()) {
                LogPrintf("Failed to flush link request leveldb!");
                return false;
            }
        }
    }
    return true;
}

bool FlushLinkAcceptDB() 
{
    {
        LOCK(cs_link_accept);
        if (pLinkAcceptDB != NULL)
        {
            if (!pLinkAcceptDB->Flush()) {
                LogPrintf("Failed to flush link accept leveldb!");
                return false;
            }
        }
    }
    return true;
}

void CleanupLinkRequestDB(int& nRemoved)
{
    if(pLinkRequestDB != NULL)
        pLinkRequestDB->CleanupMyLinkRequestDB(nRemoved);
    FlushLinkRequestDB();
}

void CleanupLinkAcceptDB(int& nRemoved)
{
    if(pLinkAcceptDB != NULL)
        pLinkAcceptDB->CleanupMyLinkAcceptDB(nRemoved);
    FlushLinkAcceptDB();
}

static bool CommonLinkParameterCheck(const vchCharString& vvchOpParameters, std::string& errorMessage)
{
    if (vvchOpParameters.size() > 4)
    {
        errorMessage = "CommonLinkParameterCheck failed! Invalid parameters.";
        return false;
    }
    // check pubkey size
    if (vvchOpParameters[0].size() != DHT_HEX_PUBLIC_KEY_LENGTH)
    {
        errorMessage = "CommonLinkParameterCheck failed! Incorrect pubkey size.";
        return false;
    }
    // check shared pubkey size
    if (vvchOpParameters[1].size() != DHT_HEX_PUBLIC_KEY_LENGTH)
    {
        errorMessage = "CommonLinkParameterCheck failed! Incorrect shared pubkey size.";
        return false;
    }
    // check expiration time is not greater than 10 digits
    if (vvchOpParameters[2].size() > 10)
    {
        errorMessage = "CommonLinkParameterCheck failed! Incorrect expiration time.";
        return false;
    }
    
    return true;
}

static bool CheckNewLinkRequestTx(const CScript& scriptData, const vchCharString& vvchOpParameters, const uint256& txid, std::string& errorMessage, bool fJustCheck)
{
    if (!CommonLinkParameterCheck(vvchOpParameters, errorMessage))
        return error(errorMessage.c_str());

    if (!scriptData.IsUnspendable()) {
        errorMessage = "CheckNewLinkRequestTx failed! Data script should be unspendable.";
        return error(errorMessage.c_str());
    }
    
    CTxOut txout(0, scriptData);
    size_t nSize = GetSerializeSize(txout, SER_DISK,0)+148u;
    LogPrint("bdap", "%s -- scriptData.size() = %u, nSize = %u \n", __func__, scriptData.size(), nSize);
    if (nSize > MAX_BDAP_LINK_DATA_SIZE) {
        errorMessage = "CheckNewLinkRequestTx failed! Data script is too large.";
        return error(errorMessage.c_str());
    }

    if (fJustCheck)
        return true;

    if (!pLinkRequestDB)
    {
        errorMessage = "CheckNewLinkRequestTx failed! Can not open LevelDB BDAP link request database.";
        return error(errorMessage.c_str());
    }

    if (!pLinkRequestDB->AddLinkRequestIndex(vvchOpParameters, txid))
    {
        errorMessage = "CheckNewLinkRequestTx failed! Error adding link request index to LevelDB.";
        return error(errorMessage.c_str());
    }

    return FlushLinkRequestDB();
}

static bool CheckNewLinkAcceptTx(const CScript& scriptData, const vchCharString& vvchOpParameters, const uint256& txid, std::string& errorMessage, bool fJustCheck)
{
    if (!CommonLinkParameterCheck(vvchOpParameters, errorMessage))
        return error(errorMessage.c_str());

    if (!scriptData.IsUnspendable()) {
        errorMessage = "CheckNewLinkAcceptTx failed! Data script should be unspendable.";
        return error(errorMessage.c_str());
    }
    
    CTxOut txout(0, scriptData);
    size_t nSize = GetSerializeSize(txout, SER_DISK,0)+148u;
    LogPrint("bdap", "%s -- scriptData.size() = %u, nSize = %u \n", __func__, scriptData.size(), nSize);
    if (nSize > 1000) {
        errorMessage = "CheckNewLinkAcceptTx failed! Data script is too large.";
        return error(errorMessage.c_str());
    }

    if (fJustCheck)
        return true;

    if (!pLinkAcceptDB)
    {
        errorMessage = "CheckNewLinkAcceptTx failed! Can not open LevelDB BDAP link accept database.";
        return error(errorMessage.c_str());
    }

    if (!pLinkAcceptDB->AddLinkAcceptIndex(vvchOpParameters, txid))
    {
        errorMessage = "CheckNewLinkAcceptTx failed! Error adding link accept index to LevelDB.";
        return error(errorMessage.c_str());
    }

    return FlushLinkAcceptDB();
}

bool CheckLinkTx(const CTransactionRef& tx, const int& op1, const int& op2, const std::vector<std::vector<unsigned char> >& vvchArgs, 
                                bool fJustCheck, int nHeight, std::string& errorMessage, bool bSanityCheck) 
{
    if (tx->IsCoinBase() && !fJustCheck && !bSanityCheck) {
        LogPrintf("*Trying to add BDAP link in coinbase transaction, skipping...");
        return true;
    }

    //if (fDebug && !bSanityCheck)
        LogPrintf("%s -- *** BDAP link nHeight=%d, chainActive.Tip()=%d, op1=%s, op2=%s, hash=%s justcheck=%s\n", __func__, nHeight, chainActive.Tip()->nHeight, BDAPFromOp(op1).c_str(), BDAPFromOp(op2).c_str(), tx->GetHash().ToString().c_str(), fJustCheck ? "JUSTCHECK" : "BLOCK");

    CScript scriptData;
    if (!GetBDAPDataScript(tx, scriptData))
        return false;

    const std::string strOperationType = GetBDAPOpTypeString(op1, op2);

    if (strOperationType == "bdap_new_link_request") {
        return CheckNewLinkRequestTx(scriptData, vvchArgs, tx->GetHash(), errorMessage, fJustCheck);
    }
    else if (strOperationType == "bdap_new_link_accept") {
        return CheckNewLinkAcceptTx(scriptData, vvchArgs, tx->GetHash(), errorMessage, fJustCheck);
    }

    return false;
}