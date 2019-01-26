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

bool CLinkRequestDB::AddMyLinkRequest(const CLinkRequest& link)
{
    LOCK(cs_link_request);
    return Write(make_pair(std::string("requestor"), link.RequestorFullObjectPath), link.RequestorPubKey)
            && Write(make_pair(std::string("recipient"), link.RecipientFullObjectPath), link.RequestorPubKey)
            && Write(make_pair(std::string("mylink"), link.RequestorPubKey), link);
}

bool CLinkRequestDB::ReadMyLinkRequest(const std::vector<unsigned char>& vchPubKey, CLinkRequest& link)
{
    LOCK(cs_link_request);
    return CDBWrapper::Read(make_pair(std::string("mylink"), vchPubKey), link);
}

// Lists active pending link requests.
bool CLinkRequestDB::ListMyLinkRequests(std::vector<CLinkRequest>& vchLinkRequests)
{
    int index = 0;
    std::pair<std::string, CharString> key;
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        try {
            if (pcursor->GetKey(key) && key.first == "mylink") {
                CLinkRequest link;
                pcursor->GetValue(link);
                vchLinkRequests.push_back(link);
                index++;
            }
            pcursor->Next();
        }
        catch (std::exception& e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
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
           CDBWrapper::Erase(make_pair(std::string("recipient"), link.RecipientFullObjectPath)) &&
           CDBWrapper::Erase(make_pair(std::string("requestor"), link.RequestorFullObjectPath));
}

bool CLinkRequestDB::MyLinkRequestExists(const std::vector<unsigned char>& vchPubKey)
{
    LOCK(cs_link_request);
    return CDBWrapper::Exists(make_pair(std::string("mylink"), vchPubKey));
}

bool CLinkRequestDB::LinkageExists(const std::string& strRequestorFQDN, const std::string& strRecipientFQDN)
{
    std::vector<unsigned char> vchRequestorFQDN = vchFromString(strRequestorFQDN);
    std::vector<unsigned char> vchRecipientFQDN = vchFromString(strRecipientFQDN);
    LOCK(cs_link_request);
    if (CDBWrapper::Exists(make_pair(std::string("recipient"), vchRequestorFQDN))) {
        std::vector<unsigned char> vchPubKey;
        if (CDBWrapper::Read(make_pair(std::string("recipient"), vchRequestorFQDN), vchPubKey)) {
            CLinkRequest link;
            if (CDBWrapper::Read(make_pair(std::string("mylink"), vchPubKey), link))
                if (link.Matches(strRequestorFQDN, strRecipientFQDN))
                    return true;
        }
    }
    if (CDBWrapper::Exists(make_pair(std::string("recipient"), vchRecipientFQDN))) {
        std::vector<unsigned char> vchPubKey;
        if (CDBWrapper::Read(make_pair(std::string("recipient"), vchRecipientFQDN), vchPubKey)) {
            CLinkRequest link;
            if (CDBWrapper::Read(make_pair(std::string("mylink"), vchPubKey), link)) 
                if (link.Matches(strRequestorFQDN, strRecipientFQDN))
                    return true;
        }
    }
    if (CDBWrapper::Exists(make_pair(std::string("requestor"), vchRequestorFQDN))) {
        std::vector<unsigned char> vchPubKey;
        if (CDBWrapper::Read(make_pair(std::string("requestor"), vchRequestorFQDN), vchPubKey)) {
            CLinkRequest link;
            if (CDBWrapper::Read(make_pair(std::string("mylink"), vchPubKey), link))
                if (link.Matches(strRequestorFQDN, strRecipientFQDN))
                    return true;
        }
    }
    if (CDBWrapper::Exists(make_pair(std::string("requestor"), vchRecipientFQDN))) {
        std::vector<unsigned char> vchPubKey;
        if (CDBWrapper::Read(make_pair(std::string("requestor"), vchRecipientFQDN), vchPubKey)) {
            CLinkRequest link;
            if (CDBWrapper::Read(make_pair(std::string("mylink"), vchPubKey), link))
                if (link.Matches(strRequestorFQDN, strRecipientFQDN))
                    return true;
        }
    }

    return false;
}

// Removes expired records from databases.
bool CLinkRequestDB::CleanupMyLinkRequestDB(int& nRemoved)
{
    LOCK(cs_link_request);
    boost::scoped_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    std::pair<std::string, std::vector<unsigned char> > key;
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        try {
            if (pcursor->GetKey(key) && key.first == "mylink") {
                CLinkRequest link;
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
        writeState = Write(make_pair(std::string("pubkey"), stringFromVch(vvchOpParameters[0])), txid);
        if (writeState && vvchOpParameters.size() > 1)
            writeState = Write(make_pair(std::string("pubkey"), stringFromVch(vvchOpParameters[1])), txid);
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

// Link Accept DB
bool CLinkAcceptDB::AddMyLinkAccept(const CLinkAccept& link)
{
    LOCK(cs_link_accept);
    return Write(make_pair(std::string("requestor"), link.RequestorFullObjectPath), link.RecipientPubKey)
            && Write(make_pair(std::string("recipient"), link.RecipientFullObjectPath), link.RecipientPubKey)
            && Write(make_pair(std::string("mylink"), link.RecipientPubKey), link);
}

bool CLinkAcceptDB::ReadMyLinkAccept(const std::vector<unsigned char>& vchPubKey, CLinkAccept& link)
{
    LOCK(cs_link_accept);
    return CDBWrapper::Read(make_pair(std::string("mylink"), vchPubKey), link);
}

// Lists my active links pending accept tx.
bool CLinkAcceptDB::ListMyLinkAccepts(std::vector<CLinkAccept>& vchLinkAccepts)
{
    int index = 0;
    std::pair<std::string, CharString> key;
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        try {
            if (pcursor->GetKey(key) && key.first == "mylink") {
                CLinkAccept link;
                pcursor->GetValue(link);
                vchLinkAccepts.push_back(link);
                index++;
            }
            pcursor->Next();
        }
        catch (std::exception& e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
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
           CDBWrapper::Erase(make_pair(std::string("recipient"), link.RecipientFullObjectPath)) &&
           CDBWrapper::Erase(make_pair(std::string("requestor"), link.RequestorFullObjectPath));
}

bool CLinkAcceptDB::MyLinkAcceptExists(const std::vector<unsigned char>& vchPubKey)
{
    LOCK(cs_link_accept);
    return CDBWrapper::Exists(make_pair(std::string("mylink"), vchPubKey));
}

bool CLinkAcceptDB::LinkageExists(const std::string& strRequestorFQDN, const std::string& strRecipientFQDN)
{
    std::vector<unsigned char> vchRequestorFQDN = vchFromString(strRequestorFQDN);
    std::vector<unsigned char> vchRecipientFQDN = vchFromString(strRecipientFQDN);
    LOCK(cs_link_request);
    if (CDBWrapper::Exists(make_pair(std::string("recipient"), vchRequestorFQDN))) {
        std::vector<unsigned char> vchPubKey;
        if (CDBWrapper::Read(make_pair(std::string("recipient"), vchRequestorFQDN), vchPubKey)) {
            CLinkAccept link;
            if (CDBWrapper::Read(make_pair(std::string("mylink"), vchPubKey), link))
                if (link.Matches(strRequestorFQDN, strRecipientFQDN))
                    return true;
        }
    }
    if (CDBWrapper::Exists(make_pair(std::string("recipient"), vchRecipientFQDN))) {
        std::vector<unsigned char> vchPubKey;
        if (CDBWrapper::Read(make_pair(std::string("recipient"), vchRecipientFQDN), vchPubKey)) {
            CLinkAccept link;
            if (CDBWrapper::Read(make_pair(std::string("mylink"), vchPubKey), link)) 
                if (link.Matches(strRequestorFQDN, strRecipientFQDN))
                    return true;
        }
    }
    if (CDBWrapper::Exists(make_pair(std::string("requestor"), vchRequestorFQDN))) {
        std::vector<unsigned char> vchPubKey;
        if (CDBWrapper::Read(make_pair(std::string("requestor"), vchRequestorFQDN), vchPubKey)) {
            CLinkAccept link;
            if (CDBWrapper::Read(make_pair(std::string("mylink"), vchPubKey), link))
                if (link.Matches(strRequestorFQDN, strRecipientFQDN))
                    return true;
        }
    }
    if (CDBWrapper::Exists(make_pair(std::string("requestor"), vchRecipientFQDN))) {
        std::vector<unsigned char> vchPubKey;
        if (CDBWrapper::Read(make_pair(std::string("requestor"), vchRecipientFQDN), vchPubKey)) {
            CLinkAccept link;
            if (CDBWrapper::Read(make_pair(std::string("mylink"), vchPubKey), link))
                if (link.Matches(strRequestorFQDN, strRecipientFQDN))
                    return true;
        }
    }

    return false;
}

// Removes expired records from databases.
bool CLinkAcceptDB::CleanupMyLinkAcceptDB(int& nRemoved)
{
    LOCK(cs_link_accept);
    boost::scoped_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    std::pair<std::string, std::vector<unsigned char> > key;
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        try {
            if (pcursor->GetKey(key) && key.first == "mylink") {
                CLinkRequest link;
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
        writeState = Write(make_pair(std::string("pubkey"), stringFromVch(vvchOpParameters[0])), txid);
        if (writeState && vvchOpParameters.size() > 1)
            writeState = Write(make_pair(std::string("pubkey"), stringFromVch(vvchOpParameters[1])), txid);
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
    txid.SetNull();
    if (!pLinkRequestDB || !pLinkRequestDB->ReadLinkRequestIndex(vchPubKey, txid)) {
        return false;
    }
    return !txid.IsNull();
}

bool GetLinkAcceptIndex(const std::vector<unsigned char>& vchPubKey, uint256& txid)
{
    txid.SetNull();
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
                LogPrintf("%s -- Failed to flush link request leveldb!\n", __func__);
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
                LogPrintf("%s -- Failed to flush link accept leveldb!\n", __func__);
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
    if (vvchOpParameters.size() > 3)
    {
        errorMessage = "CommonLinkParameterCheck failed! Invalid parameters.";
        return false;
    }
    // check pubkey size
    if (vvchOpParameters[0].size() > DHT_HEX_PUBLIC_KEY_LENGTH)
    {
        errorMessage = "CommonLinkParameterCheck failed! Incorrect pubkey size.";
        return false;
    }
    // check shared pubkey size
    if (vvchOpParameters.size() > 1 && vvchOpParameters[1].size() > DHT_HEX_PUBLIC_KEY_LENGTH)
    {
        errorMessage = "CommonLinkParameterCheck failed! Incorrect shared pubkey size.";
        return false;
    }
    // check expiration time is not greater than 10 digits
    if (vvchOpParameters.size() > 2 && vvchOpParameters[2].size() > 10)
    {
        errorMessage = "CommonLinkParameterCheck failed! Incorrect expiration time.";
        return false;
    }
    
    return true;
}

static bool CheckNewLinkRequestTx(const CScript& scriptData, const vchCharString& vvchOpParameters, const uint256& txid, std::string& errorMessage, bool fJustCheck)
{
    LogPrint("bdap", "%s -- start\n", __func__);
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
    if (!FlushLinkRequestDB())
    {
        errorMessage = "CheckNewLinkRequestTx failed! Error flushing LevelDB.";
        return error(errorMessage.c_str());
    }
    return true;
}

static bool CheckNewLinkAcceptTx(const CScript& scriptData, const vchCharString& vvchOpParameters, const uint256& txid, std::string& errorMessage, bool fJustCheck)
{
    LogPrint("bdap", "%s -- start\n", __func__);
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
    if (!FlushLinkAcceptDB()) 
    {
        errorMessage = "CheckNewLinkAcceptTx failed! Error flushing LevelDB.";
        return error(errorMessage.c_str());
    }
    return true;
}

bool CheckLinkTx(const CTransactionRef& tx, const int& op1, const int& op2, const std::vector<std::vector<unsigned char> >& vvchArgs, 
                                bool fJustCheck, int nHeight, std::string& errorMessage, bool bSanityCheck) 
{
    if (tx->IsCoinBase() && !fJustCheck && !bSanityCheck) {
        LogPrintf("%s -- Trying to add BDAP link in coinbase transaction, skipping...\n", __func__);
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

bool CheckLinkageRequestExists(const std::string& strRequestorFQDN, const std::string& strRecipientFQDN)
{
    // For link requests, check accept links as well.
    if (CheckLinkageAcceptExists(strRequestorFQDN, strRecipientFQDN))
        return true;

    if (pLinkRequestDB->LinkageExists(strRequestorFQDN, strRecipientFQDN))
        return true;

    return false;
}

bool CheckLinkageAcceptExists(const std::string& strRequestorFQDN, const std::string& strRecipientFQDN)
{
    if (pLinkAcceptDB->LinkageExists(strRequestorFQDN, strRecipientFQDN))
        return true;

    return false;
}