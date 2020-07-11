// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/linkingdb.h"

#include "amount.h"
#include "bdap/fees.h"
#include "bdap/utils.h"
#include "base58.h"
#include "utilmoneystr.h"
#include "validation.h"
#include "validationinterface.h"

#include <boost/thread.hpp>

CLinkDB *pLinkDB = NULL;

bool UndoLinkData(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchSharedPubKey)
{
    if (!pLinkDB)
        return false;

    return pLinkDB->EraseLinkIndex(vchPubKey, vchSharedPubKey);
}

bool CLinkDB::AddLinkIndex(const vchCharString& vvchOpParameters, const uint256& txid)
{ 
    bool writeState = false;
    {
        LOCK(cs_link);
        writeState = Write(make_pair(std::string("pubkey"), stringFromVch(vvchOpParameters[0])), txid);
        if (writeState && vvchOpParameters.size() > 1)
            writeState = Write(make_pair(std::string("pubkey"), stringFromVch(vvchOpParameters[1])), txid);
    }

    return writeState;
}

bool CLinkDB::ReadLinkIndex(const std::vector<unsigned char>& vchPubKey, uint256& txid)
{
    LOCK(cs_link);
    return CDBWrapper::Read(make_pair(std::string("pubkey"), vchPubKey), txid);
}

bool CLinkDB::EraseLinkIndex(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchSharedPubKey)
{
    if (!LinkExists(vchPubKey)) 
        return false;
    if (!LinkExists(vchSharedPubKey)) 
        return false;

    bool result = false;
    LOCK(cs_link);
    result = CDBWrapper::Erase(make_pair(std::string("pubkey"), vchPubKey));
    if (!result)
        return false;

    return CDBWrapper::Erase(make_pair(std::string("pubkey"), vchSharedPubKey));
}

bool CLinkDB::LinkExists(const std::vector<unsigned char>& vchPubKey)
{
    LOCK(cs_link);
    return CDBWrapper::Exists(make_pair(std::string("pubkey"), vchPubKey));
}

bool GetLinkIndex(const std::vector<unsigned char>& vchPubKey, uint256& txid)
{
    txid.SetNull();
    if (!pLinkDB || !pLinkDB->ReadLinkIndex(vchPubKey, txid)) {
        return false;
    }
    return !txid.IsNull();
}

bool CheckLinkDB()
{
    if (!pLinkDB)
        return false;

    return true;
}

bool FlushLinkDB() 
{
    {
        LOCK(cs_link);
        if (pLinkDB != NULL)
        {
            if (!pLinkDB->Flush()) {
                LogPrintf("%s -- Failed to flush link leveldb!\n", __func__);
                return false;
            }
        }
    }
    return true;
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

static bool CheckNewLinkTx(const CScript& scriptData, const vchCharString& vvchOpParameters, const uint256& txid, std::string& errorMessage, bool fJustCheck)
{
    if (!CommonLinkParameterCheck(vvchOpParameters, errorMessage))
        return error(errorMessage.c_str());

    if (!scriptData.IsUnspendable()) {
        errorMessage = "CheckNewLinkTx failed! Data script should be unspendable.";
        return error(errorMessage.c_str());
    }
    CTxOut txout(0, scriptData);
    size_t nSize = GetSerializeSize(txout, SER_DISK,0)+148u;
    LogPrint("bdap", "%s -- scriptData.size() = %u, Serialize Size = %u \n", __func__, scriptData.size(), nSize);
    if (nSize > MAX_BDAP_LINK_DATA_SIZE) {
        errorMessage = "CheckNewLinkTx failed! Data script is too large.";
        return error(errorMessage.c_str());
    }
    if (fJustCheck)
        return true;

    if (!pLinkDB)
    {
        errorMessage = "CheckNewLinkTx failed! Can not open LevelDB BDAP link database.";
        return error(errorMessage.c_str());
    }
    if (!pLinkDB->AddLinkIndex(vvchOpParameters, txid))
    {
        errorMessage = "CheckNewLinkTx failed! Error adding link index to LevelDB.";
        return error(errorMessage.c_str());
    }
    if (!FlushLinkDB())
    {
        errorMessage = "CheckNewLinkTx failed! Error flushing link LevelDB.";
        return error(errorMessage.c_str());
    }
    return true;
}

bool CheckLinkTx(const CTransactionRef& tx, const int& op1, const int& op2, const std::vector<std::vector<unsigned char> >& vvchArgs, 
                                const bool fJustCheck, const int& nHeight, const uint32_t& nBlockTime, const bool bSanityCheck, std::string& errorMessage) 
{
    if (tx->IsCoinBase() && !fJustCheck && !bSanityCheck) {
        LogPrintf("%s -- Trying to add BDAP link in coinbase transaction, skipping...\n", __func__);
        return true;
    }

    LogPrint("bdap", "%s -- *** BDAP link nHeight=%d, chainActive.Tip()=%d, op1=%s, op2=%s, hash=%s justcheck=%s\n", __func__, nHeight, chainActive.Tip()->nHeight, BDAPFromOp(op1).c_str(), BDAPFromOp(op2).c_str(), tx->GetHash().ToString().c_str(), fJustCheck ? "JUSTCHECK" : "BLOCK");

    CScript scriptData;
    if (!GetBDAPDataScript(tx, scriptData))
        return false;

    // extract amounts from tx.
    CAmount dataAmount, opAmount;
    if (!ExtractAmountsFromTx(tx, dataAmount, opAmount)) {
        errorMessage = "Unable to extract BDAP amounts from transaction";
        return false;
    }

    const std::string strOperationType = GetBDAPOpTypeString(op1, op2);

    CAmount monthlyFee, oneTimeFee, depositFee;
    if (strOperationType == "bdap_new_link_request") {
        uint16_t nMonths = 0;
        if (!GetBDAPFees(OP_BDAP_NEW, OP_BDAP_LINK_REQUEST, BDAP::ObjectType::BDAP_LINK_REQUEST, nMonths, monthlyFee, oneTimeFee, depositFee)) {
            errorMessage = "Failed to get BDAP fees for new link request";
            return false;
        }
        if (oneTimeFee > dataAmount) {
            LogPrintf("%s -- Invalid BDAP one-time fee amount for a new BDAP link request. Total paid %d but should be %d\n", __func__, 
                            FormatMoney(dataAmount), FormatMoney(oneTimeFee));
            errorMessage = "Invalid BDAP fee amount for a new BDAP link request";
            return false;
        }
        else {
            LogPrint("bdap", "%s -- Valid BDAP one-time fee amount for new BDAP request link. One-time paid %d, should be %d.\n", __func__, 
                                    FormatMoney(dataAmount), FormatMoney(oneTimeFee));
        }
        if (depositFee > opAmount) {
            LogPrintf("%s -- Invalid BDAP deposit amount for a new BDAP link request. Total paid %d but should be %d\n", __func__, 
                            FormatMoney(opAmount), FormatMoney(depositFee));
            errorMessage = "Invalid BDAP fee amount for a new BDAP link request";
            return false;
        } else {
            LogPrint("bdap", "%s -- Valid BDAP deposit fee amount for new BDAP request link. Deposit paid %d, should be %d\n", __func__, 
                                    FormatMoney(opAmount), FormatMoney(depositFee));
        }
        return CheckNewLinkTx(scriptData, vvchArgs, tx->GetHash(), errorMessage, fJustCheck);
    }
    else if (strOperationType == "bdap_new_link_accept") {
        uint16_t nMonths = 0;
        if (!GetBDAPFees(OP_BDAP_NEW, OP_BDAP_LINK_ACCEPT, BDAP::ObjectType::BDAP_LINK_ACCEPT, nMonths, monthlyFee, oneTimeFee, depositFee)) {
            errorMessage = "Failed to get BDAP fees for new link accept";
            return false;
        }
        if (oneTimeFee > dataAmount) {
            LogPrintf("%s -- Invalid BDAP one-time fee amount for a new BDAP link accept. Total paid %d but should be %d\n", __func__, 
                            FormatMoney(dataAmount), FormatMoney(oneTimeFee));
            errorMessage = "Invalid BDAP fee amount for a new BDAP link accept";
            return false;
        } else {
            LogPrint("bdap", "%s -- Valid BDAP one-time fee amount for new BDAP accept link. One-time paid %d, should be %d.\n", __func__, 
                                    FormatMoney(dataAmount), FormatMoney(oneTimeFee));
        }
        if (depositFee > opAmount) {
            LogPrintf("%s -- Invalid BDAP deposit amount for a new BDAP link accept. Total paid %d but should be %d\n", __func__, 
                            FormatMoney(opAmount), FormatMoney(depositFee));
            errorMessage = "Invalid BDAP fee amount for a new BDAP link accept";
            return false;
        } else {
            LogPrint("bdap", "%s -- Valid BDAP deposit fee amount for new BDAP accept link. Deposit paid %d, should be %d\n", __func__, 
                                    FormatMoney(opAmount), FormatMoney(depositFee));
        }
        return CheckNewLinkTx(scriptData, vvchArgs, tx->GetHash(), errorMessage, fJustCheck);
    }

    return false;
}

bool CheckPreviousLinkInputs(const std::string& strOpType, const CScript& scriptOp, const std::vector<std::vector<unsigned char>>& vvchOpParameters, std::string& errorMessage, bool fJustCheck)
{
    // finds the previous link txid and makes sure this operation is coming from the same wallet address as the new or update entry
    CTxDestination bdapDest;
    if (!ExtractDestination(scriptOp, bdapDest))
    {
        errorMessage = "CheckPreviousLinkInputs: - " + _("Cannot extract destination of BDAP input; this delete operation failed!");
        return error(errorMessage.c_str());
    }
    else
    {
        std::vector<unsigned char> vchPubKey = vvchOpParameters[0];
        uint256 prevTxId;
        if (strOpType == "bdap_delete_link_request") {
            if (!GetLinkIndex(vchPubKey, prevTxId)) {
                errorMessage = "CheckPreviousLinkInputs: - " + _("Cannot get previous link request txid; this delete operation failed!");
                return error(errorMessage.c_str());
            }
        }
        else if (strOpType == "bdap_delete_link_accept") {
            if (!GetLinkIndex(vchPubKey, prevTxId)) {
                errorMessage = "CheckPreviousLinkInputs: - " + _("Cannot get previous link accept txid; this delete operation failed!");
                return error(errorMessage.c_str());
            }
        }
        CTransactionRef prevTx;
        if (!GetPreviousTxRefById(prevTxId, prevTx)) {
            errorMessage = "CheckPreviousLinkInputs: - " + _("Cannot extract previous transaction from BDAP link output; this delete operation failed!");
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
            errorMessage = "CheckPreviousLinkInputs: - " + _("Changes to link data must use the previous wallet address; this delete operation failed!");
            return error(errorMessage.c_str());
        }
        if (fJustCheck)
            return true;

        std::vector<unsigned char> vchSharedPubKey;
        // TODO (BDAP): make sure the DHT pubkeys have not changed before allowing a global delete.
        if (vvchOpParameters.size() > 1)
            vchSharedPubKey = vvchOpParameters[1];

        if (strOpType == "bdap_delete_link_request") {
            pLinkDB->EraseLinkIndex(vchPubKey, vchSharedPubKey);
        }
        else if (strOpType == "bdap_delete_link_accept") {
            pLinkDB->EraseLinkIndex(vchPubKey, vchSharedPubKey);
        }
    }
    return true;
}

bool LinkPubKeyExists(const std::vector<unsigned char>& vchPubKey)
{
    if (!CheckLinkDB())
        return false;

    return pLinkDB->LinkExists(vchPubKey);
}