// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2011-2017 Namecoin Developers
// Copyright (c) 2013-2017 Emercoin Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.


#include "dns/dns.h"
#include "keystore.h"
#include "script/interpreter.h"
#include "script/script.h"
#include "script/sign.h"
#include "policy/policy.h"
#include "wallet/wallet.h"
#include "rpcserver.h"
#include "txmempool.h"

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/xpressive/xpressive_dynamic.hpp>
#include <fstream>

using namespace std;

#include <univalue.h>

map<CNameVal, set<uint256> > mapNamePending; // for pending tx

// forward decls
extern string _(const char* psz);
extern map<uint256, CTransaction> mapTransactions;
extern CWallet* pwalletMain;

class CNamecoinHooks : public CHooks
{
public:
    virtual bool IsNameFeeEnough(const CTransaction& tx, const CAmount& txFee);
    virtual bool CheckInputs(const CTransaction& tx, const CBlockIndex* pindexBlock, vector<nameTempProxy> &vName, const CDiskTxPos& pos, const CAmount& txFee);
    virtual bool DisconnectInputs(const CTransaction& tx);
    virtual bool ConnectBlock(CBlockIndex* pindex, const vector<nameTempProxy>& vName);
    virtual bool ExtractAddress(const CScript& script, string& address);
    virtual void AddToPendingNames(const CTransaction& tx);
    virtual bool IsNameScript(CScript scr);
    virtual bool getNameValue(const string& sName, string& sValue);
    virtual bool DumpToTextFile();
};

bool CTransaction::ReadFromDisk(const CDiskTxPos& postx)
{
    if (!fTxIndex)
        return false;

    CAutoFile file(OpenBlockFile(postx, true), SER_DISK, CLIENT_VERSION);
    CBlockHeader header;
    try {
        file >> header;
        fseek(file.Get(), postx.nTxOffset, SEEK_CUR);
        file >> *this;
    } catch (std::exception& e) {
        return error("%s() : deserialize or I/O error\n%s", __PRETTY_FUNCTION__, e.what());
    }
    return true;
}

CNameVal nameValFromValue(const UniValue& value) {
    string strName = value.get_str();
    unsigned char *strbeg = (unsigned char*)strName.c_str();
    return CNameVal(strbeg, strbeg + strName.size());
}

CNameVal nameValFromString(const string& str) {
    unsigned char *strbeg = (unsigned char*)str.c_str();
    return CNameVal(strbeg, strbeg + str.size());
}

string limitString(const string& inp, unsigned int size, string message = "")
{
    string ret = inp;
    if (inp.size() > size)
    {
        ret.resize(size);
        ret += message;
    }

    return ret;
}

// Calculate at which block will expire.
bool CalculateExpiresAt(CNameRecord& nameRec)
{
    if (nameRec.deleted())
    {
        nameRec.nExpiresAt = 0;
        return true;
    }

    int64_t sum = 0;
    for(unsigned int i = nameRec.nLastActiveChainIndex; i < nameRec.vtxPos.size(); i++)
    {
        CTransaction tx;
        if (!tx.ReadFromDisk(nameRec.vtxPos[i].txPos))
            return error("CalculateExpiresAt() : could not read tx from disk");

        NameTxInfo nti;
        if (!DecodeNameTx(tx, nti))
            return error("CalculateExpiresAt() : %s is not namecoin tx, this should never happen", tx.GetHash().GetHex());

        sum += nti.nRentalDays * 1350; //days to blocks. 1350 is average number of PoS blocks per day
    }

    //limit to INT_MAX value
    sum += nameRec.vtxPos[nameRec.nLastActiveChainIndex].nHeight;
    nameRec.nExpiresAt = sum > INT_MAX ? INT_MAX : sum;

    return true;
}

// Tests if name is active. You can optionaly specify at which height it is/was active.
bool NameActive(CNameDB& dbName, const CNameVal& name, int currentBlockHeight = -1)
{
    CNameRecord nameRec;
    if (!dbName.ReadName(name, nameRec))
        return false;

    if (currentBlockHeight < 0)
        currentBlockHeight = chainActive.Height();

    if (nameRec.deleted()) // last name op was name_delete
        return false;

    return currentBlockHeight <= nameRec.nExpiresAt;
}

bool NameActive(const CNameVal& name, int currentBlockHeight = -1)
{
    CNameDB dbName("r");
    return NameActive(dbName, name, currentBlockHeight);
}

// Returns minimum name operation fee rounded down to cents. Should be used during|before transaction creation.
// If you wish to calculate if fee is enough - use IsNameFeeEnough() function.
// Generaly:  GetNameOpFee() > IsNameFeeEnough().
CAmount GetNameOpFee(const unsigned int& nRentalDays, const int& op)
{
    if (op == OP_NAME_MULTISIG)
        return MIN_MULTISIG_NAME_FEE;
    
    if (op == OP_NAME_DELETE)
        return MIN_TX_FEE;

    CAmount txMinFee = (CAmount)(nRentalDays * NAME_REGISTRATION_DAILY_FEE); // 

    return txMinFee;
}

// scans nameindex.dat and return names with their last CNameIndex
bool CNameDB::ScanNames(const CNameVal& name, unsigned int nMax,
        vector<
            pair<
                CNameVal,
                pair<CNameIndex, int>
            >
        > &nameScan)
{
    Dbc* pcursor = GetCursor();
    if (!pcursor)
        return false;

    unsigned int fFlags = DB_SET_RANGE;
    while (true)
    {
        // Read next record
        CDataStream ssKey(SER_DISK, CLIENT_VERSION);
        if (fFlags == DB_SET_RANGE)
            ssKey << make_pair(string("namei"), name);
        CDataStream ssValue(SER_DISK, CLIENT_VERSION);
        int ret = ReadAtCursor(pcursor, ssKey, ssValue, fFlags);
        fFlags = DB_NEXT;
        if (ret == DB_NOTFOUND)
            break;
        else if (ret != 0)
            return false;

        // Unserialize
        string strType;
        ssKey >> strType;
        if (strType == "namei")
        {
            CNameVal name2;
            ssKey >> name2;
            CNameRecord val;
            ssValue >> val;
            if (val.deleted() || val.vtxPos.empty())
                continue;
            nameScan.push_back(make_pair(name2, make_pair(val.vtxPos.back(), val.nExpiresAt)));
        }

        if (nameScan.size() >= nMax)
            break;
    }
    pcursor->close();
    return true;
}

bool CNameDB::ReadName(const CNameVal& name, CNameRecord& rec)
{
    bool ret = Read(make_pair(std::string("namei"), name), rec);
    int s = rec.vtxPos.size();

     // check if array index is out of array bounds
    if (s > 0 && rec.nLastActiveChainIndex >= s)
    {
        // delete nameindex and kill the application. nameindex should be recreated on next start
        boost::system::error_code err;
        boost::filesystem::remove(GetDataDir() / this->strFile, err);
        LogPrintf("Nameindex is corrupt! It will be recreated on next start.");
        assert(rec.nLastActiveChainIndex < s);
    }
    return ret;
}

CHooks* InitHook()
{
    return new CNamecoinHooks();
}

bool IsNameFeeEnough(const CTransaction& tx, const NameTxInfo& nti, const CBlockIndex* pindexBlock, const CAmount& txFee)
{
    // scan last 10 PoW block for tx fee that matches the one specified in tx
    const CBlockIndex* lastPoW = GetLastBlockIndex(pindexBlock);
    if (lastPoW == NULL) {
        LogPrintf("IsNameFeeEnough(): Error finding lastPoW using GetLastBlockIndex(pindexBlock)");
        return false;
    }
    //LogPrintf("IsNameFeeEnough(): pindexBlock->nHeight = %d, nameSize = %lu, valueSize = %lu, nRentalDays = %d, txFee = %d\n",
    //       lastPoW->nHeight, nti.name.size(), nti.value.size(), nti.nRentalDays, txFee);
    bool txFeePass = false;
    for (int i = 1; i <= 10; i++)
    {
        CAmount netFee = GetNameOpFee(nti.nRentalDays, nti.op);
        //LogPrintf("dDNS Operation: netFee = %d, lastPoW->nHeight = %d\n", netFee, lastPoW->nHeight);
        if (txFee >= netFee)
        {
            txFeePass = true;
            break;
        }
        if (lastPoW != NULL) {
            lastPoW = GetLastBlockIndex(lastPoW->pprev);
        }
        else {
            i = 11;
        }
    }
    return txFeePass;
}

bool CNamecoinHooks::IsNameFeeEnough(const CTransaction& tx, const CAmount& txFee)
{
    if (tx.nVersion != NAMECOIN_TX_VERSION)
        return false;

    NameTxInfo nti;
    if (!DecodeNameTx(tx, nti))
        return false;

    return ::IsNameFeeEnough(tx, nti, chainActive.Tip(), txFee);
}

//returns first name operation. I.e. name_new from chain like name_new->name_update->name_update->...->name_update
bool GetFirstTxOfName(CNameDB& dbName, const CNameVal& name, CTransaction& tx)
{
    CNameRecord nameRec;
    if (!dbName.ReadName(name, nameRec) || nameRec.vtxPos.empty())
        return false;
    CNameIndex& txPos = nameRec.vtxPos[nameRec.nLastActiveChainIndex];

    if (!tx.ReadFromDisk(txPos.txPos))
        return error("GetFirstTxOfName() : could not read tx from disk");
    return true;
}

bool GetLastTxOfName(CNameDB& dbName, const CNameVal& name, CTransaction& tx, CNameRecord& nameRec)
{
    if (!dbName.ReadName(name, nameRec))
        return false;
    if (nameRec.deleted() || nameRec.vtxPos.empty())
        return false;

    CNameIndex& txPos = nameRec.vtxPos.back();

    if (!tx.ReadFromDisk(txPos.txPos))
        return error("GetLastTxOfName() : could not read tx from disk");
    return true;
}

bool GetLastTxOfName(CNameDB& dbName, const CNameVal& name, CTransaction& tx)
{
    CNameRecord nameRec;
    return GetLastTxOfName(dbName, name, tx, nameRec);
}


UniValue sendtoname(const UniValue& params, bool fHelp)
{
    if (fHelp || params.size() < 2 || params.size() > 4)
        throw runtime_error(
            "sendtoname <name> <amount> [comment] [comment-to]\n"
            "<amount> is a real and is rounded to the nearest 0.01"
            + HelpRequiringPassphrase());

    if (IsInitialBlockDownload())
        throw JSONRPCError(RPC_CLIENT_IN_INITIAL_DOWNLOAD, "DarkSilk is downloading blocks...");

    CNameVal name = nameValFromValue(params[0]);
    CAmount nAmount = AmountFromValue(params[1]);

    // Wallet comments
    CWalletTx wtx;
    if (params.size() > 2 && !params[2].isNull() && !params[2].get_str().empty())
        wtx.mapValue["comment"] = params[2].get_str();
    if (params.size() > 3 && !params[3].isNull() && !params[3].get_str().empty())
        wtx.mapValue["to"]      = params[3].get_str();

    string error;
    CDarkSilkAddress address;
    if (!GetNameCurrentAddress(name, address, error))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, error);

    SendMoney(address.Get(), nAmount, wtx);

    UniValue res(UniValue::VOBJ);
    res.push_back(Pair("sending to", address.ToString()));
    res.push_back(Pair("transaction", wtx.GetHash().GetHex()));
    return res;
}

bool GetNameCurrentAddress(const CNameVal& name, CDarkSilkAddress& address, string& error)
{
    CNameDB dbName("r");
    if (!dbName.ExistsName(name))
    {
        error = "Name not found";
        return false;
    }

    CTransaction tx;
    NameTxInfo nti;
    if (!(GetLastTxOfName(dbName, name, tx) && DecodeNameTx(tx, nti, true)))
    {
        error = "Failed to read/decode last name transaction";
        return false;
    }

    address.SetString(nti.strAddress);
    if (!address.IsValid())
    {
        error = "Name contains invalid address"; // this error should never happen, and if it does - this probably means that client blockchain database is corrupted
        return false;
    }

    if (!NameActive(dbName, name))
    {
        stringstream ss;
        ss << "This name have expired. If you still wish to send money to it's last owner you can use this command:\n"
           << "sendtoaddress " << address.ToString() << " <your_amount> ";
        error = ss.str();
        return false;
    }

    return true;
}

UniValue name_list(const UniValue& params, bool fHelp)
{
    if (fHelp || params.size() > 1)
        throw runtime_error(
                "name_list [<name>]\n"
                "list my own names"
                );

    if (IsInitialBlockDownload())
        throw JSONRPCError(RPC_CLIENT_IN_INITIAL_DOWNLOAD, "DarkSilk is downloading blocks...");

    CNameVal nameUniq;
    if (params.size() == 1)
        nameUniq = nameValFromValue(params[0]);

    map<CNameVal, NameTxInfo> mapNames, mapPending;
    GetNameList(nameUniq, mapNames, mapPending);

    UniValue oRes(UniValue::VARR);
    BOOST_FOREACH(const PAIRTYPE(CNameVal, NameTxInfo)& item, mapNames)
    {
        UniValue oName(UniValue::VOBJ);
        oName.push_back(Pair("name", stringFromNameVal(item.second.name)));
        oName.push_back(Pair("value", stringFromNameVal(item.second.value)));
        if (item.second.fIsMine == false)
            oName.push_back(Pair("transferred", true));
        oName.push_back(Pair("address", item.second.strAddress));
        oName.push_back(Pair("expires_in", item.second.nExpiresAt - chainActive.Height()));
        if (item.second.nExpiresAt - chainActive.Height() <= 0)
            oName.push_back(Pair("expired", true));

        oRes.push_back(oName);
    }
    return oRes;
}

// read wallet name txs and extract: name, value, rentalDays, nOut and nExpiresAt.  Based On name_filter
void GetNameList(const CNameVal& nameUniq, std::map<CNameVal, NameTxInfo>& mapNames, std::map<CNameVal, NameTxInfo>& mapPending)
{
    if (IsInitialBlockDownload())
        return;

    CNameDB dbName("r");
    //vector<UniValue> oRes;

    CNameVal name;
    vector<pair<CNameVal, pair<CNameIndex,int> > > nameScan;
    if (!dbName.ScanNames(name, 100000000, nameScan))
        return; // throw JSONRPCError(RPC_WALLET_ERROR, "scan failed");

    pair<CNameVal, pair<CNameIndex,int> > pairScan;
    BOOST_FOREACH(pairScan, nameScan)
    {
        CNameVal name = pairScan.first;
        CNameIndex txName = pairScan.second.first;
        CNameVal value = txName.value;
        int nHeightRegister = txName.nHeight;
        int op = txName.op;

        CNameRecord nameRec;
        if (!dbName.ReadName(pairScan.first, nameRec))
            continue;
        int nExpiresAt = nameRec.nExpiresAt;

        int nRentalDays = nExpiresAt - nHeightRegister;
        NameTxInfo nti(name, value, nRentalDays, op, 0, "");
        CTransaction tx;
        GetLastTxOfName(dbName, name, tx, nameRec);
        if (!DecodeNameTx(tx, nti, true))
            continue;

        nti.nExpiresAt = nExpiresAt;
        mapNames[nti.name] = nti;
    }

    // add all pending names
    BOOST_FOREACH(const PAIRTYPE(CNameVal, set<uint256>) &item, mapNamePending)
    {
        if (!item.second.size())
            continue;

        // TODO (Amir) DDNS, test to see if we can use nLockTime instead of nTime.
        // if there is a set of pending op on a single name - select last one, by nTime
        CTransaction tx;
        uint32_t nTime = 0;
        bool found = false;
        BOOST_FOREACH(const uint256& hash, item.second)
        {
            if (!mempool.exists(hash))
                continue;
            if (mempool.mapTx.find(hash)->GetTx().nLockTime > nTime)
            {
                tx = mempool.mapTx.find(hash)->GetTx();
                nTime = tx.nLockTime;
                found = true;
            }
        }

        if (!found)
            continue;

        NameTxInfo nti;
        if (!DecodeNameTx(tx, nti, true))
            continue;

        if (nameUniq.size() > 0 && nameUniq != nti.name)
            continue;

        mapPending[nti.name] = nti;
    }
}

UniValue name_debug(const UniValue& params, bool fHelp)
{
    if (fHelp || params.size() < 1)
        throw runtime_error(
            "name_debug\n"
            "Dump pending transactions id in the debug file.\n");

    LogPrintf("Pending:\n----------------------------\n");

    {
        LOCK(cs_main);
        BOOST_FOREACH(const PAIRTYPE(CNameVal, set<uint256>) &pairPending, mapNamePending)
        {
            string name = stringFromNameVal(pairPending.first);
            LogPrintf("%s :\n", name);
            uint256 hash;
            BOOST_FOREACH(hash, pairPending.second)
            {
                LogPrintf("    ");
                if (!pwalletMain->mapWallet.count(hash))
                    LogPrintf("foreign ");
                LogPrintf("    %s\n", hash.GetHex());
            }
        }
    }
    LogPrintf("----------------------------\n");
    return true;
}

UniValue name_show(const UniValue& params, bool fHelp)
{
    if (fHelp || params.size() < 1 || params.size() > 2)
        throw runtime_error(
            "name_show <name> [filepath]\n"
            "Show values of a name.\n"
            "If filepath is specified name value will be saved in that file in binary format (file will be overwritten!).\n"
            );

    if (IsInitialBlockDownload())
        throw JSONRPCError(RPC_CLIENT_IN_INITIAL_DOWNLOAD, "DarkSilk is downloading blocks...");

    UniValue oName(UniValue::VOBJ);
    CNameVal name = nameValFromValue(params[0]);
    string sName = stringFromNameVal(name);
    NameTxInfo nti;
    {
        LOCK(cs_main);
        CNameRecord nameRec;
        CNameDB dbName("r");
        if (!dbName.ReadName(name, nameRec))
            throw JSONRPCError(RPC_WALLET_ERROR, "failed to read from name DB");

        if (nameRec.vtxPos.size() < 1)
            throw JSONRPCError(RPC_WALLET_ERROR, "no result returned");

        CTransaction tx;
        if (!tx.ReadFromDisk(nameRec.vtxPos.back().txPos))
            throw JSONRPCError(RPC_WALLET_ERROR, "failed to read from from disk");

        if (!DecodeNameTx(tx, nti, true))
            throw JSONRPCError(RPC_WALLET_ERROR, "failed to decode name");

        oName.push_back(Pair("name", sName));
        string value = stringFromNameVal(nti.value);
        oName.push_back(Pair("value", value));
        oName.push_back(Pair("txid", tx.GetHash().GetHex()));
        oName.push_back(Pair("address", nti.strAddress));
        oName.push_back(Pair("expires_in", nameRec.nExpiresAt - chainActive.Height()));
        oName.push_back(Pair("expires_at", nameRec.nExpiresAt));
        oName.push_back(Pair("time", (boost::int64_t)tx.nLockTime));
        if (nameRec.deleted())
            oName.push_back(Pair("deleted", true));
        else
            if (nameRec.nExpiresAt - chainActive.Height() <= 0)
                oName.push_back(Pair("expired", true));
    }

    if (params.size() > 1)
    {
        string filepath = params[1].get_str();
        ofstream file;
        file.open(filepath.c_str(), ios::out | ios::binary | ios::trunc);
        if (!file.is_open())
            throw JSONRPCError(RPC_PARSE_ERROR, "Failed to open file. Check if you have permission to open it.");

        file.write((const char*)&nti.value[0], nti.value.size());
        file.close();
    }

    return oName;
}

UniValue name_history (const UniValue& params, bool fHelp)
{
    if (fHelp || params.size() < 1 || params.size() > 2)
        throw std::runtime_error (
            "name_history \"name\" ( fullhistory )\n"
            "\nLook up the current and all past data for the given name.\n"
            "\nArguments:\n"
            "1. \"name\"           (string, required) the name to query for\n"
            "2. \"fullhistory\"    (boolean, optional) shows full history, even if name is not active\n"
            "\nResult:\n"
            "[\n"
            "  {\n"
            "    \"txid\": \"xxxx\",            (string) transaction id"
            "    \"time\": xxxxx,               (numeric) transaction time"
            "    \"height\": xxxxx,             (numeric) height of block with this transaction"
            "    \"address\": \"xxxx\",         (string) address to which transaction was sent"
            "    \"address_is_mine\": \"xxxx\", (string) shows \"true\" if this is your address, otherwise not visible"
            "    \"operation\": \"xxxx\",       (string) name operation that was performed in this transaction"
            "    \"days_added\": xxxx,          (numeric) days added (1 day = 1350 blocks) to name expiration time, not visible if 0"
            "    \"value\": xxxx,               (numeric) name value in this transaction; not visible when name_delete was used"
            "  }\n"
            "]\n"
            "\nExamples:\n"
            + HelpExampleCli ("name_history", "\"myname\"")
            + HelpExampleRpc ("name_history", "\"myname\"")
        );

    if (IsInitialBlockDownload())
        throw JSONRPCError(RPC_CLIENT_IN_INITIAL_DOWNLOAD, "DarkSilk is downloading blocks...");

    CNameVal name = nameValFromValue(params[0]);
    bool fFullHistory = false;
    if (params.size() > 1)
        fFullHistory = params[1].get_bool();

    CNameRecord nameRec;
    {
        LOCK(cs_main);
        CNameDB dbName("r");
        if (!dbName.ReadName(name, nameRec))
            throw JSONRPCError(RPC_DATABASE_ERROR, "failed to read from name DB");
    }

    if (nameRec.vtxPos.empty())
        throw JSONRPCError(RPC_DATABASE_ERROR, "record for this name exists, but transaction list is empty");

    if (!fFullHistory && !NameActive(name))
        throw JSONRPCError(RPC_MISC_ERROR, "record for this name exists, but this name is not active");

    UniValue res(UniValue::VARR);
    for (unsigned int i = fFullHistory ? 0 : nameRec.nLastActiveChainIndex; i < nameRec.vtxPos.size(); i++)
    {
        CTransaction tx;
        if (!tx.ReadFromDisk(nameRec.vtxPos[i].txPos))
            throw JSONRPCError(RPC_DATABASE_ERROR, "could not read transaction from disk");

        NameTxInfo nti;
        if (!DecodeNameTx(tx, nti, true))
            throw JSONRPCError(RPC_DATABASE_ERROR, "failed to decode namecoin transaction");

        UniValue obj(UniValue::VOBJ);
        obj.push_back(Pair("txid",             tx.GetHash().ToString()));
        obj.push_back(Pair("time",             (boost::int64_t)tx.nLockTime));
        obj.push_back(Pair("height",           nameRec.vtxPos[i].nHeight));
        obj.push_back(Pair("address",          nti.strAddress));
        if (nti.fIsMine)
            obj.push_back(Pair("address_is_mine",  "true"));
        obj.push_back(Pair("operation",        stringFromOp(nti.op)));
        if (nti.op == OP_NAME_UPDATE || nti.op == OP_NAME_NEW || nti.op == OP_NAME_MULTISIG)
            obj.push_back(Pair("days_added",       nti.nRentalDays));
        if (nti.op == OP_NAME_UPDATE || nti.op == OP_NAME_NEW || nti.op == OP_NAME_MULTISIG)
            obj.push_back(Pair("value",            stringFromNameVal(nti.value)));

        res.push_back(obj);
    }

    return res;
}

UniValue name_mempool (const UniValue& params, bool fHelp)
{
    if (fHelp || params.size() > 0)
        throw std::runtime_error (
            "name_mempool\n"
            "\nList pending name transactions in mempool.\n"
            "\nResult:\n"
            "[\n"
            "  {\n"
            "    \"name\": \"xxxx\",            (string) name"
            "    \"txid\": \"xxxx\",            (string) transaction id"
            "    \"time\": xxxxx,               (numeric) transaction time"
            "    \"address\": \"xxxx\",         (string) address to which transaction was sent"
            "    \"address_is_mine\": \"xxxx\", (string) shows \"true\" if this is your address, otherwise not visible"
            "    \"operation\": \"xxxx\",       (string) name operation that was performed in this transaction"
            "    \"days_added\": xxxx,          (numeric) days added (1 day = 1350 blocks) to name expiration time, not visible if 0"
            "    \"value\": xxxx,               (numeric) name value in this transaction; not visible when name_delete was used"
            "  }\n"
            "]\n"
            "\nExamples:\n"
            + HelpExampleCli ("name_mempool", "" )
            + HelpExampleRpc ("name_mempool", "" )
        );

    UniValue res(UniValue::VARR);
    BOOST_FOREACH(const PAIRTYPE(CNameVal, set<uint256>) &pairPending, mapNamePending)
    {
        string sName = stringFromNameVal(pairPending.first);
        BOOST_FOREACH(const uint256& hash, pairPending.second)
        {
            if (!mempool.exists(hash))
                continue;

            CTransaction tx = mempool.mapTx.find(hash)->GetTx();
            NameTxInfo nti;
            if (!DecodeNameTx(tx, nti, true))
                throw JSONRPCError(RPC_DATABASE_ERROR, "failed to decode namecoin transaction");

            UniValue obj(UniValue::VOBJ);
            obj.push_back(Pair("name",             sName));
            obj.push_back(Pair("txid",             hash.ToString()));
            obj.push_back(Pair("time",             (boost::int64_t)tx.nLockTime));
            obj.push_back(Pair("address",          nti.strAddress));
            if (nti.fIsMine)
                obj.push_back(Pair("address_is_mine",  "true"));
            obj.push_back(Pair("operation",        stringFromOp(nti.op)));
            if (nti.op == OP_NAME_UPDATE || nti.op == OP_NAME_NEW || nti.op == OP_NAME_MULTISIG)
                obj.push_back(Pair("days_added",       nti.nRentalDays));
            if (nti.op == OP_NAME_UPDATE || nti.op == OP_NAME_NEW || nti.op == OP_NAME_MULTISIG)
                obj.push_back(Pair("value",            stringFromNameVal(nti.value)));

            res.push_back(obj);
        }
    }
    return res;
}

// used for sorting in name_filter by nHeight
bool mycompare2 (const UniValue& lhs, const UniValue& rhs)
{
    int pos = 2; //this should exactly match field name position in name_filter

    return lhs[pos].get_int() < rhs[pos].get_int();
}

UniValue name_filter(const UniValue& params, bool fHelp)
{
    if (fHelp || params.size() > 5)
        throw runtime_error(
                "name_filter [[[[[regexp] maxage=0] from=0] nb=0] stat]\n"
                "scan and filter names\n"
                "[regexp] : apply [regexp] on names, empty means all names\n"
                "[maxage] : look in last [maxage] blocks\n"
                "[from] : show results from number [from]\n"
                "[nb] : show [nb] results, 0 means all\n"
                "[stats] : show some stats instead of results\n"
                "name_filter \"\" 5 # list names updated in last 5 blocks\n"
                "name_filter \"^id/\" # list all names from the \"id\" namespace\n"
                "name_filter \"^id/\" 0 0 0 stat # display stats (number of names) on active names from the \"id\" namespace\n"
                );

    if (IsInitialBlockDownload())
        throw JSONRPCError(RPC_CLIENT_IN_INITIAL_DOWNLOAD, "DarkSilk is downloading blocks...");

    string strRegexp;
    int nFrom = 0;
    int nNb = 0;
    int nMaxAge = 0;
    bool fStat = false;
    int nCountFrom = 0;
    int nCountNb = 0;

    if (params.size() > 0)
        strRegexp = params[0].get_str();

    if (params.size() > 1)
        nMaxAge = params[1].get_int();

    if (params.size() > 2)
        nFrom = params[2].get_int();

    if (params.size() > 3)
        nNb = params[3].get_int();

    if (params.size() > 4)
        fStat = (params[4].get_str() == "stat" ? true : false);


    CNameDB dbName("r");
    vector<UniValue> oRes;

    CNameVal name;
    vector<pair<CNameVal, pair<CNameIndex,int> > > nameScan;
    if (!dbName.ScanNames(name, 100000000, nameScan))
        throw JSONRPCError(RPC_WALLET_ERROR, "scan failed");

    // compile regex once
    using namespace boost::xpressive;
    smatch nameparts;
    sregex cregex = sregex::compile(strRegexp);

    pair<CNameVal, pair<CNameIndex,int> > pairScan;
    BOOST_FOREACH(pairScan, nameScan)
    {
        string name = stringFromNameVal(pairScan.first);

        //don't show multisig names
        if (name.length() >= 8 && name.substr(0,8) == "address:")
            continue;

        //don't show multisig names
        if (name.length() >= 6 && name.substr(0,6) == "txsig:")
            continue;

        // regexp
        if(strRegexp != "" && !regex_search(name, nameparts, cregex))
            continue;

        CNameIndex txName = pairScan.second.first;

        CNameRecord nameRec;
        if (!dbName.ReadName(pairScan.first, nameRec))
            continue;

        // max age
        int nHeight = nameRec.vtxPos[nameRec.nLastActiveChainIndex].nHeight;
        if(nMaxAge != 0 && chainActive.Height() - nHeight >= nMaxAge)
            continue;

        // from limits
        nCountFrom++;
        if(nCountFrom < nFrom + 1)
            continue;

        UniValue oName(UniValue::VOBJ);
        if (!fStat) {
            oName.push_back(Pair("name", name));

            string value = stringFromNameVal(txName.value);
            oName.push_back(Pair("value", limitString(value, 300, "\n...(value too large - use name_show to see full value)")));

            oName.push_back(Pair("registered_at", nHeight)); // pos = 2 in comparison function (above name_filter)

            int nExpiresIn = nameRec.nExpiresAt - chainActive.Height();
            oName.push_back(Pair("expires_in", nExpiresIn));
            if (nExpiresIn <= 0)
                oName.push_back(Pair("expired", true));
        }
        oRes.push_back(oName);

        nCountNb++;
        // nb limits
        if(nNb > 0 && nCountNb >= nNb)
            break;
    }

    UniValue oRes2(UniValue::VARR);
    if (!fStat)
    {
        std::sort(oRes.begin(), oRes.end(), mycompare2); //sort by nHeight
        BOOST_FOREACH(const UniValue& res, oRes)
            oRes2.push_back(res);
    }
    else
    {
        UniValue oStat(UniValue::VOBJ);
        oStat.push_back(Pair("blocks",    chainActive.Height()));
        oStat.push_back(Pair("count",     (int)oRes2.size()));
        //oStat.push_back(Pair("sha256sum", SHA256(oRes), true));
        return oStat;
    }

    return oRes2;
}

UniValue name_scan(const UniValue& params, bool fHelp)
{
    if (fHelp || params.size() > 3)
        throw runtime_error(
                "name_scan [start-name] [max-returned] [max-value-length=-1]\n"
                "Scan all names, starting at start-name and returning a maximum number of entries (default 500)\n"
                "You can also control the length of shown value (0 = full value)\n"
                );

    if (IsInitialBlockDownload())
        throw JSONRPCError(RPC_CLIENT_IN_INITIAL_DOWNLOAD, "DarkSilk is downloading blocks...");

    CNameVal name;
    string strSearchName = "";
    unsigned int nMax = 500;
    unsigned int mMaxShownValue = 96;
    if (params.size() > 0)
    {
        name = nameValFromValue(params[0]);
        strSearchName = params[0].get_str();
    }

    if (params.size() > 1)
        nMax = params[1].get_int();

    if (params.size() > 2)
        mMaxShownValue = params[2].get_int();

    CNameDB dbName("r");
    UniValue oRes(UniValue::VARR);

    vector<pair<CNameVal, pair<CNameIndex,int> > > nameScan;
    if (!dbName.ScanNames(name, nMax, nameScan))
        throw JSONRPCError(RPC_WALLET_ERROR, "scan failed");

    pair<CNameVal, pair<CNameIndex,int> > pairScan;
    BOOST_FOREACH(pairScan, nameScan)
    {
        string ddnsName = stringFromNameVal(pairScan.first);
        // search for input string
        if (ddnsName.size() > strSearchName.size() && ddnsName.find(strSearchName) != std::string::npos)
        {
            UniValue oName(UniValue::VOBJ);
            string ddnsName = stringFromNameVal(pairScan.first);
            oName.push_back(Pair("name", ddnsName));

            CNameIndex txName = pairScan.second.first;
            int nExpiresAt    = pairScan.second.second;
            CNameVal value = txName.value;
            string sValue = stringFromNameVal(value);

            oName.push_back(Pair("value", limitString(sValue, mMaxShownValue, "\n...(value too large - use name_show to see full value)")));
            oName.push_back(Pair("expires_in", nExpiresAt - chainActive.Height()));
            if (nExpiresAt - chainActive.Height() <= 0)
                oName.push_back(Pair("expired", true));

            oRes.push_back(oName);
        }
    }

    return oRes;
}

bool createNameScript(CScript& nameScript, const CNameVal& name, const CNameVal& value, int nRentalDays, int op, string& err_msg)
{
    if (op == OP_NAME_DELETE)
    {
        nameScript << op << OP_DROP << name << OP_DROP;
        return true;
    }

    NameTxInfo nti(name, value, nRentalDays, op, -1, err_msg);
    if (!checkNameValues(nti))
    {
        err_msg = nti.err_msg;
        return false;
    }

    vector<unsigned char> vchRentalDays = CScriptNum(nRentalDays).getvch();

    //add name and rental days
    nameScript << op << OP_DROP << name << vchRentalDays << OP_2DROP;

    // split value in 520 bytes chunks and add it to script
    {
        unsigned int nChunks = ceil(value.size() / 520.0);

        for (unsigned int i = 0; i < nChunks; i++)
        {   // insert data
            vector<unsigned char>::const_iterator sliceBegin = value.begin() + i*520;
            vector<unsigned char>::const_iterator sliceEnd = min(value.begin() + (i+1)*520, value.end());
            vector<unsigned char> vchSubValue(sliceBegin, sliceEnd);
            nameScript << vchSubValue;
        }

            //insert end markers
        for (unsigned int i = 0; i < nChunks / 2; i++)
            nameScript << OP_2DROP;
        if (nChunks % 2 != 0)
            nameScript << OP_DROP;
    }
    return true;
}

bool IsWalletLocked(NameTxReturn& ret)
{
    if (pwalletMain->IsLocked())
    {
        ret.err_code = RPC_WALLET_UNLOCK_NEEDED;
        ret.err_msg = "Error: Please enter the wallet passphrase with walletpassphrase first.";
        return true;
    }
    return false;
}

static CNameVal CNameValToLowerCase(const CNameVal& nameVal) {
    string strNameVal;
    CNameVal::const_iterator vi = nameVal.begin();
    while (vi != nameVal.end()) 
    {
        strNameVal += std::tolower(*vi);
        vi++;
    }
    CNameVal returnNameVal(strNameVal.begin(), strNameVal.end());
    return returnNameVal;
}

UniValue name_new(const UniValue& params, bool fHelp)
{
    if (fHelp || params.size() < 2 || params.size() > 5)
        throw runtime_error(
                "name_new <name> <value> <days> [toaddress] [valueAsFilepath]\n"
                "Creates new key->value pair which expires after specified number of days.\n"
                "Cost is square root of (1% of last PoW + 1% per year of last PoW)."
                "If [valueAsFilepath] is non-zero it will interpret <value> as a filepath and try to write file contents in binary format.\n"
                + HelpRequiringPassphrase());

    // make sure the DDNS entry is all lowercase.
    CNameVal name = CNameValToLowerCase(nameValFromValue(params[0]));
    CNameVal value = nameValFromValue(params[1]);
    int nRentalDays = params[2].get_int();
    string strAddress = "";
    if (params.size() == 4)
        strAddress = params[3].get_str();

    bool fValueAsFilepath = false;
    if (params.size() > 4)
        fValueAsFilepath = (params[4].get_int() != 0);

    NameTxReturn ret = name_operation(OP_NAME_NEW, name, value, nRentalDays, strAddress, fValueAsFilepath);
    if (!ret.ok)
        throw JSONRPCError(ret.err_code, ret.err_msg);
    return ret.hex.GetHex();
}

UniValue name_update(const UniValue& params, bool fHelp)
{
    if (fHelp || params.size() < 2 || params.size() > 5)
        throw runtime_error(
                "name_update <name> <value> <days> [toaddress] [valueAsFilepath]\n"
                "Update name value, add days to expiration time and possibly transfer a name to diffrent address.\n"
                "If [valueAsFilepath] is non-zero it will interpret <value> as a filepath and try to write file contents in binary format.\n"
                + HelpRequiringPassphrase());

    // make sure the DDNS entry is all lowercase.
    CNameVal name = CNameValToLowerCase(nameValFromValue(params[0]));
    CNameVal value = nameValFromValue(params[1]);
    int nRentalDays = params[2].get_int();
    string strAddress = "";
    if (params.size() == 4)
        strAddress = params[3].get_str();

    bool fValueAsFilepath = false;
    if (params.size() > 4)
        fValueAsFilepath = (params[4].get_int() != 0);

    NameTxReturn ret = name_operation(OP_NAME_UPDATE, name, value, nRentalDays, strAddress, fValueAsFilepath);
     if (!ret.ok)
        throw JSONRPCError(ret.err_code, ret.err_msg);
    return ret.hex.GetHex();
}

UniValue name_delete(const UniValue& params, bool fHelp)
{
    if (fHelp || params.size() != 1)
        throw runtime_error(
                "name_delete <name>\nDelete a name if you own it. Others may do name_new after this command."
                + HelpRequiringPassphrase());

    // make sure the DDNS entry is all lowercase.
    CNameVal name = CNameValToLowerCase(nameValFromValue(params[0]));

    NameTxReturn ret = name_operation(OP_NAME_DELETE, name, CNameVal(), 0, "");
    if (!ret.ok)
        throw JSONRPCError(ret.err_code, ret.err_msg);
    return ret.hex.GetHex();

}

NameTxReturn name_operation(const int op, const CNameVal& name, CNameVal value, const int nRentalDays, const string strAddress, bool fValueAsFilepath)
{
    NameTxReturn ret;
    ret.err_code = RPC_INTERNAL_ERROR; // default value in case of abnormal exit
    ret.err_msg = "unkown error";
    ret.ok = false;

    if ((op == OP_NAME_NEW || op == OP_NAME_UPDATE || op == OP_NAME_MULTISIG) && value.empty())
    {
        ret.err_msg = "value must not be empty";
        return ret;
    }

    // currently supports only new, update, multisig and delete operations.
    if (op != OP_NAME_NEW && op != OP_NAME_UPDATE && op != OP_NAME_DELETE && op != OP_NAME_MULTISIG)
    {
        ret.err_msg = "illegal name op";
        return ret;
    }

    if (fValueAsFilepath)
    {
        string filepath = stringFromNameVal(value);
        std::ifstream ifs;
        ifs.open(filepath.c_str(), std::ios::binary | std::ios::ate);
        if (!ifs)
        {
            ret.err_msg = "failed to open file";
            return ret;
        }
        std::streampos fileSize = ifs.tellg();
        if (fileSize > MAX_VALUE_LENGTH)
        {
            ret.err_msg = "file is larger than maximum allowed size";
            return ret;
        }

        ifs.clear();
        ifs.seekg(0, std::ios::beg);

        value.resize(fileSize);
        if (!ifs.read(reinterpret_cast<char*>(&value[0]), fileSize))
        {
            ret.err_msg = "failed to read file";
            return ret;
        }
    }

    if (IsInitialBlockDownload())
    {
        ret.err_code = RPC_CLIENT_IN_INITIAL_DOWNLOAD;
        ret.err_msg = "DarkSilk is downloading blocks...";
        return ret;
    }

    
    if (op == OP_NAME_MULTISIG)
    {  
        if (!AddressMatchesPubKey(name, value, ret.err_msg))
        {
            //ret.err_msg = "Public key and address are invalid or do not match! " + stringFromNameVal(name) + " " + stringFromNameVal(value);
            return ret;
        }
    }

    if (IsWalletLocked(ret))
        return ret;

    CMutableTransaction tmpTx;
    tmpTx.nVersion = NAMECOIN_TX_VERSION;
    CWalletTx wtx(pwalletMain, tmpTx);
    stringstream ss;
    CScript scriptPubKey;

    {
        LOCK2(cs_main, pwalletMain->cs_wallet);

        // wait until other name operation on this name are completed
        if (mapNamePending.count(name) && mapNamePending[name].size())
        {
            ss << "there are " << mapNamePending[name].size() <<
                  " pending operations on that name, including " << mapNamePending[name].begin()->GetHex();
            ret.err_msg = ss.str();
            return ret;
        }

        // check if op can be aplied to name remaining time
        if (NameActive(name))
        {
            if (op == OP_NAME_NEW || op == OP_NAME_MULTISIG)
            {
                ret.err_msg = stringFromOp(op) + " on an unexpired name";
                return ret;
            }
        }
        else
        {
            if (op == OP_NAME_UPDATE || op == OP_NAME_DELETE)
            {
                ret.err_msg = stringFromOp(op) + " on an unexpired name";
                return ret;
            }
        }

        // grab last tx in name chain and check if it can be spent by us
        CWalletTx wtxIn = CWalletTx();
        if (op == OP_NAME_UPDATE || op == OP_NAME_DELETE)
        {
            CNameDB dbName("r");
            CTransaction prevTx;
            if (!GetLastTxOfName(dbName, name, prevTx))
            {
                ret.err_msg = "could not find tx with this name";
                return ret;
            }

            uint256 wtxInHash = prevTx.GetHash();
            if (!pwalletMain->mapWallet.count(wtxInHash))
            {
                ret.err_msg = "this name tx is not in your wallet: " + wtxInHash.GetHex();
                return ret;
            }

            wtxIn = pwalletMain->mapWallet[wtxInHash];
            int nTxOut = IndexOfNameOutput(wtxIn);

            if (::IsMine(*pwalletMain, wtxIn.vout[nTxOut].scriptPubKey) != ISMINE_SPENDABLE)
            {
                ret.err_msg = "this name tx is not yours or is not spendable: " + wtxInHash.GetHex();
                return ret;
            }
        }

        // create namescript
        CScript nameScript;
        string prevMsg = ret.err_msg;
        if (!createNameScript(nameScript, name, value, nRentalDays, op, ret.err_msg))
        {
            if (prevMsg == ret.err_msg)  // in case error message not changed, but error still occurred
                ret.err_msg = "failed to create name script";
            return ret;
        }

        // add destination to namescript
        if ((op == OP_NAME_UPDATE || op == OP_NAME_NEW || op == OP_NAME_MULTISIG) && strAddress != "")
        {
            CDarkSilkAddress address(strAddress);
            if (!address.IsValid())
            {
                ret.err_code = RPC_INVALID_ADDRESS_OR_KEY;
                ret.err_msg = "DarkSilk address is invalid";
                return ret;
            }
            scriptPubKey = GetScriptForDestination(address.Get());
        }
        else
        {
            CPubKey vchPubKey;
            if(!pwalletMain->GetKeyFromPool(vchPubKey))
            {
                ret.err_msg = "failed to get key from pool";
                return ret;
            }
            scriptPubKey = GetScriptForDestination(vchPubKey.GetID());
        }
        nameScript += scriptPubKey;

        // verify namescript
        NameTxInfo nti;
        if (!DecodeNameScript(nameScript, nti))
        {
            ret.err_msg = nti.err_msg;
            return ret;
        }

        // set fee and send!
        CAmount nameFee = GetNameOpFee(nRentalDays, op);
        CAmount minAmount = MIN_TXOUT_AMOUNT;
        if (op == OP_NAME_MULTISIG)
            minAmount = (CAmount)(MIN_TXOUT_AMOUNT/10);

        SendName(nameScript, minAmount, wtx, wtxIn, nameFee);
    }

    //success! collect info and return
    CTxDestination address;
    if (ExtractDestination(scriptPubKey, address))
    {
        ret.address = CDarkSilkAddress(address).ToString();
    }

    ret.hex = wtx.GetHash();
    ret.ok = true;
    
    return ret;
}

bool createNameIndexFile()
{
    LogPrintf("Scanning blockchain for names to create fast index...\n");
    
    CNameDB dbName("cr+");
    
    if (!fTxIndex)
        return error("createNameIndexFile() : transaction index not available");

    int maxHeight = chainActive.Height();
    for (int nHeight=0; nHeight<=maxHeight; nHeight++)
    {
        CBlockIndex* pindex = chainActive[nHeight];
        CBlock block;
        if (!ReadBlockFromDisk(block, pindex))
            return error("createNameIndexFile() : *** ReadBlockFromDisk failed at %d, hash=%s", pindex->nHeight, pindex->GetBlockHash().ToString());

        // collect name tx from block
        vector<nameTempProxy> vName;
        CDiskTxPos pos(pindex->GetBlockPos(), GetSizeOfCompactSize(block.vtx.size())); // start position
        for (unsigned int i=0; i<block.vtx.size(); i++)
        {
            const CTransaction& tx = block.vtx[i];
            if (tx.IsCoinBase())
            {
                pos.nTxOffset += ::GetSerializeSize(tx, SER_DISK, CLIENT_VERSION);  // set next tx position
                continue;
            }

            // calculate tx fee
            CAmount input = 0;
            BOOST_FOREACH(const CTxIn& txin, tx.vin)
            {
                CTransaction txPrev;
                uint256 hashBlock = uint256S("0");
                if (!GetTransaction(txin.prevout.hash, txPrev, Params().GetConsensus(), hashBlock, true))
                    return error("createNameIndexFile() : prev transaction not found");

                input += txPrev.vout[txin.prevout.n].nValue;
            }
            CAmount fee = input - tx.GetValueOut();

            hooks->CheckInputs(tx, pindex, vName, pos, fee);                    // collect valid name tx to vName
            pos.nTxOffset += ::GetSerializeSize(tx, SER_DISK, CLIENT_VERSION);  // set next tx position
        }

        // execute name operations, if any
        hooks->ConnectBlock(pindex, vName);
    }
    return true;
}

// read name tx and extract: name, value and rentalDays
// optionaly it can extract destination address and check if tx is mine (note: it does not check if address is valid)
bool DecodeNameTx(const CTransaction& tx, NameTxInfo& nti, bool checkAddressAndIfIsMine /* = false */)
{
    if (tx.nVersion != NAMECOIN_TX_VERSION)
        return false;

    bool found = false;
    for (unsigned int i = 0; i < tx.vout.size(); i++)
    {
        const CTxOut& out = tx.vout[i];
        NameTxInfo ntiTmp;
        CScript::const_iterator pc = out.scriptPubKey.begin();
        if (DecodeNameScript(out.scriptPubKey, ntiTmp, pc))
        {
            // If more than one name op, fail
            if (found)
                return false;

            nti = ntiTmp;
            nti.nOut = i;

            if (checkAddressAndIfIsMine)
            {
                //read address
                CTxDestination address;
                CScript scriptPubKey(pc, out.scriptPubKey.end());
                if (!ExtractDestination(scriptPubKey, address))
                    nti.strAddress = "";
                nti.strAddress = CDarkSilkAddress(address).ToString();

                // check if this is mine destination
                nti.fIsMine = IsMine(*pwalletMain, address) == ISMINE_SPENDABLE;
            }

            found = true;
        }
    }

    if (found) nti.err_msg = "";
    return found;
}

int IndexOfNameOutput(const CTransaction& tx)
{
    NameTxInfo nti;
    if (!DecodeNameTx(tx, nti))
        throw runtime_error("IndexOfNameOutput() : name output not found");
    return nti.nOut;
}

void CNamecoinHooks::AddToPendingNames(const CTransaction& tx)
{
    if (tx.nVersion != NAMECOIN_TX_VERSION)
        return;

    CCoins coins;
    if (pcoinsTip->GetCoins(tx.GetHash(), coins)) // try to ignore coins that are in blockchain
        return;

    if (tx.vout.size() < 1)
    {
        error("AddToPendingNames() : no output in tx %s\n", tx.ToString());
        return;
    }

    NameTxInfo nti;
    if (!DecodeNameTx(tx, nti))
    {
        error("AddToPendingNames() : could not decode name script in tx %s", tx.ToString());
        return;
    }

    mapNamePending[nti.name].insert(tx.GetHash());
    LogPrintf("AddToPendingNames(): added %s %s from tx %s", stringFromOp(nti.op), stringFromNameVal(nti.name), tx.ToString());
}

// Checks name tx and save name data to vName if valid
// returns true if: (tx is valid name tx) OR (tx is not a name tx)
// returns false if tx is invalid name tx
bool CNamecoinHooks::CheckInputs(const CTransaction& tx, const CBlockIndex* pindexBlock, vector<nameTempProxy> &vName, const CDiskTxPos& pos, const CAmount& txFee)
{
    if (tx.nVersion != NAMECOIN_TX_VERSION)
        return true;

    //read name tx
    NameTxInfo nti;
    if (!DecodeNameTx(tx, nti))
    {
        if (pindexBlock->nHeight > RELEASE_HEIGHT)
            return error("CheckInputsHook() : could not decode namecoin tx %s in block %d", tx.GetHash().GetHex(), pindexBlock->nHeight);
        return false;
    }

    CNameVal name = nti.name;
    string sName = stringFromNameVal(name);
    string info = str( boost::format("name %s, tx=%s, block=%d, value=%s") %
        sName % tx.GetHash().GetHex() % pindexBlock->nHeight % stringFromNameVal(nti.value));

    //check if last known tx on this name matches any of inputs of this tx
    CNameDB dbName("r");
    CNameRecord nameRec;
    if (dbName.ExistsName(name) && !dbName.ReadName(name, nameRec))
        return error("CheckInputsHook() : failed to read from name DB for %s", info);

    bool found = false;
    NameTxInfo prev_nti;
    if (!nameRec.vtxPos.empty() && !nameRec.deleted())
    {
        CTransaction lastKnownNameTx;
        if (!lastKnownNameTx.ReadFromDisk(nameRec.vtxPos.back().txPos))
            return error("CheckInputsHook() : failed to read from name DB for %s", info);
        uint256 lasthash = lastKnownNameTx.GetHash();
        if (!DecodeNameTx(lastKnownNameTx, prev_nti))
            return error("CheckInputsHook() : Failed to decode existing previous name tx for %s. Your blockchain or nameindex.dat may be corrupt.", info);

        for (unsigned int i = 0; i < tx.vin.size(); i++) //this scans all scripts of tx.vin
        {
            if (tx.vin[i].prevout.hash != lasthash)
                continue;
            found = true;
            break;
        }
    }

    switch (nti.op)
    {
        case OP_NAME_NEW:
        {
            //scan last 10 PoW block for tx fee that matches the one specified in tx
            if (!::IsNameFeeEnough(tx, nti, pindexBlock, txFee))
            {
                if (pindexBlock->nHeight > RELEASE_HEIGHT)
                    return error("CheckInputsHook() : rejected name_new because not enough fee for %s", info);
                return false;
            }

            if (NameActive(dbName, name, pindexBlock->nHeight))
            {
                if (pindexBlock->nHeight > RELEASE_HEIGHT)
                    return error("CheckInputsHook() : name_new on an unexpired name for %s", info);
                return false;
            }
            break;
        }
        case OP_NAME_MULTISIG:
        {
            if (txFee < (CAmount)(MIN_TXOUT_AMOUNT/10))
            {
                if (pindexBlock->nHeight > RELEASE_HEIGHT)
                    return error("CheckInputsHook() : rejected name_multisig because not enough fee for %s", info);
                return false;
            }

            if (NameActive(dbName, name, pindexBlock->nHeight))
            {
                if (pindexBlock->nHeight > RELEASE_HEIGHT)
                    return error("CheckInputsHook() : name_multisig on an unexpired name for %s", info);
                return false;
            }
            break;
        }
        case OP_NAME_UPDATE:
        {
            //scan last 10 PoW block for tx fee that matches the one specified in tx
            if (!::IsNameFeeEnough(tx, nti, pindexBlock, txFee))
            {
                if (pindexBlock->nHeight > RELEASE_HEIGHT)
                    return error("CheckInputsHook() : rejected name_update because not enough fee for %s", info);
                return false;
            }

            if (!found || (prev_nti.op != OP_NAME_NEW && prev_nti.op != OP_NAME_UPDATE))
                return error("name_update without previous new or update tx for %s", info);

            if (prev_nti.name != name)
                return error("CheckInputsHook() : name_update name mismatch for %s", info);

            if (!NameActive(dbName, name, pindexBlock->nHeight))
                return error("CheckInputsHook() : name_update on an unexpired name for %s", info);
            break;
        }
        case OP_NAME_DELETE:
        {
            if (!found || (prev_nti.op != OP_NAME_NEW && prev_nti.op != OP_NAME_UPDATE))
                return error("name_delete without previous new or update tx, for %s", info);

            if (prev_nti.name != name)
                return error("CheckInputsHook() : name_delete name mismatch for %s", info);

            if (!NameActive(dbName, name, pindexBlock->nHeight))
                return error("CheckInputsHook() : name_delete on expired name for %s", info);
            break;
        }
        default:
            return error("CheckInputsHook() : unknown name operation for %s", info);
    }

    // all checks passed - record tx information to vName. It will be sorted by nTime and writen to nameindex.dat at the end of ConnectBlock
    CNameIndex txPos2;
    txPos2.nHeight = pindexBlock->nHeight;
    txPos2.value = nti.value;
    txPos2.txPos = pos;

    nameTempProxy tmp;
    tmp.nTime = tx.nLockTime;
    tmp.name = name;
    tmp.op = nti.op;
    tmp.hash = tx.GetHash();
    tmp.ind = txPos2;

    vName.push_back(tmp);
    return true;
}

bool CNamecoinHooks::DisconnectInputs(const CTransaction& tx)
{
    if (tx.nVersion != NAMECOIN_TX_VERSION)
        return true;

    NameTxInfo nti;
    if (!DecodeNameTx(tx, nti))
        return error("DisconnectInputsHook() : could not decode namecoin tx");

    {
        CNameDB dbName("cr+");
        dbName.TxnBegin();

        CNameRecord nameRec;
        if (!dbName.ReadName(nti.name, nameRec))
            return error("DisconnectInputsHook() : failed to read from name DB");

        // vtxPos might be empty if we pruned expired transactions.  However, it should normally still not
        // be empty, since a reorg cannot go that far back.  Be safe anyway and do not try to pop if empty.
        if (nameRec.vtxPos.size() > 0)
        {
            // check if tx matches last tx in nameindex.dat
            CTransaction lastTx;
            lastTx.ReadFromDisk(nameRec.vtxPos.back().txPos);
            assert(lastTx.GetHash() == tx.GetHash());

            // remove tx
            nameRec.vtxPos.pop_back();

            if (nameRec.vtxPos.size() == 0) // delete empty record
                return dbName.EraseName(nti.name);

            // if we have deleted name_new - recalculate Last Active Chain Index
            if (nti.op == OP_NAME_NEW || nti.op == OP_NAME_MULTISIG)
                for (int i = nameRec.vtxPos.size() - 1; i >= 0; i--)
                    if (nameRec.vtxPos[i].op == OP_NAME_NEW || nameRec.vtxPos[i].op == OP_NAME_MULTISIG)
                    {
                        nameRec.nLastActiveChainIndex = i;
                        break;
                    }
        }
        else
            return dbName.EraseName(nti.name); // delete empty record

        if (!CalculateExpiresAt(nameRec))
            return error("DisconnectInputsHook() : failed to calculate expiration time before writing to name DB");
        if (!dbName.WriteName(nti.name, nameRec))
            return error("DisconnectInputsHook() : failed to write to name DB");

        dbName.TxnCommit();
    }

    return true;
}

string stringFromOp(int op)
{
    switch (op)
    {
        case OP_NAME_UPDATE:
            return "Update Name";
        case OP_NAME_NEW:
            return "New Name";
        case OP_NAME_DELETE:
            return "Delete Name";
        case OP_NAME_MULTISIG:
            return "Multisig Name";
        default:
            return "<unknown name op>";
    }
}

bool CNamecoinHooks::ExtractAddress(const CScript& script, string& address)
{
    NameTxInfo nti;
    if (!DecodeNameScript(script, nti))
        return false;

    string strOp = stringFromOp(nti.op);
    address = strOp + ": " + stringFromNameVal(nti.name);
    return true;
}

// Executes name operations in vName and writes result to nameindex.dat.
// NOTE: the block should already be written to blockchain by now - otherwise this may fail.
bool CNamecoinHooks::ConnectBlock(CBlockIndex* pindex, const vector<nameTempProxy> &vName)
{
    if (vName.empty())
        return true;

    // All of these name ops should succed. If there is an error - nameindex.dat is probably corrupt.
    CNameDB dbName("r+");
    set<CNameVal> sNameNew;

    BOOST_FOREACH(const nameTempProxy& i, vName)
    {
        CNameRecord nameRec;
        if (dbName.ExistsName(i.name) && !dbName.ReadName(i.name, nameRec))
            return error("ConnectBlockHook() : failed to read from name DB");

        dbName.TxnBegin();

        // only first name_new for same name in same block will get written
        if  ((i.op == OP_NAME_NEW || i.op == OP_NAME_MULTISIG) && sNameNew.count(i.name))
            continue;

        nameRec.vtxPos.push_back(i.ind); // add

        // if starting new chain - save position of where it starts
        if (i.op == OP_NAME_NEW || i.op == OP_NAME_MULTISIG)
            nameRec.nLastActiveChainIndex = nameRec.vtxPos.size()-1;

        // limit to 1000 tx per name or a full single chain - whichever is larger
        static size_t maxSize = 0;
        if(maxSize == 0)
            GetArg("-nameindexchainsize", NAMEINDEX_CHAIN_SIZE);

        if (nameRec.vtxPos.size() > maxSize &&
            nameRec.vtxPos.size() - nameRec.nLastActiveChainIndex + 1 <= maxSize)
        {
            int d = nameRec.vtxPos.size() - maxSize; // number of elements to delete
            nameRec.vtxPos.erase(nameRec.vtxPos.begin(), nameRec.vtxPos.begin() + d);
            nameRec.nLastActiveChainIndex -= d; // move last index backwards by d elements
            assert(nameRec.nLastActiveChainIndex >= 0);
        }

        // save name op
        nameRec.vtxPos.back().op = i.op;

        if (!CalculateExpiresAt(nameRec))
            return error("ConnectBlockHook() : failed to calculate expiration time before writing to name DB for %s", i.hash.GetHex());
        if (!dbName.WriteName(i.name, nameRec))
            return error("ConnectBlockHook() : failed to write to name DB");
        if  (i.op == OP_NAME_NEW || i.op == OP_NAME_MULTISIG)
            sNameNew.insert(i.name);
        LogPrintf("ConnectBlockHook(): writing %s %s in block %d to ddns.dat\n", stringFromOp(i.op), stringFromNameVal(i.name), pindex->nHeight);

        {
            // remove from pending names list
            LOCK(cs_main);
            map<CNameVal, set<uint256> >::iterator mi = mapNamePending.find(i.name);
            if (mi != mapNamePending.end())
            {
                mi->second.erase(i.hash);
                if (mi->second.empty())
                    mapNamePending.erase(i.name);
            }
        }
        if (!dbName.TxnCommit())
            return error("ConnectBlockHook(): failed to write %s to name DB", stringFromNameVal(i.name));
    }

    return true;
}

bool CNamecoinHooks::IsNameScript(CScript scr)
{
    NameTxInfo nti;
    return DecodeNameScript(scr, nti);
}

bool CNamecoinHooks::getNameValue(const string& sName, string& sValue)
{
    CNameVal name = nameValFromString(sName);
    CNameDB dbName("r");
    if (!dbName.ExistsName(name))
        return false;

    CTransaction tx;
    NameTxInfo nti;
    if (!(GetLastTxOfName(dbName, name, tx) && DecodeNameTx(tx, nti, true)))
        return false;

    if (!NameActive(dbName, name))
        return false;

    sValue = stringFromNameVal(nti.value);

    return true;
}

bool GetNameValue(const CNameVal& name, CNameVal& value)
{
    CNameDB dbName("r");
    CNameRecord nameRec;

    if (!NameActive(name))
        return false;
    if (!dbName.ReadName(name, nameRec))
        return false;
    if (nameRec.vtxPos.empty())
        return false;

    value = nameRec.vtxPos.back().value;
    return true;
}

bool CNamecoinHooks::DumpToTextFile()
{
    CNameDB dbName("r");
    return dbName.DumpToTextFile();
}


bool CNameDB::DumpToTextFile()
{
    ofstream myfile ("example111.txt");
    if (!myfile.is_open())
        return false;

    Dbc* pcursor = GetCursor();
    if (!pcursor)
        return false;

    CNameVal name;
    unsigned int fFlags = DB_SET_RANGE;
    while (true)
    {
        // Read next record
        CDataStream ssKey(SER_DISK, CLIENT_VERSION);
        if (fFlags == DB_SET_RANGE)
            ssKey << make_pair(string("namei"), name);
        CDataStream ssValue(SER_DISK, CLIENT_VERSION);
        int ret = ReadAtCursor(pcursor, ssKey, ssValue, fFlags);
        fFlags = DB_NEXT;
        if (ret == DB_NOTFOUND)
            break;
        else if (ret != 0)
            return false;

        // Unserialize
        string strType;
        ssKey >> strType;
        if (strType == "namei")
        {
            CNameVal name2;
            ssKey >> name2;
            CNameRecord val;
            ssValue >> val;
            if (val.vtxPos.empty())
                continue;

            myfile << "name =  " << stringFromNameVal(name2) << "\n";
            myfile << "nExpiresAt " << val.nExpiresAt << "\n";
            myfile << "nLastActiveChainIndex " << val.nLastActiveChainIndex << "\n";
            myfile << "vtxPos:\n";
            for (unsigned int i = 0; i < val.vtxPos.size(); i++)
            {
                myfile << "    nHeight = " << val.vtxPos[i].nHeight << "\n";
                myfile << "    op = " << val.vtxPos[i].op << "\n";
                myfile << "    value = " << stringFromNameVal(val.vtxPos[i].value) << "\n";
            }
            myfile << "\n\n";
        }
    }
    pcursor->close();
    myfile.close();
    return true;
}

bool SignNameSignature(const CKeyStore& keystore, const CTransaction& txFrom, CMutableTransaction& txTo, const unsigned int nIn, const CScript& scriptDDNS, const int nHashType)
{
    assert(nIn < txTo.vin.size());
    CTxIn& txin = txTo.vin[nIn];
    assert(txin.prevout.n < txFrom.vout.size());
    const CTxOut& txout = txFrom.vout[txin.prevout.n];

    // Leave out the signature from the hash, since a signature can't sign itself.
    // The checksig op will also drop the signatures from its hash.

    uint256 hash = SignatureHash(scriptDDNS, txTo, nIn, nHashType);

    CScript scriptPubKey;
    if (!RemoveNameScriptPrefix(scriptDDNS, scriptPubKey))
        return error(strprintf("SignNameSignature() failed to remove name script prefix scriptSig=%s", txout.scriptPubKey.ToString()).c_str());

    txnouttype whichType;
    if (!Solver(keystore, txout.scriptPubKey, hash, nHashType, txin.scriptSig, whichType))
        return error(strprintf("SignNameSignature() failed to remove name script prefix scriptSig=%s", txout.scriptPubKey.ToString()).c_str());

    // Test solution
    return VerifyScript(txin.scriptSig, txout.scriptPubKey, STANDARD_SCRIPT_VERIFY_FLAGS, MutableTransactionSignatureChecker(&txTo, nIn), NULL, txTo.nVersion == NAMECOIN_TX_VERSION);
}

std::string MultiSigGetPubKeyFromAddress(const std::string& strAddress)
{
    std::string strErrorMessage = "";
    CNameVal nameVal;
    unsigned int nMax = 500;
    
    CNameDB dbName("r");

    vector<pair<CNameVal, pair<CNameIndex,int> > > nameScan;
    if (!dbName.ScanNames(nameVal, nMax, nameScan))
        throw JSONRPCError(RPC_WALLET_ERROR, "scan failed");

    pair<CNameVal, pair<CNameIndex,int> > pairScan;
    BOOST_FOREACH(pairScan, nameScan)
    {
        CNameIndex nameIndex = pairScan.second.first;
        CNameVal name = pairScan.first;
        CNameVal value = nameIndex.value;
        const std::string ddnsName = stringFromNameVal(name);
        if (ddnsName == ("address:" + strAddress) && nameIndex.op == OP_NAME_MULTISIG && AddressMatchesPubKey(name, value, strErrorMessage))
        {
            return stringFromNameVal(value);
        }
    }
    return strErrorMessage;
}
