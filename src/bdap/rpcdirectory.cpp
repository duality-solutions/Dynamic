// Copyright (c) 2018 Duality Blockchain Solutions Developers 
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/directory.h"
#include "bdap/directorydb.h"
#include "core_io.h" // needed for ScriptToAsmStr
#include "rpcprotocol.h"
#include "rpcserver.h"
#include "primitives/transaction.h"
#include "wallet/wallet.h"

#include <univalue.h>

extern void SendBDAPTransaction(const CScript bdapDataScript, const CScript bdapOPScript, CWalletTx& wtxNew, CAmount nOPValue, CAmount nDataValue);

static constexpr bool fPrintDebug = true;

UniValue addpublicname(const JSONRPCRequest& request) 
{
    if (request.params.size() != 2) 
    {
        throw std::runtime_error("addpublicname <userid> <common name>\nAdd public name entry to blockchain directory.\n");
    }

    EnsureWalletIsUnlocked();

    // Adds a new name to channel zero.  OID = 0.0.block-height.tx-ordinal.0.0.0.0
    // Format object and domain names to lower case.
    std::string strObjectID = request.params[0].get_str();
    ToLowerCase(strObjectID);

    CharString vchObjectID = vchFromValue(request.params[0]);
    ToLowerCase(vchObjectID);
    CharString vchCommonName = vchFromValue(request.params[1]);

    

    CDirectory txDirectory;
    txDirectory.OID = vchDefaultOIDPrefix;
    txDirectory.DomainComponent = vchDefaultDomainName;
    txDirectory.OrganizationalUnit = vchDefaultPublicOU;
    txDirectory.CommonName = vchCommonName;
    txDirectory.OrganizationName = vchDefaultOrganizationName;
    txDirectory.ObjectID = vchObjectID;
    txDirectory.fPublicObject = 1; //make entry public
    txDirectory.transactionFee = 100;

    // Check if name already exists
    if (GetDirectory(txDirectory.vchFullObjectPath(), txDirectory))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3500 - " + txDirectory.GetFullObjectPath() + _(" entry already exists.  Can not add duplicate."));

    // TODO: Add ability to pass in the wallet address and public key
    CKey privWalletKey;
    privWalletKey.MakeNewKey(true);
    CPubKey pubWalletKey = privWalletKey.GetPubKey();
    CKeyID keyWalletID = pubWalletKey.GetID();
    
    if (pwalletMain && !pwalletMain->AddKeyPubKey(privWalletKey, pubWalletKey))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3502 - " + _("Error adding receiving address key wo wallet for BDAP"));

    pwalletMain->SetAddressBook(keyWalletID, strObjectID, "receive");
    
    std::string strWalletAddress = CDynamicAddress(keyWalletID).ToString();
    CharString vchWalletAddress(strWalletAddress.begin(), strWalletAddress.end());
    txDirectory.WalletAddress = vchWalletAddress;

    CKey privEncryptKey;
    privEncryptKey.MakeNewKey(true);
    CPubKey pubEncryptKey = privEncryptKey.GetPubKey();
    std::string strPrivateEncryptKey = CDynamicSecret(privEncryptKey).ToString();
    CharString vchEncryptPriKey(strPrivateEncryptKey.begin(), strPrivateEncryptKey.end());
    CharString vchEncryptPubKey(pubEncryptKey.begin(), pubEncryptKey.end());
    if (pwalletMain && !pwalletMain->AddKeyPubKey(privEncryptKey, pubEncryptKey))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3503 - " + _("Error adding encrypt key to wallet for BDAP"));

    txDirectory.EncryptPublicKey = vchEncryptPubKey;

    CKey privSignKey;
    privSignKey.MakeNewKey(true);
    CPubKey pubSignKey = privSignKey.GetPubKey();
    CKeyID keySignID = pubSignKey.GetID();
    CDynamicAddress signWallet = CDynamicAddress(keySignID);
    std::string strSignWalletAddress = signWallet.ToString();
    CharString vchSignWalletAddress(strSignWalletAddress.begin(), strSignWalletAddress.end());

    if (pwalletMain && !pwalletMain->AddKeyPubKey(privSignKey, pubSignKey))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3504 - " + _("Error adding signature key to wallet for BDAP"));

    txDirectory.SignWalletAddress = vchSignWalletAddress;

    CharString data;
    txDirectory.Serialize(data);
    
    // Create BDAP OP_RETURN Signature Scripts
    CScript scriptPubKey;
    std::vector<unsigned char> vchFullObjectPath = txDirectory.vchFullObjectPath();
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP) << CScript::EncodeOP_N(OP_BDAP_NEW) << vchFullObjectPath << OP_2DROP << OP_DROP;

    CScript scriptDestination;
    scriptDestination = GetScriptForDestination(signWallet.Get());
    scriptPubKey += scriptDestination;
    LogPrintf("BDAP GetDirectoryType = %s \n", GetDirectoryOpTypeString(scriptPubKey));

    CScript scriptData;
    scriptData << OP_RETURN << data;

    // Send the transaction
    float fYears = 1.0; //TODO use a variable for registration years.
    CWalletTx wtx;
    CAmount nOperationFee = GetBDAPFee(scriptPubKey) * powf(3.1, fYears);
    CAmount nDataFee = GetBDAPFee(scriptData) * powf(3.1, fYears);

    // check BDAP values
    std::string strMessage;
    if (!txDirectory.ValidateValues(strMessage))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3501 - " + strMessage);

    SendBDAPTransaction(scriptData, scriptPubKey, wtx, nDataFee, nOperationFee);
    txDirectory.txHash = wtx.GetHash();

    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txDirectory, oName))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3502 - " + _("Failed to read from BDAP JSON object"));
    
    if (fPrintDebug) {
        // make sure we can deserialize the transaction from the scriptData and get a valid CDirectory class
        LogPrintf("Directory Scripts:\nscriptData = %s\n", ScriptToAsmStr(scriptData, true));

        const CTransaction testTx = (CTransaction)wtx;
        CDirectory testDirectory(testTx); //loads the class from a transaction

        LogPrintf("CDirectory Values:\nnVersion = %u\nFullObjectPath = %s\nCommonName = %s\nOrganizationalUnit = %s\nEncryptPublicKey = %s\nPrivateData = %s\n", 
            testDirectory.nVersion, testDirectory.GetFullObjectPath(), stringFromVch(testDirectory.CommonName), 
            stringFromVch(testDirectory.OrganizationalUnit), HexStr(testDirectory.EncryptPublicKey), stringFromVch(testDirectory.PrivateData));
    }

    return oName;
}

UniValue getdirectories(const JSONRPCRequest& request) 
{
    if (request.params.size() > 2) 
    {
        throw std::runtime_error("directorylist <records per page> <page returned>\nLists all BDAP entries.\n");
    }
    
    unsigned int nRecordsPerPage = 100;
    unsigned int nPage = 1;
    if (request.params.size() > 0)
        nRecordsPerPage = request.params[0].get_int();

    if (request.params.size() == 2)
        nPage = request.params[1].get_int();
    
    // only return entries from the default public domain OU
    std::string strObjectLocation = DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    CharString vchObjectLocation(strObjectLocation.begin(), strObjectLocation.end());

    UniValue oDirectoryList(UniValue::VARR);
    if (CheckDirectoryDB())
        pDirectoryDB->ListDirectories(vchObjectLocation, nRecordsPerPage, nPage, oDirectoryList);

    return oDirectoryList;
}

UniValue getdirectoryinfo(const JSONRPCRequest& request) 
{
    if (request.params.size() != 1) 
    {
        throw std::runtime_error("getdirectoryinfo <public name>\nList BDAP entry.\n");
    }

    CharString vchObjectID = vchFromValue(request.params[0]);
    ToLowerCase(vchObjectID);

    CDirectory directory;
    directory.DomainComponent = vchDefaultDomainName;
    directory.OrganizationalUnit = vchDefaultPublicOU;
    directory.ObjectID = vchObjectID;
    
    UniValue oDirectoryInfo(UniValue::VOBJ);
    if (CheckDirectoryDB()) {
        if (!pDirectoryDB->GetDirectoryInfo(directory.vchFullObjectPath(), oDirectoryInfo)) {
            throw std::runtime_error("BDAP_SELECT_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3600 - " + directory.GetFullObjectPath() + _(" can not be found.  Get info failed!"));
        }
    }
    else {
        throw std::runtime_error("BDAP_SELECT_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3601 - " + _("Can not access BDAP LevelDB database.  Get info failed!"));
    }

    return oDirectoryInfo;
}

UniValue updatedirectory(const JSONRPCRequest& request) {
    if (request.params.size() != 2) 
    {
        throw std::runtime_error("updatedirectory <userid> <common name>\nUpdate an existing public name blockchain directory entry.\n");
    }

    EnsureWalletIsUnlocked();

    // Format object and domain names to lower case.
    CharString vchObjectID = vchFromValue(request.params[0]);
    ToLowerCase(vchObjectID);
    
    CDirectory txDirectory;
    txDirectory.DomainComponent = vchDefaultDomainName;
    txDirectory.OrganizationalUnit = vchDefaultPublicOU;
    txDirectory.ObjectID = vchObjectID;

    // Check if name already exists
    if (!GetDirectory(txDirectory.vchFullObjectPath(), txDirectory))
        throw std::runtime_error("BDAP_UPDATE_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3700 - " + txDirectory.GetFullObjectPath() + _(" does not exists.  Can not update."));

    // TODO: Check if Signature Wallet address is owned.  If not, return an error before submitting to the mem pool.
    CharString vchCommonName = vchFromValue(request.params[1]);
    txDirectory.CommonName = vchCommonName;

    CharString data;
    txDirectory.Serialize(data);
    
    // Create BDAP OP_RETURN Signature Scripts
    CScript scriptPubKey;
    std::vector<unsigned char> vchFullObjectPath = txDirectory.vchFullObjectPath();
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP) << CScript::EncodeOP_N(OP_BDAP_MODIFY) << vchFullObjectPath << OP_2DROP << OP_DROP;

    CDynamicAddress signWallet(stringFromVch(txDirectory.SignWalletAddress));
    CScript scriptDestination;
    scriptDestination = GetScriptForDestination(signWallet.Get());
    scriptPubKey += scriptDestination;
    LogPrintf("BDAP GetDirectoryType = %s \n", GetDirectoryOpTypeString(scriptPubKey));

    CScript scriptData;
    scriptData << OP_RETURN << data;

    // Send the transaction
    float fYears = 1.0; //TODO use a variable for registration years.
    CWalletTx wtx;
    CAmount nOperationFee = GetBDAPFee(scriptPubKey) * powf(3.1, fYears);
    CAmount nDataFee = GetBDAPFee(scriptData) * powf(3.1, fYears);

    // check BDAP values
    std::string strMessage;
    if (!txDirectory.ValidateValues(strMessage))
        throw std::runtime_error("BDAP_UPDATE_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3701 - " + strMessage);

    SendBDAPTransaction(scriptData, scriptPubKey, wtx, nDataFee, nOperationFee);
    txDirectory.txHash = wtx.GetHash();

    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txDirectory, oName))
        throw std::runtime_error("BDAP_UPDATE_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3702 - " + _("Failed to read from BDAP JSON object"));
    
    if (fPrintDebug) {
        // make sure we can deserialize the transaction from the scriptData and get a valid CDirectory class
        LogPrintf("Directory Scripts:\nscriptData = %s\n", ScriptToAsmStr(scriptData, true));

        const CTransaction testTx = (CTransaction)wtx;
        CDirectory testDirectory(testTx); //loads the class from a transaction

        LogPrintf("CDirectory Values:\nnVersion = %u\nFullObjectPath = %s\nCommonName = %s\nOrganizationalUnit = %s\nEncryptPublicKey = %s\nPrivateData = %s\n", 
            testDirectory.nVersion, testDirectory.GetFullObjectPath(), stringFromVch(testDirectory.CommonName), 
            stringFromVch(testDirectory.OrganizationalUnit), HexStr(testDirectory.EncryptPublicKey), stringFromVch(testDirectory.PrivateData));
    }

    return oName;
}

static const CRPCCommand commands[] =
{   //  category         name                        actor (function)           okSafeMode
#ifdef ENABLE_WALLET
    /* BDAP */
    { "bdap",            "addpublicname",            &addpublicname,                true  },
    { "bdap",            "getdirectories",           &getdirectories,               true  },
    { "bdap",            "getdirectoryinfo",         &getdirectoryinfo,             true  },
    { "bdap",            "updatedirectory",          &updatedirectory,              true  },
#endif //ENABLE_WALLET
};

void RegisterDirectoryRPCCommands(CRPCTable &tableRPC)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        tableRPC.appendCommand(commands[vcidx].name, &commands[vcidx]);
}