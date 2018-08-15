// Copyright (c) 2018 Duality Blockchain Solutions Developers 
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/domainentry.h"
#include "bdap/domainentrydb.h"
#include "core_io.h" // needed for ScriptToAsmStr
#include "rpcprotocol.h"
#include "rpcserver.h"
#include "primitives/transaction.h"
#include "wallet/wallet.h"
#include "validation.h"

#include <univalue.h>

extern void SendBDAPTransaction(const CScript bdapDataScript, const CScript bdapOPScript, CWalletTx& wtxNew, CAmount nOPValue, CAmount nDataValue);

static constexpr bool fPrintDebug = true;

UniValue adddomainentry(const JSONRPCRequest& request) 
{
    if (request.params.size() < 2 || request.params.size() > 3) 
    {
        throw std::runtime_error("adddomainentry <userid> <common name> <registration days>\nAdd public name entry to blockchain directory.\n");
    }

    EnsureWalletIsUnlocked();

    // Adds a new name to channel zero.  OID = 0.0.block-height.tx-ordinal.0.0.0.0
    // Format object and domain names to lower case.
    std::string strObjectID = request.params[0].get_str();
    ToLowerCase(strObjectID);

    CharString vchObjectID = vchFromValue(request.params[0]);
    ToLowerCase(vchObjectID);
    CharString vchCommonName = vchFromValue(request.params[1]);

    CDomainEntry txDomainEntry;
    txDomainEntry.OID = vchDefaultOIDPrefix;
    txDomainEntry.DomainComponent = vchDefaultDomainName;
    txDomainEntry.OrganizationalUnit = vchDefaultPublicOU;
    txDomainEntry.CommonName = vchCommonName;
    txDomainEntry.OrganizationName = vchDefaultOrganizationName;
    txDomainEntry.ObjectID = vchObjectID;
    txDomainEntry.fPublicObject = 1; //make entry public

    // Check if name already exists
    if (GetDomainEntry(txDomainEntry.vchFullObjectPath(), txDomainEntry))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3500 - " + txDomainEntry.GetFullObjectPath() + _(" entry already exists.  Can not add duplicate."));

    // TODO: Add ability to pass in the wallet address
    CKey privWalletKey;
    privWalletKey.MakeNewKey(true);
    CPubKey pubWalletKey = privWalletKey.GetPubKey();
    CKeyID keyWalletID = pubWalletKey.GetID();
    CDynamicAddress walletAddress = CDynamicAddress(keyWalletID);

    if (pwalletMain && !pwalletMain->AddKeyPubKey(privWalletKey, pubWalletKey))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3502 - " + _("Error adding receiving address key wo wallet for BDAP"));

    pwalletMain->SetAddressBook(keyWalletID, strObjectID, "receive");
    
    CharString vchWalletAddress = vchFromString(walletAddress.ToString());
    txDomainEntry.WalletAddress = vchWalletAddress;

    // TODO: Add ability to pass in the encryption public key
    CKey privEncryptKey;
    privEncryptKey.MakeNewKey(true);
    CPubKey pubEncryptKey = privEncryptKey.GetPubKey();
    std::string strPrivateEncryptKey = CDynamicSecret(privEncryptKey).ToString();
    CharString vchEncryptPriKey(strPrivateEncryptKey.begin(), strPrivateEncryptKey.end());
    CharString vchEncryptPubKey(pubEncryptKey.begin(), pubEncryptKey.end());
    if (pwalletMain && !pwalletMain->AddKeyPubKey(privEncryptKey, pubEncryptKey))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3503 - " + _("Error adding encrypt key to wallet for BDAP"));

    txDomainEntry.EncryptPublicKey = vchEncryptPubKey;

    // TODO: Add ability to pass in the link address
    CKey privLinkKey;
    privLinkKey.MakeNewKey(true);
    CPubKey pubLinkKey = privLinkKey.GetPubKey();
    CKeyID keyLinkID = pubLinkKey.GetID();
    CDynamicAddress linkAddress = CDynamicAddress(keyLinkID);
    if (pwalletMain && !pwalletMain->AddKeyPubKey(privLinkKey, pubLinkKey))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3504 - " + _("Error adding receiving address key wo wallet for BDAP"));

    pwalletMain->SetAddressBook(keyLinkID, strObjectID, "link");
    
    CharString vchLinkAddress = vchFromString(linkAddress.ToString());
    txDomainEntry.LinkAddress = vchLinkAddress;

    uint64_t nDays = 1461;  //default to 4 years.
    if (request.params.size() >= 3) {
        nDays = request.params[2].get_int();
    }
    uint64_t nSeconds = nDays * SECONDS_PER_DAY;
    txDomainEntry.nExpireTime = chainActive.Tip()->GetMedianTimePast() + nSeconds;

    CharString data;
    txDomainEntry.Serialize(data);
    
    // Create BDAP operation script
    CScript scriptPubKey;
    std::vector<unsigned char> vchFullObjectPath = txDomainEntry.vchFullObjectPath();
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP) << CScript::EncodeOP_N(OP_BDAP_NEW) << vchFullObjectPath << OP_2DROP << OP_DROP;

    CScript scriptDestination;
    scriptDestination = GetScriptForDestination(walletAddress.Get());
    scriptPubKey += scriptDestination;

    // Create BDAP OP_RETURN script
    CScript scriptData;
    scriptData << OP_RETURN << data;

    // Send the transaction
    CWalletTx wtx;
    float fYears = ((float)nDays/365.25);
    CAmount nOperationFee = GetBDAPFee(scriptPubKey) * powf(3.1, fYears);
    CAmount nDataFee = GetBDAPFee(scriptData) * powf(3.1, fYears);

    // check BDAP values
    std::string strMessage;
    if (!txDomainEntry.ValidateValues(strMessage))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3505 - " + strMessage);

    SendBDAPTransaction(scriptData, scriptPubKey, wtx, nDataFee, nOperationFee);
    txDomainEntry.txHash = wtx.GetHash();

    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txDomainEntry, oName))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3506 - " + _("Failed to read from BDAP JSON object"));
    
    if (fPrintDebug) {
        // make sure we can deserialize the transaction from the scriptData and get a valid CDomainEntry class
        LogPrintf("DomainEntry Scripts:\nscriptData = %s\n", ScriptToAsmStr(scriptData, true));

        const CTransaction testTx = (CTransaction)wtx;
        CDomainEntry testDomainEntry(testTx); //loads the class from a transaction

        LogPrintf("CDomainEntry Values:\nnVersion = %u\nFullObjectPath = %s\nCommonName = %s\nOrganizationalUnit = %s\nEncryptPublicKey = %s\n", 
            testDomainEntry.nVersion, testDomainEntry.GetFullObjectPath(), stringFromVch(testDomainEntry.CommonName), 
            stringFromVch(testDomainEntry.OrganizationalUnit), HexStr(testDomainEntry.EncryptPublicKey));
    }

    return oName;
}

UniValue getdomainentries(const JSONRPCRequest& request) 
{
    if (request.params.size() > 2) 
    {
        throw std::runtime_error("getdomainentries <records per page> <page returned>\nLists all BDAP entries.\n");
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

    UniValue oDomainEntryList(UniValue::VARR);
    if (CheckDomainEntryDB())
        pDomainEntryDB->ListDirectories(vchObjectLocation, nRecordsPerPage, nPage, oDomainEntryList);

    return oDomainEntryList;
}

UniValue getdomainentryinfo(const JSONRPCRequest& request) 
{
    if (request.params.size() != 1) 
    {
        throw std::runtime_error("getdomainentryinfo <public name>\nList BDAP entry.\n");
    }

    CharString vchObjectID = vchFromValue(request.params[0]);
    ToLowerCase(vchObjectID);

    CDomainEntry directory;
    directory.DomainComponent = vchDefaultDomainName;
    directory.OrganizationalUnit = vchDefaultPublicOU;
    directory.ObjectID = vchObjectID;
    
    UniValue oDomainEntryInfo(UniValue::VOBJ);
    if (CheckDomainEntryDB()) {
        if (!pDomainEntryDB->GetDomainEntryInfo(directory.vchFullObjectPath(), oDomainEntryInfo)) {
            throw std::runtime_error("BDAP_SELECT_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3600 - " + directory.GetFullObjectPath() + _(" can not be found.  Get info failed!"));
        }
    }
    else {
        throw std::runtime_error("BDAP_SELECT_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3601 - " + _("Can not access BDAP LevelDB database.  Get info failed!"));
    }

    return oDomainEntryInfo;
}

UniValue updatedomainentry(const JSONRPCRequest& request) {
    if (request.params.size() < 2 || request.params.size() > 3) 
    {
        throw std::runtime_error("updatedomainentry <userid> <common name> <registration days>\nUpdate an existing public name blockchain directory entry.\n");
    }

    EnsureWalletIsUnlocked();

    // Format object and domain names to lower case.
    CharString vchObjectID = vchFromValue(request.params[0]);
    ToLowerCase(vchObjectID);
    
    CDomainEntry txPreviousEntry;
    txPreviousEntry.DomainComponent = vchDefaultDomainName;
    txPreviousEntry.OrganizationalUnit = vchDefaultPublicOU;
    txPreviousEntry.ObjectID = vchObjectID;

    // Check if name already exists
    if (!GetDomainEntry(txPreviousEntry.vchFullObjectPath(), txPreviousEntry))
        throw std::runtime_error("BDAP_UPDATE_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3700 - " + txPreviousEntry.GetFullObjectPath() + _(" does not exists.  Can not update."));

    int nIn = GetDomainEntryOperationOutIndex(txPreviousEntry.nHeight, txPreviousEntry.txHash);
    COutPoint outpoint = COutPoint(txPreviousEntry.txHash, nIn);
    if(pwalletMain->IsMine(CTxIn(outpoint)) != ISMINE_SPENDABLE)
        throw std::runtime_error("BDAP_UPDATE_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3701 - You do not own the " + txPreviousEntry.GetFullObjectPath() + _(" entry.  Can not update."));

    CDomainEntry txUpdatedEntry = txPreviousEntry;
    CharString vchCommonName = vchFromValue(request.params[1]);
    txUpdatedEntry.CommonName = vchCommonName;

    uint64_t nDays = 1461;  //default to 4 years.
    if (request.params.size() >= 3) {
        nDays = request.params[2].get_int();
    }
    uint64_t nSeconds = nDays * SECONDS_PER_DAY;
    txUpdatedEntry.nExpireTime = chainActive.Tip()->GetMedianTimePast() + nSeconds;

    CharString data;
    txUpdatedEntry.Serialize(data);
    
    // Create BDAP operation script
    CScript scriptPubKey;
    std::vector<unsigned char> vchFullObjectPath = txUpdatedEntry.vchFullObjectPath();
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP) << CScript::EncodeOP_N(OP_BDAP_MODIFY) << vchFullObjectPath << OP_2DROP << OP_DROP;

    CDynamicAddress walletAddress(stringFromVch(txUpdatedEntry.WalletAddress));
    CScript scriptDestination;
    scriptDestination = GetScriptForDestination(walletAddress.Get());
    scriptPubKey += scriptDestination;

    // Create BDAP OP_RETURN script
    CScript scriptData;
    scriptData << OP_RETURN << data;

    // Send the transaction
    CWalletTx wtx;
    float fYears = ((float)nDays/365.25);
    CAmount nOperationFee = GetBDAPFee(scriptPubKey) * powf(3.1, fYears);
    CAmount nDataFee = GetBDAPFee(scriptData) * powf(3.1, fYears);

    // check BDAP values
    std::string strMessage;
    if (!txUpdatedEntry.ValidateValues(strMessage))
        throw std::runtime_error("BDAP_UPDATE_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3702 - " + strMessage);

    SendBDAPTransaction(scriptData, scriptPubKey, wtx, nDataFee, nOperationFee);
    txUpdatedEntry.txHash = wtx.GetHash();

    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txUpdatedEntry, oName))
        throw std::runtime_error("BDAP_UPDATE_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3703 - " + _("Failed to read from BDAP JSON object"));
    
    if (fPrintDebug) {
        // make sure we can deserialize the transaction from the scriptData and get a valid CDomainEntry class
        LogPrintf("DomainEntry Scripts:\nscriptData = %s\n", ScriptToAsmStr(scriptData, true));

        const CTransaction testTx = (CTransaction)wtx;
        CDomainEntry testDomainEntry(testTx); //loads the class from a transaction

        LogPrintf("CDomainEntry Values:\nnVersion = %u\nFullObjectPath = %s\nCommonName = %s\nOrganizationalUnit = %s\nEncryptPublicKey = %s\n", 
            testDomainEntry.nVersion, testDomainEntry.GetFullObjectPath(), stringFromVch(testDomainEntry.CommonName), 
            stringFromVch(testDomainEntry.OrganizationalUnit), HexStr(testDomainEntry.EncryptPublicKey));
    }

    return oName;
}

static const CRPCCommand commands[] =
{   //  category         name                        actor (function)           okSafeMode
#ifdef ENABLE_WALLET
    /* BDAP */
    { "bdap",            "adddomainentry",           &adddomainentry,               true  },
    { "bdap",            "getdomainentries",         &getdomainentries,             true  },
    { "bdap",            "getdomainentryinfo",       &getdomainentryinfo,           true  },
    { "bdap",            "updatedomainentry",        &updatedomainentry,            true  },
#endif //ENABLE_WALLET
};

void RegisterDomainEntryRPCCommands(CRPCTable &tableRPC)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        tableRPC.appendCommand(commands[vcidx].name, &commands[vcidx]);
}