// Copyright (c) 2019 Duality Blockchain Solutions Developers 
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/domainentry.h"
#include "bdap/domainentrydb.h"
#include "bdap/utils.h"
#include "dht/ed25519.h"
#include "core_io.h" // needed for ScriptToAsmStr
#include "dynodeman.h"
#include "rpcprotocol.h"
#include "rpcserver.h"
#include "primitives/transaction.h"
#include "wallet/wallet.h"
#include "validation.h"

#include <univalue.h>

extern void SendBDAPTransaction(const CScript& bdapDataScript, const CScript& bdapOPScript, CWalletTx& wtxNew, const CAmount& nOPValue, const CAmount& nDataValue, const bool fUseInstantSend);

static constexpr bool fPrintDebug = true;

static UniValue AddDomainEntry(const JSONRPCRequest& request, BDAP::ObjectType bdapType)
{
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
    txDomainEntry.fPublicObject = 1; // make entry public
    txDomainEntry.nObjectType = GetObjectTypeInt(bdapType);

    // Check if name already exists
    if (GetDomainEntry(txDomainEntry.vchFullObjectPath(), txDomainEntry))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3500 - " + txDomainEntry.GetFullObjectPath() + _(" entry already exists.  Can not add duplicate."));

    // TODO: Add ability to pass in the wallet address
    CKey privWalletKey;
    privWalletKey.MakeNewKey(true);
    CPubKey pubWalletKey = privWalletKey.GetPubKey();
    CKeyID keyWalletID = pubWalletKey.GetID();
    CDynamicAddress walletAddress = CDynamicAddress(keyWalletID);

    if (pwalletMain && !pwalletMain->AddKeyPubKey(privWalletKey, pubWalletKey))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3502 - " + _("Error adding receiving address key wo wallet for BDAP"));

    pwalletMain->SetAddressBook(keyWalletID, strObjectID, "bdap-wallet");
    
    CharString vchWalletAddress = vchFromString(walletAddress.ToString());
    txDomainEntry.WalletAddress = vchWalletAddress;

    // TODO: Add ability to pass in the DHT public key
    CKeyEd25519 privDHTKey;
    CharString vchDHTPubKey = privDHTKey.GetPubKey();
    
    if (pwalletMain && !pwalletMain->AddDHTKey(privDHTKey, vchDHTPubKey))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3503 - " + _("Error adding ed25519 key to wallet for BDAP"));

    txDomainEntry.DHTPublicKey = vchDHTPubKey;
    pwalletMain->SetAddressBook(privDHTKey.GetID(), strObjectID, "bdap-dht-key");

    // TODO: Add ability to pass in the link address
    // TODO: Use stealth address for the link address so linking will be private
    CKey privLinkKey;
    privLinkKey.MakeNewKey(true);
    CPubKey pubLinkKey = privLinkKey.GetPubKey();
    CKeyID keyLinkID = pubLinkKey.GetID();
    CDynamicAddress linkAddress = CDynamicAddress(keyLinkID);
    if (pwalletMain && !pwalletMain->AddKeyPubKey(privLinkKey, pubLinkKey))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3504 - " + _("Error adding receiving address key wo wallet for BDAP"));

    pwalletMain->SetAddressBook(keyLinkID, strObjectID, "bdap-link");
    
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
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_NEW) << CScript::EncodeOP_N(OP_BDAP_ACCOUNT_ENTRY) 
                 << vchFullObjectPath << txDomainEntry.DHTPublicKey << txDomainEntry.nExpireTime << OP_2DROP << OP_2DROP << OP_DROP;

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
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3505 - " + strMessage);

    bool fUseInstantSend = false;
    int enabled = dnodeman.CountEnabled();
    if (enabled > 5) {
        // TODO (bdap): calculate cost for instant send.
        nOperationFee = nOperationFee * 2;
        fUseInstantSend = true;
    }
    SendBDAPTransaction(scriptData, scriptPubKey, wtx, nDataFee, nOperationFee, fUseInstantSend);
    txDomainEntry.txHash = wtx.GetHash();

    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txDomainEntry, oName))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3506 - " + _("Failed to read from BDAP JSON object"));
    
    if (fPrintDebug) {
        // make sure we can deserialize the transaction from the scriptData and get a valid CDomainEntry class
        LogPrintf("DomainEntry Scripts:\nscriptData = %s\n", ScriptToAsmStr(scriptData, true));

        const CTransactionRef testTx = MakeTransactionRef((CTransaction)wtx);
        CDomainEntry testDomainEntry(testTx); //loads the class from a transaction

        LogPrintf("CDomainEntry Values:\nnVersion = %u\nFullObjectPath = %s\nCommonName = %s\nOrganizationalUnit = %s\nDHTPublicKey = %s\n", 
            testDomainEntry.nVersion, testDomainEntry.GetFullObjectPath(), stringFromVch(testDomainEntry.CommonName), 
            stringFromVch(testDomainEntry.OrganizationalUnit), stringFromVch(testDomainEntry.DHTPublicKey));
    }
    std::string strPubKeyHash = privDHTKey.GetHash().ToString();
    std::string strPubKeyID = privDHTKey.GetID().ToString();
    oName.push_back(Pair("dht_pubkey_hash", strPubKeyHash));
    oName.push_back(Pair("dht_pubkey_id", strPubKeyID));
    return oName;
}

UniValue adduser(const JSONRPCRequest& request) 
{
    if (request.params.size() < 2 || request.params.size() > 3) 
    {
        throw std::runtime_error("adduser <userid> <common name> <registration days>\nAdd public name entry to blockchain directory.\n");
    }
    BDAP::ObjectType bdapType = BDAP::ObjectType::BDAP_USER;
    return AddDomainEntry(request, bdapType);
}

UniValue getusers(const JSONRPCRequest& request) 
{
    if (request.params.size() > 2) 
    {
        throw std::runtime_error("getusers <records per page> <page returned>\nLists all BDAP public users.\n");
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

UniValue getgroups(const JSONRPCRequest& request) 
{
    if (request.params.size() > 2) 
    {
        throw std::runtime_error("getgroups <records per page> <page returned>\nLists all BDAP public groups.\n");
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

UniValue getuserinfo(const JSONRPCRequest& request) 
{
    if (request.params.size() != 1) 
    {
        throw std::runtime_error("getuserinfo <public name>\nList BDAP entry.\n");
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
            throw std::runtime_error("BDAP_SELECT_PUBLIC_USER_RPC_ERROR: ERRCODE: 3600 - " + directory.GetFullObjectPath() + _(" can not be found.  Get info failed!"));
        }
    }
    else {
        throw std::runtime_error("BDAP_SELECT_PUBLIC_USER_RPC_ERROR: ERRCODE: 3601 - " + _("Can not access BDAP LevelDB database.  Get info failed!"));
    }

    return oDomainEntryInfo;
}

UniValue getgroupinfo(const JSONRPCRequest& request) 
{
    if (request.params.size() != 1) 
    {
        throw std::runtime_error("getgroupinfo <public group>\nList BDAP entry.\n");
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
            throw std::runtime_error("BDAP_SELECT_PUBLIC_GROUP_RPC_ERROR: ERRCODE: 3600 - " + directory.GetFullObjectPath() + _(" can not be found.  Get info failed!"));
        }
    }
    else {
        throw std::runtime_error("BDAP_SELECT_PUBLIC_GROUP_RPC_ERROR: ERRCODE: 3601 - " + _("Can not access BDAP LevelDB database.  Get info failed!"));
    }

    return oDomainEntryInfo;
}

static UniValue UpdateDomainEntry(const JSONRPCRequest& request, BDAP::ObjectType bdapType) 
{
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
        throw std::runtime_error("BDAP_UPDATE_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3700 - " + txPreviousEntry.GetFullObjectPath() + _(" does not exists.  Can not update."));

    int nIn = GetBDAPOperationOutIndex(txPreviousEntry.nHeight, txPreviousEntry.txHash);
    COutPoint outpoint = COutPoint(txPreviousEntry.txHash, nIn);
    if(pwalletMain->IsMine(CTxIn(outpoint)) != ISMINE_SPENDABLE)
        throw std::runtime_error("BDAP_UPDATE_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3701 - You do not own the " + txPreviousEntry.GetFullObjectPath() + _(" entry.  Can not update."));

    CDomainEntry txUpdatedEntry = txPreviousEntry;
    CharString vchCommonName = vchFromValue(request.params[1]);
    txUpdatedEntry.CommonName = vchCommonName;
    txUpdatedEntry.nObjectType = GetObjectTypeInt(bdapType);

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
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_MODIFY) << CScript::EncodeOP_N(OP_BDAP_ACCOUNT_ENTRY) 
                 << vchFullObjectPath << txUpdatedEntry.DHTPublicKey << txUpdatedEntry.nExpireTime << OP_2DROP << OP_2DROP << OP_DROP;

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
        throw std::runtime_error("BDAP_UPDATE_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3702 - " + strMessage);

    bool fUseInstantSend = false;
    int enabled = dnodeman.CountEnabled();
    if (enabled > 5) {
        // TODO (bdap): calculate cost for instant send.
        nOperationFee = nOperationFee * 2;
        fUseInstantSend = true;
    }
    SendBDAPTransaction(scriptData, scriptPubKey, wtx, nDataFee, nOperationFee, fUseInstantSend);
    txUpdatedEntry.txHash = wtx.GetHash();

    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txUpdatedEntry, oName))
        throw std::runtime_error("BDAP_UPDATE_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3703 - " + _("Failed to read from BDAP JSON object"));
    
    if (fPrintDebug) {
        // make sure we can deserialize the transaction from the scriptData and get a valid CDomainEntry class
        LogPrintf("DomainEntry Scripts:\nscriptData = %s\n", ScriptToAsmStr(scriptData, true));

        const CTransactionRef testTx = MakeTransactionRef((CTransaction)wtx);
        CDomainEntry testDomainEntry(testTx); //loads the class from a transaction

        LogPrintf("CDomainEntry Values:\nnVersion = %u\nFullObjectPath = %s\nCommonName = %s\nOrganizationalUnit = %s\nDHTPublicKey = %s\n", 
            testDomainEntry.nVersion, testDomainEntry.GetFullObjectPath(), stringFromVch(testDomainEntry.CommonName), 
            stringFromVch(testDomainEntry.OrganizationalUnit), stringFromVch(testDomainEntry.DHTPublicKey));
    }

    return oName;
}

UniValue updateuser(const JSONRPCRequest& request) {
    if (request.params.size() < 2 || request.params.size() > 3) 
    {
        throw std::runtime_error("updateuser <userid> <common name> <registration days>\nUpdate an existing public name blockchain directory entry.\n");
    }

    BDAP::ObjectType bdapType = BDAP::ObjectType::BDAP_USER;
    return UpdateDomainEntry(request, bdapType);
}

UniValue updategroup(const JSONRPCRequest& request) {
    if (request.params.size() < 2 || request.params.size() > 3) 
    {
        throw std::runtime_error("updategroup <groupid> <common name> <registration days>\nUpdate an existing public name blockchain directory entry.\n");
    }

    BDAP::ObjectType bdapType = BDAP::ObjectType::BDAP_GROUP;
    return UpdateDomainEntry(request, bdapType);
}

static UniValue DeleteDomainEntry(const JSONRPCRequest& request, BDAP::ObjectType bdapType) 
{

    EnsureWalletIsUnlocked();

    // Format object and domain names to lower case.
    CharString vchObjectID = vchFromValue(request.params[0]);
    ToLowerCase(vchObjectID);
    
    CDomainEntry txSearchEntry;
    txSearchEntry.DomainComponent = vchDefaultDomainName;
    txSearchEntry.OrganizationalUnit = vchDefaultPublicOU;
    txSearchEntry.ObjectID = vchObjectID;
    CDomainEntry txDeletedEntry = txSearchEntry;
    
    // Check if name already exists
    if (!GetDomainEntry(txSearchEntry.vchFullObjectPath(), txSearchEntry))
        throw std::runtime_error("BDAP_DELETE_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3700 - " + txSearchEntry.GetFullObjectPath() + _(" does not exists.  Can not delete."));

    int nIn = GetBDAPOperationOutIndex(txSearchEntry.nHeight, txSearchEntry.txHash);
    COutPoint outpoint = COutPoint(txSearchEntry.txHash, nIn);
    if(pwalletMain->IsMine(CTxIn(outpoint)) != ISMINE_SPENDABLE)
        throw std::runtime_error("BDAP_DELETE_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3701 - You do not own the " + txSearchEntry.GetFullObjectPath() + _(" entry.  Can not delete."));
    
    txDeletedEntry.WalletAddress = txSearchEntry.WalletAddress;
    txDeletedEntry.CommonName = txSearchEntry.CommonName;
    txDeletedEntry.nObjectType = GetObjectTypeInt(bdapType);

    CharString data;
    txDeletedEntry.Serialize(data);
    
    // Create BDAP operation script
    CScript scriptPubKey;
    std::vector<unsigned char> vchFullObjectPath = txDeletedEntry.vchFullObjectPath();
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_DELETE) << CScript::EncodeOP_N(OP_BDAP_ACCOUNT_ENTRY) 
                 << vchFullObjectPath << txDeletedEntry.DHTPublicKey << OP_2DROP << OP_2DROP;

    CDynamicAddress walletAddress(stringFromVch(txDeletedEntry.WalletAddress));
    CScript scriptDestination;
    scriptDestination = GetScriptForDestination(walletAddress.Get());
    scriptPubKey += scriptDestination;

    // Create BDAP OP_RETURN script
    CScript scriptData;
    scriptData << OP_RETURN << data;

    // Send the transaction
    CWalletTx wtx;
    float fYears = ((float)1/365.25);
    CAmount nOperationFee = GetBDAPFee(scriptPubKey) * powf(3.1, fYears);
    CAmount nDataFee = GetBDAPFee(scriptData) * powf(3.1, fYears);

    bool fUseInstantSend = false;
    int enabled = dnodeman.CountEnabled();
    if (enabled > 5) {
        // TODO (bdap): calculate cost for instant send.
        nOperationFee = nOperationFee * 2;
        fUseInstantSend = true;
    }
    SendBDAPTransaction(scriptData, scriptPubKey, wtx, nDataFee, nOperationFee, fUseInstantSend);
    txDeletedEntry.txHash = wtx.GetHash();

    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txDeletedEntry, oName))
        throw std::runtime_error("BDAP_DELETE_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3703 - " + _("Failed to read from BDAP JSON object"));
    
    if (fPrintDebug) {
        // make sure we can deserialize the transaction from the scriptData and get a valid CDomainEntry class
        LogPrintf("DomainEntry Scripts:\nscriptData = %s\n", ScriptToAsmStr(scriptData, true));

        const CTransactionRef testTx = MakeTransactionRef((CTransaction)wtx);
        CDomainEntry testDomainEntry(testTx); //loads the class from a transaction

        LogPrintf("CDomainEntry Values:\nnVersion = %u\nFullObjectPath = %s\nCommonName = %s\nOrganizationalUnit = %s\nDHTPublicKey = %s\n", 
            testDomainEntry.nVersion, testDomainEntry.GetFullObjectPath(), stringFromVch(testDomainEntry.CommonName), 
            stringFromVch(testDomainEntry.OrganizationalUnit), stringFromVch(testDomainEntry.DHTPublicKey));
    }

    return oName;
}

UniValue deleteuser(const JSONRPCRequest& request) {
    if (request.params.size() != 1) 
    {
        throw std::runtime_error("deleteuser <userid>\nDelete an existing public name blockchain directory entry.\n");
    }

    BDAP::ObjectType bdapType = BDAP::ObjectType::BDAP_USER;
    return DeleteDomainEntry(request, bdapType);
}

UniValue deletegroup(const JSONRPCRequest& request) {
    if (request.params.size() != 1) 
    {
        throw std::runtime_error("deletegroup <groupid>\nDelete an existing public name blockchain directory entry.\n");
    }

    BDAP::ObjectType bdapType = BDAP::ObjectType::BDAP_GROUP;
    return DeleteDomainEntry(request, bdapType);
}

UniValue makekeypair(const JSONRPCRequest& request)
{
    if (request.params.size() > 1)
        throw std::runtime_error(
            "makekeypair [prefix]\n"
            "Make a public/private key pair.\n"
            "[prefix] is optional preferred prefix for the public key.\n");

    std::string strPrefix = "";
    if (request.params.size() > 0)
        strPrefix = request.params[0].get_str();

    CKey key;
    uint32_t nCount = 0;
    do
    {
        key.MakeNewKey(false);
        nCount++;
    } while (nCount < 10000 && strPrefix != HexStr(key.GetPubKey()).substr(0, strPrefix.size()));

    if (strPrefix != HexStr(key.GetPubKey()).substr(0, strPrefix.size()))
        return NullUniValue;

    CPrivKey vchPrivKey = key.GetPrivKey();
    CKeyID keyID = key.GetPubKey().GetID();
    CKey vchSecret = CKey();
    vchSecret.SetPrivKey(vchPrivKey, false);
    UniValue result(UniValue::VOBJ);
    result.push_back(Pair("PrivateKey", HexStr<CPrivKey::iterator>(vchPrivKey.begin(), vchPrivKey.end())));
    result.push_back(Pair("PublicKey", HexStr(key.GetPubKey())));
    result.push_back(Pair("WalletAddress", CDynamicAddress(keyID).ToString()));
    result.push_back(Pair("WalletPrivateKey", CDynamicSecret(vchSecret).ToString()));
    return result;
}

UniValue addgroup(const JSONRPCRequest& request) 
{
    if (request.params.size() < 2 || request.params.size() > 3) 
    {
        throw std::runtime_error("addgroup <groupid> <common name> <registration days>\nAdd public group entry to blockchain directory.\n");
    }

    BDAP::ObjectType bdapType = BDAP::ObjectType::BDAP_GROUP;
    return AddDomainEntry(request, bdapType);
}

UniValue mybdapaccounts(const JSONRPCRequest& request)
{
    if (request.params.size() != 0)
        throw std::runtime_error(
            "mybdapaccounts\n"
            "Returns your BDAP accounts.\n");

    std::vector<std::vector<unsigned char>> vvchDHTPubKeys;
    if (pwalletMain && !pwalletMain->GetDHTPubKeys(vvchDHTPubKeys))
        throw std::runtime_error("MY_BDAP_ACCOUNTS_RPC_ERROR: ERRCODE: 3800 - " + _("Error adding receiving address key wo wallet for BDAP"));

    LogPrint("bdap", "%s -- pubkey size = %u\n", __func__, vvchDHTPubKeys.size());

    UniValue result(UniValue::VOBJ);
    uint32_t nCount = 1;
    for (const std::vector<unsigned char>& vchPubKey : vvchDHTPubKeys) {
        CDomainEntry entry;
        LogPrint("bdap", "%s -- pubkey = %s\n", __func__, stringFromVch(vchPubKey));
        if (pDomainEntryDB->ReadDomainEntryPubKey(vchPubKey, entry)) {
            LogPrint("bdap", "%s -- entry = %s\n", __func__, entry.GetFullObjectPath());
            UniValue oAccount(UniValue::VOBJ);
            if (BuildBDAPJson(entry, oAccount, false)) {
                result.push_back(Pair("account_" + std::to_string(nCount) , oAccount));
                nCount++;
            }
        }
    }
    return result;
}

static const CRPCCommand commands[] =
{ //  category              name                     actor (function)               okSafe argNames
  //  --------------------- ------------------------ -----------------------        ------ --------------------
#ifdef ENABLE_WALLET
    /* BDAP */
    { "bdap",            "adduser",                  &adduser,                      true, {"userid","common name", "registration days"} },
    { "bdap",            "getusers",                 &getusers,                     true, {"records per page","page returned"} },
    { "bdap",            "getgroups",                &getgroups,                    true, {"records per page","page returned"} },
    { "bdap",            "getuserinfo",              &getuserinfo,                  true, {"public name"} },
    { "bdap",            "updateuser",               &updateuser,                   true, {"userid","common name", "registration days"} },
    { "bdap",            "updategroup",              &updategroup,                  true, {"groupid","common name", "registration days"} },
    { "bdap",            "deleteuser",               &deleteuser,                   true, {"userid"} },
    { "bdap",            "deletegroup",              &deletegroup,                  true, {"groupid"} },
    { "bdap",            "addgroup",                 &addgroup,                     true, {"groupid","common name"} },
    { "bdap",            "getgroupinfo",             &getgroupinfo,                 true, {"groupid"} },
    { "bdap",            "mybdapaccounts",           &mybdapaccounts,               true, {} },
#endif //ENABLE_WALLET
    { "bdap",            "makekeypair",              &makekeypair,                  true, {} },
};

void RegisterDomainEntryRPCCommands(CRPCTable &t)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        t.appendCommand(commands[vcidx].name, &commands[vcidx]);
}