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
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3502 - " + _("Error adding receiving address key to wallet for BDAP"));

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
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3504 - " + _("Error adding receiving address key to wallet for BDAP"));

    pwalletMain->SetAddressBook(keyLinkID, strObjectID, "bdap-link");
    
    CharString vchLinkAddress = vchFromString(linkAddress.ToString());
    txDomainEntry.LinkAddress = vchLinkAddress;

    int64_t nDays = DEFAULT_REGISTRATION_DAYS;  // default to 2 years.
    if (request.params.size() >= 3) {
        if (!ParseInt64(request.params[2].get_str(), &nDays))
            throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3505 - " + _("Error converting registration days to int"));
    }
    int64_t nSeconds = nDays * SECONDS_PER_DAY;
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
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3506 - " + strMessage);

    bool fUseInstantSend = false;
    int enabled = dnodeman.CountEnabled();
    if (enabled > 5) {
        // TODO (bdap): calculate cost for instant send.
        nOperationFee = nOperationFee * 2;
        fUseInstantSend = true;
    }
    SendBDAPTransaction(scriptData, scriptPubKey, wtx, nOperationFee, nDataFee, fUseInstantSend);
    txDomainEntry.txHash = wtx.GetHash();

    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txDomainEntry, oName))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3507 - " + _("Failed to read from BDAP JSON object"));
    
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
    if (request.fHelp || request.params.size() < 2 || request.params.size() > 3)
        throw std::runtime_error(
            "adduser \"account id\" \"common name\" \"registration days\"\n"
            "\nArguments:\n"
            "1. account id         (string)             The account userid\n"
            "2. common name        (string)             The account common name used for searching\n"
            "3. registration days  (int, optional)      Number of days to register account\n"
            "\nAdds a new bdap.io public name account entry to the blockchain directory.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"oid\"                        (string)  Account OID\n"
            "  \"version\"                    (int)     Recipient's BDAP full path\n"
            "  \"domain_component\"           (string)  Account domain name\n"
            "  \"common_name\"                (string)  Account common name\n"
            "  \"organizational_unit\"        (string)  Account organizational unit or OU\n"
            "  \"organization_name\"          (string)  Account organizational name\n"
            "  \"object_id\"                  (string)  Account object ID\n"
            "  \"object_full_path\"           (string)  Account fully qualified domain name (FQDN)\n"
            "  \"object_type\"                (string)  Account type. User or Group\n"
            "  \"wallet_address\"             (string)  Account wallet address\n"
            "  \"public\"                     (boolean) True when account is public\n"
            "  \"dht_publickey\"              (string)  Account DHT public key\n"
            "  \"link_address\"               (string)  Account link address\n"
            "  \"txid\"                       (string)  Last txid for account data\n"
            "  \"time\"                       (int)     Last transaction time for account data\n"
            "  \"height\"                     (int)     Last transaction height for account data\n"
            "  \"expires_on\"                 (int)     Account expiration epoch time\n"
            "  \"expired\"                    (boolean) Account expired\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("adduser", "Alice \"Wonderland, Alice\"") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("adduser", "Alice \"Wonderland, Alice\""));

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3000 - " + _("Can not create BDAP transactions until spork is active."));

    BDAP::ObjectType bdapType = BDAP::ObjectType::BDAP_USER;
    return AddDomainEntry(request, bdapType);
}

UniValue getusers(const JSONRPCRequest& request) 
{
    if (request.fHelp || request.params.size() > 2)
        throw std::runtime_error(
            "getusers \"records per page\" \"page returned\"\n"
            "\nArguments:\n"
            "1. records per page     (int, optional)  If paging, the number of records per page\n"
            "2. page returned        (int, optional)  If paging, the page number to return\n"
            "\nLists all BDAP user accounts in the \"public\" OU for the \"bdap.io\" domain.\n"
            "\nResult:\n"
            "{(json objects)\n"
            "  \"common_name\"             (string)  Account common name\n"
            "  \"object_full_path\"        (string)  Account fully qualified domain name (FQDN)\n"
            "  \"wallet_address\"          (string)  Account wallet address\n"
            "  \"dht_publickey\"           (string)  Account DHT public key\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("getusers", "") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("getusers", ""));

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
        pDomainEntryDB->ListDirectories(vchObjectLocation, nRecordsPerPage, nPage, oDomainEntryList, BDAP::ObjectType::BDAP_USER);

    return oDomainEntryList;
}

UniValue getgroups(const JSONRPCRequest& request) 
{
    if (request.fHelp || request.params.size() > 2)
        throw std::runtime_error(
            "getgroups \"records per page\" \"page returned\"\n"
            "\nArguments:\n"
            "1. records per page     (int, optional)  If paging, the number of records per page\n"
            "2. page returned        (int, optional)  If paging, the page number to return\n"
            "\nLists all BDAP group accounts in the \"public\" OU for the \"bdap.io\" domain.\n"
            "\nResult:\n"
            "{(json objects)\n"
            "  \"common_name\"             (string)  Account common name\n"
            "  \"object_full_path\"        (string)  Account fully qualified domain name (FQDN)\n"
            "  \"wallet_address\"          (string)  Account wallet address\n"
            "  \"dht_publickey\"           (string)  Account DHT public key\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("getgroups", "") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("getgroups", ""));

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
        pDomainEntryDB->ListDirectories(vchObjectLocation, nRecordsPerPage, nPage, oDomainEntryList, BDAP::ObjectType::BDAP_GROUP);

    return oDomainEntryList;
}

UniValue getuserinfo(const JSONRPCRequest& request) 
{
    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "getuserinfo \"account id\"\n"
            "\nArguments:\n"
            "1. account id            (string)  Account object ID (aka UserID)\n"
            "\nShows detailed user account information.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"oid\"                        (string)  Account OID\n"
            "  \"version\"                    (int)     Recipient's BDAP full path\n"
            "  \"domain_component\"           (string)  Account domain name\n"
            "  \"common_name\"                (string)  Account common name\n"
            "  \"organizational_unit\"        (string)  Account organizational unit or OU\n"
            "  \"organization_name\"          (string)  Account organizational name\n"
            "  \"object_id\"                  (string)  Account userid\n"
            "  \"object_full_path\"           (string)  Account fully qualified domain name (FQDN)\n"
            "  \"object_type\"                (string)  Account type. User or Group\n"
            "  \"wallet_address\"             (string)  Account wallet address\n"
            "  \"public\"                     (boolean) True when account is public\n"
            "  \"dht_publickey\"              (string)  Account DHT public key\n"
            "  \"link_address\"               (string)  Account link address\n"
            "  \"txid\"                       (string)  Last txid for account data\n"
            "  \"time\"                       (int)     Last transaction time for account data\n"
            "  \"height\"                     (int)     Last transaction height for account data\n"
            "  \"expires_on\"                 (int)     Account expiration epoch time\n"
            "  \"expired\"                    (boolean) Account expired\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("getuserinfo", "Alice") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("getuserinfo", "Alice"));

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
    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "getgroupinfo \"account id\"\n"
            "\nArguments:\n"
            "1. account id            (string)  Account object ID (aka UserID)\n"
            "\nShows detailed group account information.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"oid\"                        (string)  Account OID\n"
            "  \"version\"                    (int)     Recipient's BDAP full path\n"
            "  \"domain_component\"           (string)  Account domain name\n"
            "  \"common_name\"                (string)  Account common name\n"
            "  \"organizational_unit\"        (string)  Account organizational unit or OU\n"
            "  \"organization_name\"          (string)  Account organizational name\n"
            "  \"object_id\"                  (string)  Account userid\n"
            "  \"object_full_path\"           (string)  Account fully qualified domain name (FQDN)\n"
            "  \"object_type\"                (string)  Account type. User or Group\n"
            "  \"wallet_address\"             (string)  Account wallet address\n"
            "  \"public\"                     (boolean) True when account is public\n"
            "  \"dht_publickey\"              (string)  Account DHT public key\n"
            "  \"link_address\"               (string)  Account link address\n"
            "  \"txid\"                       (string)  Last txid for account data\n"
            "  \"time\"                       (int)     Last transaction time for account data\n"
            "  \"height\"                     (int)     Last transaction height for account data\n"
            "  \"expires_on\"                 (int)     Account expiration epoch time\n"
            "  \"expired\"                    (boolean) Account expired\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("getgroupinfo", "Duality") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("getgroupinfo", "Duality"));

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

    int64_t nDays = DEFAULT_REGISTRATION_DAYS;  // default to 2 years.
    if (request.params.size() >= 3) {
        if (!ParseInt64(request.params[2].get_str(), &nDays))
            throw std::runtime_error("BDAP_UPDATE_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3702 - " + _("Error converting registration days to int"));
    }
    int64_t nSeconds = nDays * SECONDS_PER_DAY;
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
        throw std::runtime_error("BDAP_UPDATE_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3703 - " + strMessage);

    bool fUseInstantSend = false;
    int enabled = dnodeman.CountEnabled();
    if (enabled > 5) {
        // TODO (bdap): calculate cost for instant send.
        nOperationFee = nOperationFee * 2;
        fUseInstantSend = true;
    }
    SendBDAPTransaction(scriptData, scriptPubKey, wtx, nOperationFee, nDataFee, fUseInstantSend);
    txUpdatedEntry.txHash = wtx.GetHash();

    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txUpdatedEntry, oName))
        throw std::runtime_error("BDAP_UPDATE_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3704 - " + _("Failed to read from BDAP JSON object"));
    
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

UniValue updateuser(const JSONRPCRequest& request) 
{
    if (request.fHelp || request.params.size() < 2 || request.params.size() > 3)
        throw std::runtime_error(
            "updateuser \"account id\"  \"common name\"  \"registration days\"\n"
            "\nArguments:\n"
            "1. account id         (string)             The account objectid within public.bdap.io\n"
            "2. common name        (string)             The account common name used for searching\n"
            "3. registration days  (int, optional)      Number of additional days to register account\n"
            "\nUpdates an existing bdap.io public name account entry in the blockchain directory.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"oid\"                        (string)  Account OID\n"
            "  \"version\"                    (int)     Recipient's BDAP full path\n"
            "  \"domain_component\"           (string)  Account domain name\n"
            "  \"common_name\"                (string)  Account common name\n"
            "  \"organizational_unit\"        (string)  Account organizational unit or OU\n"
            "  \"organization_name\"          (string)  Account organizational name\n"
            "  \"object_id\"                  (string)  Account object id\n"
            "  \"object_full_path\"           (string)  Account fully qualified domain name (FQDN)\n"
            "  \"object_type\"                (string)  Account type. User or Group\n"
            "  \"wallet_address\"             (string)  Account wallet address\n"
            "  \"public\"                     (boolean) True when account is public\n"
            "  \"dht_publickey\"              (string)  Account DHT public key\n"
            "  \"link_address\"               (string)  Account link address\n"
            "  \"txid\"                       (string)  Last txid for account data\n"
            "  \"time\"                       (int)     Last transaction time for account data\n"
            "  \"height\"                     (int)     Last transaction height for account data\n"
            "  \"expires_on\"                 (int)     Account expiration epoch time\n"
            "  \"expired\"                    (boolean) Account expired\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("updateuser", "Alice \"Updated, Alice\" 365" ) +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("updateuser", "Alice \"Updated, Alice\" 365"));

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3000 - " + _("Can not create BDAP transactions until spork is active."));

    BDAP::ObjectType bdapType = BDAP::ObjectType::BDAP_USER;
    return UpdateDomainEntry(request, bdapType);
}

UniValue updategroup(const JSONRPCRequest& request) 
{
    if (request.fHelp || request.params.size() < 2 || request.params.size() > 3)
        throw std::runtime_error(
            "updategroup \"account id\"  \"common name\"  \"registration days\"\n"
            "\nArguments:\n"
            "1. groupid            (string)             The account objectid within public.bdap.io\n"
            "2. common name        (string)             The account common name used for searching\n"
            "3. registration days  (int, optional)      Number of additional days to register account\n"
            "\nUpdates an existing bdap.io public group account entry in the blockchain directory.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"oid\"                        (string)  Account OID\n"
            "  \"version\"                    (int)     Recipient's BDAP full path\n"
            "  \"domain_component\"           (string)  Account domain name\n"
            "  \"common_name\"                (string)  Account common name\n"
            "  \"organizational_unit\"        (string)  Account organizational unit or OU\n"
            "  \"organization_name\"          (string)  Account organizational name\n"
            "  \"object_id\"                  (string)  Account object id\n"
            "  \"object_full_path\"           (string)  Account fully qualified domain name (FQDN)\n"
            "  \"object_type\"                (string)  Account type. User or Group\n"
            "  \"wallet_address\"             (string)  Account wallet address\n"
            "  \"public\"                     (boolean) True when account is public\n"
            "  \"dht_publickey\"              (string)  Account DHT public key\n"
            "  \"link_address\"               (string)  Account link address\n"
            "  \"txid\"                       (string)  Last txid for account data\n"
            "  \"time\"                       (int)     Last transaction time for account data\n"
            "  \"height\"                     (int)     Last transaction height for account data\n"
            "  \"expires_on\"                 (int)     Account expiration epoch time\n"
            "  \"expired\"                    (boolean) Account expired\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("updategroup", "Duality \"Updated, Duality Blockchain Solutions Group\" 700" ) +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("updategroup", "Duality \"Updated, Duality Blockchain Solutions Group\" 700"));

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3000 - " + _("Can not create BDAP transactions until spork is active."));

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
    
    // Create BDAP operation script
    CScript scriptPubKey;
    std::vector<unsigned char> vchFullObjectPath = txDeletedEntry.vchFullObjectPath();
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_DELETE) << CScript::EncodeOP_N(OP_BDAP_ACCOUNT_ENTRY) 
                 << vchFullObjectPath << txDeletedEntry.DHTPublicKey << OP_2DROP << OP_2DROP;

    CDynamicAddress walletAddress(stringFromVch(txDeletedEntry.WalletAddress));
    CScript scriptDestination;
    scriptDestination = GetScriptForDestination(walletAddress.Get());
    scriptPubKey += scriptDestination;

    // Create empty BDAP OP_RETURN script
    CScript scriptData;

    // Send the transaction
    CWalletTx wtx;
    CAmount nOperationFee = (GetBDAPFee(scriptPubKey) * powf(3.1, 1)) + GetDataFee(scriptPubKey);
    CAmount nDataFee = 0; // No OP_RETURN data needed for deleted account transactions

    bool fUseInstantSend = false;
    SendBDAPTransaction(scriptData, scriptPubKey, wtx, nOperationFee, nDataFee, fUseInstantSend);
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

UniValue deleteuser(const JSONRPCRequest& request) 
{
    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "deleteuser"
            "deleteuser \"account id\"\n"
            "\nArguments:\n"
            "1. account id         (string)             The account objectid within public.bdap.io\n"
            "\nDeletes an existing bdap.io public user account entry from the blockchain directory.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"oid\"                        (string)  Account OID\n"
            "  \"version\"                    (int)     Recipient's BDAP full path\n"
            "  \"domain_component\"           (string)  Account domain name\n"
            "  \"common_name\"                (string)  Account common name\n"
            "  \"organizational_unit\"        (string)  Account organizational unit or OU\n"
            "  \"organization_name\"          (string)  Account organizational name\n"
            "  \"object_id\"                  (string)  Account object id\n"
            "  \"object_full_path\"           (string)  Account fully qualified domain name (FQDN)\n"
            "  \"object_type\"                (string)  Account type. User or Group\n"
            "  \"wallet_address\"             (string)  Account wallet address\n"
            "  \"public\"                     (boolean) True when account is public\n"
            "  \"dht_publickey\"              (string)  Account DHT public key\n"
            "  \"link_address\"               (string)  Account link address\n"
            "  \"txid\"                       (string)  Last txid for account data\n"
            "  \"time\"                       (int)     Last transaction time for account data\n"
            "  \"height\"                     (int)     Last transaction height for account data\n"
            "  \"expires_on\"                 (int)     Account expiration epoch time\n"
            "  \"expired\"                    (boolean) Account expired\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("deleteuser", "Alice" ) +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("deleteuser", "Alice"));

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3000 - " + _("Can not create BDAP transactions until spork is active."));

    BDAP::ObjectType bdapType = BDAP::ObjectType::BDAP_USER;
    return DeleteDomainEntry(request, bdapType);
}

UniValue deletegroup(const JSONRPCRequest& request) 
{
    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "deletegroup \"account id\"\n"
            "\nArguments:\n"
            "1. account id        (string)              The account objectid within public.bdap.io\n"
            "\nDeletes an existing bdap.io public group account entry from the blockchain directory.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"oid\"                        (string)  Account OID\n"
            "  \"version\"                    (int)     Recipient's BDAP full path\n"
            "  \"domain_component\"           (string)  Account domain name\n"
            "  \"common_name\"                (string)  Account common name\n"
            "  \"organizational_unit\"        (string)  Account organizational unit or OU\n"
            "  \"organization_name\"          (string)  Account organizational name\n"
            "  \"object_id\"                  (string)  Account object id\n"
            "  \"object_full_path\"           (string)  Account fully qualified domain name (FQDN)\n"
            "  \"object_type\"                (string)  Account type. User or Group\n"
            "  \"wallet_address\"             (string)  Account wallet address\n"
            "  \"public\"                     (boolean) True when account is public\n"
            "  \"dht_publickey\"              (string)  Account DHT public key\n"
            "  \"link_address\"               (string)  Account link address\n"
            "  \"txid\"                       (string)  Last txid for account data\n"
            "  \"time\"                       (int)     Last transaction time for account data\n"
            "  \"height\"                     (int)     Last transaction height for account data\n"
            "  \"expires_on\"                 (int)     Account expiration epoch time\n"
            "  \"expired\"                    (boolean) Account expired\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("deletegroup", "GroupName" ) +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("deletegroup", "GroupName"));

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3000 - " + _("Can not create BDAP transactions until spork is active."));

    BDAP::ObjectType bdapType = BDAP::ObjectType::BDAP_GROUP;
    return DeleteDomainEntry(request, bdapType);
}

UniValue makekeypair(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() > 1)
        throw std::runtime_error(
            "makekeypair \"prefix\"\n"
            "\nArguments:\n"
            "1. prefix                (string, optional)     preferred prefix for the public key\n"
            "\nCreates a new public/private key pair without adding them to the local wallet.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"Private Key\"        (string)  Private key\n"
            "  \"Public Key\"         (string)  Public key\n"
            "  \"Wallet Address\"     (string)  Wallet address\n"
            "  \"Wallet Private Key\" (string)  Wallet address private key\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("deletegroup", "GroupName" ) +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("deletegroup", "GroupName"));

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
    if (request.fHelp || request.params.size() < 2 || request.params.size() > 3)
        throw std::runtime_error(
            "addgroup \"account id\" \"common name\" \"registration days\"\n"
            "\nArguments:\n"
            "1. account id         (string)             The new group account id\n"
            "2. common name        (string)             The group account common name used for searching\n"
            "3. registration days  (int, optional)      Number of days to register the account\n"
            "\nAdds a new bdap.io public group account entry to the blockchain directory.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"oid\"                        (string)  Account OID\n"
            "  \"version\"                    (int)     Recipient's BDAP full path\n"
            "  \"domain_component\"           (string)  Account domain name\n"
            "  \"common_name\"                (string)  Account common name\n"
            "  \"organizational_unit\"        (string)  Account organizational unit or OU\n"
            "  \"organization_name\"          (string)  Account organizational name\n"
            "  \"object_id\"                  (string)  Account object ID\n"
            "  \"object_full_path\"           (string)  Account fully qualified domain name (FQDN)\n"
            "  \"object_type\"                (string)  Account type. User or Group\n"
            "  \"wallet_address\"             (string)  Account wallet address\n"
            "  \"public\"                     (boolean) True when account is public\n"
            "  \"dht_publickey\"              (string)  Account DHT public key\n"
            "  \"link_address\"               (string)  Account link address\n"
            "  \"txid\"                       (string)  Last txid for account data\n"
            "  \"time\"                       (int)     Last transaction time for account data\n"
            "  \"height\"                     (int)     Last transaction height for account data\n"
            "  \"expires_on\"                 (int)     Account expiration epoch time\n"
            "  \"expired\"                    (boolean) Account expired\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("addgroup", "Duality \"Duality Blockchain Solutions Group\"") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("addgroup", "Duality \"Duality Blockchain Solutions Group\""));

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3000 - " + _("Can not create BDAP transactions until spork is active."));

    BDAP::ObjectType bdapType = BDAP::ObjectType::BDAP_GROUP;
    return AddDomainEntry(request, bdapType);
}

UniValue mybdapaccounts(const JSONRPCRequest& request)
{
    if (request.params.size() > 1)
        throw std::runtime_error(
            "mybdapaccounts\n"
            + HelpRequiringPassphrase() +
            "\nReturns a list of your BDAP accounts.\n"
            "\nResult:\n"
            "{(json objects)\n"
            "  \"common_name\"                (string)  BDAP account common name\n"
            "  \"object_full_path\"           (string)  BDAP account full path\n"
            "  \"wallet_address\"             (string)  BDAP account wallet address\n"
            "  \"dht_publickey\"              (string)  BDAP account DHT pubkey\n"
            "  \"link_address\"               (string)  BDAP account link address\n"
            "  \"object_type\"                (string)  Type of object (User or Group)\n"
            "  },...n \n"
            "\nExamples:\n"
            + HelpExampleCli("mybdapaccounts", "") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("mybdapaccounts", ""));

    if (!pwalletMain)
        throw std::runtime_error("MY_BDAP_ACCOUNTS_RPC_ERROR: ERRCODE: 3800 - " + _("Error accessing wallet."));

    std::string accountType {""};

    if (request.params.size() == 1) 
        accountType = request.params[0].get_str();

    if (!((accountType == "users") || (accountType == "groups") || (accountType == "")))
        throw std::runtime_error("MY_BDAP_ACCOUNTS_RPC_ERROR: ERRCODE: 3801 - " + _("Unkown account type"));
  
    std::vector<std::vector<unsigned char>> vvchDHTPubKeys;
    if (!pwalletMain->GetDHTPubKeys(vvchDHTPubKeys))
        return NullUniValue;

    UniValue result(UniValue::VOBJ);
    uint32_t nCount = 1;
    for (const std::vector<unsigned char>& vchPubKey : vvchDHTPubKeys) {
        CDomainEntry entry;
        if (pDomainEntryDB->ReadDomainEntryPubKey(vchPubKey, entry)) {
            UniValue oAccount(UniValue::VOBJ);
            if (BuildBDAPJson(entry, oAccount, false)) {
                if ( (accountType == "") || ((accountType == "users") && (entry.nObjectType == GetObjectTypeInt(BDAP::ObjectType::BDAP_USER))) || ((accountType == "groups") && (entry.nObjectType == GetObjectTypeInt(BDAP::ObjectType::BDAP_GROUP))) ) {
                    result.push_back(Pair("account_" + std::to_string(nCount) , oAccount));
                    nCount++;
                } //if accountType
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
    { "bdap",            "adduser",                  &adduser,                      true, {"account id", "common name", "registration days"} },
    { "bdap",            "getusers",                 &getusers,                     true, {"records per page", "page returned"} },
    { "bdap",            "getgroups",                &getgroups,                    true, {"records per page", "page returned"} },
    { "bdap",            "getuserinfo",              &getuserinfo,                  true, {"account id"} },
    { "bdap",            "updateuser",               &updateuser,                   true, {"account id", "common name", "registration days"} },
    { "bdap",            "updategroup",              &updategroup,                  true, {"account id", "common name", "registration days"} },
    { "bdap",            "deleteuser",               &deleteuser,                   true, {"account id"} },
    { "bdap",            "deletegroup",              &deletegroup,                  true, {"account id"} },
    { "bdap",            "addgroup",                 &addgroup,                     true, {"account id", "common name", "registration days"} },
    { "bdap",            "getgroupinfo",             &getgroupinfo,                 true, {"account id"} },
    { "bdap",            "mybdapaccounts",           &mybdapaccounts,               true, {} },
#endif //ENABLE_WALLET
    { "bdap",            "makekeypair",              &makekeypair,                  true, {"prefix"} },
};

void RegisterDomainEntryRPCCommands(CRPCTable &t)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        t.appendCommand(commands[vcidx].name, &commands[vcidx]);
}