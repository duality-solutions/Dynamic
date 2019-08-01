// Copyright (c) 2019 Duality Blockchain Solutions Developers 
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/domainentry.h"
#include "bdap/domainentrydb.h"
#include "bdap/fees.h"
#include "bdap/utils.h"
#include "core_io.h" // needed for ScriptToAsmStr
#include "dynodeman.h"
#include "rpcprotocol.h"
#include "rpcserver.h"
#include "primitives/transaction.h"
#include "spork.h"
#include "wallet/wallet.h"
#include "utilmoneystr.h"
#include "validation.h"
#include "dynode-sync.h"

#include <univalue.h>

extern void SendBDAPTransaction(const CScript& bdapDataScript, const CScript& bdapOPScript, CWalletTx& wtxNew, const CAmount& nDataAmount, const CAmount& nOpAmount, const bool fUseInstantSend);
extern void SendColorTransaction(const CScript& scriptColorCoins, CWalletTx& wtxNew, const CAmount& nColorAmount, const CCoinControl* coinControl, const bool fUseInstantSend, const bool fUsePrivateSend);

static UniValue AddDomainEntry(const JSONRPCRequest& request, BDAP::ObjectType bdapType)
{
    EnsureWalletIsUnlocked();

    // Adds a new name to channel zero.  OID = 2.16.840.1.114564.block-height.tx-ordinal
    // Format object and domain names to lower case.
    std::string strObjectID = request.params[0].get_str();
    ToLowerCase(strObjectID);

    CharString vchObjectID = vchFromValue(request.params[0]);
    ToLowerCase(vchObjectID);
    CharString vchCommonName = vchFromValue(request.params[1]);

    CDomainEntry txDomainEntry;
    txDomainEntry.RootOID = vchDefaultOIDPrefix;
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

    CPubKey pubWalletKey;
    CharString vchDHTPubKey;
    CStealthAddress sxAddr;
    if (!pwalletMain->GetKeysFromPool(pubWalletKey, vchDHTPubKey, sxAddr, true))
        throw JSONRPCError(RPC_WALLET_KEYPOOL_RAN_OUT, "Error: Keypool ran out, please call keypoolrefill first");

    CKeyID keyWalletID = pubWalletKey.GetID();
    CDynamicAddress walletAddress = CDynamicAddress(keyWalletID);

    pwalletMain->SetAddressBook(keyWalletID, strObjectID, "bdap-wallet");
    txDomainEntry.WalletAddress = vchFromString(walletAddress.ToString());

    txDomainEntry.DHTPublicKey = vchDHTPubKey;
    CKeyID vchDHTPubKeyID = GetIdFromCharVector(vchDHTPubKey); 
    pwalletMain->SetAddressBook(vchDHTPubKeyID, strObjectID, "bdap-dht-key"); 

    //pwalletMain->SetAddressBook(keyLinkID, strObjectID, "bdap-link");

    txDomainEntry.LinkAddress = vchFromString(sxAddr.ToString());

    int32_t nMonths = DEFAULT_REGISTRATION_MONTHS; // default to 1 year or 12 months
    if (request.params.size() >= 3) {
        if (!ParseInt32(request.params[2].get_str(), &nMonths))
            throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3505 - " + _("Error converting registration months to int, only whole numbers allowed (no decimals)"));
        if (nMonths <= 0)
            throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3505 - " + _("Error: registration months must be greater than 0"));
    }

    CharString data;
    txDomainEntry.Serialize(data);

    // Create BDAP operation script
    CScript scriptPubKey;
    std::string strMonths = std::to_string(nMonths) + "Month";
    std::vector<unsigned char> vchMonths = vchFromString(strMonths);
    std::vector<unsigned char> vchFullObjectPath = txDomainEntry.vchFullObjectPath();
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_NEW) << CScript::EncodeOP_N(OP_BDAP_ACCOUNT_ENTRY) 
                 << vchFullObjectPath << txDomainEntry.DHTPublicKey << vchMonths << OP_2DROP << OP_2DROP << OP_DROP;

    CScript scriptDestination;
    scriptDestination = GetScriptForDestination(walletAddress.Get());
    scriptPubKey += scriptDestination;

    // Create BDAP OP_RETURN script
    CScript scriptData;
    scriptData << OP_RETURN << data;

    // Get BDAP fees
    CAmount monthlyFee, oneTimeFee, depositFee;
    if (!GetBDAPFees(OP_BDAP_NEW, OP_BDAP_ACCOUNT_ENTRY, bdapType, nMonths, monthlyFee, oneTimeFee, depositFee))
        throw JSONRPCError(RPC_BDAP_FEE_UNKNOWN, strprintf("Error calculating BDAP fees."));
    LogPrint("bdap", "%s -- monthlyFee %d, oneTimeFee %d, depositFee %d\n", __func__, monthlyFee, oneTimeFee, depositFee);
    // check BDAP values
    std::string strMessage;
    if (!txDomainEntry.ValidateValues(strMessage))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3506 - " + strMessage);

    bool fUseInstantSend = false;
    //if (dnodeman.EnoughActiveForInstandSend() && sporkManager.IsSporkActive(SPORK_2_INSTANTSEND_ENABLED))
    //    fUseInstantSend = true;

    // Send the transaction
    CWalletTx wtx;
    SendBDAPTransaction(scriptData, scriptPubKey, wtx, monthlyFee, oneTimeFee + depositFee, fUseInstantSend);
    txDomainEntry.txHash = wtx.GetHash();

    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txDomainEntry, oName))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3507 - " + _("Failed to read from BDAP JSON object"));
    
    if (LogAcceptCategory("bdap")) {
        // make sure we can deserialize the transaction from the scriptData and get a valid CDomainEntry class
        LogPrint("bdap", "DomainEntry Scripts:\nscriptData = %s\n", ScriptToAsmStr(scriptData, true));

        const CTransactionRef testTx = MakeTransactionRef((CTransaction)wtx);
        CDomainEntry testDomainEntry(testTx); //loads the class from a transaction

        LogPrint("bdap", "CDomainEntry Values:\nnVersion = %u\nFullObjectPath = %s\nCommonName = %s\nOrganizationalUnit = %s\nDHTPublicKey = %s\n", 
            testDomainEntry.nVersion, testDomainEntry.GetFullObjectPath(), stringFromVch(testDomainEntry.CommonName), 
            stringFromVch(testDomainEntry.OrganizationalUnit), stringFromVch(testDomainEntry.DHTPublicKey));
    }
    std::string strPubKeyHash = GetHashFromCharVector(vchDHTPubKey).ToString();
    std::string strPubKeyID = GetIdFromCharVector(vchDHTPubKey).ToString();

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
            "1. account id           (string)             The account userid\n"
            "2. common name          (string)             The account common name used for searching\n"
            "3. registration months  (int, optional)      Number of months to register account\n"
            "\nAdds a new bdap.io public name account entry to the blockchain directory.\n"
            "\nResult:\n"
            "{(json objects)\n"
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
            "  },...n \n"
            "\nExamples\n" +
           HelpExampleCli("adduser", "Alice \"Wonderland, Alice\"") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("adduser", "Alice \"Wonderland, Alice\""));

    if (!dynodeSync.IsBlockchainSynced()) {
        throw std::runtime_error("Error: Cannot create BDAP Objects while wallet is not synced.");
    }

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3000 - " + _("Can not create BDAP transactions until spork is active."));

    BDAP::ObjectType bdapType = BDAP::ObjectType::BDAP_USER;
    return AddDomainEntry(request, bdapType);
}

UniValue getusers(const JSONRPCRequest& request) 
{
    if (request.fHelp || request.params.size() > 3 || request.params.size() == 2)
        throw std::runtime_error(
            "getusers \"search string\" \"records per page\" \"page returned\"\n"
            "\nArguments:\n"
            "1. search string        (string, optional)  Search for userid\n"
            "2. records per page     (int, optional)  If paging, the number of records per page\n"
            "3. page returned        (int, optional)  If paging, the page number to return\n"
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

    int nRecordsPerPage = 100;
    int nPage = 1;
    std::string searchString = "";

    if (request.params.size() > 0)
    {
         searchString = request.params[0].get_str();
    }

    if (request.params.size() == 3)
    {
        try {
            nRecordsPerPage = atoi(request.params[1].get_str());
            nPage = atoi(request.params[2].get_str());

            if ((nRecordsPerPage <= 0) || (nPage <= 0))
            {
                throw std::runtime_error("Error: parameters must be positive integers");
                return NullUniValue;
            }       
        } catch (std::exception& e) {
            throw std::runtime_error("Error: parameters must be positive integers");
            return NullUniValue;
        }
    }
    
    // only return entries from the default public domain OU
    std::string strObjectLocation = DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    CharString vchObjectLocation(strObjectLocation.begin(), strObjectLocation.end());

    UniValue oDomainEntryList(UniValue::VARR);
    if (CheckDomainEntryDB())
        pDomainEntryDB->ListDirectories(vchObjectLocation, nRecordsPerPage, nPage, oDomainEntryList, BDAP::ObjectType::BDAP_USER, searchString);

    return oDomainEntryList;
}

UniValue getgroups(const JSONRPCRequest& request) 
{
    if (request.fHelp || request.params.size() > 3 || request.params.size() == 2)
        throw std::runtime_error(
            "getgroups \"search string\" \"records per page\" \"page returned\"\n"
            "\nArguments:\n"
            "1. search string        (string, optional)  Search for userid\n"
            "2. records per page     (int, optional)  If paging, the number of records per page\n"
            "3. page returned        (int, optional)  If paging, the page number to return\n"
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

    int nRecordsPerPage = 100;
    int nPage = 1;
    std::string searchString = "";

    if (request.params.size() > 0)
    {
         searchString = request.params[0].get_str();
    }

    if (request.params.size() == 3)
    {
        try {
            nRecordsPerPage = atoi(request.params[1].get_str());
            nPage = atoi(request.params[2].get_str());

            if ((nRecordsPerPage <= 0) || (nPage <= 0))
            {
                throw std::runtime_error("Error: parameters must be positive integers");
                return NullUniValue;
            }       
        } catch (std::exception& e) {
            throw std::runtime_error("Error: parameters must be positive integers");
            return NullUniValue;
        }
    }
    
    // only return entries from the default public domain OU
    std::string strObjectLocation = DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    CharString vchObjectLocation(strObjectLocation.begin(), strObjectLocation.end());

    UniValue oDomainEntryList(UniValue::VARR);
    if (CheckDomainEntryDB())
        pDomainEntryDB->ListDirectories(vchObjectLocation, nRecordsPerPage, nPage, oDomainEntryList, BDAP::ObjectType::BDAP_GROUP, searchString);

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

    int32_t nMonths = 0;
    if (request.params.size() >= 3) {
        if (!ParseInt32(request.params[2].get_str(), &nMonths))
            throw std::runtime_error("BDAP_UPDATE_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3702 - " + _("Error converting registration days to int"));
    }
    //txUpdatedEntry.nExpireTime = AddMonthsToCurrentEpoch((short)nMonths);

    CharString data;
    txUpdatedEntry.Serialize(data);
    
    // Create BDAP operation script
    CScript scriptPubKey;
    std::string strMonths = std::to_string(nMonths) + "Month";
    std::vector<unsigned char> vchMonths = vchFromString(strMonths);
    std::vector<unsigned char> vchFullObjectPath = txUpdatedEntry.vchFullObjectPath();
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_MODIFY) << CScript::EncodeOP_N(OP_BDAP_ACCOUNT_ENTRY) 
                 << vchFullObjectPath << txUpdatedEntry.DHTPublicKey << vchMonths << OP_2DROP << OP_2DROP << OP_DROP;

    CDynamicAddress walletAddress(stringFromVch(txUpdatedEntry.WalletAddress));
    CScript scriptDestination;
    scriptDestination = GetScriptForDestination(walletAddress.Get());
    scriptPubKey += scriptDestination;

    // Create BDAP OP_RETURN script
    CScript scriptData;
    scriptData << OP_RETURN << data;

    // Get BDAP fees
    CAmount monthlyFee, oneTimeFee, depositFee;
    if (!GetBDAPFees(OP_BDAP_MODIFY, OP_BDAP_ACCOUNT_ENTRY, bdapType, nMonths, monthlyFee, oneTimeFee, depositFee))
        throw JSONRPCError(RPC_BDAP_FEE_UNKNOWN, strprintf("Error calculating BDAP fees."));
    LogPrint("bdap", "%s -- monthlyFee %d, oneTimeFee %d, depositFee %d\n", __func__, monthlyFee, oneTimeFee, depositFee);
    // check BDAP values
    std::string strMessage;
    if (!txUpdatedEntry.ValidateValues(strMessage))
        throw std::runtime_error("BDAP_UPDATE_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3703 - " + strMessage);

    bool fUseInstantSend = false;
    //if (dnodeman.EnoughActiveForInstandSend() && sporkManager.IsSporkActive(SPORK_2_INSTANTSEND_ENABLED))
    //    fUseInstantSend = true;

    // Send the transaction
    CWalletTx wtx;
    SendBDAPTransaction(scriptData, scriptPubKey, wtx, monthlyFee, oneTimeFee + depositFee, fUseInstantSend);
    txUpdatedEntry.txHash = wtx.GetHash();

    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txUpdatedEntry, oName))
        throw std::runtime_error("BDAP_UPDATE_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3704 - " + _("Failed to read from BDAP JSON object"));
    
    if (LogAcceptCategory("bdap")) {
        // make sure we can deserialize the transaction from the scriptData and get a valid CDomainEntry class
        LogPrint("bdap", "DomainEntry Scripts:\nscriptData = %s\n", ScriptToAsmStr(scriptData, true));

        const CTransactionRef testTx = MakeTransactionRef((CTransaction)wtx);
        CDomainEntry testDomainEntry(testTx); //loads the class from a transaction

        LogPrint("bdap", "CDomainEntry Values:\nnVersion = %u\nFullObjectPath = %s\nCommonName = %s\nOrganizationalUnit = %s\nDHTPublicKey = %s\n", 
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
            "1. account id           (string)             The account objectid within public.bdap.io\n"
            "2. common name          (string)             The account common name used for searching\n"
            "3. registration months  (int, optional)      Number of additional months to register the account\n"
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

    if (!dynodeSync.IsBlockchainSynced()) {
        throw std::runtime_error("Error: Cannot create BDAP Objects while wallet is not synced.");
    }

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
            "1. groupid              (string)             The account objectid within public.bdap.io\n"
            "2. common name          (string)             The account common name used for searching\n"
            "3. registration months  (int, optional)      Number of additional months to register the account\n"
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

    if (!dynodeSync.IsBlockchainSynced()) {
        throw std::runtime_error("Error: Cannot create BDAP Objects while wallet is not synced.");
    }

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

    // Get BDAP fees
    uint16_t nMonths = 0;
    CAmount monthlyFee, oneTimeFee, depositFee;
    if (!GetBDAPFees(OP_BDAP_DELETE, OP_BDAP_ACCOUNT_ENTRY, bdapType, nMonths, monthlyFee, oneTimeFee, depositFee))
        throw JSONRPCError(RPC_BDAP_FEE_UNKNOWN, strprintf("Error calculating BDAP fees."));
    LogPrint("bdap", "%s -- monthlyFee %d, oneTimeFee %d, depositFee %d\n", __func__, monthlyFee, oneTimeFee, depositFee);

    // Send the transaction
    CWalletTx wtx;
    bool fUseInstantSend = false;
    SendBDAPTransaction(scriptData, scriptPubKey, wtx, monthlyFee, oneTimeFee + depositFee, fUseInstantSend);
    txDeletedEntry.txHash = wtx.GetHash();

    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txDeletedEntry, oName))
        throw std::runtime_error("BDAP_DELETE_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3703 - " + _("Failed to read from BDAP JSON object"));
    
    if (LogAcceptCategory("bdap")) {
        // make sure we can deserialize the transaction from the scriptData and get a valid CDomainEntry class
        LogPrint("bdap", "DomainEntry Scripts:\nscriptData = %s\n", ScriptToAsmStr(scriptData, true));

        const CTransactionRef testTx = MakeTransactionRef((CTransaction)wtx);
        CDomainEntry testDomainEntry(testTx); //loads the class from a transaction

        LogPrint("bdap", "CDomainEntry Values:\nnVersion = %u\nFullObjectPath = %s\nCommonName = %s\nOrganizationalUnit = %s\nDHTPublicKey = %s\n", 
            testDomainEntry.nVersion, testDomainEntry.GetFullObjectPath(), stringFromVch(testDomainEntry.CommonName), 
            stringFromVch(testDomainEntry.OrganizationalUnit), stringFromVch(testDomainEntry.DHTPublicKey));
    }

    return oName;
}

UniValue deleteuser(const JSONRPCRequest& request) 
{
    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
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

    if (!dynodeSync.IsBlockchainSynced()) {
        throw std::runtime_error("Error: Cannot create BDAP Objects while wallet is not synced.");
    }

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

    if (!dynodeSync.IsBlockchainSynced()) {
        throw std::runtime_error("Error: Cannot create BDAP Objects while wallet is not synced.");
    }

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
        key.MakeNewKey(true);
        nCount++;
    } while (nCount < 10000 && strPrefix != HexStr(key.GetPubKey()).substr(0, strPrefix.size()));

    if (strPrefix != HexStr(key.GetPubKey()).substr(0, strPrefix.size()))
        return NullUniValue;

    UniValue result(UniValue::VOBJ);

    //key.SetCompressedBoolean(true);
   //get compressed versions
    CPrivKey vchPrivKeyC = key.GetPrivKey();
    CKeyID keyIDC = key.GetPubKey().GetID();
    CKey vchSecretC = CKey();
    vchSecretC.SetPrivKey(vchPrivKeyC, true);    
    result.push_back(Pair("private_key", HexStr<CPrivKey::iterator>(vchPrivKeyC.begin(), vchPrivKeyC.end())));
    result.push_back(Pair("public_key", HexStr(key.GetPubKey())));
    result.push_back(Pair("address", CDynamicAddress(keyIDC).ToString()));
    result.push_back(Pair("address_private_key", CDynamicSecret(vchSecretC).ToString()));

   //get uncompressed versions
    key.SetCompressedBoolean(false); //toggle to false
    CPrivKey vchPrivKey = key.GetPrivKey();
    CKeyID keyID = key.GetPubKey().GetID();
    CKey vchSecret = CKey();
    vchSecret.SetPrivKey(vchPrivKey, false);
    //result.push_back(Pair("private_key_uncompressed", HexStr<CPrivKey::iterator>(vchPrivKey.begin(), vchPrivKey.end())));
    result.push_back(Pair("public_key_uncompressed", HexStr(key.GetPubKey())));
    result.push_back(Pair("address_uncompressed", CDynamicAddress(keyID).ToString()));
    result.push_back(Pair("address_private_key_uncompressed", CDynamicSecret(vchSecret).ToString()));
 
    return result;
}

UniValue addgroup(const JSONRPCRequest& request) 
{
    if (request.fHelp || request.params.size() < 2 || request.params.size() > 3)
        throw std::runtime_error(
            "addgroup \"account id\" \"common name\" \"registration days\"\n"
            "\nArguments:\n"
            "1. account id           (string)             The new group account id\n"
            "2. common name          (string)             The group account common name used for searching\n"
            "3. registration months  (int, optional)      Number of months to register the account\n"
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

    if (!dynodeSync.IsBlockchainSynced()) {
        throw std::runtime_error("Error: Cannot create BDAP Objects while wallet is not synced.");
    }

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

UniValue makecredits(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() < 2 || request.params.size() > 3)
        throw std::runtime_error(
            "makecredits \"dynamicaddress\" \"amount\"\n"
            "\nConvert your Dynamic (DYN) to BDAP colored credits\n"
            + HelpRequiringPassphrase() +
            "\nArguments:\n"
            "1. \"dynamicaddress\"       (string)            The destination wallet address\n"
            "2. \"amount\"               (int)               The amount in " + CURRENCY_UNIT + " to color. eg 0.1\n"
            "\nResult:\n"
            "  \"tx id\"                 (string)            The transaction id for the coin coloring\n"
            "\nExamples:\n"
            + HelpExampleCli("makecredits", "\"DKkDJn9bjoXJiiPysSVEeUc3ve6SaWLzVv\" 100.98") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("makecredits", "\"DKkDJn9bjoXJiiPysSVEeUc3ve6SaWLzVv\" 100.98"));

    EnsureWalletIsUnlocked();

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw JSONRPCError(RPC_BDAP_SPORK_INACTIVE, strprintf("Can not use the makecredits RPC command until the BDAP spork is active."));

    if (!pwalletMain)
        throw JSONRPCError(RPC_WALLET_ERROR, strprintf("Error accessing wallet."));

    CTxDestination dest = DecodeDestination(request.params[0].get_str());
    if (!IsValidDestination(dest))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid address");

    CAmount nColorAmount = AmountFromValue(request.params[1]);
    if (nColorAmount <= 0)
        throw JSONRPCError(RPC_TYPE_ERROR, "Invalid amount for coloring");

    std::vector<unsigned char> vchMoveSource = vchFromString(std::string("DYN"));
    std::vector<unsigned char> vchMoveDestination = vchFromString(std::string("BDAP"));

    // Create BDAP move asset operation script
    CScript scriptColorCoins;
    scriptColorCoins << CScript::EncodeOP_N(OP_BDAP_MOVE) << CScript::EncodeOP_N(OP_BDAP_ASSET) 
                        << vchMoveSource << vchMoveDestination << OP_2DROP << OP_2DROP;

    CScript scriptDestination;
    scriptDestination = GetScriptForDestination(dest);
    scriptColorCoins += scriptDestination;

    CWalletTx wtx;
    SendColorTransaction(scriptColorCoins, wtx, nColorAmount, NULL, false, false);

    return wtx.GetHash().GetHex();
}

UniValue getcredits(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 0)
        throw std::runtime_error(
            "getcredits\n"
            "\nGet available BDAP credit balance\n"
            + HelpRequiringPassphrase() +
            "\nResult:\n"
            "{(json objects)\n"
            "  \"type\"                    (string)            The credit type.\n"
            "  \"operation\"               (string)            The operation code used in the tx output\n"
            "  \"address\"                 (string)            The address holding the unspent BDAP credits\n"
            "  \"dynamic_amount\"          (int)               The unspent BDAP amount int DYN\n"
            "  \"credits\"                 (int)               The unspent BDAP credits\n"
            "  },...n \n"
            "\"total_credits\"             (int)               The total credits available.\n"
            "\"total_deposits\"            (int)               The total deposits available.\n"
            "\"total_dynamic\"             (int)               The total Dynamic available that are BDAP colored.\n"
            "\nExamples:\n"
            + HelpExampleCli("getcredits", "") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("getcredits", ""));

    if (!pwalletMain)
        throw JSONRPCError(RPC_WALLET_ERROR, strprintf("Error accessing wallet."));

    std::vector<std::pair<CTxOut, COutPoint>> vCredits;
    pwalletMain->AvailableBDAPCredits(vCredits);

    CAmount nTotalAmount = 0;
    CAmount nTotalCredits = 0;
    CAmount nTotalDeposits = 0;
    UniValue result(UniValue::VOBJ);
    for (const std::pair<CTxOut, COutPoint>& credit : vCredits) {
        UniValue oCredit(UniValue::VOBJ);
        int opCode1 = -1; int opCode2 = -1;
        std::vector<std::vector<unsigned char>> vvch;
        credit.first.GetBDAPOpCodes(opCode1, opCode2, vvch);
        std::string strOpType = GetBDAPOpTypeString(opCode1, opCode2);
        const CDynamicAddress address = GetScriptAddress(credit.first.scriptPubKey);
        std::string strType = "unknown";
        std::string strAccount =  "";
        std::string strPubKey =  "";
        std::string strSharedPubKey =  "";
        if (strOpType == "bdap_new_account" || strOpType == "bdap_update_account") {
            strType = "account deposit";
            if (vvch.size() > 0)
                strAccount = stringFromVch(vvch[0]);
            if (vvch.size() > 1)
                strPubKey = stringFromVch(vvch[1]);
            nTotalDeposits += credit.first.nValue;
        } else if (strOpType == "bdap_new_link_request" || strOpType == "bdap_new_link_accept" || 
                    strOpType == "bdap_delete_link_request" || strOpType == "bdap_delete_link_accept") {
            strType = "link deposit";
            if (vvch.size() > 0)
                strPubKey = stringFromVch(vvch[0]);
            if (vvch.size() > 1)
                strSharedPubKey = stringFromVch(vvch[1]);
            nTotalDeposits += credit.first.nValue;
        } else if (strOpType == "bdap_move_asset") {
            if (vvch.size() > 1) {
                std::string strMoveSource = stringFromVch(vvch[0]);
                std::string strMoveDestination = stringFromVch(vvch[1]);
                strType = strprintf("credit (%s to %s)", strMoveSource, strMoveDestination);
                if (strMoveSource == "DYN" && strMoveDestination == "BDAP")
                    nTotalCredits += credit.first.nValue;
            } else {
                strType = strprintf("credit (unknown)");
            }
        }
        oCredit.push_back(Pair("type", strType));
        oCredit.push_back(Pair("operation", strOpType));
        if (strAccount.size() > 0)
            oCredit.push_back(Pair("account", strAccount));
        oCredit.push_back(Pair("address", address.ToString()));
        if (strPubKey.size() > 0)
            oCredit.push_back(Pair("pubkey", strPubKey));
        if (strSharedPubKey.size() > 0)
            oCredit.push_back(Pair("shared_pubkey", strSharedPubKey));
        oCredit.push_back(Pair("dynamic_amount", FormatMoney(credit.first.nValue)));
        oCredit.push_back(Pair("credits", credit.first.nValue/BDAP_CREDIT));
        std::string strOutput = credit.second.hash.ToString() + "-" + std::to_string(credit.second.n);
        result.push_back(Pair(strOutput, oCredit));
        nTotalAmount += credit.first.nValue;
    }
    // Add total amounts and credits

    result.push_back(Pair("total_credits", nTotalCredits/BDAP_CREDIT));
    result.push_back(Pair("total_deposits", nTotalDeposits/BDAP_CREDIT));
    result.push_back(Pair("total_dynamic", FormatMoney(nTotalAmount)));

    return result;
}

UniValue bdapfees(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 0)
        throw std::runtime_error(
            "bdapfees\n"
            "\nGet current BDAP fee schedule\n"
            + HelpRequiringPassphrase() +
            "\nResult:\n"
            "{(json objects)\n"
            "  \"monthly\"                    (int)            The credit type.\n"
            "  \"deposit\"                    (int)            The operation code used in the tx output\n"
            "  \"one-time\"                   (int)            The unspent BDAP amount int DYN\n"
            "  },...n \n"
            "\nExamples:\n"
            + HelpExampleCli("bdapfees", "") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("bdapfees", ""));

    UniValue oFees(UniValue::VOBJ);

    const uint16_t nMonths = 1;
    CAmount monthlyFee, oneTimeFee, depositFee;
    GetBDAPFees(OP_BDAP_NEW, OP_BDAP_ACCOUNT_ENTRY, BDAP::ObjectType::BDAP_USER, nMonths, monthlyFee, oneTimeFee, depositFee);
    CAmount nNewUserMonthly = 0;
    CAmount nNewUserDeposit = 0;
    CAmount nNewUserOnetime = 0;
    UniValue oNewUserFee(UniValue::VOBJ);
    nNewUserMonthly = monthlyFee;
    nNewUserDeposit = depositFee;
    nNewUserOnetime = oneTimeFee;
    oNewUserFee.push_back(Pair("monthly_dynamic", FormatMoney(nNewUserMonthly)));
    oNewUserFee.push_back(Pair("monthly_credits", nNewUserMonthly/BDAP_CREDIT));
    oNewUserFee.push_back(Pair("deposit_dynamic", FormatMoney(nNewUserDeposit)));
    oNewUserFee.push_back(Pair("deposit_credits", nNewUserDeposit/BDAP_CREDIT));
    oNewUserFee.push_back(Pair("one-time", FormatMoney(nNewUserOnetime)));
    oNewUserFee.push_back(Pair("one-time_credits", nNewUserOnetime/BDAP_CREDIT));
    oNewUserFee.push_back(Pair("total_dynamic", FormatMoney((nNewUserMonthly + nNewUserDeposit + nNewUserOnetime))));
    oNewUserFee.push_back(Pair("total_credits", (nNewUserMonthly + nNewUserDeposit + nNewUserOnetime)/BDAP_CREDIT));
    oFees.push_back(Pair("new_user", oNewUserFee));

    GetBDAPFees(OP_BDAP_MODIFY, OP_BDAP_ACCOUNT_ENTRY, BDAP::ObjectType::BDAP_USER, nMonths, monthlyFee, oneTimeFee, depositFee);
    CAmount nUpdateUserMonthly = 0;
    CAmount nUpdateUserDeposit = 0;
    CAmount nUpdateUserOnetime = 0;
    UniValue oUpdateUserFee(UniValue::VOBJ);
    nUpdateUserMonthly = monthlyFee;
    nUpdateUserDeposit = depositFee;
    nUpdateUserOnetime = oneTimeFee;
    oUpdateUserFee.push_back(Pair("monthly_dynamic", FormatMoney(nUpdateUserMonthly)));
    oUpdateUserFee.push_back(Pair("monthly_credits", nUpdateUserMonthly/BDAP_CREDIT));
    oUpdateUserFee.push_back(Pair("deposit_dynamic", FormatMoney(nUpdateUserDeposit)));
    oUpdateUserFee.push_back(Pair("deposit_credits", nUpdateUserDeposit/BDAP_CREDIT));
    oUpdateUserFee.push_back(Pair("one-time_dynamic", FormatMoney(nUpdateUserOnetime)));
    oUpdateUserFee.push_back(Pair("one-time_credits", nUpdateUserOnetime/BDAP_CREDIT));
    oUpdateUserFee.push_back(Pair("total_dynamic", FormatMoney((nUpdateUserMonthly + nUpdateUserDeposit + nUpdateUserOnetime))));
    oUpdateUserFee.push_back(Pair("total_credits", (nUpdateUserMonthly + nUpdateUserDeposit + nUpdateUserOnetime)/BDAP_CREDIT));
    oFees.push_back(Pair("update_user", oUpdateUserFee));

    GetBDAPFees(OP_BDAP_NEW, OP_BDAP_LINK_REQUEST, BDAP::ObjectType::BDAP_LINK_REQUEST, nMonths, monthlyFee, oneTimeFee, depositFee);
    CAmount nLinkRequestMonthly = 0;
    CAmount nLinkRequestDeposit = 0;
    CAmount nLinkRequestOnetime = 0;
    UniValue oLinkRequestFee(UniValue::VOBJ);
    nLinkRequestMonthly = monthlyFee;
    nLinkRequestDeposit = depositFee;
    nLinkRequestOnetime = oneTimeFee;
    oLinkRequestFee.push_back(Pair("monthly_dynamic", FormatMoney(nLinkRequestMonthly)));
    oLinkRequestFee.push_back(Pair("monthly_credits", nLinkRequestMonthly/BDAP_CREDIT));
    oLinkRequestFee.push_back(Pair("deposit_dynamic", FormatMoney(nLinkRequestDeposit)));
    oLinkRequestFee.push_back(Pair("deposit_credits", nLinkRequestDeposit/BDAP_CREDIT));
    oLinkRequestFee.push_back(Pair("one-time_dynamic", FormatMoney(nLinkRequestOnetime)));
    oLinkRequestFee.push_back(Pair("one-time_credits", nLinkRequestOnetime/BDAP_CREDIT));
    oLinkRequestFee.push_back(Pair("total_dynamic", FormatMoney((nLinkRequestMonthly + nLinkRequestDeposit + nLinkRequestOnetime))));
    oLinkRequestFee.push_back(Pair("total_credits", (nLinkRequestMonthly + nLinkRequestDeposit + nLinkRequestOnetime)/BDAP_CREDIT));
    oFees.push_back(Pair("new_link_request", oLinkRequestFee));

    GetBDAPFees(OP_BDAP_NEW, OP_BDAP_LINK_ACCEPT, BDAP::ObjectType::BDAP_LINK_ACCEPT, nMonths, monthlyFee, oneTimeFee, depositFee);
    CAmount nLinkAcceptMonthly = 0;
    CAmount nLinkAcceptDeposit = 0;
    CAmount nLinkAcceptOnetime = 0;
    UniValue oLinkAcceptFee(UniValue::VOBJ);
    nLinkAcceptMonthly = monthlyFee;
    nLinkAcceptDeposit = depositFee;
    nLinkAcceptOnetime = oneTimeFee;
    oLinkAcceptFee.push_back(Pair("monthly_dynamic", FormatMoney(nLinkAcceptMonthly)));
    oLinkAcceptFee.push_back(Pair("monthly_credits", nLinkAcceptMonthly/BDAP_CREDIT));
    oLinkAcceptFee.push_back(Pair("deposit_dynamic", FormatMoney(nLinkAcceptDeposit)));
    oLinkAcceptFee.push_back(Pair("deposit_credits", nLinkAcceptDeposit/BDAP_CREDIT));
    oLinkAcceptFee.push_back(Pair("one-time_dynamic", FormatMoney(nLinkAcceptOnetime)));
    oLinkAcceptFee.push_back(Pair("one-time_credits", nLinkAcceptOnetime/BDAP_CREDIT));
    oLinkAcceptFee.push_back(Pair("total_dynamic", FormatMoney((nLinkAcceptMonthly + nLinkAcceptDeposit + nLinkAcceptOnetime))));
    oLinkAcceptFee.push_back(Pair("total_credits", (nLinkAcceptMonthly + nLinkAcceptDeposit + nLinkAcceptOnetime)/BDAP_CREDIT));
    oFees.push_back(Pair("new_link_accept", oLinkAcceptFee));

    return oFees;
}

static const CRPCCommand commands[] =
{ //  category              name                     actor (function)               okSafe argNames
  //  --------------------- ------------------------ -----------------------        ------ --------------------
#ifdef ENABLE_WALLET
    /* BDAP */
    { "bdap",            "adduser",                  &adduser,                      true, {"account id", "common name", "registration days"} },
    { "bdap",            "getusers",                 &getusers,                     true, {"search string", "records per page", "page returned"} },
    { "bdap",            "getgroups",                &getgroups,                    true, {"search string", "records per page", "page returned"} },
    { "bdap",            "getuserinfo",              &getuserinfo,                  true, {"account id"} },
    { "bdap",            "updateuser",               &updateuser,                   true, {"account id", "common name", "registration days"} },
    { "bdap",            "updategroup",              &updategroup,                  true, {"account id", "common name", "registration days"} },
    { "bdap",            "deleteuser",               &deleteuser,                   true, {"account id"} },
    { "bdap",            "deletegroup",              &deletegroup,                  true, {"account id"} },
    { "bdap",            "addgroup",                 &addgroup,                     true, {"account id", "common name", "registration days"} },
    { "bdap",            "getgroupinfo",             &getgroupinfo,                 true, {"account id"} },
    { "bdap",            "mybdapaccounts",           &mybdapaccounts,               true, {} },
    { "bdap",            "makecredits",              &makecredits,                  true, {"dynamicaddress", "amount"} },
    { "bdap",            "getcredits",               &getcredits,                   true, {} },
    { "bdap",            "bdapfees",                 &bdapfees,                     true, {} },
#endif //ENABLE_WALLET
    { "bdap",            "makekeypair",              &makekeypair,                  true, {"prefix"} },
};

void RegisterDomainEntryRPCCommands(CRPCTable &t)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        t.appendCommand(commands[vcidx].name, &commands[vcidx]);
}