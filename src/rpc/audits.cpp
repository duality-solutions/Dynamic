// Copyright (c) 2020 Duality Blockchain Solutions Developers 
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/audit.h"
#include "bdap/auditdb.h"
#include "bdap/domainentry.h"
#include "bdap/domainentrydb.h"
#include "bdap/fees.h"
#include "bdap/utils.h"
#include "core_io.h" // needed for ScriptToAsmStr
#include "dynode-sync.h"
#include "dynodeman.h"
#include "rpc/protocol.h"
#include "rpc/server.h"
#include "primitives/transaction.h"
#include "spork.h"
#include "timedata.h"
#include "utilmoneystr.h"
#include "validation.h"
#include "wallet/wallet.h"

#include <univalue.h>

extern void SendBDAPTransaction(const CScript& bdapDataScript, const CScript& bdapOPScript, CWalletTx& wtxNew, const CAmount& nDataAmount, const CAmount& nOpAmount, const bool fUseInstantSend);

static UniValue AddAudit(const JSONRPCRequest& request)
{
#ifdef ENABLE_WALLET
    // audit add "0ac6a1e929c006cc63b9220bcb40e0af1e4776c4223e6f41e0da5d16f6ea2026,cc01a8d4e3d7271c4c2d6af1f5247a87cb324a5e59c5018e9251d81681877ce7" "username@public.bfap.io" 
    if (request.fHelp || (request.params.size() < 2 || request.params.size() > 5))
        throw std::runtime_error(
            "audit add \"hash_array\" ( \"account\" ) ( \"description\" ) ( \"algorithm\" )\n"
            "\nAdds an audit record to the blockchain.\n"
            "\nArguments:\n"
            "1. \"hash_array\"       (string, required)  Audit hashes\n"
            "2. \"account\"          (string, optional)  BDAP account that created audit\n"
            "3. \"description\"      (string, optional)  Description of audit\n"
            "4. \"algorithm\"        (string, optional)  Algorithm used to create hash\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"audit hashes\"      (string)            Audit hashes\n"
            "  \"account\"           (string, optional)  BDAP account that created audit\n"
            "  \"description\"       (string, optional)  Description of audit\n"
            "  \"algorithm\"         (string, optional)  Algorithm used to create hash\n"
            "  \"txid\"              (string)            Audit record transaction id\n"
            "}\n"
            "\nExamples\n" +
           HelpExampleCli("audit add", "\"0ac6a1e929c006cc63b9220bcb40e0af1e4776c4223e6f41e0da5d16f6ea2026\" \"testuser\" \"testdescription\" \"sha256\"") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("audit add", "\"0ac6a1e929c006cc63b9220bcb40e0af1e4776c4223e6f41e0da5d16f6ea2026\" \"testuser\"  \"testdescription\" \"sha256\""));

    EnsureWalletIsUnlocked();

    //handle HASH array [REQUIRED]
    std::string strAudits = request.params[1].get_str();
    CharString vchOwnerFQDN;
    CAuditData auditData;
    if (strAudits.find(",") > 0) {
        std::vector<std::string> vAudits = SplitString(strAudits, ',');
        for(const std::string& strAuditHash : vAudits)
            auditData.vAuditData.push_back(vchFromString(TrimString(strAuditHash)));
    } else {
        auditData.vAuditData.push_back(vchFromValue(strAudits));
    }
    auditData.nTimeStamp = GetAdjustedTime();
    CPubKey pubWalletKey;
    CAudit txAudit(auditData);
    bool signedAudit = false;

    //handle description [OPTIONAL]
    if (request.params.size() > 3) {
            std::string auditDescription = request.params[3].get_str();
            if (auditDescription.size() > 0) {
                txAudit.vchDescription = vchFromString(auditDescription);
            }
    }

    //handle algorithm [OPTIONAL]
    if (request.params.size() > 4) {
            std::string auditAlgorithm = request.params[4].get_str();
            if (auditAlgorithm.size() > 0) {
                txAudit.vchAlgorithmType = vchFromString(auditAlgorithm);
            }
    }

    //handle OWNER [OPTIONAL]
    if (request.params.size() > 2) {

        if (request.params[2].get_str().size() > 0) {
            // TODO: add ability to use specified wallet address
            std::string strOwnerFQDN = request.params[2].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
            ToLowerCase(strOwnerFQDN);
            vchOwnerFQDN = vchFromString(strOwnerFQDN);
            // Check if name exists
            CDomainEntry domainEntry;
            if (!GetDomainEntry(vchOwnerFQDN, domainEntry))
                throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", strOwnerFQDN));

            txAudit.vchOwnerFullObjectPath = domainEntry.vchFullObjectPath();

            CDynamicAddress address = domainEntry.GetWalletAddress();
            CKeyID keyID;
            if (!address.GetKeyID(keyID))
                throw JSONRPCError(RPC_TYPE_ERROR, "BDAP account wallet address does not refer to a key");

            CKey walletKey;
            if (!pwalletMain->GetKey(keyID, walletKey))
                throw JSONRPCError(RPC_WALLET_PRIV_KEY_NOT_FOUND, "Private key for address " + address.ToString() + " is not known");

            if (!txAudit.Sign(walletKey))
                throw JSONRPCError(RPC_WALLET_PRIV_KEY_NOT_FOUND, "Signing audit data failed.");

            signedAudit = true;
            pubWalletKey = walletKey.GetPubKey();

            if (!txAudit.CheckSignature(pubWalletKey.Raw()))
                throw JSONRPCError(RPC_BDAP_INVALID_SIGNATURE, "Check Signature failed.");

        } else {
            CharString vchEd25519PubKey;
            if (!pwalletMain->GetKeysFromPool(pubWalletKey, vchEd25519PubKey, false))
                throw JSONRPCError(RPC_WALLET_KEYPOOL_RAN_OUT, "Error: Keypool ran out, please call keypoolrefill first");
        }

    }

    // check BDAP values
    std::string strMessage;
    if (!txAudit.ValidateValues(strMessage))
        throw JSONRPCError(RPC_BDAP_AUDIT_INVALID, strprintf("Invalid audit transaction. %s", strMessage));

    // Create BDAP operation script
    CScript scriptPubKey;
    std::vector<unsigned char> vchAuditCount = vchFromString(std::to_string(auditData.vAuditData.size()));
    if (signedAudit) {
        scriptPubKey << CScript::EncodeOP_N(OP_BDAP_NEW) << CScript::EncodeOP_N(OP_BDAP_AUDIT) 
                 << vchAuditCount << vchOwnerFQDN << pubWalletKey.Raw() << OP_2DROP << OP_2DROP << OP_DROP; 
    } else {
        scriptPubKey << CScript::EncodeOP_N(OP_BDAP_NEW) << CScript::EncodeOP_N(OP_BDAP_AUDIT) 
                 << vchAuditCount << OP_2DROP << OP_DROP; 
    }

    CKeyID keyWalletID = pubWalletKey.GetID();
    CDynamicAddress walletAddress = CDynamicAddress(keyWalletID);

    CScript scriptDestination;
    scriptDestination = GetScriptForDestination(walletAddress.Get());
    scriptPubKey += scriptDestination;

    // Create BDAP OP_RETURN script
    CharString data;
    txAudit.Serialize(data);
    CScript scriptData;
    scriptData << OP_RETURN << data;

    // Get BDAP fees
    BDAP::ObjectType bdapType = BDAP::ObjectType::BDAP_AUDIT;
    CAmount monthlyFee, oneTimeFee, depositFee;
    if (!GetBDAPFees(OP_BDAP_NEW, OP_BDAP_AUDIT, bdapType, auditData.vAuditData.size(), monthlyFee, oneTimeFee, depositFee))
        throw JSONRPCError(RPC_BDAP_FEE_UNKNOWN, strprintf("Error calculating BDAP fees."));

    CAmount curBalance = pwalletMain->GetBalance() + pwalletMain->GetBDAPDynamicAmount();
    if (monthlyFee + oneTimeFee + depositFee > curBalance)
        throw JSONRPCError(RPC_WALLET_INSUFFICIENT_FUNDS, strprintf("Insufficient funds for BDAP transaction. %s DYN required.", FormatMoney(monthlyFee + oneTimeFee + depositFee)));

    bool fUseInstantSend = false;
    // Send the transaction
    CWalletTx wtx;
    SendBDAPTransaction(scriptData, scriptPubKey, wtx, oneTimeFee, monthlyFee + depositFee, fUseInstantSend);
    txAudit.txHash = wtx.GetHash();
    UniValue oAuditTransaction(UniValue::VOBJ);
    BuildAuditJson(txAudit, oAuditTransaction);
    return oAuditTransaction;
#else
    throw JSONRPCError(RPC_WALLET_ERROR, strprintf("Add audit transaction is not available when the wallet is disabled."));
#endif
}

static UniValue VerifyAudit(const JSONRPCRequest& request)
{
    if (request.fHelp || (request.params.size() < 2 || request.params.size() > 5))
        throw std::runtime_error(
            "audit verify  \"audit_hash\" \n"
            "\nVerify audits from blockchain\n"
            "\nArguments:\n"
            "1. \"audit_hash         (string, required)  audit hash\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"verified\"          (boolean)           Audit verified\n"
            "  \"version\"           (string)            Audit version\n"
            "  \"audit_count\"       (int)               Number of audit hashes stored\n"
            "  \"timestamp\"         (int)               Time when audit was created\n"
            "  \"owner\"             (string, optional)  BDAP account that created audit\n"
            "  \"signed\"            (boolean)           Audit signed\n"
            "  \"algorithm_type\"    (string, optional)  Algorithm used to create hash\n"
            "  \"description\"       (string, optional)  Description of audit\n"
            "  \"txid\"              (string)            Audit record transaction id\n"
            "  \"block_time\"        (int)               Block time where audit is stored\n"
            "  \"block_height\"      (int)               Block where audit is stored\n"
            "}\n"
            "\nExamples\n" +
           HelpExampleCli("audit verify", "\"0ac6a1e929c006cc63b9220bcb40e0af1e4776c4223e6f41e0da5d16f6ea2026\" ") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("audit verify", "\"0ac6a1e929c006cc63b9220bcb40e0af1e4776c4223e6f41e0da5d16f6ea2026\" "));

    std::vector<unsigned char> vchAudit;
    std::vector<CAudit> vAudits;

    std::string parameter1 = request.params[1].get_str();
    vchAudit = vchFromString(parameter1);

    bool readAuditState = false;

    UniValue oAuditLists(UniValue::VARR);
    UniValue oAuditList(UniValue::VOBJ);

    readAuditState = pAuditDB->ReadAuditHash(vchAudit, vAudits);

    if (readAuditState) {
        for (const CAudit& audit : vAudits) {
            UniValue oAuditList(UniValue::VOBJ);
            oAuditList.push_back((Pair("verified", "True")));
            BuildVerifyAuditJson(audit, oAuditList);
            oAuditLists.push_back(oAuditList);
        }
    }
    else {
        CAudit emptyAudit;
        oAuditList.push_back((Pair("verified", "False")));
        BuildVerifyAuditJson(emptyAudit, oAuditList);
        oAuditLists.push_back(oAuditList);       
    }

    return oAuditLists;
}

static UniValue GetAudits(const JSONRPCRequest& request)
{
    if (request.fHelp || (request.params.size() < 2 || request.params.size() > 4))
        throw std::runtime_error(
            "audit get  \"account | txid | audit_hash\" (start_time) (stop_time) \n"
            "\nGets list of audits from blockchain\n"
            "\nArguments:\n"
            "1. \"account | txid |   (string, required)  BDAP account that created audit or Transaction ID or audit hash\n"
            "    audit_hash\"    \n"
            "2. \"start_time\"       (int64, optional)   Epoch start time\n"
            "3. \"stop_time\"        (int64, optional)   Epoch stop time\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"version\"           (string)            Audit version\n"
            "  \"audit_count\"       (int)               Number of audit hashes stored\n"
            "  \"audit_hashes\"      (string)            List of Audit hashes\n"
            "  \"timestamp\"         (int)               Time when audit was created\n"
            "  \"owner\"             (string, optional)  BDAP account that created audit\n"
            "  \"signed\"            (boolean)           Audit signed\n"
            "  \"algorithm_type\"    (string, optional)  Algorithm used to create hash\n"
            "  \"description\"       (string, optional)  Description of audit\n"
            "  \"txid\"              (string)            Audit record transaction id\n"
            "  \"block_time\"        (int)               Block time where audit is stored\n"
            "  \"block_height\"      (int)               Block where audit is stored\n"
            "}\n"
            "\nExamples\n" +
           HelpExampleCli("audit get", "\"testuser\" ") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("audit get", "\"testuser\" "));

    std::vector<unsigned char> vchOwnerFQDN;
    std::vector<unsigned char> vchTxId;
    std::vector<unsigned char> vchAudit;

    bool searchByOwner = false;
    bool readAuditState = false;
    bool foundTxId = false;
    std::string parameter1 = request.params[1].get_str();
    std::string strOwnerFQDN = parameter1 + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strOwnerFQDN);
    vchOwnerFQDN = vchFromString(strOwnerFQDN);
    vchTxId = vchFromString(parameter1);
    vchAudit = vchFromString(parameter1);

    int64_t epochStart;
    int64_t epochStop;
    int64_t retrievedTimeStamp;
 
    bool startDetected = false;
    bool stopDetected = false;
    bool startOK = true;
    bool stopOK = true;
 
    //handle starttime [OPTIONAL]
    if (request.params.size() > 2) {
        if (request.params[2].get_str().size() > 0) { //only process if there's a value
            if (!ParseInt64(request.params[2].get_str(), &epochStart))
                throw JSONRPCError(RPC_TYPE_ERROR, "Cannot determine epoch time");

            startDetected = true;
        }
    }

    //handle stoptime [OPTIONAL]
    if (request.params.size() > 3) {
        if (!ParseInt64(request.params[3].get_str(), &epochStop))
            throw JSONRPCError(RPC_TYPE_ERROR, "Cannot determine epoch time");

        stopDetected = true;
    }

    // Check if name exists
    CDomainEntry domainEntry;
    if (GetDomainEntry(vchOwnerFQDN, domainEntry))
        searchByOwner = true;

    std::vector<CAudit> vAudits;
    CAudit singleAudit;
    UniValue oAuditLists(UniValue::VARR);
    UniValue oAuditList(UniValue::VOBJ);

    //Search by owner
    if (searchByOwner) {
        readAuditState = pAuditDB->ReadAuditDN(vchOwnerFQDN, vAudits);
    }
    else { //Search by TxId
        readAuditState = pAuditDB->ReadAuditTxId(vchTxId,singleAudit);
        if (readAuditState) {
            foundTxId = true;
            BuildAuditJson(singleAudit, oAuditList);
        }
        if (readAuditState)
            return oAuditList;
    }

    //If can't find owner or TxId, try Hash
    if ((!searchByOwner) && (!foundTxId)) {
        readAuditState = pAuditDB->ReadAuditHash(vchAudit, vAudits);
    }

    if (vAudits.size() > 0) {
        for (const CAudit& audit : vAudits) {
            UniValue oAuditList(UniValue::VOBJ);

            if (startDetected || stopDetected)
                retrievedTimeStamp = audit.GetTimeStamp();

            if (startDetected) {
                if (retrievedTimeStamp >= epochStart) {
                    startOK = true;
                }
                else {
                    startOK = false;
                }
            }

            if (stopDetected) {
                if (retrievedTimeStamp <= epochStop) {
                    stopOK = true;
                }
                else {
                    stopOK = false;
                }
            }

            if (startOK && stopOK) {
                BuildAuditJson(audit, oAuditList);
                oAuditLists.push_back(oAuditList);
            }

        };
        return oAuditLists;
    }

    //if all fails, return nothing
   return oAuditLists;
}

UniValue audit_rpc(const JSONRPCRequest& request) 
{
    std::string strCommand;
    if (request.params.size() >= 1) {
        strCommand = request.params[0].get_str();
        ToLowerCase(strCommand);
    }
    else {
        throw std::runtime_error(
            "audit \"command\"...\n"
            "\nAvailable commands:\n"
            "  add                - Add new audit or list of audits\n"
            "  get                - Get audit by hash or BDAP account or TxID\n"
            "  verify             - Verify audit by hash\n"
            "\nExamples:\n"
            + HelpExampleCli("audit add", "\"0ac6a1e929c006cc63b9220bcb40e0af1e4776c4223e6f41e0da5d16f6ea2026\" \"testuser\" \"testdescription\" \"sha256\"") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("audit add", "\"0ac6a1e929c006cc63b9220bcb40e0af1e4776c4223e6f41e0da5d16f6ea2026\" \"testuser\" \"testdescription\" \"sha256\""));
    }
    if (strCommand == "add" || strCommand == "get" || strCommand == "verify") {
        if (!sporkManager.IsSporkActive(SPORK_32_BDAP_V2))
            throw JSONRPCError(RPC_BDAP_SPORK_INACTIVE, strprintf("Can not use audit functionality until the BDAP version 2 spork is active."));
    }
    if (strCommand == "add") {
        return AddAudit(request);
    }
    else if (strCommand == "get") {
        return GetAudits(request);
    }
    else if (strCommand == "verify") {
        return VerifyAudit(request);
    } else {
        throw JSONRPCError(RPC_METHOD_NOT_FOUND, strprintf("%s is an unknown BDAP audit method command.", strCommand));
    }
    return NullUniValue;
}

static const CRPCCommand commands[] =
{ //  category              name                     actor (function)               okSafe argNames
  //  --------------------- ------------------------ -----------------------        ------ --------------------
    { "bdap",               "audit",                 &audit_rpc,                    true,  {"command", "param1", "param2", "param3", "param4"} },
};

void RegisterAuditRPCCommands(CRPCTable &t)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        t.appendCommand(commands[vcidx].name, &commands[vcidx]);
}