// Copyright (c) 2020 Duality Blockchain Solutions Developers 
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/audit.h"
//#include "bdap/auditdatadb.h"
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

template <typename Out>
void split(const std::string &s, char delim, Out result) {
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        *result++ = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

static UniValue AddAudit(const JSONRPCRequest& request)
{
#ifdef ENABLE_WALLET
    // audit add "0ac6a1e929c006cc63b9220bcb40e0af1e4776c4223e6f41e0da5d16f6ea2026,cc01a8d4e3d7271c4c2d6af1f5247a87cb324a5e59c5018e9251d81681877ce7" "username@public.bfap.io" 
    if (request.fHelp || (request.params.size() < 2 || request.params.size() > 3))
        throw std::runtime_error(
            "audit add \"0ac6a1e929c006cc63b9220bcb40e0af1e4776c4223e6f41e0da5d16f6ea2026\" \"testuser\"\n"
            "\nAdds an audit record to the blockchain.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"audit hashes\"      (string)            Audit hashes\n"
            "  \"account\"           (string, optional)  BDAP account that created audit\n"
            "  \"txid\"              (string)            Audit record transaction id\n"
            "}\n"
            "\nExamples\n" +
           HelpExampleCli("audit add", "\"0ac6a1e929c006cc63b9220bcb40e0af1e4776c4223e6f41e0da5d16f6ea2026\" \"testuser\"") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("audit add", "\"0ac6a1e929c006cc63b9220bcb40e0af1e4776c4223e6f41e0da5d16f6ea2026\" \"testuser\""));

    EnsureWalletIsUnlocked();

    std::string strAudits = request.params[1].get_str();
    CharString vchOwnerFQDN;
    CAuditData auditData;
    if (strAudits.find(",") > 0) {
        std::vector<std::string> vAudits = split(strAudits, ',');
        for(const std::string& strAuditHash : vAudits)
            auditData.vAuditData.push_back(vchFromString(strAuditHash));
    } else {
        auditData.vAuditData.push_back(vchFromValue(strAudits));
    }
    auditData.nTimeStamp = GetAdjustedTime();
    CPubKey pubWalletKey;
    CAudit txAudit(auditData);
    if (request.params.size() == 3) {
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

        pubWalletKey = walletKey.GetPubKey();
        //LogPrintf("%s: txAudit CheckSignature %s\n", __func__, txAudit.CheckSignature(pubWalletKey.Raw()) ? "Valid" : "Invalid");

    } else {
        CharString vchEd25519PubKey;
        if (!pwalletMain->GetKeysFromPool(pubWalletKey, vchEd25519PubKey, false))
            throw JSONRPCError(RPC_WALLET_KEYPOOL_RAN_OUT, "Error: Keypool ran out, please call keypoolrefill first");
    }

    // Create BDAP operation script
    CScript scriptPubKey;
    std::vector<unsigned char> vchAuditCount = vchFromString(std::to_string(auditData.vAuditData.size()));
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_NEW) << CScript::EncodeOP_N(OP_BDAP_AUDIT) 
                 << vchOwnerFQDN << vchAuditCount << OP_2DROP << OP_2DROP;

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
    //LogPrintf("%s -- monthlyFee %d, oneTimeFee %d, depositFee %d\n", __func__, monthlyFee, oneTimeFee, depositFee);
    // check BDAP values
    std::string strMessage;
    if (!txAudit.ValidateValues(strMessage))
        throw JSONRPCError(RPC_BDAP_AUDIT_INVALID, strprintf("Invalid audit transaction. %s", strMessage));

    CAmount curBalance = pwalletMain->GetBalance() + pwalletMain->GetBDAPDynamicAmount();
    if (monthlyFee + oneTimeFee + depositFee > curBalance)
        throw JSONRPCError(RPC_WALLET_INSUFFICIENT_FUNDS, strprintf("Insufficient funds for BDAP transaction. %s DYN required.", FormatMoney(monthlyFee + oneTimeFee + depositFee)));

    //LogPrintf("%s --txAudit %s\n", __func__, txAudit.ToString());
    bool fUseInstantSend = false;
    // Send the transaction
    CWalletTx wtx;
    SendBDAPTransaction(scriptData, scriptPubKey, wtx, monthlyFee, oneTimeFee + depositFee, fUseInstantSend);
    txAudit.txHash = wtx.GetHash();
    UniValue oAuditTransaction(UniValue::VOBJ);
    BuildAuditJson(txAudit, oAuditTransaction);
    return oAuditTransaction;
#else
    throw JSONRPCError(RPC_WALLET_ERROR, strprintf("Add audit transaction is not available when the wallet is disabled."));
#endif
}

static UniValue GetAudits(const JSONRPCRequest& request)
{
    UniValue oAuditList(UniValue::VOBJ);
    return oAuditList;
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
            "  get                - Get audit by hash or BDAP account\n"
            "\nExamples:\n"
            + HelpExampleCli("audit add", "\"0ac6a1e929c006cc63b9220bcb40e0af1e4776c4223e6f41e0da5d16f6ea2026\" \"testuser@public.bdap.io\"") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("audit add", "\"0ac6a1e929c006cc63b9220bcb40e0af1e4776c4223e6f41e0da5d16f6ea2026\" \"testuser@public.bdap.io\""));
    }
    if (strCommand == "add" || strCommand == "get") {
        if (!sporkManager.IsSporkActive(SPORK_32_BDAP_V2))
            throw JSONRPCError(RPC_BDAP_SPORK_INACTIVE, strprintf("Can not use audit functionality until the BDAP version 2 spork is active."));
    }
    if (strCommand == "add") {
        return AddAudit(request);
    }
    else if (strCommand == "get") {
        return GetAudits(request);
    } else {
        throw JSONRPCError(RPC_METHOD_NOT_FOUND, strprintf("%s is an unknown BDAP audit method command.", strCommand));
    }
    return NullUniValue;
}

static const CRPCCommand commands[] =
{ //  category              name                     actor (function)               okSafe argNames
  //  --------------------- ------------------------ -----------------------        ------ --------------------
    { "bdap",               "audit",                 &audit_rpc,                    true,  {"command", "param1", "param2", "param3"} },
};

void RegisterAuditRPCCommands(CRPCTable &t)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        t.appendCommand(commands[vcidx].name, &commands[vcidx]);
}