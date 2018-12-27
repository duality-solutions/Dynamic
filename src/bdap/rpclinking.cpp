// Copyright (c) 2018 Duality Blockchain Solutions Developers 
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/bdap.h"
#include "bdap/domainentry.h"
#include "bdap/domainentrydb.h"
#include "bdap/linking.h"
#include "bdap/linkingdb.h"
#include "bdap/utils.h"
#include "dht/ed25519.h"
#include "core_io.h" // needed for ScriptToAsmStr
#include "rpcprotocol.h"
#include "rpcserver.h"
#include "primitives/transaction.h"
#include "wallet/wallet.h"
#include "validation.h"

#include <univalue.h>

#ifdef ENABLE_WALLET

extern void SendLinkingTransaction(const CScript& bdapDataScript, const CScript& bdapOPScript, const CScript& sendAddress, 
                                    CWalletTx& wtxNew, const CAmount& nOPValue, const CAmount& nDataValue);

static bool BuildJsonLinkInfo(const CLinkRequest& link, const CDomainEntry& requestor, const CDomainEntry& recipient, UniValue& oLink)
{
    bool expired = false;
    int64_t expired_time = 0;
    int64_t nTime = 0;
    oLink.push_back(Pair("requestor_fqdn", requestor.GetFullObjectPath()));
    oLink.push_back(Pair("recipient_fqdn", recipient.GetFullObjectPath()));
    oLink.push_back(Pair("requestor_link_pubkey", link.RequestorPubKeyString()));
    oLink.push_back(Pair("requestor_link_address", stringFromVch(requestor.LinkAddress)));
    oLink.push_back(Pair("recipient_link_address", stringFromVch(recipient.LinkAddress)));
    oLink.push_back(Pair("txid", link.txHash.GetHex()));
    if ((unsigned int)chainActive.Height() >= link.nHeight-1) {
        CBlockIndex *pindex = chainActive[link.nHeight-1];
        if (pindex) {
            nTime = pindex->GetMedianTimePast();
        }
    }
    oLink.push_back(Pair("time", nTime));
    expired_time = link.nExpireTime;
    if(expired_time <= (unsigned int)chainActive.Tip()->GetMedianTimePast())
    {
        expired = true;
    }
    oLink.push_back(Pair("expires_on", expired_time));
    oLink.push_back(Pair("expired", expired));
    
    return true;
}

static UniValue SendLinkRequest(const JSONRPCRequest& request)
{
     if (request.fHelp || request.params.size() < 3 || request.params.size() > 4)
        throw std::runtime_error(
            "link send userid-from userid-to\n"
            "Creates a link request transaction on the blockchain."
            + HelpRequiringPassphrase() +
            "\nLink Send Arguments:\n"
            "1. requestor          (string)             BDAP account requesting the link\n"
            "2. recipient          (string)             Link recipient's BDAP account.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"Requestor FQDN\"             (string)  Requestor's BDAP full BDAP path\n"
            "  \"Recipient FQDN\"             (string)  Recipient's BDAP full BDAP path\n"
            "  \"Recipient Link Address\"     (string)  Recipient's link address\n"
            "  \"Requestor Link Pubkey\"      (string)  Requestor's link pubkey used for DHT storage\n"
            "  \"Requestor Link Address\"     (string)  Requestor's link address\n"
            "  \"Link Request TxID\"          (string)  Transaction ID for the link request\n"
            "  \"Time\"                       (int)     Transaction time\n"
            "  \"Expires On\"                 (int)     Link request expiration\n"
            "  \"Expired\"                    (boolean) Is link request expired\n"
            "  }\n"
            "\nExamples:\n"
            + HelpExampleCli("link send", "superman batman") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link send", "superman batman"));

    std::string strRequestorFQDN = request.params[1].get_str();
    ToLowerCase(strRequestorFQDN);
    CharString vchRequestorFQDN = vchFromString(strRequestorFQDN + "@" + stringFromVch(vchDefaultPublicOU) + "." + stringFromVch(vchDefaultDomainName));

    std::string strRecipientFQDN = request.params[2].get_str();
    ToLowerCase(strRecipientFQDN);
    CharString vchRecipientFQDN = vchFromString(strRequestorFQDN + "@" + stringFromVch(vchDefaultPublicOU) + "." + stringFromVch(vchDefaultDomainName));
    
    CLinkRequest txLink;
    txLink.RequestorFullObjectPath = vchRequestorFQDN;
    txLink.RecipientFullObjectPath = vchRecipientFQDN;
    CKeyEd25519 privReqDHTKey;
    CharString vchDHTPubKey = privReqDHTKey.GetPubKey();
    if (pwalletMain && !pwalletMain->AddDHTKey(privReqDHTKey, vchDHTPubKey))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4000 - " + _("Error adding ed25519 key to wallet for BDAP link"));

    txLink.RequestorPubKey = vchDHTPubKey;

    pwalletMain->SetAddressBook(privReqDHTKey.GetID(), strRequestorFQDN, "bdap-dht-key");

    // Check if name already exists
    if (GetLinkRequest(txLink.RequestorPubKey, txLink))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4001 - " + txLink.RequestorPubKeyString() + _(" entry already exists.  Can not add duplicate."));

    // Get requestor link address
    CDomainEntry entryRequestor;
    if (!GetDomainEntry(vchRequestorFQDN, entryRequestor))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4002 - Requestor " + strRequestorFQDN + _(" not found."));

    // Get recipient link address
    CDomainEntry entryRecipient;
    if (!GetDomainEntry(vchRecipientFQDN, entryRecipient))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4003 - Recipient " + strRecipientFQDN + _(" not found."));

    uint64_t nDays = 1461;  //default to 4 years.
// TODO (bdap): fix invalid int error when passing registration days.
//    if (request.params.size() >= 3) {
//        nDays = request.params[3].get_int();
//    }

    uint64_t nSeconds = nDays * SECONDS_PER_DAY;
    txLink.nExpireTime = chainActive.Tip()->GetMedianTimePast() + nSeconds;

    // Create BDAP operation script
    CScript scriptPubKey;
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_NEW) << CScript::EncodeOP_N(OP_BDAP_LINK_ACCEPT) << vchDHTPubKey << OP_2DROP << OP_DROP;
    CScript scriptDest = GetScriptForDestination(CPubKey(entryRecipient.LinkAddress).GetID());
    scriptPubKey += scriptDest;
    CScript scriptSend = GetScriptForDestination(CPubKey(entryRequestor.LinkAddress).GetID());

    // TODO (bdap): encrypt data before adding it to OP_RETURN.
    // Create BDAP OP_RETURN script
    CharString data;
    txLink.Serialize(data);
    CScript scriptData;
    scriptData << OP_RETURN << data;

    // Send the transaction
    CWalletTx wtx;
    float fYears = ((float)nDays/365.25);
    CAmount nOperationFee = GetBDAPFee(scriptPubKey) * powf(3.1, fYears);
    CAmount nDataFee = GetBDAPFee(scriptData) * powf(3.1, fYears);

    // check BDAP values
    std::string strMessage;
    if (!txLink.ValidateValues(strMessage))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4004 - Error validating link request values: " + strMessage);

    SendLinkingTransaction(scriptData, scriptPubKey, scriptSend, wtx, nDataFee, nOperationFee);
    txLink.txHash = wtx.GetHash();

    UniValue oLink(UniValue::VOBJ);
    if(!BuildJsonLinkInfo(txLink, entryRequestor, entryRecipient, oLink))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4005 - " + _("Failed to build BDAP link JSON object"));

    return oLink;
}

UniValue link(const JSONRPCRequest& request) 
{
    std::string strCommand;
    if (request.params.size() >= 1) {
        strCommand = request.params[0].get_str();
        ToLowerCase(strCommand);
    }
    if (strCommand == "send") {
        return SendLinkRequest(request);
    }
    // TODO (bdap): implement link RPC commands below
    if (strCommand == "accept") {
        //return AcceptLinkRequest(request);
    }
    if (strCommand == "pending") {
        //return ListPendingLinks(request);
    }
    if (strCommand == "list") {
        //return ListAcceptedLinks(request);
    }
    if (strCommand == "remove") {
        //return RemoveLinkAccept(request);
    }
    if (strCommand == "cancel") {
        //return CancelLinkRequest(request);
    }
    else {
        throw std::runtime_error("BDAP_LINK_RPC_ERROR: ERRCODE: 4010 - " + strCommand + _(" is an unknown link command."));
    }
    return NullUniValue;
}
#endif // ENABLE_WALLET
static const CRPCCommand commands[] =
{ //  category              name                     actor (function)               okSafe argNames
  //  --------------------- ------------------------ -----------------------        ------ --------------------
#ifdef ENABLE_WALLET
    /* BDAP */
    { "bdap",            "link",                     &link,                         true, {"opration","common name", "registration days"} },

};
#endif // ENABLE_WALLET
void RegisterLinkingRPCCommands(CRPCTable &t)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        t.appendCommand(commands[vcidx].name, &commands[vcidx]);
}