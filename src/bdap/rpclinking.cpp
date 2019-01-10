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
#include "dynodeman.h"
#include "hash.h"
#include "rpcprotocol.h"
#include "rpcserver.h"
#include "primitives/transaction.h"
#include "wallet/wallet.h"
#include "uint256.h"
#include "validation.h"

#include <univalue.h>

#ifdef ENABLE_WALLET

extern void SendLinkingTransaction(const CScript& bdapDataScript, const CScript& bdapOPScript, const CScript& sendAddress, 
                                    CWalletTx& wtxNew, const CAmount& nOPValue, const CAmount& nDataValue, const bool fUseInstantSend);

static bool BuildJsonLinkRequestInfo(const CLinkRequest& link, const CDomainEntry& requestor, const CDomainEntry& recipient, UniValue& oLink)
{
    bool expired = false;
    int64_t expired_time = 0;
    int64_t nTime = 0;
    oLink.push_back(Pair("requestor_fqdn", requestor.GetFullObjectPath()));
    oLink.push_back(Pair("recipient_fqdn", recipient.GetFullObjectPath()));
    oLink.push_back(Pair("requestor_link_pubkey", link.RequestorPubKeyString()));
    oLink.push_back(Pair("requestor_link_address", stringFromVch(requestor.LinkAddress)));
    oLink.push_back(Pair("recipient_link_address", stringFromVch(recipient.LinkAddress)));
    oLink.push_back(Pair("link_message", stringFromVch(link.LinkMessage)));
    oLink.push_back(Pair("signature_proof", stringFromVch(link.SignatureProof)));
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

static bool BuildJsonLinkAcceptInfo(const CLinkAccept& link, const CDomainEntry& requestor, const CDomainEntry& recipient, UniValue& oLink)
{
    bool expired = false;
    int64_t expired_time = 0;
    int64_t nTime = 0;
    oLink.push_back(Pair("requestor_fqdn", requestor.GetFullObjectPath()));
    oLink.push_back(Pair("recipient_fqdn", recipient.GetFullObjectPath()));
    oLink.push_back(Pair("recipient_link_pubkey", link.RecipientPubKeyString()));
    oLink.push_back(Pair("requestor_link_address", stringFromVch(requestor.LinkAddress)));
    oLink.push_back(Pair("recipient_link_address", stringFromVch(recipient.LinkAddress)));
    oLink.push_back(Pair("signature_proof", stringFromVch(link.SignatureProof)));
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
     if (request.fHelp || request.params.size() != 4)
        throw std::runtime_error(
            "link request userid-from userid-to message\n"
            "Creates a link request transaction on the blockchain."
            + HelpRequiringPassphrase() +
            "\nLink Send Arguments:\n"
            "1. requestor          (string)             BDAP account requesting the link\n"
            "2. recipient          (string)             Link recipient's BDAP account.\n"
            "3. invite message     (string)             Message from requestor to recipient.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"Requestor FQDN\"             (string)  Requestor's BDAP full path\n"
            "  \"Recipient FQDN\"             (string)  Recipient's BDAP full path\n"
            "  \"Recipient Link Address\"     (string)  Recipient's link address\n"
            "  \"Requestor Link Pubkey\"      (string)  Requestor's link pubkey used for DHT storage\n"
            "  \"Requestor Link Address\"     (string)  Requestor's link address\n"
            "  \"Link Message\"               (string)  Message from requestor to recipient.\n"
            "  \"Signature Proof\"            (string)  Encoded signature to prove it is from the requestor\n"
            "  \"Link Request TxID\"          (string)  Transaction ID for the link request\n"
            "  \"Time\"                       (int)     Transaction time\n"
            "  \"Expires On\"                 (int)     Link request expiration\n"
            "  \"Expired\"                    (boolean) Is link request expired\n"
            "  }\n"
            "\nExamples:\n"
            + HelpExampleCli("link request", "superman batman") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link request", "superman batman"));

    std::string strRequestorFQDN = request.params[1].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strRequestorFQDN);
    CharString vchRequestorFQDN = vchFromString(strRequestorFQDN);

    std::string strRecipientFQDN = request.params[2].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strRecipientFQDN);
    CharString vchRecipientFQDN = vchFromString(strRecipientFQDN);
    
    std::string strLinkMessage = request.params[3].get_str();

    CLinkRequest txLink;
    txLink.RequestorFullObjectPath = vchRequestorFQDN;
    txLink.RecipientFullObjectPath = vchRecipientFQDN;
    txLink.LinkMessage = vchFromString(strLinkMessage);
    CKeyEd25519 privReqDHTKey;
    CharString vchDHTPubKey = privReqDHTKey.GetPubKey();
    if (pwalletMain && !pwalletMain->AddDHTKey(privReqDHTKey, vchDHTPubKey))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4000 - " + _("Error adding ed25519 key to wallet for BDAP link"));
 
    txLink.RequestorPubKey = vchDHTPubKey;

    pwalletMain->SetAddressBook(privReqDHTKey.GetID(), strRequestorFQDN, "bdap-dht-key");

    // Check if pubkey already exists
    uint256 prevTxID;
    //GetLinkRequestIndex(const std::vector<unsigned char>& vchPubKey, uint256& txid)
    if (GetLinkRequestIndex(txLink.RequestorPubKey, prevTxID))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4001 - " + txLink.RequestorPubKeyString() + _(" entry already exists.  Can not add duplicate."));

    // Get requestor link address
    CDomainEntry entryRequestor;
    if (!GetDomainEntry(vchRequestorFQDN, entryRequestor))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4002 - Requestor " + strRequestorFQDN + _(" not found."));

    // Get recipient link address
    CDomainEntry entryRecipient;
    if (!GetDomainEntry(vchRecipientFQDN, entryRecipient))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4003 - Recipient " + strRecipientFQDN + _(" not found."));

    CDynamicAddress addressRequestor = entryRequestor.GetWalletAddress();
    CKeyID keyID;
    if (!addressRequestor.GetKeyID(keyID))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4004 - Could not get " + strRequestorFQDN + _("'s wallet address key ") + addressRequestor.ToString());

    CKey key;
    if (pwalletMain && !pwalletMain->GetKey(keyID, key))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4005 - Could not get " + strRequestorFQDN + _("'s private key ") + addressRequestor.ToString());
    
    CHashWriter ss(SER_GETHASH, 0);
    ss << strMessageMagic;
    ss << strRecipientFQDN;
    std::vector<unsigned char> vchSig;
    if (!key.SignCompact(ss.GetHash(), vchSig))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4006 - Error signing " + strRequestorFQDN + _("'s signature proof."));

    std::vector<unsigned char> vchSignatureProof = vchFromString(EncodeBase64(&vchSig[0], vchSig.size()));
    txLink.SignatureProof = vchSignatureProof;

    uint64_t nDays = 1461;  //default to 4 years.
    // TODO (bdap): fix invalid int error when passing registration days.
    //    if (request.params.size() >= 4) {
    //        nDays = request.params[4].get_int();
    //    }
    uint64_t nSeconds = nDays * SECONDS_PER_DAY;
    txLink.nExpireTime = chainActive.Tip()->GetMedianTimePast() + nSeconds;
    CKeyEd25519 dhtKey;
    std::vector<unsigned char> vchSharedPubKey = GetLinkSharedPubKey(privReqDHTKey, entryRecipient.DHTPublicKey);
    txLink.SharedPubKey = vchSharedPubKey;

    // Create BDAP operation script
    CScript scriptPubKey;
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_NEW) << CScript::EncodeOP_N(OP_BDAP_LINK_REQUEST) 
                 << vchDHTPubKey << vchSharedPubKey << txLink.nExpireTime << OP_2DROP << OP_2DROP << OP_DROP;
    CScript scriptDest = GetScriptForDestination(CPubKey(entryRecipient.LinkAddress).GetID());
    scriptPubKey += scriptDest;
    CScript scriptSend = GetScriptForDestination(CPubKey(entryRequestor.LinkAddress).GetID());

    // check BDAP values
    std::string strMessage;
    if (!txLink.ValidateValues(strMessage))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4007 - Error validating link request values: " + strMessage);

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

    bool fUseInstantSend = false;
    int enabled = dnodeman.CountEnabled();
    if (enabled > 5) {
        // TODO (bdap): calculate cost for instant send.
        nOperationFee = nOperationFee * 2;
        fUseInstantSend = true;
    }
    SendLinkingTransaction(scriptData, scriptPubKey, scriptSend, wtx, nDataFee, nOperationFee, fUseInstantSend);
    txLink.txHash = wtx.GetHash();

    UniValue oLink(UniValue::VOBJ);
    if(!BuildJsonLinkRequestInfo(txLink, entryRequestor, entryRecipient, oLink))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4005 - " + _("Failed to build BDAP link JSON object"));

    return oLink;
}

static UniValue SendLinkAccept(const JSONRPCRequest& request)
{
     if (request.fHelp || request.params.size() != 3)
        throw std::runtime_error(
            "link accept userid-from userid-to\n"
            "Creates a link accept transaction on the blockchain."
            + HelpRequiringPassphrase() +
            "\nLink Send Arguments:\n"
            "1. accept account          (string)             BDAP account accepting the link\n"
            "2. requestor account       (string)             Link requestor's BDAP account.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"Requestor FQDN\"             (string)  Requestor's BDAP full path\n"
            "  \"Recipient FQDN\"             (string)  Recipient's BDAP full path\n"
            "  \"Recipient Link Address\"     (string)  Recipient's link address\n"
            "  \"Requestor Link Pubkey\"      (string)  Requestor's link pubkey used for DHT storage\n"
            "  \"Requestor Link Address\"     (string)  Requestor's link address\n"
            "  \"Link Message\"               (string)  Message from requestor to recipient.\n"
            "  \"Signature Proof\"            (string)  Encoded signature to prove it is from the requestor\n"
            "  \"Link Request TxID\"          (string)  Transaction ID for the link request\n"
            "  \"Time\"                       (int)     Transaction time\n"
            "  \"Expires On\"                 (int)     Link request expiration\n"
            "  \"Expired\"                    (boolean) Is link request expired\n"
            "  }\n"
            "\nExamples:\n"
            + HelpExampleCli("link send", "superman batman") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link send", "superman batman"));

    std::string strAcceptorFQDN = request.params[1].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strAcceptorFQDN);
    CharString vchAcceptorFQDN = vchFromString(strAcceptorFQDN);

    std::string strRequestorFQDN = request.params[2].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strRequestorFQDN);
    CharString vchRequestorFQDN = vchFromString(strRequestorFQDN);
    
    CLinkAccept txLinkAccept;
    txLinkAccept.RequestorFullObjectPath = vchRequestorFQDN;
    txLinkAccept.RecipientFullObjectPath = vchAcceptorFQDN;

    CKeyEd25519 privAcceptDHTKey;
    CharString vchDHTPubKey = privAcceptDHTKey.GetPubKey();
    if (pwalletMain && !pwalletMain->AddDHTKey(privAcceptDHTKey, vchDHTPubKey))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4100 - " + _("Error adding ed25519 key to wallet for BDAP link"));
 
    txLinkAccept.RecipientPubKey = vchDHTPubKey;

    pwalletMain->SetAddressBook(privAcceptDHTKey.GetID(), strAcceptorFQDN, "bdap-dht-key");

    // Check if pubkey already exists
    uint256 prevTxID;
    if (GetLinkAcceptIndex(txLinkAccept.RecipientPubKey, prevTxID))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4101 - " + txLinkAccept.RecipientPubKeyString() + _(" entry already exists.  Can not add duplicate."));

    // Get link accepting address
    CDomainEntry entryAcceptor;
    if (!GetDomainEntry(vchAcceptorFQDN, entryAcceptor))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4102 - Acceptor " + strAcceptorFQDN + _(" not found."));

    // Get requestor link address
    CDomainEntry entryRequestor;
    if (!GetDomainEntry(vchRequestorFQDN, entryRequestor))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4103 - Requestor " + strRequestorFQDN + _(" not found."));

    CDynamicAddress addressAcceptor = entryAcceptor.GetWalletAddress();
    CKeyID keyID;
    if (!addressAcceptor.GetKeyID(keyID))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4104 - Could not get " + strAcceptorFQDN + _("'s wallet address key ") + addressAcceptor.ToString());

    CKey key;
    if (pwalletMain && !pwalletMain->GetKey(keyID, key))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4105 - Could not get " + strAcceptorFQDN + _("'s private key ") + addressAcceptor.ToString());
    
    CHashWriter ss(SER_GETHASH, 0);
    ss << strMessageMagic;
    ss << strRequestorFQDN;
    std::vector<unsigned char> vchSig;
    if (!key.SignCompact(ss.GetHash(), vchSig))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4106 - Error signing " + strRequestorFQDN + _("'s signature proof."));

    std::vector<unsigned char> vchSignatureProof = vchFromString(EncodeBase64(&vchSig[0], vchSig.size()));
    txLinkAccept.SignatureProof = vchSignatureProof;

    uint64_t nDays = 1461;  //default to 4 years.
    // TODO (bdap): fix invalid int error when passing registration days.
    //    if (request.params.size() >= 4) {
    //        nDays = request.params[4].get_int();
    //    }
    uint64_t nSeconds = nDays * SECONDS_PER_DAY;
    txLinkAccept.nExpireTime = chainActive.Tip()->GetMedianTimePast() + nSeconds;
    CKeyEd25519 dhtKey;
    std::vector<unsigned char> vchSharedPubKey = GetLinkSharedPubKey(privAcceptDHTKey, entryRequestor.DHTPublicKey);
    txLinkAccept.SharedPubKey = vchSharedPubKey;

    // Create BDAP operation script
    CScript scriptPubKey;
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_NEW) << CScript::EncodeOP_N(OP_BDAP_LINK_ACCEPT) 
                 << vchDHTPubKey << vchSharedPubKey << txLinkAccept.nExpireTime << OP_2DROP << OP_2DROP << OP_DROP;
    CScript scriptDest = GetScriptForDestination(CPubKey(entryRequestor.LinkAddress).GetID());
    scriptPubKey += scriptDest;
    CScript scriptSend = GetScriptForDestination(CPubKey(entryAcceptor.LinkAddress).GetID());

    // check BDAP values
    std::string strMessage;
    if (!txLinkAccept.ValidateValues(strMessage))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4007 - Error validating link request values: " + strMessage);

    // TODO (bdap): encrypt data before adding it to OP_RETURN.
    // Create BDAP OP_RETURN script
    CharString data;
    txLinkAccept.Serialize(data);
    CScript scriptData;
    scriptData << OP_RETURN << data;

    // Send the transaction
    CWalletTx wtx;
    float fYears = ((float)nDays/365.25);
    CAmount nOperationFee = GetBDAPFee(scriptPubKey) * powf(3.1, fYears);
    CAmount nDataFee = GetBDAPFee(scriptData) * powf(3.1, fYears);
    bool fUseInstantSend = false;
    int enabled = dnodeman.CountEnabled();
    if (enabled > 5) {
        // TODO (bdap): calculate cost for instant send.
        nOperationFee = nOperationFee * 2;
        fUseInstantSend = true;
    }
    SendLinkingTransaction(scriptData, scriptPubKey, scriptSend, wtx, nDataFee, nOperationFee, fUseInstantSend);
    txLinkAccept.txHash = wtx.GetHash();

    UniValue oLink(UniValue::VOBJ);
    if(!BuildJsonLinkAcceptInfo(txLinkAccept, entryRequestor, entryAcceptor, oLink))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4005 - " + _("Failed to build BDAP link JSON object"));

    return oLink;
}

static bool BuildJsonMyListRequests(const std::vector<CLinkRequest>& vchLinkRequests, const std::string& strFromAccount, const std::string& strToAccount, UniValue& oLinkRequests)
{
    int nCount = 1;
    for (const CLinkRequest& link : vchLinkRequests) {
        UniValue oLink(UniValue::VOBJ);
        bool expired = false;
        int64_t expired_time = 0;
        int64_t nTime = 0;
        if (strFromAccount.empty() || strFromAccount == stringFromVch(link.RequestorFullObjectPath)) {
            if (strToAccount.empty() || strToAccount == stringFromVch(link.RecipientFullObjectPath)) {
                oLink.push_back(Pair("requestor_fqdn", stringFromVch(link.RequestorFullObjectPath)));
                oLink.push_back(Pair("recipient_fqdn", stringFromVch(link.RecipientFullObjectPath)));
                oLink.push_back(Pair("requestor_link_pubkey", link.RequestorPubKeyString()));
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
                oLinkRequests.push_back(Pair("link_request-" + std::to_string(nCount) , oLink));
                nCount ++;
            }
        }
    }
    
    return true;
}

static UniValue ListPendingLinkRequests(const JSONRPCRequest& request)
{
     if (request.fHelp || request.params.size() > 4)
        throw std::runtime_error(
            "lists pending link requests sent by your account\n"
            + HelpRequiringPassphrase() +
            "\nLink Send Arguments:\n"
            "1. from account                  (string, optional)    BDAP from account sending the link request\n"
            "2. to account                    (string, optional)    BDAP to account receiving the link request\n"
            "\nResult:\n"
            "{(json objects)\n"
            "  \"From Account\"               (string)  Requestor's BDAP full path\n"
            "  \"To Account\"                 (string)  Recipient's BDAP full path\n"
            "  \"Link Message\"               (string)  Message from requestor to recipient.\n"
            "  \"Link Request TxID\"          (string)  Transaction ID for the link request\n"
            "  \"Time\"                       (int)     Transaction time\n"
            "  \"Expires On\"                 (int)     Link request expiration\n"
            "  \"Expired\"                    (boolean) Is link request expired\n"
            "  },...n \n"
            "\nExamples:\n"
            + HelpExampleCli("link pending request", "superman batman") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link pending request", "superman batman"));


    std::string strFromAccountFQDN, strToAccountFQDN;
    if (request.params.size() > 2) {
        strFromAccountFQDN = request.params[2].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
        ToLowerCase(strFromAccountFQDN);
    }
    if (request.params.size() > 3) {
        strToAccountFQDN = request.params[3].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
        ToLowerCase(strToAccountFQDN);
    }

    if (!CheckLinkRequestDB())
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_RPC_ERROR: ERRCODE: 4200 - Can not open link request leveldb database");

    std::vector<CLinkRequest> vchLinkRequests;
    if (!pLinkRequestDB->ListMyLinkRequests(vchLinkRequests))
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_RPC_ERROR: ERRCODE: 4201 - Error listing link requests from leveldb");

    UniValue oLinks(UniValue::VOBJ);
    if (!BuildJsonMyListRequests(vchLinkRequests, strFromAccountFQDN, strToAccountFQDN, oLinks))
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_RPC_ERROR: ERRCODE: 4202 - Error creating JSON link requests.");

    return oLinks;
}

static bool BuildJsonMyListAccepts(const std::vector<CLinkAccept>& vchLinkAccepts, const std::string& strFromAccount, const std::string& strToAccount, UniValue& oLinkAccepts)
{
    int nCount = 1;
    for (const CLinkAccept& link : vchLinkAccepts) {
        UniValue oLink(UniValue::VOBJ);
        bool expired = false;
        int64_t expired_time = 0;
        int64_t nTime = 0;
        if (strFromAccount.empty() || strFromAccount == stringFromVch(link.RequestorFullObjectPath)) {
            if (strToAccount.empty() || strToAccount == stringFromVch(link.RecipientFullObjectPath)) {
                oLink.push_back(Pair("requestor_fqdn", stringFromVch(link.RequestorFullObjectPath)));
                oLink.push_back(Pair("recipient_fqdn", stringFromVch(link.RecipientFullObjectPath)));
                oLink.push_back(Pair("recipient_link_pubkey", link.RecipientPubKeyString()));
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
                oLinkAccepts.push_back(Pair("link_accept-" + std::to_string(nCount) , oLink));
                nCount ++;
            }
        }
    }
    
    return true;
}

static UniValue ListPendingLinkAccepts(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() > 4)
        throw std::runtime_error(
            "lists pending link accepts sent to your account\n"
            + HelpRequiringPassphrase() +
            "\nLink Send Arguments:\n"
            "1. from account                  (string, optional)    BDAP from account sending the link request\n"
            "2. to account                    (string, optional)    BDAP to account receiving the link request\n"
            "\nResult:\n"
            "{(json objects)\n"
            "  \"From Account\"               (string)  Requestor's BDAP full path\n"
            "  \"To Account\"                 (string)  Recipient's BDAP full path\n"
            "  \"Link Message\"               (string)  Message from requestor to recipient.\n"
            "  \"Link Request TxID\"          (string)  Transaction ID for the link request\n"
            "  \"Time\"                       (int)     Transaction time\n"
            "  \"Expires On\"                 (int)     Link request expiration\n"
            "  \"Expired\"                    (boolean) Is link request expired\n"
            "  },...n \n"
            "\nExamples:\n"
            + HelpExampleCli("link pending accept", "superman batman") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link pending accept", "superman batman"));


    std::string strFromAccountFQDN, strToAccountFQDN;
    if (request.params.size() > 2) {
        strFromAccountFQDN = request.params[2].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
        ToLowerCase(strFromAccountFQDN);
    }
    if (request.params.size() > 3) {
        strToAccountFQDN = request.params[3].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
        ToLowerCase(strToAccountFQDN);
    }
    
    if (!CheckLinkAcceptDB())
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_RPC_ERROR: ERRCODE: 4200 - Can not open link request leveldb database");

    std::vector<CLinkAccept> vchLinkAccepts;
    if (!pLinkAcceptDB->ListMyLinkAccepts(vchLinkAccepts))
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_RPC_ERROR: ERRCODE: 4201 - Error listing link requests from leveldb");

    UniValue oLinks(UniValue::VOBJ);
    if (!BuildJsonMyListAccepts(vchLinkAccepts, strFromAccountFQDN, strToAccountFQDN, oLinks))
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_RPC_ERROR: ERRCODE: 4202 - Error creating JSON link requests.");

    return oLinks;
}

UniValue link(const JSONRPCRequest& request) 
{
    std::string strCommand;
    if (request.params.size() >= 2) {
        strCommand = request.params[0].get_str();
        ToLowerCase(strCommand);
    }
    else {
        throw std::runtime_error(
            "link\n"
            + HelpRequiringPassphrase() +
            "\nLink commands are request, accept, pending, complete, and delete.\n"
            "\nExamples:\n"
            + HelpExampleCli("link accept", "superman batman") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link accept", "superman batman"));
    }
    if (strCommand == "request") {
        return SendLinkRequest(request);
    }
    else if (strCommand == "accept") {
        return SendLinkAccept(request);
    }
    else if (strCommand == "pending") {
        std::string strSubCommand = request.params[1].get_str();
        ToLowerCase(strSubCommand);
        LogPrintf("%s -- strCommand = %s, strSubCommand = %s\n", __func__, strCommand, strSubCommand);
        if (strSubCommand == "request") {
            return ListPendingLinkRequests(request);
        }
        else if (strSubCommand == "accept") {
            return ListPendingLinkAccepts(request);
        }
        else {
            throw std::runtime_error(
            "link pending\n"
            + HelpRequiringPassphrase() +
            "\nLink Pending sub-commands are request or accept.\n"
            "\nExamples:\n"
            + HelpExampleCli("link pending request", "") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link pending request", ""));
        }
    }
    /*
    else if (strCommand == "complete") {
        return ListCompletedLinks(request);
    }
    else if (strCommand == "delete") {
        std::string strSubCommand = request.params[1].get_str();
        if (strSubCommand == "request") {
            return DeletePendingLinkRequest(request);
        }
        else if (strSubCommand == "accept") {
            return DeletePendingLinkAccept(request);
        }
        else {
            throw std::runtime_error(
            "link delete\n"
            + HelpRequiringPassphrase() +
            "\nDelete Link sub-commands are request or accept.\n"
            "\nExamples:\n"
            + HelpExampleCli("link delete request", "superman batman") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link delete request", "superman batman"));
        }
    }
    */
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