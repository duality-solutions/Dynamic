// Copyright (c) 2018 Duality Blockchain Solutions Developers 
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/bdap.h"
#include "bdap/domainentry.h"
#include "bdap/domainentrydb.h"
#include "encryption.h"
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
    oLink.push_back(Pair("signature_proof", link.SignatureProofString()));
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
    oLink.push_back(Pair("signature_proof", link.SignatureProofString()));
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
    if (request.fHelp || request.params.size() < 4 || request.params.size() > 5)
        throw std::runtime_error(
            "link request userid-from userid-to message\n"
            "Creates a link request transaction on the blockchain."
            + HelpRequiringPassphrase() +
            "\nLink Send Arguments:\n"
            "1. requestor          (string)             BDAP account requesting the link\n"
            "2. recipient          (string)             Link recipient's BDAP account.\n"
            "3. invite message     (string)             Message from requestor to recipient.\n"
            "4. registration days  (int, optional)      Number of days to keep the link request on the blockchain before pruning.\n"
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

    // Check if link request or accept already exists
    if (!CheckLinkRequestDB())
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4000 - Can not open link request leveldb database");

    if (CheckLinkageRequestExists(strRequestorFQDN, strRecipientFQDN))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4001 - Link request or accept already exists for these accounts.");

    CLinkRequest txLink;
    txLink.nVersion = 1; // version 1 = encrytped or a private link.
    txLink.RequestorFullObjectPath = vchRequestorFQDN;
    txLink.RecipientFullObjectPath = vchRecipientFQDN;
    txLink.LinkMessage = vchFromString(strLinkMessage);
    CKeyEd25519 privReqDHTKey;
    CharString vchDHTPubKey = privReqDHTKey.GetPubKey();
    if (pwalletMain && !pwalletMain->AddDHTKey(privReqDHTKey, vchDHTPubKey))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4002 - " + _("Error adding ed25519 key to wallet for BDAP link"));
 
    txLink.RequestorPubKey = vchDHTPubKey;

    pwalletMain->SetAddressBook(privReqDHTKey.GetID(), strRequestorFQDN, "bdap-dht-key");

    // Check if pubkey already exists
    uint256 prevTxID;
    if (GetLinkRequestIndex(txLink.RequestorPubKey, prevTxID))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4003 - " + txLink.RequestorPubKeyString() + _(" entry already exists.  Can not add duplicate."));

    // Get requestor link address
    CDomainEntry entryRequestor;
    if (!GetDomainEntry(vchRequestorFQDN, entryRequestor))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4004 - Requestor " + strRequestorFQDN + _(" not found."));

    // Get recipient link address
    CDomainEntry entryRecipient;
    if (!GetDomainEntry(vchRecipientFQDN, entryRecipient))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4005 - Recipient " + strRecipientFQDN + _(" not found."));

    CDynamicAddress addressRequestor = entryRequestor.GetWalletAddress();
    CKeyID keyID;
    if (!addressRequestor.GetKeyID(keyID))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4006 - Could not get " + strRequestorFQDN + _("'s wallet address key ") + addressRequestor.ToString());

    CKey key;
    if (pwalletMain && !pwalletMain->GetKey(keyID, key))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4007 - Could not get " + strRequestorFQDN + _("'s private key ") + addressRequestor.ToString());
    
    if (!CreateSignatureProof(key, strRecipientFQDN, txLink.SignatureProof))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4008 - Error signing " + strRequestorFQDN + _("'s signature proof."));

    int64_t nDays = DEFAULT_REGISTRATION_DAYS;  //default to 4 years.
    if (request.params.size() > 4) {
         if (!ParseInt64(request.params[4].get_str(), &nDays))
            throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4009 - Error parsing registration days parameter = " + request.params[4].get_str());
    }

    int64_t nSeconds = nDays * SECONDS_PER_DAY;
    txLink.nExpireTime = chainActive.Tip()->GetMedianTimePast() + nSeconds;
    CKeyEd25519 dhtKey;
    std::vector<unsigned char> vchSharedPubKey = GetLinkSharedPubKey(privReqDHTKey, entryRecipient.DHTPublicKey);
    txLink.SharedPubKey = vchSharedPubKey;

    // Create BDAP operation script
    CScript scriptPubKey;
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_NEW) << CScript::EncodeOP_N(OP_BDAP_LINK_REQUEST) 
                 << vchDHTPubKey << vchSharedPubKey << txLink.nExpireTime << OP_2DROP << OP_2DROP << OP_DROP;
    CScript scriptDest = GetScriptForDestination(entryRecipient.GetLinkAddress().Get());
    scriptPubKey += scriptDest;
    CScript scriptSend = GetScriptForDestination(entryRequestor.GetLinkAddress().Get());

    // check BDAP values
    std::string strMessage;
    if (!txLink.ValidateValues(strMessage))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4010 - Error validating link request values: " + strMessage);

    // TODO (bdap): encrypt data before adding it to OP_RETURN.
    // Create BDAP OP_RETURN script
    CharString data;
    txLink.Serialize(data);

    // Encrypt serialized data for the sender and recipient
    strMessage = "";
    std::vector<std::vector<unsigned char>> vvchPubKeys;
    vvchPubKeys.push_back(privReqDHTKey.GetPubKeyBytes()); // do we need to add the sender's pubkey?
    vvchPubKeys.push_back(EncodedPubKeyToBytes(vchSharedPubKey));
    std::vector<unsigned char> dataEncrypted;
    if (!EncryptBDAPData(vvchPubKeys, data, dataEncrypted, strMessage))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4011 - Error encrypting link data: " + strMessage);

    dataEncrypted = AddVersionToLinkData(dataEncrypted, 1);

    // Create BDAP OP_RETURN script
    CScript scriptData;
    scriptData << OP_RETURN << dataEncrypted;

    // Send the transaction
    CWalletTx wtx;
    float fYears = ((float)nDays/365.25);
    CAmount nOperationFee = GetBDAPFee(scriptPubKey) * powf(3.1, fYears);
    CAmount nDataFee = GetBDAPFee(scriptData) * powf(3.1, fYears);

    bool fUseInstantSend = false;
    SendLinkingTransaction(scriptData, scriptPubKey, scriptSend, wtx, nOperationFee, nDataFee, fUseInstantSend);
    txLink.txHash = wtx.GetHash();

    UniValue oLink(UniValue::VOBJ);
    if(!BuildJsonLinkRequestInfo(txLink, entryRequestor, entryRecipient, oLink))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4014 - " + _("Failed to build BDAP link JSON object"));

    return oLink;
}

static UniValue SendLinkAccept(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() < 3 || request.params.size() > 4)
        throw std::runtime_error(
            "link accept userid-from userid-to\n"
            "Creates a link accept transaction on the blockchain."
            + HelpRequiringPassphrase() +
            "\nLink Send Arguments:\n"
            "1. accept account          (string)             BDAP account accepting the link\n"
            "2. requestor account       (string)             Link requestor's BDAP account.\n"
            "3. registration days       (int, optional)      Number of days to keep the link accept on the blockchain before pruning.\n"
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

    if (!CheckLinkRequestDB() || !CheckLinkAcceptDB())
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4100 - Can not open link request or accept leveldb database");

    // Before creating an accept link tx, check if request exists
    if (!CheckLinkageRequestExists(strRequestorFQDN, strAcceptorFQDN))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4101 - Link request not found. Accept link failed.");

    // Check if link accept already exists
    if (CheckLinkageAcceptExists(strRequestorFQDN, strAcceptorFQDN))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4102 - Accept link already exists for these accounts.");

    // get link request
    CLinkRequest prevLink;
    if (!pLinkRequestDB->GetMyLinkRequest(strRequestorFQDN, strAcceptorFQDN, prevLink))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4103 - Initial link request not found.");

    CLinkAccept txLinkAccept;
    txLinkAccept.nVersion = 1; // version 1 = encrytped or a private link.
    txLinkAccept.RequestorFullObjectPath = vchRequestorFQDN;
    txLinkAccept.RecipientFullObjectPath = vchAcceptorFQDN;

    CKeyEd25519 privAcceptDHTKey;
    CharString vchDHTPubKey = privAcceptDHTKey.GetPubKey();
    if (pwalletMain && !pwalletMain->AddDHTKey(privAcceptDHTKey, vchDHTPubKey))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4104 - " + _("Error adding ed25519 key to wallet for BDAP link"));
 
    txLinkAccept.RecipientPubKey = vchDHTPubKey;

    pwalletMain->SetAddressBook(privAcceptDHTKey.GetID(), strAcceptorFQDN, "bdap-dht-key");

    // Check if pubkey already exists
    uint256 prevTxID;
    if (GetLinkAcceptIndex(txLinkAccept.RecipientPubKey, prevTxID))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4105 - " + txLinkAccept.RecipientPubKeyString() + _(" entry already exists.  Can not add duplicate."));

    // Get link accepting address
    CDomainEntry entryAcceptor;
    if (!GetDomainEntry(vchAcceptorFQDN, entryAcceptor))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4106 - Acceptor " + strAcceptorFQDN + _(" not found."));

    // Get requestor link address
    CDomainEntry entryRequestor;
    if (!GetDomainEntry(vchRequestorFQDN, entryRequestor))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4107 - Requestor " + strRequestorFQDN + _(" not found."));

    CDynamicAddress addressAcceptor = entryAcceptor.GetWalletAddress();
    CKeyID keyID;
    if (!addressAcceptor.GetKeyID(keyID))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4108 - Could not get " + strAcceptorFQDN + _("'s wallet address key ") + addressAcceptor.ToString());

    CKey key;
    if (pwalletMain && !pwalletMain->GetKey(keyID, key))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4109 - Could not get " + strAcceptorFQDN + _("'s private key ") + addressAcceptor.ToString());

    if (!CreateSignatureProof(key, strRequestorFQDN, txLinkAccept.SignatureProof))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4110 - Error signing " + strRequestorFQDN + _("'s signature proof."));

    int64_t nDays = DEFAULT_REGISTRATION_DAYS;  //default to 4 years.
    if (request.params.size() > 3) {
        if (!ParseInt64(request.params[3].get_str(), &nDays))
            throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4111 - Error parsing registration days parameter = " + request.params[3].get_str());
    }
    int64_t nSeconds = nDays * SECONDS_PER_DAY;
    txLinkAccept.nExpireTime = chainActive.Tip()->GetMedianTimePast() + nSeconds;
    CKeyEd25519 dhtKey;
    std::vector<unsigned char> vchSharedPubKey = GetLinkSharedPubKey(privAcceptDHTKey, entryRequestor.DHTPublicKey);
    txLinkAccept.SharedPubKey = vchSharedPubKey;

    // Create BDAP operation script
    CScript scriptPubKey;
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_NEW) << CScript::EncodeOP_N(OP_BDAP_LINK_ACCEPT) 
                 << vchDHTPubKey << vchSharedPubKey << txLinkAccept.nExpireTime << OP_2DROP << OP_2DROP << OP_DROP;
    CScript scriptDest = GetScriptForDestination(entryRequestor.GetLinkAddress().Get());
    scriptPubKey += scriptDest;
    CScript scriptSend = GetScriptForDestination(entryAcceptor.GetLinkAddress().Get());

    // check BDAP values
    std::string strMessage;
    if (!txLinkAccept.ValidateValues(strMessage))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4012 - Error validating link request values: " + strMessage);

    CharString data;
    txLinkAccept.Serialize(data);
    // Encrypt serialized data for the sender and recipient
    strMessage = "";
    std::vector<std::vector<unsigned char>> vvchPubKeys;
    vvchPubKeys.push_back(privAcceptDHTKey.GetPubKeyBytes()); // do we need to add the recipient's pubkey?
    vvchPubKeys.push_back(EncodedPubKeyToBytes(vchSharedPubKey));
    std::vector<unsigned char> dataEncrypted;
    if (!EncryptBDAPData(vvchPubKeys, data, dataEncrypted, strMessage))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4011 - Error encrypting link data: " + strMessage);

    dataEncrypted = AddVersionToLinkData(dataEncrypted, 1);

    // Create BDAP OP_RETURN script
    CScript scriptData;
    scriptData << OP_RETURN << dataEncrypted;

    // Send the transaction
    CWalletTx wtx;
    float fYears = ((float)nDays/365.25);
    CAmount nOperationFee = GetBDAPFee(scriptPubKey) * powf(3.1, fYears);
    CAmount nDataFee = GetBDAPFee(scriptData) * powf(3.1, fYears);
    bool fUseInstantSend = false;
    SendLinkingTransaction(scriptData, scriptPubKey, scriptSend, wtx, nOperationFee, nDataFee, fUseInstantSend);
    txLinkAccept.txHash = wtx.GetHash();

    UniValue oLink(UniValue::VOBJ);
    if(!BuildJsonLinkAcceptInfo(txLinkAccept, entryRequestor, entryAcceptor, oLink))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4013 - " + _("Failed to build BDAP link JSON object"));

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
                oLinkRequests.push_back(Pair("link-" + std::to_string(nCount) , oLink));
                nCount ++;
            }
        }
    }
    
    return true;
}

static bool LinkCompleted(const CLinkRequest& link)
{
    return pLinkAcceptDB->MyLinkageExists(link.RequestorFQDN(), link.RecipientFQDN());
}

static std::vector<CLinkRequest> GetMyPendingLinks(const std::vector<CLinkRequest>& vchPendingLinks, const bool fLinkRequests)
{
    std::vector<CLinkRequest> vchMyPendingLink;
    CKeyEd25519 key;
    for (const CLinkRequest& link : vchPendingLinks) {
        if (LinkCompleted(link))
            continue;

        if (fLinkRequests) {
            std::vector<unsigned char> vchRequestorPubKey = link.RequestorPubKey;
            CKeyID keyID(Hash160(vchRequestorPubKey.begin(), vchRequestorPubKey.end()));
            bool fHaveKey = pwalletMain->HaveDHTKey(keyID);
            if ((fHaveKey)) {
                vchMyPendingLink.push_back(link);
            }
        }
        else {
            // Get link recipient domain entry
            CDomainEntry entryRecipient;
            if (GetDomainEntry(link.RecipientFullObjectPath, entryRecipient)) {
                std::vector<unsigned char> vchRecipientPubKey = entryRecipient.DHTPublicKey;
                CKeyID keyID(Hash160(vchRecipientPubKey.begin(), vchRecipientPubKey.end())); 
                bool fHaveKey = pwalletMain->HaveDHTKey(keyID);
                if ((fHaveKey)) {
                    vchMyPendingLink.push_back(link);
                }
            }
        }
    }

    return vchMyPendingLink;
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
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_REQ_RPC_ERROR: ERRCODE: 4200 - Can not open link request leveldb database");

    std::vector<CLinkRequest> vchPendingLinks;
    if (!pLinkRequestDB->ListMyLinkRequests(vchPendingLinks))
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_REQ_RPC_ERROR: ERRCODE: 4201 - Error listing link requests from leveldb");

    if (!pwalletMain)
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_REQ_RPC_ERROR: ERRCODE: 4202 - Error using wallet database.");

    std::vector<CLinkRequest> vchMyLinkRequests = GetMyPendingLinks(vchPendingLinks, true);

    UniValue oLinks(UniValue::VOBJ);
    if (!BuildJsonMyListRequests(vchMyLinkRequests, strFromAccountFQDN, strToAccountFQDN, oLinks))
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_REQ_RPC_ERROR: ERRCODE: 4203 - Error creating JSON link requests.");

    return oLinks;
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
    
    if (!CheckLinkRequestDB())
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_ACCEPT_RPC_ERROR: ERRCODE: 4210 - Can not open link request leveldb database");

    std::vector<CLinkRequest> vchPendingLinks;
    if (!pLinkRequestDB->ListMyLinkRequests(vchPendingLinks))
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_ACCEPT_RPC_ERROR: ERRCODE: 4211 - Error listing link requests from leveldb");

    if (!pwalletMain)
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_ACCEPT_RPC_ERROR: ERRCODE: 4212 - Error using wallet database.");

    std::vector<CLinkRequest> vchMyLinkAccepts = GetMyPendingLinks(vchPendingLinks, false);

    UniValue oLinks(UniValue::VOBJ);
    if (!BuildJsonMyListRequests(vchMyLinkAccepts, strFromAccountFQDN, strToAccountFQDN, oLinks))
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_ACCEPT_RPC_ERROR: ERRCODE: 4213 - Error creating JSON link requests.");

    return oLinks;
}

static bool BuildJsonMyCompletedLinks(const std::vector<CLinkAccept>& vchLinkAccepts, UniValue& oLinkAccepts)
{
    int nCount = 1;
    for (const CLinkAccept& link : vchLinkAccepts) {
        UniValue oLink(UniValue::VOBJ);
        bool expired = false;
        int64_t expired_time = 0;
        int64_t nTime = 0;

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
        oLinkAccepts.push_back(Pair("link_completed-" + std::to_string(nCount) , oLink));
        nCount ++;
    }

    return true;
}

static UniValue ListCompletedLinks(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() > 2)
        throw std::runtime_error(
            "lists completed links\n"
            "\nLink Complete Arguments: {none}\n"
            + HelpRequiringPassphrase() +
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
            + HelpExampleCli("link complete", "") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link complete", ""));

    if (!CheckLinkAcceptDB())
        throw std::runtime_error("BDAP_LINK_COMPLETED_RPC_ERROR: ERRCODE: 4220 - Can not open link request leveldb database");

    std::vector<CLinkAccept> vchLinkAccepts;
    if (!pLinkAcceptDB->ListMyLinkAccepts(vchLinkAccepts))
        throw std::runtime_error("BDAP_LINK_COMPLETED_RPC_ERROR: ERRCODE: 4221 - Error listing link requests from leveldb");

    UniValue oLinks(UniValue::VOBJ);
    if (!BuildJsonMyCompletedLinks(vchLinkAccepts, oLinks))
        throw std::runtime_error("BDAP_LINK_COMPLETED_RPC_ERROR: ERRCODE: 4222 - Error creating JSON link requests.");

    return oLinks;
}

static UniValue DeleteLink(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 3)
        throw std::runtime_error(
            "link delete userid-from userid-to\n"
            "Creates a deletes link request or link accepted transaction on the blockchain."
            + HelpRequiringPassphrase() +
            "\nLink Send Arguments:\n"
            "1. requestor          (string)             BDAP account requesting the link\n"
            "2. recipient          (string)             Link recipient's BDAP account.\n"
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
            + HelpExampleCli("link delete", "superman batman") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link delete", "superman batman"));

    std::string strRequestorFQDN = request.params[1].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strRequestorFQDN);
    CharString vchRequestorFQDN = vchFromString(strRequestorFQDN);

    std::string strRecipientFQDN = request.params[2].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strRecipientFQDN);
    CharString vchRecipientFQDN = vchFromString(strRecipientFQDN);

    if (!CheckLinkAcceptDB())
        throw std::runtime_error("BDAP_DELETE_LINK_RPC_ERROR: ERRCODE: 4230 - Can not open link request leveldb database");

    bool fLinkFound = false;
    CScript scriptPubKey, scriptSend, scriptDest, scriptData;
    UniValue oLink(UniValue::VOBJ);

    // Try to find accepted link entry
    CLinkAccept linkAccept;
    if (pLinkAcceptDB->GetMyLinkAccept(strRequestorFQDN, strRecipientFQDN, linkAccept)){
        CDomainEntry entryRequestor, entryRecipient;
        if (GetDomainEntry(linkAccept.RequestorFullObjectPath, entryRequestor) && GetDomainEntry(linkAccept.RecipientFullObjectPath, entryRecipient)) {
            // Create BDAP operation script
            scriptPubKey << CScript::EncodeOP_N(OP_BDAP_DELETE) << CScript::EncodeOP_N(OP_BDAP_LINK_ACCEPT) 
                         << linkAccept.RecipientPubKey << linkAccept.SharedPubKey << OP_2DROP << OP_2DROP;
            scriptDest = GetScriptForDestination(entryRequestor.GetLinkAddress().Get());
            scriptPubKey += scriptDest;
            scriptSend = GetScriptForDestination(entryRecipient.GetLinkAddress().Get());
            if(!BuildJsonLinkAcceptInfo(linkAccept, entryRequestor, entryRecipient, oLink))
                throw std::runtime_error("BDAP_DELETE_LINK_RPC_ERROR: ERRCODE: 4231 - " + _("Failed to build BDAP link JSON object"));
            fLinkFound = true;
        }
    }

    if (!fLinkFound) {
        if (!CheckLinkRequestDB())
            throw std::runtime_error("BDAP_DELETE_LINK_RPC_ERROR: ERRCODE: 4232 - Can not open link request leveldb database");
        // Try to find link request entry
        CLinkRequest linkRequest;
        if (pLinkRequestDB->GetMyLinkRequest(strRequestorFQDN, strRecipientFQDN, linkRequest)){
            CDomainEntry entryRequestor, entryRecipient;
            if (GetDomainEntry(linkRequest.RecipientFullObjectPath, entryRecipient) && GetDomainEntry(linkRequest.RequestorFullObjectPath, entryRequestor)) {
                // Create BDAP operation script
                scriptPubKey << CScript::EncodeOP_N(OP_BDAP_DELETE) << CScript::EncodeOP_N(OP_BDAP_LINK_REQUEST) 
                             << linkRequest.RequestorPubKey << linkRequest.SharedPubKey << OP_2DROP << OP_2DROP;
                scriptDest = GetScriptForDestination(entryRecipient.GetLinkAddress().Get());
                scriptPubKey += scriptDest;
                scriptSend = GetScriptForDestination(entryRequestor.GetLinkAddress().Get());
                if(!BuildJsonLinkRequestInfo(linkRequest, entryRequestor, entryRecipient, oLink))
                    throw std::runtime_error("BDAP_DELETE_LINK_RPC_ERROR: ERRCODE: 4233 - " + _("Failed to build BDAP link JSON object"));
                fLinkFound = true;
            }
        }
    }

    if (!fLinkFound)
        throw std::runtime_error("BDAP_DELETE_LINK_RPC_ERROR: ERRCODE: 4234 - Can not find link request or link accept in database.");

    // Send the transaction
    CWalletTx wtx;
    CAmount nOperationFee = (GetBDAPFee(scriptPubKey) * powf(3.1, 1)) + GetDataFee(scriptPubKey);
    CAmount nDataFee = 0; // No OP_RETURN data needed for deleted account transactions

    bool fUseInstantSend = false;
    SendLinkingTransaction(scriptData, scriptPubKey, scriptSend, wtx, nOperationFee, nDataFee, fUseInstantSend);

    return oLink;
}

UniValue link(const JSONRPCRequest& request) 
{
    std::string strCommand;
    if (request.params.size() >= 1) {
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
    else if (strCommand == "complete") {
        return ListCompletedLinks(request);
    }
    else if (strCommand == "delete") {
        return DeleteLink(request);
    }
    else if (strCommand == "pending") {
        if (request.params.size() >= 2) {
            std::string strSubCommand = request.params[1].get_str();
            ToLowerCase(strSubCommand);
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
    { "bdap",            "link",                     &link,                         true, {"operation", "command", "from", "to", "days"} },
#endif // ENABLE_WALLET

};
void RegisterLinkingRPCCommands(CRPCTable &t)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        t.appendCommand(commands[vcidx].name, &commands[vcidx]);
}