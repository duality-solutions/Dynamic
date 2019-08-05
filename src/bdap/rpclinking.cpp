// Copyright (c) 2019 Duality Blockchain Solutions Developers 
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/bdap.h"
#include "bdap/domainentry.h"
#include "bdap/domainentrydb.h"
#include "bdap/fees.h"
#include "bdap/vgp/include/encryption.h" // for VGP E2E encryption
#include "bdap/linking.h"
#include "bdap/linkingdb.h"
#include "bdap/linkmanager.h"
#include "bdap/utils.h"
#include "bdap/vgpmessage.h"
#include "dht/ed25519.h"
#include "dht/datarecord.h" // for CDataRecord
#include "dht/session.h" // for CDataRecord
#include "core_io.h" // needed for ScriptToAsmStr
#include "dynodeman.h"
#include "hash.h"
#include "rpcprotocol.h"
#include "rpcserver.h"
#include "spork.h"
#include "primitives/transaction.h"
#include "wallet/wallet.h"
#include "uint256.h"
#include "utilmoneystr.h"
#include "utiltime.h"
#include "validation.h"

#include <univalue.h>

#ifdef ENABLE_WALLET

extern void SendLinkingTransaction(const CScript& bdapDataScript, const CScript& bdapOPScript, const CScript& stealthScript, 
                                    CWalletTx& wtxNew, const CAmount& nOneTimeFee, const CAmount& nDepositFee, const bool fUseInstantSend);


static bool BuildJsonLinkRequestInfo(const CLinkRequest& link, const CDomainEntry& requestor, const CDomainEntry& recipient, UniValue& oLink)
{
    bool expired = false;
    int64_t expired_time = 0;
    int64_t nTime = 0;
    oLink.push_back(Pair("requestor_fqdn", requestor.GetFullObjectPath()));
    oLink.push_back(Pair("recipient_fqdn", recipient.GetFullObjectPath()));
    oLink.push_back(Pair("requestor_link_pubkey", link.RequestorPubKeyString()));
    oLink.push_back(Pair("requestor_wallet_address", stringFromVch(requestor.WalletAddress))); //was LinkAddress
    oLink.push_back(Pair("recipient_wallet_address", stringFromVch(recipient.WalletAddress))); //was LinkAddress
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
    if (expired_time != 0)
    {
        if (expired_time <= (int64_t)chainActive.Tip()->GetMedianTimePast())
        {
            expired = true;
        }
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
    oLink.push_back(Pair("requestor_wallet_address", stringFromVch(requestor.WalletAddress))); //was LinkAddress
    oLink.push_back(Pair("recipient_wallet_address", stringFromVch(recipient.WalletAddress))); //was LinkAddress
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
    if (expired_time != 0)
    {
        if (expired_time <= (int64_t)chainActive.Tip()->GetMedianTimePast())
        {
            expired = true;
        }
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

    EnsureWalletIsUnlocked();

    std::string strRequestorFQDN = request.params[1].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strRequestorFQDN);
    CharString vchRequestorFQDN = vchFromString(strRequestorFQDN);

    std::string strRecipientFQDN = request.params[2].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strRecipientFQDN);
    CharString vchRecipientFQDN = vchFromString(strRecipientFQDN);
    
    std::string strLinkMessage = request.params[3].get_str();

    // get link request
    CLink prevLink;
    uint256 linkID = GetLinkID(strRequestorFQDN, strRecipientFQDN);
    if (pLinkManager->FindLink(linkID, prevLink))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4001 - Link request already exists for those accounts.");

    CLinkRequest txLink;
    txLink.nVersion = 1; // version 1 = encrytped or a private link.
    txLink.RequestorFullObjectPath = vchRequestorFQDN;
    txLink.RecipientFullObjectPath = vchRecipientFQDN;
    txLink.LinkMessage = vchFromString(strLinkMessage);

    CPubKey pubWalletKey; //won't be needing this
    CharString vchDHTPubKey;
    CKeyEd25519 privReqDHTKey;
    if (!pwalletMain->GetKeysFromPool(pubWalletKey, vchDHTPubKey, true))
        throw JSONRPCError(RPC_WALLET_KEYPOOL_RAN_OUT, "Error: Keypool ran out, please call keypoolrefill first");

    CKeyID vchDHTPubKeyID = GetIdFromCharVector(vchDHTPubKey);
    if (!pwalletMain->GetDHTKey(vchDHTPubKeyID, privReqDHTKey))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: Unable to retrieve DHT Key");

    txLink.RequestorPubKey = vchDHTPubKey;

    pwalletMain->SetAddressBook(privReqDHTKey.GetID(), strRequestorFQDN, "bdap-dht-key");

    // Check if pubkey already exists
    uint256 prevTxID;
    if (GetLinkIndex(txLink.RequestorPubKey, prevTxID))
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

    txLink.nExpireTime = DEFAULT_LINK_EXPIRE_TIME;
    CKeyEd25519 dhtKey;
    std::vector<unsigned char> vchSharedPubKey = GetLinkSharedPubKey(privReqDHTKey, entryRecipient.DHTPublicKey);
    txLink.SharedPubKey = vchSharedPubKey;

    // Create BDAP operation script
    CScript scriptPubKey;
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_NEW) << CScript::EncodeOP_N(OP_BDAP_LINK_REQUEST) 
                 << vchDHTPubKey << vchSharedPubKey << txLink.nExpireTime << OP_2DROP << OP_2DROP << OP_DROP;

    // Create OP Script with destination address
    bool fStealthAddress = false;
    CTxDestination dest = DecodeDestination(entryRecipient.GetLinkAddress().ToString());
    if (!IsValidDestination(dest))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, strprintf("Invalid destination address %s", entryRecipient.GetLinkAddress().ToString()));

    CScript scriptDest;
    std::vector<uint8_t> vStealthData;
    if (dest.type() == typeid(CStealthAddress))
    {
        CStealthAddress sxAddr = boost::get<CStealthAddress>(dest);
        std::string sError;
        if (0 != PrepareStealthOutput(sxAddr, scriptDest, vStealthData, sError)) {
            LogPrintf("%s -- PrepareStealthOutput failed. Error = %s\n", __func__, sError);
            throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, strprintf("Invalid stealth destination address %s", entryRecipient.GetLinkAddress().ToString()));
        }
        fStealthAddress = true;
    }
    else
    {
        scriptDest = GetScriptForDestination(dest);
    }
    scriptPubKey += scriptDest;
    CScript stealthScript;
    if (fStealthAddress) {
        stealthScript << OP_RETURN << vStealthData;
    }

    // check BDAP values
    std::string strMessage;
    if (!txLink.ValidateValues(strMessage))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4010 - Error validating link request values: " + strMessage);

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

    // Get BDAP fees
    BDAP::ObjectType bdapType = ObjectType::BDAP_LINK_REQUEST;
    int32_t nMonths = 0; // Links do not expire
    CAmount monthlyFee, oneTimeFee, depositFee;
    if (!GetBDAPFees(OP_BDAP_NEW, OP_BDAP_LINK_REQUEST, bdapType, nMonths, monthlyFee, oneTimeFee, depositFee))
        throw JSONRPCError(RPC_BDAP_FEE_UNKNOWN, strprintf("Error calculating BDAP fees."));
    LogPrint("bdap", "%s -- monthlyFee %d, oneTimeFee %d, depositFee %d\n", __func__, FormatMoney(monthlyFee), FormatMoney(oneTimeFee), FormatMoney(depositFee));
    // Send the transaction
    CWalletTx wtx;
    bool fUseInstantSend = false;
    SendLinkingTransaction(scriptData, scriptPubKey, stealthScript, wtx, monthlyFee + oneTimeFee, depositFee, fUseInstantSend);

    txLink.txHash = wtx.GetHash();

    UniValue oLink(UniValue::VOBJ);
    if(!BuildJsonLinkRequestInfo(txLink, entryRequestor, entryRecipient, oLink))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4014 - " + _("Failed to build BDAP link JSON object"));

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

    EnsureWalletIsUnlocked();

    std::string strAcceptorFQDN = request.params[1].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strAcceptorFQDN);
    CharString vchAcceptorFQDN = vchFromString(strAcceptorFQDN);

    std::string strRequestorFQDN = request.params[2].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strRequestorFQDN);
    CharString vchRequestorFQDN = vchFromString(strRequestorFQDN);

    // get link request
    CLink prevLink;
    uint256 linkID = GetLinkID(strRequestorFQDN, strAcceptorFQDN);
    if (!pLinkManager->FindLink(linkID, prevLink))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4102 - Initial link request not found.");

    if (!(prevLink.nLinkState == 1))
        throw std::runtime_error("BDAP_ACCEPT_LINK_RPC_ERROR: ERRCODE: 4103 - Link request is not pending.");

    CLinkAccept txLinkAccept;
    txLinkAccept.nVersion = 1; // version 1 = encrytped or a private link.
    txLinkAccept.RequestorFullObjectPath = vchRequestorFQDN;
    txLinkAccept.RecipientFullObjectPath = vchAcceptorFQDN;

    CPubKey pubWalletKey; //won't be needing this
    CharString vchDHTPubKey;
    CKeyEd25519 privAcceptDHTKey;
    if (!pwalletMain->GetKeysFromPool(pubWalletKey, vchDHTPubKey, true))
        throw JSONRPCError(RPC_WALLET_KEYPOOL_RAN_OUT, "Error: Keypool ran out, please call keypoolrefill first");

    CKeyID vchDHTPubKeyID = GetIdFromCharVector(vchDHTPubKey);
    if (!pwalletMain->GetDHTKey(vchDHTPubKeyID, privAcceptDHTKey))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: Unable to retrieve DHT Key");
 
    txLinkAccept.RecipientPubKey = vchDHTPubKey;

    pwalletMain->SetAddressBook(privAcceptDHTKey.GetID(), strAcceptorFQDN, "bdap-dht-key");

    // Check if pubkey already exists
    uint256 prevTxID;
    if (GetLinkIndex(txLinkAccept.RecipientPubKey, prevTxID))
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

    txLinkAccept.nExpireTime = DEFAULT_LINK_EXPIRE_TIME;
    CKeyEd25519 dhtKey;
    std::vector<unsigned char> vchSharedPubKey = GetLinkSharedPubKey(privAcceptDHTKey, entryRequestor.DHTPublicKey);
    txLinkAccept.SharedPubKey = vchSharedPubKey;

    // Create BDAP operation script
    CScript scriptPubKey;
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_NEW) << CScript::EncodeOP_N(OP_BDAP_LINK_ACCEPT) 
                 << vchDHTPubKey << vchSharedPubKey << txLinkAccept.nExpireTime << OP_2DROP << OP_2DROP << OP_DROP;

    // Create OP Script with destination address
    bool fStealthAddress = false;
    CTxDestination dest = DecodeDestination(entryRequestor.GetLinkAddress().ToString());
    if (!IsValidDestination(dest))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, strprintf("Invalid destination address %s", entryRequestor.GetLinkAddress().ToString()));

    CScript scriptDest;
    std::vector<uint8_t> vStealthData;
    if (dest.type() == typeid(CStealthAddress))
    {
        CStealthAddress sxAddr = boost::get<CStealthAddress>(dest);
        std::string sError;
        if (0 != PrepareStealthOutput(sxAddr, scriptDest, vStealthData, sError)) {
            LogPrintf("%s -- PrepareStealthOutput failed. Error = %s\n", __func__, sError);
            throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, strprintf("Invalid stealth destination address %s", entryRequestor.GetLinkAddress().ToString()));
        }
        fStealthAddress = true;
    }
    else
    {
        scriptDest = GetScriptForDestination(dest);
    }
    scriptPubKey += scriptDest;
    CScript stealthScript;
    if (fStealthAddress) {
        stealthScript << OP_RETURN << vStealthData;
    }

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

    // Get BDAP fees
    BDAP::ObjectType bdapType = ObjectType::BDAP_LINK_ACCEPT;
    int32_t nMonths = 0; // Links do not expire
    CAmount monthlyFee, oneTimeFee, depositFee;
    if (!GetBDAPFees(OP_BDAP_NEW, OP_BDAP_LINK_ACCEPT, bdapType, nMonths, monthlyFee, oneTimeFee, depositFee))
        throw JSONRPCError(RPC_BDAP_FEE_UNKNOWN, strprintf("Error calculating BDAP fees."));
    LogPrint("bdap", "%s -- monthlyFee %d, oneTimeFee %d, depositFee %d\n", __func__, FormatMoney(monthlyFee), FormatMoney(oneTimeFee), FormatMoney(depositFee));

    // Send the transaction
    CWalletTx wtx;
    bool fUseInstantSend = false;
    SendLinkingTransaction(scriptData, scriptPubKey, stealthScript, wtx, monthlyFee + oneTimeFee, depositFee, fUseInstantSend);

    txLinkAccept.txHash = wtx.GetHash();

    UniValue oLink(UniValue::VOBJ);
    if(!BuildJsonLinkAcceptInfo(txLinkAccept, entryRequestor, entryAcceptor, oLink))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: ERRCODE: 4013 - " + _("Failed to build BDAP link JSON object"));

    return oLink;
}

static bool BuildJsonMyLists(const std::vector<CLink>& vchLinkRequests, const std::string& strFromAccount, const std::string& strToAccount, UniValue& oLinkRequests)
{
    int nCount = 1;
    for (const CLink& link : vchLinkRequests) {
        UniValue oLink(UniValue::VOBJ);
        bool expired = false;
        int64_t expired_time = 0;
        int64_t nTime = 0;
        if (strFromAccount.empty() || strFromAccount == stringFromVch(link.RequestorFullObjectPath)) {
            if (strToAccount.empty() || strToAccount == stringFromVch(link.RecipientFullObjectPath)) {
                oLink.push_back(Pair("requestor_fqdn", stringFromVch(link.RequestorFullObjectPath)));
                oLink.push_back(Pair("recipient_fqdn", stringFromVch(link.RecipientFullObjectPath)));

                if (stringFromVch(link.SharedRequestPubKey).length() > 0) oLink.push_back(Pair("shared_request_pubkey", stringFromVch(link.SharedRequestPubKey)));
                if (stringFromVch(link.SharedAcceptPubKey).length() > 0) oLink.push_back(Pair("shared_accept_pubkey", stringFromVch(link.SharedAcceptPubKey)));

                if (stringFromVch(link.RequestorWalletAddress).length() > 0) oLink.push_back(Pair("requestor_wallet_address", stringFromVch(link.RequestorWalletAddress)));
                if (stringFromVch(link.RecipientWalletAddress).length() > 0) oLink.push_back(Pair("recipient_wallet_address", stringFromVch(link.RecipientWalletAddress)));

                oLink.push_back(Pair("requestor_link_pubkey", stringFromVch(link.RequestorPubKey)));
                oLink.push_back(Pair("txid", link.txHashRequest.GetHex())); // TODO: rename to request_txid
                if ((unsigned int)chainActive.Height() >= link.nHeightRequest-1) {
                    CBlockIndex *pindex = chainActive[link.nHeightRequest-1];
                    if (pindex) {
                        nTime = pindex->GetMedianTimePast();
                    }
                }
                oLink.push_back(Pair("time", nTime)); // TODO: rename to request_time
                expired_time = link.nExpireTimeRequest;
                if (expired_time != 0)
                {
                    if (expired_time <= (int64_t)chainActive.Tip()->GetMedianTimePast())
                    {
                        expired = true;
                    }
                }
                oLink.push_back(Pair("expires_on", expired_time)); // TODO: rename to request_expires_on
                oLink.push_back(Pair("expired", expired)); // TODO: rename to request_expired
                if (!link.txHashAccept.IsNull()) {
                    oLink.push_back(Pair("recipient_link_pubkey", stringFromVch(link.RecipientPubKey)));
                    oLink.push_back(Pair("accept_txid", link.txHashAccept.GetHex()));
                    if ((unsigned int)chainActive.Height() >= link.nHeightAccept-1) {
                        CBlockIndex *pindex = chainActive[link.nHeightAccept-1]; //changed from Request to Accept
                        if (pindex) {
                            nTime = pindex->GetMedianTimePast();
                        }
                    }
                    oLink.push_back(Pair("accept_time", nTime));
                    expired_time = link.nExpireTimeAccept; //changed from Request to Accept
                    expired = false;
                    if (expired_time != 0)
                    {
                        if (expired_time <= (int64_t)chainActive.Tip()->GetMedianTimePast())
                        {
                            expired = true;
                        }
                    }
                    oLink.push_back(Pair("accept_expires_on", expired_time));
                    oLink.push_back(Pair("accept_expired", expired));
                }
                oLink.push_back(Pair("link_message", stringFromVch(link.LinkMessage)));
                oLinkRequests.push_back(Pair("link-" + std::to_string(nCount) , oLink));
                nCount ++;
            }
        }
    }

    return true;
}

static UniValue ListPendingLinks(const JSONRPCRequest& request)
{
     if (request.fHelp || request.params.size() > 4)
        throw std::runtime_error(
            "lists pending link requests\n"
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

    if (!pLinkManager)
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_RPC_ERROR: ERRCODE: 4200 - Link manager map is null.");

    std::vector<CLink> vchPendingLinks;
    if (!pLinkManager->ListMyPendingRequests(vchPendingLinks))
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_RPC_ERROR: ERRCODE: 4201 - Error listing link requests from memory map");

    if (!pLinkManager->ListMyPendingAccepts(vchPendingLinks))
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_RPC_ERROR: ERRCODE: 4202 - Error listing link accepts from memory map");

    UniValue oLinks(UniValue::VOBJ);
    if (!BuildJsonMyLists(vchPendingLinks, strFromAccountFQDN, strToAccountFQDN, oLinks))
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_RPC_ERROR: ERRCODE: 4203 - Error creating JSON link requests.");

    int nInQueue = (int)pLinkManager->QueueSize();
    oLinks.push_back(Pair("locked_links", nInQueue));

    return oLinks;
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

    if (!pLinkManager)
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_REQ_RPC_ERROR: ERRCODE: 4200 - Link manager map is null.");

    std::vector<CLink> vchPendingLinks;
    if (!pLinkManager->ListMyPendingRequests(vchPendingLinks))
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_REQ_RPC_ERROR: ERRCODE: 4201 - Error listing link requests from memory map");

    UniValue oLinks(UniValue::VOBJ);
    if (!BuildJsonMyLists(vchPendingLinks, strFromAccountFQDN, strToAccountFQDN, oLinks))
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_REQ_RPC_ERROR: ERRCODE: 4203 - Error creating JSON link requests.");

    int nInQueue = (int)pLinkManager->QueueSize();
    oLinks.push_back(Pair("locked_links", nInQueue));

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
    
    if (!pLinkManager)
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_ACCEPT_RPC_ERROR: ERRCODE: 4200 - Link manager map is null.");

    std::vector<CLink> vchPendingLinks;
    if (!pLinkManager->ListMyPendingAccepts(vchPendingLinks))
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_ACCEPT_RPC_ERROR: ERRCODE: 4211 - Error listing link requests from memory map");

    UniValue oLinks(UniValue::VOBJ);
    if (!BuildJsonMyLists(vchPendingLinks, strFromAccountFQDN, strToAccountFQDN, oLinks))
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_ACCEPT_RPC_ERROR: ERRCODE: 4213 - Error creating JSON link requests.");

    int nInQueue = (int)pLinkManager->QueueSize();
    oLinks.push_back(Pair("locked_links", nInQueue));

    return oLinks;
}

static UniValue ListCompletedLinks(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() > 3)
        throw std::runtime_error(
            "lists completed links\n"
            "\nLink Completed Arguments:\n"
            "1. from account                  (string, optional)    BDAP from account sending the link request\n"
            "2. to account                    (string, optional)    BDAP to account receiving the link request\n"
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
            "  \"Link Accept TxID\"           (string)  Transaction ID for the link request\n"
            "  \"Accept Time\"                (int)     Transaction time\n"
            "  \"Accept Expires On\"          (int)     Link request expiration\n"
            "  \"Accept Expired\"             (boolean) Is link request expired\n"
            "  \"Link Message\"               (string) Message from requestor to recipient\n"
            "  },...n \n"
            "\nExamples:\n"
            + HelpExampleCli("link complete", "") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link complete", ""));

    std::string strFromAccountFQDN, strToAccountFQDN;
    if (request.params.size() > 1) {
        strFromAccountFQDN = request.params[1].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
        ToLowerCase(strFromAccountFQDN);
    }
    if (request.params.size() > 2) {
        strToAccountFQDN = request.params[2].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
        ToLowerCase(strToAccountFQDN);
    }

    if (!pLinkManager)
        throw std::runtime_error("BDAP_LINK_LIST_PENDING_ACCEPT_RPC_ERROR: ERRCODE: 4200 - Link manager map is null.");

    std::vector<CLink> vchLinkCompleted;
    if (!pLinkManager->ListMyCompleted(vchLinkCompleted))
        throw std::runtime_error("BDAP_LINK_COMPLETED_RPC_ERROR: ERRCODE: 4221 - Error listing link requests from memory map");

    UniValue oLinks(UniValue::VOBJ);
    if (!BuildJsonMyLists(vchLinkCompleted, strFromAccountFQDN, strToAccountFQDN, oLinks))
        throw std::runtime_error("BDAP_LINK_COMPLETED_RPC_ERROR: ERRCODE: 4222 - Error creating JSON link requests.");

    int nInQueue = (int)pLinkManager->QueueSize();
    oLinks.push_back(Pair("locked_links", nInQueue));

    return oLinks;
}
/*
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

    EnsureWalletIsUnlocked();

    std::string strRequestorFQDN = request.params[1].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strRequestorFQDN);
    CharString vchRequestorFQDN = vchFromString(strRequestorFQDN);

    std::string strRecipientFQDN = request.params[2].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strRecipientFQDN);
    CharString vchRecipientFQDN = vchFromString(strRecipientFQDN);

    if (!CheckLinkAcceptDB())
        throw std::runtime_error("BDAP_DELETE_LINK_RPC_ERROR: ERRCODE: 4230 - Can not open link request leveldb database");

    CScript scriptPubKey, scriptSend, scriptDest, scriptData;
    UniValue oLink(UniValue::VOBJ);

    // Try to find accepted link entry
    if (!pLinkManager)
            throw std::runtime_error("BDAP_DELETE_LINK_RPC_ERROR: ERRCODE: 4232 - Can not open link request in memory map");

    CLink link;
    uint256 linkID = GetLinkID(strRequestorFQDN, strRecipientFQDN);
    if (pLinkManager->FindLink(linkID, link)){
        if (link.fAcceptFromMe) {
            // create delete accept tx
            CDomainEntry entryRequestor, entryRecipient;
            if (GetDomainEntry(link.RequestorFullObjectPath, entryRequestor) && GetDomainEntry(link.RecipientFullObjectPath, entryRecipient)) {
                // Create BDAP operation script
                scriptPubKey << CScript::EncodeOP_N(OP_BDAP_DELETE) << CScript::EncodeOP_N(OP_BDAP_LINK_ACCEPT) 
                             << link.RecipientPubKey << link.SharedAcceptPubKey << OP_2DROP << OP_2DROP;
                scriptDest = GetScriptForDestination(entryRequestor.GetLinkAddress().Get());
                scriptPubKey += scriptDest;
                scriptSend = GetScriptForDestination(entryRecipient.GetLinkAddress().Get());
                if(!BuildJsonLinkInfo(link, entryRequestor, entryRecipient, oLink))
                    throw std::runtime_error("BDAP_DELETE_LINK_RPC_ERROR: ERRCODE: 4231 - " + _("Failed to build BDAP link JSON object"));
            }
        }
        else 
        {
            // create delete request tx
            CDomainEntry entryRequestor, entryRecipient;
            if (GetDomainEntry(link.RecipientFullObjectPath, entryRecipient) && GetDomainEntry(link.RequestorFullObjectPath, entryRequestor)) {
                // Create BDAP operation script
                scriptPubKey << CScript::EncodeOP_N(OP_BDAP_DELETE) << CScript::EncodeOP_N(OP_BDAP_LINK_REQUEST) 
                             << link.RequestorPubKey << link.SharedRequestPubKey << OP_2DROP << OP_2DROP;
                scriptDest = GetScriptForDestination(entryRecipient.GetLinkAddress().Get());
                scriptPubKey += scriptDest;
                scriptSend = GetScriptForDestination(entryRequestor.GetLinkAddress().Get());
                if(!BuildJsonLinkInfo(link, entryRequestor, entryRecipient, oLink))
                    throw std::runtime_error("BDAP_DELETE_LINK_RPC_ERROR: ERRCODE: 4233 - " + _("Failed to build BDAP link JSON object"));
            }
        }
        
    }
    else {
        throw std::runtime_error("BDAP_DELETE_LINK_RPC_ERROR: ERRCODE: 4234 - Can not find link request or link accept in database.");
    }

    // Send the transaction
    CWalletTx wtx;
    CAmount nOperationFee = (GetBDAPFee(scriptPubKey) * powf(3.1, 1)) + GetDataFee(scriptPubKey);
    CAmount nDataFee = 0; // No OP_RETURN data needed for deleted account transactions

    bool fUseInstantSend = false;
    SendLinkingTransaction(scriptData, scriptPubKey, scriptSend, wtx, nOperationFee, nDataFee, fUseInstantSend);

    return oLink;
}
*/
static UniValue DenyLink(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 3)
        throw std::runtime_error(
            "link deny userid-from userid-to\n"
            "Denies a link request and writes it to the DHT."
            + HelpRequiringPassphrase() +
            "\nLink Deny Arguments:\n"
            "1. recipient          (string)             Your BDAP recipient account\n"
            "2. requestor          (string)             Requesting account that will be denied.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"Requestor FQDN\"             (string)  Requestor's BDAP full path\n"
            "  \"Recipient FQDN\"             (string)  Recipient's BDAP full path\n"
            "  }\n"
            "\nExamples:\n"
            + HelpExampleCli("link deny", "superman batman") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link deny", "superman batman"));

    EnsureWalletIsUnlocked();

    if (!DHT::SessionStatus())
        throw std::runtime_error("ERRORCODE: 5500 - DHT session not started.\n");

    std::string strRecipientFQDN = request.params[1].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strRecipientFQDN);
    CharString vchRecipientFQDN = vchFromString(strRecipientFQDN);

    UniValue oLink(UniValue::VOBJ);

    CDomainEntry entry;
    if (!pDomainEntryDB->GetDomainEntryInfo(vchRecipientFQDN, entry))
        throw std::runtime_error("BDAP_DENY_LINK_RPC_ERROR: ERRCODE: 4240 - " + strRecipientFQDN + _(" can not be found.  Get BDAP info failed!\n"));

    CKeyEd25519 getKey;
    std::vector<unsigned char> vch = entry.DHTPublicKey;
    CKeyID keyID(Hash160(vch.begin(), vch.end()));
    if (pwalletMain && !pwalletMain->GetDHTKey(keyID, getKey))
        throw std::runtime_error("BDAP_DENY_LINK_RPC_ERROR: ERRCODE: 4241 - Error getting ed25519 private key for the " + strRecipientFQDN + _(" BDAP entry.\n"));

    std::string strRequestorFQDN = request.params[2].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strRequestorFQDN);
    std::string strOperationType = "denylink";

    uint16_t nTotalSlots = 32;
    int64_t iSequence = 0;
    bool fNotFound = false;
    CDataRecord record;
    if (!DHT::SubmitGetRecord(0, getKey.GetDHTPubKey(), getKey.GetDHTPrivSeed(), strOperationType, iSequence, record))
        fNotFound = true;

    std::vector<unsigned char> vchSerializedList;
    int nRecords = 0;
    if (record.GetHeader().IsNull() || fNotFound) {
        // Make new list
        CLinkDenyList denyList;
        denyList.Add(strRequestorFQDN, GetTime());
        denyList.Serialize(vchSerializedList);
        nRecords = denyList.vDenyAccounts.size();
    }
    else {
        CLinkDenyList denyList(record.RawData());
        if (denyList.Find(strRequestorFQDN))
            throw std::runtime_error("BDAP_DENY_LINK_RPC_ERROR: ERRCODE: 4244 - Link already in list.");

        denyList.Add(strRequestorFQDN, GetTime());
        denyList.Serialize(vchSerializedList);
        nRecords = denyList.vDenyAccounts.size();
    }
    oLink.push_back(Pair("list_data_size", (int)vchSerializedList.size()));
    oLink.push_back(Pair("list_records", nRecords));
    uint16_t nVersion = 1; // VGP encryption version 1
    uint64_t nExpire = DEFAULT_LINK_EXPIRE_TIME; // Does not expire.
    std::vector<std::vector<unsigned char>> vvchPubKeys;
    vvchPubKeys.push_back(getKey.GetPubKeyBytes());
    CDataRecord newRecord(strOperationType, nTotalSlots, vvchPubKeys, vchSerializedList, nVersion, (uint32_t)nExpire, DHT::DataFormat::BinaryBlob);
    if (newRecord.HasError())
        throw std::runtime_error("BDAP_DENY_LINK_RPC_ERROR: ERRCODE: 4245 - Error creating DHT data entry. " + newRecord.ErrorMessage() + _("\n"));

    if (vchSerializedList.size() > 7000)
        throw std::runtime_error("BDAP_DENY_LINK_RPC_ERROR: ERRCODE: 4246 - List is too large for one record in the DHT. " + _("\n"));

    iSequence++;
    std::string strErrorMessage;
    if (!DHT::SubmitPut(getKey.GetDHTPubKey(), getKey.GetDHTPrivKey(), iSequence, newRecord, strErrorMessage))
        throw std::runtime_error("BDAP_DENY_LINK_RPC_ERROR: ERRCODE: 4247 - Put failed. " + strErrorMessage + _("\n"));

    oLink.push_back(Pair("recipient_fqdn", strRecipientFQDN));
    oLink.push_back(Pair("requestor_fqdn", strRequestorFQDN));

    return oLink;
}

static UniValue DeniedLinkList(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 2)
        throw std::runtime_error(
            "link denied recipient\n"
            "Returns a list of denied link requests for the passed recipient account parameter."
            + HelpRequiringPassphrase() +
            "\nLink Denied Arguments:\n"
            "1. recipient          (string)             Your BDAP recipient account\n"
            "\nResult:\n"
            "{(json objects)\n"
            "  \"Requestor FQDN\"             (string)  Requestor's BDAP full path\n"
            "  }\n"
            "\nExamples:\n"
            + HelpExampleCli("link denied", "superman") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link deny", "superman"));

    EnsureWalletIsUnlocked();

    if (!DHT::SessionStatus())
        throw std::runtime_error("ERRORCODE: 5500 - DHT session not started.\n");

    std::string strRecipientFQDN = request.params[1].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strRecipientFQDN);
    CharString vchRecipientFQDN = vchFromString(strRecipientFQDN);

    UniValue oLink(UniValue::VOBJ);

    CDomainEntry entry;
    if (!pDomainEntryDB->GetDomainEntryInfo(vchRecipientFQDN, entry))
        throw std::runtime_error("BDAP_DENY_LINK_RPC_ERROR: ERRCODE: 4240 - " + strRecipientFQDN + _(" can not be found.  Get BDAP info failed!\n"));

    CKeyEd25519 getKey;
    std::vector<unsigned char> vch = entry.DHTPublicKey;
    CKeyID keyID(Hash160(vch.begin(), vch.end()));
    if (pwalletMain && !pwalletMain->GetDHTKey(keyID, getKey))
        throw std::runtime_error("BDAP_DENY_LINK_RPC_ERROR: ERRCODE: 4241 - Error getting ed25519 private key for the " + strRecipientFQDN + _(" BDAP entry.\n"));

    std::string strOperationType = "denylink";
    int64_t iSequence = 0;
    CDataRecord record;
    if (!DHT::SubmitGetRecord(0, getKey.GetDHTPubKey(), getKey.GetDHTPrivSeed(), strOperationType, iSequence, record))
        throw std::runtime_error(strprintf("%s: ERRCODE: 5604 - Failed to get record\n", __func__));

    UniValue oDeniedLink(UniValue::VOBJ);
    CLinkDenyList denyList(record.RawData());
    int timestamp = 0;
    int nRecord = 1;
    for (const std::string& strAccountFQDN : denyList.vDenyAccounts) {
        UniValue oDenied(UniValue::VOBJ);
        std::string strKey = "item_" + std::to_string(nRecord);
        oDenied.push_back(Pair("requestor_fqdn", strAccountFQDN));
        timestamp = (int)denyList.vTimestamps[nRecord-1];
        oDenied.push_back(Pair("timestamp_epoch", timestamp));
        oDenied.push_back(Pair("timestamp", DateTimeStrFormat("%Y-%m-%dT%H:%M:%SZ", timestamp)));
        oDeniedLink.push_back(Pair(strKey, oDenied));
        nRecord++;
    }
    timestamp = (int)record.GetHeader().nTimeStamp;
    oLink.push_back(Pair("list_updated_epoch", timestamp));
    oLink.push_back(Pair("list_updated", DateTimeStrFormat("%Y-%m-%dT%H:%M:%SZ", timestamp)));
    oLink.push_back(Pair("denied_list", oDeniedLink));

    return oLink;
}

static UniValue SendMessage(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() < 5 || request.params.size() > 6)
        throw std::runtime_error(
            "link sendmessage \"account\" \"recipient\" \"type\" \"message\" \"keep_last\"\n"
            "Sends a realtime message from the account to the recipient. A link must be established before sending a secure message."
            + HelpRequiringPassphrase() +
            "\nLink Send Message Arguments:\n"
            "1. account          (string)             Your BDAP sending account\n"
            "2. recipient        (string)             Your link's recipient BDAP account\n"
            "3. type             (string)             Message type\n"
            "4. message          (string)             String message value\n"
            "5. keep_last        (bool, optional)     Only store the last message for this type. Default is false.\n"
            "\nResult:\n"
            "{(json objects)\n"
            "  \"Requestor FQDN\"             (string)  Requestor's BDAP full path\n"
            "}\n"
            "\nExamples:\n"
            + HelpExampleCli("link sendmessage", "superman batman status \"I am online\"") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link sendmessage", "superman batman status \"I am online\""));

    if (!pwalletMain)
        throw std::runtime_error(strprintf("%s -- wallet pointer is null.\n", __func__));

    EnsureWalletIsUnlocked();

    std::string strSenderFQDN = request.params[1].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strSenderFQDN);
    // Make sure this wallet owns the sending address
    std::vector<unsigned char> vchSenderFQDN = vchFromString(strSenderFQDN);
    CDomainEntry entry;
    if (!GetDomainEntry(vchSenderFQDN, entry))
        throw std::runtime_error(strprintf("%s -- sender account (%s) not found.\n", __func__, strSenderFQDN));

    CKeyEd25519 getKey;
    CKeyID senderKeyID(Hash160(entry.DHTPublicKey.begin(), entry.DHTPublicKey.end()));
    if (!pwalletMain->GetDHTKey(senderKeyID, getKey))
        throw std::runtime_error(strprintf("%s -- Error getting ed25519 private key for the sender (%s) BDAP account.\n", __func__, strSenderFQDN));

    std::string strRecipientFQDN = request.params[2].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strRecipientFQDN);

    std::vector<unsigned char> vchMessageType = vchFromValue(request.params[3]);
    std::vector<unsigned char> vchMessage = vchFromValue(request.params[4]);

    bool fKeepLast = false;
    if (request.params.size() > 5 && request.params[5].get_str() == "1") {
        fKeepLast = true;
    }

    UniValue oLink(UniValue::VOBJ);
    // get third shared key, derive subjectID and messageID.
    CKeyEd25519 key;
    std::string strErrorMessage = "";
    if (!GetSecretSharedKey(strSenderFQDN, strRecipientFQDN, key, strErrorMessage))
        throw std::runtime_error(strprintf("%s -- GetSecretSharedKey failed: %s\n", __func__, strErrorMessage));

    std::vector<std::vector<unsigned char>> vvchPubKeys;
    vvchPubKeys.push_back(key.GetPubKeyBytes());
    uint256 subjectID = GetSubjectIDFromKey(key);
    int64_t timestamp = GetAdjustedTime();
    int64_t stoptime = timestamp + 60; // stop relaying after 1 minute.
    uint256 messageID = GetMessageID(key, timestamp);
    CPubKey newPubKey;
    std::vector<unsigned char> newEdKey;
    if (!pwalletMain->GetKeysFromPool(newPubKey, newEdKey, false))
        throw JSONRPCError(RPC_WALLET_KEYPOOL_RAN_OUT, "Error: Keypool ran out, please call keypoolrefill first");

    CKeyID keyID = newPubKey.GetID();
    CDynamicAddress walletAddress = CDynamicAddress(keyID);
    std::vector<unsigned char> vchWalletPubKey(newPubKey.begin(), newPubKey.end());

    CUnsignedVGPMessage unsignedMessage(subjectID, messageID, vchWalletPubKey, timestamp, stoptime);
    if (!unsignedMessage.EncryptMessage(vchMessageType, vchMessage, vchFromString(strSenderFQDN), vvchPubKeys, fKeepLast, strErrorMessage))
    {
        throw std::runtime_error(strprintf("%s -- EncryptMessage failed: %s\n", __func__, strErrorMessage));
    }
    CVGPMessage vpgMessage(unsignedMessage);
    if (vpgMessage.vchMsg.size() > MAX_MESSAGE_SIZE)
    {
        throw std::runtime_error(strprintf("%s --Message size (%d) too large to send.\n", __func__, vpgMessage.vchMsg.size()));
    }
    CKey walletKey;
    if (!pwalletMain->GetKey(keyID, walletKey))
    {
        throw std::runtime_error(strprintf("%s -- Failed to get wallet private key\n", __func__));
    }
    oLink.push_back(Pair("timestamp_epoch", timestamp));
    oLink.push_back(Pair("shared_pubkey", key.GetPubKeyString()));
    oLink.push_back(Pair("subject_id", unsignedMessage.SubjectID.ToString()));
    oLink.push_back(Pair("message_id", unsignedMessage.MessageID.ToString()));
    vpgMessage.MineMessage();
    oLink.push_back(Pair("message_hash", vpgMessage.GetHash().ToString()));
    oLink.push_back(Pair("message_size", (int)vpgMessage.vchMsg.size()));
    vpgMessage.Sign(walletKey);
    oLink.push_back(Pair("signature_size", (int)vpgMessage.vchSig.size()));
    if (vpgMessage.CheckSignature(vchWalletPubKey)) {
        oLink.push_back(Pair("check_signature", "valid"));
        vpgMessage.RelayMessage(*g_connman);
    }
    else {
        oLink.push_back(Pair("check_signature", "invalid"));
        oLink.push_back(Pair("error_message", "failed to relay message"));
    }
    oLink.push_back(Pair("keep_last", fKeepLast ? "True" : "False"));
    return oLink;
}

static UniValue GetAccountMessages(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() > 4 || request.params.size() < 3)
        throw std::runtime_error(
            "link getaccountmessages \"account\" \"sender\" \"type\"\n"
            "Gets realtime messages for the account from the specified sender and message type"
            + HelpRequiringPassphrase() +
            "\nLink Send Message Arguments:\n"
            "1. account          (string)             Your BDAP recipient account\n"
            "2. sender           (string)             The account sending the message\n"
            "3. type             (string, optional)   Filter by message type\n"
            "\nResult:\n"
            "{(json objects)\n"
            "  \"sender_fqdn\"                (string)  Sender's BDAP full path\n"
            "  \"message\"                    (string)  Message from sender\n"
            "  \"timestamp_epoch\"            (int)     Epoch time message was created\n"
            "}\n"
            "\nExamples:\n"
            + HelpExampleCli("link getaccountmessages", "superman batman status") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link getaccountmessages", "superman batman status"));

    if (!pwalletMain)
        throw std::runtime_error(strprintf("%s -- wallet pointer is null.\n", __func__));

    EnsureWalletIsUnlocked();

    if (!pLinkManager)
        throw std::runtime_error(strprintf("%s -- link manager pointer is null.\n", __func__));

    // Get BDAP recipient account
    std::string strRecipient = request.params[1].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strRecipient);
    std::string strSender = request.params[2].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strSender);
    std::vector<unsigned char> vchSenderFQDN = vchFromString(strSender);

    uint256 linkID = GetLinkID(strRecipient, strSender);
    CLink link;
    if (!pLinkManager->FindLink(linkID, link))
        throw std::runtime_error(strprintf("%s -- link not found.\n", __func__));

    std::string strErrorMessage = "";
    if (!GetMessageInfo(link, strErrorMessage))
        throw std::runtime_error(strprintf("%s -- Could not get message info %s.\n", __func__, strErrorMessage));

    // Get message type parameter
    std::vector<unsigned char> vchMessageType;
    if (request.params.size() > 3)
        vchMessageType = vchFromValue(request.params[3]);

    bool fKeepLast = false;
    std::vector<CVGPMessage> vMessages;
    GetMyLinkMessagesBySubjectAndSender(link.SubjectID, vchSenderFQDN, vchMessageType, vMessages, fKeepLast);
    std::sort(vMessages.begin(), vMessages.end()); //sort entries by TimeStamp
    if (fKeepLast)
        KeepLastTypeBySender(vMessages);

    UniValue oMessages(UniValue::VOBJ);
    if (vMessages.size() > 0)
    {
        size_t nCounter = 1;
        for (CVGPMessage& messageWrapper : vMessages)
        {
            UniValue oMessage(UniValue::VOBJ);
            CUnsignedVGPMessage unsignedMessage(messageWrapper.vchMsg);
            oMessage.push_back(Pair("sender_fqdn", stringFromVch(unsignedMessage.SenderFQDN())));
            oMessage.push_back(Pair("type", stringFromVch(unsignedMessage.Type())));
            oMessage.push_back(Pair("message", stringFromVch(unsignedMessage.Value())));
            oMessage.push_back(Pair("message_id", unsignedMessage.MessageID.ToString()));
            oMessage.push_back(Pair("message_size", (int)messageWrapper.vchMsg.size()));
            oMessage.push_back(Pair("timestamp_epoch", unsignedMessage.nTimeStamp));
            oMessage.push_back(Pair("record_num", (int)nCounter));
            oMessages.push_back(Pair(messageWrapper.GetHash().ToString(), oMessage));
            nCounter++;
        }
    }
    return oMessages;
}

static UniValue GetMessages(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() > 3 || request.params.size() < 2)
        throw std::runtime_error(
            "link getmessages \"account\" \"type\"\n"
            "Gets realtime messages from the account for the specified message type"
            + HelpRequiringPassphrase() +
            "\nLink Send Message Arguments:\n"
            "1. account          (string)             Your BDAP recipient account\n"
            "2. type             (string, optional)   Message type\n"
            "\nResult:\n"
            "{(json objects)\n"
            "  \"sender_fqdn\"                (string)  Sender's BDAP full path\n"
            "  \"message\"                    (string)  Message from sender\n"
            "  \"timestamp_epoch\"            (int)     Epoch time message was created\n"
            "}\n"
            "\nExamples:\n"
            + HelpExampleCli("link getmessages", "superman status") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link getmessages", "superman status"));

    if (!pwalletMain)
        throw std::runtime_error(strprintf("%s -- wallet pointer is null.\n", __func__));

    EnsureWalletIsUnlocked();
    // Get BDAP recipient account
    std::string strRecipient = request.params[1].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strRecipient);
    std::vector<unsigned char> vchRecipientFQDN = vchFromString(strRecipient);
    // Get message type parameter
    std::vector<unsigned char> vchMessageType;
    if (request.params.size() > 2)
        vchMessageType = vchFromValue(request.params[2]);

    bool fKeepLast = false;
    std::vector<CVGPMessage> vMessages;
    GetMyLinkMessagesByType(vchMessageType, vchRecipientFQDN, vMessages, fKeepLast);
    std::sort(vMessages.begin(), vMessages.end()); //sort entries by TimeStamp
    if (fKeepLast)
        KeepLastTypeBySender(vMessages);

    UniValue oMessages(UniValue::VOBJ);
    if (vMessages.size() > 0)
    {
        size_t nCounter = 1;
        for (CVGPMessage& messageWrapper : vMessages)
        {
            UniValue oMessage(UniValue::VOBJ);
            CUnsignedVGPMessage unsignedMessage(messageWrapper.vchMsg);
            oMessage.push_back(Pair("sender_fqdn", stringFromVch(unsignedMessage.SenderFQDN())));
            oMessage.push_back(Pair("type", stringFromVch(unsignedMessage.Type())));
            oMessage.push_back(Pair("message", stringFromVch(unsignedMessage.Value())));
            oMessage.push_back(Pair("message_id", unsignedMessage.MessageID.ToString()));
            oMessage.push_back(Pair("message_size", (int)messageWrapper.vchMsg.size()));
            oMessage.push_back(Pair("timestamp_epoch", unsignedMessage.nTimeStamp));
            oMessage.push_back(Pair("record_num", (int)nCounter));
            oMessages.push_back(Pair(messageWrapper.GetHash().ToString(), oMessage));
            nCounter++;
        }
    }
    return oMessages;
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
            "link \"command\"...\n"
            + HelpRequiringPassphrase() +
            "\nLink commands are request, accept, pending, complete, deny, denied, getaccountmessages, getmessages, and sendmessage\n"
            "\nExamples:\n"
            + HelpExampleCli("link accept", "superman batman") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("link accept", "superman batman"));
    }
    if (strCommand == "request" || strCommand == "accept" || strCommand == "delete") {
        if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
            throw std::runtime_error("BDAP_LINK_RPC_ERROR: ERRCODE: 3000 - " + _("Can not create BDAP link transactions until spork is active."));
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
    /*
    else if (strCommand == "delete") {
        return DeleteLink(request);
    }
    */
    else if (strCommand == "deny") {
        return DenyLink(request);
    }
    else if (strCommand == "denied") {
        return DeniedLinkList(request);
    }
    else if (strCommand == "getaccountmessages") {
        return GetAccountMessages(request);
    }
    else if (strCommand == "getmessages") {
        return GetMessages(request);
    }
    else if (strCommand == "pending") {
        if (request.params.size() == 1) {
            return ListPendingLinks(request);
        }
        else if (request.params.size() >= 2) {
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
    else if (strCommand == "sendmessage") {
        return SendMessage(request);
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