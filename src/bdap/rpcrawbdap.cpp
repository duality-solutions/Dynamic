// Copyright (c) 2019 Duality Blockchain Solutions Developers 
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/domainentry.h"
#include "bdap/domainentrydb.h"
#include "bdap/utils.h"
#include "core_io.h" // for EncodeHexTx
#include "dht/ed25519.h"
#include "rpcprotocol.h"
#include "rpcserver.h"
#include "policy/policy.h"
#include "primitives/transaction.h"
#include "utilstrencodings.h"
#ifdef ENABLE_WALLET
#include "wallet/wallet.h"
#endif
#include "validation.h"

#include <stdint.h>

#include <univalue.h>

UniValue createrawbdapaccount(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() < 2 || request.params.size() > 4)
        throw std::runtime_error(
            "createrawbdapaccount \"account id\" \"common name\" \"registration days\" \"object type\"\n"
            "\nArguments:\n"
            "1. account id         (string)             BDAP account id requesting the link\n"
            "2. common name        (string)             Free text comman name for BDAP account with a max length of 95 characters\n"
            "3. registration days  (int, optional)      Number of registration days for the new account.  Defaults to 2 years.\n"
            "4. object type        (int, optional)      Type of BDAP account to create. 1 = user and 2 = group.  Default to 1 for user.\n"
            "\nCreates a raw hex encoded BDAP transaction without inputs and with new outputs from this wallet.\n"
            "\nCall fundrawtransaction to pay for the BDAP account, then signrawtransaction and last sendrawtransaction\n"
            "\nResult:\n"
            "\"raw transaction\"   (string) hex string of the raw BDAP transaction\n"
            "\nExamples\n" +
           HelpExampleCli("createrawbdapaccount", "jack \"Black, Jack\"") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("createrawbdapaccount", "jack \"Black, Jack\""));

    EnsureWalletIsUnlocked();

    // Format object and domain names to lower case.
    std::string strObjectID = request.params[0].get_str();
    ToLowerCase(strObjectID);
    CharString vchObjectID = vchFromString(strObjectID);
    CharString vchCommonName = vchFromValue(request.params[1]);

    int64_t nDays = DEFAULT_REGISTRATION_DAYS;  //default to 2 years.
    if (request.params.size() >= 3) {
        if (!ParseInt64(request.params[2].get_str(), &nDays))
            throw std::runtime_error("BDAP_CREATE_RAW_TX_RPC_ERROR: ERRCODE: 4500 - " + _("Error converting registration days to int"));
    }
    int64_t nSeconds = nDays * SECONDS_PER_DAY;
    BDAP::ObjectType bdapType = ObjectType::BDAP_USER;
    if (request.params.size() >= 4) {
        int32_t nObjectType;
        if (!ParseInt32(request.params[3].get_str(), &nObjectType))
            throw std::runtime_error("BDAP_CREATE_RAW_TX_RPC_ERROR: ERRCODE: 4501 - " + _("Error converting BDAP object type to int"));

        if (nObjectType == 1) {
            bdapType = ObjectType::BDAP_USER;
        }
        else if (nObjectType == 2) {
            bdapType = ObjectType::BDAP_GROUP;
        }
        else
            throw std::runtime_error("BDAP_CREATE_RAW_TX_RPC_ERROR: ERRCODE: 4502 - " + _("Unsupported BDAP type."));
    }

    CDomainEntry txDomainEntry;
    txDomainEntry.OID = vchDefaultOIDPrefix;
    txDomainEntry.DomainComponent = vchDefaultDomainName;
    txDomainEntry.OrganizationalUnit = vchDefaultPublicOU;
    txDomainEntry.CommonName = vchCommonName;
    txDomainEntry.OrganizationName = vchDefaultOrganizationName;
    txDomainEntry.ObjectID = vchObjectID;
    txDomainEntry.fPublicObject = 1; // make entry public
    txDomainEntry.nObjectType = GetObjectTypeInt(bdapType);
    // Add an extra 8 hours or 28,800 seconds to expire time.
    txDomainEntry.nExpireTime = chainActive.Tip()->GetMedianTimePast() + nSeconds + 28800;
    
    // Check if name already exists
    if (GetDomainEntry(txDomainEntry.vchFullObjectPath(), txDomainEntry))
        throw std::runtime_error("BDAP_CREATE_RAW_TX_RPC_ERROR: ERRCODE: 4503 - " + txDomainEntry.GetFullObjectPath() + _(" entry already exists.  Can not add duplicate."));

    // TODO: Add ability to pass in the wallet address
    CKey privWalletKey;
    privWalletKey.MakeNewKey(true);
    CPubKey pubWalletKey = privWalletKey.GetPubKey();
    CKeyID keyWalletID = pubWalletKey.GetID();
    CDynamicAddress walletAddress = CDynamicAddress(keyWalletID);

    if (pwalletMain && !pwalletMain->AddKeyPubKey(privWalletKey, pubWalletKey))
        throw std::runtime_error("BDAP_CREATE_RAW_TX_RPC_ERROR: ERRCODE: 4504 - " + _("Error adding receiving address key wo wallet for BDAP"));

    pwalletMain->SetAddressBook(keyWalletID, strObjectID, "bdap-wallet");
    
    CharString vchWalletAddress = vchFromString(walletAddress.ToString());
    txDomainEntry.WalletAddress = vchWalletAddress;

    // TODO: Add ability to pass in the DHT public key
    CKeyEd25519 privDHTKey;
    CharString vchDHTPubKey = privDHTKey.GetPubKey();
    
    if (pwalletMain && !pwalletMain->AddDHTKey(privDHTKey, vchDHTPubKey))
        throw std::runtime_error("BDAP_CREATE_RAW_TX_RPC_ERROR: ERRCODE: 4505 - " + _("Error adding ed25519 key to wallet for BDAP"));

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
        throw std::runtime_error("BDAP_CREATE_RAW_TX_RPC_ERROR: ERRCODE: 4506 - " + _("Error adding receiving address key wo wallet for BDAP"));

    pwalletMain->SetAddressBook(keyLinkID, strObjectID, "bdap-link");
    
    CharString vchLinkAddress = vchFromString(linkAddress.ToString());
    txDomainEntry.LinkAddress = vchLinkAddress;

    CMutableTransaction rawTx;
    rawTx.nVersion = BDAP_TX_VERSION;
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

    CTxOut outOP(DEFAULT_MIN_RELAY_TX_FEE, scriptPubKey);
    rawTx.vout.push_back(outOP);
    CTxOut outData(0, scriptData);
    rawTx.vout.push_back(outData);

    return EncodeHexTx(rawTx);
}

UniValue ConvertParameterValues(const std::vector<std::string>& strParams)
{
    UniValue params(UniValue::VARR);

    for (unsigned int idx = 0; idx < strParams.size(); idx++) {
        const std::string& strVal = strParams[idx];
        // insert string value directly
        params.push_back(strVal);
    }

    return params;
}

UniValue sendandpayrawbdapaccount(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "sendandpayrawbdapaccount \"hexstring\"\n"
            "\nArguments:\n"
            "1. hexstring        (string)             The hex string of the raw BDAP transaction\n"
            "\nPays for BDAP account by adding utxos, signs the inputs, and broadcasts the resulting transaction.\n"
            "\nCall createrawbdapaccount to get the hex encoded BDAP transaction string\n"
            "\nResult:\n"
            "\"transaction id\"   (string) \n"
            "\nExamples\n" +
           HelpExampleCli("sendandpayrawbdapaccount", "<hexstring>") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("sendandpayrawbdapaccount", "<hexstring>"));

    CMutableTransaction mtx;
    std::string strHexIn = request.params[0].get_str();
    if (!DecodeHexTx(mtx, strHexIn))
        throw JSONRPCError(RPC_DESERIALIZATION_ERROR, "TX decode failed");

    EnsureWalletIsUnlocked();

    // Funded BDAP transaction with utxos
    JSONRPCRequest jreqFund;
    jreqFund.strMethod = "fundrawtransaction";
    std::vector<std::string> vchFundParams;
    vchFundParams.push_back(strHexIn);
    jreqFund.params = ConvertParameterValues(vchFundParams);
    UniValue resultFund = tableRPC.execute(jreqFund);
    if (resultFund.isNull())
        throw std::runtime_error("BDAP_SEND_RAW_TX_RPC_ERROR: ERRCODE: 4510 - " + _("Error funding raw BDAP transaction."));
    UniValue oHexFund = resultFund.get_obj();
    std::string strHexFund =  oHexFund[0].get_str();

    // Sign funded BDAP transaction
    JSONRPCRequest jreqSign;
    jreqSign.strMethod = "signrawtransaction";
    std::vector<std::string> vchSignParams;
    vchSignParams.push_back(strHexFund);
    jreqSign.params = ConvertParameterValues(vchSignParams);
    UniValue resultSign = tableRPC.execute(jreqSign);
    if (resultSign.isNull())
        throw std::runtime_error("BDAP_SEND_RAW_TX_RPC_ERROR: ERRCODE: 4510 - " + _("Error signing funded raw BDAP transaction."));
    UniValue oHexSign = resultSign.get_obj();
    std::string strHexSign =  oHexSign[0].get_str();

    // Send funded and signed BDAP transaction
    JSONRPCRequest jreqSend;
    jreqSend.strMethod = "sendrawtransaction";
    std::vector<std::string> vchSendParams;
    vchSendParams.push_back(strHexSign);
    jreqSend.params = ConvertParameterValues(vchSendParams);
    UniValue resultSend = tableRPC.execute(jreqSend);
    if (resultSend.isNull())
        throw std::runtime_error("BDAP_SEND_RAW_TX_RPC_ERROR: ERRCODE: 4510 - " + _("Error sending raw funded & signed BDAP transaction."));
    std::string strTxId =  resultSend.get_str();

    return strTxId;
}

static const CRPCCommand commands[] =
{ //  category              name                          actor (function)           okSafe argNames
  //  --------------------- ----------------------------- -------------------------- ------ --------------------
#ifdef ENABLE_WALLET
    /* BDAP */
    { "bdap",               "createrawbdapaccount",       &createrawbdapaccount,      true, {"account id", "common name", "registration days", "object type"} },
    { "bdap",               "sendandpayrawbdapaccount",   &sendandpayrawbdapaccount,  true, {"hexstring"} },
#endif //ENABLE_WALLET
};

void RegisterRawBDAPAccountRPCCommands(CRPCTable &t)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        t.appendCommand(commands[vcidx].name, &commands[vcidx]);
}
