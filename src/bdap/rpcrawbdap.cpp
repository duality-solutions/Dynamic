// Copyright (c) 2019 Duality Blockchain Solutions Developers 
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/domainentry.h"
#include "bdap/domainentrydb.h"
#include "bdap/fees.h"
#include "bdap/stealth.h"
#include "bdap/utils.h"
#include "core_io.h" // for EncodeHexTx
#include "rpcprotocol.h"
#include "rpcserver.h"
#include "policy/policy.h"
#include "primitives/transaction.h"
#include "utilmoneystr.h"
#include "utilstrencodings.h"
#include "dynode-sync.h"
#include "spork.h"
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
            "createrawbdapaccount \"account id\" \"common name\" \"registration months\" \"object type\"\n"
            "\nArguments:\n"
            "1. account id           (string)             BDAP account id requesting the link\n"
            "2. common name          (string)             Free text comman name for BDAP account with a max length of 95 characters\n"
            "3. registration months  (int, optional)      Number of registration months for the new account.  Defaults to 2 years.\n"
            "4. object type          (int, optional)      Type of BDAP account to create. 1 = user and 2 = group.  Default to 1 for user.\n"
            "\nCreates a raw hex encoded BDAP transaction without inputs and with new outputs from this wallet.\n"
            "\nCall fundrawtransaction to pay for the BDAP account, then signrawtransaction and last sendrawtransaction\n"
            "\nResult:\n"
            "\"raw transaction\"   (string) hex string of the raw BDAP transaction\n"
            "\nExamples\n" +
           HelpExampleCli("createrawbdapaccount", "jack \"Black, Jack\"") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("createrawbdapaccount", "jack \"Black, Jack\""));

    EnsureWalletIsUnlocked();

    if (!dynodeSync.IsBlockchainSynced()) {
        throw std::runtime_error("Error: Cannot create BDAP Objects while wallet is not synced.");
    }

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3000 - " + _("Can not create BDAP transactions until spork is active."));

    // Format object and domain names to lower case.
    std::string strObjectID = request.params[0].get_str();
    ToLowerCase(strObjectID);
    CharString vchObjectID = vchFromString(strObjectID);
    CharString vchCommonName = vchFromValue(request.params[1]);

    int32_t nMonths = DEFAULT_REGISTRATION_MONTHS;  // default to 1 year or 12 months
    std::string strMonths = std::to_string(nMonths) + "Month";
    std::vector<unsigned char> vchMonths = vchFromString(strMonths);
    if (request.params.size() >= 3) {
        if (!ParseInt32(request.params[2].get_str(), &nMonths))
            throw std::runtime_error("BDAP_CREATE_RAW_TX_RPC_ERROR: ERRCODE: 4500 - " + _("Error converting registration days to int"));
    }
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
    txDomainEntry.RootOID = vchDefaultOIDPrefix;
    txDomainEntry.DomainComponent = vchDefaultDomainName;
    txDomainEntry.OrganizationalUnit = vchDefaultPublicOU;
    txDomainEntry.CommonName = vchCommonName;
    txDomainEntry.OrganizationName = vchDefaultOrganizationName;
    txDomainEntry.ObjectID = vchObjectID;
    txDomainEntry.fPublicObject = 1; // make entry public
    txDomainEntry.nObjectType = GetObjectTypeInt(bdapType);
    // Add an extra 8 hours or 28,800 seconds to expire time.
    txDomainEntry.nExpireTime = AddMonthsToCurrentEpoch((short)nMonths);

    // Check if name already exists
    if (GetDomainEntry(txDomainEntry.vchFullObjectPath(), txDomainEntry))
        throw std::runtime_error("BDAP_CREATE_RAW_TX_RPC_ERROR: ERRCODE: 4503 - " + txDomainEntry.GetFullObjectPath() + _(" entry already exists.  Can not add duplicate."));

    // TODO: Add ability to pass in the wallet address
    std::vector<unsigned char> vchDHTPubKey;
    CPubKey pubWalletKey;
    CStealthAddress sxAddr;
    if (!pwalletMain->GetKeysFromPool(pubWalletKey, vchDHTPubKey, sxAddr, true))
        throw std::runtime_error("Error: Keypool ran out, please call keypoolrefill first");
    CKeyID keyWalletID = pubWalletKey.GetID();
    CDynamicAddress walletAddress = CDynamicAddress(keyWalletID);

    pwalletMain->SetAddressBook(keyWalletID, strObjectID, "bdap-wallet");
    
    CharString vchWalletAddress = vchFromString(walletAddress.ToString());
    txDomainEntry.WalletAddress = vchWalletAddress;

    CKeyID vchDHTPubKeyID = GetIdFromCharVector(vchDHTPubKey);
    txDomainEntry.DHTPublicKey = vchDHTPubKey;
    pwalletMain->SetAddressBook(vchDHTPubKeyID, strObjectID, "bdap-dht-key");

    //pwalletMain->SetAddressBook(keyLinkID, strObjectID, "bdap-link");
    txDomainEntry.LinkAddress = vchFromString(sxAddr.ToString());

    CMutableTransaction rawTx;
    rawTx.nVersion = BDAP_TX_VERSION;
    CharString data;
    txDomainEntry.Serialize(data);

    // Get BDAP fees
    CAmount monthlyFee, oneTimeFee, depositFee;
    if (!GetBDAPFees(OP_BDAP_NEW, OP_BDAP_ACCOUNT_ENTRY, bdapType, nMonths, monthlyFee, oneTimeFee, depositFee))
        throw JSONRPCError(RPC_BDAP_FEE_UNKNOWN, strprintf("Error calculating BDAP user account fees."));
    LogPrintf("%s -- monthlyFee %d, oneTimeFee %d, depositFee %d\n", __func__, 
        FormatMoney(monthlyFee), FormatMoney(oneTimeFee), FormatMoney(depositFee));

    // Create BDAP operation script
    CScript scriptPubKey;
    std::vector<unsigned char> vchFullObjectPath = txDomainEntry.vchFullObjectPath();
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_NEW) << CScript::EncodeOP_N(OP_BDAP_ACCOUNT_ENTRY) 
                 << vchFullObjectPath << txDomainEntry.DHTPublicKey << vchMonths << OP_2DROP << OP_2DROP << OP_DROP;

    CScript scriptDestination;
    scriptDestination = GetScriptForDestination(walletAddress.Get());
    scriptPubKey += scriptDestination;

    // Create BDAP OP_RETURN script
    CScript scriptData;
    scriptData << OP_RETURN << data;

    // Create script to fund link transaction for this account
    CScript scriptStealth;
    std::vector<uint8_t> vStealthData;
    std::string sError;
    if (0 != PrepareStealthOutput(sxAddr, scriptStealth, vStealthData, sError)) {
        LogPrintf("%s -- PrepareStealthOutput failed. Error = %s\n", __func__, sError);
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, strprintf("Invalid stealth destination address %s", sxAddr.ToString()));
    }

    CScript stealthScript;
    stealthScript << OP_RETURN << vStealthData;

    CTxDestination newStealthDest;
    if (!ExtractDestination(scriptStealth, newStealthDest))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, strprintf("Unable to get destination address using stealth address %s", sxAddr.ToString()));

    CDynamicAddress addressStealth(newStealthDest);
    CScript stealtDestination = GetScriptForDestination(addressStealth.Get());

    // Add the Stealth OP return data
    CTxOut outStealthData(0, stealthScript);
    rawTx.vout.push_back(outStealthData);
    // Add the BDAP data output
    CTxOut outData(monthlyFee, scriptData);
    rawTx.vout.push_back(outData);
    // Add the BDAP operation output
    CTxOut outOP(depositFee, scriptPubKey);
    rawTx.vout.push_back(outOP);

    // Get fees for 10 link request transactions
    if (!GetBDAPFees(OP_BDAP_NEW, OP_BDAP_LINK_REQUEST, BDAP::ObjectType::BDAP_LINK_REQUEST, 0, monthlyFee, oneTimeFee, depositFee))
        throw JSONRPCError(RPC_BDAP_FEE_UNKNOWN, strprintf("Error calculating BDAP link fees."));

    CAmount nLinkRequestAmount((oneTimeFee + depositFee + monthlyFee) * 10); // enough for 10 link requests

    // Get fees for 10 link accept transactions
    if (!GetBDAPFees(OP_BDAP_NEW, OP_BDAP_LINK_ACCEPT, BDAP::ObjectType::BDAP_LINK_ACCEPT, 0, monthlyFee, oneTimeFee, depositFee))
        throw JSONRPCError(RPC_BDAP_FEE_UNKNOWN, strprintf("Error calculating BDAP link fees."));

    CAmount nLinkAcceptAmount((oneTimeFee + depositFee + monthlyFee) * 10); // enough for 10 link accepts
    // Get total amount need for 10 link request and 10 link accept transactions
    CAmount nTotalLinkAmount = (nLinkRequestAmount + nLinkAcceptAmount);
    LogPrintf("%s -- nLinkRequestAmount %d,  nLinkAcceptAmount %d, Total %d\n", __func__, 
        FormatMoney(nLinkRequestAmount), FormatMoney(nLinkAcceptAmount), FormatMoney(nTotalLinkAmount));

    // Create BDAP credits operation script
    std::vector<unsigned char> vchMoveSource = vchFromString(std::string("DYN"));
    std::vector<unsigned char> vchMoveDestination = vchFromString(std::string("BDAP"));
    CScript scriptBdapCredits;
    scriptBdapCredits << CScript::EncodeOP_N(OP_BDAP_MOVE) << CScript::EncodeOP_N(OP_BDAP_ASSET) 
                        << vchMoveSource << vchMoveDestination << OP_2DROP << OP_2DROP;
 
    // Add stealth link destination address to credits
    scriptBdapCredits += stealtDestination;

    // Add the BDAP link with credit funds output
    CTxOut outLinkFunds(nTotalLinkAmount, scriptBdapCredits);
    rawTx.vout.push_back(outLinkFunds);

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

    if (!dynodeSync.IsBlockchainSynced()) {
        throw std::runtime_error("Error: Cannot create BDAP Objects while wallet is not synced.");
    }

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_ADD_PUBLIC_ENTRY_RPC_ERROR: ERRCODE: 3000 - " + _("Can not create BDAP transactions until spork is active."));

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
    { "bdap",               "createrawbdapaccount",       &createrawbdapaccount,      true, {"account id", "common name", "registration months", "object type"} },
    { "bdap",               "sendandpayrawbdapaccount",   &sendandpayrawbdapaccount,  true, {"hexstring"} },
#endif //ENABLE_WALLET
};

void RegisterRawBDAPAccountRPCCommands(CRPCTable &t)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        t.appendCommand(commands[vcidx].name, &commands[vcidx]);
}
