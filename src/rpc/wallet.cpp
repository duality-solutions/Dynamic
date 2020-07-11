// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "wallet.h"

#include "amount.h"
#include "base58.h"
#include "chain.h"
#include "consensus/consensus.h"
#include "consensus/validation.h"
#include "core_io.h"
#include "httpserver.h" // urlDecode
#include "init.h"
#include "instantsend.h"
#include "keepass.h"
#include "net.h"
#include "netbase.h"
#include "policy/rbf.h"
#include "privatesend-client.h"
#include "rpc/mining.h"
#include "rpc/server.h"
#include "timedata.h"
#include "util.h"
#include "utilmoneystr.h"
#include "validation.h"
#include "wallet/coincontrol.h"
#include "wallet/walletdb.h"

#include <univalue.h>

#include <stdint.h>

#include <boost/assign/list_of.hpp>

int64_t nWalletUnlockTime;
static CCriticalSection cs_nWalletUnlockTime;

std::string HelpRequiringPassphrase()
{
    return pwalletMain && pwalletMain->IsCrypted() ? "\nRequires wallet passphrase to be set with walletpassphrase call." : "";
}

bool EnsureWalletIsAvailable(bool avoidException)
{
    if (!pwalletMain) {
        if (!avoidException)
            throw JSONRPCError(RPC_METHOD_NOT_FOUND, "Method not found (disabled)");
        else
            return false;
    }
    return true;
}

void EnsureWalletIsUnlocked()
{
    if (pwalletMain->IsLocked())
        throw JSONRPCError(RPC_WALLET_UNLOCK_NEEDED, "Error: Please enter the wallet passphrase with walletpassphrase first.");
    if (fWalletUnlockMixStakeOnly)
        throw JSONRPCError(RPC_WALLET_UNLOCK_NEEDED, "Error: Wallet unlocked for mixing and staking only.");
}

void WalletTxToJSON(const CWalletTx& wtx, UniValue& entry)
{
    int confirms = wtx.GetDepthInMainChain();
    bool fLocked = instantsend.IsLockedInstantSendTransaction(wtx.GetHash());
    entry.push_back(Pair("confirmations", confirms));
    entry.push_back(Pair("instantlock", fLocked));
    if (wtx.IsCoinBase() || wtx.IsCoinStake())
        entry.push_back(Pair("generated", true));
    if (confirms > 0) {
        entry.push_back(Pair("blockhash", wtx.hashBlock.GetHex()));
        entry.push_back(Pair("blockindex", wtx.nIndex));
        entry.push_back(Pair("blocktime", mapBlockIndex[wtx.hashBlock]->GetBlockTime()));
    } else {
        entry.push_back(Pair("trusted", wtx.IsTrusted()));
    }
    uint256 hash = wtx.GetHash();
    entry.push_back(Pair("txid", hash.GetHex()));
    UniValue conflicts(UniValue::VARR);
    for (const uint256& conflict : wtx.GetConflicts())
        conflicts.push_back(conflict.GetHex());
    entry.push_back(Pair("walletconflicts", conflicts));
    entry.push_back(Pair("time", wtx.GetTxTime()));
    entry.push_back(Pair("timereceived", (int64_t)wtx.nTimeReceived));

    // Add opt-in RBF status
    std::string rbfStatus = "no";
    if (confirms <= 0) {
        LOCK(mempool.cs);
        RBFTransactionState rbfState = IsRBFOptIn(wtx, mempool);
        if (rbfState == RBF_TRANSACTIONSTATE_UNKNOWN)
            rbfStatus = "unknown";
        else if (rbfState == RBF_TRANSACTIONSTATE_REPLACEABLE_BIP125)
            rbfStatus = "yes";
    }
    entry.push_back(Pair("bip125-replaceable", rbfStatus));

    for (const std::pair<std::string, std::string>& item : wtx.mapValue)
        entry.push_back(Pair(item.first, item.second));
}

std::string AccountFromValue(const UniValue& value)
{
    std::string strAccount = value.get_str();
    if (strAccount == "*")
        throw JSONRPCError(RPC_WALLET_INVALID_ACCOUNT_NAME, "Invalid account name");
    return strAccount;
}

UniValue getnewaddress(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() > 1)
        throw std::runtime_error(
            "getnewaddress ( \"account\" )\n"
            "\nReturns a new Dynamic address for receiving payments.\n"
            "If 'account' is specified (DEPRECATED), it is added to the address book \n"
            "so payments received with the address will be credited to 'account'.\n"
            "\nArguments:\n"
            "1. \"account\"        (string, optional) DEPRECATED. The account name for the address to be linked to. If not provided, the default account \"\" is used. It can also be set to the empty string \"\" to represent the default account. The account does not need to exist, it will be created if there is no account by the given name.\n"
            "\nResult:\n"
            "\"dynamicaddress\"    (string) The new dynamic address\n"
            "\nExamples:\n" +
            HelpExampleCli("getnewaddress", "") + HelpExampleRpc("getnewaddress", ""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    // Parse the account first so we don't generate a key if there's an error
    std::string strAccount;
    if (request.params.size() > 0)
        strAccount = AccountFromValue(request.params[0]);

    //Check to see if wallet needs upgrading
    if(pwalletMain->WalletNeedsUpgrading())
        throw JSONRPCError(RPC_WALLET_NEEDS_UPGRADING, "Error: Your wallet has not been fully upgraded to version 2.4.  Please unlock your wallet to continue.");

    if (!pwalletMain->IsLocked(true))
        pwalletMain->TopUpKeyPoolCombo();

    // Generate a new key that is added to wallet
    CPubKey newKey;
    std::vector<unsigned char> newEdKey; //not really using this here, keeps keypools synced
    if (!pwalletMain->GetKeysFromPool(newKey, newEdKey, false))
        throw JSONRPCError(RPC_WALLET_KEYPOOL_RAN_OUT, "Error: Keypool ran out, please call keypoolrefill first");
    CKeyID keyID = newKey.GetID();

    pwalletMain->SetAddressBook(keyID, strAccount, "receive");

    return CDynamicAddress(keyID).ToString();
}

UniValue getnewed25519address(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() > 1)
        throw std::runtime_error(
            "getnewed25519address ( \"account\" )\n"
            "\nReturns a new Dynamic address for receiving payments.\n"
            "If 'account' is specified (DEPRECATED), it is added to the address book \n"
            "so payments received with the address will be credited to 'account'.\n"
            "\nArguments:\n"
            "1. \"account\"        (string, optional) DEPRECATED. The account name for the address to be linked to. If not provided, the default account \"\" is used. It can also be set to the empty string \"\" to represent the default account. The account does not need to exist, it will be created if there is no account by the given name.\n"
            "\nResult:\n"
            "\"dynamicaddress\"    (string) The new dynamic address\n"
            "\nExamples:\n" +
            HelpExampleCli("getnewed25519address", "") + HelpExampleRpc("getnewed25519address", ""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    // Parse the account first so we don't generate a key if there's an error
    std::string strAccount;
    if (request.params.size() > 0)
        strAccount = AccountFromValue(request.params[0]);

    //Check to see if wallet needs upgrading
    if(pwalletMain->WalletNeedsUpgrading())
        throw JSONRPCError(RPC_WALLET_NEEDS_UPGRADING, "Error: Your wallet has not been fully upgraded to version 2.4.  Please unlock your wallet to continue.");

    if (!pwalletMain->IsLocked(true))
        pwalletMain->TopUpKeyPoolCombo(); //TopUpEdKeyPool();

    // Generate a new key that is added to wallet
    CPubKey newKey;
    std::vector<unsigned char> newEdKey;
    if (!pwalletMain->GetKeysFromPool(newKey, newEdKey, false))
        throw JSONRPCError(RPC_WALLET_KEYPOOL_RAN_OUT, "Error: EdKeypool ran out, please call edkeypoolrefill first");
    CKeyID keyID = newKey.GetID();

    pwalletMain->SetAddressBook(keyID, strAccount, "receive");

    std::string newEdKeyString(newEdKey.begin(), newEdKey.end());

    UniValue returnvalue(UniValue::VOBJ);

    returnvalue.push_back(Pair("walletaddress:",CDynamicAddress(keyID).ToString()));
    returnvalue.push_back(Pair("ed25519 PubKey:",newEdKeyString));

    return returnvalue;
}

static UniValue getnewstealthaddress(const JSONRPCRequest &request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() > 1)
        throw std::runtime_error(
            "getnewstealthaddress\n"
            "\nReturns a new stealth address for receiving links.\n"
            "\nArguments:\n"
            "\nResult:\n"
            "\"stealthaddress\"    (string) The new Dynamic stealth address\n"
            "\nExamples:\n" +
            HelpExampleCli("getnewstealthaddress", "") + HelpExampleRpc("getnewstealthaddress", ""));

    EnsureWalletIsUnlocked();

    LOCK2(cs_main, pwalletMain->cs_wallet);

    //Check to see if wallet needs upgrading
    if(pwalletMain->WalletNeedsUpgrading())
        throw JSONRPCError(RPC_WALLET_NEEDS_UPGRADING, "Error: Your wallet has not been fully upgraded to version 2.4.  Please unlock your wallet to continue.");

    CPubKey walletPubKey;
    CStealthAddress sxAddr;
    std::vector<unsigned char> vchEd25519PubKey;
    if (!pwalletMain->GetKeysFromPool(walletPubKey, vchEd25519PubKey, sxAddr, false))
        throw JSONRPCError(RPC_WALLET_KEYPOOL_RAN_OUT, "Error: Keypool ran out, please call keypoolrefill first");

    return sxAddr.ToString();
}

CDynamicAddress GetAccountAddress(std::string strAccount, bool bForceNew = false)
{
    CPubKey pubKey;
    if (!pwalletMain->GetAccountPubkey(pubKey, strAccount, bForceNew)) {
        throw JSONRPCError(RPC_WALLET_KEYPOOL_RAN_OUT, "Error: Keypool ran out, please call keypoolrefill first");
    }

    return CDynamicAddress(pubKey.GetID());
}

UniValue getaccountaddress(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "getaccountaddress \"account\"\n"
            "\nDEPRECATED. Returns the current Dynamic address for receiving payments to this account.\n"
            "\nArguments:\n"
            "1. \"account\"       (string, required) The account name for the address. It can also be set to the empty string \"\" to represent the default account. The account does not need to exist, it will be created and a new address created  if there is no account by the given name.\n"
            "\nResult:\n"
            "\"dynamicaddress\"   (string) The account dynamic address\n"
            "\nExamples:\n" +
            HelpExampleCli("getaccountaddress", "") + HelpExampleCli("getaccountaddress", "\"\"") + HelpExampleCli("getaccountaddress", "\"myaccount\"") + HelpExampleRpc("getaccountaddress", "\"myaccount\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    // Parse the account first so we don't generate a key if there's an error
    std::string strAccount = AccountFromValue(request.params[0]);

    UniValue ret(UniValue::VSTR);

    ret = GetAccountAddress(strAccount).ToString();
    return ret;
}


UniValue getrawchangeaddress(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() > 1)
        throw std::runtime_error(
            "getrawchangeaddress\n"
            "\nReturns a new Dynamic address, for receiving change.\n"
            "This is for use with raw transactions, NOT normal use.\n"
            "\nResult:\n"
            "\"address\"    (string) The address\n"
            "\nExamples:\n" +
            HelpExampleCli("getrawchangeaddress", "") + HelpExampleRpc("getrawchangeaddress", ""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    if (!pwalletMain->IsLocked(true))
        pwalletMain->TopUpKeyPoolCombo();

    CReserveKey reservekey(pwalletMain);
    CPubKey vchPubKey;
    if (!reservekey.GetReservedKey(vchPubKey, true))
        throw JSONRPCError(RPC_WALLET_KEYPOOL_RAN_OUT, "Error: Keypool ran out, please call keypoolrefill first");

    reservekey.KeepKey();

    CKeyID keyID = vchPubKey.GetID();

    return CDynamicAddress(keyID).ToString();
}


UniValue setaccount(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 1 || request.params.size() > 2)
        throw std::runtime_error(
            "setaccount \"dynamicaddress\" \"account\"\n"
            "\nDEPRECATED. Sets the account associated with the given address.\n"
            "\nArguments:\n"
            "1. \"dynamicaddress\"  (string, required) The dynamic address to be associated with an account.\n"
            "2. \"account\"         (string, required) The account to assign the address to.\n"
            "\nExamples:\n" +
            HelpExampleCli("setaccount", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" \"tabby\"") + HelpExampleRpc("setaccount", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\", \"tabby\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    CDynamicAddress address(request.params[0].get_str());
    if (!address.IsValid())
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid Dynamic address");

    std::string strAccount;
    if (request.params.size() > 1)
        strAccount = AccountFromValue(request.params[1]);

    // Only add the account if the address is yours.
    if (IsMine(*pwalletMain, address.Get())) {
        // Detect when changing the account of an address that is the 'unused current key' of another account:
        if (pwalletMain->mapAddressBook.count(address.Get())) {
            std::string strOldAccount = pwalletMain->mapAddressBook[address.Get()].name;
            if (address == GetAccountAddress(strOldAccount))
                GetAccountAddress(strOldAccount, true);
        }
        pwalletMain->SetAddressBook(address.Get(), strAccount, "receive");
    } else
        throw JSONRPCError(RPC_MISC_ERROR, "setaccount can only be used with own address");

    return NullUniValue;
}


UniValue getaccount(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "getaccount \"dynamicaddress\"\n"
            "\nDEPRECATED. Returns the account associated with the given address.\n"
            "\nArguments:\n"
            "1. \"dynamicaddress\"  (string, required) The dynamic address for account lookup.\n"
            "\nResult:\n"
            "\"accountname\"        (string) the account address\n"
            "\nExamples:\n" +
            HelpExampleCli("getaccount", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\"") + HelpExampleRpc("getaccount", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    CDynamicAddress address(request.params[0].get_str());
    if (!address.IsValid())
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid Dynamic address");

    std::string strAccount;
    std::map<CTxDestination, CAddressBookData>::iterator mi = pwalletMain->mapAddressBook.find(address.Get());
    if (mi != pwalletMain->mapAddressBook.end() && !(*mi).second.name.empty())
        strAccount = (*mi).second.name;
    return strAccount;
}


UniValue getaddressesbyaccount(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "getaddressesbyaccount \"account\"\n"
            "\nDEPRECATED. Returns the list of addresses for the given account.\n"
            "\nArguments:\n"
            "1. \"account\"  (string, required) The account name.\n"
            "\nResult:\n"
            "[                     (json array of string)\n"
            "  \"dynamicaddress\"  (string) a dynamic address associated with the given account\n"
            "  ,...\n"
            "]\n"
            "\nExamples:\n" +
            HelpExampleCli("getaddressesbyaccount", "\"tabby\"") + HelpExampleRpc("getaddressesbyaccount", "\"tabby\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    std::string strAccount = AccountFromValue(request.params[0]);

    // Find all addresses that have the given account
    UniValue ret(UniValue::VARR);
    for (const std::pair<CDynamicAddress, CAddressBookData>& item : pwalletMain->mapAddressBook) {
        const CDynamicAddress& address = item.first;
        const std::string& strName = item.second.name;
        if (strName == strAccount)
            ret.push_back(address.ToString());
    }
    return ret;
}

static void SendMoney(const CTxDestination& address, CAmount nValue, bool fSubtractFeeFromAmount, CWalletTx& wtxNew, const CCoinControl& coinControl, bool fUseInstantSend = false, bool fUsePrivateSend = false)
{
    CAmount curBalance = pwalletMain->GetBalance();

    // Check amount
    if (nValue <= 0)
        throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid amount");

    if (nValue > curBalance)
        throw JSONRPCError(RPC_WALLET_INSUFFICIENT_FUNDS, "Insufficient funds");

    if (pwalletMain->GetBroadcastTransactions() && !g_connman)
        throw JSONRPCError(RPC_CLIENT_P2P_DISABLED, "Error: Peer-to-peer functionality missing or disabled");

    if (fWalletUnlockMixStakeOnly)
        throw JSONRPCError(RPC_WALLET_ERROR, "Error: Wallet unlocked for mixing and staking only, unable to create transaction.");

    CScript scriptPubKey;
    std::vector<uint8_t> vStealthData;
    bool fStealthAddress = false;
    if (address.type() == typeid(CStealthAddress))
    {
        CStealthAddress sxAddr = boost::get<CStealthAddress>(address);
        std::string sError;
        // get sxAddr from wallet map stealthAddresses
        if (0 != PrepareStealthOutput(sxAddr, scriptPubKey, vStealthData, sError)) {
            throw JSONRPCError(RPC_INTERNAL_ERROR, std::string("SendMoney failed after PrepareStealthOutput(): ") + sError);
        }
        fStealthAddress = true;
        CTxDestination newDest;
        if (ExtractDestination(scriptPubKey, newDest))
            LogPrint("bdap", "%s -- Stealth send to address: %s\n", __func__, CDynamicAddress(newDest).ToString());
    } else {
        // Parse Dynamic address
        scriptPubKey = GetScriptForDestination(address);
    }

    // Create and send the transaction

    CReserveKey reservekey(pwalletMain);
    CAmount nFeeRequired;
    std::string strError;
    std::vector<CRecipient> vecSend;
    int nChangePosRet = -1;
    CRecipient recipient = {scriptPubKey, nValue, fSubtractFeeFromAmount};
    vecSend.push_back(recipient);
    if (fStealthAddress) {
        CScript scriptData;
        scriptData << OP_RETURN << vStealthData;
        CRecipient sendData = {scriptData, 0, fSubtractFeeFromAmount};
        vecSend.push_back(sendData);
    }
    if (!pwalletMain->CreateTransaction(vecSend, wtxNew, reservekey, nFeeRequired, nChangePosRet,
            strError, coinControl, true, fUsePrivateSend ? ONLY_DENOMINATED : ALL_COINS, fUseInstantSend)) {
        if (!fSubtractFeeFromAmount && nValue + nFeeRequired > curBalance)
            strError = strprintf("Error: This transaction requires a transaction fee of at least %s", FormatMoney(nFeeRequired));
        throw JSONRPCError(RPC_WALLET_ERROR, strError);
    }
    CValidationState state;
    if (!pwalletMain->CommitTransaction(wtxNew, reservekey, g_connman.get(), state, fUseInstantSend ? NetMsgType::TXLOCKREQUEST : NetMsgType::TX)) {
        strError = strprintf("Error: The transaction was rejected! Reason given: %s", state.GetRejectReason());
        throw JSONRPCError(RPC_WALLET_ERROR, strError);
    }
}

/*
For BDAP transactions, 
    - nDataAmount is burned in the OP_RETURN transaction.
    - nOpAmount turns to BDAP credits and can only be used to fund BDAP fees
*/
void SendBDAPTransaction(const CScript& bdapDataScript, const CScript& bdapOPScript, CWalletTx& wtxNew, const CCoinControl& coinControl, const CAmount& nDataAmount, const CAmount& nOpAmount, const bool fUseInstantSend)
{
    CAmount curBalance = pwalletMain->GetBalance() + pwalletMain->GetBDAPDynamicAmount();

    // Check amounts
    if (nOpAmount <= 0)
        throw JSONRPCError(RPC_INVALID_PARAMETER, "SendBDAPTransaction invalid amount. Data and operation amounts must be greater than zero.");

    //NOTE: nDataAmount cannot be 0
    if (nDataAmount <= 0)
        throw JSONRPCError(RPC_INVALID_PARAMETER, "SendBDAPTransaction invalid amount. Data and operation amounts must be greater than zero.");

    if (nDataAmount + nOpAmount > curBalance)
        throw JSONRPCError(RPC_WALLET_INSUFFICIENT_FUNDS, "SendBDAPTransaction insufficient funds");

    if (fWalletUnlockMixStakeOnly)
        throw JSONRPCError(RPC_WALLET_ERROR, "Error: Wallet unlocked for mixing and staking only, unable to create transaction.");

    
    // Create and send the transaction
    CReserveKey reservekey(pwalletMain);
    CAmount nFeeRequired;
    std::string strError;
    std::vector<CRecipient> vecSend;
    int nChangePosInOut = 0;

    LogPrint("bdap", "Sending BDAP Data Script: %s\n", ScriptToAsmStr(bdapDataScript));
    LogPrint("bdap", "Sending BDAP OP Script: %s\n", ScriptToAsmStr(bdapOPScript));

    if (bdapDataScript.size() > 0) {
        CRecipient recDataScript = {bdapDataScript, nDataAmount, false};
        vecSend.push_back(recDataScript);
    }

    CRecipient recOPScript = {bdapOPScript, nOpAmount, false};
    vecSend.push_back(recOPScript);

    if (!pwalletMain->CreateTransaction(vecSend, wtxNew, reservekey, nFeeRequired, nChangePosInOut,
            strError, coinControl, true, ALL_COINS, fUseInstantSend, true)) {
        if (DEFAULT_MIN_RELAY_TX_FEE + DEFAULT_MIN_RELAY_TX_FEE + nFeeRequired > curBalance)
            strError = strprintf("Error: This transaction requires a transaction fee of at least %s because of its amount, complexity, or use of recently received funds!", FormatMoney(nFeeRequired));
        throw JSONRPCError(RPC_WALLET_ERROR, strError);
    }
    CValidationState state;
    if (!pwalletMain->CommitTransaction(wtxNew, reservekey, g_connman.get(), state, NetMsgType::TX)) {
        strError = strprintf("Error: The transaction was rejected! Reason given: %s", state.GetRejectReason());
        throw JSONRPCError(RPC_WALLET_ERROR, strError);
    }
}

void SendLinkingTransaction(const CScript& bdapDataScript, const CScript& bdapOPScript, const CScript& stealthScript, 
                                CWalletTx& wtxNew, const CCoinControl& coinControl, const CAmount& nOneTimeFee, const CAmount& nDepositFee, const bool fUseInstantSend)
{
    CAmount curBalance = pwalletMain->GetBalance() + pwalletMain->GetBDAPDynamicAmount();

    // Check amount
    if (nOneTimeFee <= 0)
        throw JSONRPCError(RPC_INVALID_PARAMETER, "SendLinkingTransaction invalid amount");

    if (nOneTimeFee + nDepositFee > curBalance)
        throw JSONRPCError(RPC_WALLET_INSUFFICIENT_FUNDS, "SendLinkingTransaction insufficient funds");

    if (fWalletUnlockMixStakeOnly)
        throw JSONRPCError(RPC_WALLET_ERROR, "Error: Wallet unlocked for mixing and staking only, unable to create transaction.");

    
    // Create and send the transaction
    CReserveKey reservekey(pwalletMain);
    CAmount nFeeRequired;
    std::string strError;
    std::vector<CRecipient> vecSend;
    int nChangePosInOut = 0;

    LogPrint("bdap", "Sending BDAP Linking Data Script: %s\n", ScriptToAsmStr(bdapDataScript));
    LogPrint("bdap", "Sending BDAP Linking OP Script: %s\n", ScriptToAsmStr(bdapOPScript));

    if (nOneTimeFee > 0) {
        CRecipient recDataScript = {bdapDataScript, nOneTimeFee, false};
        vecSend.push_back(recDataScript);
        if (stealthScript.size() > 0) {
            CRecipient sendStealthData = {stealthScript, 0, false};
            vecSend.push_back(sendStealthData);
            LogPrint("bdap", "Sending Stealth Script: %s\n", ScriptToAsmStr(stealthScript));
        }
    }
    CRecipient recOPScript = {bdapOPScript, nDepositFee, false};

    vecSend.push_back(recOPScript);
    // TODO (BDAP) Make sure it uses privatesend funds
    if (!pwalletMain->CreateTransaction(vecSend, wtxNew, reservekey, nFeeRequired, nChangePosInOut,
            strError, coinControl, true, ALL_COINS, fUseInstantSend, true)) {
        if (DEFAULT_MIN_RELAY_TX_FEE + nFeeRequired > pwalletMain->GetBalance())
            strError = strprintf("Error: This transaction requires a transaction fee of at least %s because of its amount, complexity, or use of recently received funds!", FormatMoney(nFeeRequired));
        throw JSONRPCError(RPC_WALLET_ERROR, strError);
    }
    CValidationState state;
    if (!pwalletMain->CommitTransaction(wtxNew, reservekey, g_connman.get(), state, NetMsgType::TX)) {
        strError = strprintf("Error: The transaction was rejected! Reason given: %s", state.GetRejectReason());
        throw JSONRPCError(RPC_WALLET_ERROR, strError);
    }
}

void SendColorTransaction(const CScript& scriptColorCoins, const CScript& stealthDataScript, CWalletTx& wtxNew, const CCoinControl& coinControl, const CAmount& nColorAmount, const bool fUseInstantSend, const bool fUsePrivateSend)
{
    CAmount curBalance = pwalletMain->GetBalance();

    // Check amount
    if (nColorAmount <= 0)
        throw JSONRPCError(RPC_INVALID_PARAMETER, strprintf("%s invalid amount", __func__));

    if (nColorAmount > curBalance)
        throw JSONRPCError(RPC_WALLET_INSUFFICIENT_FUNDS, strprintf("%s insufficient funds", __func__));

    if (fWalletUnlockMixStakeOnly)
        throw JSONRPCError(RPC_WALLET_ERROR, "Error: Wallet unlocked for mixing and staking only, unable to create transaction.");
    
    // Create and send the transaction
    CReserveKey reservekey(pwalletMain);
    CAmount nFeeRequired;
    std::string strError;
    std::vector<CRecipient> vecSend;
    int nChangePosInOut = 0;

    LogPrint("bdap", "Sending color coin script: %s\n", ScriptToAsmStr(scriptColorCoins));

    if (stealthDataScript.size() > 0) {
        CRecipient recStealthData = {stealthDataScript, 0, false};
        vecSend.push_back(recStealthData);
    }

    CRecipient recScript = {scriptColorCoins, nColorAmount, false};
    vecSend.push_back(recScript);

    if (!pwalletMain->CreateTransaction(vecSend, wtxNew, reservekey, nFeeRequired, nChangePosInOut,
            strError, coinControl, true, fUsePrivateSend ? ONLY_DENOMINATED : ALL_COINS, fUseInstantSend, true)) {
        if (nColorAmount + nFeeRequired > pwalletMain->GetBalance())
            strError = strprintf("Error: This transaction requires a transaction fee of at least %s because of its amount, complexity, or use of recently received funds!", FormatMoney(nFeeRequired));
        throw JSONRPCError(RPC_WALLET_ERROR, strError);
    }
    CValidationState state;
    if (!pwalletMain->CommitTransaction(wtxNew, reservekey, g_connman.get(), state, NetMsgType::TX)) {
        strError = strprintf("Error: The transaction was rejected! Reason given: %s", state.GetRejectReason());
        throw JSONRPCError(RPC_WALLET_ERROR, strError);
    }
}

void SendCustomTransaction(const CScript& generatedScript, CWalletTx& wtxNew, const CCoinControl& coinControl, CAmount nValue, bool fUseInstantSend = false)
{
    CAmount curBalance = pwalletMain->GetBalance();

    // Check amount
    if (nValue <= 0)
        throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid amount");

    if (nValue > curBalance)
        throw JSONRPCError(RPC_WALLET_INSUFFICIENT_FUNDS, "Insufficient funds");

    if (fWalletUnlockMixStakeOnly)
        throw JSONRPCError(RPC_WALLET_ERROR, "Error: Wallet unlocked for mixing and staking only, unable to create transaction.");

    // Parse Dynamic address
    CScript scriptPubKey = generatedScript;

    LogPrintf("Script Public Key to be sent over to custom transaction processing: %s\n", ScriptToAsmStr(scriptPubKey));

    // Create and send the transaction
    CReserveKey reservekey(pwalletMain);
    CAmount nFeeRequired;
    std::string strError;
    std::vector<CRecipient> vecSend;
    int nChangePosRet = -1;
    CRecipient recipient = {scriptPubKey, nValue, false};
    vecSend.push_back(recipient);

    if (!pwalletMain->CreateTransaction(vecSend, wtxNew, reservekey, nFeeRequired, nChangePosRet,
            strError, coinControl, true, ALL_COINS, false)) {
        if (nValue + nFeeRequired > pwalletMain->GetBalance())
            strError = strprintf("Error: This transaction requires a transaction fee of at least %s because of its amount, complexity, or use of recently received funds!", FormatMoney(nFeeRequired));
        throw JSONRPCError(RPC_WALLET_ERROR, strError);
    }
    CValidationState state;
    if (!pwalletMain->CommitTransaction(wtxNew, reservekey, g_connman.get(), state, fUseInstantSend ? NetMsgType::TXLOCKREQUEST : NetMsgType::TX)) {
        strError = strprintf("Error: The transaction was rejected! Reason given: %s", state.GetRejectReason());
        throw JSONRPCError(RPC_WALLET_ERROR, strError);
    }
}

static bool GetCoinControl(const CScript& sendAddressScript, CCoinControl* ccCoins)
{
    CTxDestination sendAddress;
    bool fValidAddress = ExtractDestination(sendAddressScript, sendAddress);
    if (!fValidAddress)
        return false;

    unsigned int nCount = 0;
    ccCoins->SetNull();
    std::vector<COutput> vecOutputs;
    pwalletMain->AvailableCoins(vecOutputs, true, nullptr, false, ALL_COINS, false);
    for (const COutput& out : vecOutputs) {
        CTxDestination outputAddress;
        const CScript& scriptPubKey = out.tx->tx->vout[out.i].scriptPubKey;
        fValidAddress = ExtractDestination(scriptPubKey, outputAddress);
        if (!fValidAddress)
            continue;

        if (sendAddress == outputAddress) {
            COutPoint outPoint(out.tx->GetHash(), out.i);
            ccCoins->Select(outPoint);
            nCount++;
        }
    }
    return (nCount > 0);
}

void SendBurnTransaction(const CScript& burnScript, CWalletTx& wtxNew, const CAmount& nValue, const CScript& scriptSendFrom)
{

    CAmount curBalance = pwalletMain->GetBalance();

    // Check amount
    if (nValue <= 0)
        throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid amount");

    if (nValue > curBalance)
        throw JSONRPCError(RPC_WALLET_INSUFFICIENT_FUNDS, "Insufficient funds");

    if (fWalletUnlockMixStakeOnly)
        throw JSONRPCError(RPC_WALLET_ERROR, "Error: Wallet unlocked for mixing and staking only, unable to create transaction.");

    LogPrintf("%s - Script public key to be sent over to the burn transaction processing: %s\n", __func__, ScriptToAsmStr(burnScript));

    // Create and send the transaction
    CReserveKey reservekey(pwalletMain);
    CAmount nFeeRequired;
    std::string strError;
    std::vector<CRecipient> vecSend;
    int nChangePosRet = -1;
    CRecipient recipient = {burnScript, nValue, false};
    vecSend.push_back(recipient);
    CCoinControl ccCoins;
    if (!scriptSendFrom.empty()) {
        if (!GetCoinControl(scriptSendFrom, &ccCoins)) {
            strError = strprintf("Error: GetCoinControl failed");
            throw JSONRPCError(RPC_WALLET_ERROR, strError);
        }
    }
    bool fUseInstantSend = false;
    if (!pwalletMain->CreateTransaction(vecSend, wtxNew, reservekey, nFeeRequired, nChangePosRet, strError, ccCoins, true, ALL_COINS, fUseInstantSend)) {
        if (nValue + nFeeRequired > pwalletMain->GetBalance())
            strError = strprintf("Error: This transaction requires a transaction fee of at least %s because of its amount, complexity, or use of recently received funds!", FormatMoney(nFeeRequired));
        throw JSONRPCError(RPC_WALLET_ERROR, strError);
    }
    CValidationState state;
    if (!pwalletMain->CommitTransaction(wtxNew, reservekey, g_connman.get(), state, fUseInstantSend ? NetMsgType::TXLOCKREQUEST : NetMsgType::TX)) {
        strError = strprintf("Error: The transaction was rejected! Reason given: %s", state.GetRejectReason());
        throw JSONRPCError(RPC_WALLET_ERROR, strError);
    }
}


UniValue sendtoaddress(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 2 || request.params.size() > 9)
        throw std::runtime_error(
            "sendtoaddress \"dynamicaddress\" amount ( \"comment\" \"comment-to\" subtractfeefromamount use_is use_ps )\n"
            "\nSend an amount to a given address.\n" +
            HelpRequiringPassphrase() +
            "\nArguments:\n"
            "1. \"dynamicaddress\" (string, required) The Dynamic address to send to.\n"
            "2. \"amount\"      (numeric or string, required) The amount in " +
            CURRENCY_UNIT + " to send. eg 0.1\n"
                            "3. \"comment\"     (string, optional) A comment used to store what the transaction is for. \n"
                            "                             This is not part of the transaction, just kept in your wallet.\n"
                            "4. \"comment-to\"  (string, optional) A comment to store the name of the person or organization \n"
                            "                             to which you're sending the transaction. This is not part of the \n"
                            "                             transaction, just kept in your wallet.\n"
                            "5. subtractfeefromamount  (boolean, optional, default=false) The fee will be deducted from the amount being sent.\n"
                            "                             The recipient will receive less Dynamic than you enter in the amount field.\n"
                            "6. \"use_is\"      (bool, optional) Send this transaction as InstantSend (default: false)\n"
                            "7. \"use_ps\"      (bool, optional) Use anonymized funds only (default: false)\n"
                            "8. conf_target            (numeric, optional) Confirmation target (in blocks)\n"
                            "9. \"estimate_mode\"      (string, optional, default=UNSET) The fee estimate mode, must be one of:\n"
                            "       \"UNSET\"\n"
                            "       \"ECONOMICAL\"\n"
                            "       \"CONSERVATIVE\"\n"
                            "\nResult:\n"
                            "\"txid\"                  (string) The transaction id.\n"
                            "\nExamples:\n" +
            HelpExampleCli("sendtoaddress", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" 0.1") + HelpExampleCli("sendtoaddress", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" 0.1 \"donation\" \"seans outpost\"") + HelpExampleCli("sendtoaddress", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" 0.1 \"\" \"\" true") + HelpExampleRpc("sendtoaddress", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\", 0.1, \"donation\", \"seans outpost\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    CTxDestination dest = DecodeDestination(request.params[0].get_str());
    if (!IsValidDestination(dest))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid address");

    // Amount
    CAmount nAmount = AmountFromValue(request.params[1]);
    if (nAmount <= 0)
        throw JSONRPCError(RPC_TYPE_ERROR, "Invalid amount for send");

    // Wallet comments
    CWalletTx wtx;
    if (request.params.size() > 2 && !request.params[2].isNull() && !request.params[2].get_str().empty())
        wtx.mapValue["comment"] = request.params[2].get_str();
    if (request.params.size() > 3 && !request.params[3].isNull() && !request.params[3].get_str().empty())
        wtx.mapValue["to"] = request.params[3].get_str();

    bool fSubtractFeeFromAmount = false;
    if (request.params.size() > 4)
        fSubtractFeeFromAmount = request.params[4].get_bool();

    bool fUseInstantSend = false;
    bool fUsePrivateSend = false;
    if (request.params.size() > 5)
        fUseInstantSend = request.params[5].get_bool();
    if (request.params.size() > 6)
        fUsePrivateSend = request.params[6].get_bool();

    CCoinControl coinControl;

    if (!request.params[7].isNull()) {
        coinControl.m_confirm_target = ParseConfirmTarget(request.params[8]);
    }

    if (!request.params[8].isNull()) {
        if (!FeeModeFromString(request.params[9].get_str(), coinControl.m_fee_mode)) {
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid estimate_mode parameter");
        }
    }

    EnsureWalletIsUnlocked();

    SendMoney(dest, nAmount, fSubtractFeeFromAmount, wtx, coinControl, fUseInstantSend, fUsePrivateSend);

    return wtx.GetHash().GetHex();
}

UniValue instantsendtoaddress(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 2 || request.params.size() > 5)
        throw std::runtime_error(
            "instantsendtoaddress \"dynamicaddress\" amount ( \"comment\" \"comment-to\" subtractfeefromamount )\n"
            "\nSend an amount to a given address. The amount is a real and is rounded to the nearest 0.00000001\n" +
            HelpRequiringPassphrase() +
            "\nArguments:\n"
            "1. \"dynamicaddress\"  (string, required) The dynamic address to send to.\n"
            "2. \"amount\"      (numeric, required) The amount in btc to send. eg 0.1\n"
            "3. \"comment\"     (string, optional) A comment used to store what the transaction is for. \n"
            "                             This is not part of the transaction, just kept in your wallet.\n"
            "4. \"comment-to\"  (string, optional) A comment to store the name of the person or organization \n"
            "                             to which you're sending the transaction. This is not part of the \n"
            "                             transaction, just kept in your wallet.\n"
            "5. subtractfeefromamount  (boolean, optional, default=false) The fee will be deducted from the amount being sent.\n"
            "                             The recipient will receive less Dynamic than you enter in the amount field.\n"
            "\nResult:\n"
            "\"transactionid\"  (string) The transaction id.\n"
            "\nExamples:\n" +
            HelpExampleCli("instantsendtoaddress", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" 0.1") + HelpExampleCli("instantsendtoaddress", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" 0.1 \"donation\" \"seans outpost\"") + HelpExampleCli("instantsendtoaddress", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" 0.1 \"\" \"\" true") + HelpExampleRpc("instantsendtoaddress", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\", 0.1, \"donation\", \"seans outpost\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    CTxDestination dest = DecodeDestination(request.params[0].get_str());
    if (!IsValidDestination(dest))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid Dynamic address");

    // Amount
    CAmount nAmount = AmountFromValue(request.params[1]);
    if (nAmount <= 0)
        throw JSONRPCError(RPC_TYPE_ERROR, "Invalid amount for send");

    // Wallet comments
    CWalletTx wtx;
    if (request.params.size() > 2 && !request.params[2].isNull() && !request.params[2].get_str().empty())
        wtx.mapValue["comment"] = request.params[2].get_str();
    if (request.params.size() > 3 && !request.params[3].isNull() && !request.params[3].get_str().empty())
        wtx.mapValue["to"] = request.params[3].get_str();

    bool fSubtractFeeFromAmount = false;
    if (request.params.size() > 4)
        fSubtractFeeFromAmount = request.params[4].get_bool();

    EnsureWalletIsUnlocked();

    CCoinControl coinControl;

    SendMoney(dest, nAmount, fSubtractFeeFromAmount, wtx, coinControl, true);

    return wtx.GetHash().GetHex();
}

UniValue sendfromaddress(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 3 || request.params.size() > 10)
        throw std::runtime_error(
            "sendfromaddress \"from_address\" \"to_address\" amount ( \"comment\" \"comment_to\" subtractfeefromamount replaceable conf_target \"estimate_mode\")\n"
            "\nSend an amount from a specific address to a given address. All DYN change will get sent back to the from_address\n"
            + HelpRequiringPassphrase() +
            "\nArguments:\n"
            "1. \"from_address\"       (string, required) The Dynamic address to send from.\n"
            "2. \"to_address\"            (string, required) The Dynamic address to send to.\n"
            "3. \"amount\"             (numeric or string, required) The amount in " + CURRENCY_UNIT + " to send. eg 0.1\n"
            "4. \"comment\"            (string, optional) A comment used to store what the transaction is for. \n"
            "                             This is not part of the transaction, just kept in your wallet.\n"
            "5. \"comment_to\"         (string, optional) A comment to store the name of the person or organization \n"
            "                             to which you're sending the transaction. This is not part of the \n"
            "                             transaction, just kept in your wallet.\n"
            "6. subtractfeefromamount  (boolean, optional, default=false) The fee will be deducted from the amount being sent.\n"
            "                             The recipient will receive less DYN than you enter in the amount field.\n"
            "7. \"use_is\"      (bool, optional) Send this transaction as InstantSend (default: false)\n"
            "8. \"use_ps\"      (bool, optional) Use anonymized funds only (default: false)\n"               
            "9. conf_target            (numeric, optional) Confirmation target (in blocks)\n"
            "10. \"estimate_mode\"      (string, optional, default=UNSET) The fee estimate mode, must be one of:\n"
            "       \"UNSET\"\n"
            "       \"ECONOMICAL\"\n"
            "       \"CONSERVATIVE\"\n"
            "\nResult:\n"
            "\"txid\"                  (string) The transaction id.\n"
            "\nExamples:\n"
            + HelpExampleCli("sendfromaddress", "\"1M72Sfpbz1BPpXFHz9m3CdqATR44Jvaydd\" \"1M72Sfpbz1BPpXFHz9m3CdqATR44Jvaydd\" 0.1")
            + HelpExampleCli("sendfromaddress", "\"1M72Sfpbz1BPpXFHz9m3CdqATR44Jvaydd\" \"1M72Sfpbz1BPpXFHz9m3CdqATR44Jvaydd\" 0.1 \"donation\" \"seans outpost\"")
            + HelpExampleCli("sendfromaddress", "\"1M72Sfpbz1BPpXFHz9m3CdqATR44Jvaydd\" \"1M72Sfpbz1BPpXFHz9m3CdqATR44Jvaydd\" 0.1 \"\" \"\" true")
            + HelpExampleRpc("sendfromaddress", "\"1M72Sfpbz1BPpXFHz9m3CdqATR44Jvaydd\" \"1M72Sfpbz1BPpXFHz9m3CdqATR44Jvaydd\", 0.1, \"donation\", \"seans outpost\"")
        );

    LOCK2(cs_main, pwalletMain->cs_wallet);

    CCoinControl coinControl;

    std::string from_address = request.params[0].get_str();
    CTxDestination from_dest = DecodeDestination(from_address);
    if (!IsValidDestination(from_dest)) {
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid from address");
    }
    coinControl.destChange = from_dest;

    CTxDestination dest = DecodeDestination(request.params[1].get_str());
    if (!IsValidDestination(dest)) {
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid address");
    }

    // Amount
    CAmount nAmount = AmountFromValue(request.params[2]);
    if (nAmount <= 0)
        throw JSONRPCError(RPC_TYPE_ERROR, "Invalid amount for send");

    // Get the coins that belong to the from address
    std::vector<COutput> coins;
    pwalletMain->AvailableCoins(coins);

    CAmount nAmountFromCoins = 0;
    for (auto out : coins) {
        // Get the address that the coin resides in, because to send a valid message. You need to send it to the same address that it currently resides in.
        CTxDestination coin_dest;
        CAmount nCoinAmount = out.tx->tx->vout[out.i].nValue;
        ExtractDestination(out.tx->tx->vout[out.i].scriptPubKey, coin_dest);

        if (nCoinAmount <= 0)
            continue;

        if (from_address != EncodeDestination(coin_dest))
            continue;

        coinControl.Select(COutPoint(out.tx->GetHash(), out.i));
        nAmountFromCoins += out.tx->tx->vout[out.i].nValue;

        if (nAmountFromCoins >= nAmount)
            break;
    }

    if (nAmountFromCoins < nAmount)
        throw JSONRPCError(RPC_TYPE_ERROR, "From Address doesn't contain enough funds");

    // Wallet comments
    CWalletTx wtx;
    if (!request.params[3].isNull() && !request.params[2].get_str().empty())
        wtx.mapValue["comment"] = request.params[2].get_str();
    if (!request.params[4].isNull() && !request.params[3].get_str().empty())
        wtx.mapValue["to"]      = request.params[3].get_str();

    bool fSubtractFeeFromAmount = false;
    if (!request.params[5].isNull()) {
        fSubtractFeeFromAmount = request.params[4].get_bool();
    }

    bool fUseInstantSend = false;
    bool fUsePrivateSend = false;
    if (request.params.size() > 6)
        fUseInstantSend = request.params[6].get_bool();
    if (request.params.size() > 7)
        fUsePrivateSend = request.params[7].get_bool();

    if (!request.params[8].isNull()) {
        coinControl.m_confirm_target = ParseConfirmTarget(request.params[6]);
    }

    if (!request.params[9].isNull()) {
        if (!FeeModeFromString(request.params[8].get_str(), coinControl.m_fee_mode)) {
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid estimate_mode parameter");
        }
    }

    EnsureWalletIsUnlocked();

    SendMoney(dest, nAmount, fSubtractFeeFromAmount, wtx, coinControl, fUseInstantSend, fUsePrivateSend);

    return wtx.GetHash().GetHex();
}

UniValue listaddressgroupings(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp)
        throw std::runtime_error(
            "listaddressgroupings\n"
            "\nLists groups of addresses which have had their common ownership\n"
            "made public by common use as inputs or as the resulting change\n"
            "in past transactions\n"
            "\nResult:\n"
            "[\n"
            "  [\n"
            "    [\n"
            "      \"dynamicaddress\",     (string) The dynamic address\n"
            "      amount,                 (numeric) The amount in " +
            CURRENCY_UNIT + "\n"
                            "      \"account\"             (string, optional) The account (DEPRECATED)\n"
                            "    ]\n"
                            "    ,...\n"
                            "  ]\n"
                            "  ,...\n"
                            "]\n"
                            "\nExamples:\n" +
            HelpExampleCli("listaddressgroupings", "") + HelpExampleRpc("listaddressgroupings", ""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    UniValue jsonGroupings(UniValue::VARR);
    std::map<CTxDestination, CAmount> balances = pwalletMain->GetAddressBalances();
    for (const std::set<CTxDestination>& grouping : pwalletMain->GetAddressGroupings()) {
        UniValue jsonGrouping(UniValue::VARR);
        for (CTxDestination address : grouping) {
            UniValue addressInfo(UniValue::VARR);
            addressInfo.push_back(CDynamicAddress(address).ToString());
            addressInfo.push_back(ValueFromAmount(balances[address]));
            {
                if (pwalletMain->mapAddressBook.find(CDynamicAddress(address).Get()) != pwalletMain->mapAddressBook.end())
                    addressInfo.push_back(pwalletMain->mapAddressBook.find(CDynamicAddress(address).Get())->second.name);
            }
            jsonGrouping.push_back(addressInfo);
        }
        jsonGroupings.push_back(jsonGrouping);
    }
    return jsonGroupings;
}

UniValue listaddressbalances(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() > 1)
        throw std::runtime_error(
            "listaddressbalances ( minamount )\n"
            "\nLists addresses of this wallet and their balances\n"
            "\nArguments:\n"
            "1. minamount               (numeric, optional, default=0) Minimum balance in " +
            CURRENCY_UNIT + " an address should have to be shown in the list\n"
                            "\nResult:\n"
                            "{\n"
                            "  \"address\": amount,       (string) The Dynamic address and the amount in " +
            CURRENCY_UNIT + "\n"
                            "  ,...\n"
                            "}\n"
                            "\nExamples:\n" +
            HelpExampleCli("listaddressbalances", "") + HelpExampleCli("listaddressbalances", "10") + HelpExampleRpc("listaddressbalances", "") + HelpExampleRpc("listaddressbalances", "10"));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    CAmount nMinAmount = 0;
    if (request.params.size() > 0)
        nMinAmount = AmountFromValue(request.params[0]);

    if (nMinAmount < 0)
        throw JSONRPCError(RPC_TYPE_ERROR, "Invalid amount");

    UniValue jsonBalances(UniValue::VOBJ);
    std::map<CTxDestination, CAmount> balances = pwalletMain->GetAddressBalances();
    for (auto& balance : balances)
        if (balance.second >= nMinAmount)
            jsonBalances.push_back(Pair(CDynamicAddress(balance.first).ToString(), ValueFromAmount(balance.second)));

    return jsonBalances;
}

UniValue signmessage(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() != 2)
        throw std::runtime_error(
            "signmessage \"dynamicaddress\" \"message\"\n"
            "\nSign a message with the private key of an address" +
            HelpRequiringPassphrase() + "\n"
                                        "\nArguments:\n"
                                        "1. \"dynamicaddress\"  (string, required) The dynamic address to use for the private key.\n"
                                        "2. \"message\"         (string, required) The message to create a signature of.\n"
                                        "\nResult:\n"
                                        "\"signature\"          (string) The signature of the message encoded in base 64\n"
                                        "\nExamples:\n"
                                        "\nUnlock the wallet for 30 seconds\n" +
            HelpExampleCli("walletpassphrase", "\"mypassphrase\" 30") +
            "\nCreate the signature\n" + HelpExampleCli("signmessage", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" \"my message\"") +
            "\nVerify the signature\n" + HelpExampleCli("verifymessage", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" \"signature\" \"my message\"") +
            "\nAs json rpc\n" + HelpExampleRpc("signmessage", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\", \"my message\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    EnsureWalletIsUnlocked();

    std::string strAddress = request.params[0].get_str();
    std::string strMessage = request.params[1].get_str();

    CDynamicAddress addr(strAddress);
    if (!addr.IsValid())
        throw JSONRPCError(RPC_TYPE_ERROR, "Invalid address");

    CKeyID keyID;
    if (!addr.GetKeyID(keyID))
        throw JSONRPCError(RPC_TYPE_ERROR, "Address does not refer to key");

    CKey key;
    if (!pwalletMain->GetKey(keyID, key))
        throw JSONRPCError(RPC_WALLET_ERROR, "Private key not available");

    CHashWriter ss(SER_GETHASH, 0);
    ss << strMessageMagic;
    ss << strMessage;

    std::vector<unsigned char> vchSig;
    if (!key.SignCompact(ss.GetHash(), vchSig))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Sign failed");

    return EncodeBase64(&vchSig[0], vchSig.size());
}

UniValue getreceivedbyaddress(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 1 || request.params.size() > 3)
        throw std::runtime_error(
            "getreceivedbyaddress \"dynamicaddress\" ( minconf addlocked )\n"
            "\nReturns the total amount received by the given dynamicaddress in transactions with specified minimum number of confirmations.\n"
            "\nArguments:\n"
            "1. \"dynamicaddress\"  (string, required) The dynamic address for transactions.\n"
            "2. minconf        (numeric, optional, default=1) Only include transactions confirmed at least this many times.\n"
            "3. addlocked           (bool, optional, default=false) Whether to include transactions locked via InstantSend.\n"
                            "\nResult:\n"
                            "amount            (numeric) The total amount in " +
            CURRENCY_UNIT + " received at this address.\n"
                            "\nExamples:\n"
                            "\nThe amount from transactions with at least 1 confirmation\n" +
            HelpExampleCli("getreceivedbyaddress", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\"") +
            "\nThe amount including unconfirmed transactions, zero confirmations\n" + HelpExampleCli("getreceivedbyaddress", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" 0") +
            "\nThe amount with at least 10 confirmation, very safe\n" + HelpExampleCli("getreceivedbyaddress", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" 10") +
            "\nAs a json rpc call\n" + HelpExampleRpc("getreceivedbyaddress", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\", 10"));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    // Dynamic address
    CDynamicAddress address = CDynamicAddress(request.params[0].get_str());
    if (!address.IsValid())
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid Dynamic address");
    CScript scriptPubKey = GetScriptForDestination(address.Get());
    if (!IsMine(*pwalletMain, scriptPubKey))
        return ValueFromAmount(0);

    // Minimum confirmations
    int nMinDepth = 1;
    if (request.params.size() > 1)
        nMinDepth = request.params[1].get_int();
    bool fAddLocked = (request.params.size() > 2 && request.params[2].get_bool());

    // Tally
    CAmount nAmount = 0;
    for (std::map<uint256, CWalletTx>::iterator it = pwalletMain->mapWallet.begin(); it != pwalletMain->mapWallet.end(); ++it) {
        const CWalletTx& wtx = (*it).second;
        if (wtx.IsCoinBase() || wtx.IsCoinStake() || !CheckFinalTx(*wtx.tx))
            continue;

        for (const CTxOut& txout : wtx.tx->vout)
            if (txout.scriptPubKey == scriptPubKey)
                if ((wtx.GetDepthInMainChain() >= nMinDepth) || (fAddLocked && wtx.IsLockedByInstantSend()))
                    nAmount += txout.nValue;
    }

    return ValueFromAmount(nAmount);
}


UniValue getreceivedbyaccount(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 1 || request.params.size() > 3)
        throw std::runtime_error(
            "getreceivedbyaccount \"account\" ( minconf addlocked )\n"
            "\nDEPRECATED. Returns the total amount received by addresses with <account> in transactions with specified minimum number of confirmations.\n"
            "\nArguments:\n"
            "1. \"account\"      (string, required) The selected account, may be the default account using \"\".\n"
            "2. minconf        (numeric, optional, default=1) Only include transactions confirmed at least this many times.\n"
            "3. addlocked      (bool, optional, default=false) Whether to include transactions locked via InstantSend.\n"
                            "\nResult:\n"
                            "amount            (numeric) The total amount in " +
            CURRENCY_UNIT + " received for this account.\n"
                            "\nExamples:\n"
                            "\nAmount received by the default account with at least 1 confirmation\n" +
            HelpExampleCli("getreceivedbyaccount", "\"\"") +
            "\nAmount received at the tabby account including unconfirmed amounts with zero confirmations\n" + HelpExampleCli("getreceivedbyaccount", "\"tabby\" 0") +
            "\nThe amount with at least 10 confirmation, very safe\n" + HelpExampleCli("getreceivedbyaccount", "\"tabby\" 10") +
            "\nAs a json rpc call\n" + HelpExampleRpc("getreceivedbyaccount", "\"tabby\", 10"));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    // Minimum confirmations
    int nMinDepth = 1;
    if (request.params.size() > 1)
        nMinDepth = request.params[1].get_int();
    bool fAddLocked = (request.params.size() > 2 && request.params[2].get_bool());

    // Get the set of pub keys assigned to account
    std::string strAccount = AccountFromValue(request.params[0]);
    std::set<CTxDestination> setAddress = pwalletMain->GetAccountAddresses(strAccount);

    // Tally
    CAmount nAmount = 0;
    for (std::map<uint256, CWalletTx>::iterator it = pwalletMain->mapWallet.begin(); it != pwalletMain->mapWallet.end(); ++it) {
        const CWalletTx& wtx = (*it).second;
        if (wtx.IsCoinBase() || wtx.IsCoinStake() || !CheckFinalTx(*wtx.tx))
            continue;

        for (const CTxOut& txout : wtx.tx->vout) {
            CTxDestination address;
            if (ExtractDestination(txout.scriptPubKey, address) && IsMine(*pwalletMain, address) && setAddress.count(address))
                if ((wtx.GetDepthInMainChain() >= nMinDepth) || (fAddLocked && wtx.IsLockedByInstantSend()))
                    nAmount += txout.nValue;
        }
    }

    return ValueFromAmount(nAmount);
}

UniValue getbalance(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() > 4)
        throw std::runtime_error(
            "getbalance ( \"account\" minconf addlocked includeWatchonly )\n"
            "\nIf account is not specified, returns the server's total available balance.\n"
            "If account is specified (DEPRECATED), returns the balance in the account.\n"
            "Note that the account \"\" is not the same as leaving the parameter out.\n"
            "The server total may be different to the balance in the default \"\" account.\n"
            "\nArguments:\n"
            "1. \"account\"        (string, optional) DEPRECATED. The selected account, or \"*\" for entire wallet. It may be the default account using \"\".\n"
            "2. minconf          (numeric, optional, default=1) Only include transactions confirmed at least this many times.\n"
            "3. addlocked      (bool, optional, default=false) Whether to add  balance from transactions locked via InstantSend.\n"
            "4. includeWatchonly (bool, optional, default=false) Also include balance in watchonly addresses (see 'importaddress')\n"
                            "\nResult:\n"
                            "amount              (numeric) The total amount in " +
            CURRENCY_UNIT + " received for this account.\n"
                            "\nExamples:\n"
                            "\nThe total amount in the wallet\n" +
            HelpExampleCli("getbalance", "") +
            "\nThe total amount in the wallet at least 5 blocks confirmed\n" + HelpExampleCli("getbalance", "\"*\" 10") +
            "\nAs a json rpc call\n" + HelpExampleRpc("getbalance", "\"*\", 10"));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    if (request.params.size() == 0)
        return ValueFromAmount(pwalletMain->GetBalance());

    int nMinDepth = 1;
    if (request.params.size() > 1)
        nMinDepth = request.params[1].get_int();
    bool fAddLocked = (request.params.size() > 2 && request.params[2].get_bool());
    isminefilter filter = ISMINE_SPENDABLE;
    if (request.params.size() > 3)
        if (request.params[3].get_bool())
            filter = filter | ISMINE_WATCH_ONLY;

    if (request.params[0].get_str() == "*") {
        // Calculate total balance a different way from GetBalance()
        // (GetBalance() sums up all unspent TxOuts)
        // getbalance and "getbalance * 1 true" should return the same number
        CAmount nBalance = 0;
        for (std::map<uint256, CWalletTx>::iterator it = pwalletMain->mapWallet.begin(); it != pwalletMain->mapWallet.end(); ++it) {
            const CWalletTx& wtx = (*it).second;
            if (!CheckFinalTx(wtx) || wtx.GetBlocksToMaturity() > 0 || wtx.GetDepthInMainChain() < 0)
                continue;

            CAmount allFee;
            std::string strSentAccount;
            std::list<COutputEntry> listReceived;
            std::list<COutputEntry> listSent;
            wtx.GetAmounts(listReceived, listSent, allFee, strSentAccount, filter);
            if ((wtx.GetDepthInMainChain() >= nMinDepth) || (fAddLocked && wtx.IsLockedByInstantSend())) {
                for (const COutputEntry& r : listReceived)
                    nBalance += r.amount;
            }
            for (const COutputEntry& s : listSent)
                nBalance -= s.amount;
            nBalance -= allFee;
        }
        return ValueFromAmount(nBalance);
    }

    std::string strAccount = AccountFromValue(request.params[0]);

    CAmount nBalance = pwalletMain->GetAccountBalance(strAccount, nMinDepth, filter, fAddLocked);

    return ValueFromAmount(nBalance);
}

UniValue getunconfirmedbalance(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() > 0)
        throw std::runtime_error(
            "getunconfirmedbalance\n"
            "Returns the server's total unconfirmed balance\n");

    LOCK2(cs_main, pwalletMain->cs_wallet);

    return ValueFromAmount(pwalletMain->GetUnconfirmedBalance());
}


UniValue movecmd(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 3 || request.params.size() > 5)
        throw std::runtime_error(
            "move \"fromaccount\" \"toaccount\" amount ( minconf \"comment\" )\n"
            "\nDEPRECATED. Move a specified amount from one account in your wallet to another.\n"
            "\nArguments:\n"
            "1. \"fromaccount\"    (string, required) The name of the account to move funds from. May be the default account using \"\".\n"
            "2. \"toaccount\"      (string, required) The name of the account to move funds to. May be the default account using \"\".\n"
            "3. amount           (numeric) Quantity of " +
            CURRENCY_UNIT + " to move between accounts.\n"
                            "4. minconf          (numeric, optional, default=1) Only use funds with at least this many confirmations.\n"
                            "5. \"comment\"        (string, optional) An optional comment, stored in the wallet only.\n"
                            "\nResult:\n"
                            "true|false          (boolean) true if successful.\n"
                            "\nExamples:\n"
                            "\nMove 0.01 " +
            CURRENCY_UNIT + " from the default account to the account named tabby\n" + HelpExampleCli("move", "\"\" \"tabby\" 0.01") +
            "\nMove 0.01 " + CURRENCY_UNIT + " timotei to akiko with a comment and funds have 10 confirmations\n" + HelpExampleCli("move", "\"timotei\" \"akiko\" 0.01 10 \"happy birthday!\"") +
            "\nAs a json rpc call\n" + HelpExampleRpc("move", "\"timotei\", \"akiko\", 0.01, 10, \"happy birthday!\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    std::string strFrom = AccountFromValue(request.params[0]);
    std::string strTo = AccountFromValue(request.params[1]);
    CAmount nAmount = AmountFromValue(request.params[2]);
    if (nAmount <= 0)
        throw JSONRPCError(RPC_TYPE_ERROR, "Invalid amount for send");
    if (request.params.size() > 3)
        // unused parameter, used to be nMinDepth, keep type-checking it though
        (void)request.params[3].get_int();
    std::string strComment;
    if (request.params.size() > 4)
        strComment = request.params[4].get_str();

    if (!pwalletMain->AccountMove(strFrom, strTo, nAmount, strComment))
        throw JSONRPCError(RPC_DATABASE_ERROR, "database error");

    return true;
}


UniValue sendfrom(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 3 || request.params.size() > 7)
        throw std::runtime_error(
            "sendfrom \"fromaccount\" \"todynamicaddress\" amount ( minconf addlocked \"comment\" \"comment-to\" )\n"
            "\nDEPRECATED (use sendtoaddress). Sent an amount from an account to a dynamic address." +
            HelpRequiringPassphrase() + "\n"
                                        "\nArguments:\n"
                                        "1. \"fromaccount\"    (string, required) The name of the account to send funds from. May be the default account using \"\".\n"
                                        "2. \"todynamicaddress\"  (string, required) The dynamic address to send funds to.\n"
                                        "3. amount           (numeric or string, required) The amount in " +
            CURRENCY_UNIT + " (transaction fee is added on top).\n"
                            "4. minconf          (numeric, optional, default=1) Only use funds with at least this many confirmations.\n"
                            "5. addlocked         (bool, optional, default=false) Whether to include transactions locked via InstantSend.\n"
                                                "6. \"comment\"        (string, optional) A comment used to store what the transaction is for. \n"
                                                "                                     This is not part of the transaction, just kept in your wallet.\n"
                                                "7. \"comment-to\"     (string, optional) An optional comment to store the name of the person or organization \n"
                                                "                                     to which you're sending the transaction. This is not part of the transaction, \n"
                                                "                                     it is just kept in your wallet.\n"
                                                "\nResult:\n"
                                                "\"transactionid\"     (string) The transaction id.\n"
                                                "\nExamples:\n"
                                                "\nSend 0.01 " +
            CURRENCY_UNIT + " from the default account to the address, must have at least 1 confirmation\n" + HelpExampleCli("sendfrom", "\"\" \"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" 0.01") +
            "\nSend 0.01 from the tabby account to the given address, funds must have at least 10 confirmations\n" + HelpExampleCli("sendfrom", "\"tabby\" \"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" 0.01 10 false \"donation\" \"seans outpost\"") +
            "\nAs a json rpc call\n" + HelpExampleRpc("sendfrom", "\"tabby\", \"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\", 0.01, 10, false \"donation\", \"seans outpost\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    std::string strAccount = AccountFromValue(request.params[0]);

    const CDynamicAddress address(request.params[1].get_str());
    if (!address.IsValid())
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid Dynamic address");

    CAmount nAmount = AmountFromValue(request.params[2]);
    if (nAmount <= 0)
        throw JSONRPCError(RPC_TYPE_ERROR, "Invalid amount for send");
    int nMinDepth = 1;
    if (request.params.size() > 3)
        nMinDepth = request.params[3].get_int();
    bool fAddLocked = (request.params.size() > 4 && request.params[4].get_bool());

    CWalletTx wtx;
    wtx.strFromAccount = strAccount;
    if (request.params.size() > 5 && !request.params[5].isNull() && !request.params[5].get_str().empty())
        wtx.mapValue["comment"] = request.params[5].get_str();
    if (request.params.size() > 6 && !request.params[6].isNull() && !request.params[6].get_str().empty())
        wtx.mapValue["to"] = request.params[6].get_str();

    EnsureWalletIsUnlocked();

    // Check funds
    CAmount nBalance = pwalletMain->GetAccountBalance(strAccount, nMinDepth, ISMINE_SPENDABLE, fAddLocked);
    if (nAmount > nBalance)
        throw JSONRPCError(RPC_WALLET_INSUFFICIENT_FUNDS, "Account has insufficient funds");

    CCoinControl coinControl;

    SendMoney(address.Get(), nAmount, false, wtx, coinControl);

    return wtx.GetHash().GetHex();
}


UniValue sendmany(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 2 || request.params.size() > 11)
        throw std::runtime_error(
            "sendmany \"fromaccount\" {\"address\":amount,...} ( minconf addlocked \"comment\" [\"address\",...] subtractfeefromamount use_is use_ps )\n"
            "\nSend multiple times. Amounts are double-precision floating point numbers." +
            HelpRequiringPassphrase() + "\n"
                                        "\nArguments:\n"
                                        "1. \"fromaccount\"           (string, required) DEPRECATED. The account to send the funds from. Should be \"\" for the default account\n"
                                        "2. \"amounts\"               (string, required) A json object with addresses and amounts\n"
                                        "    {\n"
                                        "      \"address\":amount     (numeric or string) The dynamic address is the key, the numeric amount (can be string) in " +
            CURRENCY_UNIT + " is the value\n"
                            "      ,...\n"
                            "    }\n"
                            "3. minconf                 (numeric, optional, default=1) Only use the balance confirmed at least this many times.\n"
                            "4. addlocked               (bool, optional, default=false) Whether to include transactions locked via InstantSend.\n"
                            "5. \"comment\"               (string, optional) A comment\n"
                            "6. subtractfeefromamount   (string, optional) A json array with addresses.\n"
                            "                           The fee will be equally deducted from the amount of each selected address.\n"
                            "                           Those recipients will receive less Dynamic than you enter in their corresponding amount field.\n"
                            "                           If no addresses are specified here, the sender pays the fee.\n"
                            "    [\n"
                            "      \"address\"            (string) Subtract fee from this address\n"
                            "      ,...\n"
                            "    ]\n"
                            "7. \"use_is\"                (bool, optional) Send this transaction as InstantSend (default: false)\n"
                            "8. \"use_ps\"                (bool, optional) Use anonymized funds only (default: false)\n"
                            "9. conf_target            (numeric, optional) Confirmation target (in blocks)\n"
                            "10. \"estimate_mode\"      (string, optional, default=UNSET) The fee estimate mode, must be one of:\n"
                            "       \"UNSET\"\n"
                            "       \"ECONOMICAL\"\n"
                            "       \"CONSERVATIVE\"\n"
                             "\nResult:\n"
                            "\"txid\"                   (string) The transaction id for the send. Only 1 transaction is created regardless of \n"
                            "                                    the number of addresses.\n"
                            "\nExamples:\n"
                            "\nSend two amounts to two different addresses:\n" +
            HelpExampleCli("sendmany", "\"tabby\" \"{\\\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\\\":0.01,\\\"D1MfcDTf7Zsef8gMGL2fhWA9ZslrP4K5tf\\\":0.02}\"") +
            "\nSend two amounts to two different addresses setting the confirmation and comment:\n" + HelpExampleCli("sendmany", "\"tabby\" \"{\\\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\\\":0.01,\\\"D1MfcDTf7Zsef8gMGL2fhWA9ZslrP4K5tf\\\":0.02}\" 10 false \"testing\"") +
            "\nAs a json rpc call\n" + HelpExampleRpc("sendmany", "\"tabby\", \"{\\\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\\\":0.01,\\\"D1MfcDTf7Zsef8gMGL2fhWA9ZslrP4K5tf\\\":0.02}\", 10, false \"testing\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    if (pwalletMain->GetBroadcastTransactions() && !g_connman)
        throw JSONRPCError(RPC_CLIENT_P2P_DISABLED, "Error: Peer-to-peer functionality missing or disabled");

    std::string strAccount = AccountFromValue(request.params[0]);
    UniValue sendTo = request.params[1].get_obj();
    int nMinDepth = 1;
    if (request.params.size() > 2)
        nMinDepth = request.params[2].get_int();
    bool fAddLocked = (request.params.size() > 3 && request.params[3].get_bool());

    CWalletTx wtx;
    wtx.strFromAccount = strAccount;
    if (request.params.size() > 4 && !request.params[4].isNull() && !request.params[4].get_str().empty())
        wtx.mapValue["comment"] = request.params[4].get_str();

    UniValue subtractFeeFromAmount(UniValue::VARR);
    if (request.params.size() > 5)
        subtractFeeFromAmount = request.params[5].get_array();

    std::set<CTxDestination> setDestAddress;
    std::vector<CRecipient> vecSend;

    CAmount totalAmount = 0;
    std::vector<std::string> keys = sendTo.getKeys();
    for (const std::string& name_ : keys) {
        bool fStealthAddress = false;
        CTxDestination dest = DecodeDestination(name_);
        if (!IsValidDestination(dest))
            throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, strprintf("Invalid Dynamic address %s", name_));

        if (setDestAddress.count(dest))
            throw JSONRPCError(RPC_INVALID_PARAMETER, strprintf("Invalid parameter, duplicated address:  %s", name_));
        setDestAddress.insert(dest);

        std::vector<uint8_t> vStealthData;
        CScript scriptPubKey;
        if (dest.type() == typeid(CStealthAddress))
        {
            CStealthAddress sxAddr = boost::get<CStealthAddress>(dest);
            std::string sError;
            if (0 != PrepareStealthOutput(sxAddr, scriptPubKey, vStealthData, sError)) {
                throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, strprintf("PrepareStealthOutput failed for address:  %s, Error: %s", name_, sError));
            }
            fStealthAddress = true;
            CTxDestination newDest;
            if (ExtractDestination(scriptPubKey, newDest))
                LogPrintf("%s -- Stealth send to address: %s\n", __func__, CDynamicAddress(newDest).ToString());

        } else
        {
            scriptPubKey = GetScriptForDestination(dest);
        }
        CAmount nAmount = AmountFromValue(sendTo[name_]);
        if (nAmount <= 0)
            throw JSONRPCError(RPC_TYPE_ERROR, "Invalid amount for send");
        totalAmount += nAmount;

        bool fSubtractFeeFromAmount = false;
        for (unsigned int idx = 0; idx < subtractFeeFromAmount.size(); idx++) {
            const UniValue& addr = subtractFeeFromAmount[idx];
            if (addr.get_str() == name_)
                fSubtractFeeFromAmount = true;
        }

        CRecipient recipient = {scriptPubKey, nAmount, fSubtractFeeFromAmount};
        vecSend.push_back(recipient);
        if (fStealthAddress) {
            CScript scriptData;
            scriptData << OP_RETURN << vStealthData;
            CRecipient sendData = {scriptData, 0, fSubtractFeeFromAmount};
            vecSend.push_back(sendData);
        }
    }

    EnsureWalletIsUnlocked();

    // Check funds
    CAmount nBalance = pwalletMain->GetAccountBalance(strAccount, nMinDepth, ISMINE_SPENDABLE, fAddLocked);
    if (totalAmount > nBalance)
        throw JSONRPCError(RPC_WALLET_INSUFFICIENT_FUNDS, "Account has insufficient funds");

    // Send
    CReserveKey keyChange(pwalletMain);
    CAmount nFeeRequired = 0;
    int nChangePosRet = -1;
    std::string strFailReason;
    bool fUseInstantSend = false;
    bool fUsePrivateSend = false;
    if (request.params.size() > 6)
        fUseInstantSend = request.params[6].get_bool();
    if (request.params.size() > 7)
        fUsePrivateSend = request.params[7].get_bool();

    CCoinControl coinControl;
    
    if (!request.params[8].isNull()) {
        coinControl.m_confirm_target = ParseConfirmTarget(request.params[9]);
    }

    if (!request.params[9].isNull()) {
        if (!FeeModeFromString(request.params[10].get_str(), coinControl.m_fee_mode)) {
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid estimate_mode parameter");
        }
    }

    bool fCreated = pwalletMain->CreateTransaction(vecSend, wtx, keyChange, nFeeRequired, nChangePosRet, strFailReason,
        coinControl, true, fUsePrivateSend ? ONLY_DENOMINATED : ALL_COINS, fUseInstantSend);
    if (!fCreated)
        throw JSONRPCError(RPC_WALLET_INSUFFICIENT_FUNDS, strFailReason);
    CValidationState state;
    if (!pwalletMain->CommitTransaction(wtx, keyChange, g_connman.get(), state, fUseInstantSend ? NetMsgType::TXLOCKREQUEST : NetMsgType::TX)) {
        strFailReason = strprintf("Transaction commit failed:: %s", state.GetRejectReason());
        throw JSONRPCError(RPC_WALLET_ERROR, strFailReason);
    }

    return wtx.GetHash().GetHex();
}

// Defined in rpcmisc.cpp
extern CScript _createmultisig_redeemScript(const UniValue& params);

UniValue addmultisigaddress(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 2 || request.params.size() > 3) {
        std::string msg = "addmultisigaddress nrequired [\"key\",...] ( \"account\" )\n"
                          "\nAdd a nrequired-to-sign multisignature address to the wallet.\n"
                          "Each key is a Dynamic address or hex-encoded public key.\n"
                          "If 'account' is specified (DEPRECATED), assign address to that account.\n"

                          "\nArguments:\n"
                          "1. nrequired        (numeric, required) The number of required signatures out of the n keys or addresses.\n"
                          "2. \"keysobject\"   (string, required) A json array of dynamic addresses or hex-encoded public keys\n"
                          "     [\n"
                          "       \"address\"  (string) dynamic address or hex-encoded public key\n"
                          "       ...,\n"
                          "     ]\n"
                          "3. \"account\"      (string, optional) DEPRECATED. An account to assign the addresses to.\n"

                          "\nResult:\n"
                          "\"dynamicaddress\"  (string) A dynamic address associated with the keys.\n"

                          "\nExamples:\n"
                          "\nAdd a multisig address from 2 addresses\n" +
                          HelpExampleCli("addmultisigaddress", "2 \"[\\\"D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\\\",\\\"D2sMrF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\\\"]\"") +
                          "\nAs json rpc call\n" + HelpExampleRpc("addmultisigaddress", "2, \"[\\\"D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\\\",\\\"D2sMrF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\\\"]\"");
        throw std::runtime_error(msg);
    }

    LOCK2(cs_main, pwalletMain->cs_wallet);

    std::string strAccount;
    if (request.params.size() > 2)
        strAccount = AccountFromValue(request.params[2]);

    // Construct using pay-to-script-hash:
    CScript inner = _createmultisig_redeemScript(request.params);
    CScriptID innerID(inner);
    pwalletMain->AddCScript(inner);

    pwalletMain->SetAddressBook(innerID, strAccount, "send");
    return CDynamicAddress(innerID).ToString();
}


struct tallyitem {
    CAmount nAmount;
    int nConf;
    std::vector<uint256> txids;
    bool fIsWatchonly;
    tallyitem()
    {
        nAmount = 0;
        nConf = std::numeric_limits<int>::max();
        fIsWatchonly = false;
    }
};

UniValue ListReceived(const UniValue& params, bool fByAccounts)
{
    // Minimum confirmations
    int nMinDepth = 1;
    if (params.size() > 0)
        nMinDepth = params[0].get_int();
    bool fAddLocked = (params.size() > 1 && params[1].get_bool());

    // Whether to include empty accounts
    bool fIncludeEmpty = false;
    if (params.size() > 2)
        fIncludeEmpty = params[2].get_bool();

    isminefilter filter = ISMINE_SPENDABLE;
    if (params.size() > 3)
        if (params[3].get_bool())
            filter = filter | ISMINE_WATCH_ONLY;

    // Tally
    std::map<CDynamicAddress, tallyitem> mapTally;
    for (std::map<uint256, CWalletTx>::iterator it = pwalletMain->mapWallet.begin(); it != pwalletMain->mapWallet.end(); ++it) {
        const CWalletTx& wtx = (*it).second;

        if (wtx.IsCoinBase() || wtx.IsCoinStake() || !CheckFinalTx(*wtx.tx))
            continue;

        int nDepth = wtx.GetDepthInMainChain();
        if ((nDepth < nMinDepth) && !(fAddLocked && wtx.IsLockedByInstantSend()))
            continue;

        for (const CTxOut& txout : wtx.tx->vout) {
            CTxDestination address;
            if (!ExtractDestination(txout.scriptPubKey, address))
                continue;

            isminefilter mine = IsMine(*pwalletMain, address);
            if (!(mine & filter))
                continue;

            tallyitem& item = mapTally[address];
            item.nAmount += txout.nValue;
            item.nConf = std::min(item.nConf, nDepth);
            item.txids.push_back(wtx.GetHash());
            if (mine & ISMINE_WATCH_ONLY)
                item.fIsWatchonly = true;
        }
    }

    // Reply
    UniValue ret(UniValue::VARR);
    std::map<std::string, tallyitem> mapAccountTally;
    for (const std::pair<CDynamicAddress, CAddressBookData>& item : pwalletMain->mapAddressBook) {
        const CDynamicAddress& address = item.first;
        const std::string& strAccount = item.second.name;
        std::map<CDynamicAddress, tallyitem>::iterator it = mapTally.find(address);
        if (it == mapTally.end() && !fIncludeEmpty)
            continue;

        isminefilter mine = IsMine(*pwalletMain, address.Get());
        if (!(mine & filter))
            continue;

        CAmount nAmount = 0;
        int nConf = std::numeric_limits<int>::max();
        bool fIsWatchonly = false;
        if (it != mapTally.end()) {
            nAmount = (*it).second.nAmount;
            nConf = (*it).second.nConf;
            fIsWatchonly = (*it).second.fIsWatchonly;
        }

        if (fByAccounts) {
            tallyitem& _item = mapAccountTally[strAccount];
            _item.nAmount += nAmount;
            _item.nConf = std::min(_item.nConf, nConf);
            _item.fIsWatchonly = fIsWatchonly;
        } else {
            UniValue obj(UniValue::VOBJ);
            if (fIsWatchonly)
                obj.push_back(Pair("involvesWatchonly", true));
            obj.push_back(Pair("address", address.ToString()));
            obj.push_back(Pair("account", strAccount));
            obj.push_back(Pair("amount", ValueFromAmount(nAmount)));
            obj.push_back(Pair("confirmations", (nConf == std::numeric_limits<int>::max() ? 0 : nConf)));
            if (!fByAccounts)
                obj.push_back(Pair("label", strAccount));
            UniValue transactions(UniValue::VARR);
            if (it != mapTally.end()) {
                for (const uint256& _item : (*it).second.txids) {
                    transactions.push_back(_item.GetHex());
                }
            }
            obj.push_back(Pair("txids", transactions));
            ret.push_back(obj);
        }
    }

    if (fByAccounts) {
        for (std::map<std::string, tallyitem>::iterator it = mapAccountTally.begin(); it != mapAccountTally.end(); ++it) {
            CAmount nAmount = (*it).second.nAmount;
            int nConf = (*it).second.nConf;
            UniValue obj(UniValue::VOBJ);
            if ((*it).second.fIsWatchonly)
                obj.push_back(Pair("involvesWatchonly", true));
            obj.push_back(Pair("account", (*it).first));
            obj.push_back(Pair("amount", ValueFromAmount(nAmount)));
            obj.push_back(Pair("confirmations", (nConf == std::numeric_limits<int>::max() ? 0 : nConf)));
            ret.push_back(obj);
        }
    }

    return ret;
}

UniValue listreceivedbyaddress(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() > 4)
        throw std::runtime_error(
            "listreceivedbyaddress ( minconf addlocked include_empty include_watchonly)\n"
            "\nList incoming payments grouped by receiving address.\n"
            "\nArguments:\n"
            "1. minconf           (numeric, optional, default=1) The minimum number of confirmations before payments are included.\n"
            "2. addlocked         (bool, optional, default=false) Whether to include transactions locked via InstantSend.\n"
            "3. include_empty     (bool, optional, default=false) Whether to include addresses that haven't received any payments.\n"
            "4. include_watchonly (bool, optional, default=false) Whether to include watch-only addresses (see 'importaddress').\n"

            "\nResult:\n"
            "[\n"
            "  {\n"
            "    \"involvesWatchonly\" : true,        (bool) Only returned if imported addresses were involved in transaction\n"
            "    \"address\" : \"receivingaddress\",    (string) The receiving address\n"
            "    \"account\" : \"accountname\",         (string) DEPRECATED. The account of the receiving address. The default account is \"\".\n"
            "    \"amount\" : x.xxx,                  (numeric) The total amount in " +
            CURRENCY_UNIT + " received by the address\n"
            "    \"confirmations\" : n                (numeric) The number of confirmations of the most recent transaction included.\n"
            "                                                 If 'addlocked' is true, the number of confirmations can be less than\n"
            "                                                 configured for transactions locked via InstantSend\n"
            "    \"label\" : \"label\",               (string) A comment for the address/transaction, if any\n"
            "    \"txids\": [\n"
            "       n,                                (numeric) The ids of transactions received with the address \n"
            "       ...\n"
            "    ]\n"
            "  }\n"
            "  ,...\n"
            "]\n"

            "\nExamples:\n" +
            HelpExampleCli("listreceivedbyaddress", "") + HelpExampleCli("listreceivedbyaddress", "6 false true") + HelpExampleRpc("listreceivedbyaddress", "6, false, true, true"));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    return ListReceived(request.params, false);
}

UniValue listreceivedbyaccount(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() > 4)
        throw std::runtime_error(
            "listreceivedbyaccount ( minconf addlocked includeempty includeWatchonly)\n"
            "\nDEPRECATED. List balances by account.\n"
            "\nArguments:\n"
            "1. minconf           (numeric, optional, default=1) The minimum number of confirmations before payments are included.\n"
            "2. addlocked         (bool, optional, default=false) Whether to include transactions locked via InstantSend.\n"
            "3. includeempty      (bool, optional, default=false) Whether to include accounts that haven't received any payments.\n"
            "4. includeWatchonly  (bool, optional, default=false) Whether to include watchonly addresses (see 'importaddress').\n"

            "\nResult:\n"
            "[\n"
            "  {\n"
            "    \"involvesWatchonly\" : true,   (bool) Only returned if imported addresses were involved in transaction\n"
            "    \"account\" : \"accountname\",    (string) The account name of the receiving account\n"
            "    \"amount\" : x.xxx,             (numeric) The total amount received by addresses with this account\n"
            "    \"confirmations\" : n           (numeric) The number of blockchain confirmations of the most recent transaction included\n"
            "    \"label\" : \"label\"             (string) A comment for the address/transaction, if any\n"
            "  }\n"
            "  ,...\n"
            "]\n"

            "\nExamples:\n" +
            HelpExampleCli("listreceivedbyaccount", "") + HelpExampleCli("listreceivedbyaccount", "10 false true") + HelpExampleRpc("listreceivedbyaccount", "10, false, true, true"));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    return ListReceived(request.params, true);
}

static void MaybePushAddress(UniValue& entry, const CTxDestination& dest)
{
    CDynamicAddress addr;
    if (addr.Set(dest))
        entry.push_back(Pair("address", addr.ToString()));
}

void ListTransactions(const CWalletTx& wtx, const std::string& strAccount, int nMinDepth, bool fLong, UniValue& ret, UniValue& retAssets, const isminefilter& filter)
{
    CAmount nFee;
    std::string strSentAccount;
    std::list<COutputEntry> listReceived;
    std::list<COutputEntry> listSent;

    std::list<CAssetOutputEntry> listAssetsReceived;
    std::list<CAssetOutputEntry> listAssetsSent;
    
    wtx.GetAmounts(listReceived, listSent, nFee, strSentAccount, filter, listAssetsReceived, listAssetsSent);

    bool fAllAccounts = (strAccount == std::string("*"));
    bool involvesWatchonly = wtx.IsRelevantToMe(ISMINE_WATCH_ONLY);

    // Sent
    if ((!listSent.empty() || nFee != 0) && (fAllAccounts || strAccount == strSentAccount)) {
        for (const COutputEntry& s : listSent) {
            UniValue entry(UniValue::VOBJ);
            if (involvesWatchonly || (::IsMine(*pwalletMain, s.destination) & ISMINE_WATCH_ONLY))
                entry.push_back(Pair("involvesWatchonly", true));
            entry.push_back(Pair("account", strSentAccount));
            MaybePushAddress(entry, s.destination);
            if (s.vout >= 0) {
                std::map<std::string, std::string>::const_iterator it = wtx.mapValue.find("PS");
                entry.push_back(Pair("category", (it != wtx.mapValue.end() && it->second == "1") ? "privatesend" : "send"));
            } else {
                entry.push_back(Pair("category", "mixedsend"));
            }
            entry.push_back(Pair("amount", ValueFromAmount(-s.amount)));
            if (pwalletMain->mapAddressBook.count(s.destination))
                entry.push_back(Pair("label", pwalletMain->mapAddressBook[s.destination].name));
            if (s.vout >= 0)
                entry.push_back(Pair("vout", s.vout));
            entry.push_back(Pair("fee", ValueFromAmount(-nFee)));
            if (fLong)
                WalletTxToJSON(wtx, entry);
            entry.push_back(Pair("abandoned", wtx.isAbandoned()));
            ret.push_back(entry);
        }
    }

    // Received
    if (listReceived.size() > 0 && ((wtx.GetDepthInMainChain() >= nMinDepth) || wtx.IsLockedByInstantSend())) {
        for (const COutputEntry& r : listReceived) {
            std::string account;
            if (pwalletMain->mapAddressBook.count(r.destination))
                account = pwalletMain->mapAddressBook[r.destination].name;
            if (fAllAccounts || (account == strAccount)) {
                UniValue entry(UniValue::VOBJ);
                if (involvesWatchonly || (::IsMine(*pwalletMain, r.destination) & ISMINE_WATCH_ONLY))
                    entry.push_back(Pair("involvesWatchonly", true));
                entry.push_back(Pair("account", account));
                MaybePushAddress(entry, r.destination);
                if (wtx.IsCoinBase()) {
                    if (wtx.GetDepthInMainChain() < 1)
                        entry.push_back(Pair("category", "orphan"));
                    else if (wtx.GetBlocksToMaturity() > 0)
                        entry.push_back(Pair("category", "immature"));
                    else
                        entry.push_back(Pair("category", "generate"));
                }
                else if (wtx.IsCoinStake())
                {
                    if (wtx.GetDepthInMainChain() < 1)
                            entry.push_back(Pair("category", "stake-orphan"));
                        else if (wtx.GetBlocksToMaturity() > 0)
                            entry.push_back(Pair("category", "stake"));
                        else
                            entry.push_back(Pair("category", "stake-mint"));

                } 
                else 
                {
                    entry.push_back(Pair("category", "receive"));
                }
                entry.push_back(Pair("amount", ValueFromAmount(r.amount)));
                if (pwalletMain->mapAddressBook.count(r.destination))
                    entry.push_back(Pair("label", account));
                entry.push_back(Pair("vout", r.vout));
                if (fLong)
                    WalletTxToJSON(wtx, entry);
                ret.push_back(entry);
            }
        }
    }
    /** ASSET START */
    if (AreAssetsDeployed()) {
        if (listAssetsReceived.size() > 0 && wtx.GetDepthInMainChain() >= nMinDepth) {
            for (const CAssetOutputEntry &data : listAssetsReceived){
                UniValue entry(UniValue::VOBJ);

                if (involvesWatchonly || (::IsMine(*pwalletMain, data.destination) & ISMINE_WATCH_ONLY)) {
                    entry.push_back(Pair("involvesWatchonly", true));
                }

                entry.push_back(Pair("asset_type", GetTxnOutputType(data.type)));
                entry.push_back(Pair("asset_name", data.assetName));
                entry.push_back(Pair("amount", ValueFromAmount(data.nAmount)));
                entry.push_back(Pair("message", EncodeAssetData(data.message)));
                if (!data.message.empty() && data.expireTime > 0)
                    entry.push_back(Pair("message_expires", DateTimeStrFormat("%Y-%m-%d %H:%M:%S", data.expireTime)));
                entry.push_back(Pair("destination", EncodeDestination(data.destination)));
                entry.push_back(Pair("vout", data.vout));
                entry.push_back(Pair("category", "receive"));
                if (fLong)
                    WalletTxToJSON(wtx, entry);
                entry.push_back(Pair("abandoned", wtx.isAbandoned()));
                retAssets.push_back(entry);
            }
        }

        if ((!listAssetsSent.empty() || nFee != 0) && (fAllAccounts || strAccount == strSentAccount)) {
            for (const CAssetOutputEntry &data : listAssetsSent) {
                UniValue entry(UniValue::VOBJ);

                if (involvesWatchonly || (::IsMine(*pwalletMain, data.destination) & ISMINE_WATCH_ONLY)) {
                    entry.push_back(Pair("involvesWatchonly", true));
                }

                entry.push_back(Pair("asset_type", GetTxnOutputType(data.type)));
                entry.push_back(Pair("asset_name", data.assetName));
                entry.push_back(Pair("amount", ValueFromAmount(data.nAmount)));
                entry.push_back(Pair("message", EncodeAssetData(data.message)));
                if (!data.message.empty() && data.expireTime > 0)
                    entry.push_back(Pair("message_expires", DateTimeStrFormat("%Y-%m-%d %H:%M:%S", data.expireTime)));
                entry.push_back(Pair("destination", EncodeDestination(data.destination)));
                entry.push_back(Pair("vout", data.vout));
                entry.push_back(Pair("category", "send"));
                if (fLong)
                    WalletTxToJSON(wtx, entry);
                entry.push_back(Pair("abandoned", wtx.isAbandoned()));
                retAssets.push_back(entry);
            }
        }
    }
    /** ASSET END */
}

void ListTransactions(const CWalletTx& wtx, const std::string& strAccount, int nMinDepth, bool fLong, UniValue& ret, const isminefilter& filter)
{
    UniValue assetDetails(UniValue::VARR);

    ListTransactions(wtx, strAccount, nMinDepth, fLong, ret, assetDetails, filter);
}

void AcentryToJSON(const CAccountingEntry& acentry, const std::string& strAccount, UniValue& ret)
{
    bool fAllAccounts = (strAccount == std::string("*"));

    if (fAllAccounts || acentry.strAccount == strAccount) {
        UniValue entry(UniValue::VOBJ);
        entry.push_back(Pair("account", acentry.strAccount));
        entry.push_back(Pair("category", "move"));
        entry.push_back(Pair("time", acentry.nTime));
        entry.push_back(Pair("amount", ValueFromAmount(acentry.nCreditDebit)));
        entry.push_back(Pair("otheraccount", acentry.strOtherAccount));
        entry.push_back(Pair("comment", acentry.strComment));
        ret.push_back(entry);
    }
}

UniValue listtransactions(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp)) {
        return NullUniValue;
    }

    if (request.fHelp || request.params.size() > 4)
        throw std::runtime_error(
            "listtransactions ( \"account\" count skip include_watchonly)\n"
            "\nReturns up to 'count' most recent transactions skipping the first 'from' transactions for account 'account'.\n"
            "\nArguments:\n"
            "1. \"account\"        (string, optional) DEPRECATED. The account name. Should be \"*\".\n"
            "2. count            (numeric, optional, default=10) The number of transactions to return\n"
            "3. skip           (numeric, optional, default=0) The number of transactions to skip\n"
            "4. include_watchonly (bool, optional, default=false) Include transactions to watch-only addresses (see 'importaddress')\n"
            "\nResult:\n"
            "[\n"
            "  {\n"
            "    \"account\":\"accountname\",  (string) DEPRECATED. The account name associated with the transaction. \n"
            "                                                It will be \"\" for the default account.\n"
            "    \"address\":\"address\",    (string) The dynamic address of the transaction. Not present for \n"
            "                                                move transactions (category = move).\n"
            "    \"category\":\"send|receive|move\", (string) The transaction category. 'move' is a local (off blockchain)\n"
            "                                                transaction between accounts, and not associated with an address,\n"
            "                                                transaction id or block. 'send' and 'receive' transactions are \n"
            "                                                associated with an address, transaction id and block details\n"
            "    \"amount\": x.xxx,          (numeric) The amount in " +
            CURRENCY_UNIT + ". This is negative for the 'send' category, and for the\n"
                            "                                         'move' category for moves outbound. It is positive for the 'receive' category,\n"
                            "                                         and for the 'move' category for inbound funds.\n"
                            "    \"label\": \"label\",       (string) A comment for the address/transaction, if any\n"
                            "    \"vout\": n,                (numeric) the vout value\n"
                            "    \"fee\": x.xxx,             (numeric) The amount of the fee in " +
            CURRENCY_UNIT + ". This is negative and only available for the \n"
                            "                                         'send' category of transactions.\n"
                            "    \"instantlock\" : true|false, (bool) Current transaction lock state. Available for 'send' and 'receive' category of transactions.\n"
                            "    \"confirmations\": n,       (numeric) The number of blockchain confirmations for the transaction. Available for 'send' and \n"
                            "                                         'receive' category of transactions. Negative confirmations indicate the\n"
                            "                                         transation conflicts with the block chain\n"
                            "    \"trusted\": xxx,           (bool) Whether we consider the outputs of this unconfirmed transaction safe to spend.\n"
                            "    \"blockhash\": \"hashvalue\", (string) The block hash containing the transaction. Available for 'send' and 'receive'\n"
                            "                                          category of transactions.\n"
                            "    \"blockindex\": n,          (numeric) The index of the transaction in the block that includes it. Available for 'send' and 'receive'\n"
                            "                                          category of transactions.\n"
                            "    \"blocktime\": xxx,         (numeric) The block time in seconds since epoch (1 Jan 1970 GMT).\n"
                            "    \"txid\": \"transactionid\",  (string) The transaction id. Available for 'send' and 'receive' category of transactions.\n"
                            "    \"time\": xxx,              (numeric) The transaction time in seconds since epoch (midnight Jan 1 1970 GMT).\n"
                            "    \"timereceived\": xxx,      (numeric) The time received in seconds since epoch (midnight Jan 1 1970 GMT). Available \n"
                            "                                          for 'send' and 'receive' category of transactions.\n"
                            "    \"comment\": \"...\",         (string) If a comment is associated with the transaction.\n"
                            "    \"otheraccount\": \"accountname\",  (string) DEPRECATED. For the 'move' category of transactions, the account the funds came \n"
                            "                                          from (for receiving funds, positive amounts), or went to (for sending funds,\n"
                            "                                          negative amounts).\n"
                            "    \"bip125-replaceable\": \"yes|no|unknown\",  (string) Whether this transaction could be replaced due to BIP125 (replace-by-fee);\n"
                            "                                                     may be unknown for unconfirmed transactions not in the mempool\n"
                            "    \"abandoned\": xxx          (bool) 'true' if the transaction has been abandoned (inputs are respendable). Only available for the \n"
                            "                                         'send' category of transactions.\n"
                            "  }\n"
                            "]\n"

                            "\nExamples:\n"
                            "\nList the most recent 10 transactions in the systems\n" +
            HelpExampleCli("listtransactions", "") +
            "\nList transactions 100 to 120\n" + HelpExampleCli("listtransactions", "\"*\" 20 100") +
            "\nAs a json rpc call\n" + HelpExampleRpc("listtransactions", "\"*\", 20, 100"));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    std::string strAccount = "*";
    if (request.params.size() > 0)
        strAccount = request.params[0].get_str();
    int nCount = 10;
    if (request.params.size() > 1)
        nCount = request.params[1].get_int();
    int nFrom = 0;
    if (request.params.size() > 2)
        nFrom = request.params[2].get_int();
    isminefilter filter = ISMINE_SPENDABLE;
    if (request.params.size() > 3)
        if (request.params[3].get_bool())
            filter = filter | ISMINE_WATCH_ONLY;

    if (nCount < 0)
        throw JSONRPCError(RPC_INVALID_PARAMETER, "Negative count");
    if (nFrom < 0)
        throw JSONRPCError(RPC_INVALID_PARAMETER, "Negative from");

    UniValue ret(UniValue::VARR);

    const CWallet::TxItems& txOrdered = pwalletMain->wtxOrdered;

    // iterate backwards until we have nCount items to return:
    for (CWallet::TxItems::const_reverse_iterator it = txOrdered.rbegin(); it != txOrdered.rend(); ++it) {
        CWalletTx* const pwtx = (*it).second.first;
        if (pwtx != 0)
            ListTransactions(*pwtx, strAccount, 0, true, ret, filter);
        CAccountingEntry* const pacentry = (*it).second.second;
        if (pacentry != 0)
            AcentryToJSON(*pacentry, strAccount, ret);

        if ((int)ret.size() >= (nCount + nFrom))
            break;
    }
    // ret is newest to oldest

    if (nFrom > (int)ret.size())
        nFrom = ret.size();
    if ((nFrom + nCount) > (int)ret.size())
        nCount = ret.size() - nFrom;

    std::vector<UniValue> arrTmp = ret.getValues();

    std::vector<UniValue>::iterator first = arrTmp.begin();
    std::advance(first, nFrom);
    std::vector<UniValue>::iterator last = arrTmp.begin();
    std::advance(last, nFrom + nCount);

    if (last != arrTmp.end())
        arrTmp.erase(last, arrTmp.end());
    if (first != arrTmp.begin())
        arrTmp.erase(arrTmp.begin(), first);

    std::reverse(arrTmp.begin(), arrTmp.end()); // Return oldest to newest

    ret.clear();
    ret.setArray();
    ret.push_backV(arrTmp);

    return ret;
}

UniValue listaccounts(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() > 3)
        throw std::runtime_error(
            "listaccounts ( minconf addlocked includeWatchonly)\n"
            "\nDEPRECATED. Returns Object that has account names as keys, account balances as values.\n"
            "\nArguments:\n"
            "1. minconf           (numeric, optional, default=1) Only include transactions with at least this many confirmations\n"
            "2. addlocked           (bool, optional, default=false) Whether to include transactions locked via InstantSend.\n"
            "3. includeWatchonly  (bool, optional, default=false) Include balances in watchonly addresses (see 'importaddress')\n"
            "\nResult:\n"
            "{                    (json object where keys are account names, and values are numeric balances\n"
            "  \"account\": x.xxx,  (numeric) The property name is the account name, and the value is the total balance for the account.\n"
            "  ...\n"
            "}\n"
            "\nExamples:\n"
            "\nList account balances where there at least 1 confirmation\n" +
            HelpExampleCli("listaccounts", "") +
            "\nList account balances including zero confirmation transactions\n" + HelpExampleCli("listaccounts", "0") +
            "\nList account balances for 10 or more confirmations\n" + HelpExampleCli("listaccounts", "10") +
            "\nAs json rpc call\n" + HelpExampleRpc("listaccounts", "10"));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    int nMinDepth = 1;
    if (request.params.size() > 0)
        nMinDepth = request.params[0].get_int();
    bool fAddLocked = (request.params.size() > 1 && request.params[1].get_bool());
    isminefilter includeWatchonly = ISMINE_SPENDABLE;
    if (request.params.size() > 2)
        if (request.params[2].get_bool())
            includeWatchonly = includeWatchonly | ISMINE_WATCH_ONLY;

    std::map<std::string, CAmount> mapAccountBalances;
    for (const std::pair<CTxDestination, CAddressBookData>& entry : pwalletMain->mapAddressBook) {
        if (IsMine(*pwalletMain, entry.first) & includeWatchonly) // This address belongs to me
            mapAccountBalances[entry.second.name] = 0;
    }

    for (std::map<uint256, CWalletTx>::iterator it = pwalletMain->mapWallet.begin(); it != pwalletMain->mapWallet.end(); ++it) {
        const CWalletTx& wtx = (*it).second;
        CAmount nFee;
        std::string strSentAccount;
        std::list<COutputEntry> listReceived;
        std::list<COutputEntry> listSent;
        int nDepth = wtx.GetDepthInMainChain();
        if (wtx.GetBlocksToMaturity() > 0 || nDepth < 0)
            continue;
        wtx.GetAmounts(listReceived, listSent, nFee, strSentAccount, includeWatchonly);
        mapAccountBalances[strSentAccount] -= nFee;
        for (const COutputEntry& s : listSent)
            mapAccountBalances[strSentAccount] -= s.amount;
        if ((nDepth >= nMinDepth) || (fAddLocked && wtx.IsLockedByInstantSend())) {
            for (const COutputEntry& r : listReceived)
                if (pwalletMain->mapAddressBook.count(r.destination))
                    mapAccountBalances[pwalletMain->mapAddressBook[r.destination].name] += r.amount;
                else
                    mapAccountBalances[""] += r.amount;
        }
    }

    const std::list<CAccountingEntry>& acentries = pwalletMain->laccentries;
    for (const CAccountingEntry& entry : acentries)
        mapAccountBalances[entry.strAccount] += entry.nCreditDebit;

    UniValue ret(UniValue::VOBJ);
    for (const std::pair<std::string, CAmount>& accountBalance : mapAccountBalances) {
        ret.push_back(Pair(accountBalance.first, ValueFromAmount(accountBalance.second)));
    }
    return ret;
}

UniValue listsinceblock(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp)
        throw std::runtime_error(
            "listsinceblock ( \"blockhash\" target_confirmations include_watchonly)\n"
            "\nGet all transactions in blocks since block [blockhash], or all transactions if omitted\n"
            "\nArguments:\n"
            "1. \"blockhash\"            (string, optional) The block hash to list transactions since\n"
            "2. target_confirmations:    (numeric, optional) The confirmations required, must be 1 or more\n"
            "3. include_watchonly:       (bool, optional, default=false) Include transactions to watch-only addresses (see 'importaddress')"
            "\nResult:\n"
            "{\n"
            "  \"transactions\": [\n"
            "    \"account\":\"accountname\",  (string) DEPRECATED. The account name associated with the transaction. Will be \"\" for the default account.\n"
            "    \"address\":\"address\",    (string) The dynamic address of the transaction. Not present for move transactions (category = move).\n"
            "    \"category\":\"send|receive\",  (string) The transaction category. 'send' has negative amounts, 'receive' has positive amounts.\n"
            "    \"amount\": x.xxx,          (numeric) The amount in " +
            CURRENCY_UNIT + ". This is negative for the 'send' category, and for the 'move' category for moves \n"
                            "                                          outbound. It is positive for the 'receive' category, and for the 'move' category for inbound funds.\n"
                            "    \"vout\" : n,               (numeric) the vout value\n"
                            "    \"fee\": x.xxx,             (numeric) The amount of the fee in " +
            CURRENCY_UNIT + ". This is negative and only available for the 'send' category of transactions.\n"
                            "    \"instantlock\" : true|false, (bool) Current transaction lock state. Available for 'send' and 'receive' category of transactions.\n"
                            "    \"confirmations\" : n,      (numeric) The number of blockchain confirmations for the transaction. Available for 'send' and 'receive' category of transactions.\n"
                            "                                          When it's < 0, it means the transaction conflicted that many blocks ago.\n"
                            "    \"blockhash\": \"hashvalue\", (string) The block hash containing the transaction. Available for 'send' and 'receive' category of transactions.\n"
                            "    \"blockindex\": n,          (numeric) The index of the transaction in the block that includes it. Available for 'send' and 'receive' category of transactions.\n"
                            "    \"blocktime\": xxx,         (numeric) The block time in seconds since epoch (1 Jan 1970 GMT).\n"
                            "    \"txid\": \"transactionid\",  (string) The transaction id. Available for 'send' and 'receive' category of transactions.\n"
                            "    \"time\": xxx,              (numeric) The transaction time in seconds since epoch (Jan 1 1970 GMT).\n"
                            "    \"timereceived\": xxx,      (numeric) The time received in seconds since epoch (Jan 1 1970 GMT). Available for 'send' and 'receive' category of transactions.\n"
                            "    \"bip125-replaceable\": \"yes|no|unknown\",  (string) Whether this transaction could be replaced due to BIP125 (replace-by-fee);\n"
                            "                                                   may be unknown for unconfirmed transactions not in the mempool\n"
                            "    \"abandoned\": xxx,         (bool) 'true' if the transaction has been abandoned (inputs are respendable). Only available for the 'send' category of transactions.\n"
                            "    \"comment\": \"...\",         (string) If a comment is associated with the transaction.\n"
                            "    \"label\" : \"label\"         (string) A comment for the address/transaction, if any\n"
                            "    \"to\": \"...\",              (string) If a comment to is associated with the transaction.\n"
                            "  ],\n"
                            "  \"lastblock\": \"lastblockhash\"  (string) The hash of the last block\n"
                            "}\n"
                            "\nExamples:\n" +
            HelpExampleCli("listsinceblock", "") + HelpExampleCli("listsinceblock", "\"000000000000000bacf66f7497b7dc45ef753ee9a7d38571037cdb1a57f663ad\" 6") + HelpExampleRpc("listsinceblock", "\"000000000000000bacf66f7497b7dc45ef753ee9a7d38571037cdb1a57f663ad\", 6"));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    const CBlockIndex* pindex = nullptr;
    int target_confirms = 1;
    isminefilter filter = ISMINE_SPENDABLE;

    if (request.params.size() > 0) {
        uint256 blockId;

        blockId.SetHex(request.params[0].get_str());
        BlockMap::iterator it = mapBlockIndex.find(blockId);
        if (it != mapBlockIndex.end()) {
            pindex = it->second;
            if (chainActive[pindex->nHeight] != pindex) {
                // the block being asked for is a part of a deactivated chain;
                // we don't want to depend on its perceived height in the block
                // chain, we want to instead use the last common ancestor
                pindex = chainActive.FindFork(pindex);
            }
        } else
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid blockhash");
    }

    if (request.params.size() > 1) {
        target_confirms = request.params[1].get_int();

        if (target_confirms < 1)
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid parameter");
    }

    if (request.params.size() > 2 && request.params[2].get_bool()) {
        filter = filter | ISMINE_WATCH_ONLY;
    }

    int depth = pindex ? (1 + chainActive.Height() - pindex->nHeight) : -1;

    UniValue transactions(UniValue::VARR);

    for (std::map<uint256, CWalletTx>::iterator it = pwalletMain->mapWallet.begin(); it != pwalletMain->mapWallet.end(); it++) {
        CWalletTx tx = (*it).second;

        if (depth == -1 || tx.GetDepthInMainChain() < depth)
            ListTransactions(tx, "*", 0, true, transactions, filter);
    }

    CBlockIndex* pblockLast = chainActive[chainActive.Height() + 1 - target_confirms];
    uint256 lastblock = pblockLast ? pblockLast->GetBlockHash() : uint256();

    UniValue ret(UniValue::VOBJ);
    ret.push_back(Pair("transactions", transactions));
    ret.push_back(Pair("lastblock", lastblock.GetHex()));

    return ret;
}

UniValue gettransaction(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 1 || request.params.size() > 2)
        throw std::runtime_error(
            "gettransaction \"txid\" ( include_watchonly )\n"
            "\nGet detailed information about in-wallet transaction <txid>\n"
            "\nArguments:\n"
            "1. \"txid\"                  (string, required) The transaction id\n"
            "2. \"include_watchonly\"     (bool, optional, default=false) Whether to include watch-only addresses in balance calculation and details[]\n"
            "\nResult:\n"
            "{\n"
            "  \"amount\" : x.xxx,        (numeric) The transaction amount in " +
            CURRENCY_UNIT + "\n"
                            "  \"fee\": x.xxx,            (numeric) The amount of the fee in " +
            CURRENCY_UNIT + ". This is negative and only available for the \n"
                            "                              'send' category of transactions.\n"
                            "  \"instantlock\" : true|false, (bool) Current transaction lock state\n"
                            "  \"confirmations\" : n,     (numeric) The number of blockchain confirmations\n"
                            "  \"blockhash\" : \"hash\",    (string) The block hash\n"
                            "  \"blockindex\" : xx,       (numeric) The index of the transaction in the block that includes it\n"
                            "  \"blocktime\" : ttt,       (numeric) The time in seconds since epoch (1 Jan 1970 GMT)\n"
                            "  \"txid\" : \"transactionid\",   (string) The transaction id.\n"
                            "  \"time\" : ttt,            (numeric) The transaction time in seconds since epoch (1 Jan 1970 GMT)\n"
                            "  \"timereceived\" : ttt,    (numeric) The time received in seconds since epoch (1 Jan 1970 GMT)\n"
                            "  \"bip125-replaceable\": \"yes|no|unknown\",  (string) Whether this transaction could be replaced due to BIP125 (replace-by-fee);\n"
                            "                                                   may be unknown for unconfirmed transactions not in the mempool\n"
                            "  \"details\" : [\n"
                            "    {\n"
                            "      \"account\" : \"accountname\",      (string) DEPRECATED. The account name involved in the transaction, can be \"\" for the default account.\n"
                            "      \"address\" : \"address\",          (string) The dynamic address involved in the transaction\n"
                            "      \"category\" : \"send|receive\",    (string) The category, either 'send' or 'receive'\n"
                            "      \"amount\" : x.xxx,               (numeric) The amount in " +
            CURRENCY_UNIT + "\n"
                            "      \"label\" : \"label\",              (string) A comment for the address/transaction, if any\n"
                            "      \"vout\" : n,                     (numeric) the vout value\n"
                            "      \"fee\": x.xxx,                     (numeric) The amount of the fee in " +
            CURRENCY_UNIT + ". This is negative and only available for the \n"
                            "                                           'send' category of transactions.\n"
                            "      \"abandoned\": xxx                  (bool) 'true' if the transaction has been abandoned (inputs are respendable). Only available for the \n"
                            "                                           'send' category of transactions.\n"
                            "    }\n"
                            "    ,...\n"
                            "  ],\n"
                            "  \"hex\" : \"data\"                      (string) Raw data for transaction\n"
                            "}\n"

                            "\nExamples:\n" +
            HelpExampleCli("gettransaction", "\"1075db55d416d3ca199f55b6084e2115b9345e16c5cf302fc80e9d5fbf5d48d\"") + HelpExampleCli("gettransaction", "\"1075db55d416d3ca199f55b6084e2115b9345e16c5cf302fc80e9d5fbf5d48d\" true") + HelpExampleRpc("gettransaction", "\"1075db55d416d3ca199f55b6084e2115b9345e16c5cf302fc80e9d5fbf5d48d\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    uint256 hash;
    hash.SetHex(request.params[0].get_str());

    isminefilter filter = ISMINE_SPENDABLE;
    if (request.params.size() > 1)
        if (request.params[1].get_bool())
            filter = filter | ISMINE_WATCH_ONLY;

    UniValue entry(UniValue::VOBJ);
    if (!pwalletMain->mapWallet.count(hash))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid or non-wallet transaction id");
    const CWalletTx& wtx = pwalletMain->mapWallet[hash];

    CAmount nCredit = wtx.GetCredit(filter);
    CAmount nDebit = wtx.GetDebit(filter);
    CAmount nNet = nCredit - nDebit;
    bool fFromMe = wtx.IsFromMe(filter);
    CAmount nFee = (fFromMe ? wtx.tx->GetValueOut() - nDebit : 0);

    entry.push_back(Pair("amount", ValueFromAmount(nNet - nFee)));
    if (wtx.IsFromMe(filter))
        entry.push_back(Pair("fee", ValueFromAmount(nFee)));

    WalletTxToJSON(wtx, entry);

    UniValue details(UniValue::VARR);
    ListTransactions(wtx, "*", 0, false, details, filter);
    entry.push_back(Pair("details", details));

    std::string strHex = EncodeHexTx(static_cast<CTransaction>(wtx));
    entry.push_back(Pair("hex", strHex));

    return entry;
}

UniValue abandontransaction(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "abandontransaction \"txid\"\n"
            "\nMark in-wallet transaction <txid> as abandoned\n"
            "This will mark this transaction and all its in-wallet descendants as abandoned which will allow\n"
            "for their inputs to be respent.  It can be used to replace \"stuck\" or evicted transactions.\n"
            "It only works on transactions which are not included in a block and are not currently in the mempool.\n"
            "It has no effect on transactions which are already conflicted or abandoned.\n"
            "\nArguments:\n"
            "1. \"txid\"    (string, required) The transaction id\n"
            "\nResult:\n"
            "\nExamples:\n" +
            HelpExampleCli("abandontransaction", "\"1075db55d416d3ca199f55b6084e2115b9345e16c5cf302fc80e9d5fbf5d48d\"") + HelpExampleRpc("abandontransaction", "\"1075db55d416d3ca199f55b6084e2115b9345e16c5cf302fc80e9d5fbf5d48d\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    uint256 hash;
    hash.SetHex(request.params[0].get_str());

    if (!pwalletMain->mapWallet.count(hash))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid or non-wallet transaction id");
    if (!pwalletMain->AbandonTransaction(hash))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Transaction not eligible for abandonment");

    return NullUniValue;
}


UniValue backupwallet(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "backupwallet \"destination\"\n"
            "\nSafely copies current wallet file to destination, which can be a directory or a path with filename.\n"
            "\nArguments:\n"
            "1. \"destination\"   (string) The destination directory or file\n"
            "\nExamples:\n" +
            HelpExampleCli("backupwallet", "\"backup.dat\"") + HelpExampleRpc("backupwallet", "\"backup.dat\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    std::string strDest = request.params[0].get_str();
    if (!pwalletMain->BackupWallet(strDest))
        throw JSONRPCError(RPC_WALLET_ERROR, "Error: Wallet backup failed!");

    return NullUniValue;
}


UniValue keypoolrefill(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() > 1)
        throw std::runtime_error(
            "keypoolrefill ( newsize )\n"
            "\nFills the keypool." +
            HelpRequiringPassphrase() + "\n"
                                        "\nArguments\n"
                                        "1. newsize     (numeric, optional, default=" +
            itostr(DEFAULT_KEYPOOL_SIZE) + ") The new keypool size\n"
                                           "\nExamples:\n" +
            HelpExampleCli("keypoolrefill", "") + HelpExampleRpc("keypoolrefill", ""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    // 0 is interpreted by TopUpKeyPool() as the default keypool size given by -keypool
    unsigned int kpSize = 0;
    if (request.params.size() > 0) {
        if (request.params[0].get_int() < 0)
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid parameter, expected valid size.");
        kpSize = (unsigned int)request.params[0].get_int();
    }

    //Check to see if wallet needs upgrading
    if(pwalletMain->WalletNeedsUpgrading())
        throw JSONRPCError(RPC_WALLET_NEEDS_UPGRADING, "Error: Your wallet has not been fully upgraded to version 2.4.  Please unlock your wallet to continue.");
        
    EnsureWalletIsUnlocked();
    pwalletMain->TopUpKeyPoolCombo(kpSize);

    if (pwalletMain->GetKeyPoolSize() < (pwalletMain->IsHDEnabled() ? kpSize * 2 : kpSize))
        throw JSONRPCError(RPC_WALLET_ERROR, "Error refreshing keypool.");

    return NullUniValue;
}


static void LockWallet(CWallet* pWallet)
{
    LOCK(cs_nWalletUnlockTime);
    nWalletUnlockTime = 0;
    pWallet->Lock();
}

UniValue walletpassphrase(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (pwalletMain->IsCrypted() && (request.fHelp || request.params.size() < 2 || request.params.size() > 3))
        throw std::runtime_error(
            "walletpassphrase \"passphrase\" timeout ( mixingonly )\n"
            "\nStores the wallet decryption key in memory for 'timeout' seconds.\n"
            "This is needed prior to performing transactions related to private keys such as sending Dynamic\n"
            "\nArguments:\n"
            "1. \"passphrase\"        (string, required) The wallet passphrase\n"
            "2. timeout             (numeric, required) The time to keep the decryption key in seconds.\n"
            "3. mixingonly          (boolean, optional, default=false) If is true sending functions are disabled."
            "\nNote:\n"
            "Issuing the walletpassphrase command while the wallet is already unlocked will set a new unlock\n"
            "time that overrides the old one.\n"
            "\nExamples:\n"
            "\nUnlock the wallet for 60 seconds\n" +
            HelpExampleCli("walletpassphrase", "\"my pass phrase\" 60") +
            "\nUnlock the wallet for 60 seconds but allow PrivateSend mixing only\n" + HelpExampleCli("walletpassphrase", "\"my pass phrase\" 60 true") +
            "\nLock the wallet again (before 60 seconds)\n" + HelpExampleCli("walletlock", "") +
            "\nAs json rpc call\n" + HelpExampleRpc("walletpassphrase", "\"my pass phrase\", 60"));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    if (request.fHelp)
        return true;
    if (!pwalletMain->IsCrypted())
        throw JSONRPCError(RPC_WALLET_WRONG_ENC_STATE, "Error: running with an unencrypted wallet, but walletpassphrase was called.");

    // Note that the walletpassphrase is stored in request.params[0] which is not mlock()ed
    SecureString strWalletPass;
    strWalletPass.reserve(100);
    // TODO: get rid of this .c_str() by implementing SecureString::operator=(std::string)
    // Alternately, find a way to make request.params[0] mlock()'d to begin with.
    strWalletPass = request.params[0].get_str().c_str();

    int64_t nSleepTime = request.params[1].get_int64();
    bool fForMixingOnly = false;

    if (request.params.size() > 2)
        fWalletUnlockMixStakeOnly = request.params[2].get_bool();
    else
        fWalletUnlockMixStakeOnly = false;

    if (fWalletUnlockMixStakeOnly && !pwalletMain->IsLocked(true) && pwalletMain->IsLocked())
        throw JSONRPCError(RPC_WALLET_ALREADY_UNLOCKED, "Error: Wallet is already unlocked for staking and mixing only.");

    if (!pwalletMain->IsLocked())
        throw JSONRPCError(RPC_WALLET_ALREADY_UNLOCKED, "Error: Wallet is already fully unlocked.");

    if (!pwalletMain->Unlock(strWalletPass, fForMixingOnly))
        throw JSONRPCError(RPC_WALLET_PASSPHRASE_INCORRECT, "Error: The wallet passphrase entered was incorrect.");

    pwalletMain->TopUpKeyPoolCombo(); //TopUpKeyPool();

    LOCK(cs_nWalletUnlockTime);
    nWalletUnlockTime = GetTime() + nSleepTime;
    RPCRunLater("lockwallet", boost::bind(LockWallet, pwalletMain), nSleepTime);

    UniValue ret(UniValue::VOBJ);
    ret.push_back(Pair("mixstakeonly", fForMixingOnly));

    return NullUniValue;
}

// Used only in one place - WalletModel::setWalletLocked.
void relockWalletAfterDuration(CWallet *wallet, int64_t nSeconds)
{
    wallet->TopUpKeyPoolCombo(); //TopUpKeyPool();

    LOCK(cs_nWalletUnlockTime);
    nWalletUnlockTime = GetTime() + nSeconds;
    RPCRunLater("lockwallet", boost::bind(LockWallet, wallet), nSeconds);
}

UniValue walletpassphrasechange(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (pwalletMain->IsCrypted() && (request.fHelp || request.params.size() != 2))
        throw std::runtime_error(
            "walletpassphrasechange \"oldpassphrase\" \"newpassphrase\"\n"
            "\nChanges the wallet passphrase from 'oldpassphrase' to 'newpassphrase'.\n"
            "\nArguments:\n"
            "1. \"oldpassphrase\"      (string) The current passphrase\n"
            "2. \"newpassphrase\"      (string) The new passphrase\n"
            "\nExamples:\n" +
            HelpExampleCli("walletpassphrasechange", "\"old one\" \"new one\"") + HelpExampleRpc("walletpassphrasechange", "\"old one\", \"new one\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    if (request.fHelp)
        return true;
    if (!pwalletMain->IsCrypted())
        throw JSONRPCError(RPC_WALLET_WRONG_ENC_STATE, "Error: running with an unencrypted wallet, but walletpassphrasechange was called.");

    // TODO: get rid of these .c_str() calls by implementing SecureString::operator=(std::string)
    // Alternately, find a way to make request.params[0] mlock()'d to begin with.
    SecureString strOldWalletPass;
    strOldWalletPass.reserve(100);
    strOldWalletPass = request.params[0].get_str().c_str();

    SecureString strNewWalletPass;
    strNewWalletPass.reserve(100);
    strNewWalletPass = request.params[1].get_str().c_str();

    if (strOldWalletPass.length() < 1 || strNewWalletPass.length() < 1)
        throw std::runtime_error(
            "walletpassphrasechange <oldpassphrase> <newpassphrase>\n"
            "Changes the wallet passphrase from <oldpassphrase> to <newpassphrase>.");

    if (!pwalletMain->ChangeWalletPassphrase(strOldWalletPass, strNewWalletPass))
        throw JSONRPCError(RPC_WALLET_PASSPHRASE_INCORRECT, "Error: The wallet passphrase entered was incorrect.");

    return NullUniValue;
}


UniValue walletlock(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (pwalletMain->IsCrypted() && (request.fHelp || request.params.size() != 0))
        throw std::runtime_error(
            "walletlock\n"
            "\nRemoves the wallet encryption key from memory, locking the wallet.\n"
            "After calling this method, you will need to call walletpassphrase again\n"
            "before being able to call any methods which require the wallet to be unlocked.\n"
            "\nExamples:\n"
            "\nSet the passphrase for 2 minutes to perform a transaction\n" +
            HelpExampleCli("walletpassphrase", "\"my pass phrase\" 120") +
            "\nPerform a send (requires passphrase set)\n" + HelpExampleCli("sendtoaddress", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" 1.0") +
            "\nClear the passphrase since we are done before 2 minutes is up\n" + HelpExampleCli("walletlock", "") +
            "\nAs json rpc call\n" + HelpExampleRpc("walletlock", ""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    if (request.fHelp)
        return true;
    if (!pwalletMain->IsCrypted())
        throw JSONRPCError(RPC_WALLET_WRONG_ENC_STATE, "Error: running with an unencrypted wallet, but walletlock was called.");

    {
        LOCK(cs_nWalletUnlockTime);
        pwalletMain->Lock();
        nWalletUnlockTime = 0;
    }

    return NullUniValue;
}


UniValue encryptwallet(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (!pwalletMain->IsCrypted() && (request.fHelp || request.params.size() != 1))
        throw std::runtime_error(
            "encryptwallet \"passphrase\"\n"
            "\nEncrypts the wallet with 'passphrase'. This is for first time encryption.\n"
            "After this, any calls that interact with private keys such as sending or signing \n"
            "will require the passphrase to be set prior the making these calls.\n"
            "Use the walletpassphrase call for this, and then walletlock call.\n"
            "If the wallet is already encrypted, use the walletpassphrasechange call.\n"
            "Note that this will shutdown the server.\n"
            "\nArguments:\n"
            "1. \"passphrase\"    (string) The pass phrase to encrypt the wallet with. It must be at least 1 character, but should be long.\n"
            "\nExamples:\n"
            "\nEncrypt you wallet\n" +
            HelpExampleCli("encryptwallet", "\"my pass phrase\"") +
            "\nNow set the passphrase to use the wallet, such as for signing or sending dynamic\n" + HelpExampleCli("walletpassphrase", "\"my pass phrase\"") +
            "\nNow we can so something like sign\n" + HelpExampleCli("signmessage", "\"dynamicaddress\" \"test message\"") +
            "\nNow lock the wallet again by removing the passphrase\n" + HelpExampleCli("walletlock", "") +
            "\nAs a json rpc call\n" + HelpExampleRpc("encryptwallet", "\"my pass phrase\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    if (request.fHelp)
        return true;
    if (pwalletMain->IsCrypted())
        throw JSONRPCError(RPC_WALLET_WRONG_ENC_STATE, "Error: running with an encrypted wallet, but encryptwallet was called.");

    // TODO: get rid of this .c_str() by implementing SecureString::operator=(std::string)
    // Alternately, find a way to make request.params[0] mlock()'d to begin with.
    SecureString strWalletPass;
    strWalletPass.reserve(100);
    strWalletPass = request.params[0].get_str().c_str();

    if (strWalletPass.length() < 1)
        throw std::runtime_error(
            "encryptwallet <passphrase>\n"
            "Encrypts the wallet with <passphrase>.");

    if (!pwalletMain->EncryptWallet(strWalletPass))
        throw JSONRPCError(RPC_WALLET_ENCRYPTION_FAILED, "Error: Failed to encrypt the wallet.");

    // BDB seems to have a bad habit of writing old data into
    // slack space in .dat files; that is bad if the old data is
    // unencrypted private keys. So:
    StartShutdown();
    return "Wallet encrypted; Dynamic server stopping, restart to run with encrypted wallet. The keypool has been flushed and a new HD seed was generated (if you are using HD). You need to make a new backup.";
}

UniValue lockunspent(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 1 || request.params.size() > 2)
        throw std::runtime_error(
            "lockunspent unlock ([{\"txid\":\"txid\",\"vout\":n},...])\n"
            "\nUpdates list of temporarily unspendable outputs.\n"
            "Temporarily lock (unlock=false) or unlock (unlock=true) specified transaction outputs.\n"
            "If no transaction outputs are specified when unlocking then all current locked transaction outputs are unlocked.\n"
            "A locked transaction output will not be chosen by automatic coin selection, when spending Dynamic.\n"
            "Locks are stored in memory only. Nodes start with zero locked outputs, and the locked output list\n"
            "is always cleared (by virtue of process exit) when a node stops or fails.\n"
            "Also see the listunspent call\n"
            "\nArguments:\n"
            "1. unlock            (boolean, required) Whether to unlock (true) or lock (false) the specified transactions\n"
            "2. \"transactions\"  (string, optional) A json array of objects. Each object the txid (string) vout (numeric)\n"
            "     [           (json array of json objects)\n"
            "       {\n"
            "         \"txid\":\"id\",    (string) The transaction id\n"
            "         \"vout\": n         (numeric) The output number\n"
            "       }\n"
            "       ,...\n"
            "     ]\n"

            "\nResult:\n"
            "true|false    (boolean) Whether the command was successful or not\n"

            "\nExamples:\n"
            "\nList the unspent transactions\n" +
            HelpExampleCli("listunspent", "") +
            "\nLock an unspent transaction\n" + HelpExampleCli("lockunspent", "false \"[{\\\"txid\\\":\\\"a08e6907dbbd3d809776dbfc5d82e371b764ed838b5655e72f463568df1aadf0\\\",\\\"vout\\\":1}]\"") +
            "\nList the locked transactions\n" + HelpExampleCli("listlockunspent", "") +
            "\nUnlock the transaction again\n" + HelpExampleCli("lockunspent", "true \"[{\\\"txid\\\":\\\"a08e6907dbbd3d809776dbfc5d82e371b764ed838b5655e72f463568df1aadf0\\\",\\\"vout\\\":1}]\"") +
            "\nAs a json rpc call\n" + HelpExampleRpc("lockunspent", "false, \"[{\\\"txid\\\":\\\"a08e6907dbbd3d809776dbfc5d82e371b764ed838b5655e72f463568df1aadf0\\\",\\\"vout\\\":1}]\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    if (request.params.size() == 1)
        RPCTypeCheck(request.params, boost::assign::list_of(UniValue::VBOOL));
    else
        RPCTypeCheck(request.params, boost::assign::list_of(UniValue::VBOOL)(UniValue::VARR));

    bool fUnlock = request.params[0].get_bool();

    if (request.params.size() == 1) {
        if (fUnlock)
            pwalletMain->UnlockAllCoins();
        return true;
    }

    UniValue outputs = request.params[1].get_array();
    for (unsigned int idx = 0; idx < outputs.size(); idx++) {
        const UniValue& output = outputs[idx];
        if (!output.isObject())
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid parameter, expected object");
        const UniValue& o = output.get_obj();

        RPCTypeCheckObj(o,
            {
                {"txid", UniValueType(UniValue::VSTR)},
                {"vout", UniValueType(UniValue::VNUM)},
            });

        std::string txid = find_value(o, "txid").get_str();
        if (!IsHex(txid))
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid parameter, expected hex txid");

        int nOutput = find_value(o, "vout").get_int();
        if (nOutput < 0)
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid parameter, vout must be positive");

        COutPoint outpt(uint256S(txid), nOutput);

        if (fUnlock)
            pwalletMain->UnlockCoin(outpt);
        else
            pwalletMain->LockCoin(outpt);
    }

    return true;
}

UniValue listlockunspent(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() > 0)
        throw std::runtime_error(
            "listlockunspent\n"
            "\nReturns list of temporarily unspendable outputs.\n"
            "See the lockunspent call to lock and unlock transactions for spending.\n"
            "\nResult:\n"
            "[\n"
            "  {\n"
            "    \"txid\" : \"transactionid\",     (string) The transaction id locked\n"
            "    \"vout\" : n                      (numeric) The vout value\n"
            "  }\n"
            "  ,...\n"
            "]\n"
            "\nExamples:\n"
            "\nList the unspent transactions\n" +
            HelpExampleCli("listunspent", "") +
            "\nLock an unspent transaction\n" + HelpExampleCli("lockunspent", "false \"[{\\\"txid\\\":\\\"a08e6907dbbd3d809776dbfc5d82e371b764ed838b5655e72f463568df1aadf0\\\",\\\"vout\\\":1}]\"") +
            "\nList the locked transactions\n" + HelpExampleCli("listlockunspent", "") +
            "\nUnlock the transaction again\n" + HelpExampleCli("lockunspent", "true \"[{\\\"txid\\\":\\\"a08e6907dbbd3d809776dbfc5d82e371b764ed838b5655e72f463568df1aadf0\\\",\\\"vout\\\":1}]\"") +
            "\nAs a json rpc call\n" + HelpExampleRpc("listlockunspent", ""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    std::vector<COutPoint> vOutpts;
    pwalletMain->ListLockedCoins(vOutpts);

    UniValue ret(UniValue::VARR);

    for (COutPoint& outpt : vOutpts) {
        UniValue o(UniValue::VOBJ);

        o.push_back(Pair("txid", outpt.hash.GetHex()));
        o.push_back(Pair("vout", (int)outpt.n));
        ret.push_back(o);
    }

    return ret;
}

UniValue settxfee(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 1 || request.params.size() > 1)
        throw std::runtime_error(
            "settxfee amount\n"
            "\nSet the transaction fee per kB. Overwrites the paytxfee parameter.\n"
            "\nArguments:\n"
            "1. amount         (numeric or sting, required) The transaction fee in " +
            CURRENCY_UNIT + "/kB\n"
                            "\nResult\n"
                            "true|false        (boolean) Returns true if successful\n"
                            "\nExamples:\n" +
            HelpExampleCli("settxfee", "0.00001") + HelpExampleRpc("settxfee", "0.00001"));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    // Amount
    CAmount nAmount = AmountFromValue(request.params[0]);

    payTxFee = CFeeRate(nAmount, 1000);
    return true;
}

UniValue setprivatesendrounds(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;
    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "setprivatesendrounds rounds\n"
            "\nSet the number of rounds for PrivateSend mixing.\n"
            "\nArguments:\n"
            "1. rounds         (numeric, required) The default number of rounds is " +
            std::to_string(DEFAULT_PRIVATESEND_ROUNDS) +
            " Cannot be more than " + std::to_string(MAX_PRIVATESEND_ROUNDS) + " nor less than " + std::to_string(MIN_PRIVATESEND_ROUNDS) +
            "\nExamples:\n" + HelpExampleCli("setprivatesendrounds", "4") + HelpExampleRpc("setprivatesendrounds", "16"));
    int nRounds = request.params[0].get_int();
    if (nRounds > MAX_PRIVATESEND_ROUNDS || nRounds < MIN_PRIVATESEND_ROUNDS)
        throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid number of rounds");
    privateSendClient.nPrivateSendRounds = nRounds;
    return NullUniValue;
}
UniValue setprivatesendamount(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;
    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "setprivatesendamount amount\n"
            "\nSet the goal amount in " +
            CURRENCY_UNIT + " for PrivateSend mixing.\n"
                            "\nArguments:\n"
                            "1. amount         (numeric, required) The default amount is " +
            std::to_string(DEFAULT_PRIVATESEND_AMOUNT) +
            " Cannot be more than " + std::to_string(MAX_PRIVATESEND_AMOUNT) + " nor less than " + std::to_string(MIN_PRIVATESEND_AMOUNT) +
            "\nExamples:\n" + HelpExampleCli("setprivatesendamount", "500") + HelpExampleRpc("setprivatesendamount", "208"));
    int nAmount = request.params[0].get_int();
    if (nAmount > MAX_PRIVATESEND_AMOUNT || nAmount < MIN_PRIVATESEND_AMOUNT)
        throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid amount of " + CURRENCY_UNIT + " as mixing goal amount");
    privateSendClient.nPrivateSendAmount = nAmount;
    return NullUniValue;
}

UniValue getwalletinfo(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() != 0)
        throw std::runtime_error(
            "getwalletinfo\n"
            "Returns an object containing various wallet state info.\n"
            "\nResult:\n"
            "{\n"
            "  \"walletversion\": xxxxx,     (numeric) the wallet version\n"
            "  \"balance\": xxxxxxx,         (numeric) the total confirmed balance of the wallet in " +
            CURRENCY_UNIT + "\n" + (!fLiteMode ? "  \"privatesend_balance\": xxxxxx, (numeric) the anonymized Dynamic balance of the wallet in " + CURRENCY_UNIT + "\n" : "") +
            "  \"unconfirmed_balance\": xxx, (numeric) the total unconfirmed balance of the wallet in " + CURRENCY_UNIT + "\n"
                                                                                                                          "  \"immature_balance\": xxxxxx, (numeric) the total immature balance of the wallet in " +
            CURRENCY_UNIT + "\n"
                            "  \"txcount\": xxxxxxx,         (numeric) the total number of transactions in the wallet\n"
                            "  \"keypoololdest\": xxxxxx,    (numeric) the timestamp (seconds since Unix epoch) of the oldest pre-generated key in the key pool\n"
                            "  \"keypoolsize\": xxxx,        (numeric) how many new keys are pre-generated (only counts external keys)\n"
                            "  \"keypoolsize_hd_internal\": xxxx, (numeric) how many new keys are pre-generated for internal use (used for change outputs, only appears if the wallet is using this feature, otherwise external keys are used)\n"
                            "  \"keys_left\": xxxx,          (numeric) how many new keys are left since last automatic backup\n"
                            "  \"unlocked_until\": ttt,      (numeric) the timestamp in seconds since epoch (midnight Jan 1 1970 GMT) that the wallet is unlocked for transfers, or 0 if the wallet is locked\n"
                            "  \"paytxfee\": x.xxxx,         (numeric) the transaction fee configuration, set in " +
            CURRENCY_UNIT + "/kB\n"
                            "  \"hdchainid\": \"<hash>\",      (string) the ID of the HD chain\n"
                            "  \"hdaccountcount\": xxx,      (numeric) how many accounts of the HD chain are in this wallet\n"
                            "    [\n"
                            "      {\n"
                            "      \"hdaccountindex\": xxx,         (numeric) the index of the account\n"
                            "      \"hdexternalkeyindex\": xxxx,    (numeric) current external childkey index\n"
                            "      \"hdinternalkeyindex\": xxxx,    (numeric) current internal childkey index\n"
                            "      }\n"
                            "      ,...\n"
                            "    ]\n"
                            "}\n"
                            "\nExamples:\n" +
            HelpExampleCli("getwalletinfo", "") + HelpExampleRpc("getwalletinfo", ""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    CHDChain hdChainCurrent;
    bool fHDEnabled = pwalletMain->GetHDChain(hdChainCurrent);
    UniValue obj(UniValue::VOBJ);
    obj.push_back(Pair("walletversion", pwalletMain->GetVersion()));
    obj.push_back(Pair("balance", ValueFromAmount(pwalletMain->GetBalance())));
    if (!fLiteMode)
        obj.push_back(Pair("privatesend_balance", ValueFromAmount(pwalletMain->GetAnonymizedBalance())));
    obj.push_back(Pair("unconfirmed_balance", ValueFromAmount(pwalletMain->GetUnconfirmedBalance())));
    obj.push_back(Pair("immature_balance", ValueFromAmount(pwalletMain->GetImmatureBalance())));
    obj.push_back(Pair("txcount", (int)pwalletMain->mapWallet.size()));
    obj.push_back(Pair("keypoololdest", pwalletMain->GetOldestKeyPoolTime()));
    obj.push_back(Pair("keypoolsize", (int64_t)pwalletMain->KeypoolCountExternalKeys()));
    if (fHDEnabled) {
        obj.push_back(Pair("keypoolsize_hd_internal", (int64_t)(pwalletMain->KeypoolCountInternalKeys())));
    }
    obj.push_back(Pair("keys_left", pwalletMain->nKeysLeftSinceAutoBackup));
    if (pwalletMain->IsCrypted())
        obj.push_back(Pair("unlocked_until", nWalletUnlockTime));
    obj.push_back(Pair("paytxfee", ValueFromAmount(payTxFee.GetFeePerK())));
    if (fHDEnabled) {
        obj.push_back(Pair("hdchainid", hdChainCurrent.GetID().GetHex()));
        obj.push_back(Pair("hdaccountcount", (int64_t)hdChainCurrent.CountAccounts()));
        UniValue accounts(UniValue::VARR);
        for (size_t i = 0; i < hdChainCurrent.CountAccounts(); ++i) {
            CHDAccount acc;
            UniValue account(UniValue::VOBJ);
            account.push_back(Pair("hdaccountindex", (int64_t)i));
            if (hdChainCurrent.GetAccount(i, acc)) {
                account.push_back(Pair("hdexternalkeyindex", (int64_t)acc.nExternalChainCounter));
                account.push_back(Pair("hdinternalkeyindex", (int64_t)acc.nInternalChainCounter));
            } else {
                account.push_back(Pair("error", strprintf("account %d is missing", i)));
            }
            accounts.push_back(account);
        }
        obj.push_back(Pair("hdaccounts", accounts));
    }
    return obj;
}

// ppcoin: reserve balance from being staked for network protection
UniValue reservebalance(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() > 2)
        throw std::runtime_error(
            "reservebalance ( reserve amount )\n"
            "\nShow or set the reserve amount not participating in network protection\n"
            "If no parameters provided current setting is printed.\n"

            "\nArguments:\n"
            "1. reserve     (boolean, optional) is true or false to turn balance reserve on or off.\n"
            "2. amount      (numeric, optional) is a real and rounded to cent.\n"

            "\nResult:\n"
            "{\n"
            "  \"reserve\": true|false,     (boolean) Status of the reserve balance\n"
            "  \"amount\": x.xxxx       (numeric) Amount reserved\n"
            "}\n"

            "\nExamples:\n" +
            HelpExampleCli("reservebalance", "true 5000") + HelpExampleRpc("reservebalance", "true 5000"));

    if (request.params.size() > 0) {
        bool fReserve = request.params[0].get_bool();
        if (fReserve) {
            if (request.params.size() == 1)
                throw std::runtime_error("must provide amount to reserve balance.\n");
            CAmount nAmount = AmountFromValue(request.params[1]);
            nAmount = (nAmount / CENT) * CENT; // round to cent
            if (nAmount < 0)
                throw std::runtime_error("amount cannot be negative.\n");
            nReserveBalance = nAmount;
        } else {
            if (request.params.size() > 1)
                throw std::runtime_error("cannot specify amount to turn off reserve.\n");
            nReserveBalance = 0;
        }
    }

    UniValue result(UniValue::VOBJ);
    result.push_back(Pair("reserve", (nReserveBalance > 0)));
    result.push_back(Pair("amount", ValueFromAmount(nReserveBalance)));
    return result;
}

// presstab HyperStake
UniValue setstakesplitthreshold(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "setstakesplitthreshold value\n"
            "\nThis will set the output size of your stakes to never be below this number\n" +
            HelpRequiringPassphrase() + "\n"

            "\nArguments:\n"
            "1. value   (numeric, required) Threshold value between 1 and 999999\n"

            "\nResult:\n"
            "{\n"
            "  \"threshold\": n,    (numeric) Threshold value set\n"
            "  \"saved\": true|false    (boolean) 'true' if successfully saved to the wallet file\n"
            "}\n"

            "\nExamples:\n" +
            HelpExampleCli("setstakesplitthreshold", "5000") + HelpExampleRpc("setstakesplitthreshold", "5000"));

    EnsureWalletIsUnlocked();

    uint64_t nStakeSplitThreshold = request.params[0].get_int();

    if (nStakeSplitThreshold > 999999)
        throw std::runtime_error("Value out of range, max allowed is 999999");

    CWalletDB walletdb(pwalletMain->strWalletFile);
    LOCK(pwalletMain->cs_wallet);
    {
        bool fFileBacked = pwalletMain->fFileBacked;

        UniValue result(UniValue::VOBJ);
        pwalletMain->nStakeSplitThreshold = nStakeSplitThreshold;
        result.push_back(Pair("threshold", int(pwalletMain->nStakeSplitThreshold)));
        if (fFileBacked) {
            walletdb.WriteStakeSplitThreshold(nStakeSplitThreshold);
            result.push_back(Pair("saved", "true"));
        } else
            result.push_back(Pair("saved", "false"));

        return result;
    }
}

// presstab HyperStake
UniValue getstakesplitthreshold(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 0)
        throw std::runtime_error(
            "getstakesplitthreshold\n"
            "Returns the threshold for stake splitting\n"

            "\nResult:\n"
            "n      (numeric) Threshold value\n"

            "\nExamples:\n" +
            HelpExampleCli("getstakesplitthreshold", "") + HelpExampleRpc("getstakesplitthreshold", ""));

    return int(pwalletMain->nStakeSplitThreshold);
}

UniValue autocombinerewards(const UniValue& params, bool fHelp)
{
    bool fEnable;
    if (params.size() >= 1)
        fEnable = params[0].get_bool();

    if (fHelp || params.size() < 1 || (fEnable && params.size() != 2) || params.size() > 2)
        throw std::runtime_error(
            "autocombinerewards enable ( threshold )\n"
            "\nWallet will automatically monitor for any coins with value below the threshold amount, and combine them if they reside with the same Dynamic address\n"
            "When autocombinerewards runs it will create a transaction, and therefore will be subject to transaction fees.\n"

            "\nArguments:\n"
            "1. enable          (boolean, required) Enable auto combine (true) or disable (false)\n"
            "2. threshold       (numeric, optional) Threshold amount (default: 0)\n"

            "\nExamples:\n" +
            HelpExampleCli("autocombinerewards", "true 500") + HelpExampleRpc("autocombinerewards", "true 500"));

    CWalletDB walletdb(pwalletMain->strWalletFile);
    CAmount nThreshold = 0;

    if (fEnable)
        nThreshold = params[1].get_int();

    pwalletMain->fCombineDust = fEnable;
    pwalletMain->nAutoCombineThreshold = nThreshold;

    if (!walletdb.WriteAutoCombineSettings(fEnable, nThreshold))
        throw std::runtime_error("Changed settings in wallet but failed to save to database\n");

    return NullUniValue;
}

UniValue printMultiSend()
{
    UniValue ret(UniValue::VARR);
    UniValue act(UniValue::VOBJ);
    act.push_back(Pair("MultiSendStake Activated?", pwalletMain->fMultiSendStake));
    act.push_back(Pair("MultiSendDynode Activated?", pwalletMain->fMultiSendDynodeReward));
    ret.push_back(act);

    if (pwalletMain->vDisabledAddresses.size() >= 1) {
        UniValue disAdd(UniValue::VOBJ);
        for (unsigned int i = 0; i < pwalletMain->vDisabledAddresses.size(); i++) {
            disAdd.push_back(Pair("Disabled From Sending", pwalletMain->vDisabledAddresses[i]));
        }
        ret.push_back(disAdd);
    }

    ret.push_back("MultiSend Addresses to Send To:");

    UniValue vMS(UniValue::VOBJ);
    for (unsigned int i = 0; i < pwalletMain->vMultiSend.size(); i++) {
        vMS.push_back(Pair("Address " + std::to_string(i), pwalletMain->vMultiSend[i].first));
        vMS.push_back(Pair("Percent", pwalletMain->vMultiSend[i].second));
    }

    ret.push_back(vMS);
    return ret;
}

UniValue printAddresses()
{
    std::vector<COutput> vCoins;
    pwalletMain->AvailableCoins(vCoins);
    std::map<std::string, double> mapAddresses;
    for (const COutput& out : vCoins) {
        CTxDestination utxoAddress;
        ExtractDestination(out.tx->tx->vout[out.i].scriptPubKey, utxoAddress);
        std::string strAdd = CDynamicAddress(utxoAddress).ToString();

        if (mapAddresses.find(strAdd) == mapAddresses.end()) //if strAdd is not already part of the map
            mapAddresses[strAdd] = (double)out.tx->tx->vout[out.i].nValue / (double)COIN;
        else
            mapAddresses[strAdd] += (double)out.tx->tx->vout[out.i].nValue / (double)COIN;
    }

    UniValue ret(UniValue::VARR);
    for (std::map<std::string, double>::const_iterator it = mapAddresses.begin(); it != mapAddresses.end(); ++it) {
        UniValue obj(UniValue::VOBJ);
        const std::string* strAdd = &(*it).first;
        const double* nBalance = &(*it).second;
        obj.push_back(Pair("Address ", *strAdd));
        obj.push_back(Pair("Balance ", *nBalance));
        ret.push_back(obj);
    }

    return ret;
}

unsigned int sumMultiSend()
{
    unsigned int sum = 0;
    for (unsigned int i = 0; i < pwalletMain->vMultiSend.size(); i++)
        sum += pwalletMain->vMultiSend[i].second;
    return sum;
}

UniValue multisend(const UniValue& params, bool fHelp)
{
    CWalletDB walletdb(pwalletMain->strWalletFile);
    bool fFileBacked;
    //MultiSend Commands
    if (params.size() == 1) {
        std::string strCommand = params[0].get_str();
        UniValue ret(UniValue::VOBJ);
        if (strCommand == "print") {
            return printMultiSend();
        } else if (strCommand == "printaddress" || strCommand == "printaddresses") {
            return printAddresses();
        } else if (strCommand == "clear") {
            LOCK(pwalletMain->cs_wallet);
            {
                bool erased = false;
                if (pwalletMain->fFileBacked) {
                    if (walletdb.EraseMultiSend(pwalletMain->vMultiSend))
                        erased = true;
                }

                pwalletMain->vMultiSend.clear();
                pwalletMain->setMultiSendDisabled();

                UniValue obj(UniValue::VOBJ);
                obj.push_back(Pair("Erased from database", erased));
                obj.push_back(Pair("Erased from RAM", true));

                return obj;
            }
        } else if (strCommand == "enablestake" || strCommand == "activatestake") {
            if (pwalletMain->vMultiSend.size() < 1)
                throw JSONRPCError(RPC_INVALID_REQUEST, "Unable to activate MultiSend, check MultiSend vector");

            if (CDynamicAddress(pwalletMain->vMultiSend[0].first).IsValid()) {
                pwalletMain->fMultiSendStake = true;
                if (!walletdb.WriteMSettings(true, pwalletMain->fMultiSendDynodeReward, pwalletMain->nLastMultiSendHeight)) {
                    UniValue obj(UniValue::VOBJ);
                    obj.push_back(Pair("error", "MultiSend activated but writing settings to DB failed"));
                    UniValue arr(UniValue::VARR);
                    arr.push_back(obj);
                    arr.push_back(printMultiSend());
                    return arr;
                } else
                    return printMultiSend();
            }

            throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Unable to activate MultiSend, check MultiSend vector");
        } else if (strCommand == "enabledynode" || strCommand == "activatedynode") {
            if (pwalletMain->vMultiSend.size() < 1)
                throw JSONRPCError(RPC_INVALID_REQUEST, "Unable to activate MultiSend, check MultiSend vector");

            if (CDynamicAddress(pwalletMain->vMultiSend[0].first).IsValid()) {
                pwalletMain->fMultiSendDynodeReward = true;

                if (!walletdb.WriteMSettings(pwalletMain->fMultiSendStake, true, pwalletMain->nLastMultiSendHeight)) {
                    UniValue obj(UniValue::VOBJ);
                    obj.push_back(Pair("error", "MultiSend activated but writing settings to DB failed"));
                    UniValue arr(UniValue::VARR);
                    arr.push_back(obj);
                    arr.push_back(printMultiSend());
                    return arr;
                } else
                    return printMultiSend();
            }

            throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Unable to activate MultiSend, check MultiSend vector");
        } else if (strCommand == "disable" || strCommand == "deactivate") {
            pwalletMain->setMultiSendDisabled();
            if (!walletdb.WriteMSettings(false, false, pwalletMain->nLastMultiSendHeight))
                throw JSONRPCError(RPC_DATABASE_ERROR, "MultiSend deactivated but writing settings to DB failed");

            return printMultiSend();
        } else if (strCommand == "enableall") {
            if (!walletdb.EraseMSDisabledAddresses(pwalletMain->vDisabledAddresses))
                return "failed to clear old vector from walletDB";
            else {
                pwalletMain->vDisabledAddresses.clear();
                return printMultiSend();
            }
        }
    }
    if (params.size() == 2 && params[0].get_str() == "delete") {
        int del = std::stoi(params[1].get_str().c_str());
        if (!walletdb.EraseMultiSend(pwalletMain->vMultiSend))
            throw JSONRPCError(RPC_DATABASE_ERROR, "failed to delete old MultiSend vector from database");

        pwalletMain->vMultiSend.erase(pwalletMain->vMultiSend.begin() + del);
        if (!walletdb.WriteMultiSend(pwalletMain->vMultiSend))
            throw JSONRPCError(RPC_DATABASE_ERROR, "walletdb WriteMultiSend failed!");

        return printMultiSend();
    }
    if (params.size() == 2 && params[0].get_str() == "disable") {
        std::string disAddress = params[1].get_str();
        if (!CDynamicAddress(disAddress).IsValid())
            throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "address you want to disable is not valid");
        else {
            pwalletMain->vDisabledAddresses.push_back(disAddress);
            if (!walletdb.EraseMSDisabledAddresses(pwalletMain->vDisabledAddresses))
                throw JSONRPCError(RPC_DATABASE_ERROR, "disabled address from sending, but failed to clear old vector from walletDB");

            if (!walletdb.WriteMSDisabledAddresses(pwalletMain->vDisabledAddresses))
                throw JSONRPCError(RPC_DATABASE_ERROR, "disabled address from sending, but failed to store it to walletDB");
            else
                return printMultiSend();
        }
    }

    //if no commands are used
    if (fHelp || params.size() != 2)
        throw std::runtime_error(
            "multisend <command>\n"
            "****************************************************************\n"
            "WHAT IS MULTISEND?\n"
            "MultiSend allows a user to automatically send a percent of their stake reward to as many addresses as you would like\n"
            "The MultiSend transaction is sent when the staked coins mature (100 confirmations)\n"
            "****************************************************************\n"
            "TO CREATE OR ADD TO THE MULTISEND VECTOR:\n"
            "multisend <Dynamic Address> <percent>\n"
            "This will add a new address to the MultiSend vector\n"
            "Percent is a whole number 1 to 100.\n"
            "****************************************************************\n"
            "MULTISEND COMMANDS (usage: multisend <command>)\n"
            " print - displays the current MultiSend vector \n"
            " clear - deletes the current MultiSend vector \n"
            " enablestake/activatestake - activates the current MultiSend vector to be activated on stake rewards\n"
            " enabledynode/activatedynode - activates the current MultiSend vector to be activated on dynode rewards\n"
            " disable/deactivate - disables the current MultiSend vector \n"
            " delete <Address #> - deletes an address from the MultiSend vector \n"
            " disable <address> - prevents a specific address from sending MultiSend transactions\n"
            " enableall - enables all addresses to be eligible to send MultiSend transactions\n"
            "****************************************************************\n");

    //if the user is entering a new MultiSend item
    std::string strAddress = params[0].get_str();
    CDynamicAddress address(strAddress);
    if (!address.IsValid())
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid PIV address");
    if (std::stoi(params[1].get_str().c_str()) < 0)
        throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid parameter, expected valid percentage");
    if (pwalletMain->IsLocked())
        throw JSONRPCError(RPC_WALLET_UNLOCK_NEEDED, "Error: Please enter the wallet passphrase with walletpassphrase first.");
    unsigned int nPercent = (unsigned int) std::stoul(params[1].get_str().c_str());

    LOCK(pwalletMain->cs_wallet);
    {
        fFileBacked = pwalletMain->fFileBacked;
        //Error if 0 is entered
        if (nPercent == 0) {
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Sending 0% of stake is not valid");
        }

        //MultiSend can only send 100% of your stake
        if (nPercent + sumMultiSend() > 100)
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Failed to add to MultiSend vector, the sum of your MultiSend is greater than 100%");

        for (unsigned int i = 0; i < pwalletMain->vMultiSend.size(); i++) {
            if (pwalletMain->vMultiSend[i].first == strAddress)
                throw JSONRPCError(RPC_INVALID_PARAMETER, "Failed to add to MultiSend vector, cannot use the same address twice");
        }

        if (fFileBacked)
            walletdb.EraseMultiSend(pwalletMain->vMultiSend);

        std::pair<std::string, int> newMultiSend;
        newMultiSend.first = strAddress;
        newMultiSend.second = nPercent;
        pwalletMain->vMultiSend.push_back(newMultiSend);
        if (fFileBacked) {
            if (!walletdb.WriteMultiSend(pwalletMain->vMultiSend))
                throw JSONRPCError(RPC_DATABASE_ERROR, "walletdb WriteMultiSend failed!");
        }
    }
    return printMultiSend();
}

UniValue keepass(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    std::string strCommand;

    if (request.params.size() >= 1)
        strCommand = request.params[0].get_str();

    if (request.fHelp ||
        (strCommand != "genkey" && strCommand != "init" && strCommand != "setpassphrase"))
        throw std::runtime_error(
            "keepass <genkey|init|setpassphrase>\n");

    if (strCommand == "genkey") {
        SecureString sResult;
        // Generate RSA key
        SecureString sKey = CKeePassIntegrator::generateKeePassKey();
        sResult = "Generated Key: ";
        sResult += sKey;
        return sResult.c_str();
    } else if (strCommand == "init") {
        // Generate base64 encoded 256 bit RSA key and associate with KeePassHttp
        SecureString sResult;
        SecureString sKey;
        std::string strId;
        keePassInt.rpcAssociate(strId, sKey);
        sResult = "Association successful. Id: ";
        sResult += strId.c_str();
        sResult += " - Key: ";
        sResult += sKey.c_str();
        return sResult.c_str();
    } else if (strCommand == "setpassphrase") {
        if (request.params.size() != 2) {
            return "setlogin: invalid number of parameters. Requires a passphrase";
        }

        SecureString sPassphrase = SecureString(request.params[1].get_str().c_str());

        keePassInt.updatePassphrase(sPassphrase);

        return "setlogin: Updated credentials.";
    }

    return "Invalid command";
}

UniValue resendwallettransactions(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() != 0)
        throw std::runtime_error(
            "resendwallettransactions\n"
            "Immediately re-broadcast unconfirmed wallet transactions to all peers.\n"
            "Intended only for testing; the wallet code periodically re-broadcasts\n"
            "automatically.\n"
            "Returns array of transaction ids that were re-broadcast.\n");

    if (!g_connman)
        throw JSONRPCError(RPC_CLIENT_P2P_DISABLED, "Error: Peer-to-peer functionality missing or disabled");

    LOCK2(cs_main, pwalletMain->cs_wallet);

    std::vector<uint256> txids = pwalletMain->ResendWalletTransactionsBefore(GetTime(), g_connman.get());
    UniValue result(UniValue::VARR);
    for (const uint256& txid : txids) {
        result.push_back(txid.ToString());
    }
    return result;
}

UniValue listunspent(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() > 4)
        throw std::runtime_error(
            "listunspent ( minconf maxconf  [\"addresses\",...] [include_unsafe] )\n"
            "\nReturns array of unspent transaction outputs\n"
            "with between minconf and maxconf (inclusive) confirmations.\n"
            "Optionally filter to only include txouts paid to specified addresses.\n"
            "\nArguments:\n"
            "1. minconf          (numeric, optional, default=1) The minimum confirmations to filter\n"
            "2. maxconf          (numeric, optional, default=9999999) The maximum confirmations to filter\n"
            "3. \"addresses\"      (string) A json array of dynamic addresses to filter\n"
            "    [\n"
            "      \"address\"     (string) dynamic address\n"
            "      ,...\n"
            "    ]\n"
            "4. include_unsafe (bool, optional, default=true) Include outputs that are not safe to spend\n"
            "                  because they come from unconfirmed untrusted transactions or unconfirmed\n"
            "                  replacement transactions (cases where we are less sure that a conflicting\n"
            "                  transaction won't be mined).\n"
            "\nResult:\n"
            "[                             (array of json object)\n"
            "  {\n"
            "    \"txid\" : \"txid\",          (string) the transaction id \n"
            "    \"vout\" : n,               (numeric) the vout value\n"
            "    \"address\" : \"address\",    (string) the dynamic address\n"
            "    \"account\" : \"account\",    (string) DEPRECATED. The associated account, or \"\" for the default account\n"
            "    \"scriptPubKey\" : \"key\",   (string) the script key\n"
            "    \"amount\" : x.xxx,         (numeric) the transaction output amount in " +
            CURRENCY_UNIT + "\n"
                            "    \"confirmations\" : n,      (numeric) The number of confirmations\n"
                            "    \"redeemScript\" : n        (string) The redeemScript if scriptPubKey is P2SH\n"
                            "    \"spendable\" : xxx,        (bool) Whether we have the private keys to spend this output\n"
                            "    \"solvable\" : xxx,         (bool) Whether we know how to spend this output, ignoring the lack of keys\n"
                            "    \"ps_rounds\" : n           (numeric) The number of PS rounds\n"
                            "  }\n"
                            "  ,...\n"
                            "]\n"

                            "\nExamples:\n" +
            HelpExampleCli("listunspent", "") + HelpExampleCli("listunspent", "6 9999999 \"[\\\"XwnLY9Tf7Zsef8gMGL2fhWA9ZmMjt4KPwg\\\",\\\"XuQQkwA4FYkq2XERzMY2CiAZhJTEDAbtcg\\\"]\"") + HelpExampleRpc("listunspent", "6, 9999999 \"[\\\"XwnLY9Tf7Zsef8gMGL2fhWA9ZmMjt4KPwg\\\",\\\"XuQQkwA4FYkq2XERzMY2CiAZhJTEDAbtcg\\\"]\""));

    int nMinDepth = 1;
    if (request.params.size() > 0 && !request.params[0].isNull()) {
        RPCTypeCheckArgument(request.params[0], UniValue::VNUM);
        nMinDepth = request.params[0].get_int();
    }

    int nMaxDepth = 9999999;
    if (request.params.size() > 1 && !request.params[1].isNull()) {
        RPCTypeCheckArgument(request.params[1], UniValue::VNUM);
        nMaxDepth = request.params[1].get_int();
    }

    std::set<CDynamicAddress> setAddress;
    if (request.params.size() > 2 && !request.params[2].isNull()) {
        RPCTypeCheckArgument(request.params[2], UniValue::VARR);
        UniValue inputs = request.params[2].get_array();
        for (unsigned int idx = 0; idx < inputs.size(); idx++) {
            const UniValue& input = inputs[idx];
            CDynamicAddress address(input.get_str());
            if (!address.IsValid())
                throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, std::string("Invalid Dynamic address: ") + input.get_str());
            if (setAddress.count(address))
                throw JSONRPCError(RPC_INVALID_PARAMETER, std::string("Invalid parameter, duplicated address: ") + input.get_str());
            setAddress.insert(address);
        }
    }

    bool include_unsafe = true;
    if (request.params.size() > 3 && !request.params[3].isNull()) {
        RPCTypeCheckArgument(request.params[3], UniValue::VBOOL);
        include_unsafe = request.params[3].get_bool();
    }

    UniValue results(UniValue::VARR);
    std::vector<COutput> vecOutputs;
    assert(pwalletMain != nullptr);
    LOCK2(cs_main, pwalletMain->cs_wallet);
    pwalletMain->AvailableCoins(vecOutputs, !include_unsafe, nullptr, true);
    for (const COutput& out : vecOutputs) {
        if (out.nDepth < nMinDepth || out.nDepth > nMaxDepth)
            continue;

        CTxDestination address;
        const CScript& scriptPubKey = out.tx->tx->vout[out.i].scriptPubKey;
        bool fValidAddress = ExtractDestination(scriptPubKey, address);

        if (setAddress.size() && (!fValidAddress || !setAddress.count(address)))
            continue;

        UniValue entry(UniValue::VOBJ);
        entry.push_back(Pair("txid", out.tx->GetHash().GetHex()));
        entry.push_back(Pair("vout", out.i));

        if (fValidAddress) {
            entry.push_back(Pair("address", CDynamicAddress(address).ToString()));

            if (pwalletMain->mapAddressBook.count(address))
                entry.push_back(Pair("account", pwalletMain->mapAddressBook[address].name));

            if (scriptPubKey.IsPayToScriptHash()) {
                const CScriptID& hash = boost::get<CScriptID>(address);
                CScript redeemScript;
                if (pwalletMain->GetCScript(hash, redeemScript))
                    entry.push_back(Pair("redeemScript", HexStr(redeemScript.begin(), redeemScript.end())));
            }
        }

        entry.push_back(Pair("scriptPubKey", HexStr(scriptPubKey.begin(), scriptPubKey.end())));
        entry.push_back(Pair("amount", ValueFromAmount(out.tx->tx->vout[out.i].nValue)));
        entry.push_back(Pair("confirmations", out.nDepth));
        entry.push_back(Pair("spendable", out.fSpendable));
        entry.push_back(Pair("solvable", out.fSolvable));
        entry.push_back(Pair("ps_rounds", pwalletMain->GetOutpointPrivateSendRounds(COutPoint(out.tx->GetHash(), out.i))));
        results.push_back(entry);
    }

    return results;
}

UniValue fundrawtransaction(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 1 || request.params.size() > 2)
        throw std::runtime_error(
            "fundrawtransaction \"hexstring\" ( options )\n"
            "\nAdd inputs to a transaction until it has enough in value to meet its out value.\n"
            "This will not modify existing inputs, and will add at most one change output to the outputs.\n"
            "No existing outputs will be modified unless \"subtractFeeFromOutputs\" is specified.\n"
            "Note that inputs which were signed may need to be resigned after completion since in/outputs have been added.\n"
            "The inputs added will not be signed, use signrawtransaction for that.\n"
            "Note that all existing inputs must have their previous output transaction be in the wallet.\n"
            "Note that all inputs selected must be of standard form and P2SH scripts must be\n"
            "in the wallet using importaddress or addmultisigaddress (to calculate fees).\n"
            "You can see whether this is the case by checking the \"solvable\" field in the listunspent output.\n"
            "Only pay-to-pubkey, multisig, and P2SH versions thereof are currently supported for watch-only\n"
            "\nArguments:\n"
            "1. \"hexstring\"           (string, required) The hex string of the raw transaction\n"
            "2. options                 (object, optional)\n"
            "   {\n"
            "     \"changeAddress\"          (string, optional, default pool address) The Dynamic address to receive the change\n"
            "     \"changePosition\"         (numeric, optional, default random) The index of the change output\n"
            "     \"includeWatching\"        (boolean, optional, default false) Also select inputs which are watch only\n"
            "     \"lockUnspents\"           (boolean, optional, default false) Lock selected unspent outputs\n"
            "     \"feeRate\"                (numeric, optional, default not set: makes wallet determine the fee) Set a specific fee rate in " + CURRENCY_UNIT + "/kB\n"
            "     \"subtractFeeFromOutputs\" (array, optional) A json array of integers.\n"
            "                              The fee will be equally deducted from the amount of each specified output.\n"
            "                              The outputs are specified by their zero-based index, before any change output is added.\n"
            "                              Those recipients will receive less DYN than you enter in their corresponding amount field.\n"
            "                              If no outputs are specified here, the sender pays the fee.\n"
            "                                  [vout_index,...]\n"
//                            "     \"replaceable\"            (boolean, optional) Marks this transaction as BIP125 replaceable.\n"
            "                              Allows this transaction to be replaced by a transaction with higher fees\n"
            "     \"conf_target\"            (numeric, optional) Confirmation target (in blocks)\n"
            "     \"estimate_mode\"          (string, optional, default=UNSET) The fee estimate mode, must be one of:\n"
            "         \"UNSET\"\n"
            "         \"ECONOMICAL\"\n"
            "         \"CONSERVATIVE\"\n"
            "   }\n"
            "                         for backward compatibility: passing in a true instead of an object will result in {\"includeWatching\":true}\n"
            "\nResult:\n"
            "{\n"
            "  \"hex\":       \"value\", (string)  The resulting raw transaction (hex-encoded string)\n"
            "  \"fee\":       n,         (numeric) Fee in " + CURRENCY_UNIT + " the resulting transaction pays\n"
            "  \"changepos\": n          (numeric) The position of the added change output, or -1\n"
            "}\n"
            "\nExamples:\n"
            "\nCreate a transaction with no inputs\n"
            + HelpExampleCli("createrawtransaction", "\"[]\" \"{\\\"myaddress\\\":0.01}\"") +
            "\nAdd sufficient unsigned inputs to meet the output value\n"
            + HelpExampleCli("fundrawtransaction", "\"rawtransactionhex\"") +
            "\nSign the transaction\n"
            + HelpExampleCli("signrawtransaction", "\"fundedtransactionhex\"") +
            "\nSend the transaction\n"
            + HelpExampleCli("sendrawtransaction", "\"signedtransactionhex\"")
    );

    RPCTypeCheck(request.params, boost::assign::list_of(UniValue::VSTR));

    CTxDestination changeAddress = CNoDestination();
    CCoinControl coinControl;
    int changePosition = -1;
    bool includeWatching = false;
    bool lockUnspents = false;
    UniValue subtractFeeFromOutputs;
    std::set<int> setSubtractFeeFromOutputs;

    if (request.params.size() > 1) {
        if (request.params[1].type() == UniValue::VBOOL) {
            // backward compatibility bool only fallback
            includeWatching = request.params[1].get_bool();
        } else {
            RPCTypeCheck(request.params, boost::assign::list_of(UniValue::VSTR)(UniValue::VOBJ));

            UniValue options = request.params[1];

            RPCTypeCheckObj(options,
                {
                    {"changeAddress", UniValueType(UniValue::VSTR)},
                    {"changePosition", UniValueType(UniValue::VNUM)},
                    {"includeWatching", UniValueType(UniValue::VBOOL)},
                    {"lockUnspents", UniValueType(UniValue::VBOOL)},
                    {"reserveChangeKey", UniValueType(UniValue::VBOOL)},
                    {"feeRate", UniValueType()}, // will be checked below
                    {"subtractFeeFromOutputs", UniValueType(UniValue::VARR)},
                    {"conf_target", UniValueType(UniValue::VNUM)},
                    {"estimate_mode", UniValueType(UniValue::VSTR)},
                },
                true, true);

            if (options.exists("changeAddress")) {
                CDynamicAddress address(options["changeAddress"].get_str());

                if (!address.IsValid())
                    throw JSONRPCError(RPC_INVALID_PARAMETER, "changeAddress must be a valid dynamic address");

                changeAddress = address.Get();
            }

            if (options.exists("changePosition"))
                changePosition = options["changePosition"].get_int();

            if (options.exists("includeWatching"))
                includeWatching = options["includeWatching"].get_bool();

            if (options.exists("lockUnspents"))
                lockUnspents = options["lockUnspents"].get_bool();

            if (options.exists("feeRate"))
                {
                    coinControl.m_feerate = CFeeRate(AmountFromValue(options["feeRate"]));
                    coinControl.fOverrideFeeRate = true;
                }

            if (options.exists("subtractFeeFromOutputs"))
                subtractFeeFromOutputs = options["subtractFeeFromOutputs"].get_array();
        
            if (options.exists("conf_target")) {
                if (options.exists("feeRate")) {
                    throw JSONRPCError(RPC_INVALID_PARAMETER, "Cannot specify both conf_target and feeRate");
                }
                coinControl.m_confirm_target = ParseConfirmTarget(options["conf_target"]);
            }
            if (options.exists("estimate_mode")) {
                if (options.exists("feeRate")) {
                    throw JSONRPCError(RPC_INVALID_PARAMETER, "Cannot specify both estimate_mode and feeRate");
                }
                if (!FeeModeFromString(options["estimate_mode"].get_str(), coinControl.m_fee_mode)) {
                    throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid estimate_mode parameter");
                }
            }
        }
    }

    // parse hex string from parameter
    CMutableTransaction tx;
    if (!DecodeHexTx(tx, request.params[0].get_str()))
        throw JSONRPCError(RPC_DESERIALIZATION_ERROR, "TX decode failed");

    if (tx.vout.size() == 0)
        throw JSONRPCError(RPC_INVALID_PARAMETER, "TX must have at least one output");

    if (changePosition != -1 && (changePosition < 0 || (unsigned int)changePosition > tx.vout.size()))
        throw JSONRPCError(RPC_INVALID_PARAMETER, "changePosition out of bounds");

    for (unsigned int idx = 0; idx < subtractFeeFromOutputs.size(); idx++) {
        int pos = subtractFeeFromOutputs[idx].get_int();
        if (setSubtractFeeFromOutputs.count(pos))
            throw JSONRPCError(RPC_INVALID_PARAMETER, strprintf("Invalid parameter, duplicated position: %d", pos));
        if (pos < 0)
            throw JSONRPCError(RPC_INVALID_PARAMETER, strprintf("Invalid parameter, negative position: %d", pos));
        if (pos >= int(tx.vout.size()))
            throw JSONRPCError(RPC_INVALID_PARAMETER, strprintf("Invalid parameter, position too large: %d", pos));
        setSubtractFeeFromOutputs.insert(pos);
    }

    CAmount nFeeOut;
    std::string strFailReason;

    if (!pwalletMain->FundTransaction(tx, nFeeOut, changePosition, strFailReason, includeWatching, lockUnspents, setSubtractFeeFromOutputs, coinControl))
        throw JSONRPCError(RPC_INTERNAL_ERROR, strFailReason);

    UniValue result(UniValue::VOBJ);
    result.push_back(Pair("hex", EncodeHexTx(tx)));
    result.push_back(Pair("changepos", changePosition));
    result.push_back(Pair("fee", ValueFromAmount(nFeeOut)));

    return result;
}

extern UniValue dumpprivkey(const JSONRPCRequest& request); // in rpcdump.cpp
extern UniValue importprivkey(const JSONRPCRequest& request);
extern UniValue importstealthaddress(const JSONRPCRequest& request);
extern UniValue importaddress(const JSONRPCRequest& request);
extern UniValue importpubkey(const JSONRPCRequest& request);
extern UniValue dumpwallet(const JSONRPCRequest& request);
extern UniValue importwallet(const JSONRPCRequest& request);
extern UniValue importprunedfunds(const JSONRPCRequest& request);
extern UniValue removeprunedfunds(const JSONRPCRequest& request);
extern UniValue importmulti(const JSONRPCRequest& request);
extern UniValue importmnemonic(const JSONRPCRequest& request);

extern UniValue dumphdinfo(const JSONRPCRequest& request);
extern UniValue dumpbdapkeys(const JSONRPCRequest& request);
extern UniValue importbdapkeys(const JSONRPCRequest& request);

extern UniValue importelectrumwallet(const JSONRPCRequest& request);

extern UniValue privatesend(const JSONRPCRequest& request);

static const CRPCCommand commands[] =
    {
        //  category              name                        actor (function)           okSafe  argNames
        //  --------------------- ------------------------    -----------------------    ------- ---------------------------------------------------------
        {"rawtransactions", "fundrawtransaction", &fundrawtransaction, false, {"hexstring", "options"}},
        {"hidden", "resendwallettransactions", &resendwallettransactions, true, {}},
        {"wallet", "abandontransaction", &abandontransaction, false, {"txid"}},
        {"wallet", "addmultisigaddress", &addmultisigaddress, true, {"nrequired", "keys", "account"}},
        {"wallet", "backupwallet", &backupwallet, true, {"destination"}},
        {"wallet", "dumpprivkey", &dumpprivkey, true, {"address"}},
        {"wallet", "dumpwallet", &dumpwallet, true, {"filename"}},
        {"wallet", "encryptwallet", &encryptwallet, true, {"passphrase"}},
        {"wallet", "getaccountaddress", &getaccountaddress, true, {"account"}},
        {"wallet", "getaccount", &getaccount, true, {"address"}},
        {"wallet", "getaddressesbyaccount", &getaddressesbyaccount, true, {"account"}},
        {"wallet", "getbalance", &getbalance, false, {"account", "minconf", "addlocked", "include_watchonly"}},
        {"wallet", "getnewaddress", &getnewaddress, true, {"account"}},
        {"wallet", "getnewed25519address", &getnewed25519address, true, {"account"}},
        {"wallet", "getrawchangeaddress", &getrawchangeaddress, true, {}},
        {"wallet", "getreceivedbyaccount", &getreceivedbyaccount, false, {"account", "minconf", "addlocked"}},
        {"wallet", "getreceivedbyaddress", &getreceivedbyaddress, false, {"address", "minconf", "addlocked"}},
        {"wallet", "gettransaction", &gettransaction, false, {"txid", "include_watchonly"}},
        {"wallet", "getunconfirmedbalance", &getunconfirmedbalance, false, {}},
        {"wallet", "getstakesplitthreshold", &getstakesplitthreshold, false, {}},
        {"wallet", "getwalletinfo", &getwalletinfo, false, {}},
        {"wallet", "importmulti", &importmulti, true, {"requests", "options"}},
        {"wallet", "importprivkey", &importprivkey, true, {"privkey", "label", "rescan"}},
        {"wallet", "importstealthaddress", &importstealthaddress, true, {"privkey", "label", "rescan"}},
        {"wallet", "importwallet",  &importwallet,  true, {"filename", "forcerescan"}},
        {"wallet", "importaddress", &importaddress, true, {"address", "label", "rescan", "p2sh"}},
        {"wallet", "importprunedfunds", &importprunedfunds, true, {"rawtransaction", "txoutproof"}},
        {"wallet", "importpubkey", &importpubkey, true, {"pubkey", "label", "rescan"}},
        {"wallet", "keypoolrefill", &keypoolrefill, true, {"newsize"}},
        {"wallet", "listaccounts", &listaccounts, false, {"minconf", "addlocked", "include_watchonly"}},
        {"wallet", "listaddressgroupings", &listaddressgroupings, false, {}},
        {"wallet", "listaddressbalances", &listaddressbalances, false, {"minamount"}},
        {"wallet", "listlockunspent", &listlockunspent, false, {}},
        {"wallet", "listreceivedbyaccount", &listreceivedbyaccount, false, {"minconf", "addlocked", "include_empty", "include_watchonly"}},
        {"wallet", "listreceivedbyaddress", &listreceivedbyaddress, false, {"minconf", "addlocked", "include_empty", "include_watchonly"}},
        {"wallet", "listsinceblock", &listsinceblock, false, {"blockhash", "target_confirmations", "include_watchonly"}},
        {"wallet", "listtransactions", &listtransactions, false, {"account", "count", "skip", "include_watchonly"}},
        {"wallet", "listunspent", &listunspent, false, {"minconf", "maxconf", "addresses", "include_unsafe"}},
        {"wallet", "lockunspent", &lockunspent, true, {"unlock", "transactions"}},
        {"wallet", "move", &movecmd, false, {"fromaccount", "toaccount", "amount", "minconf", "comment"}},
        {"wallet", "reservebalance", &reservebalance, true, {"amount"}},
        {"wallet", "sendfrom", &sendfrom, false, {"fromaccount", "toaddress", "amount", "minconf", "addlocked", "comment", "comment_to"}},
        {"wallet", "sendmany", &sendmany, false, {"fromaccount", "amounts", "minconf", "addlocked", "comment", "subtractfeefrom"}},
        {"wallet", "sendtoaddress", &sendtoaddress, false, {"address", "amount", "comment", "comment_to", "subtractfeefromamount"}},
        {"wallet", "sendfromaddress", &sendfromaddress, false, {"from_address","address","amount","comment","comment_to","subtractfeefromamount", "conf_target","estimate_mode"} },
        {"wallet", "setaccount", &setaccount, true, {"address", "account"}},
        {"wallet", "setprivatesendrounds", &setprivatesendrounds, true, {"rounds"}},
        {"wallet", "setprivatesendamount", &setprivatesendamount, true, {"amount"}},
        {"wallet", "setstakesplitthreshold", &setstakesplitthreshold, false, {"amount"}},
        {"wallet", "settxfee", &settxfee, true, {"amount"}},
        {"wallet", "signmessage", &signmessage, true, {"address", "message"}},
        {"wallet", "walletlock", &walletlock, true, {}},
        {"wallet", "walletpassphrasechange", &walletpassphrasechange, true, {"oldpassphrase", "newpassphrase"}},
        {"wallet", "walletpassphrase", &walletpassphrase, true, {"passphrase", "timeout", "mixingonly"}},
        {"wallet", "removeprunedfunds", &removeprunedfunds, true, {"txid"}},

        {"wallet", "keepass", &keepass, true, {}},
        {"wallet", "instantsendtoaddress", &instantsendtoaddress, false, {"address", "amount", "comment", "comment_to", "subtractfeefromamount"}},
        {"wallet", "dumphdinfo", &dumphdinfo, true, {"hdseed", "mnemonic", "mnemonicpassphrase"}},
        {"wallet", "dumpbdapkeys", &dumpbdapkeys, true, {"bdap_id"}},
        {"wallet", "importbdapkeys", &importbdapkeys,true, {"bdap_id", "wallet_privkey", "link_privkey", "DHT_privkey", "rescan"}},
        {"wallet", "importelectrumwallet", &importelectrumwallet, true, {"filename", "index"}},
        {"wallet", "importmnemonic", &importmnemonic, true, {"mnemonic", "passphrase", "begin", "end", "forcerescan"}},

        {"wallet", "getnewstealthaddress", &getnewstealthaddress, true, {}},
};

void RegisterWalletRPCCommands(CRPCTable& t)
{
    if (gArgs.GetBoolArg("-disablewallet", false))
        return;

    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        t.appendCommand(commands[vcidx].name, &commands[vcidx]);
}
