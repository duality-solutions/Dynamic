// Copyright (c) 2021-present Duality Blockchain Solutions Developers 
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "base58.h"
#include "bdap/utils.h"
#include "core_io.h"
#include "hash.h"
#include "init.h"
#include "keepass.h"
#include "net.h"
#include "netbase.h"
#include "rpc/server.h"
#include "swap/ss58.h"
#include "swap/swapdata.h"
#include "swap/swapdb.h"
#include "timedata.h"
#include "uint256.h"
#include "util.h"
#include "utilmoneystr.h"
#include "utilstrencodings.h"
#include "validation.h"
#include "wallet/wallet.h"
#include "wallet/walletdb.h"

#include <univalue.h>

#include <cmath>

#ifdef ENABLE_WALLET

extern bool EnsureWalletIsAvailable(bool avoidException);
extern bool SendSwapTransaction(const CScript& burnScript, CWalletTx& wtxNew, const CScript& sendAddress, std::string& strError);


struct SwapCompareHeight {
    bool operator()(const CSwapData& a, const CSwapData& b) {
        return (a.nHeight < b.nHeight);
    }
};

UniValue SwapDynamic(const std::string& address, const bool fSend, std::string& errorMessage)
{
    if (!EnsureWalletIsAvailable(true)) {
        errorMessage = "Wallet is not available";
        return NullUniValue;
    }
    if (pwalletMain->IsLocked()) {
        errorMessage = "Wallet is locked";
        return NullUniValue;
    }

    CSS58 ss58Address(address);
    if (!ss58Address.fValid) {
        errorMessage = "Invalid swap address. " + ss58Address.strError;
        return NullUniValue;
    }

    UniValue oResult(UniValue::VOBJ);
    oResult.push_back(Pair("address_bytes", ss58Address.nLength));
    oResult.push_back(Pair("hex_address", ss58Address.AddressHex()));
    oResult.push_back(Pair("address_type", (int64_t)ss58Address.AddressType()));
    oResult.push_back(Pair("address_pubkey", ss58Address.PublicKeyHex()));
    oResult.push_back(Pair("address_checksum", ss58Address.AddressChecksumHex()));
    if (fSend) {
        CWalletTx wtx;
        CScript scriptSendFrom;
        CScript swapScript = CScript() << OP_RETURN << ss58Address.AddressBytes();
        std::string strError = "";
        if (SendSwapTransaction(swapScript, wtx, scriptSendFrom, strError)) {
            oResult.push_back(Pair("amount", FormatMoney(wtx.GetDebit(ISMINE_SPENDABLE))));
            oResult.push_back(Pair("txid", wtx.GetHash().ToString()));
        } else {
            errorMessage = strError;
            return NullUniValue;
        }
    }
    return oResult;
}

UniValue swapdynamic(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() > 1 || request.params.size() == 0)
        throw std::runtime_error(
            "swapdynamic \"address\"\n"
            "\nSend coins to be swapped for Substrate chain\n"
            "\nArguments:\n"
            "1. \"address\"        (string, required)  The Substrate address to swap funds.\n"
            "\nExamples:\n" +
            HelpExampleCli("swapdynamic", "\"1a1LcBX6hGPKg5aQ6DXZpAHCCzWjckhea4sz3P1PvL3oc4F\"") + 
            HelpExampleRpc("swapdynamic", "\"1a1LcBX6hGPKg5aQ6DXZpAHCCzWjckhea4sz3P1PvL3oc4F\""));

    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    EnsureWalletIsUnlocked();

    std::string strSubstrateAddress = request.params[0].get_str();

    std::string errorMessage = "";
    UniValue oResult = SwapDynamic(strSubstrateAddress, true, errorMessage);
    if (errorMessage != "")
        throw JSONRPCError(RPC_TYPE_ERROR, errorMessage);

    return oResult;
}
#endif // ENABLE_WALLET

UniValue GetAddress(const std::string& address, std::string& errorMessage)
{
    CSS58 ss58Address(address);
    UniValue oResult(UniValue::VOBJ);
    oResult.push_back(Pair("address_bytes", ss58Address.nLength));
    oResult.push_back(Pair("hex_address", ss58Address.AddressHex()));
    oResult.push_back(Pair("address_type", (int64_t)ss58Address.AddressType()));
    oResult.push_back(Pair("address_pubkey", ss58Address.PublicKeyHex()));
    oResult.push_back(Pair("address_checksum", ss58Address.AddressChecksumHex()));
    oResult.push_back(Pair("calculated_checksum", ss58Address.CalulatedChecksumHex()));
    oResult.push_back(Pair("calculated_hash", ss58Address.CalculatedHashHex()));
    oResult.push_back(Pair("valid_checksum", ss58Address.ValidChecksum() ? "true" : "false"));
    oResult.push_back(Pair("valid", ss58Address.Valid() ? "true" : "false"));
    if (ss58Address.strError != "") {
        oResult.push_back(Pair("errorMessage", ss58Address.strError));
    }
    return oResult;
}

UniValue ss58valid(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() > 1 || request.params.size() == 0)
        throw std::runtime_error(
            "ss58valid \"address\"\n"
            "\nArguments:\n"
            "1. \"address\"        (string, required)  The Substrate address to swap funds.\n"
            "\nExamples:\n" +
            HelpExampleCli("ss58valid", "\"1a1LcBX6hGPKg5aQ6DXZpAHCCzWjckhea4sz3P1PvL3oc4F\"") + 
            HelpExampleRpc("ss58valid", "\"1a1LcBX6hGPKg5aQ6DXZpAHCCzWjckhea4sz3P1PvL3oc4F\""));

    std::string strSubstrateAddress = request.params[0].get_str();

    std::string errorMessage = "";
    UniValue oResult = GetAddress(strSubstrateAddress, errorMessage);
    if (errorMessage != "")
        throw JSONRPCError(RPC_TYPE_ERROR, errorMessage);

    return oResult;
}

UniValue getswaps(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() > 2)
        throw std::runtime_error(
            "getswaps start_height end_height\n"
            "\nSend coins to be swapped for Substrate chain\n"
            "\nArguments:\n"
            "1. \"start_height\"        (int, optional)  Swaps starting at this block height.\n"
            "1. \"end_height\"          (int, optional)  Swaps ending at this block height.\n"
            "\nExamples:\n" +
            HelpExampleCli("getswap", "0 100000") + 
            HelpExampleRpc("getswap", "0 100000"));

    int nStartHeight = 0;
    int nEndHeight = (std::numeric_limits<int>::max());
    if (!request.params[0].isNull()) {
        nStartHeight = request.params[0].get_int();
        if (!request.params[1].isNull()) {
            nEndHeight = request.params[1].get_int();
        }
    }

    std::vector<CSwapData> vSwaps;
    if (GetAllSwaps(vSwaps)) {
        std::sort(vSwaps.begin(), vSwaps.end(), SwapCompareHeight()); //sort entries by nHeight
        UniValue oResult(UniValue::VOBJ);
        CAmount totalAmount = 0;
        int count = 0;
        for (const CSwapData& swap : vSwaps) {
            if (swap.nHeight >= nStartHeight && swap.nHeight <= nEndHeight) {
                UniValue oSwap(UniValue::VOBJ);
                const CAmount fee = swap.GetFee();
                oSwap.push_back(Pair("address", swap.Address()));
                oSwap.push_back(Pair("out_amount", FormatMoney(swap.Amount)));
                oSwap.push_back(Pair("fee", FormatMoney(fee)));
                oSwap.push_back(Pair("swap_amount", FormatMoney(swap.Amount + fee)));
                oSwap.push_back(Pair("txid", swap.TxId.ToString()));
                oSwap.push_back(Pair("nout", swap.nOut));
                oSwap.push_back(Pair("block_height", swap.nHeight));
                if (fee <= 0) {
                    oSwap.push_back(Pair("warning", "Zero fee."));
                }
                oResult.push_back(Pair(swap.TxId.ToString(), oSwap));
                totalAmount += (swap.Amount + fee);
                count += 1;
            }
        }
        oResult.push_back(Pair("count", count));
        oResult.push_back(Pair("total_amount", FormatMoney(totalAmount)));
        return oResult;
    } else {
        throw JSONRPCError(RPC_INTERNAL_ERROR, "Get all swaps from LevelDB failed.");
    }
}

static const CRPCCommand commands[] =
    {
        //  category              name                     actor (function)           okSafe argNames
        //  --------------------- ------------------------ -----------------------    ------ --------------------
#ifdef ENABLE_WALLET
        /* Dynamic Swap To Substrate Chain */
        {"swap", "swapdynamic", &swapdynamic, true, {"address"}},
#endif //ENABLE_WALLET
        {"swap", "getswaps", &getswaps, true, {"address_hex"}},
        {"swap", "ss58valid", &ss58valid, true, {"address"}},
};

void RegisterSwapRPCCommands(CRPCTable &t)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        t.appendCommand(commands[vcidx].name, &commands[vcidx]);
}
