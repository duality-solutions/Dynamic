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

extern bool EnsureWalletIsAvailable(bool avoidException);
extern void SendSwapTransaction(const CScript& burnScript, CWalletTx& wtxNew, const CAmount& nValue, const CScript& sendAddress);

// todo: move to seperate file
UniValue swapdynamic(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() > 2 || request.params.size() == 0)
        throw std::runtime_error(
            "swapdynamic \"address\" \"amount\"\n"
            "\nSend coins to be swapped for Substrate chain\n"
            "\nArguments:\n"
            "1. \"address\"        (string, required)  The Substrate address to swap funds.\n"
            "2. \"amount\"         (numeric, optional) The amount of coins to be swapped.\n"
            "\nExamples:\n" +
            HelpExampleCli("swapdynamic", "\"1a1LcBX6hGPKg5aQ6DXZpAHCCzWjckhea4sz3P1PvL3oc4F\" \"123.456\" ") + 
            HelpExampleRpc("swapdynamic", "\"1a1LcBX6hGPKg5aQ6DXZpAHCCzWjckhea4sz3P1PvL3oc4F\" \"123.456\" "));

    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    CWalletTx wtx;

    EnsureWalletIsUnlocked();

    std::string strSubstrateAddress = request.params[0].get_str();
    std::vector<uint8_t> vchAddress;
    // Validate the address is correct.
    bool valid = DecodeBase58(strSubstrateAddress, vchAddress);
    if (!valid)
        throw JSONRPCError(RPC_TYPE_ERROR, "Invalid swap address: base58 error");

    CScript swapScript = CScript() << OP_RETURN << vchAddress;

    CAmount nAmount;
    if (!request.params[1].isNull()) {
        nAmount = AmountFromValue(request.params[1]);
        if (nAmount <= 0)
            throw JSONRPCError(RPC_TYPE_ERROR, "Invalid amount for swap");
    }
    
    UniValue oResult(UniValue::VOBJ);
    std::string strHexAddress = CharVectorToHexString(vchAddress);
    size_t addressLen = strHexAddress.length();
    std::string strTypeHex = strHexAddress.substr(0, 2);
    std::string strPubKeyHex = strHexAddress.substr(2, 64);
    oResult.push_back(Pair("address_bytes", vchAddress.size()));
    oResult.push_back(Pair("hex_address", "0x" + strHexAddress));
    oResult.push_back(Pair("address_type", "0x" + strTypeHex));
    oResult.push_back(Pair("address_pubkey", "0x" + strPubKeyHex));
    if (vchAddress.size() - 34 > 0)
    {
        oResult.push_back(Pair("checksum", "0x" + strHexAddress.substr(66, addressLen - 66)));
        // validate checksum by hashing the first 66 bytes.
        // blake2d_512(address_type + address_pubkey)
        // first two is type, next 64 is pubkey and the remaining are checksum
        std::vector<uint8_t> vchAddressChecksumData = vchAddress;
        vchAddressChecksumData.resize(34);
        std::string strPreimageBase58 = EncodeBase58(vchAddressChecksumData);
        std::string strPreimage = "SS58PRE" + strPreimageBase58;
        std::vector<uint8_t> vchData = vchFromString(strPreimage);
        oResult.push_back(Pair("preimage_base58", strPreimageBase58));
        oResult.push_back(Pair("preimage", strPreimage));
        uint256 hash = HashBlake2b_256(vchData.begin(), vchData.end());
        oResult.push_back(Pair("checksum_hash", hash.ToString()));
        oResult.push_back(Pair("original_value", EncodeBase58(vchAddress)));
    }
    CScript scriptSendFrom;
    if (nAmount > 0) {
        SendSwapTransaction(swapScript, wtx, nAmount, scriptSendFrom);
        oResult.push_back(Pair("txid", wtx.GetHash().ToString()));
    }
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
    CAmount totalAmount = 0;
    if (GetAllSwaps(vSwaps)) {
        UniValue oResult(UniValue::VOBJ);
        for (const CSwapData& swap : vSwaps) {
            if (swap.nHeight >= nStartHeight && swap.nHeight <= nEndHeight) {
                UniValue oSwap(UniValue::VOBJ);
                oSwap.push_back(Pair("address", swap.Address()));
                oSwap.push_back(Pair("amount", FormatMoney(swap.Amount)));
                oSwap.push_back(Pair("txid", swap.TxId.ToString()));
                oSwap.push_back(Pair("nout", swap.nOut));
                oSwap.push_back(Pair("block_height", swap.nHeight));
                oResult.push_back(Pair(swap.TxId.ToString(), oSwap));
                totalAmount += swap.Amount;
            }
        }
        oResult.push_back(Pair("count", vSwaps.size()));
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
        {"swap", "swapdynamic", &swapdynamic, true, {"address", "amount"}},
#endif //ENABLE_WALLET
        {"swap", "getswaps", &getswaps, true, {"address_hex"}},
};

void RegisterSwapRPCCommands(CRPCTable &t)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        t.appendCommand(commands[vcidx].name, &commands[vcidx]);
}
