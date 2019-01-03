// Copyright (c) 2017-2019 Duality Blockchain Solutions Developers

#include "fluid.h"

#include "chain.h"
#include "core_io.h"
#include "fluiddb.h"
#include "fluiddynode.h"
#include "fluidmining.h"
#include "fluidmint.h"
#include "fluidsovereign.h"
#include "init.h"
#include "keepass.h"
#include "net.h"
#include "netbase.h"
#include "rpcserver.h"
#include "timedata.h"
#include "util.h"
#include "utilmoneystr.h"
#include "utilstrencodings.h"
#include "validation.h"
#include "wallet/wallet.h"
#include "wallet/walletdb.h"

#include <univalue.h>

extern bool EnsureWalletIsAvailable(bool avoidException);
extern void SendCustomTransaction(const CScript generatedScript, CWalletTx& wtxNew, CAmount nValue, bool fUseInstantSend = false);

opcodetype getOpcodeFromString(std::string input)
{
    if (input == "OP_MINT")
        return OP_MINT;
    else if (input == "OP_REWARD_DYNODE")
        return OP_REWARD_DYNODE;
    else if (input == "OP_REWARD_MINING")
        return OP_REWARD_MINING;
    else if (input == "OP_SWAP_SOVEREIGN_ADDRESS")
        return OP_SWAP_SOVEREIGN_ADDRESS;
    else if (input == "OP_UPDATE_FEES")
        return OP_UPDATE_FEES;
    else if (input == "OP_FREEZE_ADDRESS")
        return OP_FREEZE_ADDRESS;
    else if (input == "OP_RELEASE_ADDRESS")
        return OP_RELEASE_ADDRESS;
    else
        return OP_RETURN;

    return OP_RETURN;
};

UniValue maketoken(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() < 2) {
        throw std::runtime_error(
            "maketoken \"string\"\n"
            "\nConvert String to Hexadecimal Format\n"
            "\nArguments:\n"
            "1. \"string\"         (string, required) String that has to be converted to hex.\n"
            "\nExamples:\n" +
            HelpExampleCli("maketoken", "\"Hello World!\"") + HelpExampleRpc("maketoken", "\"Hello World!\""));
    }
    std::string result;

    for (uint32_t iter = 0; iter != request.params.size(); iter++) {
        result += request.params[iter].get_str() + SubDelimiter;
    }

    result.pop_back();
    fluid.ConvertToHex(result);

    return result;
}

UniValue gettime(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 0) {
        throw std::runtime_error(
            "gettime\n"
            "\nReturns the current Epoch time (https://www.epochconverter.com).\n"
            "\nExamples:\n" +
            HelpExampleCli("gettime", "\"1535543210\"") + HelpExampleRpc("gettime", "\"1535543210\""));
    }
    return GetTime();
}

UniValue getrawpubkey(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "getrawpubkey \"address\"\n"
            "\nGet (un)compressed raw public key of an address of the wallet\n"
            "\nArguments:\n"
            "1. \"address\"         (string, required) The Dynamic Address from which the pubkey is to recovered.\n"
            "\nExamples:\n" +
            HelpExampleCli("getrawpubkey", "D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf") + HelpExampleRpc("getrawpubkey", "D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf"));
    UniValue ret(UniValue::VOBJ);

    CDynamicAddress address(request.params[0].get_str());
    bool isValid = address.IsValid();

    if (isValid) {
        CTxDestination dest = address.Get();
        CScript scriptPubKey = GetScriptForDestination(dest);
        ret.push_back(Pair("pubkey", HexStr(scriptPubKey.begin(), scriptPubKey.end())));
    } else {
        ret.push_back(Pair("errors", "Dynamic address is not valid!"));
    }

    return ret;
}

UniValue burndynamic(const UniValue& params, bool fHelp)
{
    if (!EnsureWalletIsAvailable(fHelp))
        return NullUniValue;

    if (fHelp || params.size() != 1)
        throw std::runtime_error(
            "burndynamic \"amount\"\n"
            "\nSend coins to be burnt (destroyed) onto the Dynamic Network\n"
            "\nArguments:\n"
            "1. \"account\"         (numeric or string, required) The amount of coins to be minted.\n"
            "\nExamples:\n" +
            HelpExampleCli("burndynamic", "\"123.456\" \"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\"") + HelpExampleRpc("burndynamic", "\"123.456\" \"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\""));
    CWalletTx wtx;

    EnsureWalletIsUnlocked();

    CAmount nAmount = AmountFromValue(params[0]);

    if (nAmount <= 0)
        throw JSONRPCError(RPC_TYPE_ERROR, "Invalid amount for destruction");

    std::string result = std::to_string(nAmount);
    fluid.ConvertToHex(result);

    CScript destroyScript = CScript() << OP_RETURN << ParseHex(result);

    SendCustomTransaction(destroyScript, wtx, nAmount, false);

    return wtx.GetHash().GetHex();
}

opcodetype negatif = OP_RETURN;

UniValue sendfluidtransaction(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() != 2)
        throw std::runtime_error(
            "sendfluidtransaction \"OP_MINT || OP_REWARD_DYNODE || OP_REWARD_MINING\" \"hexstring\"\n"
            "\nSend Fluid transactions to the network\n"
            "\nArguments:\n"
            "1. \"opcode\"  (string, required) The Fluid operation to be executed.\n"
            "2. \"hexstring\" (string, required) The token for that opearation.\n"
            "\nExamples:\n" +
            HelpExampleCli("sendfluidtransaction", "\"3130303030303030303030303a3a313439393336353333363a3a445148697036443655376d46335761795a32747337794478737a71687779367a5a6a20494f42447a557167773\"") + HelpExampleRpc("sendfluidtransaction", "\"3130303030303030303030303a3a313439393336353333363a3a445148697036443655376d46335761795a32747337794478737a71687779367a5a6a20494f42447a557167773\""));
    CScript finalScript;

    EnsureWalletIsUnlocked();
    opcodetype opcode = getOpcodeFromString(request.params[0].get_str());

    if (negatif == opcode)
        throw std::runtime_error("OP_CODE is either not a Fluid OP_CODE or is invalid");

    if (!IsHex(request.params[1].get_str()))
        throw std::runtime_error("Hex isn't even valid!");
    else
        finalScript = CScript() << opcode << ParseHex(request.params[1].get_str());

    std::string message;

    if (!fluid.CheckIfQuorumExists(ScriptToAsmStr(finalScript), message))
        throw std::runtime_error("Instruction does not meet required quorum for validity");

    if (opcode == OP_MINT || opcode == OP_REWARD_MINING || opcode == OP_REWARD_DYNODE) {
        CWalletTx wtx;
        SendCustomTransaction(finalScript, wtx, fluid.FLUID_TRANSACTION_COST, false);
        return wtx.GetHash().GetHex();
    } else {
        throw std::runtime_error(strprintf("OP_CODE, %s, not implemented yet!", request.params[0].get_str()));
    }
}

UniValue signtoken(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 2)
        throw std::runtime_error(
            "signtoken \"address\" \"tokenkey\"\n"
            "\nSign a Fluid Protocol Token\n"
            "\nArguments:\n"
            "1. \"address\"         (string, required) The Dynamic Address which will be used to sign.\n"
            "2. \"tokenkey\"         (string, required) The token which has to be initially signed\n"
            "\nExamples:\n" +
            HelpExampleCli("signtoken", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" \"3130303030303030303030303a3a313439393336353333363a3a445148697036443655376d46335761795a32747337794478737a71687779367a5a6a20494f42447a557167773\"") + HelpExampleRpc("signtoken", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" \"3130303030303030303030303a3a313439393336353333363a3a445148697036443655376d46335761795a32747337794478737a71687779367a5a6a20494f42447a557167773\""));
    std::string result;

    CDynamicAddress address(request.params[0].get_str());
    if (!address.IsValid())
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid Dynamic address");

    if (!fluid.VerifyAddressOwnership(address))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Address is not fluid protocol sovereign address");

    if (!fluid.VerifyAddressOwnership(address))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Address is not possessed by wallet!");

    std::string r = request.params[1].get_str();

    if (!IsHex(r))
        throw std::runtime_error("Hex isn't even valid! Cannot process ahead...");

    fluid.ConvertToString(r);

    if (!fluid.GenericSignMessage(r, result, address))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Message signing failed");

    return result;
}

UniValue verifyquorum(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "verifyquorum \"tokenkey\"\n"
            "\nVerify if the token provided has required quorum\n"
            "\nArguments:\n"
            "1. \"tokenkey\"         (string, required) The token which has to be initially signed\n"
            "\nExamples:\n" +
            HelpExampleCli("verifyquorum", "\"3130303030303030303030303a3a313439393336353333363a3a445148697036443655376d46335761795a32747337794478737a71687779367a5a6a20494f42447a557167773\"") + HelpExampleRpc("verifyquorum", "\"3130303030303030303030303a3a313439393336353333363a3a445148697036443655376d46335761795a32747337794478737a71687779367a5a6a20494f42447a557167773\""));
    std::string message;

    if (!fluid.CheckNonScriptQuorum(request.params[0].get_str(), message, false))
        throw std::runtime_error("Instruction does not meet minimum quorum for validity");

    return "Quorum is present!";
}

UniValue consenttoken(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 2)
        throw std::runtime_error(
            "consenttoken \"address\" \"tokenkey\"\n"
            "\nGive consent to a Fluid Protocol Token as a second party\n"
            "\nArguments:\n"
            "1. \"address\"         (string, required) The Dynamic Address which will be used to give consent.\n"
            "2. \"tokenkey\"         (string, required) The token which has to be been signed by one party\n"
            "\nExamples:\n" +
            HelpExampleCli("consenttoken", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" \"3130303030303030303030303a3a313439393336353333363a3a445148697036443655376d46335761795a32747337794478737a71687779367a5a6a20494f42447a557167773\"") + HelpExampleRpc("consenttoken", "\"D5nRy9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf\" \"3130303030303030303030303a3a313439393336353333363a3a445148697036443655376d46335761795a32747337794478737a71687779367a5a6a20494f42447a557167773\""));
    std::string result;

    CDynamicAddress address(request.params[0].get_str());
    if (!address.IsValid())
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid Dynamic address");

    if (!IsHex(request.params[1].get_str()))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Hex string is invalid! Token incorrect");

    if (!IsSovereignAddress(address))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Address is not fluid protocol sovereign address");

    if (!fluid.VerifyAddressOwnership(address))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Address is not possessed by wallet!");

    std::string message;

    if (!fluid.CheckNonScriptQuorum(request.params[1].get_str(), message, true))
        throw std::runtime_error("Instruction does not meet minimum quorum for validity");

    if (!fluid.GenericConsentMessage(request.params[1].get_str(), result, address))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Message signing failed");

    return result;
}

UniValue getfluidhistoryraw(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 0)
        throw std::runtime_error(
            "getfluidhistoryraw\n"
            "\nReturns raw data about each fluid command confirmed on the Dynamic blockchain.\n"
            "\nResult:\n"
            "{                   (json array of string)\n"
            "  \"fluid_command\"     (string) The operation code and raw fluid script command\n"
            "}, ...\n"
            "\nExamples\n" +
            HelpExampleCli("getfluidhistoryraw", "") + HelpExampleRpc("getfluidhistoryraw", ""));
    UniValue ret(UniValue::VOBJ);
    CAmount totalMintedCoins = 0;
    CAmount totalFluidTxCost = 0;
    int nTotal = 0;

    UniValue oMints(UniValue::VOBJ);
    // load fluid mint transaction history
    {
        std::vector<CFluidMint> mintEntries;
        if (!GetAllFluidMintRecords(mintEntries)) {
            throw std::runtime_error("GET_FLUID_HISTORY_RPC_ERROR: ERRCODE: 4000 - " + _("Error getting fluid mint entries"));
        }
        int x = 1;
        for (const CFluidMint& mintEntry : mintEntries) {
            UniValue obj(UniValue::VOBJ);
            obj.push_back(Pair("fluid_script", StringFromCharVector(mintEntry.FluidScript)));
            std::string addLabel = "mint_" + std::to_string(x);
            oMints.push_back(Pair(addLabel, obj));
            totalMintedCoins = totalMintedCoins + mintEntry.MintAmount;
            totalFluidTxCost = totalFluidTxCost + fluid.FLUID_TRANSACTION_COST;
            x++;
            nTotal++;
        }
    }
    ret.push_back(Pair("minting_history", oMints));
    // load fluid dynode update reward transaction history
    UniValue oDynodes(UniValue::VOBJ);
    {
        std::vector<CFluidDynode> dynodeEntries;
        if (!GetAllFluidDynodeRecords(dynodeEntries)) {
            throw std::runtime_error("GET_FLUID_HISTORY_RPC_ERROR: ERRCODE: 4002 - " + _("Error getting fluid dynode entries"));
        }
        int x = 1;
        for (const CFluidDynode& dynEntry : dynodeEntries) {
            UniValue obj(UniValue::VOBJ);
            obj.push_back(Pair("fluid_script", StringFromCharVector(dynEntry.FluidScript)));
            std::string addLabel = "reward_update_" + std::to_string(x);
            oDynodes.push_back(Pair(addLabel, obj));
            totalFluidTxCost = totalFluidTxCost + fluid.FLUID_TRANSACTION_COST;
            x++;
            nTotal++;
        }
    }
    ret.push_back(Pair("dynode_reward_history", oDynodes));
    // load fluid mining update reward transaction history
    UniValue oMining(UniValue::VOBJ);
    {
        std::vector<CFluidMining> miningEntries;
        if (!GetAllFluidMiningRecords(miningEntries)) {
            throw std::runtime_error("GET_FLUID_HISTORY_RPC_ERROR: ERRCODE: 4004 - " + _("Error getting fluid mining entries"));
        }
        int x = 1;
        for (const CFluidMining& miningEntry : miningEntries) {
            UniValue obj(UniValue::VOBJ);
            obj.push_back(Pair("fluid_script", StringFromCharVector(miningEntry.FluidScript)));
            std::string addLabel = "reward_update_" + std::to_string(x);
            oMining.push_back(Pair(addLabel, obj));
            totalFluidTxCost = totalFluidTxCost + fluid.FLUID_TRANSACTION_COST;
            x++;
            nTotal++;
        }
    }
    ret.push_back(Pair("mining_reward_history", oMining));
    // load fluid transaction summary
    UniValue oSummary(UniValue::VOBJ);
    {
        UniValue obj(UniValue::VOBJ);
        obj.push_back(Pair("total_minted", FormatMoney(totalMintedCoins)));
        obj.push_back(Pair("total_fluid_fee_cost", FormatMoney(totalFluidTxCost)));
        CAmount dynodeReward = GetFluidDynodeReward(chainActive.Tip()->nHeight);
        obj.push_back(Pair("current_dynode_reward", FormatMoney(dynodeReward)));

        CFluidMining lastMiningRecord;
        CAmount miningAmount = GetFluidMiningReward(chainActive.Tip()->nHeight);
        obj.push_back(Pair("current_mining_reward", FormatMoney(miningAmount)));
        obj.push_back(Pair("total_fluid_transactions", nTotal));
        oSummary.push_back(Pair("summary", obj));
    }
    ret.push_back(Pair("fluid_summary", oSummary));
    return ret;
}

UniValue getfluidhistory(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 0)
        throw std::runtime_error(
            "getfluidhistory\n"
            "\nReturns data about each fluid command confirmed on the Dynamic blockchain.\n"
            "\nResult:\n"
            "[                   (json array of object)\n"
            "  {                 (json object)\n"
            "  \"order\"               (string) order of execution.\n"
            "  \"operation\"           (string) The fluid operation code.\n"
            "  \"amount\"              (string) The fluid operation amount.\n"
            "  \"timestamp\"           (string) The fluid operation timestamp\n"
            "  \"payment address\"     (string) The fluid operation payment address\n"
            "  \"sovereign address 1\" (string) First sovereign signature address used\n"
            "  \"sovereign address 2\" (string) Second sovereign signature address used\n"
            "  \"sovereign address 3\" (string) Third sovereign signature address used\n"
            "  }, ...\n"
            "]\n"
            "\nExamples\n" +
            HelpExampleCli("getfluidhistory", "") + HelpExampleRpc("getfluidhistory", ""));
    UniValue ret(UniValue::VOBJ);
    CAmount totalMintedCoins = 0;
    CAmount totalFluidTxCost = 0;
    int nTotal = 0;

    UniValue oMints(UniValue::VOBJ);
    // load fluid mint transaction history
    {
        std::vector<CFluidMint> mintEntries;
        if (!GetAllFluidMintRecords(mintEntries)) {
            throw std::runtime_error("GET_FLUID_HISTORY_RPC_ERROR: ERRCODE: 4000 - " + _("Error getting fluid mint entries"));
        }
        int x = 1;
        for (const CFluidMint& mintEntry : mintEntries) {
            UniValue obj(UniValue::VOBJ);
            obj.push_back(Pair("operation", "Mint"));
            obj.push_back(Pair("amount", FormatMoney(mintEntry.MintAmount)));
            obj.push_back(Pair("timestamp", mintEntry.nTimeStamp));
            obj.push_back(Pair("block_height", (int)mintEntry.nHeight));
            obj.push_back(Pair("txid", mintEntry.txHash.GetHex()));
            obj.push_back(Pair("destination_address", StringFromCharVector(mintEntry.DestinationAddress)));
            int index = 1;
            for (const std::vector<unsigned char>& vchAddress : mintEntry.SovereignAddresses) {
                std::string addLabel = "sovereign_address_" + std::to_string(index);
                obj.push_back(Pair(addLabel, StringFromCharVector(vchAddress)));
                index++;
            }
            std::string addLabel = "mint_" + std::to_string(x);
            oMints.push_back(Pair(addLabel, obj));
            totalMintedCoins = totalMintedCoins + mintEntry.MintAmount;
            totalFluidTxCost = totalFluidTxCost + fluid.FLUID_TRANSACTION_COST;
            x++;
            nTotal++;
        }
    }
    ret.push_back(Pair("minting_history", oMints));
    // load fluid dynode update reward transaction history
    UniValue oDynodes(UniValue::VOBJ);
    {
        std::vector<CFluidDynode> dynodeEntries;
        if (!GetAllFluidDynodeRecords(dynodeEntries)) {
            throw std::runtime_error("GET_FLUID_HISTORY_RPC_ERROR: ERRCODE: 4002 - " + _("Error getting fluid dynode entries"));
        }
        int x = 1;
        for (const CFluidDynode& dynEntry : dynodeEntries) {
            UniValue obj(UniValue::VOBJ);
            obj.push_back(Pair("operation", "Dynode Reward Update"));
            obj.push_back(Pair("amount", FormatMoney(dynEntry.DynodeReward)));
            obj.push_back(Pair("timestamp", dynEntry.nTimeStamp));
            obj.push_back(Pair("block_height", (int)dynEntry.nHeight));
            obj.push_back(Pair("txid", dynEntry.txHash.GetHex()));
            int index = 1;
            for (const std::vector<unsigned char>& vchAddress : dynEntry.SovereignAddresses) {
                std::string addLabel = "sovereign_address_" + std::to_string(index);
                obj.push_back(Pair(addLabel, StringFromCharVector(vchAddress)));
                index++;
            }
            std::string addLabel = "reward_update_" + std::to_string(x);
            oDynodes.push_back(Pair(addLabel, obj));
            totalFluidTxCost = totalFluidTxCost + fluid.FLUID_TRANSACTION_COST;
            x++;
            nTotal++;
        }
    }
    ret.push_back(Pair("dynode_reward_history", oDynodes));
    // load fluid mining update reward transaction history
    UniValue oMining(UniValue::VOBJ);
    {
        std::vector<CFluidMining> miningEntries;
        if (!GetAllFluidMiningRecords(miningEntries)) {
            throw std::runtime_error("GET_FLUID_HISTORY_RPC_ERROR: ERRCODE: 4004 - " + _("Error getting fluid mining entries"));
        }
        int x = 1;
        for (const CFluidMining& miningEntry : miningEntries) {
            UniValue obj(UniValue::VOBJ);
            obj.push_back(Pair("operation", "Mining Reward Update"));
            obj.push_back(Pair("amount", FormatMoney(miningEntry.MiningReward)));
            obj.push_back(Pair("timestamp", miningEntry.nTimeStamp));
            obj.push_back(Pair("block_height", (int)miningEntry.nHeight));
            obj.push_back(Pair("txid", miningEntry.txHash.GetHex()));
            int index = 1;
            for (const std::vector<unsigned char>& vchAddress : miningEntry.SovereignAddresses) {
                std::string addLabel = "sovereign_address_" + std::to_string(index);
                obj.push_back(Pair(addLabel, StringFromCharVector(vchAddress)));
                index++;
            }
            std::string addLabel = "reward_update_" + std::to_string(x);
            oMining.push_back(Pair(addLabel, obj));
            totalFluidTxCost = totalFluidTxCost + fluid.FLUID_TRANSACTION_COST;
            x++;
            nTotal++;
        }
    }
    ret.push_back(Pair("mining_reward_history", oMining));
    // load fluid transaction summary
    UniValue oSummary(UniValue::VOBJ);
    {
        UniValue obj(UniValue::VOBJ);
        obj.push_back(Pair("total_minted", FormatMoney(totalMintedCoins)));
        obj.push_back(Pair("total_fluid_fee_cost", FormatMoney(totalFluidTxCost)));
        CAmount dynodeReward = GetFluidDynodeReward(chainActive.Tip()->nHeight);
        obj.push_back(Pair("current_dynode_reward", FormatMoney(dynodeReward)));
        CFluidMining lastMiningRecord;
        CAmount miningAmount = GetFluidMiningReward(chainActive.Tip()->nHeight);
        obj.push_back(Pair("current_mining_reward", FormatMoney(miningAmount)));
        obj.push_back(Pair("total_fluid_transactions", nTotal));
        oSummary.push_back(Pair("summary", obj));
    }
    ret.push_back(Pair("fluid_summary", oSummary));
    return ret;
}

UniValue getfluidsovereigns(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 0)
        throw std::runtime_error(
            "getfluidsovereigns\n"
            "\nReturns the active sovereign addresses.\n"
            "\nResult:\n"
            "{                         (json array of string)\n"
            "  \"sovereign address\"     (string) A sovereign address with permission to co-sign a fluid command\n"
            "}, ...\n"
            "\nExamples\n" +
            HelpExampleCli("getfluidsovereigns", "") + HelpExampleRpc("getfluidsovereigns", ""));

    UniValue ret(UniValue::VOBJ);
    std::vector<CFluidSovereign> sovereignEntries;
    if (!GetAllFluidSovereignRecords(sovereignEntries)) {
        throw std::runtime_error("GET_FLUID_SOVEREIGN_HISTORY_RPC_ERROR: ERRCODE: 4008 - " + _("Error getting fluid sovereign entries"));
    }
    int x = 1;
    UniValue obj(UniValue::VOBJ);
    for (const CFluidSovereign& sovereignEntry : sovereignEntries) {
        int index = 1;
        UniValue oEntry(UniValue::VOBJ);
        for (const std::vector<unsigned char>& vchAddress : sovereignEntry.SovereignAddresses) {
            std::string addLabel = "address_" + std::to_string(index);
            oEntry.push_back(Pair(addLabel, StringFromCharVector(vchAddress)));
            index++;
        }
        std::string addLabel = "sovereign" + std::to_string(x);
        obj.push_back(Pair(addLabel, oEntry));
        x++;
    }
    ret.push_back(Pair("history", obj));

    return ret;
}

static const CRPCCommand commands[] =
    {
        //  category              name                     actor (function)           okSafe argNames
        //  --------------------- ------------------------ -----------------------    ------ --------------------
#ifdef ENABLE_WALLET
        /* Fluid Protocol */
        {"fluid", "sendfluidtransaction", &sendfluidtransaction, true, {"opcode", "hexstring"}},
        {"fluid", "signtoken", &signtoken, true, {"address", "tokenkey"}},
        {"fluid", "consenttoken", &consenttoken, true, {"address", "tokenkey"}},
        {"fluid", "getrawpubkey", &getrawpubkey, true, {"address"}},
        {"fluid", "verifyquorum", &verifyquorum, true, {"tokenkey"}},
        {"fluid", "maketoken", &maketoken, true, {"string"}},
        {"fluid", "getfluidhistory", &getfluidhistory, true, {}},
        {"fluid", "getfluidhistoryraw", &getfluidhistoryraw, true, {}},
        {"fluid", "getfluidsovereigns", &getfluidsovereigns, true, {}},
        {"fluid", "gettime", &gettime, true, {}},
#endif //ENABLE_WALLET
};

void RegisterFluidRPCCommands(CRPCTable& tableRPC)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        tableRPC.appendCommand(commands[vcidx].name, &commands[vcidx]);
}