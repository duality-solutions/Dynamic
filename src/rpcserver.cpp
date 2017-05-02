// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "rpcserver.h"

#include "base58.h"
#include "init.h"
#include "random.h"
#include "sync.h"
#include "ui_interface.h"
#include "util.h"
#include "utilstrencodings.h"

#include <univalue.h>

#include <memory> // for unique_ptr

#include <boost/bind.hpp>
#include <boost/algorithm/string/case_conv.hpp> // for to_upper()
#include <boost/iostreams/concepts.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/signals2/signal.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/thread.hpp>

using namespace RPCServer;

static bool fRPCRunning = false;
static bool fRPCInWarmup = true;
static std::string rpcWarmupStatus("RPC server started");
static CCriticalSection cs_rpcWarmup;
/* Timer-creating functions */
static RPCTimerInterface* timerInterface = NULL;
/* Map of name to timer. */
static std::map<std::string, std::unique_ptr<RPCTimerBase> > deadlineTimers;

static struct CRPCSignals
{
    boost::signals2::signal<void ()> Started;
    boost::signals2::signal<void ()> Stopped;
    boost::signals2::signal<void (const CRPCCommand&)> PreCommand;
    boost::signals2::signal<void (const CRPCCommand&)> PostCommand;
} g_rpcSignals;

void RPCServer::OnStarted(boost::function<void ()> slot)
{
    g_rpcSignals.Started.connect(slot);
}

void RPCServer::OnStopped(boost::function<void ()> slot)
{
    g_rpcSignals.Stopped.connect(slot);
}

void RPCServer::OnPreCommand(boost::function<void (const CRPCCommand&)> slot)
{
    g_rpcSignals.PreCommand.connect(boost::bind(slot, _1));
}

void RPCTypeCheck(const UniValue& params,
                  const std::list<UniValue::VType>& typesExpected,
                  bool fAllowNull)
{
    unsigned int i = 0;
    BOOST_FOREACH(UniValue::VType t, typesExpected)
    {
        if (params.size() <= i)
            break;

        const UniValue& v = params[i];
        if (!((v.type() == t) || (fAllowNull && (v.isNull()))))
        {
            std::string err = strprintf("Expected type %s, got %s",
                                   uvTypeName(t), uvTypeName(v.type()));
            throw JSONRPCError(RPC_TYPE_ERROR, err);
        }
        i++;
    }
}

void RPCTypeCheckObj(const UniValue& o,
                  const std::map<std::string, UniValue::VType>& typesExpected,
                  bool fAllowNull)
{
    BOOST_FOREACH(const PAIRTYPE(std::string, UniValue::VType)& t, typesExpected)
    {
        const UniValue& v = find_value(o, t.first);
        if (!fAllowNull && v.isNull())
            throw JSONRPCError(RPC_TYPE_ERROR, strprintf("Missing %s", t.first));

        if (!((v.type() == t.second) || (fAllowNull && (v.isNull()))))
        {
            std::string err = strprintf("Expected type %s for %s, got %s",
                                   uvTypeName(t.second), t.first, uvTypeName(v.type()));
            throw JSONRPCError(RPC_TYPE_ERROR, err);
        }
    }
}

CAmount AmountFromValue(const UniValue& value)
{
    if (!value.isNum() && !value.isStr())
        throw JSONRPCError(RPC_TYPE_ERROR, "Amount is not a number or string");
    CAmount amount;
    if (!ParseFixedPoint(value.getValStr(), 8, &amount))
        throw JSONRPCError(RPC_TYPE_ERROR, "Invalid amount");
    if (!MoneyRange(amount))
        throw JSONRPCError(RPC_TYPE_ERROR, "Amount out of range");
    return amount;
}

UniValue ValueFromAmount(const CAmount& amount)
{
    bool sign = amount < 0;
    int64_t n_abs = (sign ? -amount : amount);
    int64_t quotient = n_abs / COIN;
    int64_t remainder = n_abs % COIN;
    return UniValue(UniValue::VNUM,
            strprintf("%s%d.%08d", sign ? "-" : "", quotient, remainder));
}

uint256 ParseHashV(const UniValue& v, std::string strName)
{
    std::string strHex;
    if (v.isStr())
        strHex = v.get_str();
    if (!IsHex(strHex)) // Note: IsHex("") is false
        throw JSONRPCError(RPC_INVALID_PARAMETER, strName+" must be hexadecimal string (not '"+strHex+"')");
    uint256 result;
    result.SetHex(strHex);
    return result;
}
uint256 ParseHashO(const UniValue& o, std::string strKey)
{
    return ParseHashV(find_value(o, strKey), strKey);
}
std::vector<unsigned char> ParseHexV(const UniValue& v, std::string strName)
{
    std::string strHex;
    if (v.isStr())
        strHex = v.get_str();
    if (!IsHex(strHex))
        throw JSONRPCError(RPC_INVALID_PARAMETER, strName+" must be hexadecimal string (not '"+strHex+"')");
    return ParseHex(strHex);
}

std::vector<unsigned char> ParseHexO(const UniValue& o, std::string strKey)
{
    return ParseHexV(find_value(o, strKey), strKey);
}

/**
 * Note: This interface may still be subject to change.
 */

std::string CRPCTable::help(const std::string& strCommand) const
{
    std::string strRet;
    std::string category;
    std::set<rpcfn_type> setDone;
    std::vector<std::pair<std::string, const CRPCCommand*> > vCommands;

    for (std::map<std::string, const CRPCCommand*>::const_iterator mi = mapCommands.begin(); mi != mapCommands.end(); ++mi)
        vCommands.push_back(make_pair(mi->second->category + mi->first, mi->second));
    sort(vCommands.begin(), vCommands.end());

    BOOST_FOREACH(const PAIRTYPE(std::string, const CRPCCommand*)& command, vCommands)
    {
        const CRPCCommand *pcmd = command.second;
        std::string strMethod = pcmd->name;
        // We already filter duplicates, but these deprecated screw up the sort order
        if (strMethod.find("label") != std::string::npos)
            continue;
        if ((strCommand != "" || pcmd->category == "hidden") && strMethod != strCommand)
            continue;
        try
        {
            UniValue params;
            rpcfn_type pfn = pcmd->actor;
            if (setDone.insert(pfn).second)
                (*pfn)(params, true);
        }
        catch (const std::exception& e)
        {
            // Help text is returned in an exception
            std::string strHelp = std::string(e.what());
            if (strCommand == "")
            {
                if (strHelp.find('\n') != std::string::npos)
                    strHelp = strHelp.substr(0, strHelp.find('\n'));

                if (category != pcmd->category)
                {
                    if (!category.empty())
                        strRet += "\n";
                    category = pcmd->category;
                    std::string firstLetter = category.substr(0,1);
                    boost::to_upper(firstLetter);
                    strRet += "== " + firstLetter + category.substr(1) + " ==\n";
                }
            }
            strRet += strHelp + "\n";
        }
    }
    if (strRet == "")
        strRet = strprintf("help: unknown command: %s\n", strCommand);
    strRet = strRet.substr(0,strRet.size()-1);
    return strRet;
}

UniValue help(const UniValue& params, bool fHelp)
{
    if (fHelp || params.size() > 1)
        throw std::runtime_error(
            "help ( \"command\" )\n"
            "\nList all commands, or get help for a specified command.\n"
            "\nArguments:\n"
            "1. \"command\"     (string, optional) The command to get help on\n"
            "\nResult:\n"
            "\"text\"     (string) The help text\n"
        );

    std::string strCommand;
    if (params.size() > 0)
        strCommand = params[0].get_str();

    return tableRPC.help(strCommand);
}


UniValue stop(const UniValue& params, bool fHelp)
{
    // Accept the deprecated and ignored 'detach' boolean argument
    if (fHelp || params.size() > 1)
        throw std::runtime_error(
            "stop\n"
            "\nStop Dynamic server.");
    // Event loop will exit after current HTTP requests have been handled, so
    // this reply will get back to the client.
    StartShutdown();
    return "Dynamic server stopping";
}

/**
 * Call Table
 */
static const CRPCCommand vRPCCommands[] =
{ //  category              name                      actor (function)         okSafeMode
  //  --------------------- ------------------------  -----------------------  ----------
    /* Overall control/query calls */
    { "Control",            "getinfo",                &getinfo,                true  }, /* uses wallet if enabled */
    { "Control",            "debug",                  &debug,                  true  },
    { "Control",            "help",                   &help,                   true  },
    { "Control",            "stop",                   &stop,                   true  },
    { "Control",            "getmemoryinfo",          &getmemoryinfo,          true  },

    /* P2P networking */
    { "Network",            "getnetworkinfo",         &getnetworkinfo,         true  },
    { "Network",            "addnode",                &addnode,                true  },
    { "Network",            "disconnectnode",         &disconnectnode,         true  },
    { "Network",            "getaddednodeinfo",       &getaddednodeinfo,       true  },
    { "Network",            "getconnectioncount",     &getconnectioncount,     true  },
    { "Network",            "getnettotals",           &getnettotals,           true  },
    { "Network",            "getpeerinfo",            &getpeerinfo,            true  },
    { "Network",            "ping",                   &ping,                   true  },
    { "Network",            "setban",                 &setban,                 true  },
    { "Network",            "listbanned",             &listbanned,             true  },
    { "Network",            "clearbanned",            &clearbanned,            true  },

    /* Block chain and UTXO */
    { "Blockchain",         "getblockchaininfo",      &getblockchaininfo,      true  },
    { "Blockchain",         "getbestblockhash",       &getbestblockhash,       true  },
    { "Blockchain",         "getblockcount",          &getblockcount,          true  },
    { "Blockchain",         "getblock",               &getblock,               true  },
    { "Blockchain",         "getblockhashes",         &getblockhashes,         true  },
    { "Blockchain",         "getblockhash",           &getblockhash,           true  },
    { "Blockchain",         "getblockheader",         &getblockheader,         true  },
    { "Blockchain",         "getblockheaders",        &getblockheaders,        true  },
    { "Blockchain",         "getchaintips",           &getchaintips,           true  },
    { "Blockchain",         "getdifficulty",          &getdifficulty,          true  },
    { "Blockchain",         "getmempoolinfo",         &getmempoolinfo,         true  },
    { "Blockchain",         "getrawmempool",          &getrawmempool,          true  },
    { "Blockchain",         "gettxout",               &gettxout,               true  },
    { "Blockchain",         "gettxoutproof",          &gettxoutproof,          true  },
    { "Blockchain",         "verifytxoutproof",       &verifytxoutproof,       true  },
    { "Blockchain",         "gettxoutsetinfo",        &gettxoutsetinfo,        true  },
    { "Blockchain",         "verifychain",            &verifychain,            true  },
    { "Blockchain",         "getspentinfo",           &getspentinfo,           false },

    /* Mining */
    { "Mining",             "getblocktemplate",       &getblocktemplate,       true  },
    { "Mining",             "getwork",				  &getwork,       		   true  },
    { "Mining",             "getmininginfo",          &getmininginfo,          true  },
    { "Mining",             "getnetworkhashps",       &getnetworkhashps,       true  },
    { "Mining",             "getpowrewardstart",      &getpowrewardstart,      true  },
    { "Mining",             "prioritisetransaction",  &prioritisetransaction,  true  },
    { "Mining",             "submitblock",            &submitblock,            true  },

    /* Coin generation */
    { "Generating",         "getgenerate",            &getgenerate,            true  },
    { "Generating",         "setgenerate",            &setgenerate,            true  },
    { "Generating",         "generate",               &generate,               true  },
    { "Generating",         "gethashespersec",        &gethashespersec,        true  },

    /* Raw transactions */
    { "Raw Transactions",    "createrawtransaction",   &createrawtransaction,   true  },
    { "Raw Transactions",    "decoderawtransaction",   &decoderawtransaction,   true  },
    { "Raw Transactions",    "decodescript",           &decodescript,           true  },
    { "Raw Transactions",    "getrawtransaction",      &getrawtransaction,      true  },
    { "Raw Transactions",    "sendrawtransaction",     &sendrawtransaction,     false },
    { "Raw Transactions",    "signrawtransaction",     &signrawtransaction,     false }, /* uses wallet if enabled */
#ifdef ENABLE_WALLET
    { "Raw Transactions",    "fundrawtransaction",     &fundrawtransaction,     false },
#endif

    /* Address index */
    { "Address Index",       "getaddressmempool",      &getaddressmempool,      true  },
    { "Address Index",       "getaddressutxos",        &getaddressutxos,        false },
    { "Address Index",       "getaddressdeltas",       &getaddressdeltas,       false },
    { "Address Index",       "getaddresstxids",        &getaddresstxids,        false },
    { "Address Index",       "getaddressbalance",      &getaddressbalance,      false },

    /* Utility functions */
    { "Utility Function",               "createmultisig",         &createmultisig,         true  },
    { "Utility Function",               "validateaddress",        &validateaddress,        true  }, /* uses wallet if enabled */
    { "Utility Function",               "verifymessage",          &verifymessage,          true  },
    { "Utility Function",               "estimatefee",            &estimatefee,            true  },
    { "Utility Function",               "estimatepriority",       &estimatepriority,       true  },
    { "Utility Function",               "estimatesmartfee",       &estimatesmartfee,       true  },
    { "Utility Function",               "estimatesmartpriority",  &estimatesmartpriority,  true  },

    /* Not shown in help */
    { "Hidden",             "invalidateblock",        &invalidateblock,        true  },
    { "Hidden",             "reconsiderblock",        &reconsiderblock,        true  },
    { "Hidden",             "setmocktime",            &setmocktime,            true  },
#ifdef ENABLE_WALLET
    { "Hidden",                 "resendwallettransactions", &resendwallettransactions, true},
#endif

    /* Dynamic features */
    { "Dynamic",                "dynode",                 &dynode,                 true  },
    { "Dynamic",                "dynodelist",             &dynodelist,             true  },
    { "Dynamic",                "dynodebroadcast",        &dynodebroadcast,        true  },
    { "Dynamic",                "gobject",                &gobject,                true  },
    { "Dynamic",                "getgovernanceinfo",      &getgovernanceinfo,      true  },
    { "Dynamic",                "getdynoderewardstart",   &getdynoderewardstart,   true  },
    { "Dynamic",                "getsuperblockbudget",    &getsuperblockbudget,    true  },
    { "Dynamic",                "voteraw",                &voteraw,                true  },
    { "Dynamic",                "dnsync",                 &dnsync,                 true  },
    { "Dynamic",                "spork",                  &spork,                  true  },
    { "Dynamic",                "getpoolinfo",            &getpoolinfo,            true  },
#ifdef ENABLE_WALLET
    { "Dynamic",                "privatesend",            &privatesend,            false },

    /* Wallet */
    { "Wallet",             "keepass",                &keepass,                true },
    { "Wallet",             "instantsendtoaddress",   &instantsendtoaddress,   false },
    { "Wallet",             "addmultisigaddress",     &addmultisigaddress,     true  },
    { "Wallet",             "backupwallet",           &backupwallet,           true  },
    { "Wallet",             "dumpprivkey",            &dumpprivkey,            true  },
    { "Wallet",             "dumpwallet",             &dumpwallet,             true  },
    { "Wallet",             "dumphdinfo",             &dumphdinfo,             true  },
    { "Wallet",             "encryptwallet",          &encryptwallet,          true  },
    { "Wallet",             "getaccountaddress",      &getaccountaddress,      true  },
    { "Wallet",             "getaccount",             &getaccount,             true  },
    { "Wallet",             "getaddressesbyaccount",  &getaddressesbyaccount,  true  },
    { "Wallet",             "getbalance",             &getbalance,             false },
    { "Wallet",             "getnewaddress",          &getnewaddress,          true  },
    { "Wallet",             "getrawchangeaddress",    &getrawchangeaddress,    true  },
    { "Wallet",             "getreceivedbyaccount",   &getreceivedbyaccount,   false },
    { "Wallet",             "getreceivedbyaddress",   &getreceivedbyaddress,   false },
    { "Wallet",             "gettransaction",         &gettransaction,         false },
    { "Wallet",             "abandontransaction",     &abandontransaction,     false },
    { "Wallet",             "getunconfirmedbalance",  &getunconfirmedbalance,  false },
    { "Wallet",             "getwalletinfo",          &getwalletinfo,          false },
    { "Wallet",             "importprivkey",          &importprivkey,          true  },
    { "Wallet",             "importwallet",           &importwallet,           true  },
    { "Wallet",             "importaddress",          &importaddress,          true  },
    { "Wallet",             "importpubkey",           &importpubkey,           true  },
    { "Wallet",             "keypoolrefill",          &keypoolrefill,          true  },
    { "Wallet",             "listaccounts",           &listaccounts,           false },
    { "Wallet",             "listaddressgroupings",   &listaddressgroupings,   false },
    { "Wallet",             "listlockunspent",        &listlockunspent,        false },
    { "Wallet",             "listreceivedbyaccount",  &listreceivedbyaccount,  false },
    { "Wallet",             "listreceivedbyaddress",  &listreceivedbyaddress,  false },
    { "Wallet",             "listsinceblock",         &listsinceblock,         false },
    { "Wallet",             "listtransactions",       &listtransactions,       false },
    { "Wallet",             "listunspent",            &listunspent,            false },
    { "Wallet",             "lockunspent",            &lockunspent,            true  },
    { "Wallet",             "makekeypair",            &makekeypair,            false },
    { "Wallet",             "move",                   &movecmd,                false },
    { "Wallet",             "sendfrom",               &sendfrom,               false },
    { "Wallet",             "sendmany",               &sendmany,               false },
    { "Wallet",             "sendtoaddress",          &sendtoaddress,          false },
    { "Wallet",             "setaccount",             &setaccount,             true  },
    { "Wallet",             "settxfee",               &settxfee,               true  },
    { "Wallet",             "signmessage",            &signmessage,            true  },
    { "Wallet",             "walletlock",             &walletlock,             true  },
    { "Wallet",             "walletpassphrasechange", &walletpassphrasechange, true  },
    { "Wallet",             "walletpassphrase",       &walletpassphrase,       true  },
     /* Decentralised DNS */
    { "DDNS",               "name_scan",              &name_scan,              true  },
    { "DDNS",               "name_filter",            &name_filter,            true  },
    { "DDNS",               "name_show",              &name_show,              true  },
    { "DDNS",               "name_history",           &name_history,           true  },
    { "DDNS",               "name_mempool",           &name_mempool,           true  },
    { "DDNS",               "name_new",               &name_new,               true  },
    { "DDNS",               "name_update",            &name_update,            true  },
    { "DDNS",               "name_delete",            &name_delete,            true  },
    { "DDNS",               "name_list",              &name_list,              true  },
#endif // ENABLE_WALLET
    /* Not shown in help */
    { "Hidden", "name_debug", &name_debug, false },
};

CRPCTable::CRPCTable()
{
    unsigned int vcidx;
    for (vcidx = 0; vcidx < (sizeof(vRPCCommands) / sizeof(vRPCCommands[0])); vcidx++)
    {
        const CRPCCommand *pcmd;

        pcmd = &vRPCCommands[vcidx];
        mapCommands[pcmd->name] = pcmd;
    }
}

const CRPCCommand *CRPCTable::operator[](const std::string &name) const
{
    std::map<std::string, const CRPCCommand*>::const_iterator it = mapCommands.find(name);
    if (it == mapCommands.end())
        return NULL;
    return (*it).second;
}

bool StartRPC()
{
    LogPrint("rpc", "Starting RPC\n");
    fRPCRunning = true;
    g_rpcSignals.Started();
    return true;
}

void InterruptRPC()
{
    LogPrint("rpc", "Interrupting RPC\n");
    // Interrupt e.g. running longpolls
    fRPCRunning = false;
}

void StopRPC()
{
    LogPrint("rpc", "Stopping RPC\n");
    deadlineTimers.clear();
    g_rpcSignals.Stopped();
}

bool IsRPCRunning()
{
    return fRPCRunning;
}

void SetRPCWarmupStatus(const std::string& newStatus)
{
    LOCK(cs_rpcWarmup);
    rpcWarmupStatus = newStatus;
}

void SetRPCWarmupFinished()
{
    LOCK(cs_rpcWarmup);
    assert(fRPCInWarmup);
    fRPCInWarmup = false;
}

bool RPCIsInWarmup(std::string *outStatus)
{
    LOCK(cs_rpcWarmup);
    if (outStatus)
        *outStatus = rpcWarmupStatus;
    return fRPCInWarmup;
}

void JSONRequest::parse(const UniValue& valRequest)
{
    // Parse request
    if (!valRequest.isObject())
        throw JSONRPCError(RPC_INVALID_REQUEST, "Invalid Request object");
    const UniValue& request = valRequest.get_obj();

    // Parse id now so errors from here on will have the id
    id = find_value(request, "id");

    // Parse method
    UniValue valMethod = find_value(request, "method");
    if (valMethod.isNull())
        throw JSONRPCError(RPC_INVALID_REQUEST, "Missing method");
    if (!valMethod.isStr())
        throw JSONRPCError(RPC_INVALID_REQUEST, "Method must be a string");
    strMethod = valMethod.get_str();
    if (strMethod != "getblocktemplate")
        LogPrint("rpc", "ThreadRPCServer method=%s\n", SanitizeString(strMethod));

    // Parse params
    UniValue valParams = find_value(request, "params");
    if (valParams.isArray())
        params = valParams.get_array();
    else if (valParams.isNull())
        params = UniValue(UniValue::VARR);
    else
        throw JSONRPCError(RPC_INVALID_REQUEST, "Params must be an array");
}

static UniValue JSONRPCExecOne(const UniValue& req)
{
    UniValue rpc_result(UniValue::VOBJ);

    JSONRequest jreq;
    try {
        jreq.parse(req);

        UniValue result = tableRPC.execute(jreq.strMethod, jreq.params);
        rpc_result = JSONRPCReplyObj(result, NullUniValue, jreq.id);
    }
    catch (const UniValue& objError)
    {
        rpc_result = JSONRPCReplyObj(NullUniValue, objError, jreq.id);
    }
    catch (const std::exception& e)
    {
        rpc_result = JSONRPCReplyObj(NullUniValue,
                                     JSONRPCError(RPC_PARSE_ERROR, e.what()), jreq.id);
    }

    return rpc_result;
}

std::string JSONRPCExecBatch(const UniValue& vReq)
{
    UniValue ret(UniValue::VARR);
    for (unsigned int reqIdx = 0; reqIdx < vReq.size(); reqIdx++)
        ret.push_back(JSONRPCExecOne(vReq[reqIdx]));

    return ret.write() + "\n";     
}

UniValue CRPCTable::execute(const std::string &strMethod, const UniValue &params) const
{
    // Return immediately if in warmup
    {
        LOCK(cs_rpcWarmup);
        if (fRPCInWarmup)
            throw JSONRPCError(RPC_IN_WARMUP, rpcWarmupStatus);
    }

    // Find method
    const CRPCCommand *pcmd = tableRPC[strMethod];
    if (!pcmd)
        throw JSONRPCError(RPC_METHOD_NOT_FOUND, "Method not found");

    g_rpcSignals.PreCommand(*pcmd);

    try
    {
        // Execute
        return pcmd->actor(params, false);
    }
    catch (const std::exception& e)
    {
        throw JSONRPCError(RPC_MISC_ERROR, e.what());
    }

    g_rpcSignals.PostCommand(*pcmd);
}

std::vector<std::string> CRPCTable::listCommands() const
{
    std::vector<std::string> commandList;
    typedef std::map<std::string, const CRPCCommand*> commandMap;

    std::transform( mapCommands.begin(), mapCommands.end(),
                   std::back_inserter(commandList),
                   boost::bind(&commandMap::value_type::first,_1) );
    return commandList;
}

std::string HelpExampleCli(const std::string& methodname, const std::string& args)
{
    return "> dynamic-cli " + methodname + " " + args + "\n";
}

std::string HelpExampleRpc(const std::string& methodname, const std::string& args)
{
    return "> curl --user myusername --data-binary '{\"jsonrpc\": \"1.0\", \"id\":\"curltest\", "
        "\"method\": \"" + methodname + "\", \"params\": [" + args + "] }' -H 'content-type: text/plain;' http://127.0.0.1:31350/\n";
}

void RPCSetTimerInterfaceIfUnset(RPCTimerInterface *iface)
{
    if (!timerInterface)
        timerInterface = iface;
}

void RPCSetTimerInterface(RPCTimerInterface *iface)
{
    timerInterface = iface;
}

void RPCUnsetTimerInterface(RPCTimerInterface *iface)
{
    if (timerInterface == iface)
        timerInterface = NULL;
}

void RPCRunLater(const std::string& name, boost::function<void(void)> func, int64_t nSeconds)
{
    if (!timerInterface)
        throw JSONRPCError(RPC_INTERNAL_ERROR, "No timer handler registered for RPC");
    deadlineTimers.erase(name);
    LogPrint("rpc", "queue run of timer %s in %i seconds (using %s)\n", name, nSeconds, timerInterface->Name());
    deadlineTimers.emplace(name, std::unique_ptr<RPCTimerBase>(timerInterface->NewTimer(func, nSeconds*1000)));
}

const CRPCTable tableRPC;
