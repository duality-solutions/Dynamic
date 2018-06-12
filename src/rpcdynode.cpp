// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "activedynode.h"
#include "base58.h"
#include "clientversion.h"
#include "dynode-payments.h"
#include "dynode-sync.h"
#include "dynodeconfig.h"
#include "dynodeman.h"
#include "init.h"
#include "validation.h"
#ifdef ENABLE_WALLET
#include "privatesend-client.h"
#endif // ENABLE_WALLET
#include "privatesend-server.h"
#include "rpcserver.h"
#include "util.h"
#include "utilmoneystr.h"

#include <univalue.h>

#include <fstream>
#include <iomanip>

UniValue dynodelist(const JSONRPCRequest& request);

#ifdef ENABLE_WALLET
void EnsureWalletIsUnlocked();

UniValue privatesend(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "privatesend \"command\"\n"
            "\nArguments:\n"
            "1. \"command\"        (string or set of strings, required) The command to execute\n"
            "\nAvailable commands:\n"
            "  start       - Start mixing\n"
            "  stop        - Stop mixing\n"
            "  reset       - Reset mixing\n"
            );

    if(request.params[0].get_str() == "start") {
        {
            LOCK(pwalletMain->cs_wallet);
            EnsureWalletIsUnlocked();
        }

        if(fDyNode)
            return "Mixing is not supported from Dynodes";

        privateSendClient.fEnablePrivateSend = true;
        bool result = privateSendClient.DoAutomaticDenominating(*g_connman);
        return "Mixing " + (result ? "started successfully" : ("start failed: " + privateSendClient.GetStatus() + ", will retry"));
    }

    if(request.params[0].get_str() == "stop") {
        privateSendClient.fEnablePrivateSend = false;
        return "Mixing was stopped";
    }

    if(request.params[0].get_str() == "reset") {
        privateSendClient.ResetPool();
        return "Mixing was reset";
    }

    return "Unknown command, please see \"help privatesend\"";
}
#endif // ENABLE_WALLET

UniValue getpoolinfo(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 0)
        throw std::runtime_error(
            "getpoolinfo\n"
            "Returns an object containing mixing pool related information.\n");

#ifdef ENABLE_WALLET
    CPrivateSendBase* pprivateSendBase = fDyNode ? (CPrivateSendBase*)&privateSendServer : (CPrivateSendBase*)&privateSendClient;

    UniValue obj(UniValue::VOBJ);
    obj.push_back(Pair("state",             pprivateSendBase->GetStateString()));
    obj.push_back(Pair("mixing_mode",       (!fDyNode && privateSendClient.fPrivateSendMultiSession) ? "multi-session" : "normal"));
    obj.push_back(Pair("queue",             pprivateSendBase->GetQueueSize()));
    obj.push_back(Pair("entries",           pprivateSendBase->GetEntriesCount()));
    obj.push_back(Pair("status",            privateSendClient.GetStatus()));

    dynode_info_t dnInfo;
    if (privateSendClient.GetMixingDynodeInfo(dnInfo)) {
        obj.push_back(Pair("outpoint",      dnInfo.vin.prevout.ToStringShort()));
        obj.push_back(Pair("addr",          dnInfo.addr.ToString()));
    }

    if (pwalletMain) {
        obj.push_back(Pair("keys_left",     pwalletMain->nKeysLeftSinceAutoBackup));
        obj.push_back(Pair("warnings",      pwalletMain->nKeysLeftSinceAutoBackup < PRIVATESEND_KEYS_THRESHOLD_WARNING
                                                ? "WARNING: keypool is almost depleted!" : ""));
    }
#else // ENABLE_WALLET
    UniValue obj(UniValue::VOBJ);
    obj.push_back(Pair("state",             privateSendServer.GetStateString()));
    obj.push_back(Pair("queue",             privateSendServer.GetQueueSize()));
    obj.push_back(Pair("entries",           privateSendServer.GetEntriesCount()));
#endif // ENABLE_WALLET

    return obj;
}


UniValue dynode(const JSONRPCRequest& request)
{
    std::string strCommand;
    if (request.params.size() >= 1) {
        strCommand = request.params[0].get_str();
    }

#ifdef ENABLE_WALLET
    if (strCommand == "start-many")
        throw JSONRPCError(RPC_INVALID_PARAMETER, "DEPRECATED, please use start-all instead");
#endif // ENABLE_WALLET

    if (request.fHelp  ||
        (
#ifdef ENABLE_WALLET
            strCommand != "start-alias" && strCommand != "start-all" && strCommand != "start-missing" &&
         strCommand != "start-disabled" && strCommand != "outputs" &&
#endif // ENABLE_WALLET
         strCommand != "list" && strCommand != "list-conf" && strCommand != "count" &&
         strCommand != "debug" && strCommand != "current" && strCommand != "winner" && strCommand != "winners" && strCommand != "genkey" &&
         strCommand != "connect" && strCommand != "status"))
            throw std::runtime_error(
                "dynode \"command\"...\n"
                "Set of commands to execute dynode-sync related actions\n"
                "\nArguments:\n"
                "1. \"command\"        (string or set of strings, required) The command to execute\n"
                "\nAvailable commands:\n"
                "  count        - Print number of all known Dynodes (optional: 'ps', 'enabled', 'all', 'qualify')\n"
                "  current      - Print info on current Dynode winner to be paid the next block (calculated locally)\n"
                "  debug        - Print Dynode status\n"
                "  genkey       - Generate new dynodepairingkey\n"
#ifdef ENABLE_WALLET
                "  outputs      - Print Dynode compatible outputs\n"
                "  start-alias  - Start single remote Dynode by assigned alias configured in dynode.conf\n"
                "  start-<mode> - Start remote Dynodes configured in dynode.conf (<mode>: 'all', 'missing', 'disabled')\n"
#endif // ENABLE_WALLET
                "  status       - Print Dynode status information\n"
                "  list         - Print list of all known Dynodes (see dynodelist for more info)\n"
                "  list-conf    - Print dynode.conf in JSON format\n"
                "  winner       - Print info on next dynode winner to vote for\n"
                "  winners      - Print list of Dynode winners\n"
                );

    if (strCommand == "list")
    {
        JSONRPCRequest newRequest = request;
        newRequest.params.setArray();
        // forward params but skip "list"
        for (unsigned int i = 1; i < request.params.size(); i++) {
            newRequest.params.push_back(request.params[i]);
        }
        return dynodelist(newRequest);
    }

    if(strCommand == "connect")
    {
        if (request.params.size() < 2)
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Dynode address required");

        std::string strAddress = request.params[1].get_str();

        CService addr;
        if (!Lookup(strAddress.c_str(), addr, 0, false))
            throw JSONRPCError(RPC_INTERNAL_ERROR, strprintf("Incorrect dynode address %s", strAddress));

        // TODO: Pass CConnman instance somehow and don't use global variable.
        CNode *pnode = g_connman->ConnectNode(CAddress(addr, NODE_NETWORK), NULL);
        if(!pnode)
            throw JSONRPCError(RPC_INTERNAL_ERROR, strprintf("Couldn't connect to dynode %s", strAddress));

        return "successfully connected";
    }

    if (strCommand == "count")
    {
        if (request.params.size() > 2)
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Too many parameters");

        if (request.params.size() == 1)
            return dnodeman.size();

        std::string strMode = request.params[1].get_str();

        if (strMode == "ps")
            return dnodeman.CountEnabled(MIN_PRIVATESEND_PEER_PROTO_VERSION);

        if (strMode == "enabled")
            return dnodeman.CountEnabled();

        int nCount;
        dynode_info_t dnInfo;
        dnodeman.GetNextDynodeInQueueForPayment(true, nCount, dnInfo);

        if (strMode == "qualify")
            return nCount;

        if (strMode == "all")
            return strprintf("Total: %d (PS Compatible: %d / Enabled: %d / Qualify: %d)",
                dnodeman.size(), dnodeman.CountEnabled(MIN_PRIVATESEND_PEER_PROTO_VERSION),
                dnodeman.CountEnabled(), nCount);
    }

    if (strCommand == "current" || strCommand == "winner")
    {
        int nCount;
        int nHeight;
        dynode_info_t dnInfo;
        CBlockIndex* pindex = NULL;
        {
            LOCK(cs_main);
            pindex = chainActive.Tip();
        }
        nHeight = pindex->nHeight + (strCommand == "current" ? 1 : 10);
        dnodeman.UpdateLastPaid(pindex);

        if(!dnodeman.GetNextDynodeInQueueForPayment(nHeight, true, nCount, dnInfo))
            return "unknown";

        UniValue obj(UniValue::VOBJ);

        obj.push_back(Pair("height",        nHeight));
        obj.push_back(Pair("IP:port",       dnInfo.addr.ToString()));
        obj.push_back(Pair("protocol",      (int64_t)dnInfo.nProtocolVersion));
        obj.push_back(Pair("outpoint",      dnInfo.vin.prevout.ToStringShort()));
        obj.push_back(Pair("payee",         CDynamicAddress(dnInfo.pubKeyCollateralAddress.GetID()).ToString()));
        obj.push_back(Pair("lastseen",      dnInfo.nTimeLastPing));
        obj.push_back(Pair("activeseconds", dnInfo.nTimeLastPing - dnInfo.sigTime));
        return obj;
    }

#ifdef ENABLE_WALLET       
    if (strCommand == "debug")
    {
        if(activeDynode.nState != ACTIVE_DYNODE_INITIAL || !dynodeSync.IsBlockchainSynced())
            return activeDynode.GetStatus();

        COutPoint outpoint;
        CPubKey pubkey;
        CKey key;
        if(!pwalletMain || !pwalletMain->GetDynodeOutpointAndKeys(outpoint, pubkey, key))
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Missing Dynode input, please look at the documentation for instructions on Dynode creation");
        return activeDynode.GetStatus();
    }

    if (strCommand == "start-alias")
    {
        if (request.params.size() < 2)
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Please specify an alias");

        {
            LOCK(pwalletMain->cs_wallet);
            EnsureWalletIsUnlocked();
        }

        std::string strAlias = request.params[1].get_str();

        bool fFound = false;

        UniValue statusObj(UniValue::VOBJ);
        statusObj.push_back(Pair("alias", strAlias));

        BOOST_FOREACH(CDynodeConfig::CDynodeEntry dne, dynodeConfig.getEntries()) {
            if(dne.getAlias() == strAlias) {
                fFound = true;
                std::string strError;
                CDynodeBroadcast dnb;

                bool fResult = CDynodeBroadcast::Create(dne.getIp(), dne.getPrivKey(), dne.getTxHash(), dne.getOutputIndex(), strError, dnb);

                statusObj.push_back(Pair("result", fResult ? "successful" : "failed"));
                if(fResult) {
                    dnodeman.UpdateDynodeList(dnb, *g_connman);
                    dnb.Relay(*g_connman);
                } else {
                    statusObj.push_back(Pair("errorMessage", strError));
                }
                dnodeman.NotifyDynodeUpdates(*g_connman);
                break;
            }
        }

        if(!fFound) {
            statusObj.push_back(Pair("result", "failed"));
            statusObj.push_back(Pair("errorMessage", "Could not find alias in config. Verify with list-conf."));
        }

        return statusObj;

    }

    if (strCommand == "start-all" || strCommand == "start-missing" || strCommand == "start-disabled")
    {
        {
            LOCK(pwalletMain->cs_wallet);
            EnsureWalletIsUnlocked();
        }

        if((strCommand == "start-missing" || strCommand == "start-disabled") && !dynodeSync.IsDynodeListSynced()) {
            throw JSONRPCError(RPC_CLIENT_IN_INITIAL_DOWNLOAD, "You can't use this command until Dynode list is synced");
        }

        int nSuccessful = 0;
        int nFailed = 0;

        UniValue resultsObj(UniValue::VOBJ);

        BOOST_FOREACH(CDynodeConfig::CDynodeEntry dne, dynodeConfig.getEntries()) {
            std::string strError;

            COutPoint outpoint = COutPoint(uint256S(dne.getTxHash()), uint32_t(atoi(dne.getOutputIndex().c_str())));
            CDynode dn;
            bool fFound = dnodeman.Get(outpoint, dn);
            CDynodeBroadcast dnb;

            if(strCommand == "start-missing" && fFound) continue;
            if(strCommand == "start-disabled" && fFound && dn.IsEnabled()) continue;

            bool fResult = CDynodeBroadcast::Create(dne.getIp(), dne.getPrivKey(), dne.getTxHash(), dne.getOutputIndex(), strError, dnb);

            UniValue statusObj(UniValue::VOBJ);
            statusObj.push_back(Pair("alias", dne.getAlias()));
            statusObj.push_back(Pair("result", fResult ? "successful" : "failed"));

            if (fResult) {
                nSuccessful++;
                dnodeman.UpdateDynodeList(dnb, *g_connman);
                dnb.Relay(*g_connman);
            } else {
                nFailed++;
                statusObj.push_back(Pair("errorMessage", strError));
            }

            resultsObj.push_back(Pair("status", statusObj));
        }
        dnodeman.NotifyDynodeUpdates(*g_connman);

        UniValue returnObj(UniValue::VOBJ);
        returnObj.push_back(Pair("overall", strprintf("Successfully started %d Dynodes, failed to start %d, total %d", nSuccessful, nFailed, nSuccessful + nFailed)));
        returnObj.push_back(Pair("detail", resultsObj));

        return returnObj;
    }
#endif // ENABLE_WALLET

    if (strCommand == "genkey")
    {
        CKey secret;
        secret.MakeNewKey(false);

        return CDynamicSecret(secret).ToString();
    }

    if (strCommand == "list-conf")
    {
        UniValue resultObj(UniValue::VOBJ);

        BOOST_FOREACH(CDynodeConfig::CDynodeEntry dne, dynodeConfig.getEntries()) {
            COutPoint outpoint = COutPoint(uint256S(dne.getTxHash()), uint32_t(atoi(dne.getOutputIndex().c_str())));
            CDynode dn;
            bool fFound = dnodeman.Get(outpoint, dn);

            std::string strStatus = fFound ? dn.GetStatus() : "MISSING";

            UniValue dnObj(UniValue::VOBJ);
            dnObj.push_back(Pair("alias", dne.getAlias()));
            dnObj.push_back(Pair("address", dne.getIp()));
            dnObj.push_back(Pair("privateKey", dne.getPrivKey()));
            dnObj.push_back(Pair("txHash", dne.getTxHash()));
            dnObj.push_back(Pair("outputIndex", dne.getOutputIndex()));
            dnObj.push_back(Pair("status", strStatus));
            resultObj.push_back(Pair("dynode", dnObj));
        }

        return resultObj;
    }

#ifdef ENABLE_WALLET
    if (strCommand == "outputs") {
        // Find possible candidates
        std::vector<COutput> vPossibleCoins;
        pwalletMain->AvailableCoins(vPossibleCoins, true, NULL, false, ONLY_1000);

        UniValue obj(UniValue::VOBJ);
        BOOST_FOREACH(COutput& out, vPossibleCoins) {
            obj.push_back(Pair(out.tx->GetHash().ToString(), strprintf("%d", out.i)));
        }

        return obj;
    }
#endif // ENABLE_WALLET

    if (strCommand == "status")
    {
        if (!fDyNode)
            throw JSONRPCError(RPC_INTERNAL_ERROR, "This is not a Dynode");

        UniValue dnObj(UniValue::VOBJ);

        dnObj.push_back(Pair("outpoint", activeDynode.outpoint.ToStringShort()));
        dnObj.push_back(Pair("service", activeDynode.service.ToString()));

        CDynode dn;
        if(dnodeman.Get(activeDynode.outpoint, dn)) {
            dnObj.push_back(Pair("payee", CDynamicAddress(dn.pubKeyCollateralAddress.GetID()).ToString()));
        }

        dnObj.push_back(Pair("status", activeDynode.GetStatus()));
        return dnObj;
    }

    if (strCommand == "winners")
    {
        int nHeight;
        {
            LOCK(cs_main);
            CBlockIndex* pindex = chainActive.Tip();
            if(!pindex) return NullUniValue;

            nHeight = pindex->nHeight;
        }

        int nLast = 10;
        std::string strFilter = "";

        if (request.params.size() >= 2) {
            nLast = atoi(request.params[1].get_str());
        }

        if (request.params.size() == 3) {
            strFilter = request.params[2].get_str();
        }

        if (request.params.size() > 3)
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Correct usage is 'dynode winners ( \"count\" \"filter\" )'");

        UniValue obj(UniValue::VOBJ);

        for(int i = nHeight - nLast; i < nHeight + 20; i++) {
            std::string strPayment = GetRequiredPaymentsString(i);
            if (strFilter !="" && strPayment.find(strFilter) == std::string::npos) continue;
            obj.push_back(Pair(strprintf("%d", i), strPayment));
        }

        return obj;
    }

    return NullUniValue;
}

UniValue dynodelist(const JSONRPCRequest& request)
{
    std::string strMode = "status";
    std::string strFilter = "";

    if (request.params.size() >= 1) strMode = request.params[0].get_str();
    if (request.params.size() == 2) strFilter = request.params[1].get_str();

    if (request.fHelp || (
                strMode != "activeseconds" && strMode != "addr" && strMode != "full" && strMode != "info" &&
                strMode != "lastseen" && strMode != "lastpaidtime" && strMode != "lastpaidblock" &&
                strMode != "protocol" && strMode != "payee" && strMode != "pubkey" &&
                strMode != "rank" && strMode != "status"))
    {
        throw std::runtime_error(
                "dynodelist ( \"mode\" \"filter\" )\n"
                "Get a list of Dynodes in different modes\n"
                "\nArguments:\n"
                "1. \"mode\"      (string, optional/required to use filter, defaults = status) The mode to run list in\n"
                "2. \"filter\"    (string, optional) Filter results. Partial match by outpoint by default in all modes,\n"
                "                                    additional matches in some modes are also available\n"
                "\nAvailable modes:\n"
                "  activeseconds  - Print number of seconds Dynode recognized by the network as enabled\n"
                "                   (since latest issued \"dynode start/start-many/start-alias\")\n"
                "  addr           - Print ip address associated with a Dynode (can be additionally filtered, partial match)\n"
                "  full           - Print info in format 'status protocol payee lastseen activeseconds lastpaidtime lastpaidblock IP'\n"
                "                   (can be additionally filtered, partial match)\n"
                "  info           - Print info in format 'status protocol payee lastseen activeseconds sentinelversion sentinelstate IP'\n"
                "                   (can be additionally filtered, partial match)\n"
                "  lastpaidblock  - Print the last block height a node was paid on the network\n"
                "  lastpaidtime   - Print the last time a node was paid on the network\n"
                "  lastseen       - Print timestamp of when a Dynode was last seen on the network\n"
                "  payee          - Print Dynamic address associated with a Dynode (can be additionally filtered,\n"
                "                   partial match)\n"
                "  protocol       - Print protocol of a Dynode (can be additionally filtered, exact match))\n"
                "  pubkey         - Print the Dynode (not collateral) public key\n"
                "  rank           - Print rank of a Dynode based on current block\n"
                "  status         - Print Dynode status: PRE_ENABLED / ENABLED / EXPIRED / SENTINEL_PING_EXPIRED / NEW_START_REQUIRED /\n"
                "                   UPDATE_REQUIRED / POSE_BAN / OUTPOINT_SPENT (can be additionally filtered, partial match)\n"
                );
    }

    if (strMode == "full" || strMode == "lastpaidtime" || strMode == "lastpaidblock") {
        CBlockIndex* pindex = NULL;
         {
            LOCK(cs_main);
            pindex = chainActive.Tip();
        }
        dnodeman.UpdateLastPaid(pindex);
    }

    UniValue obj(UniValue::VOBJ);
    if (strMode == "rank") {
        CDynodeMan::rank_pair_vec_t vDynodeRanks;
        dnodeman.GetDynodeRanks(vDynodeRanks);
        BOOST_FOREACH(PAIRTYPE(int, CDynode)& s, vDynodeRanks) {
            std::string strOutpoint = s.second.vin.prevout.ToStringShort();
            if (strFilter !="" && strOutpoint.find(strFilter) == std::string::npos) continue;
            obj.push_back(Pair(strOutpoint, s.first));
        }
    } else {
        std::map<COutPoint, CDynode> mapDynodes = dnodeman.GetFullDynodeMap();
        for (auto& dnpair : mapDynodes) {
            CDynode dn = dnpair.second;
            std::string strOutpoint = dnpair.first.ToStringShort();
            if (strMode == "activeseconds") {
                if (strFilter !="" && strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, (int64_t)(dn.lastPing.sigTime - dn.sigTime)));
            } else if (strMode == "addr") {
                std::string strAddress = dn.addr.ToString();
                if (strFilter !="" && strAddress.find(strFilter) == std::string::npos &&
                    strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, strAddress));
            } else if (strMode == "full") {
                std::ostringstream streamFull;
                streamFull << std::setw(18) <<
                               dn.GetStatus() << " " <<
                               dn.nProtocolVersion << " " <<
                               CDynamicAddress(dn.pubKeyCollateralAddress.GetID()).ToString() << " " <<
                               (int64_t)dn.lastPing.sigTime << " " << std::setw(8) <<
                               (int64_t)(dn.lastPing.sigTime - dn.sigTime) << " " << std::setw(10) <<
                               dn.GetLastPaidTime() << " "  << std::setw(6) <<
                               dn.GetLastPaidBlock() << " " <<
                               dn.addr.ToString();
                std::string strFull = streamFull.str();
                if (strFilter !="" && strFull.find(strFilter) == std::string::npos &&
                    strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, strFull));
            } else if (strMode == "info") {
                std::ostringstream streamInfo;
                streamInfo << std::setw(18) <<
                               dn.GetStatus() << " " <<
                               dn.nProtocolVersion << " " <<
                               CDynamicAddress(dn.pubKeyCollateralAddress.GetID()).ToString() << " " <<
                               (int64_t)dn.lastPing.sigTime << " " << std::setw(8) <<
                               (int64_t)(dn.lastPing.sigTime - dn.sigTime) << " " <<
                               SafeIntVersionToString(dn.lastPing.nSentinelVersion) << " "  <<
                               (dn.lastPing.fSentinelIsCurrent ? "current" : "expired") << " " <<
                               dn.addr.ToString();
                std::string strInfo = streamInfo.str();
                if (strFilter !="" && strInfo.find(strFilter) == std::string::npos &&
                    strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, strInfo));
            } else if (strMode == "lastpaidblock") {
                if (strFilter !="" && strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, dn.GetLastPaidBlock()));
            } else if (strMode == "lastpaidtime") {
                if (strFilter !="" && strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, dn.GetLastPaidTime()));
            } else if (strMode == "lastseen") {
                if (strFilter !="" && strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, (int64_t)dn.lastPing.sigTime));
            } else if (strMode == "payee") {
                CDynamicAddress address(dn.pubKeyCollateralAddress.GetID());
                std::string strPayee = address.ToString();
                if (strFilter !="" && strPayee.find(strFilter) == std::string::npos &&
                    strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, strPayee));
            } else if (strMode == "protocol") {
                if (strFilter !="" && strFilter != strprintf("%d", dn.nProtocolVersion) &&
                    strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, (int64_t)dn.nProtocolVersion));
            } else if (strMode == "pubkey") {
                if (strFilter !="" && strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, HexStr(dn.pubKeyDynode)));
            } else if (strMode == "status") {
                std::string strStatus = dn.GetStatus();
                if (strFilter !="" && strStatus.find(strFilter) == std::string::npos &&
                    strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, strStatus));
            }
        }
    }
    return obj;
}

bool DecodeHexVecDnb(std::vector<CDynodeBroadcast>& vecDnb, std::string strHexDnb) {

    if (!IsHex(strHexDnb))
        return false;

    std::vector<unsigned char> dnbData(ParseHex(strHexDnb));
    CDataStream ssData(dnbData, SER_NETWORK, PROTOCOL_VERSION);
    try {
        ssData >> vecDnb;
    }
    catch (const std::exception&) {
        return false;
    }

    return true;
}

UniValue dynodebroadcast(const JSONRPCRequest& request)
{
    std::string strCommand;
    if (request.params.size() >= 1)
        strCommand = request.params[0].get_str();

    if (request.fHelp  ||
        (
#ifdef ENABLE_WALLET
            strCommand != "create-alias" && strCommand != "create-all" &&
#endif // ENABLE_WALLET
            strCommand != "decode" && strCommand != "relay"))
        throw std::runtime_error(
                "dynodebroadcast \"command\"...\n"
                "Set of commands to create and relay Dynode broadcast messages\n"
                "\nArguments:\n"
                "1. \"command\"        (string or set of strings, required) The command to execute\n"
                "\nAvailable commands:\n"
#ifdef ENABLE_WALLET
                "  create-alias  - Create single remote Dynode broadcast message by assigned alias configured in dynode.conf\n"
                "  create-all    - Create remote Dynode broadcast messages for all Dynodes configured in dynode.conf\n"
#endif // ENABLE_WALLET
                "  decode        - Decode Dynode broadcast message\n"
                "  relay         - Relay Dynode broadcast message to the network\n"
                );

#ifdef ENABLE_WALLET
    if (strCommand == "create-alias")
    {
        // wait for reindex and/or import to finish
        if (fImporting || fReindex)
            throw JSONRPCError(RPC_INTERNAL_ERROR, "Wait for reindex and/or import to finish");

        if (request.params.size() < 2)
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Please specify an alias");

        {
            LOCK(pwalletMain->cs_wallet);
            EnsureWalletIsUnlocked();
        }

        bool fFound = false;
        std::string strAlias = request.params[1].get_str();

        UniValue statusObj(UniValue::VOBJ);
        std::vector<CDynodeBroadcast> vecDnb;

        statusObj.push_back(Pair("alias", strAlias));

        BOOST_FOREACH(CDynodeConfig::CDynodeEntry dne, dynodeConfig.getEntries()) {
            if(dne.getAlias() == strAlias) {
                fFound = true;
                std::string strError;
                CDynodeBroadcast dnb;

                bool fResult = CDynodeBroadcast::Create(dne.getIp(), dne.getPrivKey(), dne.getTxHash(), dne.getOutputIndex(), strError, dnb, true);

                statusObj.push_back(Pair("result", fResult ? "successful" : "failed"));
                if(fResult) {
                    vecDnb.push_back(dnb);
                    CDataStream ssVecDnb(SER_NETWORK, PROTOCOL_VERSION);
                    ssVecDnb << vecDnb;
                    statusObj.push_back(Pair("hex", HexStr(ssVecDnb.begin(), ssVecDnb.end())));
                } else {
                    statusObj.push_back(Pair("errorMessage", strError));
                }
                break;
            }
        }

        if(!fFound) {
            statusObj.push_back(Pair("result", "not found"));
            statusObj.push_back(Pair("errorMessage", "Could not find alias in config. Verify with list-conf."));
        }

        return statusObj;

    }

    if (strCommand == "create-all")
    {
        // wait for reindex and/or import to finish
        if (fImporting || fReindex)
            throw JSONRPCError(RPC_INTERNAL_ERROR, "Wait for reindex and/or import to finish");

        {
            LOCK(pwalletMain->cs_wallet);
            EnsureWalletIsUnlocked();
        }

        std::vector<CDynodeConfig::CDynodeEntry> dnEntries;
        dnEntries = dynodeConfig.getEntries();

        int nSuccessful = 0;
        int nFailed = 0;

        UniValue resultsObj(UniValue::VOBJ);
        std::vector<CDynodeBroadcast> vecDnb;

        BOOST_FOREACH(CDynodeConfig::CDynodeEntry dne, dynodeConfig.getEntries()) {
            std::string strError;
            CDynodeBroadcast dnb;

            bool fResult = CDynodeBroadcast::Create(dne.getIp(), dne.getPrivKey(), dne.getTxHash(), dne.getOutputIndex(), strError, dnb, true);

            UniValue statusObj(UniValue::VOBJ);
            statusObj.push_back(Pair("alias", dne.getAlias()));
            statusObj.push_back(Pair("result", fResult ? "successful" : "failed"));

            if(fResult) {
                nSuccessful++;
                vecDnb.push_back(dnb);
            } else {
                nFailed++;
                statusObj.push_back(Pair("errorMessage", strError));
            }

            resultsObj.push_back(Pair("status", statusObj));
        }

        CDataStream ssVecDnb(SER_NETWORK, PROTOCOL_VERSION);
        ssVecDnb << vecDnb;
        UniValue returnObj(UniValue::VOBJ);
        returnObj.push_back(Pair("overall", strprintf("Successfully created broadcast messages for %d Dynodes, failed to create %d, total %d", nSuccessful, nFailed, nSuccessful + nFailed)));
        returnObj.push_back(Pair("detail", resultsObj));
        returnObj.push_back(Pair("hex", HexStr(ssVecDnb.begin(), ssVecDnb.end())));

        return returnObj;
    }
#endif // ENABLE_WALLET

    if (strCommand == "decode")
    {
        if (request.params.size() != 2)
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Correct usage is 'dynodebroadcast decode \"hexstring\"'");

        std::vector<CDynodeBroadcast> vecDnb;

        if (!DecodeHexVecDnb(vecDnb, request.params[1].get_str()))
            throw JSONRPCError(RPC_DESERIALIZATION_ERROR, "Dynode broadcast message decode failed");

        int nSuccessful = 0;
        int nFailed = 0;
        int nDos = 0;
        UniValue returnObj(UniValue::VOBJ);

        BOOST_FOREACH(CDynodeBroadcast& dnb, vecDnb) {
            UniValue resultObj(UniValue::VOBJ);

            if(dnb.CheckSignature(nDos)) {
                nSuccessful++;
                resultObj.push_back(Pair("outpoint", dnb.vin.prevout.ToStringShort()));
                resultObj.push_back(Pair("addr", dnb.addr.ToString()));
                resultObj.push_back(Pair("pubKeyCollateralAddress", CDynamicAddress(dnb.pubKeyCollateralAddress.GetID()).ToString()));
                resultObj.push_back(Pair("pubKeyDynode", CDynamicAddress(dnb.pubKeyDynode.GetID()).ToString()));
                resultObj.push_back(Pair("vchSig", EncodeBase64(&dnb.vchSig[0], dnb.vchSig.size())));
                resultObj.push_back(Pair("sigTime", dnb.sigTime));
                resultObj.push_back(Pair("protocolVersion", dnb.nProtocolVersion));
                resultObj.push_back(Pair("nLastPsq", dnb.nLastPsq));

                UniValue lastPingObj(UniValue::VOBJ);
                lastPingObj.push_back(Pair("outpoint", dnb.lastPing.vin.prevout.ToStringShort()));
                lastPingObj.push_back(Pair("blockHash", dnb.lastPing.blockHash.ToString()));
                lastPingObj.push_back(Pair("sigTime", dnb.lastPing.sigTime));
                lastPingObj.push_back(Pair("vchSig", EncodeBase64(&dnb.lastPing.vchSig[0], dnb.lastPing.vchSig.size())));

                resultObj.push_back(Pair("lastPing", lastPingObj));
            } else {
                nFailed++;
                resultObj.push_back(Pair("errorMessage", "Dynode broadcast signature verification failed"));
            }

            returnObj.push_back(Pair(dnb.GetHash().ToString(), resultObj));
        }

        returnObj.push_back(Pair("overall", strprintf("Successfully decoded broadcast messages for %d Dynodes, failed to decode %d, total %d", nSuccessful, nFailed, nSuccessful + nFailed)));

        return returnObj;
    }

    if (strCommand == "relay")
    {
        if (request.params.size() < 2 || request.params.size() > 3)
            throw JSONRPCError(RPC_INVALID_PARAMETER,   "dynodebroadcast relay \"hexstring\" ( fast )\n"
                                                        "\nArguments:\n"
                                                        "1. \"hex\"      (string, required) Broadcast messages hex string\n"
                                                        "2. fast       (string, optional) If none, using safe method\n");

        std::vector<CDynodeBroadcast> vecDnb;

        if (!DecodeHexVecDnb(vecDnb, request.params[1].get_str()))
            throw JSONRPCError(RPC_DESERIALIZATION_ERROR, "Dynode broadcast message decode failed");

        int nSuccessful = 0;
        int nFailed = 0;
        bool fSafe = request.params.size() == 2;
        UniValue returnObj(UniValue::VOBJ);

        // verify all signatures first, bailout if any of them broken
        BOOST_FOREACH(CDynodeBroadcast& dnb, vecDnb) {
            UniValue resultObj(UniValue::VOBJ);

            resultObj.push_back(Pair("outpoint", dnb.vin.prevout.ToStringShort()));
            resultObj.push_back(Pair("addr", dnb.addr.ToString()));

            int nDos = 0;
            bool fResult;
            if (dnb.CheckSignature(nDos)) {
                if (fSafe) {
                    fResult = dnodeman.CheckDnbAndUpdateDynodeList(NULL, dnb, nDos, *g_connman);
                } else {
                    dnodeman.UpdateDynodeList(dnb, *g_connman);
                    dnb.Relay(*g_connman);
                    fResult = true;
                }
                dnodeman.NotifyDynodeUpdates(*g_connman);
            } else fResult = false;

            if(fResult) {
                nSuccessful++;
                resultObj.push_back(Pair(dnb.GetHash().ToString(), "successful"));
            } else {
                nFailed++;
                resultObj.push_back(Pair("errorMessage", "Dynode broadcast signature verification failed"));
            }

            returnObj.push_back(Pair(dnb.GetHash().ToString(), resultObj));
        }

        returnObj.push_back(Pair("overall", strprintf("Successfully relayed broadcast messages for %d Dynodes, failed to relay %d, total %d", nSuccessful, nFailed, nSuccessful + nFailed)));

        return returnObj;
    }

    return NullUniValue;
}

UniValue sentinelping(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 1) {
        throw std::runtime_error(
            "sentinelping version\n"
            "\nSentinel ping.\n"
            "\nArguments:\n"
            "1. version           (string, required) Sentinel version in the form \"x.x.x\"\n"
            "\nResult:\n"
            "state                (boolean) Ping result\n"
            "\nExamples:\n"
            + HelpExampleCli("sentinelping", "1.0.2")
            + HelpExampleRpc("sentinelping", "1.0.2")
        );
    }

    activeDynode.UpdateSentinelPing(StringVersionToInt(request.params[0].get_str()));
    return true;
}

static const CRPCCommand commands[] =
{ //  category                  name                    actor (function)     okSafeMode
    /* Dynamic features */
    { "dynamic",               "dynode",                &dynode,             true  },
    { "dynamic",               "dynodelist",            &dynodelist,         true  },
    { "dynamic",               "dynodebroadcast",       &dynodebroadcast,    true  },
    { "dynamic",               "getpoolinfo",           &getpoolinfo,        true  },
    { "dynamic",               "sentinelping",          &sentinelping,       true  },
#ifdef ENABLE_WALLET
    { "dynamic",               "privatesend",           &privatesend,        false },
#endif // ENABLE_WALLET
};

void RegisterDynodeRPCCommands(CRPCTable &tableRPC)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        tableRPC.appendCommand(commands[vcidx].name, &commands[vcidx]);
}
