// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "activestormnode.h"
#include "sandstorm.h"
#include "init.h"
#include "main.h"
#include "stormnode-payments.h"
#include "stormnode-sync.h"
#include "stormnodeconfig.h"
#include "stormnodeman.h"
#include "rpcserver.h"
#include "util.h"
#include "utilmoneystr.h"

#include <fstream>
#include <iomanip>
#include <univalue.h>

void EnsureWalletIsUnlocked();

UniValue privatesend(const UniValue& params, bool fHelp)
{
    if (fHelp || params.size() != 1)
        throw std::runtime_error(
            "privatesend \"command\"\n"
            "\nArguments:\n"
            "1. \"command\"        (string or set of strings, required) The command to execute\n"
            "\nAvailable commands:\n"
            "  start       - Start mixing\n"
            "  stop        - Stop mixing\n"
            "  reset       - Reset mixing\n"
            + HelpRequiringPassphrase());

    if(params[0].get_str() == "start") {
        if (pwalletMain->IsLocked(true))
            throw JSONRPCError(RPC_WALLET_UNLOCK_NEEDED, "Error: Please enter the wallet passphrase with walletpassphrase first.");

        if(fStormNode)
            return "Mixing is not supported from stormnodes";

        fEnablePrivateSend = true;
        bool result = sandStormPool.DoAutomaticDenominating();
        return "Mixing " + (result ? "started successfully" : ("start failed: " + sandStormPool.GetStatus() + ", will retry"));
    }

    if(params[0].get_str() == "stop") {
        fEnablePrivateSend = false;
        return "Mixing was stopped";
    }

    if(params[0].get_str() == "reset") {
        sandStormPool.ResetPool();
        return "Mixing was reset";
    }

    return "Unknown command, please see \"help privatesend\"";
}

UniValue getpoolinfo(const UniValue& params, bool fHelp)
{
    if (fHelp || params.size() != 0)
        throw std::runtime_error(
            "getpoolinfo\n"
            "Returns an object containing mixing pool related information.\n");

    UniValue obj(UniValue::VOBJ);
    obj.push_back(Pair("state",             sandStormPool.GetStateString()));
    obj.push_back(Pair("mixing_mode",       fPrivateSendMultiSession ? "multi-session" : "normal"));
    obj.push_back(Pair("queue",             sandStormPool.GetQueueSize()));
    obj.push_back(Pair("entries",           sandStormPool.GetEntriesCount()));
    obj.push_back(Pair("status",            sandStormPool.GetStatus()));

    if (sandStormPool.pSubmittedToStormnode) {
        obj.push_back(Pair("outpoint",      sandStormPool.pSubmittedToStormnode->vin.prevout.ToStringShort()));
        obj.push_back(Pair("addr",          sandStormPool.pSubmittedToStormnode->addr.ToString()));
    }

    if (pwalletMain) {
        obj.push_back(Pair("keys_left",     pwalletMain->nKeysLeftSinceAutoBackup));
        obj.push_back(Pair("warnings",      pwalletMain->nKeysLeftSinceAutoBackup < PRIVATESEND_KEYS_THRESHOLD_WARNING
                                                ? "WARNING: keypool is almost depleted!" : ""));
    }
    return obj;
}


UniValue stormnode(const UniValue& params, bool fHelp)
{
    std::string strCommand;
    if (params.size() >= 1) {
        strCommand = params[0].get_str();
    }

    if (strCommand == "start-many")
        throw JSONRPCError(RPC_INVALID_PARAMETER, "DEPRECATED, please use start-all instead");

    if (fHelp  ||
        (strCommand != "start" && strCommand != "start-alias" && strCommand != "start-all" && strCommand != "start-missing" &&
         strCommand != "start-disabled" && strCommand != "list" && strCommand != "list-conf" && strCommand != "count" &&
         strCommand != "debug" && strCommand != "current" && strCommand != "winner" && strCommand != "winners" && strCommand != "genkey" &&
         strCommand != "connect" && strCommand != "outputs" && strCommand != "status"))
            throw std::runtime_error(
                "stormnode \"command\"... ( \"passphrase\" )\n"
                "Set of commands to execute stormnode-sync related actions\n"
                "\nArguments:\n"
                "1. \"command\"        (string or set of strings, required) The command to execute\n"
                "2. \"passphrase\"     (string, optional) The wallet passphrase\n"
                "\nAvailable commands:\n"
                "  count        - Print number of all known stormnodes (optional: 'ps', 'enabled', 'all', 'qualify')\n"
                "  current      - Print info on current stormnode winner to be paid the next block (calculated locally)\n"
                "  debug        - Print stormnode status\n"
                "  genkey       - Generate new stormnodeprivkey\n"
                "  outputs      - Print stormnode compatible outputs\n"
                "  start        - Start local Hot stormnode configured in darksilk.conf\n"
                "  start-alias  - Start single remote stormnode by assigned alias configured in stormnode.conf\n"
                "  start-<mode> - Start remote stormnodes configured in stormnode.conf (<mode>: 'all', 'missing', 'disabled')\n"
                "  status       - Print stormnode status information\n"
                "  list         - Print list of all known stormnodes (see stormnodelist for more info)\n"
                "  list-conf    - Print stormnode.conf in JSON format\n"
                "  winner       - Print info on next stormnode winner to vote for\n"
                "  winners      - Print list of stormnode winners\n"
                );

    if (strCommand == "list")
    {
        UniValue newParams(UniValue::VARR);
        // forward params but skip "list"
        for (unsigned int i = 1; i < params.size(); i++) {
            newParams.push_back(params[i]);
        }
        return stormnodelist(newParams, fHelp);
    }

    if(strCommand == "connect")
    {
        if (params.size() < 2)
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Stormnode address required");

        std::string strAddress = params[1].get_str();

        CService addr = CService(strAddress);

        CNode *pnode = ConnectNode((CAddress)addr, NULL);
        if(!pnode)
            throw JSONRPCError(RPC_INTERNAL_ERROR, strprintf("Couldn't connect to stormnode %s", strAddress));

        return "successfully connected";
    }

    if (strCommand == "count")
    {
        if (params.size() > 2)
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Too many parameters");

        if (params.size() == 1)
            return snodeman.size();

        std::string strMode = params[1].get_str();

        if (strMode == "ps")
            return snodeman.CountEnabled(MIN_PRIVATESEND_PEER_PROTO_VERSION);

        if (strMode == "enabled")
            return snodeman.CountEnabled();

        int nCount;
        snodeman.GetNextStormnodeInQueueForPayment(true, nCount);

        if (strMode == "qualify")
            return nCount;

        if (strMode == "all")
            return strprintf("Total: %d (PS Compatible: %d / Enabled: %d / Qualify: %d)",
                snodeman.size(), snodeman.CountEnabled(MIN_PRIVATESEND_PEER_PROTO_VERSION),
                snodeman.CountEnabled(), nCount);
    }

    if (strCommand == "current" || strCommand == "winner")
    {
        int nCount;
        int nHeight;
        CStormnode* winner = NULL;
        {
            LOCK(cs_main);
            nHeight = chainActive.Height() + (strCommand == "current" ? 1 : 10);
        }
        snodeman.UpdateLastPaid();
        winner = snodeman.GetNextStormnodeInQueueForPayment(nHeight, true, nCount);
        if(!winner) return "unknown";

        UniValue obj(UniValue::VOBJ);

        obj.push_back(Pair("height",        nHeight));
        obj.push_back(Pair("IP:port",       winner->addr.ToString()));
        obj.push_back(Pair("protocol",      (int64_t)winner->nProtocolVersion));
        obj.push_back(Pair("vin",           winner->vin.prevout.ToStringShort()));
        obj.push_back(Pair("payee",         CDarkSilkAddress(winner->pubKeyCollateralAddress.GetID()).ToString()));
        obj.push_back(Pair("lastseen",      (winner->lastPing == CStormnodePing()) ? winner->sigTime :
                                                    winner->lastPing.sigTime));
        obj.push_back(Pair("activeseconds", (winner->lastPing == CStormnodePing()) ? 0 :
                                                    (winner->lastPing.sigTime - winner->sigTime)));
        return obj;
    }

    if (strCommand == "debug")
    {
        if(activeStormnode.nState != ACTIVE_STORMNODE_INITIAL || !stormnodeSync.IsBlockchainSynced())
            return activeStormnode.GetStatus();

        CTxIn vin;
        CPubKey pubkey;
        CKey key;

        if(!pwalletMain || !pwalletMain->GetStormnodeVinAndKeys(vin, pubkey, key))
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Missing stormnode input, please look at the documentation for instructions on stormnode creation");

        return activeStormnode.GetStatus();
    }

    if (strCommand == "start")
    {
        if(!fStormNode)
            throw JSONRPCError(RPC_INTERNAL_ERROR, "You must set stormnode=1 in the configuration");

        {
            LOCK(pwalletMain->cs_wallet);
            EnsureWalletIsUnlocked();
        }

        if(activeStormnode.nState != ACTIVE_STORMNODE_STARTED){
            activeStormnode.nState = ACTIVE_STORMNODE_INITIAL; // TODO: consider better way
            activeStormnode.ManageState();
        }

        return activeStormnode.GetStatus();
    }

    if (strCommand == "start-alias")
    {
        if (params.size() < 2)
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Please specify an alias");

        {
            LOCK(pwalletMain->cs_wallet);
            EnsureWalletIsUnlocked();
        }

        std::string strAlias = params[1].get_str();

        bool fFound = false;

        UniValue statusObj(UniValue::VOBJ);
        statusObj.push_back(Pair("alias", strAlias));

        BOOST_FOREACH(CStormnodeConfig::CStormnodeEntry sne, stormnodeConfig.getEntries()) {
            if(sne.getAlias() == strAlias) {
                fFound = true;
                std::string strError;
                CStormnodeBroadcast snb;

                bool fResult = CStormnodeBroadcast::Create(sne.getIp(), sne.getPrivKey(), sne.getTxHash(), sne.getOutputIndex(), strError, snb);

                statusObj.push_back(Pair("result", fResult ? "successful" : "failed"));
                if(fResult) {
                    snodeman.UpdateStormnodeList(snb);
                    snb.Relay();
                } else {
                    statusObj.push_back(Pair("errorMessage", strError));
                }
                snodeman.NotifyStormnodeUpdates();
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

        if((strCommand == "start-missing" || strCommand == "start-disabled") && !stormnodeSync.IsStormnodeListSynced()) {
            throw JSONRPCError(RPC_CLIENT_IN_INITIAL_DOWNLOAD, "You can't use this command until stormnode list is synced");
        }

        int nSuccessful = 0;
        int nFailed = 0;

        UniValue resultsObj(UniValue::VOBJ);

        BOOST_FOREACH(CStormnodeConfig::CStormnodeEntry sne, stormnodeConfig.getEntries()) {
            std::string strError;

            CTxIn vin = CTxIn(uint256S(sne.getTxHash()), uint32_t(atoi(sne.getOutputIndex().c_str())));
            CStormnode *psn = snodeman.Find(vin);
            CStormnodeBroadcast snb;

            if(strCommand == "start-missing" && psn) continue;
            if(strCommand == "start-disabled" && psn && psn->IsEnabled()) continue;

            bool fResult = CStormnodeBroadcast::Create(sne.getIp(), sne.getPrivKey(), sne.getTxHash(), sne.getOutputIndex(), strError, snb);

            UniValue statusObj(UniValue::VOBJ);
            statusObj.push_back(Pair("alias", sne.getAlias()));
            statusObj.push_back(Pair("result", fResult ? "successful" : "failed"));

            if (fResult) {
                nSuccessful++;
                snodeman.UpdateStormnodeList(snb);
                snb.Relay();
            } else {
                nFailed++;
                statusObj.push_back(Pair("errorMessage", strError));
            }

            resultsObj.push_back(Pair("status", statusObj));
        }
        snodeman.NotifyStormnodeUpdates();

        UniValue returnObj(UniValue::VOBJ);
        returnObj.push_back(Pair("overall", strprintf("Successfully started %d stormnodes, failed to start %d, total %d", nSuccessful, nFailed, nSuccessful + nFailed)));
        returnObj.push_back(Pair("detail", resultsObj));

        return returnObj;
    }

    if (strCommand == "genkey")
    {
        CKey secret;
        secret.MakeNewKey(false);

        return CDarkSilkSecret(secret).ToString();
    }

    if (strCommand == "list-conf")
    {
        UniValue resultObj(UniValue::VOBJ);

        BOOST_FOREACH(CStormnodeConfig::CStormnodeEntry sne, stormnodeConfig.getEntries()) {
            CTxIn vin = CTxIn(uint256S(sne.getTxHash()), uint32_t(atoi(sne.getOutputIndex().c_str())));
            CStormnode *psn = snodeman.Find(vin);

            std::string strStatus = psn ? psn->GetStatus() : "MISSING";

            UniValue snObj(UniValue::VOBJ);
            snObj.push_back(Pair("alias", sne.getAlias()));
            snObj.push_back(Pair("address", sne.getIp()));
            snObj.push_back(Pair("privateKey", sne.getPrivKey()));
            snObj.push_back(Pair("txHash", sne.getTxHash()));
            snObj.push_back(Pair("outputIndex", sne.getOutputIndex()));
            snObj.push_back(Pair("status", strStatus));
            resultObj.push_back(Pair("stormnode", snObj));
        }

        return resultObj;
    }

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

    if (strCommand == "status")
    {
        if (!fStormNode)
            throw JSONRPCError(RPC_INTERNAL_ERROR, "This is not a stormnode");

        UniValue snObj(UniValue::VOBJ);

        snObj.push_back(Pair("vin", activeStormnode.vin.ToString()));
        snObj.push_back(Pair("service", activeStormnode.service.ToString()));

        CStormnode sn;
        if(snodeman.Get(activeStormnode.vin, sn)) {
            snObj.push_back(Pair("payee", CDarkSilkAddress(sn.pubKeyCollateralAddress.GetID()).ToString()));
        }

        snObj.push_back(Pair("status", activeStormnode.GetStatus()));
        return snObj;
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

        if (params.size() >= 2) {
            nLast = atoi(params[1].get_str());
        }

        if (params.size() == 3) {
            strFilter = params[2].get_str();
        }

        if (params.size() > 3)
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Correct usage is 'stormnode winners ( \"count\" \"filter\" )'");

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

UniValue stormnodelist(const UniValue& params, bool fHelp)
{
    std::string strMode = "status";
    std::string strFilter = "";

    if (params.size() >= 1) strMode = params[0].get_str();
    if (params.size() == 2) strFilter = params[1].get_str();

    if (fHelp || (
                strMode != "activeseconds" && strMode != "addr" && strMode != "full" &&
                strMode != "lastseen" && strMode != "lastpaidtime" && strMode != "lastpaidblock" &&
                strMode != "protocol" && strMode != "payee" && strMode != "rank" && strMode != "status"))
    {
        throw std::runtime_error(
                "stormnodelist ( \"mode\" \"filter\" )\n"
                "Get a list of stormnodes in different modes\n"
                "\nArguments:\n"
                "1. \"mode\"      (string, optional/required to use filter, defaults = status) The mode to run list in\n"
                "2. \"filter\"    (string, optional) Filter results. Partial match by outpoint by default in all modes,\n"
                "                                    additional matches in some modes are also available\n"
                "\nAvailable modes:\n"
                "  activeseconds  - Print number of seconds stormnode recognized by the network as enabled\n"
                "                   (since latest issued \"stormnode start/start-many/start-alias\")\n"
                "  addr           - Print ip address associated with a stormnode (can be additionally filtered, partial match)\n"
                "  full           - Print info in format 'status protocol payee lastseen activeseconds lastpaidtime lastpaidblock IP'\n"
                "                   (can be additionally filtered, partial match)\n"
                "  lastpaidblock  - Print the last block height a node was paid on the network\n"
                "  lastpaidtime   - Print the last time a node was paid on the network\n"
                "  lastseen       - Print timestamp of when a stormnode was last seen on the network\n"
                "  payee          - Print DarkSilk address associated with a stormnode (can be additionally filtered,\n"
                "                   partial match)\n"
                "  protocol       - Print protocol of a stormnode (can be additionally filtered, exact match))\n"
                "  rank           - Print rank of a stormnode based on current block\n"
                "  status         - Print stormnode status: PRE_ENABLED / ENABLED / EXPIRED / WATCHDOG_EXPIRED / NEW_START_REQUIRED /\n"
                "                   UPDATE_REQUIRED / POSE_BAN / OUTPOINT_SPENT (can be additionally filtered, partial match)\n"
                );
    }

    if (strMode == "full" || strMode == "lastpaidtime" || strMode == "lastpaidblock") {
       snodeman.UpdateLastPaid();
    }

    UniValue obj(UniValue::VOBJ);
    if (strMode == "rank") {
        std::vector<std::pair<int, CStormnode> > vStormnodeRanks = snodeman.GetStormnodeRanks();
        BOOST_FOREACH(PAIRTYPE(int, CStormnode)& s, vStormnodeRanks) {
            std::string strOutpoint = s.second.vin.prevout.ToStringShort();
            if (strFilter !="" && strOutpoint.find(strFilter) == std::string::npos) continue;
            obj.push_back(Pair(strOutpoint, s.first));
        }
    } else {
        std::vector<CStormnode> vStormnodes = snodeman.GetFullStormnodeVector();
        BOOST_FOREACH(CStormnode& sn, vStormnodes) {
            std::string strOutpoint = sn.vin.prevout.ToStringShort();
            if (strMode == "activeseconds") {
                if (strFilter !="" && strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, (int64_t)(sn.lastPing.sigTime - sn.sigTime)));
            } else if (strMode == "addr") {
                std::string strAddress = sn.addr.ToString();
                if (strFilter !="" && strAddress.find(strFilter) == std::string::npos &&
                    strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, strAddress));
            } else if (strMode == "full") {
                std::ostringstream streamFull;
                streamFull << std::setw(15) <<
                               sn.GetStatus() << " " <<
                               sn.nProtocolVersion << " " <<
                               CDarkSilkAddress(sn.pubKeyCollateralAddress.GetID()).ToString() << " " <<
                               (int64_t)sn.lastPing.sigTime << " " << std::setw(8) <<
                               (int64_t)(sn.lastPing.sigTime - sn.sigTime) << " " << std::setw(10) <<
                               sn.GetLastPaidTime() << " "  << std::setw(6) <<
                               sn.GetLastPaidBlock() << " " <<
                               sn.addr.ToString();
                std::string strFull = streamFull.str();
                if (strFilter !="" && strFull.find(strFilter) == std::string::npos &&
                    strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, strFull));
            } else if (strMode == "lastpaidblock") {
                if (strFilter !="" && strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, sn.GetLastPaidBlock()));
            } else if (strMode == "lastpaidtime") {
                if (strFilter !="" && strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, sn.GetLastPaidTime()));
            } else if (strMode == "lastseen") {
                if (strFilter !="" && strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, (int64_t)sn.lastPing.sigTime));
            } else if (strMode == "payee") {
                CDarkSilkAddress address(sn.pubKeyCollateralAddress.GetID());
                std::string strPayee = address.ToString();
                if (strFilter !="" && strPayee.find(strFilter) == std::string::npos &&
                    strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, strPayee));
            } else if (strMode == "protocol") {
                if (strFilter !="" && strFilter != strprintf("%d", sn.nProtocolVersion) &&
                    strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, (int64_t)sn.nProtocolVersion));
            } else if (strMode == "status") {
                std::string strStatus = sn.GetStatus();
                if (strFilter !="" && strStatus.find(strFilter) == std::string::npos &&
                    strOutpoint.find(strFilter) == std::string::npos) continue;
                obj.push_back(Pair(strOutpoint, strStatus));
            }
        }
    }
    return obj;
}

bool DecodeHexVecSnb(std::vector<CStormnodeBroadcast>& vecSnb, std::string strHexSnb) {

    if (!IsHex(strHexSnb))
        return false;

    std::vector<unsigned char> snbData(ParseHex(strHexSnb));
    CDataStream ssData(snbData, SER_NETWORK, PROTOCOL_VERSION);
    try {
        ssData >> vecSnb;
    }
    catch (const std::exception&) {
        return false;
    }

    return true;
}

UniValue stormnodebroadcast(const UniValue& params, bool fHelp)
{
    std::string strCommand;
    if (params.size() >= 1)
        strCommand = params[0].get_str();

    if (fHelp  ||
        (strCommand != "create-alias" && strCommand != "create-all" && strCommand != "decode" && strCommand != "relay"))
        throw std::runtime_error(
                "stormnodebroadcast \"command\"... ( \"passphrase\" )\n"
                "Set of commands to create and relay stormnode broadcast messages\n"
                "\nArguments:\n"
                "1. \"command\"        (string or set of strings, required) The command to execute\n"
                "2. \"passphrase\"     (string, optional) The wallet passphrase\n"
                "\nAvailable commands:\n"
                "  create-alias  - Create single remote stormnode broadcast message by assigned alias configured in stormnode.conf\n"
                "  create-all    - Create remote stormnode broadcast messages for all stormnodes configured in stormnode.conf\n"
                "  decode        - Decode stormnode broadcast message\n"
                "  relay         - Relay stormnode broadcast message to the network\n"
                + HelpRequiringPassphrase());

    if (strCommand == "create-alias")
    {
        // wait for reindex and/or import to finish
        if (fImporting || fReindex)
            throw JSONRPCError(RPC_INTERNAL_ERROR, "Wait for reindex and/or import to finish");

        if (params.size() < 2)
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Please specify an alias");

        {
            LOCK(pwalletMain->cs_wallet);
            EnsureWalletIsUnlocked();
        }

        bool fFound = false;
        std::string strAlias = params[1].get_str();

        UniValue statusObj(UniValue::VOBJ);
        std::vector<CStormnodeBroadcast> vecSnb;

        statusObj.push_back(Pair("alias", strAlias));

        BOOST_FOREACH(CStormnodeConfig::CStormnodeEntry sne, stormnodeConfig.getEntries()) {
            if(sne.getAlias() == strAlias) {
                fFound = true;
                std::string strError;
                CStormnodeBroadcast snb;

                bool fResult = CStormnodeBroadcast::Create(sne.getIp(), sne.getPrivKey(), sne.getTxHash(), sne.getOutputIndex(), strError, snb, true);

                statusObj.push_back(Pair("result", fResult ? "successful" : "failed"));
                if(fResult) {
                    vecSnb.push_back(snb);
                    CDataStream ssVecSnb(SER_NETWORK, PROTOCOL_VERSION);
                    ssVecSnb << vecSnb;
                    statusObj.push_back(Pair("hex", HexStr(ssVecSnb.begin(), ssVecSnb.end())));
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

        std::vector<CStormnodeConfig::CStormnodeEntry> snEntries;
        snEntries = stormnodeConfig.getEntries();

        int nSuccessful = 0;
        int nFailed = 0;

        UniValue resultsObj(UniValue::VOBJ);
        std::vector<CStormnodeBroadcast> vecSnb;

        BOOST_FOREACH(CStormnodeConfig::CStormnodeEntry sne, stormnodeConfig.getEntries()) {
            std::string strError;
            CStormnodeBroadcast snb;

            bool fResult = CStormnodeBroadcast::Create(sne.getIp(), sne.getPrivKey(), sne.getTxHash(), sne.getOutputIndex(), strError, snb, true);

            UniValue statusObj(UniValue::VOBJ);
            statusObj.push_back(Pair("alias", sne.getAlias()));
            statusObj.push_back(Pair("result", fResult ? "successful" : "failed"));

            if(fResult) {
                nSuccessful++;
                vecSnb.push_back(snb);
            } else {
                nFailed++;
                statusObj.push_back(Pair("errorMessage", strError));
            }

            resultsObj.push_back(Pair("status", statusObj));
        }

        CDataStream ssVecSnb(SER_NETWORK, PROTOCOL_VERSION);
        ssVecSnb << vecSnb;
        UniValue returnObj(UniValue::VOBJ);
        returnObj.push_back(Pair("overall", strprintf("Successfully created broadcast messages for %d stormnodes, failed to create %d, total %d", nSuccessful, nFailed, nSuccessful + nFailed)));
        returnObj.push_back(Pair("detail", resultsObj));
        returnObj.push_back(Pair("hex", HexStr(ssVecSnb.begin(), ssVecSnb.end())));

        return returnObj;
    }

    if (strCommand == "decode")
    {
        if (params.size() != 2)
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Correct usage is 'stormnodebroadcast decode \"hexstring\"'");

        std::vector<CStormnodeBroadcast> vecSnb;

        if (!DecodeHexVecSnb(vecSnb, params[1].get_str()))
            throw JSONRPCError(RPC_DESERIALIZATION_ERROR, "Stormnode broadcast message decode failed");

        int nSuccessful = 0;
        int nFailed = 0;
        int nDos = 0;
        UniValue returnObj(UniValue::VOBJ);

        BOOST_FOREACH(CStormnodeBroadcast& snb, vecSnb) {
            UniValue resultObj(UniValue::VOBJ);

            if(snb.CheckSignature(nDos)) {
                nSuccessful++;
                resultObj.push_back(Pair("vin", snb.vin.ToString()));
                resultObj.push_back(Pair("addr", snb.addr.ToString()));
                resultObj.push_back(Pair("pubKeyCollateralAddress", CDarkSilkAddress(snb.pubKeyCollateralAddress.GetID()).ToString()));
                resultObj.push_back(Pair("pubKeyStormnode", CDarkSilkAddress(snb.pubKeyStormnode.GetID()).ToString()));
                resultObj.push_back(Pair("vchSig", EncodeBase64(&snb.vchSig[0], snb.vchSig.size())));
                resultObj.push_back(Pair("sigTime", snb.sigTime));
                resultObj.push_back(Pair("protocolVersion", snb.nProtocolVersion));
                resultObj.push_back(Pair("nLastSsq", snb.nLastSsq));

                UniValue lastPingObj(UniValue::VOBJ);
                lastPingObj.push_back(Pair("vin", snb.lastPing.vin.ToString()));
                lastPingObj.push_back(Pair("blockHash", snb.lastPing.blockHash.ToString()));
                lastPingObj.push_back(Pair("sigTime", snb.lastPing.sigTime));
                lastPingObj.push_back(Pair("vchSig", EncodeBase64(&snb.lastPing.vchSig[0], snb.lastPing.vchSig.size())));

                resultObj.push_back(Pair("lastPing", lastPingObj));
            } else {
                nFailed++;
                resultObj.push_back(Pair("errorMessage", "Stormnode broadcast signature verification failed"));
            }

            returnObj.push_back(Pair(snb.GetHash().ToString(), resultObj));
        }

        returnObj.push_back(Pair("overall", strprintf("Successfully decoded broadcast messages for %d stormnodes, failed to decode %d, total %d", nSuccessful, nFailed, nSuccessful + nFailed)));

        return returnObj;
    }

    if (strCommand == "relay")
    {
        if (params.size() < 2 || params.size() > 3)
            throw JSONRPCError(RPC_INVALID_PARAMETER,   "stormnodebroadcast relay \"hexstring\" ( fast )\n"
                                                        "\nArguments:\n"
                                                        "1. \"hex\"      (string, required) Broadcast messages hex string\n"
                                                        "2. fast       (string, optional) If none, using safe method\n");

        std::vector<CStormnodeBroadcast> vecSnb;

        if (!DecodeHexVecSnb(vecSnb, params[1].get_str()))
            throw JSONRPCError(RPC_DESERIALIZATION_ERROR, "Stormnode broadcast message decode failed");

        int nSuccessful = 0;
        int nFailed = 0;
        bool fSafe = params.size() == 2;
        UniValue returnObj(UniValue::VOBJ);

        // verify all signatures first, bailout if any of them broken
        BOOST_FOREACH(CStormnodeBroadcast& snb, vecSnb) {
            UniValue resultObj(UniValue::VOBJ);

            resultObj.push_back(Pair("vin", snb.vin.ToString()));
            resultObj.push_back(Pair("addr", snb.addr.ToString()));

            int nDos = 0;
            bool fResult;
            if (snb.CheckSignature(nDos)) {
                if (fSafe) {
                    fResult = snodeman.CheckSnbAndUpdateStormnodeList(NULL, snb, nDos);
                } else {
                    snodeman.UpdateStormnodeList(snb);
                    snb.Relay();
                    fResult = true;
                }
                snodeman.NotifyStormnodeUpdates();
            } else fResult = false;

            if(fResult) {
                nSuccessful++;
                resultObj.push_back(Pair(snb.GetHash().ToString(), "successful"));
            } else {
                nFailed++;
                resultObj.push_back(Pair("errorMessage", "Stormnode broadcast signature verification failed"));
            }

            returnObj.push_back(Pair(snb.GetHash().ToString(), resultObj));
        }

        returnObj.push_back(Pair("overall", strprintf("Successfully relayed broadcast messages for %d stormnodes, failed to relay %d, total %d", nSuccessful, nFailed, nSuccessful + nFailed)));

        return returnObj;
    }

    return NullUniValue;
}
