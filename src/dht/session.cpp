// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#include "dht/session.h"

#include "chainparams.h"
#include "dht/settings.h"
#include "dynode-sync.h"
#include "net.h"
#include "util.h"
#include "validation.h"

#include "libtorrent/hex.hpp" // for to_hex
#include "libtorrent/alert_types.hpp"
#include "libtorrent/bencode.hpp" // for bencode()
#include "libtorrent/kademlia/ed25519.hpp"
#include "libtorrent/span.hpp"

#include <boost/filesystem.hpp>

#include <cstdio> // for snprintf
#include <cinttypes> // for PRId64 et.al.
#include <cstdlib>
#include <fstream>
#include <thread>

using namespace libtorrent;

static std::shared_ptr<std::thread> pDHTTorrentThread;

static bool fShutdown;
session *pTorrentDHTSession = NULL;

static void empty_public_key(std::array<char, 32>& public_key)
{
    for( unsigned int i = 0; i < sizeof(public_key); i++) {
        public_key[i] = 0;
    }
}

alert* WaitForResponse(session* dhtSession, const int alert_type, const std::array<char, 32> public_key, const std::string strSalt)
{
    LogPrintf("DHTTorrentNetwork -- WaitForResponse start.\n");
    alert* ret = nullptr;
    bool found = false;
    std::array<char, 32> emptyKey;
    empty_public_key(emptyKey);
    std::string strEmpty = aux::to_hex(emptyKey);
    std::string strPublicKey = aux::to_hex(public_key);
    while (!found)
    {
        dhtSession->wait_for_alert(seconds(1));
        std::vector<alert*> alerts;
        dhtSession->pop_alerts(&alerts);
        for (std::vector<alert*>::iterator iAlert = alerts.begin(), end(alerts.end()); iAlert != end; ++iAlert)
        {
            std::string strAlertMessage = (*iAlert)->message();
            int iAlertType = (*iAlert)->type();
            if ((*iAlert)->category() == 0x1) {
                LogPrintf("DHTTorrentNetwork -- error alert message = %s, alert_type =%d\n", strAlertMessage, iAlertType);
            }
            else if ((*iAlert)->category() == 0x80) {
                LogPrintf("DHTTorrentNetwork -- progress alert message = %s, alert_type =%d\n", strAlertMessage, iAlertType);
            }
            else if ((*iAlert)->category() == 0x200) {
                LogPrintf("DHTTorrentNetwork -- performance warning alert message = %s, alert_type =%d\n", strAlertMessage, iAlertType);
            }
            else if ((*iAlert)->category() == 0x400) {
                LogPrintf("DHTTorrentNetwork -- dht alert message = %s, alert_type =%d\n", strAlertMessage, iAlertType);
            }
            if (iAlertType != alert_type)
            {
                continue;
            }
            
            size_t posKey = strAlertMessage.find("key=" + strPublicKey);
            size_t posSalt = strAlertMessage.find("salt=" + strSalt);
            if (strPublicKey == strEmpty || (posKey != std::string::npos && posSalt != std::string::npos)) {
                LogPrintf("DHTTorrentNetwork -- wait alert complete. message = %s, alert_type =%d\n", strAlertMessage, iAlertType);
                ret = *iAlert;
                found = true;
            }
        }
        if (fShutdown)
            return ret;
    }
    return ret;
}

static alert* WaitForResponse(session* dhtSession, const int alert_type)
{
    std::array<char, 32> emptyKey;
    empty_public_key(emptyKey);
    return WaitForResponse(dhtSession, alert_type, emptyKey, "");
}

void Bootstrap(libtorrent::session* dhtSession)
{
    LogPrintf("DHTTorrentNetwork -- bootstrapping.\n");
    WaitForResponse(dhtSession, dht_bootstrap_alert::alert_type);
    LogPrintf("DHTTorrentNetwork -- bootstrap done.\n");
}

std::string GetSessionStatePath()
{
    boost::filesystem::path path = GetDataDir() / "dht_state.dat";
    return path.string();
}

int SaveSessionState(session* dhtSession)
{
    entry torrentEntry;
    dhtSession->save_state(torrentEntry, session::save_dht_state);
    std::vector<char> state;
    bencode(std::back_inserter(state), torrentEntry);
    std::fstream f(GetSessionStatePath().c_str(), std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
    f.write(state.data(), state.size());
    LogPrintf("DHTTorrentNetwork -- SaveSessionState complete.\n");
    return 0;
}

bool LoadSessionState(session* dhtSession)
{
    std::fstream f(GetSessionStatePath().c_str(), std::ios_base::in | std::ios_base::binary | std::ios_base::ate);

    auto const size = f.tellg();
    if (static_cast<int>(size) <= 0) return false;
    f.seekg(0, std::ios_base::beg);

    std::vector<char> state;
    state.resize(static_cast<std::size_t>(size));

    f.read(state.data(), state.size());
    if (f.fail())
    {
        LogPrintf("DHTTorrentNetwork -- LoadSessionState failed to read dht-state.log\n");
        return false;
    }

    bdecode_node e;
    error_code ec;
    bdecode(state.data(), state.data() + state.size(), e, ec);
    if (ec) {
        LogPrintf("DHTTorrentNetwork -- LoadSessionState failed to parse dht-state.log file: (%d) %s\n", ec.value(), ec.message());
        return false;
    }
    else
    {
        LogPrintf("DHTTorrentNetwork -- LoadSessionState load dht state from dht-state.log\n");
        dhtSession->load_state(e);
    }
    return true;
}

void static DHTTorrentNetwork(const CChainParams& chainparams, CConnman& connman)
{
    LogPrintf("DHTTorrentNetwork -- started\n");
    SetThreadPriority(THREAD_PRIORITY_LOWEST);
    RenameThread("dht-torrent-network");
    
    try {
        CDHTSettings settings;
        // Busy-wait for the network to come online so we get a full list of Dynodes
        do {
            bool fvNodesEmpty = connman.GetNodeCount(CConnman::CONNECTIONS_ALL) == 0;
            if (!fvNodesEmpty && !IsInitialBlockDownload() && dynodeSync.IsSynced() && dynodeSync.IsBlockchainSynced())
                break;
            MilliSleep(1000);
            if (fShutdown)
                break;

        } while (true);

        // boot strap the DHT LibTorrent network
        // with current peers and Dynodes
        unsigned int iCounter = 0;
        settings.LoadSettings();
        pTorrentDHTSession = settings.GetSession();
        Bootstrap(pTorrentDHTSession);
        SaveSessionState(pTorrentDHTSession);
        if (!pTorrentDHTSession) {
            throw std::runtime_error("DHT Torrent network bootstraping error.");
        }
        while (!fShutdown) {
            MilliSleep(1000);
            iCounter ++;
            if (!pTorrentDHTSession->is_dht_running()) {
                LogPrintf("DHTTorrentNetwork -- not running.  Loading from file and restarting bootstrap.\n");
                LoadSessionState(pTorrentDHTSession);
                Bootstrap(pTorrentDHTSession);
                SaveSessionState(pTorrentDHTSession);
            }
            else {
                if (iCounter >= 300) {
                    // save DHT state every 5 minutes
                    SaveSessionState(pTorrentDHTSession);
                    iCounter = 0;
                }
            }
        }
    }
    catch (const std::runtime_error& e)
    {
        fShutdown = true;
        LogPrintf("DHTTorrentNetwork -- runtime error: %s\n", e.what());
        return;
    }
}

void StopTorrentDHTNetwork()
{
    LogPrintf("DHTTorrentNetwork -- StopTorrentDHTNetwork begin.\n");
    fShutdown = true;
    MilliSleep(1100);
    if (pDHTTorrentThread != NULL)
    {
        LogPrintf("DHTTorrentNetwork -- StopTorrentDHTNetwork trying to stop.\n");
        libtorrent::session_params params;
        params.settings.set_bool(settings_pack::enable_dht, false);
        params.settings.set_int(settings_pack::alert_mask, 0x0);
        pTorrentDHTSession->apply_settings(params.settings);
        pTorrentDHTSession->abort();
        pDHTTorrentThread->join();
        LogPrintf("DHTTorrentNetwork -- StopTorrentDHTNetwork abort.\n");
    }
    else {
        LogPrintf("DHTTorrentNetwork --StopTorrentDHTNetwork pDHTTorrentThreads is null.  Stop not needed.\n");
    }
    pDHTTorrentThread = NULL;
}

void StartTorrentDHTNetwork(const CChainParams& chainparams, CConnman& connman)
{
    LogPrintf("DHTTorrentNetwork -- Log file = %s.\n", GetSessionStatePath());
    fShutdown = false;
    if (pDHTTorrentThread != NULL)
         StopTorrentDHTNetwork();

    pDHTTorrentThread = std::make_shared<std::thread>(std::bind(&DHTTorrentNetwork, std::cref(chainparams), std::ref(connman)));
}

void GetDHTStats(session_status& stats, std::vector<dht_lookup>& vchDHTLookup, std::vector<dht_routing_bucket>& vchDHTBuckets)
{
    LogPrintf("DHTTorrentNetwork -- GetDHTStats started.\n");

    if (!pTorrentDHTSession) {
        return;
    }

    if (!pTorrentDHTSession->is_dht_running()) {
        LogPrintf("DHTTorrentNetwork -- GetDHTStats Restarting DHT.\n");
        if (!LoadSessionState(pTorrentDHTSession)) {
            LogPrintf("DHTTorrentNetwork -- GetDHTStats Couldn't load previous settings.  Trying to bootstrap again.\n");
            Bootstrap(pTorrentDHTSession);
        }
        else {
            LogPrintf("DHTTorrentNetwork -- GetDHTStats setting loaded from file.\n");
        }
    }
    else {
        LogPrintf("DHTTorrentNetwork -- GetDHTStats DHT already running.  Bootstrap not needed.\n");
    }

    pTorrentDHTSession->post_dht_stats();
    alert* dhtAlert = WaitForResponse(pTorrentDHTSession, dht_stats_alert::alert_type);
    dht_stats_alert* dhtStatsAlert = alert_cast<dht_stats_alert>(dhtAlert);
    vchDHTLookup = dhtStatsAlert->active_requests;
    vchDHTBuckets = dhtStatsAlert->routing_table;
    stats = pTorrentDHTSession->status();
}