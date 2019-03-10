// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dht/session.h"

#include "dht/sessionevents.h"
#include "chainparams.h"
#include "dht/datachunk.h"
#include "dht/dataheader.h"
#include "dht/settings.h"
#include "dynode-sync.h"
#include "net.h"
#include "spork.h"
#include "util.h"
#include "utiltime.h" // for GetTimeMillis
#include "validation.h"

#include "libtorrent/alert_types.hpp"
#include "libtorrent/bencode.hpp" // for bencode()
#include <libtorrent/hex.hpp> // for to_hex
#include "libtorrent/kademlia/ed25519.hpp"
#include <libtorrent/kademlia/item.hpp> // for sign_mutable_item
#include "libtorrent/span.hpp"

#include <boost/filesystem.hpp>

#include <cstdio> // for snprintf
#include <cinttypes> // for PRId64 et.al.
#include <cstdlib>
#include <functional>
#include <fstream>
#include <thread>

using namespace libtorrent;

static std::shared_ptr<std::thread> pDHTTorrentThread;

static bool fShutdown;
static bool fStarted;

CHashTableSession* pHashTableSession;

namespace DHT {
    std::vector<std::pair<std::string, std::string>> vPutValues;

    void put_mutable_string
    (
        libtorrent::entry& e
        ,std::array<char, 64>& sig
        ,std::int64_t& seq
        ,std::string const& salt
        ,std::array<char, 32> const& pk
        ,std::array<char, 64> const& sk
        ,char const* str
        ,std::int64_t const& iSeq
    )
    {
        using dht::sign_mutable_item;
        if (str != NULL) {
            e = std::string(str);
            std::vector<char> buf;
            bencode(std::back_inserter(buf), e);
            dht::signature sign;
            seq = iSeq;
            LogPrintf("%s --\nSalt = %s\nValue = %s\nSequence = %li\n", __func__, salt, std::string(str), seq);
            sign = sign_mutable_item(buf, salt, dht::sequence_number(seq)
                , dht::public_key(pk.data())
                , dht::secret_key(sk.data()));
            sig = sign.bytes;
        }
    }
}   

bool Bootstrap()
{
    LogPrintf("dht", "DHTTorrentNetwork -- bootstrapping.\n");
    const int64_t timeout = 30000; // 30 seconds
    const int64_t startTime = GetTimeMillis();
    while (timeout > GetTimeMillis() - startTime)
    {
        std::vector<CEvent> events;
        MilliSleep(1500);
        if (GetLastTypeEvent(DHT_BOOTSTRAP_ALERT_TYPE_CODE, startTime, events)) 
        {
            if (events.size() > 0 ) {
                LogPrint("dht", "DHTTorrentNetwork -- Bootstrap successful.\n");
                return true;
            }
        }
    }
    LogPrint("dht", "DHTTorrentNetwork -- Bootstrap failed after 30 second timeout.\n");
    return false;
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
    LogPrint("dht", "DHTTorrentNetwork -- SaveSessionState complete.\n");
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
        LogPrint("dht", "DHTTorrentNetwork -- LoadSessionState failed to read dht-state.log\n");
        return false;
    }

    bdecode_node e;
    error_code ec;
    bdecode(state.data(), state.data() + state.size(), e, ec);
    if (ec) {
        LogPrint("dht", "DHTTorrentNetwork -- LoadSessionState failed to parse dht-state.log file: (%d) %s\n", ec.value(), ec.message());
        return false;
    }
    else
    {
        LogPrint("dht", "DHTTorrentNetwork -- LoadSessionState load dht state from dht-state.log\n");
        dhtSession->load_state(e);
    }
    return true;
}

void static DHTTorrentNetwork(const CChainParams& chainparams, CConnman& connman)
{
    LogPrint("dht", "DHTTorrentNetwork -- starting\n");
    SetThreadPriority(THREAD_PRIORITY_LOWEST);
    RenameThread("dht-session");
    
    try {
        CDHTSettings settings;
        // Busy-wait for the network to come online so we get a full list of Dynodes
        do {
            bool fvNodesEmpty = connman.GetNodeCount(CConnman::CONNECTIONS_ALL) == 0;
            if (!fvNodesEmpty && !IsInitialBlockDownload() && dynodeSync.IsSynced() && 
                dynodeSync.IsBlockchainSynced() && sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
                    break;

            MilliSleep(1000);
            if (fShutdown)
                return;

        } while (true);
        
        fStarted = true;
        LogPrintf("DHTTorrentNetwork -- started\n");
        // with current peers and Dynodes
        settings.LoadSettings();
        pHashTableSession->Session = settings.GetSession();
        
        if (!pHashTableSession->Session)
            throw std::runtime_error("DHT Torrent network bootstraping error.");
        
        StartEventListener(pHashTableSession->Session);
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
    MilliSleep(300);
    StopEventListener();
    MilliSleep(30);
    if (pDHTTorrentThread != NULL)
    {
        LogPrint("dht", "DHTTorrentNetwork -- StopTorrentDHTNetwork trying to stop.\n");
        if (fStarted) { 
            libtorrent::session_params params;
            params.settings.set_bool(settings_pack::enable_dht, false);
            params.settings.set_int(settings_pack::alert_mask, 0x0);
            pHashTableSession->Session->apply_settings(params.settings);
            pHashTableSession->Session->abort();
        }
        pDHTTorrentThread->join();
        LogPrint("dht", "DHTTorrentNetwork -- StopTorrentDHTNetwork abort.\n");
    }
    else {
        LogPrint("dht", "DHTTorrentNetwork --StopTorrentDHTNetwork pDHTTorrentThreads is null.  Stop not needed.\n");
    }
    pDHTTorrentThread = NULL;
    LogPrintf("DHTTorrentNetwork -- Stopped.\n");
}

void StartTorrentDHTNetwork(const CChainParams& chainparams, CConnman& connman)
{
    LogPrint("dht", "DHTTorrentNetwork -- Log file = %s.\n", GetSessionStatePath());
    fShutdown = false;
    fStarted = false;
    if (pDHTTorrentThread != NULL)
         StopTorrentDHTNetwork();

    pDHTTorrentThread = std::make_shared<std::thread>(std::bind(&DHTTorrentNetwork, std::cref(chainparams), std::ref(connman)));
}

void GetDHTStats(session_status& stats, std::vector<dht_lookup>& vchDHTLookup, std::vector<dht_routing_bucket>& vchDHTBuckets)
{
    LogPrint("dht", "DHTTorrentNetwork -- GetDHTStats started.\n");

    if (!pHashTableSession->Session) {
        return;
    }

    if (!pHashTableSession->Session->is_dht_running()) {
        return;
        //LogPrint("dht", "DHTTorrentNetwork -- GetDHTStats Restarting DHT.\n");
        //if (!LoadSessionState(pHashTableSession->Session)) {
        //    LogPrint("dht", "DHTTorrentNetwork -- GetDHTStats Couldn't load previous settings.  Trying to bootstrap again.\n");
        //    Bootstrap();
        //}
        //else {
        //    LogPrint("dht", "DHTTorrentNetwork -- GetDHTStats setting loaded from file.\n");
        //}
    }
    else {
        LogPrint("dht", "DHTTorrentNetwork -- GetDHTStats DHT already running.  Bootstrap not needed.\n");
    }

    pHashTableSession->Session->post_dht_stats();
    //get alert from map
    //alert* dhtAlert = WaitForResponse(pHashTableSession->Session, dht_stats_alert::alert_type);
    //dht_stats_alert* dhtStatsAlert = alert_cast<dht_stats_alert>(dhtAlert);
    //vchDHTLookup = dhtStatsAlert->active_requests;
    //vchDHTBuckets = dhtStatsAlert->routing_table;
    stats = pHashTableSession->Session->status();
}

bool CHashTableSession::SubmitPut(const std::array<char, 32> public_key, const std::array<char, 64> private_key, const int64_t lastSequence, CDataRecord record)
{
    vDataEntries.push_back(record);
    DHT::vPutValues.clear();
    std::vector<std::vector<unsigned char>> vPubKeys;
    std::string strSalt = record.GetHeader().Salt;
    DHT::vPutValues.push_back(std::make_pair(strSalt, record.HeaderHex));
    //TODO (DHT): Change to LogPrint to make less chatty when not in debug mode.
    LogPrintf("CHashTableSession::%s -- PutMutableData started, Salt = %s, Value = %s, lastSequence = %li, vPutValues size = %u\n", __func__, 
                                                                strSalt, record.Value(), lastSequence, DHT::vPutValues.size());
    for(const CDataChunk& chunk: record.GetChunks()) {
        DHT::vPutValues.push_back(std::make_pair(chunk.Salt, chunk.Value));
    }

    for(const std::pair<std::string, std::string>& pair: DHT::vPutValues) {
        Session->dht_put_item(public_key, std::bind(&DHT::put_mutable_string, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, 
                                public_key, private_key, pair.second.c_str(), lastSequence), pair.first);
        //TODO (DHT): Change to LogPrint to make less chatty when not in debug mode.
        LogPrintf("CHashTableSession::%s -- salt: %s, value: %s\n", __func__, pair.first, pair.second);
    }
    return true;
}

bool CHashTableSession::SubmitGet(const std::array<char, 32>& public_key, const std::string& recordSalt)
{
    //TODO: DHT add locks
    LogPrintf("CHashTableSession::%s -- started.\n", __func__);

    if (!pHashTableSession->Session) {
        //message = "DHTTorrentNetwork -- GetDHTMutableData Error. pHashTableSession->Session is null.";
        return false;
    }

    if (!pHashTableSession->Session->is_dht_running()) {
        LogPrintf("CHashTableSession::%s -- GetDHTMutableData Restarting DHT.\n", __func__);
        if (!LoadSessionState(pHashTableSession->Session)) {
            LogPrintf("DHTTorrentNetwork -- GetDHTMutableData Couldn't load previous settings.  Trying to Bootstrap again.\n");
            if (!Bootstrap())
                return false;
        }
        else {
            LogPrintf("CHashTableSession::%s -- GetDHTMutableData  setting loaded from file.\n", __func__);
        }
    }
    else {
        LogPrintf("CHashTableSession::%s -- GetDHTMutableData DHT already running.  Bootstrap not needed.\n", __func__);
    }

    Session->dht_get_item(public_key, recordSalt);
    LogPrintf("CHashTableSession::%s -- MGET: %s, salt = %s\n", __func__, aux::to_hex(public_key), recordSalt);

    return true;
}

bool CHashTableSession::SubmitGet(const std::array<char, 32>& public_key, const std::string& recordSalt, const int64_t& timeout, 
                            std::string& recordValue, int64_t& lastSequence, bool& fAuthoritative)
{
    std::string infoHash = GetInfoHash(aux::to_hex(public_key),recordSalt);
    RemoveDHTGetEvent(infoHash);
    if (!SubmitGet(public_key, recordSalt))
        return false;

    MilliSleep(40);
    CMutableGetEvent data;
    int64_t startTime = GetTimeMillis();
    while (timeout > GetTimeMillis() - startTime)
    {
        if (FindDHTGetEvent(infoHash, data)) {
            std::string strData = data.Value();
            // TODO (DHT): check the last position for the single quote character
            if (strData.substr(0, 1) == "'") {
                recordValue = strData.substr(1, strData.size() - 2);
            }
            else {
                recordValue = strData;
            }
            lastSequence = data.SequenceNumber();
            fAuthoritative = data.Authoritative();
            LogPrintf("CHashTableSession::%s -- value = %s, seq = %d, auth = %u\n", __func__, recordValue, lastSequence, fAuthoritative);
            return true;
        }
        MilliSleep(40);
    }
    return false;
}