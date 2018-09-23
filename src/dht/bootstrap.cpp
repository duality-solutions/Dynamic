// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#include "dht/bootstrap.h"

#include "chainparams.h"
#include "dht/dhtsettings.h"
#include "dynode-sync.h"
#include "net.h"
#include "util.h"
#include "validation.h"

#include "libtorrent/hex.hpp" // for to_hex
#include "libtorrent/alert_types.hpp"
#include "libtorrent/bencode.hpp" // for bencode()
#include "libtorrent/kademlia/item.hpp" // for sign_mutable_item
#include "libtorrent/kademlia/ed25519.hpp"
#include "libtorrent/span.hpp"

#include <boost/thread.hpp>
#include <boost/filesystem.hpp>

#include <functional>
#include <cstdio> // for snprintf
#include <cinttypes> // for PRId64 et.al.
#include <cstdlib>
#include <fstream>

using namespace libtorrent;

session *pTorrentDHTSession = NULL;

static alert* wait_for_alert(session* dhtSession, int alert_type)
{
    LogPrintf("DHTTorrentNetwork -- wait_for_alert start.\n");
    alert* ret = nullptr;
    bool found = false;
    while (!found)
    {
        dhtSession->wait_for_alert(seconds(5));

        std::vector<alert*> alerts;
        dhtSession->pop_alerts(&alerts);
        for (std::vector<alert*>::iterator i = alerts.begin()
            , end(alerts.end()); i != end; ++i)
        {
            if ((*i)->type() != alert_type)
            {
                //print some alerts?
                // LogPrintf("DHTTorrentNetwork -- alert = .\n");
                continue;
            }
            ret = *i;
            found = true;
        }
    }
    LogPrintf("DHTTorrentNetwork -- wait_for_alert complete.\n");
    return ret;
}

static void bootstrap(lt::session* dhtSession)
{
    LogPrintf("DHTTorrentNetwork -- bootstrapping.\n");
    wait_for_alert(dhtSession, dht_bootstrap_alert::alert_type);
    LogPrintf("DHTTorrentNetwork -- bootstrap done.\n");
}

static std::string get_log_path()
{
    boost::filesystem::path path = GetDataDir() / "dht-state.dat";
    return path.string();
}

static int save_dht_state(session* dhtSession)
{
    entry torrentEntry;
    dhtSession->save_state(torrentEntry, session::save_dht_state);
    std::vector<char> state;
    bencode(std::back_inserter(state), torrentEntry);
    std::fstream f(get_log_path().c_str(), std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
    f.write(state.data(), state.size());
    return 0;
}

static void load_dht_state(session* dhtSession)
{
    std::fstream f(get_log_path().c_str(), std::ios_base::in | std::ios_base::binary | std::ios_base::ate);

    auto const size = f.tellg();
    if (static_cast<int>(size) <= 0) return;
    f.seekg(0, std::ios_base::beg);

    std::vector<char> state;
    state.resize(static_cast<std::size_t>(size));

    f.read(state.data(), state.size());
    if (f.fail())
    {
        LogPrintf("DHTTorrentNetwork -- failed to read dht-state.log\n");
        return;
    }

    bdecode_node e;
    error_code ec;
    bdecode(state.data(), state.data() + state.size(), e, ec);
    if (ec) {
        LogPrintf("DHTTorrentNetwork -- failed to parse dht-state.log file: (%d) %s\n", ec.value(), ec.message());
    }
    else
    {
        LogPrintf("DHTTorrentNetwork -- load dht state from dht-state.log\n");
        dhtSession->load_state(e);
    }
}

void static DHTTorrentNetwork(const CChainParams& chainparams, CConnman& connman)
{
    LogPrintf("DHTTorrentNetwork -- started\n");
    SetThreadPriority(THREAD_PRIORITY_LOWEST);
    RenameThread("dht-torrent-network");

    boost::shared_ptr<CReserveScript> coinbaseScript;
    
    try {
        CDHTSettings settings;
        // Busy-wait for the network to come online so we get a full list of Dynodes
        do {
            bool fvNodesEmpty = connman.GetNodeCount(CConnman::CONNECTIONS_ALL) == 0;
            if (!fvNodesEmpty && !IsInitialBlockDownload() && dynodeSync.IsSynced() && dynodeSync.IsBlockchainSynced())
                break;
            MilliSleep(1000);
        } while (true);

        // boot strap the DHT LibTorrent network
        // with current peers and Dynodes
        settings.LoadSettings();
        pTorrentDHTSession = new session(settings.GetSettingsPack());
        load_dht_state(pTorrentDHTSession);
        bootstrap(pTorrentDHTSession);
        save_dht_state(pTorrentDHTSession);
        if (!pTorrentDHTSession) {
            throw std::runtime_error("DHT Torrent network bootstraping error.");
        }
        while (true) {}
    }
    catch (const boost::thread_interrupted&)
    {
        LogPrintf("DHTTorrentNetwork -- terminated\n");
        throw;
    }
    catch (const std::runtime_error& e)
    {
        LogPrintf("DHTTorrentNetwork -- runtime error: %s\n", e.what());
        return;
    }
}

void StopTorrentDHTNetwork()
{
    static boost::thread_group* dhtTorrentThreads;
    if (dhtTorrentThreads != NULL)
    {
        dhtTorrentThreads->interrupt_all();
        delete dhtTorrentThreads;
        dhtTorrentThreads = NULL;
        LogPrintf("DHTTorrentNetwork -- StopTorrentDHTNetwork stopped.\n");
    }
    else {
        LogPrintf("DHTTorrentNetwork -- StopTorrentDHTNetwork dhtTorrentThreads is null.  Stop not needed.\n");
    }
}

void StartTorrentDHTNetwork(const CChainParams& chainparams, CConnman& connman)
{
    LogPrintf("DHTTorrentNetwork -- Log file = %s.\n", get_log_path());
    static boost::thread_group* dhtTorrentThreads;

    StopTorrentDHTNetwork();

    dhtTorrentThreads = new boost::thread_group();
    dhtTorrentThreads->create_thread(boost::bind(&DHTTorrentNetwork, boost::cref(chainparams), boost::ref(connman)));
}

bool GetDHTMutableData(std::array<char, 32> public_key, const std::string& entrySalt, std::string& entryValue, int64_t& lastSequence)
{
    //TODO: DHT add locks
    LogPrintf("DHTTorrentNetwork -- PutMutableData started.\n");

    if (!pTorrentDHTSession) 
        return false;

    bootstrap(pTorrentDHTSession);

    pTorrentDHTSession->dht_get_item(public_key, entrySalt);
    LogPrintf("DHTTorrentNetwork -- MGET: %s, salt = %s\n", aux::to_hex(public_key), entrySalt);

    bool authoritative = false;
    while (!authoritative)
    {
        alert* dhtAlert = wait_for_alert(pTorrentDHTSession, dht_mutable_item_alert::alert_type);

        dht_mutable_item_alert* dhtGetAlert = alert_cast<dht_mutable_item_alert>(dhtAlert);

        authoritative = dhtGetAlert->authoritative;
        entryValue = dhtGetAlert->item.to_string();
        lastSequence = dhtGetAlert->seq;
        LogPrintf("%s: %s\n", authoritative ? "auth" : "non-auth", entryValue);
    }
    save_dht_state(pTorrentDHTSession);

    return true;
}

static void put_string(
    entry& e
    ,std::array<char, 64>& sig
    ,std::int64_t& seq
    ,std::string const& salt
    ,std::array<char, 32> const& pk
    ,std::array<char, 64> const& sk
    ,char const* str)
{
    using dht::sign_mutable_item;

    e = std::string(str);
    std::vector<char> buf;
    bencode(std::back_inserter(buf), e);
    dht::signature sign;
    ++seq;
    sign = sign_mutable_item(buf, salt, dht::sequence_number(seq)
        , dht::public_key(pk.data())
        , dht::secret_key(sk.data()));
    sig = sign.bytes;
}

bool PutDHTMutableData(const std::array<char, 32>& public_key, const std::array<char, 64>& private_key, const std::string& entrySalt, const int64_t& lastSequence
                        ,char const* dhtValue, std::string& message)
{
    //TODO: (DHT) add locks
    LogPrintf("DHTTorrentNetwork -- PutMutableData started.\n");

    if (!pTorrentDHTSession) 
        return false;

    bootstrap(pTorrentDHTSession);

    entry e;
    std::array<char, 64> sig;
    pTorrentDHTSession->dht_put_item(public_key, std::bind(&put_string, e, sig, lastSequence + 1, entrySalt, public_key, private_key, dhtValue));

    LogPrintf("DHTTorrentNetwork -- MPUT public key: %s, salt = %s\n", aux::to_hex(public_key), entrySalt);
    alert* dhtAlert = wait_for_alert(pTorrentDHTSession, dht_put_alert::alert_type);
    dht_put_alert* dhtPutAlert = alert_cast<dht_put_alert>(dhtAlert);
    message = dhtPutAlert->message();
    LogPrintf("%s\n", message);
    save_dht_state(pTorrentDHTSession);

    return true;
}