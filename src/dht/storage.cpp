// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#include "storage.h"

#include "bdap/domainentry.h"
#include "dht/mutable.h"
#include "dht/mutabledb.h"
#include "util.h"

#include <libtorrent/aux_/numeric_cast.hpp>
#include <libtorrent/broadcast_socket.hpp> // for ip_v4
#include <libtorrent/config.hpp>
#include <libtorrent/hex.hpp>
#include <libtorrent/random.hpp>
#include <libtorrent/socket_io.hpp>

namespace libtorrent { 
namespace dht {

// internal
bool operator<(peer_entry const& lhs, peer_entry const& rhs)
{
    return lhs.addr.address() == rhs.addr.address()
        ? lhs.addr.port() < rhs.addr.port()
        : lhs.addr.address() < rhs.addr.address();
}

constexpr time_duration announce_interval = minutes(ANNOUNCE_INTERVAL_MINUTES);

size_t dht_bdap_storage::num_torrents() const 
{ 
    LogPrintf("********** dht_bdap_storage -- num_torrents **********\n");
    return m_map.size(); 
}

size_t dht_bdap_storage::num_peers() const
{
    LogPrintf("********** dht_bdap_storage -- num_peers **********\n");
    size_t ret = 0;
    for (auto const& t : m_map)
        ret += t.second.peers4.size() + t.second.peers6.size();
    return ret;
}

void dht_bdap_storage::update_node_ids(std::vector<node_id> const& ids)
{
    LogPrintf("********** dht_bdap_storage -- update_node_ids **********\n");
    m_node_ids = ids;
}

bool dht_bdap_storage::get_peers(sha1_hash const& info_hash, bool const noseed, bool const scrape, address const& requester, entry& peers) const
{
    //LogPrintf("********** dht_bdap_storage -- get_peers **********\n");
    auto const i = m_map.find(info_hash);
    if (i == m_map.end()) return int(m_map.size()) >= m_settings.max_torrents;

    torrent_entry const& v = i->second;
    auto const& peersv = requester.is_v4() ? v.peers4 : v.peers6;

    if (!v.name.empty()) peers["n"] = v.name;

    if (scrape)
    {
        bloom_filter<256> downloaders;
        bloom_filter<256> seeds;

        for (auto const& p : peersv)
        {
            sha1_hash const iphash = hash_address(p.addr.address());
            if (p.seed) seeds.set(iphash);
            else downloaders.set(iphash);
        }

        peers["BFpe"] = downloaders.to_string();
        peers["BFsd"] = seeds.to_string();
    }
    else
    {
        tcp const protocol = requester.is_v4() ? tcp::v4() : tcp::v6();
        int to_pick = m_settings.max_peers_reply;
        TORRENT_ASSERT(to_pick >= 0);
        // if these are IPv6 peers their addresses are 4x the size of IPv4
        // so reduce the max peers 4 fold to compensate
        // max_peers_reply should probably be specified in bytes
        if (!peersv.empty() && protocol == tcp::v6())
            to_pick /= 4;
        entry::list_type& pe = peers["values"].list();

        int candidates = int(std::count_if(peersv.begin(), peersv.end()
            , [=](peer_entry const& e) { return !(noseed && e.seed); }));

        to_pick = std::min(to_pick, candidates);

        for (auto iter = peersv.begin(); to_pick > 0; ++iter)
        {
            // if the node asking for peers is a seed, skip seeds from the
            // peer list
            if (noseed && iter->seed) continue;

            TORRENT_ASSERT(candidates >= to_pick);

            // pick this peer with probability
            // <peers left to pick> / <peers left in the set>
            if (random(std::uint32_t(candidates--)) > std::uint32_t(to_pick))
                continue;

            pe.emplace_back();
            std::string& str = pe.back().string();

            str.resize(18);
            std::string::iterator out = str.begin();
            detail::write_endpoint(iter->addr, out);
            str.resize(std::size_t(out - str.begin()));

            --to_pick;
        }
    }

    if (int(peersv.size()) < m_settings.max_peers)
        return false;

    // we're at the max peers stored for this torrent
    // only send a write token if the requester is already in the set
    // only check for a match on IP because the peer may be announcing
    // a different port than the one it is using to send DHT messages
    peer_entry requester_entry;
    requester_entry.addr.address(requester);
    auto requester_iter = std::lower_bound(peersv.begin(), peersv.end(), requester_entry);
    return requester_iter == peersv.end()
        || requester_iter->addr.address() != requester;
}

void dht_bdap_storage::announce_peer(sha1_hash const& info_hash, tcp::endpoint const& endp, string_view name, bool const seed)
{
    LogPrintf("********** dht_bdap_storage -- announce_peer **********\n");
    auto const ti = m_map.find(info_hash);
    torrent_entry* v;
    if (ti == m_map.end())
    {
        if (int(m_map.size()) >= m_settings.max_torrents)
        {
            // we're at capacity, drop the announce
            return;
        }

        m_counters.torrents += 1;
        v = &m_map[info_hash];
    }
    else
    {
        v = &ti->second;
    }

    // the peer announces a torrent name, and we don't have a name
    // for this torrent. Store it.
    if (!name.empty() && v->name.empty())
    {
        v->name = name.substr(0, 100).to_string();
    }

    auto& peersv = is_v4(endp) ? v->peers4 : v->peers6;

    peer_entry peer;
    peer.addr = endp;
    peer.added = aux::time_now();
    peer.seed = seed;
    auto i = std::lower_bound(peersv.begin(), peersv.end(), peer);
    if (i != peersv.end() && i->addr == endp)
    {
        *i = peer;
    }
    else if (int(peersv.size()) >= m_settings.max_peers)
    {
        // we're at capacity, drop the announce
        return;
    }
    else
    {
        peersv.insert(i, peer);
        m_counters.peers += 1;
    }
}

// Do not support get immutable item
bool dht_bdap_storage::get_immutable_item(sha1_hash const& target, entry& item) const
{
    //TODO: this is not the best place to reject an immutable_item, we should ban the sender as well
    LogPrintf("********** dht_bdap_storage -- get_immutable_item **********\n");
    return false;
}

void dht_bdap_storage::put_immutable_item(sha1_hash const& target, span<char const> buf, address const& addr)
{
    //TODO: this is not the best place to reject an immutable_item, we should ban the sender as well
    LogPrintf("********** dht_bdap_storage -- put_immutable_item **********\n");
}

bool dht_bdap_storage::get_mutable_item_seq(sha1_hash const& target, sequence_number& seq) const
{
    CMutableData mutableData;
    std::string strInfoHash = aux::to_hex(target.to_string());
    CharString vchInfoHash = vchFromString(strInfoHash);
    LogPrintf("********** dht_bdap_storage -- get_mutable_item_seq infohash = %s\n", strInfoHash);
    if (!GetLocalMutableData(vchInfoHash, mutableData)) {
        LogPrintf("********** dht_bdap_storage -- get_mutable_item_seq failed to get sequence_number for infohash = %s.\n", strInfoHash);
        return false;
    }
    seq = dht::sequence_number(mutableData.SequenceNumber);
    LogPrintf("********** dht_bdap_storage -- get_mutable_item_seq found seq = %u\n", mutableData.SequenceNumber);
    return true;
}

bool dht_bdap_storage::get_mutable_item(sha1_hash const& target, sequence_number const seq, bool const force_fill, entry& item) const
{
    CMutableData mutableData;
    std::string strInfoHash = aux::to_hex(target.to_string());
    CharString vchInfoHash = vchFromString(strInfoHash);
    if (!GetLocalMutableData(vchInfoHash, mutableData)) {
        LogPrintf("********** dht_bdap_storage -- get_mutable_item failed to get mutable data from leveldb.\n");
        return false;
    }
    item["seq"] = mutableData.SequenceNumber;
    if (force_fill || (sequence_number(0) <= seq && seq < sequence_number(mutableData.SequenceNumber)))
    {
        item["v"] = bdecode(mutableData.vchValue.begin(), mutableData.vchValue.end());
        std::array<char, 64> sig;
        aux::from_hex(mutableData.Signature(), sig.data());
        item["sig"] = sig;
        std::array<char, 32> pubKey;
        aux::from_hex(mutableData.PublicKey(), pubKey.data());
        item["k"] = pubKey;
    }
    LogPrintf("********** dht_bdap_storage -- get_mutable_item found, Value = %s, seq_found = %d, seq_query = %d\n", 
                                ExtractPutValue(mutableData.Value()), mutableData.SequenceNumber, seq.value);
    return true;
}

std::string ExtractPutValue(std::string value)
{
    std::string strReturn = "";
    size_t posStart = value.find(":") + 1;
    size_t posEnd = 0;
    if (!(posStart == std::string::npos)) {
        posEnd = value.find("e1:q3:put1:t");
        if (!(posEnd == std::string::npos) && value.size() > posEnd) {
            strReturn = value.substr(posStart, posEnd - posStart);
        }
    }
    return strReturn;
}

std::string ExtractSalt(std::string salt)
{
    std::string strReturn = "";
    size_t posEnd = salt.find("3:seqi1e3:");
    if (!(posEnd == std::string::npos) && salt.size() > posEnd) {
        strReturn = salt.substr(0, posEnd);
    }
    return strReturn;
}

void dht_bdap_storage::put_mutable_item(sha1_hash const& target
    , span<char const> buf
    , signature const& sig
    , sequence_number const seq
    , public_key const& pk
    , span<char const> salt
    , address const& addr)
{

    std::string strInfoHash = aux::to_hex(target.to_string());
    CharString vchInfoHash = vchFromString(strInfoHash);
    std::string strPutValue = std::string(buf.data());
    CharString vchPutValue = vchFromString(strPutValue);
    std::string strSignature = aux::to_hex(std::string(sig.bytes.data()));
    CharString vchSignature = vchFromString(strSignature);
    std::string strPublicKey = aux::to_hex(std::string(pk.bytes.data()));
    CharString vchPublicKey = vchFromString(strPublicKey);
    std::string strSalt = std::string(salt.data());
    CharString vchSalt = vchFromString(strSalt);
    CMutableData putMutableData(vchInfoHash, vchPublicKey, vchSignature, seq.value, vchSalt, vchPutValue);
    LogPrintf("********** dht_bdap_storage -- put_mutable_item info_hash = %s, buf_value = %s, sig = %s, pubkey = %s, salt = %s, seq = %d \n", 
                    strInfoHash, ExtractPutValue(strPutValue), strSignature, strPublicKey, ExtractSalt(strSalt), putMutableData.SequenceNumber);

    CMutableData previousData;
    if (!GetLocalMutableData(vchInfoHash, previousData)) {
        if (PutLocalMutableData(vchInfoHash, putMutableData)) {
            LogPrintf("********** dht_bdap_storage -- put_mutable_item added successfully**********\n");
        }
    }
    else {
        if (putMutableData.Value() != previousData.Value() || putMutableData.SequenceNumber != previousData.SequenceNumber) {
            if (UpdateLocalMutableData(vchInfoHash, putMutableData)) {
                LogPrintf("********** dht_bdap_storage -- put_mutable_item updated successfully**********\n");
            }
        }
        else {
            LogPrintf("********** dht_bdap_storage -- put_mutable_item value unchanged. No database operation needed. **********\n");
        }
    }
    // TODO: Log from address (addr). See touch_item in the default storage implementation.
    return;
}

int dht_bdap_storage::get_infohashes_sample(entry& item)
{
    LogPrintf("********** dht_bdap_storage -- get_infohashes_sample **********\n");
    item["interval"] = aux::clamp(m_settings.sample_infohashes_interval
        , 0, sample_infohashes_interval_max);
    item["num"] = int(m_map.size());

    refresh_infohashes_sample();

    aux::vector<sha1_hash> const& samples = m_infohashes_sample.samples;
    item["samples"] = span<char const>(
        reinterpret_cast<char const*>(samples.data()), samples.size() * 20);

    return m_infohashes_sample.count();
}

void dht_bdap_storage::tick()
{
    LogPrintf("********** dht_bdap_storage -- tick **********\n");
    // look through all peers and see if any have timed out
    for (auto i = m_map.begin(), end(m_map.end()); i != end;)
    {
        torrent_entry& t = i->second;
        purge_peers(t.peers4);
        purge_peers(t.peers6);

        if (!t.peers4.empty() || !t.peers6.empty())
        {
            ++i;
            continue;
        }

        // if there are no more peers, remove the entry altogether
        i = m_map.erase(i);
        m_counters.torrents -= 1;// peers is decreased by purge_peers
    }

    if (0 == m_settings.item_lifetime) return;

    time_point const now = aux::time_now();
    time_duration lifetime = seconds(m_settings.item_lifetime);
    // item lifetime must >= 120 minutes.
    if (lifetime < minutes(120)) lifetime = minutes(120);

    for (auto i = m_immutable_table.begin(); i != m_immutable_table.end();)
    {
        if (i->second.last_seen + lifetime > now)
        {
            ++i;
            continue;
        }
        i = m_immutable_table.erase(i);
        m_counters.immutable_data -= 1;
    }

    for (auto i = m_mutable_table.begin(); i != m_mutable_table.end();)
    {
        if (i->second.last_seen + lifetime > now)
        {
            ++i;
            continue;
        }
        i = m_mutable_table.erase(i);
        m_counters.mutable_data -= 1;
    }
}

dht_storage_counters dht_bdap_storage::counters() const
{
    LogPrintf("********** dht_bdap_storage -- counters **********\n");
    return m_counters;
}

void dht_bdap_storage::purge_peers(std::vector<peer_entry>& peers)
{
    LogPrintf("********** dht_bdap_storage -- purge_peers **********\n");
    auto now = aux::time_now();
    auto new_end = std::remove_if(peers.begin(), peers.end()
        , [=](peer_entry const& e)
    {
        return e.added + announce_interval * 3 / 2 < now;
    });

    m_counters.peers -= std::int32_t(std::distance(new_end, peers.end()));
    peers.erase(new_end, peers.end());
    // if we're using less than 1/4 of the capacity free up the excess
    if (!peers.empty() && peers.capacity() / peers.size() >= 4U)
        peers.shrink_to_fit();
}

void dht_bdap_storage::refresh_infohashes_sample()
{
    LogPrintf("********** dht_bdap_storage -- refresh_infohashes_sample **********\n");
    time_point const now = aux::time_now();
    int const interval = aux::clamp(m_settings.sample_infohashes_interval
        , 0, sample_infohashes_interval_max);

    int const max_count = aux::clamp(m_settings.max_infohashes_sample_count
        , 0, infohashes_sample_count_max);
    int const count = std::min(max_count, int(m_map.size()));

    if (interval > 0
        && m_infohashes_sample.created + seconds(interval) > now
        && m_infohashes_sample.count() >= max_count)
        return;

    aux::vector<sha1_hash>& samples = m_infohashes_sample.samples;
    samples.clear();
    samples.reserve(count);

    int to_pick = count;
    int candidates = int(m_map.size());

    for (auto const& t : m_map)
    {
        if (to_pick == 0)
            break;

        TORRENT_ASSERT(candidates >= to_pick);

        // pick this key with probability
        // <keys left to pick> / <keys left in the set>
        if (random(std::uint32_t(candidates--)) > std::uint32_t(to_pick))
            continue;

        samples.push_back(t.first);
        --to_pick;
    }

    TORRENT_ASSERT(int(samples.size()) == count);
    m_infohashes_sample.created = now;
}

std::unique_ptr<dht_storage_interface> dht_bdap_storage_constructor(dht_settings const& settings)
{
    return std::unique_ptr<dht_bdap_storage>(new dht_bdap_storage(settings));
}

} // namespace dht
} // namespace libtorrent