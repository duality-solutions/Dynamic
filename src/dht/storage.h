// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#ifndef DYNAMIC_DHT_STORAGE_H
#define DYNAMIC_DHT_STORAGE_H

#include <tuple>
#include <algorithm>
#include <utility>
#include <map>
#include <set>
#include <string>

#include <libtorrent/aux_/time.hpp>
#include <libtorrent/aux_/vector.hpp>
#include <libtorrent/kademlia/types.hpp>
#include <libtorrent/kademlia/node_id.hpp>
#include <libtorrent/kademlia/dht_settings.hpp>
#include <libtorrent/kademlia/dht_storage.hpp>
#include <libtorrent/bloom_filter.hpp>

static constexpr unsigned int ANNOUNCE_INTERVAL_MINUTES = 30;

namespace libtorrent { 
namespace dht {

struct dht_immutable_item
{
    // the actual value
    std::unique_ptr<char[]> value;
    // this counts the number of IPs we have seen
    // announcing this item, this is used to determine
    // popularity if we reach the limit of items to store
    bloom_filter<128> ips;
    // the last time we heard about this item
    // the correct interpretation of this field
    // requires a time reference
    time_point last_seen;
    // number of IPs in the bloom filter
    int num_announcers = 0;
    // size of malloced space pointed to by value
    int size = 0;
};

struct dht_mutable_item : dht_immutable_item
{
    signature sig{};
    sequence_number seq{};
    public_key key{};
    std::string salt;
};

// this is the entry for every peer
// the timestamp is there to make it possible
// to remove stale peers
struct peer_entry
{
    time_point added;
    tcp::endpoint addr;
    bool seed = 0;
};

// this is a group. It contains a set of group members
struct torrent_entry
{
    std::string name;
    std::vector<peer_entry> peers4;
    std::vector<peer_entry> peers6;
};

struct infohashes_sample
{
    aux::vector<sha1_hash> samples;
    time_point created = min_time();

    int count() const { return int(samples.size()); }
};

// return true of the first argument is a better candidate for removal, i.e.
// less important to keep
struct immutable_item_comparator
{
    explicit immutable_item_comparator(std::vector<node_id> const& node_ids) : m_node_ids(node_ids) {}
    immutable_item_comparator(immutable_item_comparator const&) = default;

    template <typename Item>
    bool operator()(std::pair<node_id const, Item> const& lhs
        , std::pair<node_id const, Item> const& rhs) const
    {
        int const l_distance = min_distance_exp(lhs.first, m_node_ids);
        int const r_distance = min_distance_exp(rhs.first, m_node_ids);

        // this is a score taking the popularity (number of announcers) and the
        // fit, in terms of distance from ideal storing node, into account.
        // each additional 5 announcers is worth one extra bit in the distance.
        // that is, an item with 10 announcers is allowed to be twice as far
        // from another item with 5 announcers, from our node ID. Twice as far
        // because it gets one more bit.
        return lhs.second.num_announcers / 5 - l_distance < rhs.second.num_announcers / 5 - r_distance;
    }

private:
    // explicitly disallow assignment, to silence msvc warning
    immutable_item_comparator& operator=(immutable_item_comparator const&) = delete;

    std::vector<node_id> const& m_node_ids;
};

using node_id = libtorrent::sha1_hash;

// picks the least important one (i.e. the one
// the fewest peers are announcing, and farthest
// from our node IDs)
template<class Item>
typename std::map<node_id, Item>::const_iterator pick_least_important_item(
    std::vector<node_id> const& node_ids, std::map<node_id, Item> const& table)
{
    return std::min_element(table.begin(), table.end(), immutable_item_comparator(node_ids));
}

constexpr int sample_infohashes_interval_max = 21600;
constexpr int infohashes_sample_count_max = 20;

class dht_bdap_storage final : public dht_storage_interface
{
public:

    explicit dht_bdap_storage(dht_settings const& settings)
        : m_settings(settings)
    {
        //LogPrintf("********** dht_bdap_storage -- constructor ********** \n");
        m_counters.reset();
    }

    ~dht_bdap_storage() override 
    {
        //LogPrintf("********** dht_bdap_storage -- destructor ********** \n");
    };

    dht_bdap_storage(dht_bdap_storage const&) = delete;
    dht_bdap_storage& operator=(dht_bdap_storage const&) = delete;

    size_t num_torrents() const override;
    size_t num_peers() const override;
    void update_node_ids(std::vector<node_id> const& ids) override;
    bool get_peers(sha1_hash const& info_hash, bool const noseed, bool const scrape, address const& requester, entry& peers) const override;
    void announce_peer(sha1_hash const& info_hash, tcp::endpoint const& endp, string_view name, bool const seed) override;
    // Do not support get immutable item
    bool get_immutable_item(sha1_hash const& target, entry& item) const override;
    // Do not support put immutable item   
    void put_immutable_item(sha1_hash const& target, span<char const> buf, address const& addr) override;
    bool get_mutable_item_seq(sha1_hash const& target, sequence_number& seq) const override;
    bool get_mutable_item(sha1_hash const& target, sequence_number const seq, bool const force_fill, entry& item) const override;
    void put_mutable_item(sha1_hash const& target
        , span<char const> buf
        , signature const& sig
        , sequence_number const seq
        , public_key const& pk
        , span<char const> salt
        , address const& addr) override;
    
    int get_infohashes_sample(entry& item) override;
    void tick() override;
    dht_storage_counters counters() const override;

private:
    dht_settings const& m_settings;
    dht_storage_counters m_counters;

    std::vector<node_id> m_node_ids;
    std::map<node_id, torrent_entry> m_map;
    std::map<node_id, dht_immutable_item> m_immutable_table;
    std::map<node_id, dht_mutable_item> m_mutable_table;

    infohashes_sample m_infohashes_sample;

    void purge_peers(std::vector<peer_entry>& peers);
    void refresh_infohashes_sample();
};

std::unique_ptr<dht_storage_interface> dht_bdap_storage_constructor(dht_settings const& settings);

std::string ExtractPutValue(std::string value);
std::string ExtractSalt(std::string salt);

} // namespace dht
} // namespace libtorrent


#endif // DYNAMIC_DHT_STORAGE_H
