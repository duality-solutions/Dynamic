// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#include "dht/dhtsettings.h"

#include "dht/persistence.h"
#include "clientversion.h"
#include "dynode.h"
#include "dynodeman.h"
#include "net.h"
#include "primitives/transaction.h"
#include "util.h"

#include <libtorrent/kademlia/dht_storage.hpp>

using namespace libtorrent;

CDHTSettings::CDHTSettings()
{
    user_agent = "Dynamic v" + FormatFullVersion();
    // Use ports 33307 and 33337
    listen_interfaces = "0.0.0.0:33307,[::]:33307,0.0.0.0:33317,[::]:33317";
                        //"0.0.0.0:33327,[::]:33327,0.0.0.0:33337,[::]:33337"
                        //"0.0.0.0:33347,[::]:33347";
}

void CDHTSettings::LoadPeerList()
{
    std::string strPeerList = "";
    // get all Dynodes above the minimum protocol version
    std::map<COutPoint, CDynode> mapDynodes = dnodeman.GetFullDynodeMap();
    for (auto& dnpair : mapDynodes) {
        CDynode dn = dnpair.second;
        if (dn.nProtocolVersion >= MIN_DHT_PROTO_VERSION) {
            std::string strDynodeIP = dn.addr.ToString();
            size_t pos = strDynodeIP.find(":");
            if (pos != std::string::npos && strDynodeIP.size() > 5) {
                // remove port from IP address string
                strDynodeIP = strDynodeIP.substr(0, pos);
            }
            pos = strPeerList.find(strDynodeIP);
            if (pos == std::string::npos) {
                strPeerList += strDynodeIP + ":33307,";
                //strPeerList += strDynodeIP + ":33317,";
                //strPeerList += strDynodeIP + ":33327,";
                //strPeerList += strDynodeIP + ":33337,";
                //strPeerList += strDynodeIP + ":33347,";
            }
        }
    }
    // get all peers above the minimum protocol version
    if(g_connman) {
        std::vector<CNodeStats> vstats;
        g_connman->GetNodeStats(vstats);
        for (const CNodeStats& stats : vstats) {
            if (stats.nVersion >= MIN_DHT_PROTO_VERSION) {
                std::string strPeerIP = stats.addrName;
                size_t pos = strPeerIP.find(":");
                if (pos != std::string::npos && strPeerIP.size() > 5) {
                    // remove port from IP address string
                    strPeerIP = strPeerIP.substr(0, pos);
                }
                pos = strPeerList.find(strPeerIP);
                if (pos == std::string::npos) {
                    strPeerList += strPeerIP + ":33307,";
                    //strPeerList += strPeerIP + ":33317,";
                    //strPeerList += strPeerIP + ":33327,";
                    //strPeerList += strPeerIP + ":33337,";
                    //strPeerList += strPeerIP + ":33347,";
                }
            }
        }
    }
    if (strPeerList.size() > 1) {
        dht_bootstrap_nodes = strPeerList.substr(0, strPeerList.size()-1);
    }
    LogPrintf("CDHTSettings::LoadPeerList -- dht_bootstrap_nodes = %s\n", dht_bootstrap_nodes);
}

void CDHTSettings::LoadSettings()
{
    LoadPeerList();
    
    params.settings.set_bool(settings_pack::enable_dht, false);
    session* newSession = new session(params.settings);
    ses = newSession;
    if (!ses->is_dht_running()) {
        ses->set_dht_storage(dht::dht_bdap_storage_constructor);
        // General LibTorrent Settings
        params.settings.set_int(settings_pack::alert_mask, 0xffffffff); // receive all alerts
        params.settings.set_bool(settings_pack::enable_dht, true);
        params.settings.set_str(settings_pack::user_agent, user_agent);
        params.settings.set_str(settings_pack::dht_bootstrap_nodes, dht_bootstrap_nodes); 
        params.settings.set_str(settings_pack::listen_interfaces, listen_interfaces);
        params.dht_settings.max_peers_reply = 100; // default = 100
        params.dht_settings.search_branching = 10; // default = 5
        params.dht_settings.max_fail_count = 100; // default = 20
        params.dht_settings.max_torrents = 2000;
        params.dht_settings.max_dht_items = 5000; // default = 700
        params.dht_settings.max_peers = 1000; // default = 5000
        params.dht_settings.max_torrent_search_reply = 20; // default = 20

        params.dht_settings.restrict_routing_ips = true; // default = true
        params.dht_settings.restrict_search_ips = true; // default = true

        params.dht_settings.extended_routing_table = true; // default = true
        params.dht_settings.aggressive_lookups = true; // default = true
        params.dht_settings.privacy_lookups = false; // default = false
        params.dht_settings.enforce_node_id = true; // default = false

        params.dht_settings.ignore_dark_internet = true; // default = true
        params.dht_settings.block_timeout = (5 * 60); // default = (5 * 60)
        params.dht_settings.block_ratelimit = 10; // default = 5
        params.dht_settings.read_only = false; // default = false
        params.dht_settings.item_lifetime = 0; // default = 0

        params.settings.set_bool(settings_pack::enable_natpmp, true);
        params.settings.set_int(settings_pack::dht_announce_interval, (60));
        params.settings.set_bool(settings_pack::enable_outgoing_utp, true);
        params.settings.set_bool(settings_pack::enable_incoming_utp, true);
        params.settings.set_bool(settings_pack::enable_outgoing_tcp, false);
        params.settings.set_bool(settings_pack::enable_incoming_tcp, false);

        ses->apply_settings(params.settings);
        ses->set_dht_storage(dht::dht_bdap_storage_constructor);
    }
    // Dynamic LibTorrent Settings
    // see https://www.libtorrent.org/reference-Settings.html#dht_settings
    // DHT Settings

    // Apply settings.
    ses->set_dht_settings(params.dht_settings);
    ses->apply_settings(params.settings);

    // TODO: (DHT) Evaluate and test the rest of these settings.
    // ``enable_lsd``
    // Starts and stops Local Service Discovery. This service will
    // broadcast the info-hashes of all the non-private torrents on the
    // local network to look for peers on the same swarm within multicast
    // reach.
    //params.settings.set_bool(settings_pack::enable_lsd, false);

    // ``prefer_rc4``
    // if the allowed encryption level is both, setting this to true will
    // prefer rc4 if both methods are offered, plaintext otherwise
    //params.settings.set_bool(settings_pack::prefer_rc4, true); 

    // ``announce_ip`` is the ip address passed along to trackers as the
    // ``&ip=`` parameter. If left as the default, that parameter is
    // omitted.
    //settings.set_bool(settings_pack::announce_ip, false);

    // ``allow_multiple_connections_per_ip``
    // determines if connections from the same IP address as existing
    // connections should be rejected or not. Multiple connections from
    // the same IP address is not allowed by default, to prevent abusive
    // behavior by peers. It may be useful to allow such connections in
    // cases where simulations are run on the same machine, and all peers
    // in a swarm has the same IP address.
    // allow_multiple_connections_per_ip
    //settings.set_bool(settings_pack::allow_multiple_connections_per_ip, true);

    // ``send_redundant_have`` controls if have messages will be sent to
    // peers that already have the piece. This is typically not necessary,
    // but it might be necessary for collecting statistics in some cases.
    //settings.set_bool(settings_pack::send_redundant_have, true);

    // ``use_dht_as_fallback`` determines how the DHT is used. If this is
    // true, the DHT will only be used for torrents where all trackers in
    // its tracker list has failed. Either by an explicit error message or
    // a time out. This is false by default, which means the DHT is used
    // by default regardless of if the trackers fail or not.
    //settings.set_bool(settings_pack::use_dht_as_fallback, false);

    // ``upnp_ignore_nonrouters`` indicates whether or not the UPnP
    // implementation should ignore any broadcast response from a device
    // whose address is not the configured router for this machine. i.e.
    // it's a way to not talk to other people's routers by mistake.
    //settings.set_bool(settings_pack::upnp_ignore_nonrouters, true);

    // ``use_parole_mode`` specifies if parole mode should be used. Parole
    // mode means that peers that participate in pieces that fail the hash
    // check are put in a mode where they are only allowed to download
    // whole pieces. If the whole piece a peer in parole mode fails the
    // hash check, it is banned. If a peer participates in a piece that
    // passes the hash check, it is taken out of parole mode.
    //settings.set_bool(settings_pack::use_parole_mode, true);

    // ``use_read_cache`` enable and disable caching of blocks read from disk. the purpose of
    // the read cache is partly read-ahead of requests but also to avoid
    // reading blocks back from the disk multiple times for popular
    // pieces.
    //settings.set_bool(settings_pack::use_read_cache, true);

    // ``coalesce_reads``  ``coalesce_writes``
    // allocate separate, contiguous, buffers for read and write calls.
    // Only used where writev/readv cannot be used will use more RAM but
    // may improve performance
    //settings.set_bool(settings_pack::coalesce_reads, false);
    //settings.set_bool(settings_pack::coalesce_writes, false);

    // ``auto_manage_prefer_seeds`` 
    // prefer seeding torrents when determining which torrents to give
    // active slots to, the default is false which gives preference to
    // downloading torrents
    //settings.set_bool(settings_pack::auto_manage_prefer_seeds, false);

    // if ``dont_count_slow_torrents`` is true, torrents without any
    // payload transfers are not subject to the ``active_seeds`` and
    // ``active_downloads`` limits. This is intended to make it more
    // likely to utilize all available bandwidth, and avoid having
    // torrents that don't transfer anything block the active slots.
    //settings.set_bool(settings_pack::dont_count_slow_torrents, false);

    // ``close_redundant_connections`` specifies whether libtorrent should
    // close connections where both ends have no utility in keeping the
    // connection open. For instance if both ends have completed their
    // downloads, there's no point in keeping it open.
    //settings.set_bool(settings_pack::close_redundant_connections, false);

    // If ``prioritize_partial_pieces`` is true, partial pieces are picked
    // before pieces that are more rare. If false, rare pieces are always
    // prioritized, unless the number of partial pieces is growing out of
    // proportion.
    //settings.set_bool(settings_pack::prioritize_partial_pieces, false);

    // if ``rate_limit_ip_overhead`` set to true, the estimated TCP/IP overhead is drained from the
    // rate limiters, to avoid exceeding the limits with the total traffic
    //settings.set_bool(settings_pack::rate_limit_ip_overhead, true);

    // ``announce_to_all_trackers`` controls how multi tracker torrents
    // are treated. If this is set to true, all trackers in the same tier
    // are announced to in parallel. If all trackers in tier 0 fails, all
    // trackers in tier 1 are announced as well. If it's set to false, the
    // behavior is as defined by the multi tracker specification. It
    // defaults to false, which is the same behavior previous versions of
    // libtorrent has had as well.
    //settings.set_bool(settings_pack::announce_to_all_trackers, true);

    // ``announce_to_all_tiers`` also controls how multi tracker torrents
    // are treated. When this is set to true, one tracker from each tier
    // is announced to. This is the uTorrent behavior. This is false by
    // default in order to comply with the multi-tracker specification.
    //settings.set_bool(settings_pack::announce_to_all_tiers, true);
    
    // ``prefer_udp_trackers`` is true by default. It means that trackers
    // may be rearranged in a way that udp trackers are always tried
    // before http trackers for the same hostname. Setting this to false
    // means that the trackers' tier is respected and there's no
    // preference of one protocol over another.
    //settings.set_bool(settings_pack::prefer_udp_trackers, true);

    // ``strict_super_seeding`` when this is set to true, a piece has to
    // have been forwarded to a third peer before another one is handed
    // out. This is the traditional definition of super seeding.
    //settings.set_bool(settings_pack::strict_super_seeding, true);

    // ``disable_hash_checks`` when set to true, all data downloaded from peers will be assumed to
    // be correct, and not tested to match the hashes in the torrent this
    // is only useful for simulation and testing purposes (typically
    // combined with disabled_storage)
    //settings.set_bool(settings_pack::disable_hash_checks, false);

    // ``allow_i2p_mixed`` if this is true, i2p torrents are allowed to also get peers from
    // other sources than the tracker, and connect to regular IPs, not
    // providing any anonymization. This may be useful if the user is not
    // interested in the anonymization of i2p, but still wants to be able
    // to connect to i2p peers.
    //settings.set_bool(settings_pack::allow_i2p_mixed, true);

    // ``volatile_read_cache``, if this is set to true, read cache blocks
    // that are hit by peer read requests are removed from the disk cache
    // to free up more space. This is useful if you don't expect the disk
    // cache to create any cache hits from other peers than the one who
    // triggered the cache line to be read into the cache in the first
    // place.
    //settings.set_bool(settings_pack::volatile_read_cache, true);
#ifndef WIN32
    // ``no_atime_storage`` this is a linux-only option and passes in the
    // ``O_NOATIME`` to ``open()`` when opening files. This may lead to
    // some disk performance improvements.
    //settings.set_bool(settings_pack::no_atime_storage, true);
#endif
    // ``incoming_starts_queued_torrents`` defaults to false. If a torrent
    // has been paused by the auto managed feature in libtorrent, i.e. the
    // torrent is paused and auto managed, this feature affects whether or
    // not it is automatically started on an incoming connection. The main
    // reason to queue torrents, is not to make them unavailable, but to
    // save on the overhead of announcing to the trackers, the DHT and to
    // avoid spreading one's unchoke slots too thin. If a peer managed to
    // find us, even though we're no in the torrent anymore, this setting
    // can make us start the torrent and serve it.
    //settings.set_bool(settings_pack::incoming_starts_queued_torrents, false);  //TODO: should this be true?

    // ``report_true_downloaded`` when set to true, the downloaded counter 
    // sent to trackers will include the actual number of payload 
    // bytes downloaded including redundant bytes. If set to false, it 
    // will not include any redundancy bytes
    //settings.set_bool(settings_pack::report_true_downloaded, false);

    // ``strict_end_game_mode`` defaults to true, and controls when a
    // block may be requested twice. If this is ``true``, a block may only
    // be requested twice when there's ay least one request to every piece
    // that's left to download in the torrent. This may slow down progress
    // on some pieces sometimes, but it may also avoid downloading a lot
    // of redundant bytes. If this is ``false``, libtorrent attempts to
    // use each peer connection to its max, by always requesting
    // something, even if it means requesting something that has been
    // requested from another peer already.
    //settings.set_bool(settings_pack::strict_end_game_mode, true);

    // if ``broadcast_lsd`` is set to true, the local peer discovery (or
    // Local Service Discovery) will not only use IP multicast, but also
    // broadcast its messages. This can be useful when running on networks
    // that don't support multicast. Since broadcast messages might be
    // expensive and disruptive on networks, only every 8th announce uses
    // broadcast.
    //settings.set_bool(settings_pack::broadcast_lsd, true);

    // ``seeding_outgoing_connections`` determines if seeding (and
    // finished) torrents should attempt to make outgoing connections or
    // not. By default this is true. It may be set to false in very
    // specific applications where the cost of making outgoing connections
    // is high, and there are no or small benefits of doing so. For
    // instance, if no nodes are behind a firewall or a NAT, seeds don't
    // need to make outgoing connections.
    //settings.set_bool(settings_pack::seeding_outgoing_connections, true);

    // ``no_connect_privileged_ports`` when this is true, 
    // libtorrent will not attempt to make outgoing connections
    // to peers whose port is < 1024. This is a safety precaution
    // to avoid being part of a DDoS attack
    //settings.set_bool(settings_pack::no_connect_privileged_ports, true); //TODO: should this be true?

    // ``smooth_connects`` is true by default, which means the number of
    // connection attempts per second may be limited to below the
    // ``connection_speed``, in case we're close to bump up against the
    // limit of number of connections. The intention of this setting is to
    // more evenly distribute our connection attempts over time, instead
    // of attempting to connect in batches, and timing them out in
    // batches.
    //settings.set_bool(settings_pack::smooth_connects, true);

    // ``always_send_user_agent`` 
    // always send user-agent in every web seed request. If false, only
    // the first request per http connection will include the user agent
    //settings.set_bool(settings_pack::always_send_user_agent, true);

    // ``apply_ip_filter_to_trackers`` defaults to true. It determines
    // whether the IP filter applies to trackers as well as peers. If this
    // is set to false, trackers are exempt from the IP filter (if there
    // is one). If no IP filter is set, this setting is irrelevant.
    //settings.set_bool(settings_pack::apply_ip_filter_to_trackers, true);

    // ``ban_web_seeds`` when true, web seeds sending bad data will be banned
    //settings.set_bool(settings_pack::ban_web_seeds, true); //TODO: should this be true?

    // ``allow_partial_disk_writes`` 
    // when set to false, the ``write_cache_line_size`` will apply across
    // piece boundaries. this is a bad idea unless the piece picker also
    // is configured to have an affinity to pick pieces belonging to the
    // same write cache line as is configured in the disk cache.
    //settings.set_bool(settings_pack::allow_partial_disk_writes, true);

    // ``support_share_mode`` if false, prevents libtorrent to advertise share-mode support
    //settings.set_bool(settings_pack::support_share_mode, true);

    // ``support_merkle_torrents`` if this is false, don't advertise support for the Tribler merkle
    // tree piece message
    //settings.set_bool(settings_pack::support_merkle_torrents, true);

    // ``report_redundant_bytes`` if this is true, the number of redundant bytes is sent to the
    // tracker
    //settings.set_bool(settings_pack::report_redundant_bytes, false);

    // ``listen_system_port_fallback`` if this is true, 
    // libtorrent will fall back to listening on a port chosen by the
    // operating system (i.e. binding to port 0). If a failure is
    // preferred, set this to false.
    //settings.set_bool(settings_pack::listen_system_port_fallback, true);

    // ``announce_crypto_support`` when this is true, and incoming encrypted connections are enabled,
    // &supportcrypt=1 is included in http tracker announces
    //settings.set_bool(settings_pack::announce_crypto_support, true); //TODO: should this be true?

    // ``enable_upnp``
    // Starts and stops the UPnP service. When started, the listen port
    // and the DHT port are attempted to be forwarded on local UPnP router
    // devices.
    // The upnp object returned by ``start_upnp()`` can be used to add and
    // remove arbitrary port mappings. Mapping status is returned through
    // the portmap_alert and the portmap_error_alert. The object will be
    // valid until ``stop_upnp()`` is called. See upnp-and-nat-pmp_.
    //settings.set_bool(settings_pack::enable_upnp, false);

    // ``proxy_hostnames``
    // if true, hostname lookups are done via the configured proxy (if
    // any). This is only supported by SOCKS5 and HTTP.
    //settings.set_bool(settings_pack::proxy_hostnames, true);

    // ``proxy_peer_connections``
    // if true, peer connections are made (and accepted) over the
    // configured proxy, if any. Web seeds as well as regular bittorrent
    // peer connections are considered "peer connections". Anything
    // transporting actual torrent payload (trackers and DHT traffic are
    // not considered peer connections).
    //settings.set_bool(settings_pack::proxy_peer_connections, true);

    // ``auto_sequential``
    // if this setting is true, torrents with a very high availability of
    // pieces (and seeds) are downloaded sequentially. This is more
    // efficient for the disk I/O. With many seeds, the download order is
    // unlikely to matter anyway
    //settings.set_bool(settings_pack::auto_sequential, true);

    // ``proxy_tracker_connections``
    // if true, tracker connections are made over the configured proxy, if
    // any.
    //,
    //settings.set_bool(settings_pack::proxy_tracker_connections, true);

    // Starts and stops the internal IP table route changes notifier.
    //
    // The current implementation supports multiple platforms, and it is
    // recommended to have it enable, but you may want to disable it if
    // it's supported but unreliable, or if you have a better way to
    // detect the changes. In the later case, you should manually call
    // ``session_handle::reopen_network_sockets`` to ensure network
    // changes are taken in consideration.
    //enable_ip_notifier,


    // int params:

    // ``tracker_completion_timeout`` is the number of seconds the tracker
    // connection will wait from when it sent the request until it
    // considers the tracker to have timed-out.
    //tracker_completion_timeout = int_type_base,

    // ``tracker_receive_timeout`` is the number of seconds to wait to
    // receive any data from the tracker. If no data is received for this
    // number of seconds, the tracker will be considered as having timed
    // out. If a tracker is down, this is the kind of timeout that will
    // occur.
    //tracker_receive_timeout,

    // ``stop_tracker_timeout`` is the number of seconds to wait when
    // sending a stopped message before considering a tracker to have
    // timed out. This is usually shorter, to make the client quit faster.
    // If the value is set to 0, the connections to trackers with the
    // stopped event are suppressed.
    //stop_tracker_timeout,

    // this is the maximum number of bytes in a tracker response. If a
    // response size passes this number of bytes it will be rejected and
    // the connection will be closed. On gzipped responses this size is
    // measured on the uncompressed data. So, if you get 20 bytes of gzip
    // response that'll expand to 2 megabytes, it will be interrupted
    // before the entire response has been uncompressed (assuming the
    // limit is lower than 2 megs).
    //tracker_maximum_response_length,

    // the number of seconds from a request is sent until it times out if
    // no piece response is returned.
    //piece_timeout,

    // the number of seconds one block (16kB) is expected to be received
    // within. If it's not, the block is requested from a different peer
    //request_timeout,

    // the length of the request queue given in the number of seconds it
    // should take for the other end to send all the pieces. i.e. the
    // actual number of requests depends on the download rate and this
    // number.
    //request_queue_time,

    // the number of outstanding block requests a peer is allowed to queue
    // up in the client. If a peer sends more requests than this (before
    // the first one has been sent) the last request will be dropped. the
    // higher this is, the faster upload speeds the client can get to a
    // single peer.
    //max_allowed_in_request_queue,

    // ``max_out_request_queue`` is the maximum number of outstanding
    // requests to send to a peer. This limit takes precedence over
    // ``request_queue_time``. i.e. no matter the download speed, the
    // number of outstanding requests will never exceed this limit.
    //max_out_request_queue,

    // if a whole piece can be downloaded in this number of seconds, or
    // less, the peer_connection will prefer to request whole pieces at a
    // time from this peer. The benefit of this is to better utilize disk
    // caches by doing localized accesses and also to make it easier to
    // identify bad peers if a piece fails the hash check.
    //whole_pieces_threshold,

    // ``peer_timeout`` is the number of seconds the peer connection
    // should wait (for any activity on the peer connection) before
    // closing it due to time out. This defaults to 120 seconds, since
    // that's what's specified in the protocol specification. After half
    // the time out, a keep alive message is sent.
    //peer_timeout,

    // same as peer_timeout, but only applies to url-seeds. this is
    // usually set lower, because web servers are expected to be more
    // reliable.
    //urlseed_timeout,

    // controls the pipelining size of url and http seeds. i.e. the number of HTTP
    // request to keep outstanding before waiting for the first one to
    // complete. It's common for web servers to limit this to a relatively
    // low number, like 5
    //urlseed_pipeline_size,

    // number of seconds until a new retry of a url-seed takes place.
    // Default retry value for http-seeds that don't provide a valid 'retry-after' header.
    //urlseed_wait_retry,

    // sets the upper limit on the total number of files this session will
    // keep open. The reason why files are left open at all is that some
    // anti virus software hooks on every file close, and scans the file
    // for viruses. deferring the closing of the files will be the
    // difference between a usable system and a completely hogged down
    // system. Most operating systems also has a limit on the total number
    // of file descriptors a process may have open.
    //file_pool_size,

    // ``max_failcount`` is the maximum times we try to connect to a peer
    // before stop connecting again. If a peer succeeds, the failcounter
    // is reset. If a peer is retrieved from a peer source (other than
    // DHT) the failcount is decremented by one, allowing another try.
    //max_failcount,

    // the number of seconds to wait to reconnect to a peer. this time is
    // multiplied with the failcount.
    //min_reconnect_time,

    // ``peer_connect_timeout`` the number of seconds to wait after a
    // connection attempt is initiated to a peer until it is considered as
    // having timed out. This setting is especially important in case the
    // number of half-open connections are limited, since stale half-open
    // connection may delay the connection of other peers considerably.
    //peer_connect_timeout,

    // ``connection_speed`` is the number of connection attempts that are
    // made per second. If a number < 0 is specified, it will default to
    // 200 connections per second. If 0 is specified, it means don't make
    // outgoing connections at all.
    //connection_speed,

    // if a peer is uninteresting and uninterested for longer than this
    // number of seconds, it will be disconnected. default is 10 minutes
    //inactivity_timeout,

    // ``unchoke_interval`` is the number of seconds between
    // chokes/unchokes. On this interval, peers are re-evaluated for being
    // choked/unchoked. This is defined as 30 seconds in the protocol, and
    // it should be significantly longer than what it takes for TCP to
    // ramp up to it's max rate.
    //unchoke_interval,

    // ``optimistic_unchoke_interval`` is the number of seconds between
    // each *optimistic* unchoke. On this timer, the currently
    // optimistically unchoked peer will change.
    //optimistic_unchoke_interval,

    // ``num_want`` is the number of peers we want from each tracker
    // request. It defines what is sent as the ``&num_want=`` parameter to
    // the tracker.
    //num_want,

    // ``initial_picker_threshold`` specifies the number of pieces we need
    // before we switch to rarest first picking. This defaults to 4, which
    // means the 4 first pieces in any torrent are picked at random, the
    // following pieces are picked in rarest first order.
    //initial_picker_threshold,

    // the number of allowed pieces to send to peers that supports the
    // fast extensions
    //allowed_fast_set_size,

    // ``suggest_mode`` controls whether or not libtorrent will send out
    // suggest messages to create a bias of its peers to request certain
    // pieces. The modes are:
    //
    // * ``no_piece_suggestions`` which is the default and will not send
    //   out suggest messages.
    // * ``suggest_read_cache`` which will send out suggest messages for
    //   the most recent pieces that are in the read cache.
    //suggest_mode,

    // ``max_queued_disk_bytes`` is the maximum number of bytes, to
    // be written to disk, that can wait in the disk I/O thread queue.
    // This queue is only for waiting for the disk I/O thread to receive
    // the job and either write it to disk or insert it in the write
    // cache. When this limit is reached, the peer connections will stop
    // reading data from their sockets, until the disk thread catches up.
    // Setting this too low will severely limit your download rate.
    //max_queued_disk_bytes,

    // the number of seconds to wait for a handshake response from a peer.
    // If no response is received within this time, the peer is
    // disconnected.
    //handshake_timeout,

    // ``send_buffer_low_watermark`` the minimum send buffer target size
    // (send buffer includes bytes pending being read from disk). For good
    // and snappy seeding performance, set this fairly high, to at least
    // fit a few blocks. This is essentially the initial window size which
    // will determine how fast we can ramp up the send rate
    //
    // if the send buffer has fewer bytes than ``send_buffer_watermark``,
    // we'll read another 16kB block onto it. If set too small, upload
    // rate capacity will suffer. If set too high, memory will be wasted.
    // The actual watermark may be lower than this in case the upload rate
    // is low, this is the upper limit.
    //
    // the current upload rate to a peer is multiplied by this factor to
    // get the send buffer watermark. The factor is specified as a
    // percentage. i.e. 50 -> 0.5 This product is clamped to the
    // ``send_buffer_watermark`` setting to not exceed the max. For high
    // speed upload, this should be set to a greater value than 100. For
    // high capacity connections, setting this higher can improve upload
    // performance and disk throughput. Setting it too high may waste RAM
    // and create a bias towards read jobs over write jobs.
    //send_buffer_low_watermark,
    //send_buffer_watermark,
    //send_buffer_watermark_factor,

    // ``choking_algorithm`` specifies which algorithm to use to determine
    // which peers to unchoke.
    //
    // The options for choking algorithms are:
    //
    // * ``fixed_slots_choker`` is the traditional choker with a fixed
    //   number of unchoke slots (as specified by
    //   ``settings_pack::unchoke_slots_limit``).
    //
    // * ``rate_based_choker`` opens up unchoke slots based on the upload
    //   rate achieved to peers. The more slots that are opened, the
    //   marginal upload rate required to open up another slot increases.
    //
    // * ``bittyrant_choker`` attempts to optimize download rate by
    //   finding the reciprocation rate of each peer individually and
    //   prefers peers that gives the highest *return on investment*. It
    //   still allocates all upload capacity, but shuffles it around to
    //   the best peers first. For this choker to be efficient, you need
    //   to set a global upload rate limit
    //   (``settings_pack::upload_rate_limit``). For more information
    //   about this choker, see the paper_. This choker is not fully
    //   implemented nor tested.
    //
    // .. _paper: http://bittyrant.cs.washington.edu/#papers
    //
    // ``seed_choking_algorithm`` controls the seeding unchoke behavior.
    // The available options are:
    //
    // * ``round_robin`` which round-robins the peers that are unchoked
    //   when seeding. This distributes the upload bandwidht uniformly and
    //   fairly. It minimizes the ability for a peer to download everything
    //   without redistributing it.
    //
    // * ``fastest_upload`` unchokes the peers we can send to the fastest.
    //   This might be a bit more reliable in utilizing all available
    //   capacity.
    //
    // * ``anti_leech`` prioritizes peers who have just started or are
    //   just about to finish the download. The intention is to force
    //   peers in the middle of the download to trade with each other.
    //choking_algorithm,
    //seed_choking_algorithm,

    // ``cache_size`` is the disk write and read cache. It is specified
    // in units of 16 KiB blocks. Buffers that are part of a peer's send
    // or receive buffer also count against this limit. Send and receive
    // buffers will never be denied to be allocated, but they will cause
    // the actual cached blocks to be flushed or evicted. If this is set
    // to -1, the cache size is automatically set based on the amount of
    // physical RAM on the machine. If the amount of physical RAM cannot
    // be determined, it's set to 1024 (= 16 MiB).
    //
    // ``cache_expiry`` is the number of seconds from the last cached write
    // to a piece in the write cache, to when it's forcefully flushed to
    // disk. Default is 60 second.
    //
    // On 32 bit builds, the effective cache size will be limited to 3/4 of
    // 2 GiB to avoid exceeding the virtual address space limit.
    //cache_size,

    //cache_expiry,

    // determines how files are opened when they're in read only mode
    // versus read and write mode. The options are:
    //
    // enable_os_cache
    //   This is the default and files are opened normally, with the OS
    //   caching reads and writes.
    // disable_os_cache
    //   This opens all files in no-cache mode. This corresponds to the
    //   OS not letting blocks for the files linger in the cache. This
    //   makes sense in order to avoid the bittorrent client to
    //   potentially evict all other processes' cache by simply handling
    //   high throughput and large files. If libtorrent's read cache is
    //   disabled, enabling this may reduce performance.
    //
    // One reason to disable caching is that it may help the operating
    // system from growing its file cache indefinitely.
    //disk_io_write_mode,
    //disk_io_read_mode,

    // this is the first port to use for binding outgoing connections to.
    // This is useful for users that have routers that allow QoS settings
    // based on local port. when binding outgoing connections to specific
    // ports, ``num_outgoing_ports`` is the size of the range. It should
    // be more than a few
    //
    // .. warning:: setting outgoing ports will limit the ability to keep
    //    multiple connections to the same client, even for different
    //    torrents. It is not recommended to change this setting. Its main
    //    purpose is to use as an escape hatch for cheap routers with QoS
    //    capability but can only classify flows based on port numbers.
    //
    // It is a range instead of a single port because of the problems with
    // failing to reconnect to peers if a previous socket to that peer and
    // port is in ``TIME_WAIT`` state.
    //outgoing_port,
    //num_outgoing_ports,

    // ``peer_tos`` determines the TOS byte set in the IP header of every
    // packet sent to peers (including web seeds). The default value for
    // this is ``0x0`` (no marking). One potentially useful TOS mark is
    // ``0x20``, this represents the *QBone scavenger service*. For more
    // details, see QBSS_.
    //
    // .. _`QBSS`: http://qbone.internet2.edu/qbss/
    //peer_tos,

    // for auto managed torrents, these are the limits they are subject
    // to. If there are too many torrents some of the auto managed ones
    // will be paused until some slots free up. ``active_downloads`` and
    // ``active_seeds`` controls how many active seeding and downloading
    // torrents the queuing mechanism allows. The target number of active
    // torrents is ``min(active_downloads + active_seeds, active_limit)``.
    // ``active_downloads`` and ``active_seeds`` are upper limits on the
    // number of downloading torrents and seeding torrents respectively.
    // Setting the value to -1 means unlimited.
    //
    // For example if there are 10 seeding torrents and 10 downloading
    // torrents, and ``active_downloads`` is 4 and ``active_seeds`` is 4,
    // there will be 4 seeds active and 4 downloading torrents. If the
    // settings are ``active_downloads`` = 2 and ``active_seeds`` = 4,
    // then there will be 2 downloading torrents and 4 seeding torrents
    // active. Torrents that are not auto managed are not counted against
    // these limits.
    //
    // ``active_checking`` is the limit of number of simultaneous checking
    // torrents.
    //
    // ``active_limit`` is a hard limit on the number of active (auto
    // managed) torrents. This limit also applies to slow torrents.
    //
    // ``active_dht_limit`` is the max number of torrents to announce to
    // the DHT. By default this is set to 88, which is no more than one
    // DHT announce every 10 seconds.
    //
    // ``active_tracker_limit`` is the max number of torrents to announce
    // to their trackers. By default this is 360, which is no more than
    // one announce every 5 seconds.
    //
    // ``active_lsd_limit`` is the max number of torrents to announce to
    // the local network over the local service discovery protocol. By
    // default this is 80, which is no more than one announce every 5
    // seconds (assuming the default announce interval of 5 minutes).
    //
    // You can have more torrents *active*, even though they are not
    // announced to the DHT, lsd or their tracker. If some peer knows
    // about you for any reason and tries to connect, it will still be
    // accepted, unless the torrent is paused, which means it won't accept
    // any connections.
    //active_downloads,
    //active_seeds,
    //active_checking,
    //active_dht_limit,
    //active_tracker_limit,
    //active_lsd_limit,
    //active_limit,

    // ``auto_manage_interval`` is the number of seconds between the
    // torrent queue is updated, and rotated.
    //auto_manage_interval,

    // this is the limit on the time a torrent has been an active seed
    // (specified in seconds) before it is considered having met the seed
    // limit criteria. See queuing_.
    //seed_time_limit,

    // ``auto_scrape_interval`` is the number of seconds between scrapes
    // of queued torrents (auto managed and paused torrents). Auto managed
    // torrents that are paused, are scraped regularly in order to keep
    // track of their downloader/seed ratio. This ratio is used to
    // determine which torrents to seed and which to pause.
    //
    // ``auto_scrape_min_interval`` is the minimum number of seconds
    // between any automatic scrape (regardless of torrent). In case there
    // are a large number of paused auto managed torrents, this puts a
    // limit on how often a scrape request is sent.
    //auto_scrape_interval,
    //auto_scrape_min_interval,

    // ``max_peerlist_size`` is the maximum number of peers in the list of
    // known peers. These peers are not necessarily connected, so this
    // number should be much greater than the maximum number of connected
    // peers. Peers are evicted from the cache when the list grows passed
    // 90% of this limit, and once the size hits the limit, peers are no
    // longer added to the list. If this limit is set to 0, there is no
    // limit on how many peers we'll keep in the peer list.
    //
    // ``max_paused_peerlist_size`` is the max peer list size used for
    // torrents that are paused. This default to the same as
    // ``max_peerlist_size``, but can be used to save memory for paused
    // torrents, since it's not as important for them to keep a large peer
    // list.
    //max_peerlist_size,
    //max_paused_peerlist_size,

    // this is the minimum allowed announce interval for a tracker. This
    // is specified in seconds and is used as a sanity check on what is
    // returned from a tracker. It mitigates hammering misconfigured
    // trackers.
    //min_announce_interval,

    // this is the number of seconds a torrent is considered active after
    // it was started, regardless of upload and download speed. This is so
    // that newly started torrents are not considered inactive until they
    // have a fair chance to start downloading.
    //auto_manage_startup,

    // ``seeding_piece_quota`` is the number of pieces to send to a peer,
    // when seeding, before rotating in another peer to the unchoke set.
    // It defaults to 3 pieces, which means that when seeding, any peer
    // we've sent more than this number of pieces to will be unchoked in
    // favour of a choked peer.
    //seeding_piece_quota,

    // TODO: deprecate this
    // ``max_rejects`` is the number of piece requests we will reject in a
    // row while a peer is choked before the peer is considered abusive
    // and is disconnected.
    //max_rejects,

    // specifies the buffer sizes set on peer sockets. 0 (which is the
    // default) means the OS default (i.e. don't change the buffer sizes).
    // The socket buffer sizes are changed using setsockopt() with
    // SOL_SOCKET/SO_RCVBUF and SO_SNDBUFFER.
    //recv_socket_buffer_size,
    //send_socket_buffer_size,

    // the max number of bytes a single peer connection's receive buffer is
    // allowed to grow to.
    //max_peer_recv_buffer_size,

    // ``read_cache_line_size`` is the number of blocks to read into the
    // read cache when a read cache miss occurs. Setting this to 0 is
    // essentially the same thing as disabling read cache. The number of
    // blocks read into the read cache is always capped by the piece
    // boundary.
    //
    // When a piece in the write cache has ``write_cache_line_size``
    // contiguous blocks in it, they will be flushed. Setting this to 1
    // effectively disables the write cache.
    //read_cache_line_size,
    //write_cache_line_size,

    // ``optimistic_disk_retry`` is the number of seconds from a disk
    // write errors occur on a torrent until libtorrent will take it out
    // of the upload mode, to test if the error condition has been fixed.
    //
    // libtorrent will only do this automatically for auto managed
    // torrents.
    //
    // You can explicitly take a torrent out of upload only mode using
    // set_upload_mode().
    //optimistic_disk_retry,

    // ``max_suggest_pieces`` is the max number of suggested piece indices
    // received from a peer that's remembered. If a peer floods suggest
    // messages, this limit prevents libtorrent from using too much RAM.
    // It defaults to 10.
    //max_suggest_pieces,

    // ``local_service_announce_interval`` is the time between local
    // network announces for a torrent. By default, when local service
    // discovery is enabled a torrent announces itself every 5 minutes.
    // This interval is specified in seconds.
    //local_service_announce_interval,

    // ``udp_tracker_token_expiry`` is the number of seconds libtorrent
    // will keep UDP tracker connection tokens around for. This is
    // specified to be 60 seconds, and defaults to that. The higher this
    // value is, the fewer packets have to be sent to the UDP tracker. In
    // order for higher values to work, the tracker needs to be configured
    // to match the expiration time for tokens.
    //udp_tracker_token_expiry,

    // ``num_optimistic_unchoke_slots`` is the number of optimistic
    // unchoke slots to use. It defaults to 0, which means automatic.
    // Having a higher number of optimistic unchoke slots mean you will
    // find the good peers faster but with the trade-off to use up more
    // bandwidth. When this is set to 0, libtorrent opens up 20% of your
    // allowed upload slots as optimistic unchoke slots.
    //num_optimistic_unchoke_slots,

    // ``default_est_reciprocation_rate`` is the assumed reciprocation
    // rate from peers when using the BitTyrant choker. This defaults to
    // 14 kiB/s. If set too high, you will over-estimate your peers and be
    // more altruistic while finding the true reciprocation rate, if it's
    // set too low, you'll be too stingy and waste finding the true
    // reciprocation rate.
    //
    // ``increase_est_reciprocation_rate`` specifies how many percent the
    // estimated reciprocation rate should be increased by each unchoke
    // interval a peer is still choking us back. This defaults to 20%.
    // This only applies to the BitTyrant choker.
    //
    // ``decrease_est_reciprocation_rate`` specifies how many percent the
    // estimated reciprocation rate should be decreased by each unchoke
    // interval a peer unchokes us. This default to 3%. This only applies
    // to the BitTyrant choker.
    //default_est_reciprocation_rate,
    //increase_est_reciprocation_rate,
    //decrease_est_reciprocation_rate,

    // the max number of peers we accept from pex messages from a single
    // peer. this limits the number of concurrent peers any of our peers
    // claims to be connected to. If they claim to be connected to more
    // than this, we'll ignore any peer that exceeds this limit
    //max_pex_peers,

    // ``tick_interval`` specifies the number of milliseconds between
    // internal ticks. This is the frequency with which bandwidth quota is
    // distributed to peers. It should not be more than one second (i.e.
    // 1000 ms). Setting this to a low value (around 100) means higher
    // resolution bandwidth quota distribution, setting it to a higher
    // value saves CPU cycles.
    //tick_interval,

    // ``share_mode_target`` specifies the target share ratio for share
    // mode torrents. This defaults to 3, meaning we'll try to upload 3
    // times as much as we download. Setting this very high, will make it
    // very conservative and you might end up not downloading anything
    // ever (and not affecting your share ratio). It does not make any
    // sense to set this any lower than 2. For instance, if only 3 peers
    // need to download the rarest piece, it's impossible to download a
    // single piece and upload it more than 3 times. If the
    // share_mode_target is set to more than 3, nothing is downloaded.
    //share_mode_target,

    // ``upload_rate_limit`` and ``download_rate_limit`` sets
    // the session-global limits of upload and download rate limits, in
    // bytes per second. By default peers on the local network are not rate
    // limited.
    //
    // A value of 0 means unlimited.
    //
    // For fine grained control over rate limits, including making them apply
    // to local peers, see peer-classes_.
    //upload_rate_limit,
    //download_rate_limit,

    // ``unchoke_slots_limit`` is the max number of unchoked peers in the
    // session. The number of unchoke slots may be ignored depending on
    // what ``choking_algorithm`` is set to.
    //unchoke_slots_limit,    

    // ``connections_limit`` sets a global limit on the number of
    // connections opened. The number of connections is set to a hard
    // minimum of at least two per torrent, so if you set a too low
    // connections limit, and open too many torrents, the limit will not
    // be met.
    //connections_limit,

    // ``connections_slack`` is the the number of incoming connections
    // exceeding the connection limit to accept in order to potentially
    // replace existing ones.
    //connections_slack,

    // ``utp_target_delay`` is the target delay for uTP sockets in
    // milliseconds. A high value will make uTP connections more
    // aggressive and cause longer queues in the upload bottleneck. It
    // cannot be too low, since the noise in the measurements would cause
    // it to send too slow. The default is 50 milliseconds.
    // ``utp_gain_factor`` is the number of bytes the uTP congestion
    // window can increase at the most in one RTT. This defaults to 300
    // bytes. If this is set too high, the congestion controller reacts
    // too hard to noise and will not be stable, if it's set too low, it
    // will react slow to congestion and not back off as fast.
    //
    // ``utp_min_timeout`` is the shortest allowed uTP socket timeout,
    // specified in milliseconds. This defaults to 500 milliseconds. The
    // timeout depends on the RTT of the connection, but is never smaller
    // than this value. A connection times out when every packet in a
    // window is lost, or when a packet is lost twice in a row (i.e. the
    // resent packet is lost as well).
    //
    // The shorter the timeout is, the faster the connection will recover
    // from this situation, assuming the RTT is low enough.
    // ``utp_syn_resends`` is the number of SYN packets that are sent (and
    // timed out) before giving up and closing the socket.
    // ``utp_num_resends`` is the number of times a packet is sent (and
    // lost or timed out) before giving up and closing the connection.
    // ``utp_connect_timeout`` is the number of milliseconds of timeout
    // for the initial SYN packet for uTP connections. For each timed out
    // packet (in a row), the timeout is doubled. ``utp_loss_multiplier``
    // controls how the congestion window is changed when a packet loss is
    // experienced. It's specified as a percentage multiplier for
    // ``cwnd``. By default it's set to 50 (i.e. cut in half). Do not
    // change this value unless you know what you're doing. Never set it
    // higher than 100.
    //utp_target_delay,
    //utp_gain_factor,
    //utp_min_timeout,
    //utp_syn_resends,
    //utp_fin_resends,
    //utp_num_resends,
    //utp_connect_timeout,

    //utp_loss_multiplier,

    // The ``mixed_mode_algorithm`` determines how to treat TCP
    // connections when there are uTP connections. Since uTP is designed
    // to yield to TCP, there's an inherent problem when using swarms that
    // have both TCP and uTP connections. If nothing is done, uTP
    // connections would often be starved out for bandwidth by the TCP
    // connections. This mode is ``prefer_tcp``. The ``peer_proportional``
    // mode simply looks at the current throughput and rate limits all TCP
    // connections to their proportional share based on how many of the
    // connections are TCP. This works best if uTP connections are not
    // rate limited by the global rate limiter (which they aren't by
    // default).
    //mixed_mode_algorithm,

    // ``listen_queue_size`` is the value passed in to listen() for the
    // listen socket. It is the number of outstanding incoming connections
    // to queue up while we're not actively waiting for a connection to be
    // accepted. The default is 5 which should be sufficient for any
    // normal client. If this is a high performance server which expects
    // to receive a lot of connections, or used in a simulator or test, it
    // might make sense to raise this number. It will not take affect
    // until the ``listen_interfaces`` settings is updated.
    //listen_queue_size,

    // ``torrent_connect_boost`` is the number of peers to try to connect
    // to immediately when the first tracker response is received for a
    // torrent. This is a boost to given to new torrents to accelerate
    // them starting up. The normal connect scheduler is run once every
    // second, this allows peers to be connected immediately instead of
    // waiting for the session tick to trigger connections.
    // This may not be set higher than 255.
    //torrent_connect_boost,

    // ``alert_queue_size`` is the maximum number of alerts queued up
    // internally. If alerts are not popped, the queue will eventually
    // fill up to this level. Once the alert queue is full, additional
    // alerts will be dropped, and not delievered to the client. Once the
    // client drains the queue, new alerts may be delivered again. In order
    // to know that alerts have been dropped, see
    // session_handle::dropped_alerts().
    //alert_queue_size,

    // ``max_metadata_size`` is the maximum allowed size (in bytes) to be
    // received by the metadata extension, i.e. magnet links.
    //max_metadata_size,

    // the number of blocks to keep outstanding at any given time when
    // checking torrents. Higher numbers give faster re-checks but uses
    // more memory. Specified in number of 16 kiB blocks
    //checking_mem_usage,

    // if set to > 0, pieces will be announced to other peers before they
    // are fully downloaded (and before they are hash checked). The
    // intention is to gain 1.5 potential round trip times per downloaded
    // piece. When non-zero, this indicates how many milliseconds in
    // advance pieces should be announced, before they are expected to be
    // completed.
    //predictive_piece_announce,

    // for some aio back-ends, ``aio_threads`` specifies the number of
    // io-threads to use,  and ``aio_max`` the max number of outstanding
    // jobs.
    //aio_threads,
    //aio_max,

    // ``tracker_backoff`` determines how aggressively to back off from
    // retrying failing trackers. This value determines *x* in the
    // following formula, determining the number of seconds to wait until
    // the next retry:
    //
    //    delay = 5 + 5 * x / 100 * fails^2
    //
    // This setting may be useful to make libtorrent more or less
    // aggressive in hitting trackers.
    //tracker_backoff,

    // when a seeding torrent reaches either the share ratio (bytes up /
    // bytes down) or the seed time ratio (seconds as seed / seconds as
    // downloader) or the seed time limit (seconds as seed) it is
    // considered done, and it will leave room for other torrents. These
    // are specified as percentages. Torrents that are considered done will
    // still be allowed to be seeded, they just won't have priority anymore.
    // For more, see queuing_.
    //share_ratio_limit,
    //seed_time_ratio_limit,

    // peer_turnover is the percentage of peers to disconnect every
    // turnover peer_turnover_interval (if we're at the peer limit), this
    // is specified in percent when we are connected to more than limit *
    // peer_turnover_cutoff peers disconnect peer_turnover fraction of the
    // peers. It is specified in percent peer_turnover_interval is the
    // interval (in seconds) between optimistic disconnects if the
    // disconnects happen and how many peers are disconnected is
    // controlled by peer_turnover and peer_turnover_cutoff
    //peer_turnover,
    //peer_turnover_cutoff,
    //peer_turnover_interval,

    // this setting controls the priority of downloading torrents over
    // seeding or finished torrents when it comes to making peer
    // connections. Peer connections are throttled by the connection_speed
    // and the half-open connection limit. This makes peer connections a
    // limited resource. Torrents that still have pieces to download are
    // prioritized by default, to avoid having many seeding torrents use
    // most of the connection attempts and only give one peer every now
    // and then to the downloading torrent. libtorrent will loop over the
    // downloading torrents to connect a peer each, and every n:th
    // connection attempt, a finished torrent is picked to be allowed to
    // connect to a peer. This setting controls n.
    //connect_seed_every_n_download,

    // the max number of bytes to allow an HTTP response to be when
    // announcing to trackers or downloading .torrent files via the
    // ``url`` provided in ``add_torrent_params``.
    //max_http_recv_buffer_size,

    // if binding to a specific port fails, should the port be incremented
    // by one and tried again? This setting specifies how many times to
    // retry a failed port bind
    //max_retry_port_bind,

    // a bitmask combining flags from alert::category_t defining which
    // kinds of alerts to receive
    //alert_mask,

    // control the settings for incoming and outgoing connections
    // respectively. see enc_policy enum for the available options.
    // Keep in mind that protocol encryption degrades performance in
    // several respects:
    //
    // 1. It prevents "zero copy" disk buffers being sent to peers, since
    //    each peer needs to mutate the data (i.e. encrypt it) the data
    //    must be copied per peer connection rather than sending the same
    //    buffer to multiple peers.
    // 2. The encryption itself requires more CPU than plain bittorrent
    //    protocol. The highest cost is the Diffie Hellman exchange on
    //    connection setup.
    // 3. The encryption handshake adds several round-trips to the
    //    connection setup, and delays transferring data.
    //out_enc_policy,
    //in_enc_policy,

    // determines the encryption level of the connections. This setting
    // will adjust which encryption scheme is offered to the other peer,
    // as well as which encryption scheme is selected by the client. See
    // enc_level enum for options.
    //allowed_enc_level,

    // the download and upload rate limits for a torrent to be considered
    // active by the queuing mechanism. A torrent whose download rate is
    // less than ``inactive_down_rate`` and whose upload rate is less than
    // ``inactive_up_rate`` for ``auto_manage_startup`` seconds, is
    // considered inactive, and another queued torrent may be started.
    // This logic is disabled if ``dont_count_slow_torrents`` is false.
    //inactive_down_rate,
    //inactive_up_rate,

    // proxy to use, defaults to none. see proxy_type_t.
    //proxy_type,

    // the port of the proxy server
    //proxy_port,

    // sets the i2p_ SAM bridge port to connect to. set the hostname with
    // the ``i2p_hostname`` setting.
    //
    // .. _i2p: http://www.i2p2.de
    //i2p_port,

    // this determines the max number of volatile disk cache blocks. If the
    // number of volatile blocks exceed this limit, other volatile blocks
    // will start to be evicted. A disk cache block is volatile if it has
    // low priority, and should be one of the first blocks to be evicted
    // under pressure. For instance, blocks pulled into the cache as the
    // result of calculating a piece hash are volatile. These blocks don't
    // represent potential interest among peers, so the value of keeping
    // them in the cache is limited.
    //cache_size_volatile,

    // The maximum request range of an url seed in bytes. This value
    // defines the largest possible sequential web seed request. Default
    // is 16 * 1024 * 1024. Lower values are possible but will be ignored
    // if they are lower then piece size.
    // This value should be related to your download speed to prevent
    // libtorrent from creating too many expensive http requests per
    // second. You can select a value as high as you want but keep in mind
    // that libtorrent can't create parallel requests if the first request
    // did already select the whole file.
    // If you combine bittorrent seeds with web seeds and pick strategies
    // like rarest first you may find your web seed requests split into
    // smaller parts because we don't download already picked pieces
    // twice.
    //urlseed_max_request_bytes,

    // time to wait until a new retry of a web seed name lookup
    //web_seed_name_lookup_retry,

    // the number of seconds between closing the file opened the longest
    // ago. 0 means to disable the feature. The purpose of this is to
    // periodically close files to trigger the operating system flushing
    // disk cache. Specifically it has been observed to be required on
    // windows to not have the disk cache grow indefinitely.
    // This defaults to 120 seconds on windows, and disabled on other
    // systems.
    //close_file_interval,

    // the max number of web seeds to have connected per torrent at any
    // given time.
    //max_web_seed_connections,

    // the number of seconds before the internal host name resolver
    // considers a cache value timed out, negative values are interpreted
    // as zero.
    //resolver_cache_timeout,
    
    // this is the client name and version identifier sent to peers in the
    // handshake message. If this is an empty string, the user_agent is
    // used instead
    //settings.set_str(settings_pack:handshake_client_version, "Dynamic"); //todo: add version to user agent 

    // ``outgoing_interfaces``
    // sets the network interface this session will use when it opens
    // outgoing connections. By default, it binds outgoing connections to
    // INADDR_ANY and port 0 (i.e. let the OS decide). Ths parameter must
    // be a string containing one or more, comma separated, adapter names.
    // Adapter names on unix systems are of the form "eth0", "eth1",
    // "tun0", etc. When specifying multiple interfaces, they will be
    // assigned in round-robin order. This may be useful for clients that
    // are multi-homed. Binding an outgoing connection to a local IP does
    // not necessarily make the connection via the associated NIC/Adapter.
    // Setting this to an empty string will disable binding of outgoing
    // connections.
    //settings.set_bool(settings_pack::outgoing_interfaces, false);

    // ``proxy_hostname``
    // when using a poxy, this is the hostname where the proxy is running
    // see proxy_type.
    //proxy_hostname,

    // when using a proxy, these are the credentials (if any) to use when
    // connecting to it. see proxy_type
    //proxy_username,
    //proxy_password,

    // sets the i2p_ SAM bridge to connect to. set the port with the
    // ``i2p_port`` setting.
    //
    // .. _i2p: http://www.i2p2.de
    //i2p_hostname,

    // this is the fingerprint for the client. It will be used as the
    // prefix to the peer_id. If this is 20 bytes (or longer) it will be
    // truncated to 20 bytes and used as the entire peer-id
    //
    // There is a utility function, generate_fingerprint() that can be used
    // to generate a standard client peer ID fingerprint prefix.
    //peer_fingerprint,
}

