

#include "libtorrent/session.hpp"
#include "libtorrent/hex.hpp" // for from_hex
#include "libtorrent/alert_types.hpp"
#include "libtorrent/bencode.hpp" // for bencode()
#include "libtorrent/kademlia/item.hpp" // for sign_mutable_item
#include "libtorrent/kademlia/ed25519.hpp"
#include "libtorrent/span.hpp"

#include <functional>
#include <cstdio> // for snprintf
#include <cinttypes> // for PRId64 et.al.
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace lt;
using namespace lt::dht;
using namespace std::placeholders;

using lt::aux::from_hex;
using lt::aux::to_hex;

void Usage()
{
    std::fprintf(stderr,
        "USAGE:\ndht <command> <arg>\n\nCOMMANDS:\n"
        "gen-key <key-file> -           generate ed25519 keypair and save it in\n"
        "                               the specified file\n"
        "dump-key <key-file> -          dump ed25519 keypair from the specified key\n"
        "                               file.\n"
        "put <key-file> <string> -      puts the specified string as a mutable\n"
        "                               object under the public key in key-file\n"
        "get <public-key> -             get a mutable object under the specified\n"
        "                               public key\n"
        "listen -                       listen for get/put requests\n\n\n"
        );
    exit(1);
}

void PrintAlerts(lt::session& s)
{
    s.wait_for_alert(seconds(1));
    std::vector<alert*> alerts;
    s.pop_alerts(&alerts);
    for (std::vector<alert*>::iterator i = alerts.begin(), end(alerts.end()); i != end; ++i)
    {
        std::printf("\r%s\n", (*i)->message().c_str());
        std::fflush(stdout);
    }
    std::printf("\n");
}

alert* WaitForAlert(lt::session& s, int alert_type)
{
    alert* ret = nullptr;
    bool found = false;
    while (!found)
    {
        s.wait_for_alert(seconds(5));

        std::vector<alert*> alerts;
        s.pop_alerts(&alerts);
        for (std::vector<alert*>::iterator i = alerts.begin()
            , end(alerts.end()); i != end; ++i)
        {
            if ((*i)->type() != alert_type)
            {
                static int spinner = 0;
                static const char anim[] = {'-', '\\', '|', '/'};
                std::printf("\r%c", anim[spinner]);
                std::fflush(stdout);
                spinner = (spinner + 1) & 3;
                //print some alerts?
                continue;
            }
            ret = *i;
            found = true;
        }
    }
    std::printf("\n");
    return ret;
}

void PutString(entry& e, std::array<char, 64>& sig
    , std::int64_t& seq
    , std::string const& salt
    , std::array<char, 32> const& pk
    , std::array<char, 64> const& sk
    , char const* str)
{
    using lt::dht::sign_mutable_item;

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

void Bootstrap(lt::session& s)
{
    std::printf("Bootstrapping\n");
    WaitForAlert(s, dht_bootstrap_alert::alert_type);
    std::printf("Bootstrap done.\n");
}

int SaveKey(char const* filename)
{
    std::array<char, 32> seed;

    std::fstream f(filename, std::ios_base::in | std::ios_base::binary);
    f.read(seed.data(), seed.size());
    if (f.fail())
    {
        std::fprintf(stderr, "invalid key file.\n");
        return 1;
    }

    public_key pk;
    secret_key sk;
    std::tie(pk, sk) = ed25519_create_keypair(seed);

    std::printf("public key: %s\nprivate key: %s\n"
        , to_hex(pk.bytes).c_str()
        , to_hex(sk.bytes).c_str());

    return 0;
}

int GenerateKey(char const* filename)
{
    std::array<char, 32> seed = ed25519_create_seed();

    std::fstream f(filename, std::ios_base::out | std::ios_base::binary);
    f.write(seed.data(), seed.size());
    if (f.fail())
    {
        std::fprintf(stderr, "failed to write key file.\n");
        return 1;
    }

    return 0;
}

void LoadState(lt::session& s)
{
    std::fstream f(".dht", std::ios_base::in | std::ios_base::binary | std::ios_base::ate);

    auto const size = f.tellg();
    if (static_cast<int>(size) <= 0) return;
    f.seekg(0, std::ios_base::beg);

    std::vector<char> state;
    state.resize(static_cast<std::size_t>(size));

    f.read(state.data(), size);
    if (f.fail())
    {
        std::fprintf(stderr, "failed to read .dht");
        return;
    }

    bdecode_node e;
    error_code ec;
    bdecode(state.data(), state.data() + state.size(), e, ec);
    if (ec)
        std::fprintf(stderr, "failed to parse .dht file: (%d) %s\n"
            , ec.value(), ec.message().c_str());
    else
    {
        std::printf("load dht state from .dht\n");
        s.load_state(e);
    }
}

int SaveState(lt::session& s)
{
    entry e;
    s.save_state(e, session::save_dht_state);
    std::vector<char> state;
    bencode(std::back_inserter(state), e);

    std::fstream f(".dht", std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
    f.write(state.data(), static_cast<std::streamsize>(state.size()));
    return 0;
}

int main(int argc, char* argv[])
{
    // skip pointer to self
    ++argv;
    --argc;

    if (argc < 1) Usage();

    if (argv[0] == "dump-key"_sv)
    {
        ++argv;
        --argc;
        if (argc < 1) Usage();

        return SaveKey(argv[0]);
    }

    if (argv[0] == "gen-key"_sv)
    {
        ++argv;
        --argc;
        if (argc < 1) Usage();

        return GenerateKey(argv[0]);
    }
    // BDAP LibTorrent Test Nodes
    std::string strListenInterfaces = "0.0.0.0:33444";
    std::string strBootstrapNodes = "159.203.17.98:33444,178.128.144.29:33444,206.189.30.176:33444,178.128.180.138:33444,178.128.63.114:33444,138.197.167.18:33444";
    lt::session_params params;
    params.settings.set_int(settings_pack::alert_mask, 0xffffffff); // receive all alerts
    params.settings.set_bool(settings_pack::enable_dht, true);
    params.settings.set_str(settings_pack::user_agent, "BDAP");
    params.settings.set_str(settings_pack::dht_bootstrap_nodes, strBootstrapNodes); 
    params.settings.set_str(settings_pack::listen_interfaces, strListenInterfaces);
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
    params.settings.set_bool(settings_pack::enable_outgoing_utp, false);
    params.settings.set_bool(settings_pack::enable_incoming_utp, false);
    params.settings.set_bool(settings_pack::enable_outgoing_tcp, true);
    params.settings.set_bool(settings_pack::enable_incoming_tcp, true);

    params.settings.set_bool(settings_pack::enable_dht, false);
    params.settings.set_int(settings_pack::alert_mask, 0x7fffffff);

    lt::session s(params.settings);

    params.settings.set_bool(settings_pack::enable_dht, true);
    s.apply_settings(params.settings);

    LoadState(s);

    if (argv[0] == "put"_sv)
    {
        ++argv;
        --argc;
        if (argc < 1) Usage();

        std::array<char, 32> seed;

        std::fstream f(argv[0], std::ios_base::in | std::ios_base::binary);
        f.read(seed.data(), seed.size());

        ++argv;
        --argc;
        if (argc < 1) Usage();

        public_key pk;
        secret_key sk;
        std::tie(pk, sk) = ed25519_create_keypair(seed);

        Bootstrap(s);
        s.dht_put_item(pk.bytes, std::bind(&PutString, _1, _2, _3, _4
            , pk.bytes, sk.bytes, argv[0]));

        std::printf("PUT public key: %s\n", to_hex(pk.bytes).c_str());

        alert* a = WaitForAlert(s, dht_put_alert::alert_type);
        dht_put_alert* pa = alert_cast<dht_put_alert>(a);
        std::printf("%s\n", pa->message().c_str());
    }
    else if (argv[0] == "get"_sv)
    {
        ++argv;
        --argc;
        if (argc < 1) Usage();

        auto const len = static_cast<std::ptrdiff_t>(strlen(argv[0]));
        if (len != 64)
        {
            std::fprintf(stderr, "public key is expected to be 64 hex digits\n");
            return 1;
        }
        std::array<char, 32> public_key;
        bool ret = from_hex({argv[0], len}, &public_key[0]);
        if (!ret)
        {
            std::fprintf(stderr, "invalid hex encoding of public key\n");
            return 1;
        }

        Bootstrap(s);
        s.dht_get_item(public_key);
        std::printf("GET %s\n", argv[0]);

        bool authoritative = false;

        while (!authoritative)
        {
            alert* a = WaitForAlert(s, dht_mutable_item_alert::alert_type);

            dht_mutable_item_alert* item = alert_cast<dht_mutable_item_alert>(a);

            authoritative = item->authoritative;
            std::string str = item->item.to_string();
            std::printf("%s: %s\n", authoritative ? "auth" : "non-auth", str.c_str());
        }
    }
    else if (argv[0] == "listen"_sv)
    {
        Bootstrap(s);
        while (true) {
            s.wait_for_alert(seconds(1));
            PrintAlerts(s);
            int ch = std::cin.get();
            if (ch == 27)
                break;
        }
    }
    else
    {
        Usage();
    }

    return SaveState(s);
}
