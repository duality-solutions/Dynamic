// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dht/dataentry.h"
#include "utiltime.h"

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
#include <boost/algorithm/string/classification.hpp> // Include boost::for is_any_of
#include <boost/algorithm/string/split.hpp> // Include for boost::split

using namespace lt;
using namespace lt::dht;
using namespace std::placeholders;

using lt::aux::from_hex;
using lt::aux::to_hex;

void Usage()
{
    std::fprintf(stderr,
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nCOMMANDS:\n\n"
        "gen-key <key-file> -               generate ed25519 keypair and save it in\n"
        "                                   the specified file\n"
        "dump-key <key-file> -              dump ed25519 keypair from the specified key\n"
        "                                   file.\n"
        "put <key-file> <salt> <string> -   puts the specified string as a mutable\n"
        "                                   object under the public key in key-file\n"
        "get <public-key> <salt> -          get a mutable object under the specified\n"
        "                                   public key and salt\n"
        "listen -                           listen for get/put requests\n"
        "\n"
        "stop -                             stops the program. quit and exit will also\n"
        "                                   terminate the application\n\n"
        );
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
    std::printf("PutString salt = %s\n", salt.c_str());
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

int DumpKey(char const* filename)
{
    std::array<char, 32> seed;

    std::fstream f(filename, std::ios_base::in | std::ios_base::binary);
    f.read(seed.data(), seed.size());
    if (f.fail())
    {
        std::fprintf(stderr, "ERROR: invalid key file.\n");
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
        std::fprintf(stderr, "ERROR: failed to write key file.\n");
        return 1;
    }
    f.flush();
    DumpKey(filename);
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


std::string ConvertPath(const std::string& strPath)
{
    std::string strConvertPath;
    if (strPath.substr(0,1) == "~") {
        strConvertPath = getenv("HOME") + strPath.substr(1, strPath.size()-1);
    }
    else {
        strConvertPath = strPath;
    }
    return strConvertPath;
}

std::vector<unsigned char> ConvertStringToCharVector(const std::string str)
{
    return std::vector<unsigned char>(str.begin(), str.end());;
}

std::vector<unsigned char> GetPubKeyBytes(const public_key& pk)
{
    std::vector<unsigned char> vchRawPubKey;
    for(unsigned int i = 0; i < pk.bytes.size(); i++) {
        vchRawPubKey.push_back(pk.bytes[i]);
    }
    return vchRawPubKey;
}

int main(int argc, char* argv[])
{
    // skip pointer to self
    ++argv;
    --argc;

    
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
    std::printf("\nDHT Data Test Application\n");
    bool fBootStrap = false;
    char command[256];
    Usage();
    while (true) {
        std::cout << "Enter another command or type \"stop\" to end program: ";
        std::cin.getline(command,256);
        std::string strCommand = std::string(command);
        std::vector<std::string> vArgs;
        boost::split(vArgs, strCommand, boost::is_any_of(" "), boost::token_compress_on);
        if (strCommand.substr(0, 4) == "put ")
        {
            if (vArgs.size() == 4) {
                std::string strPath = ConvertPath(vArgs[1]);
                std::array<char, 32> seed;
                std::fstream f(strPath.c_str(), std::ios_base::in | std::ios_base::binary);
                f.read(seed.data(), seed.size());
                std::printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
                public_key pk;
                secret_key sk;
                std::tie(pk, sk) = ed25519_create_keypair(seed);
                if (fBootStrap == false) {
                    LoadState(s);
                    Bootstrap(s);
                    fBootStrap = true;
                }
                int64_t nStartTime = GetTimeMillis();
                s.dht_put_item(pk.bytes, std::bind(&PutString, _1, _2, _3, _4
                    , pk.bytes, sk.bytes, vArgs[3].c_str()), vArgs[2]);

                std::printf("PUT public key: %s\n", to_hex(pk.bytes).c_str());

                alert* a = WaitForAlert(s, dht_put_alert::alert_type);
                dht_put_alert* pa = alert_cast<dht_put_alert>(a);
                std::printf("%s\n", pa->message().c_str());
                int64_t nEndTime = GetTimeMillis();
                std::printf("Milliseconds = %li\n", nEndTime - nStartTime);
            }
            else
                Usage();
        }
        else if (strCommand.substr(0, 9) == "dump-key ")
        {
            if (vArgs.size() == 2) {
                std::string strPath = ConvertPath(vArgs[1]);
                std::printf("strPath: %s\n", strPath.c_str());
                std::printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
                DumpKey(strPath.c_str());

            }
            else
                Usage();
        }
        else if (strCommand.substr(0, 8) == "gen-key ")
        {
            if (vArgs.size() == 2) {
                std::string strPath = ConvertPath(vArgs[1]);
                std::printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
                GenerateKey(strPath.c_str());
            }
            else
                Usage();
        }
        else if (strCommand.substr(0, 4) == "get ")
        {
           if (vArgs.size() == 3) {
                auto const len = static_cast<std::ptrdiff_t>(strlen(vArgs[1].c_str()));
                if (len != 64)
                {
                    std::fprintf(stderr, "ERROR: public key is expected to be 64 hex digits\n");
                }
                std::array<char, 32> public_key;
                bool ret = from_hex({vArgs[1].c_str(), len}, &public_key[0]);
                if (!ret)
                {
                    std::fprintf(stderr, "ERROR: invalid hex encoding of public key\n");
                }
                std::printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
                if (fBootStrap == false) {
                    LoadState(s);
                    Bootstrap(s);
                    fBootStrap = true;
                }
                int64_t nStartTime = GetTimeMillis();
                s.dht_get_item(public_key, vArgs[2]);
                

                bool authoritative = false;

                while (!authoritative)
                {
                    alert* a = WaitForAlert(s, dht_mutable_item_alert::alert_type);

                    dht_mutable_item_alert* item = alert_cast<dht_mutable_item_alert>(a);

                    authoritative = item->authoritative;
                    std::string str = item->item.to_string();
                    std::printf("%s: %s\n", authoritative ? "auth" : "non-auth", str.c_str());
                    if (!authoritative) {
                        int64_t nGetNonAuthTime = GetTimeMillis();
                        std::printf("Non-auth milliseconds = %li\n", nGetNonAuthTime - nStartTime);
                    }
                }
                int64_t nGetAuthTime = GetTimeMillis();
                std::printf("Auth milliseconds = %li\n", nGetAuthTime - nStartTime);
            }
            else
                Usage();
        }
        else if (strCommand == "listen")
        {
            if (vArgs.size() == 1) {
                if (fBootStrap == false) {
                    LoadState(s);
                    Bootstrap(s);
                    fBootStrap = true;
                }
                while (true) {
                    s.wait_for_alert(seconds(1));
                    PrintAlerts(s);
                    int ch = std::cin.get();
                    if (ch == 27) {
                        std::printf("Listener stopped.\n");
                        break;
                    }
                }
            }
            else
                Usage();
        }
        else if (strCommand == "stop" || strCommand == "quit" || strCommand == "exit")
        {
            std::printf("Stopping DHT data test listener\n");
            exit(1);
        }
        else {
            Usage();
        }
        std::printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    }
    return SaveState(s);
}