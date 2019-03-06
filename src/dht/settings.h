// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DHT_SETTINGS_H
#define DYNAMIC_DHT_SETTINGS_H

#include <libtorrent/session.hpp>

static constexpr int MIN_DHT_PROTO_VERSION = 71000;

class CDHTSettings {
private:
    libtorrent::session* ses;
    libtorrent::session_params params;
    std::string listen_interfaces;
    std::string dht_bootstrap_nodes;
    std::string user_agent;
    uint16_t nPort;

public:
    CDHTSettings();

    void LoadSettings();

    libtorrent::settings_pack GetSettingsPack() const { return params.settings; }
    libtorrent::dht_settings GetDHTSettings() const { return params.dht_settings; }
    libtorrent::session_params GetSessionParams() const { return params; }
    libtorrent::session* GetSession() const { return ses; }

private:
    void LoadPeerList();

};

#endif // DYNAMIC_DHT_SETTINGS_H