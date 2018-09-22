
// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#ifndef DYNAMIC_DHT_DHTSETTINGS_H
#define DYNAMIC_DHT_DHTSETTINGS_H

#include <libtorrent/session.hpp>

static const int MIN_DHT_PROTO_VERSION = 71000;

class CDHTSettings {
private:
    libtorrent::settings_pack settings;
    std::string listen_interfaces;
    std::string dht_bootstrap_nodes;
    std::string user_agent;

public:

    CDHTSettings();
    
    libtorrent::settings_pack GetSettingsPack() const;

private:

    void LoadSettings();
    void LoadPeerList();
};


#endif // DYNAMIC_DHT_DHTSETTINGS_H
