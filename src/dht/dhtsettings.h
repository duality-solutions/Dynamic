
// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#ifndef DYNAMIC_DHT_DHTSETTINGS_H
#define DYNAMIC_DHT_DHTSETTINGS_H

#include <libtorrent/session.hpp>

class CDHTSettings {
private:
    
    libtorrent::settings_pack settings;

public:

    CDHTSettings();
    
    libtorrent::settings_pack GetSettingsPack() const;

private:

	void LoadSettings();
};


#endif // DYNAMIC_DHT_DHTSETTINGS_H
