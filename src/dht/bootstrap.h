// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#ifndef DYNAMIC_DHT_BOOTSTRAP_H
#define DYNAMIC_DHT_BOOTSTRAP_H

#include "libtorrent/session.hpp"

class CChainParams;
class CConnman;
class CKeyEd25519;

/** Start the DHT libtorrent network threads */
void StartTorrentDHTNetwork(const CChainParams& chainparams, CConnman& connman);
/** Stop the DHT libtorrent network threads */
void StopTorrentDHTNetwork();
/** Get a mutable entry in the libtorrent DHT */
bool GetDHTMutableData(std::array<char, 32> public_key, const std::string& entrySalt, std::string& entryValue, int64_t& lastSequence);
/** Set a mutable entry in the libtorrent DHT */
bool PutDHTMutableData(const std::array<char, 32>& public_key, const std::array<char, 64>& private_key, const std::string& entrySalt, const int64_t& lastSequence
                        ,char const* dhtValue, std::string& message);

extern libtorrent::session *pTorrentDHTSession;

#endif // DYNAMIC_DHT_BOOTSTRAP_H
