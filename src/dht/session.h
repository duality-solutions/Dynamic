// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DHT_SESSION_H
#define DYNAMIC_DHT_SESSION_H

#include "dht/dataentry.h"

#include "libtorrent/alert.hpp"
#include "libtorrent/alert_types.hpp"
#include "libtorrent/session.hpp"
#include "libtorrent/session_status.hpp"

class CChainParams;
class CConnman;
class CKeyEd25519;

namespace libtorrent {
    class entry;
}


static constexpr int DHT_GET_ALERT_TYPE_CODE = 75;
static constexpr int DHT_PUT_ALERT_TYPE_CODE = 76;
static constexpr int DHT_BOOTSTRAP_ALERT_TYPE_CODE = 62;
static constexpr int DHT_STATS_ALERT_TYPE_CODE = 83;
static constexpr int DHT_ERROR_ALERT_TYPE_CODE = 73;

class CHashTableSession {
public:
    std::vector<CDataEntry> vDataEntries;
    libtorrent::session* Session = NULL;

    CHashTableSession() {};

    bool SubmitPut(const std::array<char, 32> public_key, const std::array<char, 64> private_key, const int64_t lastSequence, const CDataEntry entry);

    bool SubmitGet(const std::array<char, 32>& public_key, const std::string& entrySalt);
    bool SubmitGet(const std::array<char, 32>& public_key, const std::string& entrySalt, const int64_t& timeout, 
                            std::string& entryValue, int64_t& lastSequence, bool& fAuthoritative);

};

bool Bootstrap();
bool LoadSessionState(libtorrent::session* dhtSession);
int SaveSessionState(libtorrent::session* dhtSession);
std::string GetSessionStatePath();

/** Start the DHT libtorrent network threads */
void StartTorrentDHTNetwork(const CChainParams& chainparams, CConnman& connman);
/** Stop the DHT libtorrent network threads */
void StopTorrentDHTNetwork();
/** Get a mutable entry in the libtorrent DHT */

void GetDHTStats(libtorrent::session_status& stats, std::vector<libtorrent::dht_lookup>& vchDHTLookup, std::vector<libtorrent::dht_routing_bucket>& vchDHTBuckets);

libtorrent::alert* WaitForResponse(libtorrent::session* dhtSession, const int alert_type, const std::array<char, 32>& public_key, const std::string& strSalt);

void put_mutable(libtorrent::entry& e, std::array<char, 64>& sig, std::int64_t& seq, std::string const& salt, 
                        std::array<char, 32> const& pk, std::array<char, 64> const& sk, char const* str, std::int64_t const& iSeq);

extern CHashTableSession* pHashTableSession;

#endif // DYNAMIC_DHT_SESSION_H
