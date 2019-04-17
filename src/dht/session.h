// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DHT_SESSION_H
#define DYNAMIC_DHT_SESSION_H

#include "dht/datarecord.h"

#include "libtorrent/alert.hpp"
#include "libtorrent/alert_types.hpp"
#include "libtorrent/session.hpp"
#include "libtorrent/session_status.hpp"

class CChainParams;
class CConnman;
class CKeyEd25519;
class CLinkInfo;
class CMutableGetEvent;

namespace libtorrent {
    class entry;
}


static constexpr int DHT_GET_ALERT_TYPE_CODE = 75;
static constexpr int DHT_PUT_ALERT_TYPE_CODE = 76;
static constexpr int DHT_BOOTSTRAP_ALERT_TYPE_CODE = 62;
static constexpr int DHT_STATS_ALERT_TYPE_CODE = 83;
static constexpr int DHT_ERROR_ALERT_TYPE_CODE = 73;
static constexpr int64_t DHT_RECORD_LOCK_SECONDS = 16;
static constexpr uint32_t DHT_KEEP_PUT_BUFFER_SECONDS = 300;

typedef std::pair<std::array<char, 32>, std::string> HashRecordKey; // public key and salt pair

class CHashTableSession {
public:
    CDataRecordBuffer vDataEntries;
    libtorrent::session* Session = NULL;
    std::map<HashRecordKey, int64_t> mPutCommands;
    std::string strPutErrorMessage;
    uint64_t nPutRecords;

    CHashTableSession() : vDataEntries(CDataRecordBuffer(32)), strPutErrorMessage(""), nPutRecords(0) {};

    bool SubmitPut(const std::array<char, 32> public_key, const std::array<char, 64> private_key, const int64_t lastSequence, CDataRecord& record);

    bool SubmitGet(const std::array<char, 32>& public_key, const std::string& recordSalt);
    bool SubmitGet(const std::array<char, 32>& public_key, const std::string& recordSalt, const int64_t& timeout, 
                            std::string& recordValue, int64_t& lastSequence, bool& fAuthoritative);
    bool SubmitGetRecord(const std::array<char, 32>& public_key, const std::array<char, 32>& private_seed, const std::string& strOperationType, int64_t& iSequence, CDataRecord& record);
    bool SubmitGetAllRecordsAsync(const std::vector<CLinkInfo>& vchLinkInfo, const std::string& strOperationType, std::vector<CDataRecord>& vchRecords);
    bool SubmitGetAllRecordsSync(const std::vector<CLinkInfo>& vchLinkInfo, const std::string& strOperationType, std::vector<CDataRecord>& vchRecords);

private:
    void CleanUpPutCommandMap();
    int64_t GetLastPutDate(const HashRecordKey& recordKey);
    bool GetDataFromMap(const std::array<char, 32>& public_key, const std::string& recordSalt, CMutableGetEvent& event);

};

bool Bootstrap();
bool LoadSessionState(libtorrent::session* dhtSession);
int SaveSessionState(libtorrent::session* dhtSession);
std::string GetSessionStatePath();

/** Start the DHT libtorrent network threads */
void StartTorrentDHTNetwork(const CChainParams& chainparams, CConnman& connman);
/** Stop the DHT libtorrent network threads */
void StopTorrentDHTNetwork();
/** Get a mutable record in the libtorrent DHT */

void GetDHTStats(libtorrent::session_status& stats, std::vector<libtorrent::dht_lookup>& vchDHTLookup, std::vector<libtorrent::dht_routing_bucket>& vchDHTBuckets);

extern CHashTableSession* pHashTableSession;

#endif // DYNAMIC_DHT_SESSION_H