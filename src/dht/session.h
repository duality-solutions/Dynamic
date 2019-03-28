// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DHT_SESSION_H
#define DYNAMIC_DHT_SESSION_H

#include "dht/datarecord.h"
#include "dht/sessionevents.h"
#include "sync.h"

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

typedef std::pair<int64_t, CEvent> EventPair;
typedef std::multimap<int, EventPair> EventTypeMap;
typedef std::map<std::string, CMutableGetEvent> DHTGetEventMap;

static constexpr int DHT_GET_ALERT_TYPE_CODE = 75;
static constexpr int DHT_PUT_ALERT_TYPE_CODE = 76;
static constexpr int DHT_BOOTSTRAP_ALERT_TYPE_CODE = 62;
static constexpr int DHT_STATS_ALERT_TYPE_CODE = 83;
static constexpr int DHT_ERROR_ALERT_TYPE_CODE = 73;
static constexpr int64_t DHT_RECORD_LOCK_SECONDS = 16;

typedef std::pair<std::array<char, 32>, std::string> HashRecordKey; // public key and salt pair

class CHashTableSession {
public:
    CDataRecordBuffer vDataEntries;
    libtorrent::session* Session = NULL;
    std::map<HashRecordKey, uint32_t> mPutCommands;
    std::string strPutErrorMessage;
    uint64_t nPutRecords;
    bool fShutdown;
    EventTypeMap m_EventTypeMap;
    DHTGetEventMap m_DHTGetEventMap;
    CCriticalSection cs_EventMap;
    CCriticalSection cs_DHTGetEventMap;

    CHashTableSession() : vDataEntries(CDataRecordBuffer(32)), strPutErrorMessage(""), nPutRecords(0) {};

    bool SubmitPut(const std::array<char, 32> public_key, const std::array<char, 64> private_key, const int64_t lastSequence, const CDataRecord record);

    bool SubmitGet(const std::array<char, 32>& public_key, const std::string& recordSalt);
    bool SubmitGet(const std::array<char, 32>& public_key, const std::string& recordSalt, const int64_t& timeout, 
                            std::string& recordValue, int64_t& lastSequence, bool& fAuthoritative);
    /** Get a mutable record in the libtorrent DHT */
    bool SubmitGetRecord(const std::array<char, 32>& public_key, const std::array<char, 32>& private_seed, const std::string& strOperationType, int64_t& iSequence, CDataRecord& record);
    bool SubmitGetAllRecordsAsync(const std::vector<CLinkInfo>& vchLinkInfo, const std::string& strOperationType, std::vector<CDataRecord>& vchRecords);
    bool SubmitGetAllRecordsSync(const std::vector<CLinkInfo>& vchLinkInfo, const std::string& strOperationType, std::vector<CDataRecord>& vchRecords);
    void GetDHTStats(libtorrent::session_status& stats, std::vector<libtorrent::dht_lookup>& vchDHTLookup, std::vector<libtorrent::dht_routing_bucket>& vchDHTBuckets);
    bool Bootstrap();
    bool GetAllDHTGetEvents(std::vector<CMutableGetEvent>& vchGetEvents);
    void AddToDHTGetEventMap(const std::string& infoHash, const CMutableGetEvent& event);
    void AddToEventMap(const int type, const CEvent& event);
    void CleanUpEventMap(const uint32_t timeout);

private:
    void CleanUpPutCommandMap();
    uint32_t GetLastPutDate(const HashRecordKey& recordKey);
    bool GetDataFromMap(const std::array<char, 32>& public_key, const std::string& recordSalt, CMutableGetEvent& event);
    bool LoadSessionState();
    int SaveSessionState();
    std::string GetSessionStatePath();
    bool RemoveDHTGetEvent(const std::string& infoHash);
    void StopEventListener();
    bool GetLastTypeEvent(const int& type, const int64_t& startTime, std::vector<CEvent>& events);
    bool FindDHTGetEvent(const std::string& infoHash, CMutableGetEvent& event);

};

/** Start the DHT libtorrent network threads */
void StartTorrentDHTNetwork(const CChainParams& chainparams, CConnman& connman);
/** Stop the DHT libtorrent network threads */
void StopTorrentDHTNetwork();
void StartEventListener(CHashTableSession* dhtSession);
void StopEventListener();

extern CHashTableSession* pHashTableSession;

#endif // DYNAMIC_DHT_SESSION_H
