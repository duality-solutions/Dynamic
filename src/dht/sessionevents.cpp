// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#include "dht/sessionevents.h"

#include "dht/operations.h"
#include "dht/session.h"
#include "sync.h" // for LOCK and CCriticalSection
#include "util.h" // for LogPrintf
#include "utiltime.h" // for GetTimeMillis

#include <libtorrent/alert.hpp>
#include <libtorrent/alert_types.hpp>
#include <libtorrent/kademlia/types.hpp>
#include <libtorrent/kademlia/item.hpp>
#include <libtorrent/hex.hpp> // for to_hex and from_hex
#include <libtorrent/session.hpp>
#include <libtorrent/session_status.hpp>
#include <libtorrent/time.hpp>

#include <map>

using namespace libtorrent;

typedef std::pair<int64_t, CEvent> EventPair;
typedef std::multimap<int, EventPair> EventTypeMap;
typedef std::multimap<std::string, CMutableGetEvent> DHTGetEventMap;
typedef std::multimap<std::string, CMutablePutEvent> DHTPutEventMap;
typedef std::map<int64_t, CPutRequest> DHTPutRequestMap;

static CCriticalSection cs_EventMap;
static CCriticalSection cs_DHTGetEventMap;
static CCriticalSection cs_DHTPutEventMap;
static CCriticalSection cs_DHTPutRequestMap;

static bool fShutdown;
static EventTypeMap m_EventTypeMap;
static DHTGetEventMap m_DHTGetEventMap;
static DHTPutEventMap m_DHTPutEventMap;
static DHTPutRequestMap m_DHTPutRequestMap;

CEvent::CEvent(std::string _message, int _type, uint32_t _category, std::string _what)
{
    message = _message;
    type = _type;
    category = _category;
    what = _what;
    timestamp = GetTimeMillis();
}

std::string CEvent::ToString() const 
{
    return strprintf("CEvent(Message = %s\n, Type = %d\n, Category = %d\n, What = %s\n, Timestamp = %u)\n",
        Message(), Type(), Category(), What(), Timestamp());
}

static std::string GetInfoHash(const std::string pubkey, const std::string salt)
{
    std::array<char, 32> arrPubKey;
    aux::from_hex(pubkey, arrPubKey.data());
    dht::public_key pk;
    pk.bytes = arrPubKey;
    const sha1_hash infoHash = dht::item_target_id(salt, pk);
    
    return aux::to_hex(infoHash.to_string());
}

CMutableGetEvent::CMutableGetEvent() : CEvent()
{
}
CMutableGetEvent::CMutableGetEvent(std::string _message, int _type, uint32_t _category, std::string _what, 
                                   std::string _pubkey, std::string _salt, int64_t _seq, std::string _value, std::string _signature, bool _authoritative)
                : CEvent(_message, _type, _category, _what)
{
    pubkey = _pubkey;
    salt = _salt;
    seq = _seq;
    value = _value;
    signature = _signature;
    authoritative = _authoritative;
    infohash = GetInfoHash(pubkey, salt);
}

CMutablePutEvent::CMutablePutEvent(std::string _message, int _type, uint32_t _category, std::string _what, 
                                   std::string _pubkey, std::string _salt, int64_t _seq, std::string _signature, uint32_t _success_count)
                : CEvent(_message, _type, _category, _what)
{
    pubkey = _pubkey;
    salt = _salt;
    seq = _seq;
    signature = _signature;
    success_count = _success_count;
    infohash = GetInfoHash(pubkey, salt);
}

CPutRequest::CPutRequest(const CKeyEd25519 _key, const std::string _salt, const int64_t _sequence, const std::string _value)
{
    key = _key;
    salt = _salt;
    sequence = _sequence;
    value = _value;
    timestamp = GetTimeMillis();
}

void CPutRequest::DHTPut()
{
    SubmitPutDHTMutableData(key.GetDHTPubKey(), key.GetDHTPrivKey(), salt, sequence, value.c_str());
}

static void AddToDHTGetEventMap(const MutableKey& mKey, const CMutableGetEvent& event)
{
    LOCK(cs_DHTGetEventMap);
    std::string infoHash = GetInfoHash(mKey.first, mKey.second);
    std::multimap<std::string, CMutableGetEvent>::iterator iMutableEvent = m_DHTGetEventMap.find(infoHash);
    if (iMutableEvent == m_DHTGetEventMap.end()) {
        // event not found. Add a new entry to DHT event map
        LogPrint("dht", "AddToDHTGetEventMap Not found -- infohash = %s, pubkey = %s, salt = %s\n", infoHash, mKey.first, mKey.second);
        m_DHTGetEventMap.insert(std::make_pair(infoHash, event));
    }
    else {
        // event found. Update entry in DHT event map
        LogPrint("dht", "AddToDHTGetEventMap Found -- infohash = %s, pubkey = %s, salt = %s\n", infoHash, mKey.first, mKey.second);
        iMutableEvent->second = event;
    }
}

static void AddToDHTPutEventMap(const MutableKey& mKey, const CMutablePutEvent& event)
{
    LOCK(cs_DHTPutEventMap);
    std::string infoHash = GetInfoHash(mKey.first, mKey.second);
    std::multimap<std::string, CMutablePutEvent>::iterator iMutableEvent = m_DHTPutEventMap.find(infoHash);
    if (iMutableEvent == m_DHTPutEventMap.end()) {
        // event not found. Add a new entry to DHT event map
        LogPrint("dht", "AddToDHTPutEventMap Not found -- infohash = %s, pubkey = %s, salt = %s\n", infoHash, mKey.first, mKey.second);
        m_DHTPutEventMap.insert(std::make_pair(infoHash, event));
    }
    else {
        // event found. Update entry in DHT event map
        LogPrint("dht", "AddToDHTPutEventMap Found -- infohash = %s, pubkey = %s, salt = %s\n", infoHash, mKey.first, mKey.second);
        iMutableEvent->second = event;
    }
}

static void AddToEventMap(const int type, const CEvent& event)
{
    LOCK(cs_EventMap);
    m_EventTypeMap.insert(std::make_pair(type, std::make_pair(event.Timestamp(), event)));
}

static void DHTEventListener(session* dhtSession)
{
    SetThreadPriority(THREAD_PRIORITY_LOWEST);
    RenameThread("dht-events");
    unsigned int counter = 0;
    while(!fShutdown)
    {
        if (!dhtSession->is_dht_running()) {
            LogPrint("dht", "%s -- DHT is not running yet\n", __func__);
            MilliSleep(2000);
            continue;
        }
        if (m_DHTPutRequestMap.size() > 0) {
            CPutRequest put = m_DHTPutRequestMap.begin()->second;
            put.DHTPut();
            alert* dhtAlert = WaitForResponse(dhtSession, dht_put_alert::alert_type, put.Key().GetDHTPubKey(), put.Salt());
            dht_put_alert* dhtPutAlert = alert_cast<dht_put_alert>(dhtAlert);
            std::string strMessage = dhtPutAlert->message();
            LogPrint("dht", "DHTEventListener -- DHT Processing Put Request: value = %s, salt = %s, message = %s\n", put.Value(), put.Salt(), strMessage);
            m_DHTPutRequestMap.erase(m_DHTPutRequestMap.begin()->first);
        }

        dhtSession->wait_for_alert(seconds(1));
        std::vector<alert*> alerts;
        dhtSession->pop_alerts(&alerts);
        for (std::vector<alert*>::iterator iAlert = alerts.begin(), end(alerts.end()); iAlert != end; ++iAlert) {
            if ((*iAlert) == nullptr)
                continue;

            const uint32_t iAlertCategory = (*iAlert)->category();
            const std::string strAlertMessage = (*iAlert)->message();
            const int iAlertType = (*iAlert)->type();
            const std::string strAlertTypeName = alert_name(iAlertType);
            if (iAlertType == DHT_GET_ALERT_TYPE_CODE || iAlertType == DHT_PUT_ALERT_TYPE_CODE) {
                LogPrint("dht", "DHTEventListener -- DHT Alert Message = %s, Alert Type =%s, Alert Category = %u\n", strAlertMessage, strAlertTypeName, iAlertCategory);
                MutableKey mKey;
                if (iAlertType == DHT_GET_ALERT_TYPE_CODE) {
                    // DHT Get Mutable Event
                    dht_mutable_item_alert* pGet = alert_cast<dht_mutable_item_alert>((*iAlert));
                    if (pGet == nullptr)
                        continue;

                    const CMutableGetEvent event(strAlertMessage, iAlertType, iAlertCategory, strAlertTypeName, 
                          aux::to_hex(pGet->key), pGet->salt, pGet->seq, pGet->item.to_string(), aux::to_hex(pGet->signature), pGet->authoritative);

                    mKey = std::make_pair(event.PublicKey(), event.Salt());
                    AddToDHTGetEventMap(mKey, event);
                }
                else if (iAlertType == DHT_PUT_ALERT_TYPE_CODE) {
                    // DHT Put Mutable Event
                    LogPrint("dht", "DHTEventListener -- DHT Alert Message = %s, Alert Type =%s, Alert Category = %u\n", strAlertMessage, strAlertTypeName, iAlertCategory);
                    dht_put_alert* pPut = alert_cast<dht_put_alert>((*iAlert));
                    if (pPut == nullptr)
                        continue;

                    const CMutablePutEvent event(strAlertMessage, iAlertType, iAlertCategory, strAlertTypeName, 
                          aux::to_hex(pPut->public_key), pPut->salt, pPut->seq, aux::to_hex(pPut->signature), pPut->num_success);

                    mKey = std::make_pair(event.PublicKey(), event.Salt());
                    AddToDHTPutEventMap(mKey, event);
                }
            }
            else if (iAlertType == DHT_STATS_ALERT_TYPE_CODE) {
                // TODO (dht): handle stats 
                //dht_stats_alert* pAlert = alert_cast<dht_stats_alert>((*iAlert));
                LogPrintf("%s -- DHT Alert Message: AlertType = %s\n", __func__, strAlertTypeName); 
            }
            else {
                const CEvent event(strAlertMessage, iAlertType, iAlertCategory, strAlertTypeName);
                AddToEventMap(iAlertType, event);
            }
        }
        if (fShutdown)
            return;
        
        counter++;
        if (counter % 60 == 0) {
            LogPrint("dht", "DHTEventListener -- Before CleanUpEventMap. counter = %u\n", counter);
            CleanUpEventMap(300000);
        }
    }
}

void CleanUpEventMap(uint32_t timeout)
{
    unsigned int deleted = 0;
    unsigned int counter = 0;
    int64_t iTime = GetTimeMillis();
    LOCK(cs_EventMap);
    for (auto it = m_EventTypeMap.begin(); it != m_EventTypeMap.end(); ) {
        CEvent event = it->second.second;
        if ((iTime - event.Timestamp()) > timeout) {
            it = m_EventTypeMap.erase(it);
            deleted++;
        }
        else {
            ++it;
        }
        counter++;
    }
    LogPrint("dht", "DHTEventListener -- CleanUpEventMap. deleted = %u, count = %u\n", deleted, counter);
}

void StopEventListener()
{
	fShutdown = true;
    LogPrint("dht", "DHTEventListener -- stopping.\n");
    MilliSleep(2100);
}

void StartEventListener(session* dhtSession)
{
    LogPrint("dht", "StartEventListener -- start\n");
    fShutdown = false;
    DHTEventListener(dhtSession);
}

bool GetLastTypeEvent(const int& type, const int64_t& startTime, std::vector<CEvent>& events)
{
    //LOCK(cs_EventMap);
    LogPrint("dht", "GetLastTypeEvent -- m_EventTypeMap.size = %u, type = %u.\n", m_EventTypeMap.size(), type);
    std::multimap<int, EventPair>::iterator iEvents = m_EventTypeMap.find(type);
    while (iEvents != m_EventTypeMap.end()) {
        if (iEvents->second.first >= startTime) {
            events.push_back(iEvents->second.second);
        }
        iEvents++;
    }
    LogPrint("dht", "GetLastTypeEvent -- events.size() = %u\n", events.size());
    return events.size() > 0;
}

bool FindDHTGetEvent(const MutableKey& mKey, CMutableGetEvent& event)
{
    //LOCK(cs_DHTGetEventMap);
    std::string infoHash = GetInfoHash(mKey.first, mKey.second);
    std::multimap<std::string, CMutableGetEvent>::iterator iMutableEvent = m_DHTGetEventMap.find(infoHash);
    if (iMutableEvent != m_DHTGetEventMap.end()) {
        // event found.
        LogPrint("dht", "FindDHTGetEvent -- Found, infoHash = %s, pubkey = %s, salt = %s\n", infoHash, mKey.first, mKey.second);
        event = iMutableEvent->second;
        return true;
    }
    LogPrint("dht", "FindDHTGetEvent -- Not found, infoHash = %s, pubkey = %s, salt = %s\n", infoHash, mKey.first, mKey.second);
    return false;
}

bool FindDHTPutEvent(const MutableKey& mKey, CMutablePutEvent& event)
{
    //LOCK(cs_DHTPutEventMap);
    std::string infoHash = GetInfoHash(mKey.first, mKey.second);
    std::multimap<std::string, CMutablePutEvent>::iterator iMutableEvent = m_DHTPutEventMap.find(infoHash);
    if (iMutableEvent != m_DHTPutEventMap.end()) {
        // event found.
        LogPrint("dht", "FindDHTPutEvent -- Found, infoHash = %s, pubkey = %s, salt = %s\n", infoHash, mKey.first, mKey.second);
        event = iMutableEvent->second;
        return true;
    }
    return false;
}

bool GetAllDHTPutEvents(std::vector<CMutablePutEvent>& vchPutEvents)
{
    //LOCK(cs_DHTPutEventMap);
    for (std::multimap<std::string, CMutablePutEvent>::iterator it=m_DHTPutEventMap.begin(); it!=m_DHTPutEventMap.end(); ++it) {
        vchPutEvents.push_back(it->second);
    }
    return true;
}

bool GetAllDHTGetEvents(std::vector<CMutableGetEvent>& vchGetEvents)
{
    //LOCK(cs_DHTGetEventMap);
    for (std::multimap<std::string, CMutableGetEvent>::iterator it=m_DHTGetEventMap.begin(); it!=m_DHTGetEventMap.end(); ++it) {
        vchGetEvents.push_back(it->second);
    }
    return true;
}

void AddPutRequest(CPutRequest& put)
{
    LOCK(cs_DHTPutRequestMap);
    m_DHTPutRequestMap.insert(std::make_pair(put.Timestamp(), put));
}