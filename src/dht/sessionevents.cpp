// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#include "dht/sessionevents.h"

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
#include <mutex>
#include <thread>

using namespace libtorrent;

typedef std::pair<int64_t, CEvent> EventPair;
typedef std::multimap<uint32_t, EventPair> EventCategoryMap;
typedef std::map<MutableKey, CMutableGetEvent> DHTGetEventMap;
typedef std::map<MutableKey, CMutablePutEvent> DHTPutEventMap;

static CCriticalSection cs_EventMap;
static CCriticalSection cs_DHTGetEventMap;
static CCriticalSection cs_DHTPutEventMap;

static bool fShutdown;
static std::shared_ptr<std::thread> pEventListenerThread;
static EventCategoryMap m_EventCategoryMap;
static DHTGetEventMap m_DHTGetEventMap;
static DHTPutEventMap m_DHTPutEventMap;

CEvent::CEvent(libtorrent::alert* alert) {
    SetAlert(alert);
}

void CEvent::SetAlert(libtorrent::alert* alert)
{
    if (alert != nullptr) {
        pAlert = alert;
        timestamp = GetTimeMillis();
    }
}

std::string CEvent::Message() const
{
    if (pAlert == nullptr)
        return "";
    return pAlert->message();
}

int CEvent::Type() const
{
    if (pAlert == nullptr)
        return 0;
    return pAlert->type();
}

uint32_t CEvent::Category() const 
{
    if (pAlert == nullptr)
        return 0;
    return pAlert->category();
}

std::string CEvent::What() const 
{
    if (pAlert == nullptr)
        return "";
    return (std::string)pAlert->what();
}

std::int64_t CEvent::Timestamp() const 
{
    return timestamp;
}

std::string CEvent::ToString() const 
{
    if (pAlert == nullptr)
        return "";
    return strprintf("CEvent(Message = %s\n, Type = %d\n, Category = %d\n, What = %s\n, Timestamp = %u)\n",
        Message(), Type(), Category(), What(), Timestamp());
}

static bool ParseDHTMessageKey(const std::string& strAlertMessage, MutableKey& key)
{
    // parse key and salt
    size_t posKeyStart = strAlertMessage.find("key=");
    size_t posSaltBegin = strAlertMessage.find("salt=");
    if (posKeyStart == std::string::npos || posSaltBegin == std::string::npos) {
        LogPrintf("DHTEventListener -- ParseDHTMessageKey Error parsing DHT alert message start.\n");
        return false;
    }
    size_t posKeyEnd = strAlertMessage.find(" ", posKeyStart);
    size_t posSaltEnd = strAlertMessage.find(" ", posSaltBegin);
    if (posKeyEnd == std::string::npos || posSaltEnd == std::string::npos) {
        LogPrintf("DHTEventListener -- ParseDHTMessageKey Error parsing DHT alert message end.\n");
        return false;
    }
    const std::string pubkey = strAlertMessage.substr(posKeyStart + 4, posKeyEnd);
    const std::string salt = strAlertMessage.substr(posSaltBegin + 5, posSaltEnd);
    key = std::make_pair(pubkey, salt);
    
    return true;
}

static std::string GetInfoHash(const std::string pubkey, const std::string salt)
{
    std::array<char, 32> arrPubKey;
    aux::from_hex(pubkey, arrPubKey.data());
    dht::public_key pk;
    pk.bytes = arrPubKey;
    const sha1_hash infoHash = dht::item_target_id(salt, pk);
    return infoHash.to_string();
}

CMutableGetEvent::CMutableGetEvent(alert* alert) : CEvent(alert)
{
    Init();
}

CMutablePutEvent::CMutablePutEvent(alert* alert) : CEvent(alert)
{
    Init();
}

CMutableGetEvent::CMutableGetEvent() : CEvent()
{
}

bool CMutableGetEvent::Init()
{
    if (pAlert != nullptr) {
        dht_mutable_item_alert* dhtGetAlert = alert_cast<dht_mutable_item_alert>(pAlert);
        pubkey = aux::to_hex(dhtGetAlert->key);
        salt = dhtGetAlert->salt;
        authoritative = dhtGetAlert->authoritative;
        value = dhtGetAlert->item.to_string();
        signature = aux::to_hex(dhtGetAlert->signature);
        seq = dhtGetAlert->seq;
        infohash = GetInfoHash(pubkey, salt);
        LogPrintf("CMutableGetEvent::Init -- pubkey = %s, salt = %s, seq = %u.\n", pubkey, salt, seq);
        return true;
    }
    return false;
}

bool CMutablePutEvent::Init()
{
    if (pAlert != nullptr) {
        dht_put_alert* dhtPutAlert = alert_cast<dht_put_alert>(pAlert);
        pubkey = aux::to_hex(dhtPutAlert->public_key);
        salt = dhtPutAlert->salt;
        num_success = dhtPutAlert->num_success;
        seq = dhtPutAlert->seq;
        signature = aux::to_hex(dhtPutAlert->signature);
        infohash = GetInfoHash(pubkey, salt);
        LogPrintf("CMutablePutEvent::Init -- pubkey = %s, salt = %s, seq = %u.\n", pubkey, salt, seq);
        return true;
    }
    return false;
}

void CMutableGetEvent::SetAlert(libtorrent::alert* alert)
{
    if (alert != nullptr) {
        CEvent::SetAlert(alert);
        Init();
    }
}

void CMutablePutEvent::SetAlert(libtorrent::alert* alert)
{
    if (alert != nullptr) {
        CEvent::SetAlert(alert);
        Init();
    }
}

static void AddToDHTGetEventMap(const MutableKey& mKey, const CMutableGetEvent& event) 
{
    LOCK(cs_DHTGetEventMap);
    std::map<MutableKey, CMutableGetEvent>::iterator iMutableEvent = m_DHTGetEventMap.find(mKey);
    if (iMutableEvent == m_DHTGetEventMap.end()) {
        // event not found. Add a new entry to DHT event map
        m_DHTGetEventMap.insert(std::make_pair(mKey, event));
    }
    else {
        // event found. Update entry in DHT event map
        iMutableEvent->second = event;
    }
}

static void AddToDHTPutEventMap(const MutableKey& mKey, const CMutablePutEvent& event) 
{
    LOCK(cs_DHTPutEventMap);
    std::map<MutableKey, CMutablePutEvent>::iterator iMutableEvent = m_DHTPutEventMap.find(mKey);
    if (iMutableEvent == m_DHTPutEventMap.end()) {
        // event not found. Add a new entry to DHT event map
        m_DHTPutEventMap.insert(std::make_pair(mKey, event));
    }
    else {
        // event found. Update entry in DHT event map
        iMutableEvent->second = event;
    }
}

static void AddToEventMap(const uint32_t category, const CEvent& event) 
{
    LOCK(cs_EventMap);
    m_EventCategoryMap.insert(std::make_pair(category, std::make_pair(event.Timestamp(), event)));
}

static void  DHTEventListener(session* dhtSession)
{
    SetThreadPriority(THREAD_PRIORITY_LOWEST);
    RenameThread("dht-events");
    unsigned int counter = 0;
    while(!fShutdown)
    {
        dhtSession->wait_for_alert(seconds(1));
        std::vector<alert*> alerts;
        dhtSession->pop_alerts(&alerts);
        for (std::vector<alert*>::iterator iAlert = alerts.begin(), end(alerts.end()); iAlert != end; ++iAlert) {
            if ((*iAlert) == nullptr)
                continue;

            const uint32_t category = (*iAlert)->category();
            const std::string strAlertMessage = (*iAlert)->message();
            const int iAlertType = (*iAlert)->type();
            if (iAlertType == 75 || iAlertType == 76) {
                LogPrintf("DHTEventListener -- DHT Alert Message = %s, Alert Type =%d\n", strAlertMessage, iAlertType);
                MutableKey mKey; // parse key and salt
                if (!ParseDHTMessageKey(strAlertMessage, mKey)) {
                    LogPrintf("DHTEventListener -- Error parsing DHT alert message start.\n");
                    continue;
                }
                if (iAlertType == 75) {
                    const CMutableGetEvent event((*iAlert));
                    AddToDHTGetEventMap(mKey, event);
                }
                else if (iAlertType == 76) {
                    const CMutablePutEvent event((*iAlert));
                    AddToDHTPutEventMap(mKey, event);
                }
            }
            else {
                const CEvent event((*iAlert));
                AddToEventMap(category, event);
            }
        }
        if (fShutdown)
            return;
        //TODO: (DHT) remove old entries.
        counter++;
    }
}

void StopEventListener()
{
	fShutdown = true;
    LogPrintf("DHTEventListener -- stopping.\n");
    MilliSleep(1100);
    if (pEventListenerThread != nullptr) {
        pEventListenerThread->join();
        pEventListenerThread = nullptr;
    }
}

void StartEventListener(session* dhtSession)
{
    LogPrintf("DHTEventListener -- start\n");
    fShutdown = false;
    if (pEventListenerThread != nullptr)
         StopEventListener();

    pEventListenerThread = std::make_shared<std::thread>(std::bind(&DHTEventListener, std::ref(dhtSession)));
}

bool GetLastCategoryEvents(const uint32_t category, const int64_t& startTime, std::vector<CEvent>& events)
{
    LOCK(cs_EventMap);
    std::multimap<uint32_t, EventPair>::iterator iEvents = m_EventCategoryMap.find(category);
    while (iEvents != m_EventCategoryMap.end()) {
        if (iEvents->second.first >= startTime) {
            events.push_back(iEvents->second.second);
        }
        iEvents++;
    }
    return events.size() > 0;
}

bool FindDHTGetEvent(const MutableKey& mKey, CMutableGetEvent& event)
{
    LOCK(cs_DHTGetEventMap);
    std::map<MutableKey, CMutableGetEvent>::iterator iMutableEvent = m_DHTGetEventMap.find(mKey);
    if (iMutableEvent != m_DHTGetEventMap.end()) {
        // event found.
        event = iMutableEvent->second;
        return true;
    }
    return false;
}

bool FindDHTPutEvent(const MutableKey& mKey, CMutablePutEvent& event)
{
    LOCK(cs_DHTPutEventMap);
    std::map<MutableKey, CMutablePutEvent>::iterator iMutableEvent = m_DHTPutEventMap.find(mKey);
    if (iMutableEvent != m_DHTPutEventMap.end()) {
        // event found.
        event = iMutableEvent->second;
        return true;
    }
    return false;
}