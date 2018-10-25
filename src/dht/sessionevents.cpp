// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#include "dht/sessionevents.h"

#include "sync.h" // for LOCK and CCriticalSection
#include "util.h" // for LogPrintf

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
typedef std::map<MutableKey, CMutableDataEvent> DHTEventMap;

static CCriticalSection cs_EventMap;
static CCriticalSection cs_DHTEventMap;

static bool fShutdown;
static std::shared_ptr<std::thread> pEventListenerThread;
static EventCategoryMap m_EventCategoryMap;
static DHTEventMap m_DHTEventMap;

CEvent::CEvent(libtorrent::alert* alert) {
    SetAlert(alert);
}

void CEvent::SetAlert(libtorrent::alert* alert)
{
    if (alert != nullptr) {
        pAlert = alert;
        timestamp = total_milliseconds(clock_type::now() - pAlert->timestamp());
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
        LogPrintf("DHTEventListener -- ParseDHTMessage Error parsing DHT alert message start.\n");
        return false;
    }
    size_t posKeyEnd = strAlertMessage.find(" ", posKeyStart);
    size_t posSaltEnd = strAlertMessage.find(" ", posSaltBegin);
    if (posKeyEnd == std::string::npos || posSaltEnd == std::string::npos) {
        LogPrintf("DHTEventListener -- ParseDHTMessage Error parsing DHT alert message end.\n");
        return false;
    }
    const std::string pubkey = strAlertMessage.substr(posKeyStart + 4, posKeyEnd);
    const std::string salt = strAlertMessage.substr(posSaltBegin + 5, posSaltEnd);
    key = std::make_pair(pubkey, salt);
    
    return true;
}

static bool ParseDHTMessage(const std::string& strAlertMessage, MutableKey& key, std::int64_t& sequence)
{
    if (!ParseDHTMessageKey(strAlertMessage, key)) {
        return false;
    }
    // parse sequence number
    size_t posSeqBegin = strAlertMessage.find("seq=");
    if (posSeqBegin == std::string::npos) {
        LogPrintf("DHTEventListener -- ParseDHTMessage Error parsing DHT alert seq start.s %\n", strAlertMessage);
        sequence = -1;
        return true;
    }
    size_t posSeqEnd = strAlertMessage.find(" ", posSeqBegin);
    if (posSeqEnd == std::string::npos) {
        LogPrintf("DHTEventListener -- ParseDHTMessage Error parsing DHT alert seq end. %s\n", strAlertMessage);
        sequence = -1;
        return true;
    }
    sequence = std::stoi(strAlertMessage.substr(posSeqBegin + 4, posSeqEnd));
    return true;
}

CMutableDataEvent::CMutableDataEvent(alert* alert) : CEvent(alert)
{
    Init();
}

CMutableDataEvent::CMutableDataEvent() : CEvent()
{
}

bool CMutableDataEvent::Init()
{
    if (pAlert != nullptr) {
        MutableKey key;
        ParseDHTMessage(Message(), key, seq); // parse key, salt and seq
        pubkey = key.first;
        salt = key.second;
        LogPrintf("CMutableDataEvent::Init -- pubkey = %s, salt = %s, seq = %u.\n", pubkey, salt, seq);
        return true;
    }
    return false;
}

void CMutableDataEvent::SetAlert(libtorrent::alert* alert)
{
    if (alert != nullptr) {
        CEvent::SetAlert(alert);
        Init();
    }
}

std::string CMutableDataEvent::InfoHash() const
{
    std::array<char, 32> arrPubKey;
    aux::from_hex(pubkey, arrPubKey.data());
    dht::public_key pk;
    pk.bytes = arrPubKey;
    const sha1_hash infoHash = dht::item_target_id(salt, pk);
    return infoHash.to_string();
}

static void AddToDHTEventMap(const MutableKey& mKey, const CMutableDataEvent& event) 
{
    LOCK(cs_DHTEventMap);
    std::map<MutableKey, CMutableDataEvent>::iterator iMutableEvent = m_DHTEventMap.find(mKey);
    if (iMutableEvent == m_DHTEventMap.end()) {
        // event not found. Add a new entry to DHT event map
        m_DHTEventMap.insert(std::make_pair(mKey, event));
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
            if (category == 0x400) {
                LogPrintf("DHTEventListener -- DHT Alert Message = %s, Alert Type =%d\n", strAlertMessage, iAlertType);
                MutableKey mKey; // parse key and salt
                if (!ParseDHTMessageKey(strAlertMessage, mKey)) {
                    LogPrintf("DHTEventListener -- Error parsing DHT alert message start.\n");
                    continue;
                }
                const CMutableDataEvent event((*iAlert));
                AddToDHTEventMap(mKey, event);
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

bool FindLastDHTEvent(const MutableKey& mKey, CMutableDataEvent& event)
{
    LOCK(cs_DHTEventMap);
    std::map<MutableKey, CMutableDataEvent>::iterator iMutableEvent = m_DHTEventMap.find(mKey);
    if (iMutableEvent != m_DHTEventMap.end()) {
        // event found.
        event = iMutableEvent->second;
        return true;
    }
    return false;
}