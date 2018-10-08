// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#include "dht/sessionevents.h"

#include "util.h" // for LogPrintf

#include "libtorrent/alert.hpp"
#include "libtorrent/alert_types.hpp"
#include "libtorrent/session.hpp"
#include "libtorrent/session_status.hpp"
#include "libtorrent/time.hpp"

#include <map>
#include <mutex>
#include <thread>

using namespace libtorrent;

typedef std::pair<std::string, std::string> MutableKey;
typedef std::map<uint64_t, CEvent> EventMap;
typedef std::map<MutableKey, CMutableDataEvent> DHTEventMap;

static std::mutex mut_EventMap;
static std::mutex mut_DHTEventMap;

static bool fShutdown;
static std::shared_ptr<std::thread> pEventListenerThread;
static EventMap m_EventMap;
static DHTEventMap m_DHTEventMap;

CEvent::CEvent(libtorrent::alert* alert) {
    pAlert = alert;
    timestamp = total_milliseconds(clock_type::now() - pAlert->timestamp());
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

unsigned int CEvent::Category() const 
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
        return 0;
    return strprintf("CEvent(Message = %s\n, Type = %d\n, Category = %d\n, What = %s\n, Timestamp = %u)\n",
        Message(), Type(), Category(), What(), Timestamp());
}

CMutableDataEvent::CMutableDataEvent(alert* alert) : CEvent(alert)
{
    // parse key and salt
    std::string strAlertMessage = Message();
    size_t posKeyStart = strAlertMessage.find("key=");
    size_t posSaltBegin = strAlertMessage.find("salt=");
    if (posKeyStart == std::string::npos || posSaltBegin == std::string::npos) {
        LogPrintf("CMutableDataEvent::CMutableDataEvent -- Error parsing DHT alert message start. %s\n", strAlertMessage);
        return;
    }
    size_t posKeyEnd = strAlertMessage.find(" ", posKeyStart);
    size_t posSaltEnd = strAlertMessage.find(" ", posSaltBegin);
    if (posKeyEnd == std::string::npos || posSaltEnd == std::string::npos) {
        LogPrintf("CMutableDataEvent::CMutableDataEvent -- Error parsing DHT alert message end. %s\n", strAlertMessage);
        return;
    }
    pubkey = strAlertMessage.substr(posKeyStart + 4, posKeyEnd);
    salt = strAlertMessage.substr(posSaltBegin + 5, posSaltEnd);
    LogPrintf("CMutableDataEvent::CMutableDataEvent -- pubkey = %s, salt = %s.\n", pubkey, salt);
    // parse sequence number
    size_t posSeqBegin = strAlertMessage.find("seq=");
    if (posSeqBegin == std::string::npos) {
        LogPrintf("CMutableDataEvent:CMutableDataEvent -- Error parsing DHT alert seq start.s %\n", strAlertMessage);
        return;
    }
    size_t posSeqEnd = strAlertMessage.find(" ", posSeqBegin);
    if (posSeqEnd == std::string::npos) {
        LogPrintf("CMutableDataEvent -- Error parsing DHT alert seq end. %s\n", strAlertMessage);
        return;
    }
    seq = std::stoi(strAlertMessage.substr(posSeqBegin + 4, posSeqEnd));
    LogPrintf("CMutableDataEvent::CMutableDataEvent -- seq = %u.\n", seq);
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
            std::string strAlertMessage = (*iAlert)->message();
            int iAlertType = (*iAlert)->type();
            if ((*iAlert)->category() == 0x400) {
                LogPrintf("DHTEventListener -- DHT Alert Message = %s, Alert Type =%d\n", strAlertMessage, iAlertType);
                // parse key and salt
                size_t posKeyStart = strAlertMessage.find("key=");
                size_t posSaltBegin = strAlertMessage.find("salt=");
                if (posKeyStart == std::string::npos || posSaltBegin == std::string::npos) {
                    LogPrintf("DHTEventListener -- Error parsing DHT alert message start.\n");
                    continue;
                }
                size_t posKeyEnd = strAlertMessage.find(" ", posKeyStart);
                size_t posSaltEnd = strAlertMessage.find(" ", posSaltBegin);
                if (posKeyEnd == std::string::npos || posSaltEnd == std::string::npos) {
                    LogPrintf("DHTEventListener -- Error parsing DHT alert message end.\n");
                    continue;
                }
                const std::string pubkey = strAlertMessage.substr(posKeyStart + 4, posKeyEnd);
                const std::string salt = strAlertMessage.substr(posSaltBegin + 5, posSaltEnd);
                MutableKey mKey(pubkey, salt);
                //CMutableDataEvent findEvent = m_DHTEventMap.find(mKey);
                // TODO: (DHT) Add map entry
                // find existing map for this entry
                //m_DHTEventMap.insert(std::make_pair(event,value));
            }
            else {
                CEvent event((*iAlert));
                m_EventMap.insert(std::make_pair(event.Timestamp(), event));
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

bool FindLastEvent(CEvent& event)
{
    // TODO: (DHT) lookup in m_EventMap
    return false;
}

bool FindLastDHTEvent(CMutableDataEvent& event)
{
    // TODO: (DHT) lookup in m_DHTEventMap
    return false;
}