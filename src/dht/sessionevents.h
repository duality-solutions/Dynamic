// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#ifndef DYNAMIC_DHT_SESSION_EVENTS_H
#define DYNAMIC_DHT_SESSION_EVENTS_H

#include <string>

namespace libtorrent {
    struct session;
    struct alert;
}

class CEvent {
    private:
       libtorrent::alert* pAlert;
       std::int64_t timestamp;
       
    public:
       CEvent(libtorrent::alert* alert);

       std::string Message() const;
       int Type() const;
       unsigned int Category() const;
       std::string What() const;
       std::int64_t Timestamp() const; // microseconds since session started
       std::string ToString() const;
};

class CMutableDataEvent : public CEvent {
    private:
        std::string pubkey;
        std::string salt;
        std::int64_t seq;

    public:
       CMutableDataEvent(libtorrent::alert* alert);

       std::string PublicKey() const { return pubkey; }
       std::string Salt() const { return salt; }
       std::int64_t SequenceNumber() const { return seq; }
       std::string InfoHash() const;
};

void StopEventListener();
void StartEventListener(libtorrent::session* dhtSession);

bool FindLastEvent(CEvent& event);
bool FindLastDHTEvent(CMutableDataEvent& event);

#endif // DYNAMIC_DHT_SESSION_EVENTS_H
