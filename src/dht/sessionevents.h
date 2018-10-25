// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#ifndef DYNAMIC_DHT_SESSION_EVENTS_H
#define DYNAMIC_DHT_SESSION_EVENTS_H

#include <string>
#include <vector>

namespace libtorrent {
    struct session;
    struct alert;
}

typedef std::pair<std::string, std::string> MutableKey; // <pubkey, salt>

class CEvent { 
public:
    libtorrent::alert* pAlert;
    std::int64_t timestamp;

    CEvent() {};
    CEvent(libtorrent::alert* alert);

    bool IsNull() { return pAlert == nullptr; }
    void SetAlert(libtorrent::alert* alert);
    std::string Message() const;
    int Type() const;
    uint32_t Category() const;
    std::string What() const;
    std::int64_t Timestamp() const; // microseconds since session started
    std::string ToString() const;

    inline CEvent operator=(const CEvent& b) {
        SetAlert(b.pAlert);
        return *this;
    }
};

class CMutableDataEvent : public CEvent {
private:   
    std::string pubkey;
    std::string salt;
    std::int64_t seq;

public:
    CMutableDataEvent();
    CMutableDataEvent(libtorrent::alert* alert);

    bool Init();
    void SetAlert(libtorrent::alert* alert);
    std::string PublicKey() const { return pubkey; }
    std::string Salt() const { return salt; }
    std::int64_t SequenceNumber() const { return seq; }
    std::string InfoHash() const;

    inline friend bool operator==(const CMutableDataEvent& a, const CMutableDataEvent& b) {
        return (a.ToString() == b.ToString() && a.PublicKey() == b.PublicKey() && a.Salt() == b.Salt() && a.SequenceNumber() == b.SequenceNumber());
    }

    inline friend bool operator!=(const CMutableDataEvent& a, const CMutableDataEvent& b) {
        return !(a == b);
    }

    inline CMutableDataEvent operator=(const CMutableDataEvent& b) {
        SetAlert(b.pAlert);
        pubkey = b.PublicKey();
        salt = b.Salt();
        seq = b.SequenceNumber();
        return *this;
    }
};

void StopEventListener();
void StartEventListener(libtorrent::session* dhtSession);

bool GetLastCategoryEvents(const uint32_t category, const int64_t& startTime, std::vector<CEvent>& events);
bool FindLastDHTEvent(const MutableKey& mKey, CMutableDataEvent& event);

#endif // DYNAMIC_DHT_SESSION_EVENTS_H
