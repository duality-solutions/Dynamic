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
    std::int64_t Timestamp() const;
    std::string ToString() const;

    inline CEvent operator=(const CEvent& b) {
        SetAlert(b.pAlert);
        return *this;
    }
};

class CMutableGetEvent : public CEvent {
private:   
    std::string pubkey;
    std::string salt;
    std::int64_t seq;
    std::string value;
    bool authoritative;
    std::string signature;
    std::string infohash;

public:
    CMutableGetEvent();
    CMutableGetEvent(libtorrent::alert* alert);

    bool Init();
    void SetAlert(libtorrent::alert* alert);
    std::string PublicKey() const { return pubkey; }
    std::string Salt() const { return salt; }
    std::int64_t SequenceNumber() const { return seq; }
    std::string InfoHash() const { return infohash; }
    std::string Value() const { return value; }
    bool Authoritative() const { return authoritative; }

    inline friend bool operator==(const CMutableGetEvent& a, const CMutableGetEvent& b) {
        return (a.ToString() == b.ToString() && a.PublicKey() == b.PublicKey() && a.Salt() == b.Salt() && a.SequenceNumber() == b.SequenceNumber());
    }

    inline friend bool operator!=(const CMutableGetEvent& a, const CMutableGetEvent& b) {
        return !(a == b);
    }

    inline CMutableGetEvent operator=(const CMutableGetEvent& b) {
        SetAlert(b.pAlert);
        pubkey = b.PublicKey();
        salt = b.Salt();
        seq = b.SequenceNumber();
        return *this;
    }
};

class CMutablePutEvent : public CEvent {
private:   
    std::string pubkey;
    std::string salt;
    std::int64_t seq;
    std::uint32_t num_success;
    std::string signature;
    std::string infohash;

public:
    CMutablePutEvent();
    CMutablePutEvent(libtorrent::alert* alert);

    bool Init();
    void SetAlert(libtorrent::alert* alert);
    std::string PublicKey() const { return pubkey; }
    std::string Salt() const { return salt; }
    std::int64_t SequenceNumber() const { return seq; }
    std::string InfoHash() const  { return infohash; }
    std::uint32_t SuccessCount() const { return num_success; }

    inline friend bool operator==(const CMutablePutEvent& a, const CMutablePutEvent& b) {
        return (a.ToString() == b.ToString() && a.PublicKey() == b.PublicKey() && a.Salt() == b.Salt() && a.SequenceNumber() == b.SequenceNumber());
    }

    inline friend bool operator!=(const CMutablePutEvent& a, const CMutablePutEvent& b) {
        return !(a == b);
    }

    inline CMutablePutEvent operator=(const CMutablePutEvent& b) {
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
bool FindDHTGetEvent(const MutableKey& mKey, CMutableGetEvent& event);
bool FindDHTPutEvent(const MutableKey& mKey, CMutablePutEvent& event);

#endif // DYNAMIC_DHT_SESSION_EVENTS_H
