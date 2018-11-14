// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#ifndef DYNAMIC_DHT_SESSION_EVENTS_H
#define DYNAMIC_DHT_SESSION_EVENTS_H

#include "dht/ed25519.h"

#include <string>
#include <vector>

namespace libtorrent {
    class session;
    class alert;
}

typedef std::pair<std::string, std::string> MutableKey; // <pubkey, salt>

class CEvent {
private:
    std::string message;
    int type;
    uint32_t category;
    std::string what;
    std::int64_t timestamp;

public:
    CEvent() {};
    CEvent(std::string _message, int _type, uint32_t _category, std::string _what);

    std::string Message() const { return message; }
    int Type() const { return type; }
    uint32_t Category() const { return category; }
    std::string What() const { return what; }
    std::int64_t Timestamp() const { return timestamp; }
    std::string ToString() const;

    inline CEvent operator=(const CEvent& b) {
        message = b.Message();
        type = b.Type();
        category = b.Category();
        what = b.What();
        timestamp = b.Timestamp();
        return *this;
    }
};

class CMutableGetEvent : public CEvent {
private:   
    std::string pubkey;
    std::string salt;
    std::int64_t seq;
    std::string value;
    std::string signature;
    bool authoritative;
    std::string infohash;

public:
    CMutableGetEvent();
    CMutableGetEvent(std::string _message, int _type, uint32_t _category, std::string _what, 
                     std::string _pubkey, std::string _salt, int64_t _seq, std::string _value, std::string _signature, bool _authoritative);

    std::string PublicKey() const { return pubkey; }
    std::string Salt() const { return salt; }
    std::int64_t SequenceNumber() const { return seq; }
    std::string Value() const { return value; }
    std::string Signature() const { return signature; }
    bool Authoritative() const { return authoritative; }
    std::string InfoHash() const { return infohash; }

    inline friend bool operator==(const CMutableGetEvent& a, const CMutableGetEvent& b) {
        return (a.ToString() == b.ToString() && a.PublicKey() == b.PublicKey() && a.Salt() == b.Salt() && a.SequenceNumber() == b.SequenceNumber());
    }

    inline friend bool operator!=(const CMutableGetEvent& a, const CMutableGetEvent& b) {
        return !(a == b);
    }

    inline CMutableGetEvent operator=(const CMutableGetEvent& b) {
        pubkey = b.PublicKey();
        salt = b.Salt();
        seq = b.SequenceNumber();
        value = b.Value();
        authoritative = b.Authoritative();
        signature = b.Signature();
        infohash = b.InfoHash();
        return *this;
    }
};

class CMutablePutEvent : public CEvent {
private:   
    std::string pubkey;
    std::string salt;
    std::int64_t seq;
    std::string signature;
    std::uint32_t success_count;
    std::string infohash;

public:
    CMutablePutEvent();
    CMutablePutEvent(std::string _message, int _type, uint32_t _category, std::string _what, 
                     std::string _pubkey, std::string _salt, int64_t _seq, std::string _signature, uint32_t _success_count);

    std::string PublicKey() const { return pubkey; }
    std::string Salt() const { return salt; }
    std::int64_t SequenceNumber() const { return seq; }
    std::string Signature() const { return signature; }
    std::uint32_t SuccessCount() const { return success_count; }
    std::string InfoHash() const  { return infohash; }

    inline friend bool operator==(const CMutablePutEvent& a, const CMutablePutEvent& b) {
        return (a.ToString() == b.ToString() && a.PublicKey() == b.PublicKey() && a.Salt() == b.Salt() && a.SequenceNumber() == b.SequenceNumber());
    }

    inline friend bool operator!=(const CMutablePutEvent& a, const CMutablePutEvent& b) {
        return !(a == b);
    }

    inline CMutablePutEvent operator=(const CMutablePutEvent& b) {
        pubkey = b.PublicKey();
        salt = b.Salt();
        seq = b.SequenceNumber();
        signature = b.Signature();
        success_count = b.SuccessCount();
        infohash = b.InfoHash();
        return *this;
    }
};

class CPutRequest {
private:
    CKeyEd25519 key;
    std::string salt;
    std::int64_t sequence;
    std::string value;
    std::int64_t timestamp;

public:
    CPutRequest(const CKeyEd25519 _key, const std::string _salt, const int64_t _sequence, const std::string _value);

    CKeyEd25519 Key() const { return key; }
    std::string Salt() const { return salt; }
    std::int64_t SequenceNumber() const { return sequence; }
    std::string Value() const { return value; }
    std::int64_t Timestamp() const { return timestamp; }

    void DHTPut();
    
    inline CPutRequest operator=(const CPutRequest& b) {
        key = b.Key();
        salt = b.Salt();
        sequence = b.SequenceNumber();
        value = b.Value();
        timestamp = b.Timestamp();
        return *this;
    }
};

void CleanUpEventMap(uint32_t timeout = 300000);  //default to 5 minutes.

void StopEventListener();
void StartEventListener(libtorrent::session* dhtSession);

bool GetLastTypeEvent(const int& type, const int64_t& startTime, std::vector<CEvent>& events);
bool FindDHTGetEvent(const MutableKey& mKey, CMutableGetEvent& event);
bool FindDHTPutEvent(const MutableKey& mKey, CMutablePutEvent& event);
bool GetAllDHTPutEvents(std::vector<CMutablePutEvent>& vchPutEvents);
bool GetAllDHTGetEvents(std::vector<CMutableGetEvent>& vchGetEvents);
void AddPutRequest(CPutRequest& put);

#endif // DYNAMIC_DHT_SESSION_EVENTS_H
