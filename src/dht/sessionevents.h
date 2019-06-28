// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DHT_SESSION_EVENTS_H
#define DYNAMIC_DHT_SESSION_EVENTS_H

#include "dht/ed25519.h"

#include <string>
#include <vector>

namespace libtorrent {
    class alert;
    class session;
}

class CEvent {
private:
    std::string message;
    int type;
    uint32_t category;
    std::string what;
    std::int64_t timestamp;

public:
    CEvent() {};
    CEvent(const std::string& _message, const int _type, const uint32_t _category, const std::string& _what);

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
    CMutableGetEvent(const std::string& _message, const int _type, const uint32_t _category, const std::string& _what, 
                     const std::string& _pubkey, const std::string& _salt, const int64_t& _seq, const std::string& _value, const std::string& _signature, const bool _authoritative);

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
    CMutablePutEvent(const std::string& _message, const int _type, const uint32_t _category, const std::string& _what, 
                     const std::string& _pubkey, const std::string& _salt, const int64_t& _seq, const std::string& _signature, const uint32_t _success_count);

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
    CPutRequest(const CKeyEd25519& _key, const std::string& _salt, const int64_t& _sequence, const std::string& _value);

    CKeyEd25519 Key() const { return key; }
    std::string Salt() const { return salt; }
    std::int64_t SequenceNumber() const { return sequence; }
    std::string Value() const { return value; }
    std::int64_t Timestamp() const { return timestamp; }
    
    inline CPutRequest operator=(const CPutRequest& b) {
        key = b.Key();
        salt = b.Salt();
        sequence = b.SequenceNumber();
        value = b.Value();
        timestamp = b.Timestamp();
        return *this;
    }
};

std::string GetInfoHash(const std::string& pubkey, const std::string& salt);
std::string GetDynodeHashID(const std::string& service_address);

#endif // DYNAMIC_DHT_SESSION_EVENTS_H
