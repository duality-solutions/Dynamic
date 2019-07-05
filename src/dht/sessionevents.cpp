// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dht/sessionevents.h"

#include "sync.h" // for LOCK and CCriticalSection
#include "util.h" // for LogPrintf
#include "utiltime.h" // for GetTimeMillis

#include <libtorrent/alert.hpp>
#include <libtorrent/alert_types.hpp>
#include <libtorrent/kademlia/types.hpp>
#include <libtorrent/kademlia/item.hpp>
#include <libtorrent/hasher.hpp> // for to_hex and from_hex
#include <libtorrent/hex.hpp> // for to_hex and from_hex
#include <libtorrent/session.hpp>
#include <libtorrent/session_status.hpp>
#include <libtorrent/time.hpp>

using namespace libtorrent;

CEvent::CEvent(const std::string& _message, const int _type, const uint32_t _category, const std::string& _what)
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

std::string GetInfoHash(const std::string& pubkey, const std::string& salt)
{
    std::array<char, 32> arrPubKey;
    aux::from_hex(pubkey, arrPubKey.data());
    dht::public_key pk;
    pk.bytes = arrPubKey;
    const sha1_hash infoHash = dht::item_target_id(salt, pk);
    return aux::to_hex(infoHash.to_string());
}

std::string GetDynodeHashID(const std::string& service_address)
{
    hasher hashNodeID(service_address.c_str(), service_address.size());
    const sha1_hash nodeID = hashNodeID.final();
    LogPrintf("%s -- Dynode Service Address %s,  HashID %s\n", __func__, service_address, aux::to_hex(nodeID.to_string()));
    return aux::to_hex(nodeID.to_string());
}

CMutableGetEvent::CMutableGetEvent() : CEvent()
{
}

CMutableGetEvent::CMutableGetEvent(const std::string& _message, const int _type, const uint32_t _category, const std::string& _what, 
                                   const std::string& _pubkey, const std::string& _salt, const int64_t& _seq, const std::string& _value, const std::string& _signature, const bool _authoritative)
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

CMutablePutEvent::CMutablePutEvent(const std::string& _message, const int _type, const uint32_t _category, const std::string& _what, 
                                   const std::string& _pubkey, const std::string& _salt, const int64_t& _seq, const std::string& _signature, const uint32_t _success_count)
                : CEvent(_message, _type, _category, _what)
{
    pubkey = _pubkey;
    salt = _salt;
    seq = _seq;
    signature = _signature;
    success_count = _success_count;
    infohash = GetInfoHash(pubkey, salt);
}

CPutRequest::CPutRequest(const CKeyEd25519& _key, const std::string& _salt, const int64_t& _sequence, const std::string& _value)
{
    key = _key;
    salt = _salt;
    sequence = _sequence;
    value = _value;
    timestamp = GetTimeMillis();
}