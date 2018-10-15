// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#include "storage.h"

#include "bdap/domainentry.h"
#include "dht/mutable.h"
#include "dht/mutabledb.h"
#include "util.h"

#include <libtorrent/aux_/numeric_cast.hpp>
#include <libtorrent/broadcast_socket.hpp> // for ip_v4

#include <libtorrent/config.hpp>
#include <libtorrent/hex.hpp>
#include <libtorrent/random.hpp>
#include <libtorrent/packet_buffer.hpp> 
#include <libtorrent/socket_io.hpp>

#include <array>
#include <string>

using namespace libtorrent;
using namespace libtorrent::dht;

size_t CDHTStorage::num_torrents() const 
{ 
    LogPrintf("********** CDHTStorage -- num_torrents **********\n");
    return pDefaultStorage->num_torrents(); 
}

size_t CDHTStorage::num_peers() const
{
    LogPrintf("********** CDHTStorage -- num_peers **********\n");
    return pDefaultStorage->num_peers(); 
}

void CDHTStorage::update_node_ids(std::vector<libtorrent::sha1_hash> const& ids)
{
    LogPrintf("********** CDHTStorage -- update_node_ids **********\n");
    pDefaultStorage->update_node_ids(ids);
}

bool CDHTStorage::get_peers(sha1_hash const& info_hash, bool const noseed, bool const scrape, address const& requester, entry& peers) const
{
    bool ret = pDefaultStorage->get_peers(info_hash, noseed, scrape, requester, peers);
    //LogPrintf("********** CDHTStorage -- get_peers peers = %s **********\n", peers.to_string());
    return ret;
}

void CDHTStorage::announce_peer(sha1_hash const& info_hash, tcp::endpoint const& endp, string_view name, bool const seed)
{
    LogPrintf("********** CDHTStorage -- announce_peer **********\n");
    pDefaultStorage->announce_peer(info_hash, endp, name, seed);
}

bool CDHTStorage::get_immutable_item(sha1_hash const& target, entry& item) const
{
    return false;
}

void CDHTStorage::put_immutable_item(sha1_hash const& target, span<char const> buf, address const& addr)
{
    //TODO: ban nodes that try to put immutable entries.
    //pDefaultStorage->put_immutable_item(target, buf, addr);
    // Do we need immutable item support ???
    LogPrintf("********** CDHTStorage -- put_immutable_item target = %s, buf = %s, addr = %s\n", aux::to_hex(target.to_string()), std::string(buf.data()), addr.to_string());
}

bool CDHTStorage::get_mutable_item_seq(sha1_hash const& target, sequence_number& seq) const
{
    CMutableData mutableData;
    std::string strInfoHash = aux::to_hex(target.to_string());
    CharString vchInfoHash = vchFromString(strInfoHash);
    LogPrintf("********** CDHTStorage -- get_mutable_item_seq infohash = %s\n", strInfoHash);
    if (!GetLocalMutableData(vchInfoHash, mutableData)) {
        LogPrintf("********** CDHTStorage -- get_mutable_item_seq failed to get mutable entry sequence_number for infohash = %s.\n", strInfoHash);
        return false;
    }
    seq = dht::sequence_number(mutableData.SequenceNumber);
    LogPrintf("********** CDHTStorage -- get_mutable_item_seq found seq = %u\n", mutableData.SequenceNumber);
    return true;
}

bool CDHTStorage::get_mutable_item(sha1_hash const& target, sequence_number const seq, bool const force_fill, entry& item) const
{
    CMutableData mutableData;
    std::string strInfoHash = aux::to_hex(target.to_string());
    CharString vchInfoHash = vchFromString(strInfoHash);
    if (!GetLocalMutableData(vchInfoHash, mutableData)) {
        LogPrintf("********** CDHTStorage -- get_mutable_item failed to get mutable entry for infohash = %s.\n", strInfoHash);
        return false;
    }
    item["seq"] = mutableData.SequenceNumber;
    if (force_fill || (sequence_number(0) <= seq && seq < sequence_number(mutableData.SequenceNumber)))
    {
        LogPrintf("********** CDHTStorage -- get_mutable_item data found.\n");
        item["v"] = bdecode(mutableData.vchValue.begin(), mutableData.vchValue.end());
        std::array<char, 64> sig;
        aux::from_hex(mutableData.Signature(), sig.data());
        item["sig"] = sig;
        std::array<char, 32> pubKey;
        aux::from_hex(mutableData.PublicKey(), pubKey.data());
        item["k"] = pubKey;
    }
    LogPrintf("********** CDHTStorage -- get_mutable_item target = %s, item = %s\n", aux::to_hex(target.to_string()), item.to_string());
    return true;
}

void CDHTStorage::put_mutable_item(sha1_hash const& target
    , span<char const> buf
    , signature const& sig
    , sequence_number const seq
    , public_key const& pk
    , span<char const> salt
    , address const& addr)
{
    std::string strInfoHash = aux::to_hex(target.to_string());
    CharString vchInfoHash = vchFromString(strInfoHash);
    std::string strPutValue = std::string(buf.data());
    CharString vchPutValue = vchFromString(strPutValue);
    std::string strSignature = aux::to_hex(std::string(sig.bytes.data()));
    CharString vchSignature = vchFromString(strSignature);
    std::string strPublicKey = aux::to_hex(std::string(pk.bytes.data()));
    CharString vchPublicKey = vchFromString(strPublicKey);
    std::string strSalt = ExtractSalt(std::string(salt.data()));
    CharString vchSalt = vchFromString(strSalt);
    CMutableData putMutableData(vchInfoHash, vchPublicKey, vchSignature, seq.value, vchSalt, vchPutValue);
    LogPrintf("********** CDHTStorage -- put_mutable_item info_hash = %s, buf_value = %s, sig = %s, pubkey = %s, salt = %s, seq = %d \n", 
                    strInfoHash, ExtractPutValue(strPutValue), strSignature, strPublicKey, strSalt, putMutableData.SequenceNumber);

    CMutableData previousData;
    if (!GetLocalMutableData(vchInfoHash, previousData)) {
        if (PutLocalMutableData(vchInfoHash, putMutableData)) {
            LogPrintf("********** CDHTStorage -- put_mutable_item added successfully**********\n");
        }
    }
    else {
        if (putMutableData.Value() != previousData.Value() || putMutableData.SequenceNumber != previousData.SequenceNumber) {
            if (UpdateLocalMutableData(vchInfoHash, putMutableData)) {
                LogPrintf("********** CDHTStorage -- put_mutable_item updated successfully**********\n");
            }
        }
        else {
            LogPrintf("********** CDHTStorage -- put_mutable_item value unchanged. No database operation needed. **********\n");
        }
    }
    // TODO: Log from address (addr). See touch_item in the default storage implementation.
    return;
    
}

int CDHTStorage::get_infohashes_sample(entry& item)
{
    LogPrintf("********** CDHTStorage -- get_infohashes_sample **********\n");
    return pDefaultStorage->get_infohashes_sample(item);
}

void CDHTStorage::tick()
{
    LogPrintf("********** CDHTStorage -- tick **********\n");
    pDefaultStorage->tick();
}

dht_storage_counters CDHTStorage::counters() const
{
    LogPrintf("********** CDHTStorage -- counters **********\n");
    return pDefaultStorage->counters();
}

std::unique_ptr<dht_storage_interface> CDHTStorageConstructor(dht_settings const& settings)
{
    return std::unique_ptr<CDHTStorage>(new CDHTStorage(settings));
}

std::string ExtractPutValue(std::string value)
{
    std::string strReturn = "";
    size_t posStart = value.find(":") + 1;
    size_t posEnd = 0;
    if (!(posStart == std::string::npos)) {
        posEnd = value.find("e1:q3:put1:t");
        if (!(posEnd == std::string::npos) && value.size() > posEnd) {
            strReturn = value.substr(posStart, posEnd - posStart);
        }
    }
    return strReturn;
}

std::string ExtractSalt(std::string salt)
{
    std::string strReturn = "";
    size_t posEnd = salt.find("3:seqi");
    if (!(posEnd == std::string::npos) && salt.size() > posEnd) {
        strReturn = salt.substr(0, posEnd);
    }
    //LogPrintf("********** CDHTStorage -- ExtractSalt salt in = %, salt out = %s\n", salt, strReturn);
    return strReturn;
}