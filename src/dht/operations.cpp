// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#include "dht/operations.h"

#include "dht/session.h"
#include "util.h"


#include <libtorrent/hex.hpp> // for to_hex
#include <libtorrent/bencode.hpp>
#include "libtorrent/kademlia/item.hpp" // for sign_mutable_item

using namespace libtorrent;

bool GetDHTMutableData(const std::array<char, 32>& public_key, const std::string& entrySalt, std::string& entryValue, int64_t& lastSequence, bool fWaitForAuthoritative)
{
    //TODO: DHT add locks
    LogPrintf("DHTTorrentNetwork -- GetDHTMutableData started.\n");

    if (!pTorrentDHTSession) {
        //message = "DHTTorrentNetwork -- GetDHTMutableData Error. pTorrentDHTSession is null.";
        return false;
    }

    if (!pTorrentDHTSession->is_dht_running()) {
        LogPrintf("DHTTorrentNetwork -- GetDHTMutableData Restarting DHT.\n");
        if (!LoadSessionState(pTorrentDHTSession)) {
            LogPrintf("DHTTorrentNetwork -- GetDHTMutableData Couldn't load previous settings.  Trying to Bootstrap again.\n");
            Bootstrap(pTorrentDHTSession);
        }
        else {
            LogPrintf("DHTTorrentNetwork -- GetDHTMutableData  setting loaded from file.\n");
        }
    }
    else {
        LogPrintf("DHTTorrentNetwork -- GetDHTMutableData DHT already running.  Bootstrap not needed.\n");
    }

    pTorrentDHTSession->dht_get_item(public_key, entrySalt);
    LogPrintf("DHTTorrentNetwork -- MGET: %s, salt = %s\n", aux::to_hex(public_key), entrySalt);

    bool authoritative = false;
    if (fWaitForAuthoritative) {
        while (!authoritative)
        {
            alert* dhtAlert = WaitForResponse(pTorrentDHTSession, dht_mutable_item_alert::alert_type, public_key, entrySalt);

            dht_mutable_item_alert* dhtGetAlert = alert_cast<dht_mutable_item_alert>(dhtAlert);
            authoritative = dhtGetAlert->authoritative;
            entryValue = dhtGetAlert->item.to_string();
            lastSequence = dhtGetAlert->seq;
            LogPrintf("DHTTorrentNetwork -- GetDHTMutableData %s: %s\n", authoritative ? "auth" : "non-auth", entryValue);
        }
    }
    else {
        alert* dhtAlert = WaitForResponse(pTorrentDHTSession, dht_mutable_item_alert::alert_type, public_key, entrySalt);
        dht_mutable_item_alert* dhtGetAlert = alert_cast<dht_mutable_item_alert>(dhtAlert);
        authoritative = dhtGetAlert->authoritative;
        entryValue = dhtGetAlert->item.to_string();
        lastSequence = dhtGetAlert->seq;
        LogPrintf("DHTTorrentNetwork -- GetDHTMutableData %s: %s\n", authoritative ? "auth" : "non-auth", entryValue);
    }

    if (entryValue == "<uninitialized>")
        return false;

    return true;
}

static void put_mutable
(
    entry& e
    ,std::array<char, 64>& sig
    ,std::int64_t& seq
    ,std::string const& salt
    ,std::array<char, 32> const& pk
    ,std::array<char, 64> const& sk
    ,char const* str
    ,std::int64_t const& iSeq
)
{
    using dht::sign_mutable_item;
    if (str != NULL) {
        e = std::string(str);
        std::vector<char> buf;
        bencode(std::back_inserter(buf), e);
        dht::signature sign;
        seq = iSeq + 1;
        sign = sign_mutable_item(buf, salt, dht::sequence_number(seq)
            , dht::public_key(pk.data())
            , dht::secret_key(sk.data()));
        sig = sign.bytes;
    }
}

bool PutDHTMutableData(const std::array<char, 32>& public_key, const std::array<char, 64>& private_key, const std::string& entrySalt, const int64_t& lastSequence
                        ,char const* dhtValue, std::string& message)
{
    //TODO: (DHT) add locks
    LogPrintf("DHTTorrentNetwork -- PutMutableData started.\n");

    if (!pTorrentDHTSession) {
        message = "DHTTorrentNetwork -- PutDHTMutableData Error. pTorrentDHTSession is null.";
        return false;
    }

    if (!pTorrentDHTSession->is_dht_running()) {
        LogPrintf("DHTTorrentNetwork -- PutDHTMutableData Restarting DHT.\n");
        if (!LoadSessionState(pTorrentDHTSession)) {
            LogPrintf("DHTTorrentNetwork -- PutDHTMutableData Couldn't load previous settings.  Trying to Bootstrap again.\n");
            Bootstrap(pTorrentDHTSession);
        }
        else {
            LogPrintf("DHTTorrentNetwork -- PutDHTMutableData  setting loaded from file.\n");
        }
    }
    else {
        LogPrintf("DHTTorrentNetwork -- PutDHTMutableData DHT already running.  Bootstrap not needed.\n");
    }
    
    pTorrentDHTSession->dht_put_item(public_key, std::bind(&put_mutable, std::placeholders::_1, std::placeholders::_2, 
                                        std::placeholders::_3, std::placeholders::_4, public_key, private_key, dhtValue, lastSequence), entrySalt);

    LogPrintf("DHTTorrentNetwork -- MPUT public key: %s, salt = %s, seq=%d\n", aux::to_hex(public_key), entrySalt, lastSequence);
    alert* dhtAlert = WaitForResponse(pTorrentDHTSession, dht_put_alert::alert_type, public_key, entrySalt);
    dht_put_alert* dhtPutAlert = alert_cast<dht_put_alert>(dhtAlert);
    message = dhtPutAlert->message();
    LogPrintf("DHTTorrentNetwork -- PutMutableData %s\n", message);

    if (dhtPutAlert->num_success == 0)
        return false;

    return true;
}