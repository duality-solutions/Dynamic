// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dht/operations.h"

#include "dht/putbuffer.h"
#include "dht/session.h"
#include "dht/sessionevents.h"
#include "util.h"
#include "utiltime.h"

#include <libtorrent/hex.hpp> // for to_hex
#include <libtorrent/bencode.hpp>
#include <libtorrent/kademlia/item.hpp> // for sign_mutable_item

#include <functional>

using namespace libtorrent;

bool SubmitGetDHTMutableData(const std::array<char, 32>& public_key, const std::string& entrySalt)
{
    //TODO: DHT add locks
    LogPrintf("DHTTorrentNetwork -- GetDHTMutableData started.\n");

    if (!pHashTableSession->Session) {
        //message = "DHTTorrentNetwork -- GetDHTMutableData Error. pHashTableSession->Session is null.";
        return false;
    }

    if (!pHashTableSession->Session->is_dht_running()) {
        LogPrintf("DHTTorrentNetwork -- GetDHTMutableData Restarting DHT.\n");
        if (!LoadSessionState(pHashTableSession->Session)) {
            LogPrintf("DHTTorrentNetwork -- GetDHTMutableData Couldn't load previous settings.  Trying to Bootstrap again.\n");
            if (!Bootstrap())
                return false;
        }
        else {
            LogPrintf("DHTTorrentNetwork -- GetDHTMutableData  setting loaded from file.\n");
        }
    }
    else {
        LogPrintf("DHTTorrentNetwork -- GetDHTMutableData DHT already running.  Bootstrap not needed.\n");
    }

    pHashTableSession->Session->dht_get_item(public_key, entrySalt);
    LogPrintf("DHTTorrentNetwork -- MGET: %s, salt = %s\n", aux::to_hex(public_key), entrySalt);

    return true;
}

bool GetDHTMutableData(const std::array<char, 32>& public_key, const std::string& entrySalt, const int64_t& timeout, 
                            std::string& entryValue, int64_t& lastSequence, bool& fAuthoritative)
{
    if (!SubmitGetDHTMutableData(public_key, entrySalt))
        return false;

    MutableKey mKey = std::make_pair(aux::to_hex(public_key), entrySalt);
    CMutableGetEvent data;
    int64_t startTime = GetTimeMillis();
    while (timeout > GetTimeMillis() - startTime)
    {
        if (FindDHTGetEvent(mKey, data)) {
            std::string strData = data.Value();
            // TODO (DHT): check the last position for the single quote character
            if (strData.substr(0, 1) == "'") {
                entryValue = strData.substr(1, strData.size() - 2);
            }
            else {
                entryValue = strData;
            }
            lastSequence = data.SequenceNumber();
            fAuthoritative = data.Authoritative();
            //LogPrintf("DHTTorrentNetwork -- GetDHTMutableData: value = %s, seq = %d, auth = %u\n", entryValue, lastSequence, fAuthoritative);
            return true;
        }
        MilliSleep(40);
    }
    return false;
}