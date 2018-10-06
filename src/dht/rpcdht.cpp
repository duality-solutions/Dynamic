// Copyright (c) 2018 Duality Blockchain Solutions Developers 
// TODO: Add License

#include "bdap/domainentry.h"
#include "dht/ed25519.h"
#include "dht/mutable.h"
#include "dht/mutabledb.h"
#include "dht/storage.h"
#include "dht/operations.h"
#include "rpcprotocol.h"
#include "rpcserver.h"
#include "util.h"
#include "utilstrencodings.h"

#include <libtorrent/alert_types.hpp>
#include <libtorrent/hex.hpp> // for to_hex and from_hex

#include <univalue.h>

UniValue getmutable(const JSONRPCRequest& request)
{
    if (request.params.size() != 2)
        throw std::runtime_error(
            "getmutable\n"
            "\n");

    UniValue result(UniValue::VOBJ);
    if (!pTorrentDHTSession)
        throw std::runtime_error("getdhtmutable failed. DHT session not started.\n");

    const std::string strPubKey = request.params[0].get_str();
    const std::string strSalt = request.params[1].get_str();

    bool fRet = false;
    int64_t iSequence = 0;
    std::string strValue = "";
    std::array<char, 32> pubKey;
    libtorrent::aux::from_hex(strPubKey, pubKey.data());
    fRet = GetDHTMutableData(pubKey, strSalt, strValue, iSequence, false);
    if (fRet) {
        result.push_back(Pair("Get_PubKey", strPubKey));
        result.push_back(Pair("Get_Salt", strSalt));
        result.push_back(Pair("Get_Seq", iSequence));
        result.push_back(Pair("Get_Value", strValue));
    }
    else {
        throw std::runtime_error("getmutable failed.  Check the debug.log for details.\n");
    }

    return result;
}

UniValue putmutable(const JSONRPCRequest& request)
{
    if (request.params.size() < 2 || request.params.size() > 4 || request.params.size() == 3)
        throw std::runtime_error(
            "putmutable\n"
            "\n");

    UniValue result(UniValue::VOBJ);
    if (!pTorrentDHTSession)
        throw std::runtime_error("putmutable failed. DHT session not started.\n");

    //TODO: Check putValue is not > 1000 bytes.
    bool fNewEntry = false;
    char const* putValue = request.params[0].get_str().c_str();
    const std::string strSalt = request.params[1].get_str();
    std::string strPrivKey;
    std::string strPubKey;
    if (request.params.size() == 4) {
        strPubKey = request.params[2].get_str();
        strPrivKey = request.params[3].get_str();
    }
    else if (request.params.size() == 2) {
        CKeyEd25519 key;
        key.MakeNewKeyPair();
        strPubKey = stringFromVch(key.GetPubKey());
        strPrivKey = stringFromVch(key.GetPrivKey());
        fNewEntry = true;
    }

    bool fRet = false;
    int64_t iSequence = 0;
    std::string strPutValue = "";
    std::array<char, 32> pubKey;
    libtorrent::aux::from_hex(strPubKey, pubKey.data());

    std::array<char, 64> privKey;
    libtorrent::aux::from_hex(strPrivKey, privKey.data());
    if (!fNewEntry) {
        // we need the last sequence number to update an existing DHT entry.
        GetDHTMutableData(pubKey, strSalt, strPutValue, iSequence, true);
        iSequence++;
    }
    std::string dhtMessage = "";
    fRet = PutDHTMutableData(pubKey, privKey, strSalt, iSequence, putValue, dhtMessage);
    if (fRet) {
        std::string dhtMessage = "";
        fRet = PutDHTMutableData(pubKey, privKey, strSalt, iSequence, putValue, dhtMessage);
        if (fRet) {
            result.push_back(Pair("Put_PubKey", strPubKey));
            result.push_back(Pair("Put_PrivKey", strPrivKey));
            result.push_back(Pair("Put_Salt", strSalt));
            result.push_back(Pair("Put_Seq", iSequence));
            result.push_back(Pair("Put_Value", request.params[0].get_str()));
            result.push_back(Pair("Put_Message", dhtMessage));
        }
        else {
            throw std::runtime_error("putdhtmutable failed. Put failed. Check the debug.log for details.\n");
        }
    }
    else {
        throw std::runtime_error("putmutable failed. Put failed. Check the debug.log for details.\n");
    }
    return result;
}

UniValue dhtinfo(const JSONRPCRequest& request)
{
    if (request.params.size() != 0)
        throw std::runtime_error(
            "dhtinfo\n"
            "\n");

    if (!pTorrentDHTSession)
        throw std::runtime_error("dhtinfo failed. DHT session not started.\n");

    libtorrent::session_status stats;
    std::vector<libtorrent::dht_lookup> vchDHTLookup; 
    std::vector<libtorrent::dht_routing_bucket> vchDHTBuckets;
    GetDHTStats(stats, vchDHTLookup, vchDHTBuckets);

    UniValue result(UniValue::VOBJ);
    result.push_back(Pair("num_peers", stats.num_peers));
    result.push_back(Pair("peerlist_size", stats.peerlist_size));
    result.push_back(Pair("active_request_size", (int)stats.active_requests.size()));
    result.push_back(Pair("dht_node_cache", stats.dht_node_cache));
    result.push_back(Pair("dht_global_nodes", stats.dht_global_nodes));
    result.push_back(Pair("dht_download_rate", stats.dht_download_rate));
    result.push_back(Pair("dht_upload_rate", stats.dht_upload_rate));
    result.push_back(Pair("dht_total_allocations", stats.dht_total_allocations));
    result.push_back(Pair("download_rate", stats.download_rate));
    result.push_back(Pair("upload_rate", stats.upload_rate));
    result.push_back(Pair("total_download", stats.total_download));
    result.push_back(Pair("total_upload", stats.total_upload));
    result.push_back(Pair("total_dht_download", stats.total_dht_download));
    result.push_back(Pair("total_dht_upload", stats.total_dht_upload));
    result.push_back(Pair("total_ip_overhead_download", stats.total_ip_overhead_download));
    result.push_back(Pair("total_ip_overhead_upload", stats.total_ip_overhead_upload));
    result.push_back(Pair("total_payload_download", stats.total_payload_download));
    result.push_back(Pair("total_payload_upload", stats.total_payload_upload));
    result.push_back(Pair("dht_nodes", stats.dht_nodes));
    result.push_back(Pair("dht_torrents", stats.dht_torrents));

    for (const libtorrent::dht_routing_bucket& bucket : vchDHTBuckets){
        UniValue oBucket(UniValue::VOBJ);
        oBucket.push_back(Pair("num_nodes", bucket.num_nodes));
        oBucket.push_back(Pair("num_replacements", bucket.num_replacements));
        oBucket.push_back(Pair("last_active", bucket.last_active));
        result.push_back(Pair("bucket", oBucket)); 
    }

    for (const libtorrent::dht_lookup& lookup : vchDHTLookup) {
        UniValue oLookup(UniValue::VOBJ);
        oLookup.push_back(Pair("outstanding_requests", lookup.outstanding_requests));
        oLookup.push_back(Pair("timeouts", lookup.timeouts));
        oLookup.push_back(Pair("responses", lookup.responses));
        oLookup.push_back(Pair("branch_factor", lookup.branch_factor));
        oLookup.push_back(Pair("nodes_left", lookup.nodes_left));
        oLookup.push_back(Pair("last_sent", lookup.last_sent));
        oLookup.push_back(Pair("first_timeout", lookup.first_timeout));
        // string literal indicating which kind of lookup this is
        // char const* type;
        // the node-id or info-hash target for this lookup
        //sha1_hash target;
        result.push_back(oLookup);
        result.push_back(Pair("lookup", oLookup)); 
    }
/*
    result.push_back(Pair("ip_overhead_download_rate", stats.ip_overhead_download_rate));
    result.push_back(Pair("ip_overhead_upload_rate", stats.ip_overhead_upload_rate));
    result.push_back(Pair("payload_download_rate", stats.payload_download_rate));
    result.push_back(Pair("payload_upload_rate", stats.payload_upload_rate));
    result.push_back(Pair("tracker_upload_rate", stats.tracker_upload_rate));
    result.push_back(Pair("tracker_download_rate", stats.tracker_download_rate));
    result.push_back(Pair("total_tracker_download", stats.total_tracker_download));
    result.push_back(Pair("total_tracker_upload", stats.total_tracker_upload));
    result.push_back(Pair("total_redundant_bytes", stats.total_redundant_bytes));
    result.push_back(Pair("total_failed_bytes", stats.total_failed_bytes));
    result.push_back(Pair("num_unchoked", stats.num_unchoked));
    result.push_back(Pair("allowed_upload_slots", stats.allowed_upload_slots));
    result.push_back(Pair("up_bandwidth_queue", stats.up_bandwidth_queue));
    result.push_back(Pair("down_bandwidth_queue", stats.down_bandwidth_queue));
    result.push_back(Pair("up_bandwidth_bytes_queue", stats.up_bandwidth_bytes_queue));
    result.push_back(Pair("down_bandwidth_bytes_queue", stats.down_bandwidth_bytes_queue));
    result.push_back(Pair("optimistic_unchoke_counter", stats.optimistic_unchoke_counter));
    result.push_back(Pair("unchoke_counter", stats.unchoke_counter));
    result.push_back(Pair("has_incoming_connections", stats.has_incoming_connections));
*/
    return result;
}

UniValue dhtdb(const JSONRPCRequest& request)
{
    if (request.params.size() != 0)
        throw std::runtime_error(
            "dhtdb\n"
            "\n");

    UniValue result(UniValue::VOBJ);

    std::vector<CMutableData> vchMutableData;
    
    bool fRet = GetAllLocalMutableData(vchMutableData);
    int nCounter = 0;
    if (fRet) {
        for(const CMutableData& data : vchMutableData) {
            UniValue oMutableData(UniValue::VOBJ);
            oMutableData.push_back(Pair("info_hash", data.InfoHash()));
            oMutableData.push_back(Pair("public_key", data.PublicKey()));
            oMutableData.push_back(Pair("signature", data.Signature()));
            oMutableData.push_back(Pair("seq_num", data.SequenceNumber));
            oMutableData.push_back(Pair("salt", data.Salt()));
            oMutableData.push_back(Pair("value", libtorrent::dht::ExtractPutValue(data.Value())));
            result.push_back(Pair("dht_entry_" + std::to_string(nCounter + 1), oMutableData));
            nCounter++;
        }
    }
    else {
        throw std::runtime_error("dhtdb failed.  Check the debug.log for details.\n");
    }
    UniValue oCounter(UniValue::VOBJ);
    oCounter.push_back(Pair("record_count", nCounter));
    result.push_back(Pair("summary", oCounter));
    return result;
}

static const CRPCCommand commands[] =
{   //  category         name                        actor (function)           okSafeMode
    /* DHT */
    { "dht",             "getmutable",               &getmutable,                   true  },
    { "dht",             "putmutable",               &putmutable,                   true  },
    { "dht",             "dhtinfo",                  &dhtinfo,                      true  },
    { "dht",             "dhtdb",                    &dhtdb,                        true  },
};

void RegisterDHTRPCCommands(CRPCTable &tableRPC)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        tableRPC.appendCommand(commands[vcidx].name, &commands[vcidx]);
}