// Copyright (c) 2018 Duality Blockchain Solutions Developers 
// TODO: Add License

#include "bdap/domainentry.h"
#include "bdap/domainentrydb.h"
#include "bdap/utils.h"
#include "dht/ed25519.h"
#include "dht/mutable.h"
#include "dht/mutabledb.h"
#include "dht/storage.h"
#include "dht/operations.h"
#include "dht/sessionevents.h"
#include "hash.h"
#include "pubkey.h"
#include "rpcprotocol.h"
#include "rpcserver.h"
#include "util.h"
#include "utilstrencodings.h"
#include "wallet/wallet.h"

#include <libtorrent/alert_types.hpp>
#include <libtorrent/hex.hpp> // for to_hex and from_hex

#include <univalue.h>

UniValue getmutable(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 2)
        throw std::runtime_error(
            "getmutable \"pubkey\" \"operation\"\n"
            "\nArguments:\n"
            "1. pubkey             (string)             DHT public key\n"
            "2. operation          (string)             DHT data operation\n"
            "\nGets mutable data from the DHT.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"Public Key\"                 (string)  Mutable entry public key\n"
            "  \"Salt\"                       (string)  Mutable entry salt\n"
            "  \"Sequence Number\"            (int)     Mutable entry sequence Number\n"
            "  \"Authoritative\"              (bool)    Response authoritative\n"
            "  \"DHT Entry Value\"            (string)  Mutable entry value\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("getmutable", "517c4242c95214e5eb631e1ddf4e7dac5e815f0578f88491b81fd36df3c2a16a avatar") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("getmutable", "517c4242c95214e5eb631e1ddf4e7dac5e815f0578f88491b81fd36df3c2a16a avatar"));

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
    bool fAuthoritative;
    fRet = GetDHTMutableData(pubKey, strSalt, 5000, strValue, iSequence, fAuthoritative);
    if (fRet) {
        result.push_back(Pair("Public Key", strPubKey));
        result.push_back(Pair("Salt", strSalt));
        result.push_back(Pair("Sequence Number", iSequence));
        result.push_back(Pair("Authoritative", fAuthoritative ? "True" : "False"));
        result.push_back(Pair("DHT Entry Value", strValue));
    }
    else {
        throw std::runtime_error("getmutable failed.  Check the debug.log for details.\n");
    }

    return result;
}

UniValue putmutable(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() < 2 || request.params.size() > 4 || request.params.size() == 3)
        throw std::runtime_error(
            "putmutable \"data value\" \"operation\" \"pubkey\" \"privkey\"\n"
            "\nArguments:\n"
            "1. data value                    (string)  Put data value for the DHT entry\n"
            "2. operation                     (string)  DHT data operation\n"
            "3. pubkey                        (string)  Base64 encoded DHT public key\n"
            "4. privkey                       (string)  Base64 encoded DHT private key\n"
            "\nPuts or saves mutable data on the DHT for a pubkey/salt pair\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"Public Key\"                 (string)  Mutable entry public key\n"
            "  \"Salt\"                       (string)  Mutable entry salt\n"
            "  \"Sequence Number\"            (int)     Mutable entry sequence Number\n"
            "  \"Authoritative\"              (bool)    Response authoritative\n"
            "  \"DHT Entry Value\"            (string)  Mutable entry value\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("putmutable", "\"https://duality.solutions/duality/logos/dual.png\" avatar 517c4242c95214e5eb631e1ddf4e7dac5e815f0578f88491b81fd36df3c2a16a bf8b4f66bdd9e7dc526ddc3637a4edf8e0ac86b7df5e249fc6514a0a1c047cd0") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("getmutable", "\"https://duality.solutions/duality/logos/dual.png\" avatar 517c4242c95214e5eb631e1ddf4e7dac5e815f0578f88491b81fd36df3c2a16a bf8b4f66bdd9e7dc526ddc3637a4edf8e0ac86b7df5e249fc6514a0a1c047cd0"));

    UniValue result(UniValue::VOBJ);
    if (!pTorrentDHTSession)
        throw std::runtime_error("putmutable failed. DHT session not started.\n");

    //TODO: Check putValue is not > 1000 bytes.
    bool fNewEntry = false;
    const std::string strValue = request.params[1].get_str();
    const std::string strOperationType = request.params[1].get_str();
    std::string strPrivKey;
    std::string strPubKey;
    
    if (request.params.size() == 4) {
        strPubKey = request.params[2].get_str();
        strPrivKey = request.params[3].get_str();
    }
    else if (request.params.size() == 2) {
        CKeyEd25519 newkey;
        strPubKey = stringFromVch(newkey.GetPubKey());
        strPrivKey = stringFromVch(newkey.GetPrivKey());
        fNewEntry = true;
    }

    int64_t iSequence = 0;
    std::string strPutValue = "";
    std::array<char, 32> pubKey;
    libtorrent::aux::from_hex(strPubKey, pubKey.data());
    std::array<char, 64> privKey;
    libtorrent::aux::from_hex(strPrivKey, privKey.data());
    
    CKeyEd25519 key;
    std::vector<unsigned char> vch = vchFromString(strPubKey);
    CKeyID keyID(Hash160(vch.begin(), vch.end()));
    if (pwalletMain && !pwalletMain->GetDHTKey(keyID, key)) {
        throw std::runtime_error("putmutable: ERRCODE: 5400 - Error getting ed25519 private key for " + strPubKey + "\n");
    }

    if (!fNewEntry) {
        // we need the last sequence number to update an existing DHT entry.
        bool fAuthoritative;
        GetDHTMutableData(pubKey, strOperationType, 2000, strPutValue, iSequence, fAuthoritative);
        iSequence++;
    }
    CPutRequest put(key, strOperationType, iSequence, strValue);
    AddPutRequest(put);

    result.push_back(Pair("Public Key", strPubKey));
    //result.push_back(Pair("Put_PrivKey", strPrivKey));
    result.push_back(Pair("Salt", strOperationType));
    result.push_back(Pair("Sequence Number", iSequence));
    result.push_back(Pair("DHT Entry Value", request.params[0].get_str()));
    
    return result;
}

UniValue dhtinfo(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 0)
        throw std::runtime_error(
            "dhtinfo"
            "\nGets DHT network stats and info.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"num_peers\"                     (int)      Number of torrent peers\n"
            "  \"peerlist_size\"                 (int)      Torrent peer list size\n"
            "  \"active_request_size\"           (int)      Active request size\n"
            "  \"dht_node_cache\"                (int)      DHT node cache\n"
            "  \"dht_global_nodes\"              (int)      DHT global nodes\n"
            "  \"dht_download_rate\"             (int)      DHT download rate\n"
            "  \"dht_upload_rate\"               (int)      DHT upload rate\n"
            "  \"dht_total_allocations\"         (int)      DHT total allocations\n"
            "  \"download_rate\"                 (decimal)  Torrent download rate\n"
            "  \"upload_rate\"                   (decimal)  Torrent upload rate\n"
            "  \"total_download\"                (int)      Total torrent downloads\n"
            "  \"total_upload\"                  (int)      Total torrent uploads\n"
            "  \"total_dht_download\"            (int)      Total DHT downloads\n"
            "  \"total_dht_upload\"              (int)      Total DHT uploads\n"
            "  \"total_ip_overhead_download\"    (int)      Total torrent IP overhead for downloads\n"
            "  \"total_ip_overhead_upload\"      (int)      Total torrent IP overhead for uploads\n"
            "  \"total_payload_download\"        (int)      Total torrent payload for downloads\n"
            "  \"total_payload_upload\"          (int)      Total torrent payload for uploads\n"
            "  \"dht_nodes\"                     (int)      Number of DHT nodes\n"
            "  {(dht_bucket)\n"
            "    \"num_nodes\"                   (int)      Number of nodes in DHT bucket\n"
            "    \"num_replacements\"            (int)      Number of replacements in DHT bucket\n"
            "    \"last_active\"                 (int)      DHT bucket last active\n"
            "  }\n"
            "  {(dht_lookup)\n"
            "    \"outstanding_requests\"        (int)      DHT lookup outstanding requests\n"
            "    \"timeouts\"                    (int)      DHT lookup timeouts\n"
            "    \"responses\"                   (int)      DHT lookup responses\n"
            "    \"branch_factor\"               (int)      DHT lookup branch factor\n"
            "    \"nodes_left\"                  (int)      DHT lookup nodes left\n"
            "    \"last_sent\"                   (int)      DHT lookup last sent\n"
            "    \"first_timeout\"               (int)      DHT lookup first timeouts\n"
            "  }\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("dhtinfo", "") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("dhtinfo", ""));

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
    if (request.fHelp || request.params.size() != 0)
        throw std::runtime_error(
            "dhtdb"
            "\nGets the local DHT cache database contents.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"info_hash\"                  (string)      Mutable data info hash\n"
            "  \"public_key\"                 (string)      Mutable data public key\n"
            "  \"signature\"                  (string)      Mutable data entry signature\n"
            "  \"seq_num\"                    (int)         Mutable data sequsence number\n"
            "  \"salt\"                       (string)      Mutable data entry salt or operation code\n"
            "  \"value\"                      (string)      Mutable data entry value\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("dhtdb", "") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("dhtdb", ""));

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
            oMutableData.push_back(Pair("value", data.Value()));
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

UniValue putbdapdata(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 3)
        throw std::runtime_error(
            "putbdapdata \"account id\" \"data value\" \"operation\"\n"
            "\nSaves mutable data in the DHT for a BDAP entry and operation code.\n"
            "\nArguments:\n"
            "1. account id             (string)      BDAP account id\n"
            "2. data value             (string)      Mutable data value to save in the DHT\n"
            "3. operation              (string)      Mutable data operation used for DHT entry\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"entry_path\"          (string)      BDAP account FQDN\n"
            "  \"wallet_address\"      (string)      BDAP account wallet address\n"
            "  \"link_address\"        (string)      BDAP account link address\n"
            "  \"put_pubkey\"          (string)      BDAP account DHT public key\n"
            "  \"put_operation\"       (string)      Mutable data put operation or salt\n"
            "  \"put_seq\"             (string)      Mutable data sequence number\n"
            "  \"put_value\"           (string)      Mutable data entry value\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("putbdapdata", "Duality \"https://duality.solutions/duality/graphics/header/bdap.png\" avatar") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("putbdapdata", "Duality \"https://duality.solutions/duality/graphics/header/bdap.png\" avatar"));

    UniValue result(UniValue::VOBJ);
   
    EnsureWalletIsUnlocked();

    if (!pTorrentDHTSession)
        throw std::runtime_error("putbdapdata: ERRCODE: 5500 - DHT session not started.\n");

    if (!CheckDomainEntryDB())
        throw std::runtime_error("putbdapdata: ERRCODE: 5501 - Can not access BDAP domain entry database.\n");

    CharString vchObjectID = vchFromValue(request.params[0]);
    const std::string strValue = request.params[1].get_str();
    const std::string strOperationType = request.params[2].get_str();

    ToLowerCase(vchObjectID);
    CDomainEntry entry;
    entry.DomainComponent = vchDefaultDomainName;
    entry.OrganizationalUnit = vchDefaultPublicOU;
    entry.ObjectID = vchObjectID;
    std::string strFullObjectPath = entry.GetFullObjectPath();
    if (!pDomainEntryDB->GetDomainEntryInfo(entry.vchFullObjectPath(), entry))
        throw std::runtime_error("putbdapdata: ERRCODE: 5502 - " + strFullObjectPath + _(" can not be found.  Get BDAP info failed!\n"));

    CKeyEd25519 getKey;
    std::vector<unsigned char> vch = entry.DHTPublicKey;
    CKeyID keyID(Hash160(vch.begin(), vch.end()));
    if (pwalletMain && !pwalletMain->GetDHTKey(keyID, getKey))
        throw std::runtime_error("putbdapdata: ERRCODE: 5503 - Error getting ed25519 private key for the " + strFullObjectPath + _(" BDAP entry.\n"));

    if (getKey.GetPubKey() != entry.DHTPublicKey)
        throw std::runtime_error("putbdapdata: ERRCODE: 5504 - Error getting ed25519. Public key from wallet doesn't match entry for " + strFullObjectPath + _(" BDAP entry.\n"));

    result.push_back(Pair("entry_path", strFullObjectPath));
    result.push_back(Pair("wallet_address", stringFromVch(entry.WalletAddress)));
    result.push_back(Pair("link_address", stringFromVch(entry.LinkAddress)));
    result.push_back(Pair("put_pubkey", stringFromVch(getKey.GetPubKey())));
    //result.push_back(Pair("wallet_privkey", stringFromVch(getKey.GetPrivKey())));
    result.push_back(Pair("put_operation", strOperationType));

    int64_t iSequence = 0;
    bool fAuthoritative = false;
    std::string strGetLastValue;
    // we need the last sequence number to update an existing DHT entry. 
    GetDHTMutableData(getKey.GetDHTPubKey(), strOperationType, 1200, strGetLastValue, iSequence, fAuthoritative);
    iSequence++;
    CPutRequest put(getKey, strOperationType, iSequence, strValue);
    AddPutRequest(put);
    
    result.push_back(Pair("put_seq", iSequence));
    result.push_back(Pair("put_value", strValue));

    return result;
}

UniValue getbdapdata(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 2)
        throw std::runtime_error(
            "getbdapdata \"account id\" \"operation\"\n"
            "\nGets mutable data from the DHT for a BDAP entry and operation code.\n"
            "\nArguments:\n"
            "1. account id             (string)      BDAP account id\n"
            "2. operation              (string)      Mutable data operation\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"entry_path\"          (string)      BDAP account FQDN\n"
            "  \"wallet_address\"      (string)      BDAP account wallet address\n"
            "  \"link_address\"        (string)      BDAP account link address\n"
            "  \"get_pubkey\"          (string)      BDAP account DHT public key\n"
            "  \"get_operation\"       (string)      Mutable data operation code or salt\n"
            "  \"get_seq\"             (string)      Mutable data sequence number\n"
            "  \"get_value\"           (string)      Mutable data entry value\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("getbdapdata", "Duality avatar") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("getbdapdata", "Duality avatar"));

    UniValue result(UniValue::VOBJ);
   
    if (!pTorrentDHTSession)
        throw std::runtime_error("getbdapdata: ERRCODE: 5600 - DHT session not started.\n");

    if (!CheckDomainEntryDB())
        throw std::runtime_error("getbdapdata: ERRCODE: 5601 - Can not access BDAP domain entry database.\n");

    CharString vchObjectID = vchFromValue(request.params[0]);
    ToLowerCase(vchObjectID);
    CDomainEntry entry;
    entry.DomainComponent = vchDefaultDomainName;
    entry.OrganizationalUnit = vchDefaultPublicOU;
    entry.ObjectID = vchObjectID;
    std::string strFullObjectPath = entry.GetFullObjectPath();
    if (!pDomainEntryDB->GetDomainEntryInfo(entry.vchFullObjectPath(), entry))
        throw std::runtime_error("getbdapdata: ERRCODE: 5602 - " + strFullObjectPath + _(" can not be found.  Get BDAP entry failed!\n"));

    const std::string strOperationType = request.params[1].get_str();
    std::string strPubKey = stringFromVch(entry.DHTPublicKey);
    result.push_back(Pair("entry_path", strFullObjectPath));
    result.push_back(Pair("wallet_address", stringFromVch(entry.WalletAddress)));
    result.push_back(Pair("link_address", stringFromVch(entry.LinkAddress)));
    result.push_back(Pair("get_pubkey", strPubKey));
    result.push_back(Pair("get_operation", strOperationType));
    
    bool fRet = false;
    int64_t iSequence = 0;
    std::array<char, 32> arrPubKey;
    libtorrent::aux::from_hex(strPubKey, arrPubKey.data());
    
    std::string strValue = "";
    bool fAuthoritative;
    fRet = GetDHTMutableData(arrPubKey, strOperationType, 5000, strValue, iSequence, fAuthoritative);
    if (fRet) {
        result.push_back(Pair("get_seq", iSequence));
        result.push_back(Pair("get_value", strValue));
    }
    else {
        throw std::runtime_error("getbdapdata: ERRCODE: 5603 - Error getting data from DHT. Check the debug.log for details.\n");
    }

    return result;
}

UniValue dhtputmessages(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 0)
        throw std::runtime_error(
            "dhtputmessages"
            "\nGets all DHT put messages in memory.\n"
            "\nResult:\n"
            "{(json objects)\n"
            "  \"info_hash\"          (string)      Put entry info hash\n"
            "  \"public_key\"         (string)      Put entry public key\n"
            "  \"salt\"               (string)      Put entry salt or operation code\n"
            "  \"seq_num\"            (string)      Put entry sequence number\n"
            "  \"success_count\"      (string)      Put entry success count\n"
            "  \"message\"            (string)      Put entry message\n"
            "  \"what\"               (string)      Type of DHT alert event\n"
            "  \"timestamp\"          (string)      Put entry timestamp\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("dhtputmessages", "") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("dhtputmessages", ""));

    UniValue result(UniValue::VOBJ);

    std::vector<CMutablePutEvent> vchMutableData;
    bool fRet = GetAllDHTPutEvents(vchMutableData);
    int nCounter = 0;
    if (fRet) {
        for(const CMutablePutEvent& data : vchMutableData) {
            UniValue oMutableData(UniValue::VOBJ);
            oMutableData.push_back(Pair("info_hash", data.InfoHash()));
            oMutableData.push_back(Pair("public_key", data.PublicKey()));
            oMutableData.push_back(Pair("salt", data.Salt()));
            oMutableData.push_back(Pair("seq_num", data.SequenceNumber()));
            oMutableData.push_back(Pair("success_count", (int64_t)data.SuccessCount()));
            oMutableData.push_back(Pair("message", data.Message()));
            oMutableData.push_back(Pair("what", data.What()));
            oMutableData.push_back(Pair("timestamp", data.Timestamp()));
            result.push_back(Pair("dht_entry_" + std::to_string(nCounter + 1), oMutableData));
            nCounter++;
        }
    }
    else {
        throw std::runtime_error("dhtputmessages failed.  Check the debug.log for details.\n");
    }
    UniValue oCounter(UniValue::VOBJ);
    oCounter.push_back(Pair("record_count", nCounter));
    result.push_back(Pair("summary", oCounter));
    return result;
}

UniValue dhtgetmessages(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 0)
        throw std::runtime_error(
            "dhtgetmessages"
            "\nGets all DHT get messages in memory.\n"
            "\nResult:\n"
            "{(json objects)\n"
            "  \"info_hash\"          (string)      Get entry info hash\n"
            "  \"public_key\"         (string)      Get entry public key\n"
            "  \"salt\"               (string)      Get entry salt or operation code\n"
            "  \"seq_num\"            (string)      Get entry sequence number\n"
            "  \"authoritative\"      (string)      Get entry value is authoritative\n"
            "  \"message\"            (string)      Get entry message\n"
            "  \"what\"               (string)      Type of DHT alert event\n"
            "  \"timestamp\"          (string)      Get entry timestamp\n"
            "  \"value\"              (string)      Get entry value\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("dhtgetmessages", "") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("dhtgetmessages", ""));

    UniValue result(UniValue::VOBJ);

    std::vector<CMutableGetEvent> vchMutableData;
    bool fRet = GetAllDHTGetEvents(vchMutableData);
    int nCounter = 0;
    if (fRet) {
        for(const CMutableGetEvent& data : vchMutableData) {
            UniValue oMutableData(UniValue::VOBJ);
            oMutableData.push_back(Pair("info_hash", data.InfoHash()));
            oMutableData.push_back(Pair("public_key", data.PublicKey()));
            oMutableData.push_back(Pair("salt", data.Salt()));
            oMutableData.push_back(Pair("seq_num", data.SequenceNumber()));
            oMutableData.push_back(Pair("authoritative", data.Authoritative() ? "Yes" : "No"));
            oMutableData.push_back(Pair("message", data.Message()));
            oMutableData.push_back(Pair("what", data.What()));
            oMutableData.push_back(Pair("timestamp", data.Timestamp()));
            oMutableData.push_back(Pair("value", data.Value()));
            result.push_back(Pair("dht_entry_" + std::to_string(nCounter + 1), oMutableData));
            nCounter++;
        }
    }
    else {
        throw std::runtime_error("dhtgetmessages failed.  Check the debug.log for details.\n");
    }
    UniValue oCounter(UniValue::VOBJ);
    oCounter.push_back(Pair("record_count", nCounter));
    result.push_back(Pair("summary", oCounter));
    return result;
}

static const CRPCCommand commands[] =
{ //  category              name                     actor (function)               okSafe   argNames
  //  --------------------- ------------------------ -----------------------        ------   --------------------
    /* DHT */
    { "dht",             "getmutable",               &getmutable,                   true,    {"pubkey","operation"}  },
    { "dht",             "putmutable",               &putmutable,                   true,    {"data value","operation", "pubkey", "privkey"} },
    { "dht",             "dhtinfo",                  &dhtinfo,                      true,    {} },
    { "dht",             "dhtdb",                    &dhtdb,                        true,    {} },
    { "dht",             "putbdapdata",              &putbdapdata,                  true,    {"account id","data value", "operation"} },
    { "dht",             "getbdapdata",              &getbdapdata,                  true,    {"account id","operation"} },
    { "dht",             "dhtputmessages",           &dhtputmessages,               true,    {} },
    { "dht",             "dhtgetmessages",           &dhtgetmessages,               true,    {} },
};

void RegisterDHTRPCCommands(CRPCTable &tableRPC)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        tableRPC.appendCommand(commands[vcidx].name, &commands[vcidx]);
}