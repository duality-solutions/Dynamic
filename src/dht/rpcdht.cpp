// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/domainentry.h"
#include "bdap/domainentrydb.h"
#include "bdap/linking.h"
#include "bdap/linkmanager.h"
#include "bdap/utils.h"
#include "dht/datachunk.h" // for CDataChunk
#include "dht/dataheader.h" // for CRecordHeader
#include "dht/datarecord.h" // for CDataRecord
#include "dht/ed25519.h"
#include "dht/mutable.h"
#include "dht/mutabledb.h"
#include "dht/storage.h"
#include "dht/session.h"
#include "dht/sessionevents.h"
#include "bdap/vgp/include/encryption.h" // for VGP E2E encryption
#include "hash.h"
#include "pubkey.h"
#include "rpcprotocol.h"
#include "rpcserver.h"
#include "spork.h"
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

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_DHT_RPC_ERROR: ERRCODE: 3000 - " + _("Can not use DHT until BDAP spork is active."));

    UniValue result(UniValue::VOBJ);
    if (!pHashTableSession->Session)
        throw std::runtime_error("getdhtmutable failed. DHT session not started.\n");

    const std::string strPubKey = request.params[0].get_str();
    const std::string strSalt = request.params[1].get_str();

    bool fRet = false;
    int64_t iSequence = 0;
    std::string strValue = "";
    std::array<char, 32> pubKey;
    libtorrent::aux::from_hex(strPubKey, pubKey.data());
    bool fAuthoritative;
    fRet = pHashTableSession->SubmitGet(pubKey, strSalt, 2000, strValue, iSequence, fAuthoritative);
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

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_DHT_RPC_ERROR: ERRCODE: 3000 - " + _("Can not use DHT until BDAP spork is active."));

    UniValue result(UniValue::VOBJ);
    if (!pHashTableSession->Session)
        throw std::runtime_error("putmutable failed. DHT session not started.\n");

    //TODO: Check putValue is not > 1000 bytes.
    bool fNewEntry = false;
    const std::string strValue = request.params[1].get_str();
    const std::string strOperationType = request.params[1].get_str();
    std::string strPrivKey;
    std::string strPubKey;
    CKeyEd25519 key;
    if (request.params.size() == 4) {
        strPubKey = request.params[2].get_str();
        strPrivKey = request.params[3].get_str();
        std::vector<unsigned char> vch = vchFromString(strPubKey);
        CKeyID keyID(Hash160(vch.begin(), vch.end()));
        if (pwalletMain && !pwalletMain->GetDHTKey(keyID, key)) {
            throw std::runtime_error("putmutable: ERRCODE: 5400 - Error getting ed25519 private key for " + strPubKey + "\n");
        }
    }
    else if (request.params.size() == 2) {
        strPubKey = stringFromVch(key.GetPubKey());
        strPrivKey = stringFromVch(key.GetPrivKey());
        fNewEntry = true;
    }
    int64_t iSequence = 0;
    std::array<char, 32> pubKey;
    libtorrent::aux::from_hex(strPubKey, pubKey.data());
    std::array<char, 64> privKey;
    libtorrent::aux::from_hex(strPrivKey, privKey.data());
    bool fAuthoritative = false;
    if (!fNewEntry) {
        std::string strGetLastValue;
        // we need the last sequence number to update an existing DHT entry.
        pHashTableSession->SubmitGet(pubKey, strOperationType, 2000, strGetLastValue, iSequence, fAuthoritative);
        iSequence++;
    }
    uint16_t nTotalSlots = 32;
    std::vector<unsigned char> vchValue = vchFromValue(request.params[1]);
    std::vector<std::vector<unsigned char>> vvchPubKeys;
    vvchPubKeys.push_back(key.GetPubKeyBytes());
    uint16_t nVersion = 1; //TODO (DHT): Default is encrypted but add parameter to use version 0 (unencrypted)
    uint32_t nExpire = GetTime() + 2592000; // TODO (DHT): Default to 30 days but add an expiration date parameter.
    CDataRecord record(strOperationType, nTotalSlots, vvchPubKeys, vchValue, nVersion, nExpire, DHT::DataFormat::BinaryBlob);
    if (record.HasError())
        throw std::runtime_error("putbdapdata: ERRCODE: 5401 - Error creating DHT data entry. " + record.ErrorMessage() + _("\n"));

    record.GetHeader().Salt = strOperationType;
    if (!pHashTableSession->SubmitPut(key.GetDHTPubKey(), key.GetDHTPrivKey(), iSequence, record))
        throw std::runtime_error("putbdapdata: ERRCODE: 5402 - - Put failed. " + pHashTableSession->strPutErrorMessage + _("\n"));

    result.push_back(Pair("put_seq", iSequence));
    result.push_back(Pair("put_data_size", (int)vchValue.size()));
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

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_DHT_RPC_ERROR: ERRCODE: 3000 - " + _("Can not use DHT until BDAP spork is active."));

    if (!pHashTableSession->Session)
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
            "putbdapdata \"account id\" \"operation\" \"data value\"\n"
            "\nSaves mutable data in the DHT for a BDAP entry and operation code.\n"
            "\nArguments:\n"
            "1. account id             (string)      BDAP account id\n"
            "2. operation              (string)      Mutable data operation used for DHT entry\n"
            "3. data value             (string)      Mutable data value to save in the DHT\n"
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
            HelpExampleCli("putbdapdata", "duality avatar \"https://duality.solutions/duality/graphics/header/bdap.png\"") +
            "\nAs a JSON-RPC call\n" + 
            HelpExampleRpc("putbdapdata", "duality avatar \"https://duality.solutions/duality/graphics/header/bdap.png\""));

    EnsureWalletIsUnlocked();

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_DHT_RPC_ERROR: ERRCODE: 3000 - " + _("Can not use DHT until BDAP spork is active."));

    UniValue result(UniValue::VOBJ);
   
    EnsureWalletIsUnlocked();

    if (!pHashTableSession->Session)
        throw std::runtime_error("putbdapdata: ERRCODE: 5500 - DHT session not started.\n");

    if (!CheckDomainEntryDB())
        throw std::runtime_error("putbdapdata: ERRCODE: 5501 - Can not access BDAP domain entry database.\n");

    CharString vchObjectID = vchFromValue(request.params[0]);
    std::string strOperationType = request.params[1].get_str();
    ToLowerCase(strOperationType);

    const std::string strValue = request.params[2].get_str();

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
    std::string strHeaderHex;
    std::string strHeaderSalt = strOperationType + ":" + std::to_string(0);
    // we need the last sequence number to update an existing DHT entry. 
    pHashTableSession->SubmitGet(getKey.GetDHTPubKey(), strHeaderSalt, 2000, strHeaderHex, iSequence, fAuthoritative);
    CRecordHeader header(strHeaderHex);

    if (header.nUnlockTime  > GetTime())
        throw std::runtime_error(strprintf("%s:  ERRCODE: 5505 - DHT data entry is locked for another %lli seconds\n", __func__, (header.nUnlockTime  - GetTime())));

    iSequence++;
    uint16_t nVersion = 1; //TODO (DHT): Default is encrypted but add parameter for use cases where we want clear text.
    uint32_t nExpire = GetTime() + 2592000; // TODO (DHT): Default to 30 days but add an expiration date parameter.
    uint16_t nTotalSlots = 32;
    std::vector<unsigned char> vchValue = vchFromValue(request.params[2]);
    std::vector<std::vector<unsigned char>> vvchPubKeys;
    vvchPubKeys.push_back(getKey.GetPubKeyBytes());
    CDataRecord record(strOperationType, nTotalSlots, vvchPubKeys, vchValue, nVersion, nExpire, DHT::DataFormat::BinaryBlob);
    if (record.HasError())
        throw std::runtime_error("putbdapdata: ERRCODE: 5505 - Error creating DHT data entry. " + record.ErrorMessage() + _("\n"));

    pHashTableSession->SubmitPut(getKey.GetDHTPubKey(), getKey.GetDHTPrivKey(), iSequence, record);
    result.push_back(Pair("put_seq", iSequence));
    result.push_back(Pair("put_data_size", (int)vchValue.size()));
    return result;
}

UniValue clearbdapdata(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 2)
        throw std::runtime_error(
            "clearbdapdata \"account id\" \"operation\"\n"
            "\nClears mutable data entry in the DHT for a BDAP account and operation code.\n"
            "\nArguments:\n"
            "1. account id             (string)      BDAP account id\n"
            "2. operation              (string)      Mutable data operation used for DHT entry\n"
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
            HelpExampleCli("clearbdapdata", "duality auth") +
            "\nAs a JSON-RPC call\n" + 
            HelpExampleRpc("clearbdapdata", "duality auth"));

    EnsureWalletIsUnlocked();

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_DHT_RPC_ERROR: ERRCODE: 3000 - " + _("Can not use DHT until BDAP spork is active."));

    UniValue result(UniValue::VOBJ);
   
    EnsureWalletIsUnlocked();

    if (!pHashTableSession->Session)
        throw std::runtime_error("clearbdapdata: ERRCODE: 5510 - DHT session not started.\n");

    if (!CheckDomainEntryDB())
        throw std::runtime_error("clearbdapdata: ERRCODE: 5511 - Can not access BDAP domain entry database.\n");

    CharString vchObjectID = vchFromValue(request.params[0]);
    ToLowerCase(vchObjectID);
    std::string strOperationType = request.params[1].get_str();
    ToLowerCase(strOperationType);

    CDomainEntry entry;
    entry.DomainComponent = vchDefaultDomainName;
    entry.OrganizationalUnit = vchDefaultPublicOU;
    entry.ObjectID = vchObjectID;
    std::string strFullObjectPath = entry.GetFullObjectPath();
    if (!pDomainEntryDB->GetDomainEntryInfo(entry.vchFullObjectPath(), entry))
        throw std::runtime_error("clearbdapdata: ERRCODE: 5512 - " + strFullObjectPath + _(" can not be found.  Get BDAP info failed!\n"));

    CKeyEd25519 getKey;
    std::vector<unsigned char> vch = entry.DHTPublicKey;
    CKeyID keyID(Hash160(vch.begin(), vch.end()));
    if (pwalletMain && !pwalletMain->GetDHTKey(keyID, getKey))
        throw std::runtime_error("clearbdapdata: ERRCODE: 5513 - Error getting ed25519 private key for the " + strFullObjectPath + _(" BDAP entry.\n"));

    if (getKey.GetPubKey() != entry.DHTPublicKey)
        throw std::runtime_error("clearbdapdata: ERRCODE: 5514 - Error getting ed25519. Public key from wallet doesn't match entry for " + strFullObjectPath + _(" BDAP entry.\n"));

    result.push_back(Pair("entry_path", strFullObjectPath));
    result.push_back(Pair("wallet_address", stringFromVch(entry.WalletAddress)));
    result.push_back(Pair("link_address", stringFromVch(entry.LinkAddress)));
    result.push_back(Pair("put_pubkey", stringFromVch(getKey.GetPubKey())));
    //result.push_back(Pair("wallet_privkey", stringFromVch(getKey.GetPrivKey())));
    result.push_back(Pair("put_operation", strOperationType));

    int64_t iSequence = 0;
    bool fAuthoritative = false;
    std::string strHeaderHex;
    std::string strHeaderSalt = strOperationType + ":" + std::to_string(0);
    // we need the last sequence number to update an existing DHT entry. 
    pHashTableSession->SubmitGet(getKey.GetDHTPubKey(), strHeaderSalt, 2000, strHeaderHex, iSequence, fAuthoritative);
    CRecordHeader header(strHeaderHex);

    if (header.nUnlockTime  > GetTime())
        throw std::runtime_error(strprintf("%s:  ERRCODE: 5505 - DHT data entry is locked for another %lli seconds\n", __func__, (header.nUnlockTime  - GetTime())));

    iSequence++;
    uint16_t nVersion = 0;
    uint32_t nExpire = 0;
    uint16_t nTotalSlots = 32;
    std::vector<unsigned char> vchValue = ZeroCharVector();
    std::vector<std::vector<unsigned char>> vvchPubKeys;
    CDataRecord record(strOperationType, nTotalSlots, vvchPubKeys, vchValue, nVersion, nExpire, DHT::DataFormat::Null);
    if (record.HasError())
        throw std::runtime_error("clearbdapdata: ERRCODE: 5515 - Error creating DHT data entry. " + record.ErrorMessage() + _("\n"));

    if (!pHashTableSession->SubmitPut(getKey.GetDHTPubKey(), getKey.GetDHTPrivKey(), iSequence, record))
        throw std::runtime_error("putbdapdata: ERRCODE: 5516 - - Put failed. " + pHashTableSession->strPutErrorMessage + _("\n"));

    result.push_back(Pair("put_seq", iSequence));
    result.push_back(Pair("put_data_size", (int)vchValue.size()));
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

    EnsureWalletIsUnlocked();

    int64_t nStart = GetTimeMillis();
    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_DHT_RPC_ERROR: ERRCODE: 3000 - " + _("Can not use DHT until BDAP spork is active."));

    UniValue result(UniValue::VOBJ);
   
    if (!pHashTableSession->Session)
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
    std::vector<unsigned char> vchDHTPublicKey = entry.DHTPublicKey;
    CKeyEd25519 getKey;
    CKeyID keyID(Hash160(vchDHTPublicKey.begin(), vchDHTPublicKey.end()));
    if (pwalletMain && !pwalletMain->GetDHTKey(keyID, getKey))
        throw std::runtime_error("getbdapdata: ERRCODE: 5603 - Error getting ed25519 private key for the " + entry.GetFullObjectPath() + _(" BDAP entry.\n"));

    int64_t iSequence = 0;
    std::array<char, 32> arrPubKey;
    libtorrent::aux::from_hex(strPubKey, arrPubKey.data());
    CDataRecord record;
    if (!pHashTableSession->SubmitGetRecord(arrPubKey, getKey.GetDHTPrivSeed(), strOperationType, iSequence, record))
        throw std::runtime_error(strprintf("%s: ERRCODE: 5604 - Failed to get record: %s\n", __func__, pHashTableSession->strPutErrorMessage));

    result.push_back(Pair("get_seq", iSequence));
    result.push_back(Pair("data_encrypted", record.GetHeader().Encrypted() ? "true" : "false"));
    result.push_back(Pair("data_version", record.GetHeader().nVersion));
    result.push_back(Pair("data_chunks", record.GetHeader().nChunks));
    result.push_back(Pair("get_value", record.Value()));
    result.push_back(Pair("get_value_size", (int)record.Value().size()));

    int64_t nEnd = GetTimeMillis();
    result.push_back(Pair("get_milliseconds", (nEnd - nStart)));

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

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_DHT_RPC_ERROR: ERRCODE: 3000 - " + _("Can not use DHT until BDAP spork is active."));

    UniValue result(UniValue::VOBJ);

    std::vector<CMutablePutEvent> vchMutableData;
    bool fRet = false;//GetAllDHTPutEvents(vchMutableData);
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

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_DHT_RPC_ERROR: ERRCODE: 3000 - " + _("Can not use DHT until BDAP spork is active."));

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

UniValue getbdaplinkdata(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 3)
        throw std::runtime_error(
            "getbdaplinkdata \"account1\" \"account2\" \"operation\"\n"
            "\nGets mutable data from the DHT for a BDAP entry and operation code.\n"
            "\nArguments:\n"
            "1. account1               (string)      BDAP link account 1, gets the value for this account\n"
            "2. account2               (string)      BDAP link account 2, other account in the link\n"
            "3. operation              (string)      Mutable data operation code\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"link_requestor\"      (string)      BDAP account that initiated the link\n"
            "  \"link_acceptor\"       (string)      BDAP account that accepted the link\n"
            "  \"get_pubkey\"          (string)      BDAP account DHT public key for account1\n"
            "  \"get_operation\"       (string)      Mutable data operation code or salt\n"
            "  \"get_seq\"             (string)      Mutable data sequence number\n"
            "  \"get_value\"           (string)      Mutable data entry value\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("getbdaplinkdata", "duality bob auth") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("getbdaplinkdata", "duality bob auth"));

    EnsureWalletIsUnlocked();

    int64_t nStart = GetTimeMillis();
    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_DHT_RPC_ERROR: ERRCODE: 3000 - " + _("Can not use DHT until BDAP spork is active."));

    UniValue result(UniValue::VOBJ);
   
    if (!pHashTableSession->Session)
        throw std::runtime_error("getbdaplinkdata: ERRCODE: 5620 - DHT session not started.\n");

    if (!CheckDomainEntryDB())
        throw std::runtime_error("getbdaplinkdata: ERRCODE: 5621 - Can not access BDAP domain entry database.\n");

    if (!pLinkManager)
        throw std::runtime_error("getbdaplinkdata: ERRCODE: 5700 - Can not open link request in memory map");

    CharString vchObjectID1 = vchFromValue(request.params[0]);
    ToLowerCase(vchObjectID1);
    CDomainEntry entry1;
    entry1.DomainComponent = vchDefaultDomainName;
    entry1.OrganizationalUnit = vchDefaultPublicOU;
    entry1.ObjectID = vchObjectID1;
    if (!pDomainEntryDB->GetDomainEntryInfo(entry1.vchFullObjectPath(), entry1))
        throw std::runtime_error("getbdaplinkdata: ERRCODE: 5622 - " + entry1.GetFullObjectPath() + _(" can not be found.  Get BDAP entry failed!\n"));

    CharString vchObjectID2 = vchFromValue(request.params[1]);
    ToLowerCase(vchObjectID2);
    CDomainEntry entry2;
    entry2.DomainComponent = vchDefaultDomainName;
    entry2.OrganizationalUnit = vchDefaultPublicOU;
    entry2.ObjectID = vchObjectID2;
    if (!pDomainEntryDB->GetDomainEntryInfo(entry2.vchFullObjectPath(), entry2))
        throw std::runtime_error("getbdaplinkdata: ERRCODE: 5623 - " + entry2.GetFullObjectPath() + _(" can not be found.  Get BDAP entry failed!\n"));

    std::string strOperationType = request.params[2].get_str();
    ToLowerCase(strOperationType);

    if (!pLinkManager)
        throw std::runtime_error("getbdaplinkdata: ERRCODE: 5700 - Can not open link request in memory map");

    CLink link;
    std::string strPubKey;
    std::vector<unsigned char> vchDHTPublicKey;
    uint256 linkID = GetLinkID(entry1.GetFullObjectPath(), entry2.GetFullObjectPath());
    if (!pLinkManager->FindLink(linkID, link))
        throw std::runtime_error(strprintf("%s: ERRCODE: 5625 - Link id %s for %s and %s not found. Get BDAP link failed!", __func__, linkID.ToString(), entry1.GetFullObjectPath(), entry2.GetFullObjectPath()));

    CKeyEd25519 getKey;
    if (entry1.GetFullObjectPath() == link.RequestorFQDN()) {
        strPubKey = link.RequestorPubKeyString();
        vchDHTPublicKey = link.RequestorPubKey;
        CKeyID keyID(Hash160(vchDHTPublicKey.begin(), vchDHTPublicKey.end()));
        if (pwalletMain && !pwalletMain->GetDHTKey(keyID, getKey)) {
            std::vector<unsigned char> vchPublicKey = link.RecipientPubKey;
            CKeyID keyID2(Hash160(vchPublicKey.begin(), vchPublicKey.end()));
            if (pwalletMain && !pwalletMain->GetDHTKey(keyID2, getKey)) {
                throw std::runtime_error("getbdaplinkdata: ERRCODE: 5625 - DHT link private key for " + entry1.GetFullObjectPath() + " or " + entry2.GetFullObjectPath() +
                                             _(" not found. Get BDAP link entry failed!\n"));
            }
        }
    }
    else if (entry1.GetFullObjectPath() == link.RecipientFQDN()) {
        strPubKey = link.RecipientPubKeyString();
        vchDHTPublicKey = link.RecipientPubKey;
        CKeyID keyID(Hash160(vchDHTPublicKey.begin(), vchDHTPublicKey.end()));
        if (pwalletMain && !pwalletMain->GetDHTKey(keyID, getKey)) {
            std::vector<unsigned char> vchPublicKey = link.RequestorPubKey;
            CKeyID keyID2(Hash160(vchPublicKey.begin(), vchPublicKey.end()));
            if (pwalletMain && !pwalletMain->GetDHTKey(keyID2, getKey)) {
                throw std::runtime_error("getbdaplinkdata: ERRCODE: 5625 - DHT link private key for " + entry1.GetFullObjectPath() + " or " + entry2.GetFullObjectPath() +
                                             _(" not found. Get BDAP link entry failed!\n"));
            }
        }
    }

    result.push_back(Pair("link_requestor", link.RequestorFQDN()));
    result.push_back(Pair("link_acceptor", link.RecipientFQDN()));

    result.push_back(Pair("get_pubkey", strPubKey));
    result.push_back(Pair("get_operation", strOperationType));

    int64_t iSequence = 0;
    std::array<char, 32> arrPubKey;
    libtorrent::aux::from_hex(strPubKey, arrPubKey.data());
    CDataRecord record;
    if (!pHashTableSession->SubmitGetRecord(arrPubKey, getKey.GetDHTPrivSeed(), strOperationType, iSequence, record))
        throw std::runtime_error(strprintf("%s: ERRCODE: 5626 - Failed to get record: %s\n", __func__, pHashTableSession->strPutErrorMessage));

    result.push_back(Pair("get_seq", iSequence));
    result.push_back(Pair("data_encrypted", record.GetHeader().Encrypted() ? "true" : "false"));
    result.push_back(Pair("data_version", record.GetHeader().nVersion));
    result.push_back(Pair("data_chunks", record.GetHeader().nChunks));
    result.push_back(Pair("get_value", record.Value()));
    result.push_back(Pair("get_value_size", (int)record.Value().size()));

    int64_t nEnd = GetTimeMillis();
    result.push_back(Pair("get_milliseconds", (nEnd - nStart)));

    return result;
}

UniValue getallbdaplinkdata(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 2)
        throw std::runtime_error(
            "getallbdaplinkdata \"account\" \"operation\"\n"
            "\nGets mutable data from the DHT for a BDAP entry and operation code.\n"
            "\nArguments:\n"
            "1. account                (string)      BDAP account used to get all data for this operation\n"
            "2. operation              (string)      Mutable data operation code\n"
            "\nResult:\n"
            "{(json objects)\n"
            "  \"link_requestor\"      (string)      BDAP account that initiated the link\n"
            "  \"link_acceptor\"       (string)      BDAP account that accepted the link\n"
            "  \"get_pubkey\"          (string)      BDAP account DHT public key for account1\n"
            "  \"get_operation\"       (string)      Mutable data operation code or salt\n"
            "  \"get_seq\"             (string)      Mutable data sequence number\n"
            "  \"get_value\"           (string)      Mutable data entry value\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("getallbdaplinkdata", "duality pshare-offer") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("getallbdaplinkdata", "duality pshare-offer"));

    EnsureWalletIsUnlocked();

    int64_t nStart = GetTimeMillis();
    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_DHT_RPC_ERROR: ERRCODE: 3000 - " + _("Can not use DHT until BDAP spork is active."));

    UniValue results(UniValue::VOBJ);
   
    if (!pHashTableSession->Session)
        throw std::runtime_error("getallbdaplinkdata: ERRCODE: 5720 - DHT session not started.\n");

    if (!CheckDomainEntryDB())
        throw std::runtime_error("getallbdaplinkdata: ERRCODE: 5721 - Can not access BDAP domain entry database.\n");

    if (!pLinkManager)
        throw std::runtime_error("getallbdaplinkdata: ERRCODE: 5700 - Can not open link request in memory map");

    CharString vchObjectID = vchFromValue(request.params[0]);
    ToLowerCase(vchObjectID);
    CDomainEntry entry;
    entry.DomainComponent = vchDefaultDomainName;
    entry.OrganizationalUnit = vchDefaultPublicOU;
    entry.ObjectID = vchObjectID;
    if (!pDomainEntryDB->GetDomainEntryInfo(entry.vchFullObjectPath(), entry))
        throw std::runtime_error("getallbdaplinkdata: ERRCODE: 5722 - " + entry.GetFullObjectPath() + _(" can not be found.  Get BDAP entry failed!\n"));

    std::string strOperationType = request.params[1].get_str();
    ToLowerCase(strOperationType);

    if (!pLinkManager)
        throw std::runtime_error("getallbdaplinkdata: ERRCODE: 5700 - Can not open link request in memory map");

    std::vector<CLinkInfo> vchLinkInfo = pLinkManager->GetCompletedLinkInfo(entry.vchFullObjectPath());
    for (CLinkInfo& linkInfo : vchLinkInfo) {
        CKeyEd25519 getKey;
        std::vector<unsigned char> vchDHTPublicKey = linkInfo.vchReceivePubKey;
        CKeyID keyID(Hash160(vchDHTPublicKey.begin(), vchDHTPublicKey.end()));
        if (pwalletMain && !pwalletMain->GetDHTKey(keyID, getKey))
           throw std::runtime_error(strprintf("%s: ERRCODE: 5725 - Failed to get DHT private key for account %s (pubkey = %s)\n", __func__, entry.GetFullObjectPath(), entry.DHTPubKeyString()));

       linkInfo.arrReceivePrivateSeed = getKey.GetDHTPrivSeed();
    }

    std::vector<CDataRecord> vchRecords;
    if (!pHashTableSession->SubmitGetAllRecordsSync(vchLinkInfo, strOperationType, vchRecords))
        throw std::runtime_error(strprintf("%s: ERRCODE: 5726 - Failed to get records: %s\n", __func__, pHashTableSession->strPutErrorMessage));

    results.push_back(Pair("get_operation", strOperationType));
    int nRecordItem = 1;
    for (CDataRecord& record : vchRecords) // loop through records
    {
        UniValue result(UniValue::VOBJ);
        result.push_back(Pair("account", stringFromVch(record.vchOwnerFQDN)));
        result.push_back(Pair("data_encrypted", record.Encrypted() ? "true" : "false"));
        result.push_back(Pair("data_version", record.Version()));
        result.push_back(Pair("data_chunks", record.GetHeader().nChunks));
        result.push_back(Pair("get_value", record.Value()));
        result.push_back(Pair("get_value_size", (int)record.Value().size()));
        results.push_back(Pair("record_" + std::to_string(nRecordItem), result));
        nRecordItem++;
    }
    int64_t nEnd = GetTimeMillis();
    results.push_back(Pair("get_milliseconds", (nEnd - nStart)));

    return results;
}

UniValue putbdaplinkdata(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 4)
        throw std::runtime_error(
            "putbdaplinkdata \"account1\" \"account2\" \"operation\" \"value\"\n"
            "\nSaves a link mutable data entry in the DHT for a BDAP entry and operation code.\n"
            "\nArguments:\n"
            "1. account1               (string)      BDAP link account 1, gets the value for this account\n"
            "2. account2               (string)      BDAP link account 2, other account in the link\n"
            "3. operation              (string)      Mutable data operation code\n"
            "4. value                  (string)                                 \n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"link_requestor\"      (string)      BDAP account that initiated the link\n"
            "  \"link_acceptor\"       (string)      BDAP account that accepted the link\n"
            "  \"put_pubkey\"          (string)      BDAP account DHT public key for account1\n"
            "  \"put_operation\"       (string)      Mutable data operation code or salt\n"
            "  \"put_seq\"             (string)      Mutable data sequence number\n"
            "  \"put_value\"           (string)      Mutable data entry value\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("putbdaplinkdata", "duality bob auth \"save this auth data\"") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("putbdaplinkdata", "duality bob auth \"save this auth data\""));

    EnsureWalletIsUnlocked();

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_DHT_RPC_ERROR: ERRCODE: 3000 - " + _("Can not use DHT until BDAP spork is active."));

    UniValue result(UniValue::VOBJ);

    if (!pHashTableSession->Session)
        throw std::runtime_error("putbdaplinkdata: ERRCODE: 5640 - DHT session not started.\n");

    if (!CheckDomainEntryDB())
        throw std::runtime_error("putbdaplinkdata: ERRCODE: 5641 - Can not access BDAP domain entry database.\n");

    if (!pLinkManager)
        throw std::runtime_error("putbdaplinkdata: ERRCODE: 5700 - Can not open link request in memory map");

    CharString vchObjectID1 = vchFromValue(request.params[0]);
    ToLowerCase(vchObjectID1);
    CDomainEntry entry1;
    entry1.DomainComponent = vchDefaultDomainName;
    entry1.OrganizationalUnit = vchDefaultPublicOU;
    entry1.ObjectID = vchObjectID1;
    if (!pDomainEntryDB->GetDomainEntryInfo(entry1.vchFullObjectPath(), entry1))
        throw std::runtime_error("putbdaplinkdata: ERRCODE: 5642 - " + entry1.GetFullObjectPath() + _(" can not be found.  Put BDAP DHT entry failed!\n"));

    CharString vchObjectID2 = vchFromValue(request.params[1]);
    ToLowerCase(vchObjectID2);
    CDomainEntry entry2;
    entry2.DomainComponent = vchDefaultDomainName;
    entry2.OrganizationalUnit = vchDefaultPublicOU;
    entry2.ObjectID = vchObjectID2;
    if (!pDomainEntryDB->GetDomainEntryInfo(entry2.vchFullObjectPath(), entry2))
        throw std::runtime_error("putbdaplinkdata: ERRCODE: 5643 - " + entry2.GetFullObjectPath() + _(" can not be found.  Put BDAP DHT entry failed!\n"));

    std::string strOperationType = request.params[2].get_str();
    ToLowerCase(strOperationType);

    CLink link;
    uint256 linkID = GetLinkID(entry1.GetFullObjectPath(), entry2.GetFullObjectPath());
    if (!pLinkManager->FindLink(linkID, link))
        throw std::runtime_error(strprintf("%s: ERRCODE: 5644 - Link id %s for %s and %s not found. Get BDAP link failed!", __func__, linkID.ToString(), entry1.GetFullObjectPath(), entry2.GetFullObjectPath()));

    LogPrintf("%s -- req: %s, rec: %s \n", __func__, link.RequestorFQDN(), link.RecipientFQDN());
    std::vector<unsigned char> vchDHTPublicKey;
    if (entry1.GetFullObjectPath() == link.RequestorFQDN()) {
        vchDHTPublicKey = link.RequestorPubKey;
    }
    else if (entry1.GetFullObjectPath() == link.RecipientFQDN()) {
        vchDHTPublicKey = link.RecipientPubKey;
    }
    else {
        throw std::runtime_error("putbdaplinkdata: ERRCODE: 5645 - DHT link public key for " + entry1.GetFullObjectPath() + _(" not found. Put BDAP DHT entry failed!\n"));
    }
    result.push_back(Pair("link_requestor", link.RequestorFQDN()));
    result.push_back(Pair("link_acceptor", link.RecipientFQDN()));
    result.push_back(Pair("put_pubkey", stringFromVch(vchDHTPublicKey)));
    result.push_back(Pair("put_operation", strOperationType));


    CKeyEd25519 getKey;
    CKeyID keyID(Hash160(vchDHTPublicKey.begin(), vchDHTPublicKey.end()));
    if (pwalletMain && !pwalletMain->GetDHTKey(keyID, getKey))
        throw std::runtime_error("putbdaplinkdata: ERRCODE: 5546 - Error getting ed25519 private key for the " + entry1.GetFullObjectPath() + _(" BDAP entry.\n"));

    int64_t iSequence = 0;
    bool fAuthoritative = false;
    std::string strHeaderHex;

    // we need the last sequence number to update an existing DHT entry.
    std::string strHeaderSalt = strOperationType + ":" + std::to_string(0);
    pHashTableSession->SubmitGet(getKey.GetDHTPubKey(), strHeaderSalt, 2000, strHeaderHex, iSequence, fAuthoritative);
    CRecordHeader header(strHeaderHex);

    if (header.nUnlockTime  > GetTime())
        throw std::runtime_error(strprintf("%s:  ERRCODE: 5505 - DHT data entry is locked for another %lli seconds\n", __func__, (header.nUnlockTime  - GetTime())));

    iSequence++;

    uint16_t nVersion = 1; //TODO (DHT): Default is encrypted but add parameter for use cases where we want clear text.
    uint32_t nExpire = GetTime() + 2592000; // TODO (DHT): Default to 30 days but add an expiration date parameter.
    uint16_t nTotalSlots = 32;
    std::vector<unsigned char> vchValue = vchFromValue(request.params[3]);
    std::vector<std::vector<unsigned char>> vvchPubKeys;
    vvchPubKeys.push_back(EncodedPubKeyToBytes(link.RequestorPubKey));
    vvchPubKeys.push_back(EncodedPubKeyToBytes(link.RecipientPubKey));
    CDataRecord record(strOperationType, nTotalSlots, vvchPubKeys, vchValue, nVersion, nExpire, DHT::DataFormat::BinaryBlob);
    if (record.HasError())
        throw std::runtime_error("putbdaplinkdata: ERRCODE: 5547 - Error creating DHT data entry. " + record.ErrorMessage() + _("\n"));

    if (!pHashTableSession->SubmitPut(getKey.GetDHTPubKey(), getKey.GetDHTPrivKey(), iSequence, record))
        throw std::runtime_error("putbdaplinkdata: ERRCODE: 5548 - - Put failed. " + pHashTableSession->strPutErrorMessage + _("\n"));

    result.push_back(Pair("put_seq", iSequence));
    result.push_back(Pair("put_data_size", (int)vchValue.size()));

    return result;
}

UniValue clearbdaplinkdata(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 3)
        throw std::runtime_error(
            "clearbdaplinkdata \"account1\" \"account2\" \"operation\"\n"
            "\nClears a link mutable data entry from the DHT for a BDAP entry and operation code.\n"
            "\nArguments:\n"
            "1. account1               (string)      BDAP link account 1, gets the value for this account\n"
            "2. account2               (string)      BDAP link account 2, other account in the link\n"
            "3. operation              (string)      Mutable data operation code\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"link_requestor\"      (string)      BDAP account that initiated the link\n"
            "  \"link_acceptor\"       (string)      BDAP account that accepted the link\n"
            "  \"put_pubkey\"          (string)      BDAP account DHT public key for account1\n"
            "  \"put_operation\"       (string)      Mutable data operation code or salt\n"
            "  \"put_seq\"             (string)      Mutable data sequence number\n"
            "  }\n"
            "\nExamples\n" +
           HelpExampleCli("clearbdaplinkdata", "duality bob auth") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("clearbdaplinkdata", "duality bob auth"));

    EnsureWalletIsUnlocked();

    if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        throw std::runtime_error("BDAP_DHT_RPC_ERROR: ERRCODE: 3000 - " + _("Can not use DHT until BDAP spork is active."));

    UniValue result(UniValue::VOBJ);

    if (!pHashTableSession->Session)
        throw std::runtime_error("clearbdaplinkdata: ERRCODE: 5650 - DHT session not started.\n");

    if (!CheckDomainEntryDB())
        throw std::runtime_error("clearbdaplinkdata: ERRCODE: 5651 - Can not access BDAP domain entry database.\n");

    if (!pLinkManager)
        throw std::runtime_error("clearbdaplinkdata: ERRCODE: 5700 - Can not open link request in memory map");

    CharString vchObjectID1 = vchFromValue(request.params[0]);
    ToLowerCase(vchObjectID1);
    CDomainEntry entry1;
    entry1.DomainComponent = vchDefaultDomainName;
    entry1.OrganizationalUnit = vchDefaultPublicOU;
    entry1.ObjectID = vchObjectID1;
    if (!pDomainEntryDB->GetDomainEntryInfo(entry1.vchFullObjectPath(), entry1))
        throw std::runtime_error("clearbdaplinkdata: ERRCODE: 5642 - " + entry1.GetFullObjectPath() + _(" can not be found.  Put BDAP DHT entry failed!\n"));

    CharString vchObjectID2 = vchFromValue(request.params[1]);
    ToLowerCase(vchObjectID2);
    CDomainEntry entry2;
    entry2.DomainComponent = vchDefaultDomainName;
    entry2.OrganizationalUnit = vchDefaultPublicOU;
    entry2.ObjectID = vchObjectID2;
    if (!pDomainEntryDB->GetDomainEntryInfo(entry2.vchFullObjectPath(), entry2))
        throw std::runtime_error("clearbdaplinkdata: ERRCODE: 5643 - " + entry2.GetFullObjectPath() + _(" can not be found.  Put BDAP DHT entry failed!\n"));

    std::string strOperationType = request.params[2].get_str();
    ToLowerCase(strOperationType);

    CLink link;
    uint256 linkID = GetLinkID(entry1.GetFullObjectPath(), entry2.GetFullObjectPath());
    if (!pLinkManager->FindLink(linkID, link))
        throw std::runtime_error(strprintf("%s: ERRCODE: 5644 - Link id %s for %s and %s not found. Get BDAP link failed!", __func__, linkID.ToString(), entry1.GetFullObjectPath(), entry2.GetFullObjectPath()));

    LogPrintf("%s -- req: %s, rec: %s \n", __func__, link.RequestorFQDN(), link.RecipientFQDN());
    std::vector<unsigned char> vchDHTPublicKey;
    if (entry1.GetFullObjectPath() == link.RequestorFQDN()) {
        vchDHTPublicKey = link.RequestorPubKey;
    }
    else if (entry1.GetFullObjectPath() == link.RecipientFQDN()) {
        vchDHTPublicKey = link.RecipientPubKey;
    }
    else {
        throw std::runtime_error("clearbdaplinkdata: ERRCODE: 5645 - DHT link public key for " + entry1.GetFullObjectPath() + _(" not found. Put BDAP DHT entry failed!\n"));
    }
    result.push_back(Pair("link_requestor", link.RequestorFQDN()));
    result.push_back(Pair("link_acceptor", link.RecipientFQDN()));
    result.push_back(Pair("put_pubkey", stringFromVch(vchDHTPublicKey)));
    result.push_back(Pair("put_operation", strOperationType));


    CKeyEd25519 getKey;
    CKeyID keyID(Hash160(vchDHTPublicKey.begin(), vchDHTPublicKey.end()));
    if (pwalletMain && !pwalletMain->GetDHTKey(keyID, getKey))
        throw std::runtime_error("clearbdaplinkdata: ERRCODE: 5546 - Error getting ed25519 private key for the " + entry1.GetFullObjectPath() + _(" BDAP entry.\n"));

    int64_t iSequence = 0;
    bool fAuthoritative = false;
    std::string strHeaderHex;

    // we need the last sequence number to update an existing DHT entry.
    std::string strHeaderSalt = strOperationType + ":" + std::to_string(0);
    pHashTableSession->SubmitGet(getKey.GetDHTPubKey(), strHeaderSalt, 2000, strHeaderHex, iSequence, fAuthoritative);
    CRecordHeader header(strHeaderHex);

    if (header.nUnlockTime  > GetTime())
        throw std::runtime_error(strprintf("%s:  ERRCODE: 5505 - DHT data entry is locked for another %lli seconds\n", __func__, (header.nUnlockTime  - GetTime())));

    iSequence++;

    uint16_t nVersion = 0;
    uint32_t nExpire = 0;
    uint16_t nTotalSlots = 32;
    std::vector<unsigned char> vchValue = ZeroCharVector();
    std::vector<std::vector<unsigned char>> vvchPubKeys;
    CDataRecord record(strOperationType, nTotalSlots, vvchPubKeys, vchValue, nVersion, nExpire, DHT::DataFormat::Null);
    if (record.HasError())
        throw std::runtime_error("clearbdaplinkdata: ERRCODE: 5547 - Error creating DHT data entry. " + record.ErrorMessage() + _("\n"));

    if (!pHashTableSession->SubmitPut(getKey.GetDHTPubKey(), getKey.GetDHTPrivKey(), iSequence, record))
        throw std::runtime_error("clearbdaplinkdata: ERRCODE: 5548 - - Put failed. " + pHashTableSession->strPutErrorMessage + _("\n"));

    result.push_back(Pair("put_seq", iSequence));
    result.push_back(Pair("put_data_size", (int)vchValue.size()));

    return result;
}

static const CRPCCommand commands[] =
{ //  category              name                     actor (function)               okSafe   argNames
  //  --------------------- ------------------------ -----------------------        ------   --------------------
    /* DHT */
    { "dht",             "getmutable",               &getmutable,                   true,    {"pubkey", "operation"}  },
    { "dht",             "putmutable",               &putmutable,                   true,    {"data value", "operation", "pubkey", "privkey"} },
    { "dht",             "dhtinfo",                  &dhtinfo,                      true,    {} },
    { "dht",             "dhtdb",                    &dhtdb,                        true,    {} },
    { "dht",             "putbdapdata",              &putbdapdata,                  true,    {"account id", "operation", "data value"} },
    { "dht",             "getbdapdata",              &getbdapdata,                  true,    {"account id", "operation"} },
    { "dht",             "dhtputmessages",           &dhtputmessages,               true,    {} },
    { "dht",             "dhtgetmessages",           &dhtgetmessages,               true,    {} },
    { "dht",             "getbdaplinkdata",          &getbdaplinkdata,              true,    {"account1", "account2", "operation"} },
    { "dht",             "putbdaplinkdata",          &putbdaplinkdata,              true,    {"account1", "account2", "operation", "value"} },
    { "dht",             "clearbdapdata",            &clearbdapdata,                true,    {"account1", "account2", "operation"} },
    { "dht",             "clearbdaplinkdata",        &clearbdaplinkdata,            true,    {"account1", "account2", "operation"} },
    { "dht",             "getallbdaplinkdata",       &getallbdaplinkdata,           true,    {"account1", "operation"} },
};

void RegisterDHTRPCCommands(CRPCTable &tableRPC)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        tableRPC.appendCommand(commands[vcidx].name, &commands[vcidx]);
}
