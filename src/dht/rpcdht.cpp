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

static UniValue GetMutable(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 3)
        throw std::runtime_error(
            "dht getmutable \"pubkey\" \"operation\"\n"
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
           HelpExampleCli("dht getmutable", "517c4242c95214e5eb631e1ddf4e7dac5e815f0578f88491b81fd36df3c2a16a avatar") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("dht getmutable", "517c4242c95214e5eb631e1ddf4e7dac5e815f0578f88491b81fd36df3c2a16a avatar"));

    UniValue result(UniValue::VOBJ);

    if (!DHT::SessionStatus())
        throw JSONRPCError(RPC_DHT_NOT_STARTED, strprintf("dht %s failed. DHT session not started.", request.params[0].get_str()));

    const std::string strPubKey = request.params[1].get_str();
    const std::string strSalt = request.params[2].get_str();

    bool fRet = false;
    int64_t iSequence = 0;
    std::string strValue = "";
    std::array<char, 32> pubKey;
    libtorrent::aux::from_hex(strPubKey, pubKey.data());
    bool fAuthoritative;
    fRet = DHT::SubmitGet(0, pubKey, strSalt, 2000, strValue, iSequence, fAuthoritative);
    if (fRet) {
        result.push_back(Pair("Public Key", strPubKey));
        result.push_back(Pair("Salt", strSalt));
        result.push_back(Pair("Sequence Number", iSequence));
        result.push_back(Pair("Authoritative", fAuthoritative ? "True" : "False"));
        result.push_back(Pair("DHT Entry Value", strValue));
    }
    else {
        throw JSONRPCError(RPC_DHT_ERROR, strprintf("dht %s failed. Check the debug.log for details.", request.params[0].get_str()));
    }

    return result;
}

static UniValue PutMutable(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() < 3 || request.params.size() > 5 || request.params.size() == 4)
        throw std::runtime_error(
            "dht putmutable \"data value\" \"operation\" \"pubkey\" \"privkey\"\n"
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
           HelpExampleCli("dht putmutable", "\"https://duality.solutions/duality/logos/dual.png\" \"avatar\" \"517c4242c95214e5eb631e1ddf4e7dac5e815f0578f88491b81fd36df3c2a16a\" \"bf8b4f66bdd9e7dc526ddc3637a4edf8e0ac86b7df5e249fc6514a0a1c047cd0\"") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("dht putmutable", "\"https://duality.solutions/duality/logos/dual.png\" \"avatar\" \"517c4242c95214e5eb631e1ddf4e7dac5e815f0578f88491b81fd36df3c2a16a\" \"bf8b4f66bdd9e7dc526ddc3637a4edf8e0ac86b7df5e249fc6514a0a1c047cd0\""));

    UniValue result(UniValue::VOBJ);

    if (!DHT::SessionStatus())
        throw JSONRPCError(RPC_DHT_NOT_STARTED, strprintf("dht %s failed. DHT session not started.", request.params[0].get_str()));

    //TODO: Check putValue is not > 1000 bytes.
    bool fNewEntry = false;
    const std::string strValue = request.params[1].get_str();
    const std::string strOperationType = request.params[2].get_str();
    std::string strPrivKey;
    std::string strPubKey;
    CKeyEd25519 key;
    if (request.params.size() == 5) {
        strPubKey = request.params[3].get_str();
        strPrivKey = request.params[4].get_str();
        std::vector<unsigned char> vch = vchFromString(strPubKey);
        CKeyID keyID(Hash160(vch.begin(), vch.end()));
        if (pwalletMain && !pwalletMain->GetDHTKey(keyID, key)) {
            throw JSONRPCError(RPC_DHT_GET_KEY_FAILED, strprintf("Error getting ed25519 private key for %s", strPubKey));
        }
    }
    else if (request.params.size() == 3) {
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
        DHT::SubmitGet(0, pubKey, strOperationType, 2000, strGetLastValue, iSequence, fAuthoritative);
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
        throw JSONRPCError(RPC_DHT_GET_FAILED, strprintf("Error creating DHT data entry. %s", record.ErrorMessage()));

    record.GetHeader().Salt = strOperationType;
    std::string strErrorMessage;
    if (!DHT::SubmitPut(key.GetDHTPubKey(), key.GetDHTPrivKey(), iSequence, record, strErrorMessage))
        throw JSONRPCError(RPC_DHT_PUT_FAILED, strprintf("Put failed. %s", strErrorMessage));

    result.push_back(Pair("put_seq", iSequence));
    result.push_back(Pair("put_data_size", (int)vchValue.size()));
    return result;
}

static UniValue GetDHTStatus(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "dht status"
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

    if (!DHT::SessionStatus())
        throw JSONRPCError(RPC_DHT_NOT_STARTED, strprintf("dht %s failed. DHT session not started.", request.params[0].get_str()));

    CSessionStats stats;
    DHT::GetDHTStats(stats);
    UniValue result(UniValue::VOBJ);
    result.push_back(Pair("num_sessions", stats.nSessions));
    result.push_back(Pair("put_records", stats.nPutRecords));
    result.push_back(Pair("put_pieces", stats.nPutPieces));
    result.push_back(Pair("put_bytes", stats.nPutBytes));
    result.push_back(Pair("get_records", stats.nGetRecords));
    result.push_back(Pair("get_pieces", stats.nGetPieces));
    result.push_back(Pair("get_bytes", stats.nGetBytes));
    result.push_back(Pair("get_errors", stats.nGetErrors));

    for (const std::pair<std::string, std::string>& pairMessage : stats.vMessages)
    {
        result.push_back(Pair(pairMessage.first, pairMessage.second));
    }

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
            oMutableData.push_back(Pair("value", EncodeBase64(data.Value())));
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

static UniValue PutRecord(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() < 4 || request.params.size() > 5)
        throw std::runtime_error(
            "dht putrecord \"account id\" \"operation\" \"data value\"\n"
            "\nSaves mutable data in the DHT for a BDAP entry and operation code.\n"
            "\nArguments:\n"
            "1. account id             (string)           BDAP account id\n"
            "2. operation              (string)           Mutable data operation used for DHT entry\n"
            "3. data value             (string)           Mutable data value to save in the DHT\n"
            "4. encrypt                (bool, optional)   Encrypt the data with the user's public key\n"
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
            HelpExampleCli("dht putrecord", "duality avatar \"https://duality.solutions/duality/graphics/header/bdap.png\" 0") +
            "\nAs a JSON-RPC call\n" + 
            HelpExampleRpc("dht putrecord", "duality avatar \"https://duality.solutions/duality/graphics/header/bdap.png\" 0"));

    EnsureWalletIsUnlocked();

    UniValue result(UniValue::VOBJ);

    if (!DHT::SessionStatus())
        throw JSONRPCError(RPC_DHT_NOT_STARTED, strprintf("dht %s failed. DHT session not started.", request.params[0].get_str()));

    if (!CheckDomainEntryDB())
        throw JSONRPCError(RPC_BDAP_DB_ERROR, strprintf("Can not access BDAP domain entry database."));

    CharString vchObjectID = vchFromValue(request.params[1]);
    ToLowerCase(vchObjectID);

    std::string strOperationType = request.params[2].get_str();
    ToLowerCase(strOperationType);

    CDomainEntry entry;
    entry.DomainComponent = vchDefaultDomainName;
    entry.OrganizationalUnit = vchDefaultPublicOU;
    entry.ObjectID = vchObjectID;
    std::string strFullObjectPath = entry.GetFullObjectPath();
    if (!pDomainEntryDB->GetDomainEntryInfo(entry.vchFullObjectPath(), entry))
        throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", strFullObjectPath));

    CKeyEd25519 getKey;
    std::vector<unsigned char> vch = entry.DHTPublicKey;
    CKeyID keyID(Hash160(vch.begin(), vch.end()));
    if (pwalletMain && !pwalletMain->GetDHTKey(keyID, getKey))
        throw JSONRPCError(RPC_DHT_GET_KEY_FAILED, strprintf("Failed to get DHT private key for account %s (pubkey = %s)", strFullObjectPath, stringFromVch(vch)));

    if (getKey.GetPubKey() != entry.DHTPublicKey)
        throw std::runtime_error("putbdapdata: ERRCODE: 5504 - Error getting ed25519. Public key from wallet doesn't match entry for " + strFullObjectPath + _(" BDAP entry.\n"));

    result.push_back(Pair("entry_path", strFullObjectPath));
    result.push_back(Pair("wallet_address", stringFromVch(entry.WalletAddress)));
    result.push_back(Pair("link_address", stringFromVch(entry.LinkAddress)));
    result.push_back(Pair("put_pubkey", stringFromVch(getKey.GetPubKey())));
    result.push_back(Pair("put_operation", strOperationType));

    int64_t iSequence = 0;
    bool fAuthoritative = false;
    std::string strHeaderHex;
    std::string strHeaderSalt = strOperationType + ":" + std::to_string(0);
    // we need the last sequence number to update an existing DHT entry. 
    DHT::SubmitGet(0, getKey.GetDHTPubKey(), strHeaderSalt, 2000, strHeaderHex, iSequence, fAuthoritative);
    CRecordHeader header(strHeaderHex);
    if (header.nUnlockTime  > GetTime())
        throw JSONRPCError(RPC_DHT_RECORD_LOCKED, strprintf("DHT data entry is locked for another %lli seconds", (header.nUnlockTime  - GetTime())));

    iSequence++;
    uint16_t nVersion = 1;
    uint32_t nExpire = GetTime() + 2592000; // TODO (DHT): Default to 30 days but add an expiration date parameter.
    uint16_t nTotalSlots = 32;
    const std::vector<unsigned char> vchValue = vchFromValue(request.params[3]);
    bool fEncrypt = true;
    if (request.params.size() > 4)
        fEncrypt = request.params[4].getBool();

    std::vector<std::vector<unsigned char>> vvchPubKeys;
    if (fEncrypt) {
        vvchPubKeys.push_back(getKey.GetPubKeyBytes());
    }
    else {
        nVersion = 0; // clear text
    }

    CDataRecord record(strOperationType, nTotalSlots, vvchPubKeys, vchValue, nVersion, nExpire, DHT::DataFormat::BinaryBlob);
    if (record.HasError())
        throw JSONRPCError(RPC_DHT_INVALID_RECORD, strprintf("Error creating DHT data entry. %s", record.ErrorMessage()));

    std::string strErrorMessage;
    if (!DHT::SubmitPut(getKey.GetDHTPubKey(), getKey.GetDHTPrivKey(), iSequence, record, strErrorMessage))
        throw JSONRPCError(RPC_DHT_PUT_FAILED, strprintf("Put failed. %s", strErrorMessage));

    result.push_back(Pair("put_seq", iSequence));
    result.push_back(Pair("put_data_size", (int)vchValue.size()));
    return result;
}

static UniValue ClearRecord(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 3)
        throw std::runtime_error(
            "dht clearrecord \"account id\" \"operation\"\n"
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
            HelpExampleCli("dht clearrecord", "duality auth") +
            "\nAs a JSON-RPC call\n" + 
            HelpExampleRpc("dht clearrecord", "duality auth"));

    EnsureWalletIsUnlocked();

    UniValue result(UniValue::VOBJ);

    if (!DHT::SessionStatus())
        throw JSONRPCError(RPC_DHT_NOT_STARTED, strprintf("dht %s failed. DHT session not started.", request.params[0].get_str()));

    if (!CheckDomainEntryDB())
        throw JSONRPCError(RPC_BDAP_DB_ERROR, strprintf("Can not access BDAP domain entry database."));

    CharString vchObjectID = vchFromValue(request.params[1]);
    ToLowerCase(vchObjectID);
    std::string strOperationType = request.params[2].get_str();
    ToLowerCase(strOperationType);

    CDomainEntry entry;
    entry.DomainComponent = vchDefaultDomainName;
    entry.OrganizationalUnit = vchDefaultPublicOU;
    entry.ObjectID = vchObjectID;
    std::string strFullObjectPath = entry.GetFullObjectPath();
    if (!pDomainEntryDB->GetDomainEntryInfo(entry.vchFullObjectPath(), entry))
        throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", strFullObjectPath));

    CKeyEd25519 getKey;
    std::vector<unsigned char> vch = entry.DHTPublicKey;
    CKeyID keyID(Hash160(vch.begin(), vch.end()));
    if (pwalletMain && !pwalletMain->GetDHTKey(keyID, getKey))
        throw JSONRPCError(RPC_DHT_GET_KEY_FAILED, strprintf("Failed to get DHT private key for account %s (pubkey = %s)", entry.GetFullObjectPath(), stringFromVch(vch)));

    if (getKey.GetPubKey() != entry.DHTPublicKey)
        throw JSONRPCError(RPC_DHT_PUBKEY_MISMATCH, strprintf(" Error getting ed25519. Public key from wallet doesn't match entry for %s BDAP entry", strFullObjectPath));

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
    DHT::SubmitGet(0, getKey.GetDHTPubKey(), strHeaderSalt, 2000, strHeaderHex, iSequence, fAuthoritative);
    CRecordHeader header(strHeaderHex);

    if (header.nUnlockTime  > GetTime())
        throw JSONRPCError(RPC_DHT_RECORD_LOCKED, strprintf("DHT data entry is locked for another %lli seconds", (header.nUnlockTime  - GetTime())));

    iSequence++;
    uint16_t nVersion = 0;
    uint32_t nExpire = 0;
    uint16_t nTotalSlots = 32;
    std::vector<unsigned char> vchValue = ZeroCharVector();
    std::vector<std::vector<unsigned char>> vvchPubKeys;
    CDataRecord record(strOperationType, nTotalSlots, vvchPubKeys, vchValue, nVersion, nExpire, DHT::DataFormat::Null);
    if (record.HasError())
        throw JSONRPCError(RPC_DHT_INVALID_RECORD, strprintf("Error creating DHT data entry. %s", record.ErrorMessage()));

    std::string strErrorMessage;
    if (!DHT::SubmitPut(getKey.GetDHTPubKey(), getKey.GetDHTPrivKey(), iSequence, record, strErrorMessage))
        throw JSONRPCError(RPC_DHT_PUT_FAILED, strprintf("Put failed. %s", strErrorMessage));

    result.push_back(Pair("put_seq", iSequence));
    result.push_back(Pair("put_data_size", (int)vchValue.size()));
    return result;
}

static UniValue GetRecord(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 3)
        throw std::runtime_error(
            "dht getrecord \"account id\" \"operation\"\n"
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
           HelpExampleCli("dht getrecord", "Duality avatar") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("dht getrecord", "Duality avatar"));

    EnsureWalletIsUnlocked();

    int64_t nStart = GetTimeMillis();
    UniValue result(UniValue::VOBJ);
   
    if (!DHT::SessionStatus())
        throw JSONRPCError(RPC_DHT_NOT_STARTED, strprintf("dht %s failed. DHT session not started.", request.params[0].get_str()));

    if (!CheckDomainEntryDB())
        throw JSONRPCError(RPC_BDAP_DB_ERROR, strprintf("Can not access BDAP domain entry database."));

    CharString vchObjectID = vchFromValue(request.params[1]);
    ToLowerCase(vchObjectID);
    CDomainEntry entry;
    entry.DomainComponent = vchDefaultDomainName;
    entry.OrganizationalUnit = vchDefaultPublicOU;
    entry.ObjectID = vchObjectID;
    std::string strFullObjectPath = entry.GetFullObjectPath();
    if (!pDomainEntryDB->GetDomainEntryInfo(entry.vchFullObjectPath(), entry))
        throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", strFullObjectPath));

    const std::string strOperationType = request.params[2].get_str();
    std::string strPubKey = stringFromVch(entry.DHTPublicKey);
    result.push_back(Pair("entry_path", strFullObjectPath));
    result.push_back(Pair("wallet_address", stringFromVch(entry.WalletAddress)));
    result.push_back(Pair("link_address", stringFromVch(entry.LinkAddress)));
    result.push_back(Pair("get_pubkey", strPubKey));
    result.push_back(Pair("get_operation", strOperationType));
    std::vector<unsigned char> vchDHTPublicKey = entry.DHTPublicKey;
    CKeyEd25519 getKey(true);
    CKeyID keyID(Hash160(vchDHTPublicKey.begin(), vchDHTPublicKey.end()));
    if (!pwalletMain->GetDHTKey(keyID, getKey)) {
        getKey.SetNull();
    }

    int64_t iSequence = 0;
    std::array<char, 32> arrPubKey;
    libtorrent::aux::from_hex(strPubKey, arrPubKey.data());
    CDataRecord record;
    if (!DHT::SubmitGetRecord(0, arrPubKey, getKey.GetDHTPrivSeed(), strOperationType, iSequence, record))
        throw JSONRPCError(RPC_DHT_GET_FAILED, strprintf("Failed to get record"));

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
        throw JSONRPCError(RPC_BDAP_SPORK_INACTIVE, strprintf("Can not use the DHT until the BDAP spork is active."));

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
        throw JSONRPCError(RPC_BDAP_SPORK_INACTIVE, strprintf("Can not use the DHT until the BDAP spork is active."));

    UniValue result(UniValue::VOBJ);

    std::vector<CMutableGetEvent> vchMutableData;
    bool fRet = DHT::GetAllDHTGetEvents(0, vchMutableData);
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

static UniValue GetLinkRecord(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 4)
        throw std::runtime_error(
            "dht getlinkrecord \"account1\" \"account2\" \"operation\"\n"
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
           HelpExampleCli("dht getlinkrecord", "duality bob auth") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("dht getlinkrecord", "duality bob auth"));

    EnsureWalletIsUnlocked();

    int64_t nStart = GetTimeMillis();
    UniValue result(UniValue::VOBJ);

    if (!DHT::SessionStatus())
        throw JSONRPCError(RPC_DHT_NOT_STARTED, strprintf("dht %s failed. DHT session not started.", request.params[0].get_str()));

    if (!CheckDomainEntryDB())
        throw JSONRPCError(RPC_BDAP_DB_ERROR, strprintf("Can not access BDAP domain entry database."));

    if (!pLinkManager)
        throw JSONRPCError(RPC_BDAP_LINK_MNGR_ERROR, strprintf("Can not open link request in memory map"));

    CharString vchObjectID1 = vchFromValue(request.params[1]);
    ToLowerCase(vchObjectID1);
    CDomainEntry entry1;
    entry1.DomainComponent = vchDefaultDomainName;
    entry1.OrganizationalUnit = vchDefaultPublicOU;
    entry1.ObjectID = vchObjectID1;
    if (!pDomainEntryDB->GetDomainEntryInfo(entry1.vchFullObjectPath(), entry1))
        throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", entry1.GetFullObjectPath()));

    CharString vchObjectID2 = vchFromValue(request.params[2]);
    ToLowerCase(vchObjectID2);
    CDomainEntry entry2;
    entry2.DomainComponent = vchDefaultDomainName;
    entry2.OrganizationalUnit = vchDefaultPublicOU;
    entry2.ObjectID = vchObjectID2;
    if (!pDomainEntryDB->GetDomainEntryInfo(entry2.vchFullObjectPath(), entry2))
        throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", entry2.GetFullObjectPath()));

    std::string strOperationType = request.params[3].get_str();
    ToLowerCase(strOperationType);

    CLink link;
    std::string strPubKey;
    std::vector<unsigned char> vchDHTPublicKey;
    uint256 linkID = GetLinkID(entry1.GetFullObjectPath(), entry2.GetFullObjectPath());
    if (!pLinkManager->FindLink(linkID, link))
        throw JSONRPCError(RPC_BDAP_LINK_MNGR_ERROR, strprintf("Link id %s for %s and %s not found. Get BDAP link failed!", linkID.ToString(), entry1.GetFullObjectPath(), entry2.GetFullObjectPath()));

    CKeyEd25519 getKey;
    if (entry1.GetFullObjectPath() == link.RequestorFQDN()) {
        strPubKey = link.RequestorPubKeyString();
        vchDHTPublicKey = link.RequestorPubKey;
        CKeyID keyID(Hash160(vchDHTPublicKey.begin(), vchDHTPublicKey.end()));
        if (pwalletMain && !pwalletMain->GetDHTKey(keyID, getKey)) {
            std::vector<unsigned char> vchPublicKey = link.RecipientPubKey;
            CKeyID keyID2(Hash160(vchPublicKey.begin(), vchPublicKey.end()));
            if (pwalletMain && !pwalletMain->GetDHTKey(keyID2, getKey))
                throw JSONRPCError(RPC_DHT_GET_KEY_FAILED, strprintf("Failed to get DHT private key for account %s (pubkey = %s)", link.RequestorFQDN(), stringFromVch(vchDHTPublicKey)));
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
                throw JSONRPCError(RPC_DHT_GET_KEY_FAILED, strprintf("DHT link private key for %s or %s not found.", entry1.GetFullObjectPath(), entry2.GetFullObjectPath()));
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
    if (!DHT::SubmitGetRecord(0, arrPubKey, getKey.GetDHTPrivSeed(), strOperationType, iSequence, record))
        throw JSONRPCError(RPC_DHT_GET_FAILED, strprintf("Failed to get record"));

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

static UniValue GetAllLinkRecords(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 3)
        throw std::runtime_error(
            "dht getalllinkrecords \"account\" \"operation\"\n"
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
           HelpExampleCli("dht getalllinkrecords", "duality pshare-offer") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("dht getalllinkrecords", "duality pshare-offer"));

    EnsureWalletIsUnlocked();

    int64_t nStart = GetTimeMillis();

    UniValue results(UniValue::VOBJ);
   
    if (!DHT::SessionStatus())
        throw JSONRPCError(RPC_DHT_NOT_STARTED, strprintf("dht %s failed. DHT session not started.", request.params[0].get_str()));

    if (!CheckDomainEntryDB())
        throw JSONRPCError(RPC_BDAP_DB_ERROR, strprintf("Can not access BDAP domain entry database."));

    if (!pLinkManager)
        throw JSONRPCError(RPC_BDAP_LINK_MNGR_ERROR, strprintf("Can not open link request in memory map"));

    CharString vchObjectID = vchFromValue(request.params[1]);
    ToLowerCase(vchObjectID);
    CDomainEntry entry;
    entry.DomainComponent = vchDefaultDomainName;
    entry.OrganizationalUnit = vchDefaultPublicOU;
    entry.ObjectID = vchObjectID;
    if (!pDomainEntryDB->GetDomainEntryInfo(entry.vchFullObjectPath(), entry))
        throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", entry.GetFullObjectPath()));

    std::string strOperationType = request.params[2].get_str();
    ToLowerCase(strOperationType);

    std::vector<CLinkInfo> vchLinkInfo = pLinkManager->GetCompletedLinkInfo(entry.vchFullObjectPath());
    for (CLinkInfo& linkInfo : vchLinkInfo) {
        CKeyEd25519 getKey;
        std::vector<unsigned char> vchDHTPublicKey = linkInfo.vchReceivePubKey;
        CKeyID keyID(Hash160(vchDHTPublicKey.begin(), vchDHTPublicKey.end()));
        if (pwalletMain && !pwalletMain->GetDHTKey(keyID, getKey))
            throw JSONRPCError(RPC_DHT_GET_KEY_FAILED, strprintf("Failed to get DHT private key for account %s (pubkey = %s)", stringFromVch(linkInfo.vchFullObjectPath), stringFromVch(vchDHTPublicKey)));

       linkInfo.arrReceivePrivateSeed = getKey.GetDHTPrivSeed();
    }

    std::vector<CDataRecord> vchRecords;
    if (!DHT::SubmitGetAllRecordsSync(0, vchLinkInfo, strOperationType, vchRecords))
        throw JSONRPCError(RPC_DHT_GET_FAILED, strprintf("Failed to get records"));

    int nRecordItem = 1;
    for (CDataRecord& record : vchRecords) // loop through records
    {
        UniValue result(UniValue::VOBJ);
        result.push_back(Pair("account", stringFromVch(record.vchOwnerFQDN)));
        result.push_back(Pair("get_operation", strOperationType));
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

static UniValue PutLinkRecord(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() < 4 || request.params.size() > 6)
        throw std::runtime_error(
            "dht putlinkrecord \"account1\" \"account2\" \"operation\" \"value\"\n"
            "\nSaves a link mutable data entry in the DHT for a BDAP entry and operation code.\n"
            "\nArguments:\n"
            "1. account1               (string)           BDAP link account 1, gets the value for this account\n"
            "2. account2               (string)           BDAP link account 2, other account in the link\n"
            "3. operation              (string)           Mutable data operation code\n"
            "4. value                  (string)           Mutable data value to save in the DHT\n"
            "5. encrypt                (bool, optional)   Encrypt the data with the user's public key\n"
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
           HelpExampleCli("dht putlinkrecord", "duality bob auth \"save this auth data\"") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("dht putlinkrecord", "duality bob auth \"save this auth data\""));

    EnsureWalletIsUnlocked();

    UniValue result(UniValue::VOBJ);

    if (!DHT::SessionStatus())
        throw JSONRPCError(RPC_DHT_NOT_STARTED, strprintf("dht %s failed. DHT session not started.", request.params[0].get_str()));

    if (!CheckDomainEntryDB())
        throw JSONRPCError(RPC_BDAP_DB_ERROR, strprintf("Can not access BDAP domain entry database."));

    if (!pLinkManager)
        throw JSONRPCError(RPC_BDAP_LINK_MNGR_ERROR, strprintf("Can not open link request in memory map"));

    CharString vchObjectID1 = vchFromValue(request.params[1]);
    ToLowerCase(vchObjectID1);
    CDomainEntry entry1;
    entry1.DomainComponent = vchDefaultDomainName;
    entry1.OrganizationalUnit = vchDefaultPublicOU;
    entry1.ObjectID = vchObjectID1;
    if (!pDomainEntryDB->GetDomainEntryInfo(entry1.vchFullObjectPath(), entry1))
        throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", entry1.GetFullObjectPath()));

    CharString vchObjectID2 = vchFromValue(request.params[2]);
    ToLowerCase(vchObjectID2);
    CDomainEntry entry2;
    entry2.DomainComponent = vchDefaultDomainName;
    entry2.OrganizationalUnit = vchDefaultPublicOU;
    entry2.ObjectID = vchObjectID2;
    if (!pDomainEntryDB->GetDomainEntryInfo(entry2.vchFullObjectPath(), entry2))
        throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", entry2.GetFullObjectPath()));

    std::string strOperationType = request.params[3].get_str();
    ToLowerCase(strOperationType);

    CLink link;
    uint256 linkID = GetLinkID(entry1.GetFullObjectPath(), entry2.GetFullObjectPath());
    if (!pLinkManager->FindLink(linkID, link))
        throw JSONRPCError(RPC_BDAP_LINK_MNGR_ERROR, strprintf("Link id %s for %s and %s not found. Get BDAP link failed!", linkID.ToString(), entry1.GetFullObjectPath(), entry2.GetFullObjectPath()));

    std::vector<unsigned char> vchDHTPublicKey;
    if (entry1.GetFullObjectPath() == link.RequestorFQDN()) {
        vchDHTPublicKey = link.RequestorPubKey;
    }
    else if (entry1.GetFullObjectPath() == link.RecipientFQDN()) {
        vchDHTPublicKey = link.RecipientPubKey;
    }
    else {
        throw JSONRPCError(RPC_BDAP_LINK_MNGR_ERROR, strprintf("DHT link public key for %s not found", entry1.GetFullObjectPath()));
    }
    result.push_back(Pair("link_requestor", link.RequestorFQDN()));
    result.push_back(Pair("link_acceptor", link.RecipientFQDN()));
    result.push_back(Pair("put_pubkey", stringFromVch(vchDHTPublicKey)));
    result.push_back(Pair("put_operation", strOperationType));


    CKeyEd25519 getKey;
    CKeyID keyID(Hash160(vchDHTPublicKey.begin(), vchDHTPublicKey.end()));
    if (pwalletMain && !pwalletMain->GetDHTKey(keyID, getKey))
        throw JSONRPCError(RPC_DHT_GET_KEY_FAILED, strprintf("Error getting ed25519 private key for the %s BDAP entry", entry1.GetFullObjectPath()));

    int64_t iSequence = 0;
    bool fAuthoritative = false;
    std::string strHeaderHex;

    // we need the last sequence number to update an existing DHT entry.
    std::string strHeaderSalt = strOperationType + ":" + std::to_string(0);
    DHT::SubmitGet(0, getKey.GetDHTPubKey(), strHeaderSalt, 2000, strHeaderHex, iSequence, fAuthoritative);
    CRecordHeader header(strHeaderHex);
    if (header.nUnlockTime  > GetTime())
        throw JSONRPCError(RPC_DHT_RECORD_LOCKED, strprintf("DHT data entry is locked for another %lli seconds", (header.nUnlockTime  - GetTime())));

    iSequence++;

    uint16_t nVersion = 1; //TODO (DHT): Default is encrypted but add parameter for use cases where we want clear text.
    uint32_t nExpire = GetTime() + 2592000; // TODO (DHT): Default to 30 days but add an expiration date parameter.
    uint16_t nTotalSlots = 32;
    std::vector<unsigned char> vchValue = vchFromValue(request.params[4]);
    std::vector<std::vector<unsigned char>> vvchPubKeys;
    bool fEncrypt = true;
    if (request.params.size() > 5)
        fEncrypt = request.params[5].getBool();

    if (fEncrypt) {
        vvchPubKeys.push_back(EncodedPubKeyToBytes(link.RequestorPubKey));
        vvchPubKeys.push_back(EncodedPubKeyToBytes(link.RecipientPubKey));
    }
    else {
        nVersion = 0; // clear text
    }

    CDataRecord record(strOperationType, nTotalSlots, vvchPubKeys, vchValue, nVersion, nExpire, DHT::DataFormat::BinaryBlob);
    if (record.HasError())
        throw JSONRPCError(RPC_DHT_INVALID_RECORD, strprintf("Error creating DHT data entry. %s", record.ErrorMessage()));

    std::string strErrorMessage;
    if (!DHT::SubmitPut(getKey.GetDHTPubKey(), getKey.GetDHTPrivKey(), iSequence, record, strErrorMessage))
        throw JSONRPCError(RPC_DHT_PUT_FAILED, strprintf("Put failed. %s", strErrorMessage));

    result.push_back(Pair("put_seq", iSequence));
    result.push_back(Pair("put_data_size", (int)vchValue.size()));

    return result;
}

static UniValue ClearLinkRecord(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 4)
        throw std::runtime_error(
            "dht clearlinkrecord \"account1\" \"account2\" \"operation\"\n"
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
           HelpExampleCli("dht clearlinkrecord", "duality bob auth") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("dht clearlinkrecord", "duality bob auth"));

    EnsureWalletIsUnlocked();

    UniValue result(UniValue::VOBJ);

    if (!DHT::SessionStatus())
        throw JSONRPCError(RPC_DHT_NOT_STARTED, strprintf("dht %s failed. DHT session not started.", request.params[0].get_str()));

    if (!CheckDomainEntryDB())
        throw JSONRPCError(RPC_BDAP_DB_ERROR, strprintf("Can not access BDAP domain entry database."));

    if (!pLinkManager)
        throw JSONRPCError(RPC_BDAP_LINK_MNGR_ERROR, strprintf("Can not open link request in memory map"));

    CharString vchObjectID1 = vchFromValue(request.params[1]);
    ToLowerCase(vchObjectID1);
    CDomainEntry entry1;
    entry1.DomainComponent = vchDefaultDomainName;
    entry1.OrganizationalUnit = vchDefaultPublicOU;
    entry1.ObjectID = vchObjectID1;
    if (!pDomainEntryDB->GetDomainEntryInfo(entry1.vchFullObjectPath(), entry1))
        throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", entry1.GetFullObjectPath()));

    CharString vchObjectID2 = vchFromValue(request.params[2]);
    ToLowerCase(vchObjectID2);
    CDomainEntry entry2;
    entry2.DomainComponent = vchDefaultDomainName;
    entry2.OrganizationalUnit = vchDefaultPublicOU;
    entry2.ObjectID = vchObjectID2;
    if (!pDomainEntryDB->GetDomainEntryInfo(entry2.vchFullObjectPath(), entry2))
        throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", entry2.GetFullObjectPath()));

    std::string strOperationType = request.params[3].get_str();
    ToLowerCase(strOperationType);

    CLink link;
    uint256 linkID = GetLinkID(entry1.GetFullObjectPath(), entry2.GetFullObjectPath());
    if (!pLinkManager->FindLink(linkID, link))
        throw JSONRPCError(RPC_BDAP_LINK_MNGR_ERROR, strprintf("Link id %s for %s and %s not found. Get BDAP link failed!", linkID.ToString(), entry1.GetFullObjectPath(), entry2.GetFullObjectPath()));

    LogPrintf("%s -- req: %s, rec: %s \n", __func__, link.RequestorFQDN(), link.RecipientFQDN());
    std::vector<unsigned char> vchDHTPublicKey;
    if (entry1.GetFullObjectPath() == link.RequestorFQDN()) {
        vchDHTPublicKey = link.RequestorPubKey;
    }
    else if (entry1.GetFullObjectPath() == link.RecipientFQDN()) {
        vchDHTPublicKey = link.RecipientPubKey;
    }
    else {
        throw JSONRPCError(RPC_BDAP_LINK_MNGR_ERROR, strprintf("DHT link public key for %s not found", entry1.GetFullObjectPath()));
    }
    result.push_back(Pair("link_requestor", link.RequestorFQDN()));
    result.push_back(Pair("link_acceptor", link.RecipientFQDN()));
    result.push_back(Pair("put_pubkey", stringFromVch(vchDHTPublicKey)));
    result.push_back(Pair("put_operation", strOperationType));


    CKeyEd25519 getKey;
    CKeyID keyID(Hash160(vchDHTPublicKey.begin(), vchDHTPublicKey.end()));
    if (pwalletMain && !pwalletMain->GetDHTKey(keyID, getKey))
        throw JSONRPCError(RPC_DHT_GET_KEY_FAILED, strprintf("Error getting ed25519 private key for the %s BDAP entry", entry1.GetFullObjectPath()));

    int64_t iSequence = 0;
    bool fAuthoritative = false;
    std::string strHeaderHex;

    // we need the last sequence number to update an existing DHT entry.
    std::string strHeaderSalt = strOperationType + ":" + std::to_string(0);
    DHT::SubmitGet(0, getKey.GetDHTPubKey(), strHeaderSalt, 2000, strHeaderHex, iSequence, fAuthoritative);
    CRecordHeader header(strHeaderHex);

    if (header.nUnlockTime  > GetTime())
        throw JSONRPCError(RPC_DHT_RECORD_LOCKED, strprintf("DHT data entry is locked for another %lli seconds", (header.nUnlockTime  - GetTime())));

    iSequence++;

    uint16_t nVersion = 0;
    uint32_t nExpire = 0;
    uint16_t nTotalSlots = 32;
    std::vector<unsigned char> vchValue = ZeroCharVector();
    std::vector<std::vector<unsigned char>> vvchPubKeys;
    CDataRecord record(strOperationType, nTotalSlots, vvchPubKeys, vchValue, nVersion, nExpire, DHT::DataFormat::Null);
    if (record.HasError())
        throw JSONRPCError(RPC_DHT_INVALID_RECORD, strprintf("Error creating DHT data entry. %s", record.ErrorMessage()));

    std::string strErrorMessage;
    if (!DHT::SubmitPut(getKey.GetDHTPubKey(), getKey.GetDHTPrivKey(), iSequence, record, strErrorMessage))
        throw JSONRPCError(RPC_DHT_PUT_FAILED, strprintf("Put failed. %s", strErrorMessage));

    result.push_back(Pair("put_seq", iSequence));
    result.push_back(Pair("put_data_size", (int)vchValue.size()));

    return result;
}

static UniValue ReannounceLocalMutable(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 2)
        throw std::runtime_error(
            "dht reannounce \"infohash\"\n"
            "\nReannounces signed mutable item from local leveldb.\n"
            "\nArguments:\n"
            "1. infohash               (string)      DHT Infohash reannouncing\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"link_requestor\"      (string)      BDAP account that initiated the link\n"
            "}\n"
            "\nExamples\n" +
           HelpExampleCli("dht reannounce", "\"88196b9f8ca5f1dfb095bd48e18d97157f7a4435\"") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("dht reannounce", "\"88196b9f8ca5f1dfb095bd48e18d97157f7a4435\""));

    if (!DHT::SessionStatus())
        throw JSONRPCError(RPC_DHT_NOT_STARTED, strprintf("dht %s failed. DHT session not started.", request.params[0].get_str()));

    if (!CheckMutableItemDB())
        throw JSONRPCError(RPC_BDAP_DB_ERROR, strprintf("Can not access mutable data item database."));

    std::string strInfoHash = request.params[1].get_str();
    CharString vchInfoHash = vchFromString(strInfoHash);

    UniValue result(UniValue::VOBJ);

    CMutableData mutableData;
    if (!GetLocalMutableData(vchInfoHash, mutableData))
        throw JSONRPCError(RPC_BDAP_DB_ERROR, strprintf("Mutable data infohash %s not found in local leveldb.", strInfoHash));

    if (!DHT::ReannounceEntry(mutableData))
        result.push_back(Pair("error", "DHT re-announce failed."));

    result.push_back(Pair("target_hash", mutableData.InfoHash()));
    result.push_back(Pair("public_key", mutableData.PublicKey()));
    result.push_back(Pair("salt", mutableData.Salt()));
    result.push_back(Pair("signature", mutableData.Signature()));
    result.push_back(Pair("value", mutableData.Value()));

    return result;
}

static UniValue GetHashTableEvents(const JSONRPCRequest& request)
{
    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "dht events\n"
            "\nReturns DHT events.\n"
            "\nResult:\n"
            "{(json object)\n"
            "  \"link_requestor\"      (string)      BDAP account that initiated the link\n"
            "}\n"
            "\nExamples\n" +
           HelpExampleCli("dht reannounce", "\"88196b9f8ca5f1dfb095bd48e18d97157f7a4435\"") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("dht reannounce", "\"88196b9f8ca5f1dfb095bd48e18d97157f7a4435\""));

    if (!DHT::SessionStatus())
        throw JSONRPCError(RPC_DHT_NOT_STARTED, strprintf("dht %s failed. DHT session not started.", request.params[0].get_str()));

    UniValue results(UniValue::VOBJ);
    std::vector<CEvent> events;
    DHT::GetEvents(0, events);
    size_t nCount = 0;
    for (const CEvent& event : events) {
        nCount++;
        UniValue oEventItem(UniValue::VOBJ);
        oEventItem.push_back(Pair("message", event.Message()));
        oEventItem.push_back(Pair("type", (int)event.Type()));
        oEventItem.push_back(Pair("category", (int)event.Category()));
        oEventItem.push_back(Pair("what", event.What()));
        oEventItem.push_back(Pair("timestamp", (int)event.Timestamp()));
        results.push_back(Pair("event_" + std::to_string(nCount), oEventItem));
    }

    return results;
}

UniValue dht_rpc(const JSONRPCRequest& request) 
{
    std::string strCommand;
    if (request.params.size() >= 1) {
        strCommand = request.params[0].get_str();
        ToLowerCase(strCommand);
    }
    else {
        throw std::runtime_error(
            "dht \"command\"...\n"
            "\nAvailable commands:\n"
            "  getmutable         - Get mutable entry\n"
            "  putmutable         - Put mutable entry\n"
            "  getrecord          - Get DHT record\n"
            "  putrecord          - Put DHT record\n"
            "  clearrecord        - Clear DHT record\n"
            "  getlinkrecord      - Get DHT link record\n"
            "  putlinkrecord      - Put DHT link record\n"
            "  clearlinkrecord    - Clear DHT link record\n"
            "  getalllinkrecords  - Get all DHT link records\n"
            "  status             - DHT status\n"
            "\nExamples:\n"
            + HelpExampleCli("dht getrecord", "superman avatar") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("dht getrecord", "superman avatar"));
    }
    if (strCommand == "getmutable" || strCommand == "putmutable" || 
            strCommand == "getrecord" || strCommand == "putrecord" || strCommand == "clearrecord" || 
            strCommand == "getlinkrecord" || strCommand == "putlinkrecord" || strCommand == "clearlinkrecord" || strCommand == "getalllinkrecords" ||
            strCommand == "status" || strCommand == "reannounce" || strCommand == "events") 
    {
        if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
            throw JSONRPCError(RPC_BDAP_SPORK_INACTIVE, strprintf("Can not use the DHT until the BDAP spork is active."));
    }
    if (strCommand == "getmutable") {
        return GetMutable(request);
    }
    else if (strCommand == "putmutable") {
        return PutMutable(request);
    }
    else if (strCommand == "getrecord") {
        return GetRecord(request);
    }
    else if (strCommand == "putrecord") {
        return PutRecord(request);
    }
    else if (strCommand == "clearrecord") {
        return ClearRecord(request);
    }
    else if (strCommand == "getlinkrecord") {
        return GetLinkRecord(request);
    }
    else if (strCommand == "putlinkrecord") {
        return PutLinkRecord(request);
    }
    else if (strCommand == "clearlinkrecord") {
        return ClearLinkRecord(request);
    }
    else if (strCommand == "getalllinkrecords") {
        return GetAllLinkRecords(request);
    }
    else if (strCommand == "status") {
        return GetDHTStatus(request);
    }
    else if (strCommand == "reannounce") {
        return ReannounceLocalMutable(request);
    }
    else if (strCommand == "events") {
        return GetHashTableEvents(request);
    }
    else {
        throw JSONRPCError(RPC_METHOD_NOT_FOUND, strprintf("%s is an unknown DHT method command.", strCommand));
    }
    return NullUniValue;
}

static const CRPCCommand commands[] =
{ //  category              name                     actor (function)               okSafe   argNames
  //  --------------------- ------------------------ -----------------------        ------   --------------------
    /* DHT */
    { "dht",             "dht",                      &dht_rpc,                      true,    {"command", "param1", "param2", "param3"}  },
    { "dht",             "dhtdb",                    &dhtdb,                        true,    {} },
    { "dht",             "dhtputmessages",           &dhtputmessages,               true,    {} },
    { "dht",             "dhtgetmessages",           &dhtgetmessages,               true,    {} },
};

void RegisterDHTRPCCommands(CRPCTable &tableRPC)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        tableRPC.appendCommand(commands[vcidx].name, &commands[vcidx]);
}
