// Copyright (c) 2018 Duality Blockchain Solutions Developers 
// TODO: Add License

#include "bdap/domainentry.h"
#include "dht/bootstrap.h"
#include "dht/keyed25519.h"
#include "libtorrent/hex.hpp" // for to_hex and from_hex
#include "rpcprotocol.h"
#include "rpcserver.h"
#include "util.h"
#include "utilstrencodings.h"

#include <univalue.h>


UniValue getdhtmutable(const JSONRPCRequest& request)
{
    if (request.params.size() != 2)
        throw std::runtime_error(
            "getdhtdata\n"
            "\n");

    UniValue result(UniValue::VOBJ);
    if (!pTorrentDHTSession)
        return result;

    const std::string strPubKey = request.params[0].get_str();
    const std::string strSalt = request.params[1].get_str();

    bool fRet = false;
    int64_t iSequence = 0;
    std::string strValue = "";
    std::array<char, 32> pubKey;
    libtorrent::aux::from_hex(strPubKey, pubKey.data());
    fRet = GetDHTMutableData(pubKey, strSalt, strValue, iSequence);
    if (fRet) {
        result.push_back(Pair("Get_PubKey", strPubKey));
        result.push_back(Pair("Get_Salt", strSalt));
        result.push_back(Pair("Get_Seq", iSequence));
        result.push_back(Pair("Get_Value", strValue));
    }

    return result;
}

UniValue putdhtmutable(const JSONRPCRequest& request)
{
    if (request.params.size() < 2 || request.params.size() > 4 || request.params.size() == 3)
        throw std::runtime_error(
            "putdhtdata\n"
            "\n");

    UniValue result(UniValue::VOBJ);
    if (!pTorrentDHTSession)
        return result;

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
    }

    bool fRet = false;
    int64_t iSequence = 0;
    std::string strPutValue = "";
    std::array<char, 32> pubKey;
    libtorrent::aux::from_hex(strPubKey, pubKey.data());

    std::array<char, 64> privKey;
    libtorrent::aux::from_hex(strPrivKey, privKey.data());

    fRet = GetDHTMutableData(pubKey, strSalt, strPutValue, iSequence);
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

    return result;
}

static const CRPCCommand commands[] =
{   //  category         name                        actor (function)           okSafeMode
    /* DHT */
    { "dht",             "getdhtmutable",            &getdhtmutable,                true  },
    { "dht",             "putdhtmutable",            &putdhtmutable,                true  },
};

void RegisterDHTRPCCommands(CRPCTable &tableRPC)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        tableRPC.appendCommand(commands[vcidx].name, &commands[vcidx]);
}