// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_GLOBALDATA_H
#define DYNAMIC_BDAP_GLOBALDATA_H

#include "serialize.h"
#include "sync.h"
#include "uint256.h"

#include <stdint.h>
#include <string>
#include <vector>

static constexpr size_t MAX_AVATAR_DATA_SIZE = 320;
static constexpr size_t MAX_SCIM_PROFILE_DATA = 2560;
static constexpr size_t MAX_GLOBAL_MESSAGE_SIGNATURE_SIZE = 72;
static constexpr size_t MAX_GLOBAL_MESSAGE_DATA_LENGTH = 2700;
static constexpr int RELAY_KEEP_ALIVE_SECONDS = 30;
static constexpr int MAX_GLOBAL_MESSAGE_DRIFT_SECONDS = 90;
static constexpr int MIN_GLOBAL_DATA_PEER_PROTO_VERSION = 71300; // TODO (BDAP): Update minimum protocol version before v2.4 release

class CConnman;
class CDynamicAddress;
class CKey;
class CNode;
class CPubKey;

class CGlobalData
{
public:
    static constexpr int CURRENT_VERSION = 1;

    int nVersion;
    int64_t nTimeStamp;
    std::uint8_t nType;
    std::vector<unsigned char> vchWalletPubKey;
    std::vector<unsigned char> vchFQDN;
    std::vector<unsigned char> vchData;

    CGlobalData()
    {
        SetNull();
    }

    CGlobalData(const std::vector<unsigned char>& vchData)
    {
       UnserializeFromData(vchData);
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(this->nVersion);
        READWRITE(nTimeStamp);
        READWRITE(nType);
        READWRITE(vchWalletPubKey);
        READWRITE(vchFQDN);
        READWRITE(vchData);
    }

    void SetNull()
    {
        nVersion = CURRENT_VERSION;
        nTimeStamp = 0;
        nType = 0;
        vchWalletPubKey.clear();
        vchFQDN.clear();
        vchData.clear();
    }

    std::string ToString() const;

    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData);

    CPubKey GetPubKey() const;
    CDynamicAddress WalletAddress() const;

};

/**
Global BDAP avatar formats can use a URL, IPFS filename or torrent file
**/
class CGlobalAvatar
{
public:
    static constexpr int CURRENT_VERSION = 1;

    enum AvatarType : std::uint8_t {
       URL = 1,
       IPFS = 2,
       TORRENT_FILE = 3
    };

    std::vector<unsigned char> vchGlobalData;
    std::vector<unsigned char> vchSignature;

    CGlobalAvatar()
    {
        SetNull();
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(vchGlobalData);
        READWRITE(vchSignature);
    }

    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData);

    void SetNull()
    {
        vchGlobalData.clear();
        vchSignature.clear();
    }

    std::string TypeName() const;
    uint256 GetHash() const;
    bool RelayMessage(CConnman& connman) const;
    bool Sign(const CKey& key);
    bool CheckSignature(const std::vector<unsigned char>& vchPubKey) const;
    int ProcessMessage(std::string& strErrorMessage) const;
    bool RelayTo(CNode* pnode, CConnman& connman) const;

};

/**
Global BDAP identity profiles use the SCIM v2 IETF standard format
- System for Cross-domain Identity Management (SCIM) v2
- https://tools.ietf.org/html/rfc7643
- http://www.simplecloud.info
**/
class CGlobalProfile
{
public:
    static constexpr int CURRENT_VERSION = 1;

    enum ProfileType : std::uint8_t {
       SCIM1 = 65,
       SCIM2 = 66
    };

    std::vector<unsigned char> vchGlobalData;
    std::vector<unsigned char> vchSignature;

    CGlobalProfile()
    {
        SetNull();
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(vchGlobalData);
        READWRITE(vchSignature);
    }

    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData);

    void SetNull()
    {
        vchGlobalData.clear();
        vchSignature.clear();
    }

    uint256 GetHash() const;
    bool RelayMessage(CConnman& connman) const;
    bool Sign(const CKey& key);
    bool CheckSignature(const std::vector<unsigned char>& vchPubKey) const;
    int ProcessMessage(std::string& strErrorMessage) const;
    bool RelayTo(CNode* pnode, CConnman& connman) const;

};

bool ReceivedGlobalDataMessage(const uint256& messageHash);

#endif // DYNAMIC_BDAP_GLOBALDATA_H
