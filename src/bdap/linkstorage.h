// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_LINKSTORAGE_H
#define DYNAMIC_BDAP_LINKSTORAGE_H

#include "serialize.h"
#include "uint256.h"

#include <vector>

// Used to store account FQDN and public key
typedef std::pair<std::vector<unsigned char>, std::vector<unsigned char>> LinkInfo;

namespace BDAP {

    enum LinkType : std::uint8_t
    {
        UnknownType = 0,
        RequestType = 1,
        AcceptType = 2
    };
}

class CLinkStorage {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    std::vector<unsigned char> vchRawData;
    std::vector<unsigned char> vchLinkPubKey;
    std::vector<unsigned char> vchSharedPubKey;
    uint8_t nType;
    uint64_t nHeight;
    uint64_t nExpireTime;
    uint64_t nTime;
    uint256 txHash;

    CLinkStorage() 
    {
        SetNull();
    }

    CLinkStorage(const std::vector<unsigned char>& data, const std::vector<unsigned char>& pubkey, const std::vector<unsigned char>& sharedPubkey, const uint8_t type, 
                    const uint64_t& height, const uint64_t& expire, const uint64_t& time, const uint256& txid)
                    : vchRawData(data), vchLinkPubKey(pubkey), vchSharedPubKey(sharedPubkey), nType(type), nHeight(height), nExpireTime(expire), nTime(time), txHash(txid) 
    {
    }

    inline void SetNull()
    {
        nVersion = CLinkStorage::CURRENT_VERSION;
        vchRawData.clear();
        vchLinkPubKey.clear();
        vchSharedPubKey.clear();
        nType = 0;
        nHeight = 0;
        nExpireTime = 0;
        nTime = 0;
        txHash.SetNull();
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(vchRawData);
        READWRITE(vchLinkPubKey);
        READWRITE(vchSharedPubKey);
        READWRITE(VARINT(nType));
        READWRITE(VARINT(nHeight));
        READWRITE(VARINT(nExpireTime));
        READWRITE(VARINT(nTime));
        READWRITE(txHash);
    }

    inline friend bool operator==(const CLinkStorage& a, const CLinkStorage& b) {
        return (a.vchRawData == b.vchRawData && a.txHash == b.txHash);
    }

    inline friend bool operator!=(const CLinkStorage& a, const CLinkStorage& b) {
        return !(a == b);
    }

    inline CLinkStorage operator=(const CLinkStorage& b) {
        nVersion = b.nVersion;
        vchRawData = b.vchRawData;
        vchLinkPubKey = b.vchLinkPubKey;
        vchSharedPubKey = b.vchSharedPubKey;
        nType = b.nType;
        nHeight = b.nHeight;
        nExpireTime = b.nExpireTime;
        nTime = b.nTime;
        txHash = b.txHash;
        return *this;
    }

    bool UnserializeFromData(const std::vector<unsigned char>& vchData);
    void Serialize(std::vector<unsigned char>& vchData);

    inline bool IsNull() const { return (vchRawData.empty()); }
    bool Encrypted() const;
    int DataVersion() const;

};

void ProcessLink(const CLinkStorage& storage, const bool fStoreInQueueOnly = false);
void ProcessLinkQueue();

#endif // DYNAMIC_BDAP_LINKSTORAGE_H