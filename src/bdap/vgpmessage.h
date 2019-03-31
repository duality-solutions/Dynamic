// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_RELAYMESSAGE_H
#define DYNAMIC_BDAP_RELAYMESSAGE_H

#include "serialize.h"
#include "sync.h"
#include "uint256.h"

#include <map>
#include <set>
#include <stdint.h>
#include <vector>

class CRelayMessage;
class CConnman;
class CNode;
class uint256;

extern std::map<uint256, CRelayMessage> mapRelayMessages;
extern CCriticalSection cs_mapRelayMessages;

class CUnsignedRelayMessage
{
public:
    static constexpr uint32_t MIN_CLIENT_VERSION = 2041400; // TODO (BDAP): Update minimum client version before v2.4 release
    static constexpr uint32_t MIN_PROTOCOL_VERSION = 71200; // TODO (BDAP): Update minimum protocol version before v2.4 release
    static constexpr size_t MAX_MESSAGE_DATA_LENGTH = 8192;
    static const int CURRENT_VERSION = 1;
    int nVersion;
    uint256 SubjectID;
    bool fEncrypted;
    uint16_t nDataFormatVersion;
    std::vector<unsigned char> vchRelayWallet;
    std::vector<unsigned char> vchMessageData;
    int64_t nTimeStamp;
    int64_t nRelayUntil; // when newer nodes stop relaying to newer nodes

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(this->nVersion);
        READWRITE(SubjectID);
        READWRITE(fEncrypted);
        READWRITE(nDataFormatVersion);
        READWRITE(vchRelayWallet);
        READWRITE(vchMessageData);
        READWRITE(nTimeStamp);
        READWRITE(nRelayUntil);
    }

    void SetNull();

    std::string ToString() const;
};

/** An relay message is a combination of a serialized CUnsignedRelayMessage and a signature. */
class CRelayMessage : public CUnsignedRelayMessage
{
public:
    std::vector<unsigned char> vchMsg;
    std::vector<unsigned char> vchSig;

    CRelayMessage()
    {
        SetNull();
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(vchMsg);
        READWRITE(vchSig);
    }

    void SetNull();
    bool IsNull() const;
    uint256 GetHash() const;
    bool IsInEffect() const;
    bool AppliesTo(int nVersion, const std::string& strSubVerIn) const;
    //bool AppliesToMe() const;
    bool RelayMessage(CNode* pnode, CConnman& connman) const;
    bool Sign();
    bool CheckSignature(const std::vector<unsigned char>& vchPubKey) const;
    bool ProcessRelayMessage(const std::vector<unsigned char>& vchPubKey, bool fThread = true) const; // fThread means run -alertnotify in a free-running thread
    static void Notify(const std::string& strMessage, bool fThread = true);

    /*
     * Get copy of (active) relay message object by hash. Returns a null relay message if it is not found.
     */
    static CRelayMessage getAlertByHash(const uint256& hash);
};

#endif // DYNAMIC_BDAP_RELAYMESSAGE_H
