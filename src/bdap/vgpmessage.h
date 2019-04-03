// Copyright (c) 2019 Duality Blockchain Solutions Developers
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
#include <string>
#include <vector>

class CConnman;
class CKey;
class CKeyEd25519;
class CNode;
class CVGPMessage;
class uint256;

static const size_t MAX_MESSAGE_SIZE = 8192;
static constexpr int MIN_VGP_MESSAGE_PEER_PROTO_VERSION = 71200; // TODO (BDAP): Update minimum protocol version before v2.4 release
static constexpr size_t MAX_MESSAGE_DATA_LENGTH = 8192;
static constexpr uint32_t MIN_CLIENT_VERSION = 2041400; // TODO (BDAP): Update minimum client version before v2.4 release
static constexpr size_t MAX_SIGNATURE_SIZE = 72;
static constexpr size_t MAX_WALLET_PUBKEY_SIZE = 40;

class CUnsignedVGPMessage
{
public:
    static const int CURRENT_VERSION = 1;
    int nVersion;
    uint256 SubjectID;
    uint256 MessageID; // derived by hashing the public key + nTimestamp
    bool fEncrypted;
    std::vector<unsigned char> vchWalletPubKey;
    int64_t nTimeStamp;
    int64_t nRelayUntil; // when newer nodes stop relaying to newer nodes
    std::vector<unsigned char> vchMessageData;

    CUnsignedVGPMessage(const uint256& subjectID, const uint256& messageID, const std::vector<unsigned char> wallet, int64_t timestamp, int64_t stoptime)
        : SubjectID(subjectID), MessageID(messageID), vchWalletPubKey(wallet), nTimeStamp(timestamp), nRelayUntil(stoptime)
    {
        nVersion = CUnsignedVGPMessage::CURRENT_VERSION;
        vchMessageData.clear();
    }

    CUnsignedVGPMessage(const std::vector<unsigned char>& vchData)
    {
       UnserializeFromData(vchData);
    }

    CUnsignedVGPMessage()
    {
        SetNull();
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(this->nVersion);
        READWRITE(SubjectID);
        READWRITE(MessageID);
        READWRITE(fEncrypted);
        READWRITE(vchWalletPubKey);
        READWRITE(nTimeStamp);
        READWRITE(nRelayUntil);
        READWRITE(vchMessageData);
    }

    inline CUnsignedVGPMessage operator=(const CUnsignedVGPMessage& b)
    {
        nVersion = b.nVersion;
        SubjectID = b.SubjectID;
        MessageID = b.MessageID;
        fEncrypted = b.fEncrypted;
        vchWalletPubKey = b.vchWalletPubKey;
        nTimeStamp = b.nTimeStamp;
        nRelayUntil = b.nRelayUntil;
        vchMessageData = b.vchMessageData;
        return *this;
    }

    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData);

    void SetNull();

    bool EncryptMessage(const std::vector<unsigned char>& vchType, const std::vector<unsigned char>& vchMessage, const std::vector<unsigned char>& vchSenderFQDN, 
                        const std::vector<std::vector<unsigned char>>& vvchPubKeys, std::string& strErrorMessage);

    bool DecryptMessage(const std::array<char, 32>& arrPrivateSeed, std::vector<unsigned char>& vchType, 
                        std::vector<unsigned char>& vchMessage, std::vector<unsigned char>& vchSenderFQDN, std::string& strErrorMessage);

    std::vector<unsigned char> Type();
    std::vector<unsigned char> Value();
    std::vector<unsigned char> SenderFQDN();
    std::string ToString() const;

};

/** A VGP message is a combination of a serialized CUnsignedVGPMessage and a signature. */
class CVGPMessage
{
public:
    std::vector<unsigned char> vchMsg;
    std::vector<unsigned char> vchSig;

    CVGPMessage()
    {
        SetNull();
    }

    CVGPMessage(CUnsignedVGPMessage& unsignedMessage);

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
    bool RelayMessage(CConnman& connman) const;
    bool Sign(const CKey& key);
    bool CheckSignature(const std::vector<unsigned char>& vchPubKey) const;
    int ProcessMessage(std::string& strErrorMessage) const;
    bool RelayTo(CNode* pnode, CConnman& connman) const;

};

bool GetSecretSharedKey(const std::string& strSenderFQDN, const std::string& strRecipientFQDN, CKeyEd25519& key, std::string& strErrorMessage);
uint256 GetSubjectIDFromKey(const CKeyEd25519& key);

bool ReceivedMessage(const uint256& messageHash);
void CleanupRecentMessageLog();
void CleanupMyMessageMap();
bool DecryptMessage(CUnsignedVGPMessage& unsignedMessage);
void AddMyMessage(const CVGPMessage& message);
void GetMyLinkMessages(const uint256& subjectID, std::vector<CUnsignedVGPMessage>& vMessages);
void GetMyLinkMessagesByType(const std::vector<unsigned char>& vchType, const std::vector<unsigned char>& vchRecipientFQDN, std::vector<CVGPMessage>& vMessages);
void GetMyLinkMessagesBySubjectAndSender(const uint256& subjectID, const std::vector<unsigned char>& vchSenderFQDN, 
                                            const std::vector<unsigned char>& vchType, std::vector<CVGPMessage>& vchMessages);

#endif // DYNAMIC_BDAP_RELAYMESSAGE_H
