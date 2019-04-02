// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/vgpmessage.h"

#include "base58.h"
#include "bdap/linkmanager.h"
#include "bdap/utils.h"
#include "bdap/vgp/include/encryption.h" // for VGP DecryptBDAPData
#include "clientversion.h"
#include "dht/ed25519.h"
#include "hash.h"
#include "key.h"
#include "net.h" // for g_connman
#include "netmessagemaker.h"
#include "script/script.h"
#include "streams.h"
#include "timedata.h"
#include "util.h"

class CMessage
{
public:
    static const int CURRENT_VERSION = 1;
    int nMessageVersion;
    std::vector<unsigned char> vchMessageType;
    std::vector<unsigned char> vchMessage;

    CMessage()
    {
        SetNull();
    }

    CMessage(const int& version, const std::vector<unsigned char>& type, const std::vector<unsigned char>& message) 
                : nMessageVersion(version), vchMessageType(type), vchMessage(message) {}

    CMessage(const std::vector<unsigned char>& vchData)
    {
        UnserializeFromData(vchData);
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(nMessageVersion);
        READWRITE(vchMessageType);
        READWRITE(vchMessage);
    }

    void SetNull()
    {
        nMessageVersion = -1;
        vchMessageType.clear();
        vchMessage.clear();
    }

    inline bool IsNull() const { return (nMessageVersion == -1); }

    inline CMessage operator=(const CMessage& b)
    {
        nMessageVersion = b.nMessageVersion;
        vchMessageType = b.vchMessageType;
        vchMessage = b.vchMessage;
        return *this;
    }

    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData);
};

void CMessage::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsMessageData(SER_NETWORK, PROTOCOL_VERSION);
    dsMessageData << *this;
    vchData = std::vector<unsigned char>(dsMessageData.begin(), dsMessageData.end());
}

bool CMessage::UnserializeFromData(const std::vector<unsigned char>& vchData)
{
    try {
        CDataStream dsMessageData(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsMessageData >> *this;
    } catch (std::exception &e) {
        SetNull();
        return false;
    }
    return true;
}

void CUnsignedVGPMessage::SetNull()
{
    nVersion = CURRENT_VERSION;
    SubjectID.SetNull();
    MessageID.SetNull();
    fEncrypted = false;
    vchWalletPubKey.clear();
    nTimeStamp = 0;
    nRelayUntil = 0;
    vchMessageData.clear();
}

std::string CUnsignedVGPMessage::ToString() const
{
    return strprintf(
        "CUnsignedVGPMessage(\n"
        "    nVersion             = %d\n"
        "    SubjectID            = %s\n"
        "    MessageID            = %s\n"
        "    Encrypted            = %s\n"
        "    RelayWallet          = %s\n"
        "    nTimeStamp           = %d\n"
        "    nRelayUntil          = %d\n"
        "    Message Size         = %d\n"
        ")\n",
        nVersion,
        SubjectID.ToString(),
        MessageID.ToString(),
        fEncrypted ? "true" : "false",
        stringFromVch(vchWalletPubKey),
        nTimeStamp,
        nRelayUntil,
        vchMessageData.size());
}

bool CUnsignedVGPMessage::EncryptMessage(const std::vector<unsigned char>& vchType, const std::vector<unsigned char>& vchMessage, const std::vector<std::vector<unsigned char>>& vvchPubKeys, std::string& strErrorMessage)
{
    CMessage message(CMessage::CURRENT_VERSION, vchType, vchMessage);
    std::vector<unsigned char> vchData;
    message.Serialize(vchData);
    std::vector<unsigned char> vchCipherText;
    if (!EncryptBDAPData(vvchPubKeys, vchData, vchCipherText, strErrorMessage))
    {
        return false;
    }
    fEncrypted = true;
    vchMessageData = vchCipherText;
    return true;
}

bool CUnsignedVGPMessage::DecryptMessage(const std::array<char, 32>& arrPrivateSeed, std::vector<unsigned char>& vchType, std::vector<unsigned char>& vchMessage, std::string& strErrorMessage)
{
    if (!fEncrypted)
        return false;

    std::vector<unsigned char> vchRawPrivSeed;
    for (unsigned int i = 0; i < sizeof(arrPrivateSeed); i++)
    {
        vchRawPrivSeed.push_back(arrPrivateSeed[i]);
    }
    std::vector<unsigned char> vchDecryptMessage;
    if (!DecryptBDAPData(vchRawPrivSeed, vchMessageData, vchDecryptMessage, strErrorMessage))
    {
        return false;
    }
    CMessage message(vchDecryptMessage);
    vchType = message.vchMessageType;
    vchMessage = message.vchMessage;
    return true;
}

void CUnsignedVGPMessage::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsMessageData(SER_NETWORK, PROTOCOL_VERSION);
    dsMessageData << *this;
    vchData = std::vector<unsigned char>(dsMessageData.begin(), dsMessageData.end());
}

bool CUnsignedVGPMessage::UnserializeFromData(const std::vector<unsigned char>& vchData)
{
    try {
        CDataStream dsMessageData(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsMessageData >> *this;
    } catch (std::exception &e) {
        SetNull();
        return false;
    }
    return true;
}

CVGPMessage::CVGPMessage(CUnsignedVGPMessage& unsignedMessage)
{
    unsignedMessage.Serialize(vchMsg);
}

void CVGPMessage::SetNull()
{
    vchMsg.clear();
    vchSig.clear();
}

bool CVGPMessage::IsNull() const
{
    return (vchMsg.size() == 0);
}

uint256 CVGPMessage::GetHash() const
{
    return Hash(this->vchMsg.begin(), this->vchMsg.end());
}

bool CVGPMessage::IsInEffect() const
{
    if (vchMsg.size() == 0)
        return false;
    // only keep for 1 minute
    CUnsignedVGPMessage unsignedMessage(vchMsg);
    return (unsignedMessage.nTimeStamp + 60 >= GetAdjustedTime());
}

bool CVGPMessage::RelayMessage(CConnman& connman) const
{
    CUnsignedVGPMessage unsignedMessage(vchMsg);
    if (!IsInEffect())
        return false;

    connman.ForEachNode([&connman, this, unsignedMessage](CNode* pnode) {
        if (pnode->nVersion != 0 && pnode->nVersion >= MIN_VGP_MESSAGE_PEER_PROTO_VERSION)
        {
            CNetMsgMaker msgMaker(pnode->GetSendVersion());
            // returns true if wasn't already contained in the set
            if (pnode->setKnown.insert(GetHash()).second) {
                if (GetAdjustedTime() < unsignedMessage.nRelayUntil) {
                    connman.PushMessage(pnode, msgMaker.Make(NetMsgType::VGPMESSAGE, (*this)));
                }
            }
        }
    });
    return true;
}

bool CVGPMessage::Sign(const CKey& key)
{
    if (!key.Sign(Hash(vchMsg.begin(), vchMsg.end()), vchSig)) {
        LogPrintf("CVGPMessage::%s -- Failed to sign VGP message.\n", __func__);
        return false;
    }

    return true;
}

bool CVGPMessage::CheckSignature(const std::vector<unsigned char>& vchPubKey) const
{
    CPubKey key(vchPubKey);
    if (!key.Verify(Hash(vchMsg.begin(), vchMsg.end()), vchSig))
        return error("CVGPMessage::%s(): verify signature failed", __func__);

    return true;
}

int CVGPMessage::ProcessMessage(std::string& strErrorMessage) const
{
    // TODO (BDAP): Check pubkey is allowed to broadcast VGP messages, set ban score if not.
    // TODO (BDAP): Check number of messages from this pubkey. make sure it isn't spamming, set ban if too many messages per minute.
    CUnsignedVGPMessage unsignedMessage(vchMsg);
    if (unsignedMessage.vchMessageData.size() > MAX_MESSAGE_DATA_LENGTH)
    {
        strErrorMessage = "Message length exceeds limit. Adding 10 to ban score.";
        return 10; // this will add 100 to the peer's ban score
    }
    if (vchMsg.size() > MAX_MESSAGE_DATA_LENGTH) // create a const for this.
    {
        strErrorMessage = "Message length exceeds limit. Adding 10 to ban score.";
        return 10; // this will add 100 to the peer's ban score
    }
    if (vchSig.size() > 80) // create a const for this.
    {
        strErrorMessage = "Signature size is too large. Adding 100 to ban score.";
        return 100; // this will add 100 to the peer's ban score
    }
    if (unsignedMessage.vchWalletPubKey.size() > 178)
    {
        strErrorMessage = "Wallet pubkey is too large. Adding 100 to ban score.";
        return 100; // this will add 100 to the peer's ban score
    }
    if (!CheckSignature(unsignedMessage.vchWalletPubKey))
    {
        strErrorMessage = "VGP message has an invalid signature. Adding 100 to ban score.";
        return 100; // this will add 100 to the peer's ban score
    }
    // Add message hash to map that stored relayed message hashes.
    // check if message is for me. if, validate MessageID. If MessageID validates okay, store in memory map.
    return 0; // All checks okay, relay message to peers.
}

bool CVGPMessage::RelayTo(CNode* pnode, CConnman& connman) const
{
    if (pnode->nVersion != 0 && pnode->nVersion >= MIN_VGP_MESSAGE_PEER_PROTO_VERSION)
    {
        CNetMsgMaker msgMaker(pnode->GetSendVersion());
        CUnsignedVGPMessage unsignedMessage(vchMsg);
        if (pnode->setKnown.insert(GetHash()).second) {
            if (GetAdjustedTime() < unsignedMessage.nRelayUntil) {
                connman.PushMessage(pnode, msgMaker.Make(NetMsgType::VGPMESSAGE, (*this)));
            }
            else {
                return false;
            }
        }
        else {
            return false;
        }
    }
    else {
        return false;
    }
    return true;
}

bool GetSecretSharedKey(const std::string& strSenderFQDN, const std::string& strRecipientFQDN, CKeyEd25519& key, std::string& strErrorMessage)
{
    if (!pLinkManager)
    {
        strErrorMessage = "Link manager is null.";
        return false;
    }
    uint256 id = GetLinkID(strSenderFQDN, strRecipientFQDN);
    CLink link;
    if (!pLinkManager->FindLink(id, link))
    {
        strErrorMessage = "Link not found.";
        return false;
    }
    std::array<char, 32> seed;
    if (!GetSharedPrivateSeed(link, seed, strErrorMessage))
    {
        strErrorMessage = "Failed to get shared secret link key.";
        return false;
    }
    CKeyEd25519 getKey(seed);
    key = getKey;

    return true;
}

uint256 GetSubjectIDFromKey(const CKeyEd25519& key)
{
    std::vector<unsigned char> vchPubKey = key.GetPubKeyBytes();
    return Hash(vchPubKey.begin(), vchPubKey.end());
}

uint256 GetMessageID(const CKeyEd25519& key, const int64_t& timestamp)
{
    CScript scriptMessage;
    scriptMessage << key.GetPubKeyBytes() << timestamp;
    return Hash(scriptMessage.begin(), scriptMessage.end());
}