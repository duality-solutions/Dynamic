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
#include "wallet/wallet.h"

#include <cstdlib>

static std::map<uint256, CVGPMessage> mapMyVGPMessages;
static CCriticalSection cs_mapMyVGPMessages;
static int nMyMessageCounter = 0;

static std::map<uint256, int64_t> mapRecentMessageLog;
static CCriticalSection cs_mapRecentMessageLog;
static int nMessageCounter = 0;

class CMessage
{
public:
    static const int CURRENT_VERSION = 1;
    int nMessageVersion;
    std::vector<unsigned char> vchMessageType;
    std::vector<unsigned char> vchMessage;
    std::vector<unsigned char> vchSenderFQDN;

    CMessage()
    {
        SetNull();
    }

    CMessage(const int& version, const std::vector<unsigned char>& type, const std::vector<unsigned char>& message, const std::vector<unsigned char>& sender) 
                : nMessageVersion(version), vchMessageType(type), vchMessage(message), vchSenderFQDN(sender) {}

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
        READWRITE(vchSenderFQDN);
    }

    void SetNull()
    {
        nMessageVersion = -1;
        vchMessageType.clear();
        vchMessage.clear();
        vchSenderFQDN.clear();
    }

    inline bool IsNull() const { return (nMessageVersion == -1); }

    inline CMessage operator=(const CMessage& b)
    {
        nMessageVersion = b.nMessageVersion;
        vchMessageType = b.vchMessageType;
        vchMessage = b.vchMessage;
        vchSenderFQDN = b.vchSenderFQDN;
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

bool CUnsignedVGPMessage::EncryptMessage(const std::vector<unsigned char>& vchType, const std::vector<unsigned char>& vchMessage, const std::vector<unsigned char>& vchSenderFQDN, 
                                         const std::vector<std::vector<unsigned char>>& vvchPubKeys, std::string& strErrorMessage)
{
    CMessage message(CMessage::CURRENT_VERSION, vchType, vchMessage, vchSenderFQDN);
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

bool CUnsignedVGPMessage::DecryptMessage(const std::array<char, 32>& arrPrivateSeed, std::vector<unsigned char>& vchType, std::vector<unsigned char>& vchMessage, std::vector<unsigned char>& vchSenderFQDN, std::string& strErrorMessage)
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
    vchSenderFQDN = message.vchSenderFQDN;
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

std::vector<unsigned char> CUnsignedVGPMessage::Type()
{
    std::vector<unsigned char> vchType;
    if (fEncrypted)
        return vchType;

    CMessage message(vchMessageData);
    return message.vchMessageType;
}

std::vector<unsigned char> CUnsignedVGPMessage::Value()
{
    std::vector<unsigned char> vchValue;
    if (fEncrypted)
        return vchValue;

    CMessage message(vchMessageData);
    return message.vchMessage;
}

std::vector<unsigned char> CUnsignedVGPMessage::SenderFQDN()
{
    std::vector<unsigned char> vchSenderFQDN;
    if (fEncrypted)
        return vchSenderFQDN;

    CMessage message(vchMessageData);
    return message.vchSenderFQDN;
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
    CUnsignedVGPMessage unsignedMessage(vchMsg);
    // TODO (BDAP): Check pubkey is allowed to broadcast VGP messages, set ban score if not.
    // TODO (BDAP): Check number of messages from this pubkey. make sure it isn't spamming, set ban if too many messages per minute.
    //std::std::vector<unsigned char> vchWalletPubKey = unsignedMessage.vchWalletPubKey;
    if (ReceivedMessage(GetHash()))
    {
        strErrorMessage = "Message already received.";
        return -1; // do not relay message again
    }
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
    if (vchSig.size() > MAX_SIGNATURE_SIZE) // create a const for this.
    {
        strErrorMessage = "Signature size is too large. Adding 100 to ban score.";
        return 100; // this will add 100 to the peer's ban score
    }
    if (unsignedMessage.vchWalletPubKey.size() > MAX_WALLET_PUBKEY_SIZE)
    {
        strErrorMessage = "Wallet pubkey is too large. Adding 100 to ban score.";
        return 100; // this will add 100 to the peer's ban score
    }
    if (!CheckSignature(unsignedMessage.vchWalletPubKey))
    {
        strErrorMessage = "VGP message has an invalid signature. Adding 100 to ban score.";
        return 100; // this will add 100 to the peer's ban score
    }
    // check if message is for me. if, validate MessageID. If MessageID validates okay, store in memory map.
    int nMyLinkStatus = pLinkManager->IsMyMessage(unsignedMessage.SubjectID, unsignedMessage.MessageID, unsignedMessage.nTimeStamp);
    if (nMyLinkStatus == 1)
    {
        AddMyMessage((*this));
    }
    else if (nMyLinkStatus < 0)
    {   
        return std::abs(nMyLinkStatus);
    }
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

void CleanupRecentMessageLog()
{
    std::map<uint256, int64_t>::iterator itr = mapRecentMessageLog.begin();
    while (itr != mapRecentMessageLog.end())
    {
        int64_t nTimeStamp = (*itr).second;
        if (nTimeStamp + 600 > GetAdjustedTime())
        {
           itr = mapRecentMessageLog.erase(itr);
        }
        else
        {
           ++itr;
        }
    }
}

bool ReceivedMessage(const uint256& messageHash)
{
    LOCK(cs_mapRecentMessageLog);
    std::map<uint256, int64_t>::iterator iMessage = mapRecentMessageLog.find(messageHash);
    if (iMessage != mapRecentMessageLog.end()) 
    {
        return true;
    }
    else
    {
        int64_t nTimeStamp =  GetAdjustedTime();
        mapRecentMessageLog[messageHash] = nTimeStamp;
        nMessageCounter++;
        if ((nMessageCounter % 10) == 0)
            CleanupRecentMessageLog();
    }
    return false;
}

void CleanupMyMessageMap()
{
    int64_t nCurrentTimeStamp =  GetAdjustedTime();
    std::map<uint256, CVGPMessage>::iterator itr = mapMyVGPMessages.begin();
    while (itr != mapMyVGPMessages.end())
    {
        CVGPMessage message = (*itr).second;
        CUnsignedVGPMessage unsignedMessage(message.vchMsg);
        if (unsignedMessage.nTimeStamp + 1800 > nCurrentTimeStamp)
        {
            itr = mapMyVGPMessages.erase(itr);
        }
        else
        {
           ++itr;
        }
    }
}

bool DecryptMessage(CUnsignedVGPMessage& unsignedMessage)
{
    CLink link;
    if (pLinkManager->FindLinkBySubjectID(unsignedMessage.SubjectID, link))
    {
        std::array<char, 32> seed;
        std::string strErrorMessage = "";
        if (GetSharedPrivateSeed(link, seed, strErrorMessage))
        {
            std::vector<unsigned char> vchType, vchMessage, vchSenderFQDN;
            if (unsignedMessage.DecryptMessage(seed, vchType, vchMessage, vchSenderFQDN, strErrorMessage))
            {
                LogPrint("bdap", "%s -- Found and decrypted type = %s, message = %s, sender = %s\n", __func__, stringFromVch(vchType), stringFromVch(vchMessage), stringFromVch(vchSenderFQDN));
                CMessage message(1, vchType, vchMessage, vchSenderFQDN);
                unsignedMessage.fEncrypted = false;
                message.Serialize(unsignedMessage.vchMessageData);
                return true;
            }
            else
            {
                LogPrintf("%s -- GetSharedPrivateSeed failed. message = %s\n", __func__, strErrorMessage);
            }
        }
        else
        {
            LogPrintf("%s -- GetSharedPrivateSeed failed. message = %s\n", __func__, strErrorMessage);
        }
    }
    else
    {
        LogPrintf("%s -- FindLinkBySubjectID failed to find %s\n", __func__, unsignedMessage.SubjectID.ToString());
    }
    return false;
}

void AddMyMessage(const CVGPMessage& message)
{
    bool fFound = false;
    CUnsignedVGPMessage unsignedMessage(message.vchMsg);
    LogPrint("bdap", "%s -- Message hash = %s, Link MessageID = %s\n", __func__, message.GetHash().ToString(), unsignedMessage.MessageID.ToString());
    CVGPMessage storeMessage;
    if (pwalletMain && pLinkManager && !pwalletMain->IsLocked() && unsignedMessage.fEncrypted)
    {
        if (DecryptMessage(unsignedMessage))
            fFound = true;
    }
    if (fFound)
    {
        CVGPMessage newMessage(unsignedMessage);
        storeMessage = newMessage;
    }
    else
    {
        storeMessage = message;
    }
    LOCK(cs_mapMyVGPMessages);
    mapMyVGPMessages[storeMessage.GetHash()] = storeMessage;
    nMyMessageCounter++;
    if ((nMyMessageCounter % 10) == 0)
        CleanupMyMessageMap();
}

void GetMyLinkMessages(const uint256& subjectID, std::vector<CUnsignedVGPMessage>& vMessages)
{
    LOCK(cs_mapMyVGPMessages);
    std::map<uint256, CVGPMessage>::iterator itr = mapMyVGPMessages.begin();
    while (itr != mapMyVGPMessages.end())
    {
        CVGPMessage message = (*itr).second;
        CUnsignedVGPMessage unsignedMessage(message.vchMsg);
        if (unsignedMessage.SubjectID == subjectID)
        {
            if (unsignedMessage.fEncrypted)
            {
                if (pwalletMain && !pwalletMain->IsLocked() && DecryptMessage(unsignedMessage))
                {
                    vMessages.push_back(unsignedMessage);
                }
            }
            else
            {
                vMessages.push_back(unsignedMessage);
            }
        }
    }
}

void GetMyLinkMessagesByType(const std::vector<unsigned char>& vchType, const std::vector<unsigned char>& vchRecipientFQDN, std::vector<CVGPMessage>& vMessages)
{
    LOCK(cs_mapMyVGPMessages);
    std::map<uint256, CVGPMessage>::iterator itr = mapMyVGPMessages.begin();
    while (itr != mapMyVGPMessages.end())
    {
        CVGPMessage message = (*itr).second;
        CUnsignedVGPMessage unsignedMessage(message.vchMsg);
        if (unsignedMessage.fEncrypted && pwalletMain && !pwalletMain->IsLocked())
        {
            DecryptMessage(unsignedMessage);
        }
        if (!unsignedMessage.fEncrypted && (vchType.size() == 0 || vchType == unsignedMessage.Type()) && unsignedMessage.SenderFQDN() != vchRecipientFQDN)
        {
            vMessages.push_back(unsignedMessage);
        }
        itr++;
    }
}

void GetMyLinkMessagesBySubjectAndSender(const uint256& subjectID, const std::vector<unsigned char>& vchSenderFQDN, 
                                            const std::vector<unsigned char>& vchType, std::vector<CVGPMessage>& vchMessages)
{
    LOCK(cs_mapMyVGPMessages);
    std::map<uint256, CVGPMessage>::iterator itr = mapMyVGPMessages.begin();
    while (itr != mapMyVGPMessages.end())
    {
        CVGPMessage message = (*itr).second;
        CUnsignedVGPMessage unsignedMessage(message.vchMsg);
        if (unsignedMessage.SubjectID == subjectID && unsignedMessage.SenderFQDN() == vchSenderFQDN && (vchType.size() == 0 || vchType == unsignedMessage.Type()))
        {
            vchMessages.push_back(message);
        }
        itr++;
    }
}