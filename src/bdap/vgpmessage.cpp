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
    bool fKeepLast;

    CMessage()
    {
        SetNull();
    }

    CMessage(const int& version, const std::vector<unsigned char>& type, const std::vector<unsigned char>& message, const std::vector<unsigned char>& sender, bool keeplast) 
                : nMessageVersion(version), vchMessageType(type), vchMessage(message), vchSenderFQDN(sender), fKeepLast(keeplast) {}

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
        READWRITE(fKeepLast);
    }

    void SetNull()
    {
        nMessageVersion = -1;
        vchMessageType.clear();
        vchMessage.clear();
        vchSenderFQDN.clear();
        fKeepLast = false;
    }

    inline bool IsNull() const { return (nMessageVersion == -1); }

    inline CMessage operator=(const CMessage& b)
    {
        nMessageVersion = b.nMessageVersion;
        vchMessageType = b.vchMessageType;
        vchMessage = b.vchMessage;
        vchSenderFQDN = b.vchSenderFQDN;
        fKeepLast = b.fKeepLast;
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
    nNonce = 0;
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
        "    nNonce               = %d\n"
        "    Hash                 = %s\n"
        ")\n",
        nVersion,
        SubjectID.ToString(),
        MessageID.ToString(),
        fEncrypted ? "true" : "false",
        stringFromVch(vchWalletPubKey),
        nTimeStamp,
        nRelayUntil,
        vchMessageData.size(),
        nNonce,
        GetHash().ToString()
        );
}

uint256 CUnsignedVGPMessage::GetHash() const
{
    CDataStream dsMessageData(SER_NETWORK, PROTOCOL_VERSION);
    dsMessageData << *this;
    return hash_Argon2d(dsMessageData.begin(), dsMessageData.end(), 1);
}

bool CUnsignedVGPMessage::EncryptMessage(const std::vector<unsigned char>& vchType, const std::vector<unsigned char>& vchMessage, const std::vector<unsigned char>& vchSenderFQDN, 
                                         const std::vector<std::vector<unsigned char>>& vvchPubKeys, const bool fKeepLast, std::string& strErrorMessage)
{
    CMessage message(1, vchType, vchMessage, vchSenderFQDN, fKeepLast);
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

bool CUnsignedVGPMessage::DecryptMessage(const std::array<char, 32>& arrPrivateSeed, std::vector<unsigned char>& vchType, std::vector<unsigned char>& vchMessage, std::vector<unsigned char>& vchSenderFQDN, bool& fKeepLast, std::string& strErrorMessage)
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
    fKeepLast = message.fKeepLast;
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

bool CUnsignedVGPMessage::KeepLast()
{
    bool keepLast = false;
    if (fEncrypted)
        return keepLast;

    CMessage message(vchMessageData);
    return message.fKeepLast;
}

CVGPMessage::CVGPMessage(CUnsignedVGPMessage& unsignedMessage)
{
    unsignedMessage.Serialize(vchMsg);
}

int CVGPMessage::Version() const
{
    if (vchMsg.size() == 0)
        return -1;

    return CUnsignedVGPMessage(vchMsg).nVersion;
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
    return CUnsignedVGPMessage(vchMsg).GetHash();
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
    int64_t nCurrentTimeStamp = GetAdjustedTime();
    CUnsignedVGPMessage unsignedMessage(vchMsg);
    // TODO (BDAP): Check pubkey is allowed to broadcast VGP messages, set ban score if not.
    // TODO (BDAP): Check number of messages from this pubkey. make sure it isn't spamming, set ban if too many messages per minute.
    //std::std::vector<unsigned char> vchWalletPubKey = unsignedMessage.vchWalletPubKey;
    if (ReceivedMessage(GetHash()))
    {
        strErrorMessage = "Message already received.";
        return -1; // do not relay message again
    }
    if (std::abs(nCurrentTimeStamp - unsignedMessage.nTimeStamp) > MAX_MESAGGE_DRIFT_SECONDS)
    {
        strErrorMessage = "Message exceeds maximum time drift.";
        return -2; // message too old or into the future (time drift exceeds maximum allowed)
    }
    if (unsignedMessage.nTimeStamp >= unsignedMessage.nRelayUntil)
    {
        strErrorMessage = "Timestamp is greater than relay until time. Malformed message.";
        return -3; // timestamp is greater than relay until time
    }
    if (std::abs(unsignedMessage.nTimeStamp - unsignedMessage.nRelayUntil) > MAX_MESAGGE_RELAY_SECONDS)
    {
        strErrorMessage = "Too much span between timestamp and relay until time.";
        return -4; // relay time is too much.  max relay is 120 seconds
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
    if (UintToArith256(unsignedMessage.GetHash()) > UintToArith256(VGP_MESSAGE_MIN_HASH_TARGET))
    {
        LogPrintf("%s -- message proof hash failed to meet target %s\n", __func__, unsignedMessage.ToString());
        strErrorMessage = "Message proof of work is invalid and under the target.";
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

void CVGPMessage::MineMessage()
{
    int64_t nStart = GetTimeMillis();
    CUnsignedVGPMessage message(vchMsg);
    message.nNonce = 0;
    arith_uint256 besthash = UintToArith256(uint256S("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"));
    arith_uint256 hashTarget = UintToArith256(VGP_MESSAGE_MIN_HASH_TARGET);
    arith_uint256 newhash = UintToArith256(message.GetHash());
    while (newhash > hashTarget) {
        message.nNonce++;
        if (message.nNonce == 0) {
            ++message.nTimeStamp;
            ++message.nRelayUntil;
        }
        if (newhash < besthash) {
            besthash = newhash;
            LogPrint("bdap", "%s -- New best: %s\n", __func__, newhash.GetHex());
        }
        newhash = UintToArith256(message.GetHash());
    }
    message.Serialize(vchMsg);
    LogPrintf("%s -- Milliseconds %d, nNonce %d, Hash %s\n", __func__, GetTimeMillis() - nStart, message.nNonce, GetHash().ToString());
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
    int64_t nCurrentTimeStamp =  GetAdjustedTime();
    std::map<uint256, int64_t>::iterator itr = mapRecentMessageLog.begin();
    while (itr != mapRecentMessageLog.end())
    {
        int64_t nTimeStamp = (*itr).second;
        if (nCurrentTimeStamp > nTimeStamp + KEEP_MESSAGE_LOG_ALIVE_SECONDS)
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
    // map with message type, message sender and timestamp.  Used to keep last message from a sender/type pair.
    std::map<std::pair<std::vector<unsigned char>, std::vector<unsigned char>>, int64_t> mapMessageTypeFromTimestamp;
    int64_t nCurrentTimeStamp =  GetAdjustedTime();
    std::map<uint256, CVGPMessage>::iterator itr = mapMyVGPMessages.begin();
    while (itr != mapMyVGPMessages.end())
    {
        CVGPMessage message = (*itr).second;
        CUnsignedVGPMessage unsignedMessage(message.vchMsg);
        if (!unsignedMessage.fEncrypted && nCurrentTimeStamp > unsignedMessage.nTimeStamp + KEEP_MY_MESSAGE_ALIVE_SECONDS)
        {
            CMessage message(unsignedMessage.vchMessageData);
            std::pair<std::vector<unsigned char>, std::vector<unsigned char>> pairTypeFrom = std::make_pair(message.vchMessageType, message.vchSenderFQDN);
            if (!message.fKeepLast) {
                itr = mapMyVGPMessages.erase(itr);
            }
            else {
                std::map<std::pair<std::vector<unsigned char>, std::vector<unsigned char>>, int64_t>::iterator itTypeFrom = mapMessageTypeFromTimestamp.find(pairTypeFrom);
                if (itTypeFrom != mapMessageTypeFromTimestamp.end()) {
                    if (itTypeFrom->second > unsignedMessage.nTimeStamp) {
                        itr = mapMyVGPMessages.erase(itr);
                    }
                    else {
                        mapMessageTypeFromTimestamp[pairTypeFrom] = unsignedMessage.nTimeStamp;
                        ++itr;
                    }
                }
                else {
                    mapMessageTypeFromTimestamp[pairTypeFrom] = unsignedMessage.nTimeStamp;
                    ++itr;
                }
            }
        }
        else
        {
           ++itr;
        }
    }
    LogPrintf("%s -- Size %d\n", __func__, mapMyVGPMessages.size());
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
            bool fKeepLast;
            std::vector<unsigned char> vchType, vchMessage, vchSenderFQDN;
            if (unsignedMessage.DecryptMessage(seed, vchType, vchMessage, vchSenderFQDN, fKeepLast, strErrorMessage))
            {
                LogPrint("bdap", "%s -- Found and decrypted type = %s, message = %s, sender = %s\n", __func__, stringFromVch(vchType), stringFromVch(vchMessage), stringFromVch(vchSenderFQDN));
                CMessage message(1, vchType, vchMessage, vchSenderFQDN, fKeepLast);
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
        itr++;
    }
}

void GetMyLinkMessagesByType(const std::vector<unsigned char>& vchType, const std::vector<unsigned char>& vchRecipientFQDN, std::vector<CVGPMessage>& vMessages, bool& fKeepLast)
{
    LOCK(cs_mapMyVGPMessages);
    std::map<uint256, CVGPMessage>::iterator itr = mapMyVGPMessages.begin();
    while (itr != mapMyVGPMessages.end())
    {
        CVGPMessage messageWrapper = (*itr).second;
        CUnsignedVGPMessage unsignedMessage(messageWrapper.vchMsg);
        if (unsignedMessage.fEncrypted && pwalletMain && !pwalletMain->IsLocked())
        {
            DecryptMessage(unsignedMessage);
        }
        if (!unsignedMessage.fEncrypted && (vchType.size() == 0 || vchType == unsignedMessage.Type()) && unsignedMessage.SenderFQDN() != vchRecipientFQDN)
        {
            if (unsignedMessage.KeepLast())
                fKeepLast = true;

            vMessages.push_back(unsignedMessage);
        }
        itr++;
    }
}

void GetMyLinkMessagesBySubjectAndSender(const uint256& subjectID, const std::vector<unsigned char>& vchSenderFQDN, 
                                            const std::vector<unsigned char>& vchType, std::vector<CVGPMessage>& vchMessages, bool& fKeepLast)
{
    LOCK(cs_mapMyVGPMessages);
    std::map<uint256, CVGPMessage>::iterator itr = mapMyVGPMessages.begin();
    while (itr != mapMyVGPMessages.end())
    {
        CVGPMessage messageWrapper = (*itr).second;
        CUnsignedVGPMessage unsignedMessage(messageWrapper.vchMsg);
        if (unsignedMessage.SubjectID == subjectID && unsignedMessage.SenderFQDN() == vchSenderFQDN && (vchType.size() == 0 || vchType == unsignedMessage.Type()))
        {
            if (unsignedMessage.KeepLast())
                fKeepLast = true;

            vchMessages.push_back(messageWrapper);
        }
        itr++;
    }
}

void KeepLastTypeBySender(std::vector<CVGPMessage>& vMessages)
{
    std::map<std::pair<std::vector<unsigned char>, std::vector<unsigned char>>, std::pair<CVGPMessage, int64_t> > mapFromMessageTime;
    for (const CVGPMessage& messageWrapper : vMessages)
    {
        CUnsignedVGPMessage unsignedMessage(messageWrapper.vchMsg);
        if (!unsignedMessage.fEncrypted)
        {
            CMessage message(unsignedMessage.vchMessageData);
            std::pair<std::vector<unsigned char>, std::vector<unsigned char>> pairTypeFrom = std::make_pair(message.vchMessageType, message.vchSenderFQDN);
            std::pair<CVGPMessage, int64_t> pairMessageTime = std::make_pair(messageWrapper, unsignedMessage.nTimeStamp);
            std::map<std::pair<std::vector<unsigned char>, std::vector<unsigned char>>, std::pair<CVGPMessage, int64_t> >::iterator itFind = mapFromMessageTime.find(pairTypeFrom);
            if (itFind == mapFromMessageTime.end()) {
                mapFromMessageTime[pairTypeFrom] = pairMessageTime;
            }
            else if (unsignedMessage.nTimeStamp > itFind->second.second) {
                mapFromMessageTime[pairTypeFrom] = pairMessageTime;
            }
        }
    }
    vMessages.clear();
    std::map<std::pair<std::vector<unsigned char>, std::vector<unsigned char>>, std::pair<CVGPMessage, int64_t> >::iterator itr = mapFromMessageTime.begin();
    while (itr != mapFromMessageTime.end())
    {
        vMessages.push_back(itr->second.first);
        itr++;
    }
}