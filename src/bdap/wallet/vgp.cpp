// Copyright (c) 2019-2021 Duality Blockchain Solutions Developers
// Copyright (c) 2009-2021 The Bitcoin Developers
// Copyright (c) 2009-2021 Satoshi Nakamoto
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/vgpmessage.h"

#include "arith_uint256.h"
#include "base58.h"
#include "bdap/linkmanager.h"
#include "bdap/utils.h"
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

#include "encryption.h" // for VGP DecryptBDAPData

#include <cstdlib>

static std::map<uint256, CVGPMessage> mapMyVGPMessages;
static CCriticalSection cs_mapMyVGPMessages;
static int nMyMessageCounter = 0;

#ifdef ENABLE_WALLET
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
#endif // ENABLE_WALLET


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
