// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/globaldata.h"

#include "base58.h"
#include "bdap/domainentry.h"
#include "bdap/domainentrydb.h"
#include "bdap/utils.h"
#include "hash.h"
#include "net.h" // for g_connman
#include "netmessagemaker.h"
#include "streams.h"
#include "timedata.h"
#include "util.h"


static std::map<uint256, int64_t> mapRecentMessageLog;
static CCriticalSection cs_mapRecentMessageLog;
static int nMessageCounter = 0;

void CGlobalData::Serialize(std::vector<unsigned char>& vchData)
{
    CDataStream dsGlobalData(SER_NETWORK, PROTOCOL_VERSION);
    dsGlobalData << *this;
    vchData = std::vector<unsigned char>(dsGlobalData.begin(), dsGlobalData.end());
}

bool CGlobalData::UnserializeFromData(const std::vector<unsigned char>& vchData)
{
    try {
        CDataStream dsGlobalData(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsGlobalData >> *this;
    } catch (std::exception &e) {
        SetNull();
        return false;
    }
    return true;
}

std::string CGlobalData::ToString() const
{
    return strprintf(
        "CGlobalData(\n"
        "    nVersion         = %d\n"
        "    nTimeStamp       = %d\n"
        "    nType            = %d\n"
        "    vchFQDN          = %s\n"
        "    vchData          = %s\n"
        ")\n",
        nVersion,
        nTimeStamp,
        nType,
        stringFromVch(vchFQDN),
        stringFromVch(vchData));
}

CPubKey CGlobalData::GetPubKey() const
{
    CPubKey pubKey;
    pubKey.Set(vchWalletPubKey.begin(), vchWalletPubKey.end());
    return pubKey;
}

CDynamicAddress CGlobalData::WalletAddress() const
{
    return CDynamicAddress(GetPubKey().GetID());
}

uint256 CGlobalAvatar::GetHash() const
{
    return Hash(this->vchGlobalData.begin(), this->vchGlobalData.end());
}

void CGlobalAvatar::Serialize(std::vector<unsigned char>& vchData)
{
    CDataStream dsAvatarData(SER_NETWORK, PROTOCOL_VERSION);
    dsAvatarData << *this;
    vchData = std::vector<unsigned char>(dsAvatarData.begin(), dsAvatarData.end());
}

bool CGlobalAvatar::UnserializeFromData(const std::vector<unsigned char>& vchData)
{
    try {
        CDataStream dsAvatarData(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsAvatarData >> *this;
    } catch (std::exception &e) {
        SetNull();
        return false;
    }
    return true;
}

bool CGlobalAvatar::Sign(const CKey& key)
{
    if (!key.Sign(Hash(vchGlobalData.begin(), vchGlobalData.end()), vchSignature)) {
        LogPrintf("CGlobalAvatar::%s -- Failed to sign global avatar message.\n", __func__);
        return false;
    }

    return true;
}

bool CGlobalAvatar::CheckSignature(const std::vector<unsigned char>& vchPubKey) const
{
    CPubKey key(vchPubKey);
    if (!key.Verify(Hash(vchGlobalData.begin(), vchGlobalData.end()), vchSignature))
        return error("CGlobalAvatar::%s(): verify signature failed", __func__);

    return true;
}

bool CGlobalAvatar::RelayMessage(CConnman& connman) const
{
    CGlobalData data(vchGlobalData); 
    if (RELAY_KEEP_ALIVE_SECONDS > std::abs(GetAdjustedTime() - data.nTimeStamp))
        return false;

    connman.ForEachNode([&connman, this](CNode* pnode) {
        if (pnode->nVersion != 0 && pnode->nVersion >= MIN_GLOBAL_DATA_PEER_PROTO_VERSION)
        {
            CNetMsgMaker msgMaker(pnode->GetSendVersion());
            // returns true if wasn't already contained in the set
            if (pnode->setKnown.insert(GetHash()).second) {
                connman.PushMessage(pnode, msgMaker.Make(NetMsgType::AVATAR, (*this)));
            }
        }
    });
    return true;
}

int CGlobalAvatar::ProcessMessage(std::string& strErrorMessage) const
{
    int64_t nCurrentTimeStamp = GetAdjustedTime();
    CGlobalData data(vchGlobalData);
    if (ReceivedGlobalDataMessage(GetHash()))
    {
        strErrorMessage = "Message already received.";
        return -1; // do not relay message again
    }
    if (std::abs(nCurrentTimeStamp - data.nTimeStamp) > MAX_GLOBAL_MESSAGE_DRIFT_SECONDS)
    {
        strErrorMessage = "Message exceeds maximum time drift.";
        return -2; // message too old or into the future (time drift exceeds maximum allowed)
    }
    if (data.vchFQDN.size() > MAX_OBJECT_FULL_PATH_LENGTH)
    {
        strErrorMessage = "Message length exceeds limit. Adding 10 to ban score.";
        return 10; // this will add 100 to the peer's ban score
    }
    if (vchGlobalData.size() > MAX_GLOBAL_MESSAGE_SIGNATURE_SIZE)
    {
        strErrorMessage = "Signature size is too large. Adding 100 to ban score.";
        return 10; // this will add 100 to the peer's ban score
    }
    if (vchSignature.size() > MAX_GLOBAL_MESSAGE_SIGNATURE_SIZE)
    {
        strErrorMessage = "Signature size is too large. Adding 100 to ban score.";
        return 100; // this will add 100 to the peer's ban score
    }
    // Get the BDAP account entry
    CDomainEntry entry;
    // TODO (BDAP): Make sure local node has a synced blockchain
    if (!GetDomainEntry(data.vchFQDN, entry)) {
        strErrorMessage = "BDAP account in the global avatar message not found. Adding 100 to ban score.";
        return 100; // this will add 100 to the peer's ban score
    }
    // Verify message wallet address pubkey matches BDAP account wallet address
    if (data.WalletAddress() != entry.GetWalletAddress()) {
        strErrorMessage = "Global avatar message wallet address does not match BDAP entry. Adding 100 to ban score.";
        return 100; // this will add 100 to the peer's ban score
    }
    // Verify signature is correct for the given wallet address pubkey.
    if (!CheckSignature(data.vchWalletPubKey))
    {
        strErrorMessage = "Global avatar message has an invalid signature. Adding 100 to ban score.";
        return 100; // this will add 100 to the peer's ban score
    }
    // TODO (BDAP): Write to BDAP account database

    return 0; // All checks okay, relay message to peers.
}

bool CGlobalAvatar::RelayTo(CNode* pnode, CConnman& connman) const
{
    if (pnode->nVersion != 0 && pnode->nVersion >= MIN_GLOBAL_DATA_PEER_PROTO_VERSION)
    {
        CNetMsgMaker msgMaker(pnode->GetSendVersion());
        CGlobalData data(vchGlobalData);
        if (pnode->setKnown.insert(GetHash()).second) {
            if (GetAdjustedTime() < data.nTimeStamp) {
                connman.PushMessage(pnode, msgMaker.Make(NetMsgType::AVATAR, (*this)));
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

uint256 CGlobalProfile::GetHash() const
{
    return Hash(this->vchGlobalData.begin(), this->vchGlobalData.end());
}

void CGlobalProfile::Serialize(std::vector<unsigned char>& vchData)
{
    CDataStream dsProfileData(SER_NETWORK, PROTOCOL_VERSION);
    dsProfileData << *this;
    vchData = std::vector<unsigned char>(dsProfileData.begin(), dsProfileData.end());
}

bool CGlobalProfile::UnserializeFromData(const std::vector<unsigned char>& vchData)
{
    try {
        CDataStream dsProfileData(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsProfileData >> *this;
    } catch (std::exception &e) {
        SetNull();
        return false;
    }
    return true;
}

bool CGlobalProfile::Sign(const CKey& key)
{
    if (!key.Sign(Hash(vchGlobalData.begin(), vchGlobalData.end()), vchSignature)) {
        LogPrintf("CGlobalProfile::%s -- Failed to sign global profile message.\n", __func__);
        return false;
    }

    return true;
}

bool CGlobalProfile::CheckSignature(const std::vector<unsigned char>& vchPubKey) const
{
    CPubKey key(vchPubKey);
    if (!key.Verify(Hash(vchGlobalData.begin(), vchGlobalData.end()), vchSignature))
        return error("CGlobalProfile::%s(): verify signature failed", __func__);

    return true;
}

bool CGlobalProfile::RelayMessage(CConnman& connman) const
{
    CGlobalData data(vchGlobalData); 
    if (RELAY_KEEP_ALIVE_SECONDS > std::abs(GetAdjustedTime() - data.nTimeStamp))
        return false;

    connman.ForEachNode([&connman, this](CNode* pnode) {
        if (pnode->nVersion != 0 && pnode->nVersion >= MIN_GLOBAL_DATA_PEER_PROTO_VERSION)
        {
            CNetMsgMaker msgMaker(pnode->GetSendVersion());
            // returns true if wasn't already contained in the set
            if (pnode->setKnown.insert(GetHash()).second) {
                connman.PushMessage(pnode, msgMaker.Make(NetMsgType::PROFILE, (*this)));
            }
        }
    });
    return true;
}

int CGlobalProfile::ProcessMessage(std::string& strErrorMessage) const
{
    int64_t nCurrentTimeStamp = GetAdjustedTime();
    CGlobalData data(vchGlobalData);
    if (ReceivedGlobalDataMessage(GetHash()))
    {
        strErrorMessage = "Message already received.";
        return -1; // do not relay message again
    }
    if (std::abs(nCurrentTimeStamp - data.nTimeStamp) > MAX_GLOBAL_MESSAGE_DRIFT_SECONDS)
    {
        strErrorMessage = "Message exceeds maximum time drift.";
        return -2; // message too old or into the future (time drift exceeds maximum allowed)
    }
    if (data.vchFQDN.size() > MAX_OBJECT_FULL_PATH_LENGTH)
    {
        strErrorMessage = "Message length exceeds limit. Adding 10 to ban score.";
        return 10; // this will add 100 to the peer's ban score
    }
    if (vchGlobalData.size() > MAX_GLOBAL_MESSAGE_DATA_LENGTH)
    {
        strErrorMessage = "Signature size is too large. Adding 100 to ban score.";
        return 10; // this will add 100 to the peer's ban score
    }
    if (vchSignature.size() > MAX_GLOBAL_MESSAGE_SIGNATURE_SIZE)
    {
        strErrorMessage = "Signature size is too large. Adding 100 to ban score.";
        return 100; // this will add 100 to the peer's ban score
    }
    // Get the BDAP account entry
    CDomainEntry entry;
    // TODO (BDAP): Make sure local node has a synced blockchain
    if (!GetDomainEntry(data.vchFQDN, entry)) {
        strErrorMessage = "BDAP account in the global profile message not found. Adding 100 to ban score.";
        return 100; // this will add 100 to the peer's ban score
    }
    // Verify message wallet address pubkey matches BDAP account wallet address
    if (data.WalletAddress() != entry.GetWalletAddress()) {
        strErrorMessage = "Global profile message wallet address does not match BDAP entry. Adding 100 to ban score.";
        return 100; // this will add 100 to the peer's ban score
    }
    // Verify signature is correct for the given wallet address pubkey.
    if (!CheckSignature(data.vchWalletPubKey))
    {
        strErrorMessage = "Global profile message has an invalid signature. Adding 100 to ban score.";
        return 100; // this will add 100 to the peer's ban score
    }
    // TODO (BDAP): Write to BDAP account database

    return 0; // All checks okay, relay message to peers.
}

bool CGlobalProfile::RelayTo(CNode* pnode, CConnman& connman) const
{
    if (pnode->nVersion != 0 && pnode->nVersion >= MIN_GLOBAL_DATA_PEER_PROTO_VERSION)
    {
        CNetMsgMaker msgMaker(pnode->GetSendVersion());
        CGlobalData data(vchGlobalData);
        if (pnode->setKnown.insert(GetHash()).second) {
            if (GetAdjustedTime() < data.nTimeStamp) {
                connman.PushMessage(pnode, msgMaker.Make(NetMsgType::PROFILE, (*this)));
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

bool ReceivedGlobalDataMessage(const uint256& messageHash)
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
    }
    return false;
}