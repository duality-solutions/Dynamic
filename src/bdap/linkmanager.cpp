// Copyright (c) 2019-2021 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/linkmanager.h"

#include "bdap/domainentry.h"
#include "bdap/domainentrydb.h"
#include "bdap/linking.h"
#include "bdap/utils.h"
#include "bdap/vgp/include/encryption.h" // for VGP DecryptBDAPData
#include "dht/ed25519.h"
#include "pubkey.h"
#include "wallet/wallet.h"

CLinkManager* pLinkManager = NULL;

std::string CLink::LinkState() const
{
    if (nLinkState == 0) {
        return "Unknown";
    }
    else if (nLinkState == 1) {
        return "Pending";
    }
    else if (nLinkState == 2) {
        return "Complete";
    }
    else if (nLinkState == 3) {
        return "Deleted";
    }
    return "Undefined";
}

std::string CLink::ToString() const
{
    return strprintf(
            "CLink(\n"
            "    nVersion                   = %d\n"
            "    LinkID                     = %s\n"
            "    fRequestFromMe             = %s\n"
            "    fAcceptFromMe              = %s\n"
            "    LinkState                  = %s\n"
            "    RequestorFullObjectPath    = %s\n"
            "    RecipientFullObjectPath    = %s\n"
            "    RequestorPubKey            = %s\n"
            "    RecipientPubKey            = %s\n"
            "    SharedRequestPubKey        = %s\n"
            "    SharedAcceptPubKey         = %s\n"
            "    LinkMessage                = %s\n"
            "    nHeightRequest             = %d\n"
            "    nExpireTimeRequest         = %d\n"
            "    txHashRequest              = %s\n"
            "    nHeightAccept              = %d\n"
            "    nExpireTimeAccept          = %d\n"
            "    txHashAccept               = %s\n"
            "    SubjectID                  = %s\n"
            "    RequestorWalletAddress     = %s\n"
            "    RecipientWalletAddress     = %s\n"
            ")\n",
            nVersion,
            LinkID.ToString(),
            fRequestFromMe ? "true" : "false",
            fAcceptFromMe ? "true" : "false",
            LinkState(),
            stringFromVch(RequestorFullObjectPath),
            stringFromVch(RecipientFullObjectPath),
            stringFromVch(RequestorPubKey),
            stringFromVch(RecipientPubKey),
            stringFromVch(SharedRequestPubKey),
            stringFromVch(SharedAcceptPubKey),
            stringFromVch(LinkMessage),
            nHeightRequest,
            nExpireTimeRequest,
            txHashRequest.ToString(),
            nHeightAccept,
            nExpireTimeAccept,
            txHashAccept.ToString(),
            SubjectID.ToString(),
            stringFromVch(RequestorWalletAddress),
            stringFromVch(RecipientWalletAddress)
        );
}

std::string CLink::RequestorFQDN() const
{
    return stringFromVch(RequestorFullObjectPath);
}

std::string CLink::RecipientFQDN() const
{
    return stringFromVch(RecipientFullObjectPath);
}

std::string CLink::RequestorPubKeyString() const
{
    return stringFromVch(RequestorPubKey);
}

std::string CLink::RecipientPubKeyString() const
{
    return stringFromVch(RecipientPubKey);
}

bool CLinkManager::FindLink(const uint256& id, CLink& link)
{
    if (m_Links.count(id) > 0) {
        link = m_Links.at(id);
        return true;
    }
    return false;
}

bool CLinkManager::FindLinkBySubjectID(const uint256& subjectID, CLink& getLink)
{
    for (const std::pair<uint256, CLink>& link : m_Links)
    {
        if (link.second.SubjectID == subjectID) // pending request
        {
            getLink = link.second;
            return true;
        }
    }
    return false;
}

#ifndef ENABLE_WALLET
void CLinkManager::ProcessQueue()
{
    return;
}
#endif // ENABLE_WALLET

bool CLinkManager::ListMyPendingRequests(std::vector<CLink>& vchLinks)
{
    for (const std::pair<uint256, CLink>& link : m_Links)
    {
        if (link.second.nLinkState == 1 && link.second.fRequestFromMe) // pending request
        {
            vchLinks.push_back(link.second);
        }
    }
    return true;
}

bool CLinkManager::ListMyPendingAccepts(std::vector<CLink>& vchLinks)
{
    for (const std::pair<uint256, CLink>& link : m_Links)
    {
        //LogPrintf("%s -- link:\n%s\n", __func__, link.second.ToString());
        if (link.second.nLinkState == 1 && (!link.second.fRequestFromMe || (link.second.fRequestFromMe && link.second.fAcceptFromMe))) // pending accept
        {
            vchLinks.push_back(link.second);
        }
    }
    return true;
}

bool CLinkManager::ListMyCompleted(std::vector<CLink>& vchLinks)
{
    for (const std::pair<uint256, CLink>& link : m_Links)
    {
        if (link.second.nLinkState == 2 && !link.second.txHashRequest.IsNull()) // completed link
        {
            vchLinks.push_back(link.second);
        }
    }
    return true;
}

#ifndef ENABLE_WALLET
bool CLinkManager::ProcessLink(const CLinkStorage& storage, const bool fStoreInQueueOnly)
{
        linkQueue.push(storage);
        return true;
}
#endif // ENABLE_WALLET

std::vector<CLinkInfo> CLinkManager::GetCompletedLinkInfo(const std::vector<unsigned char>& vchFullObjectPath)
{
    std::vector<CLinkInfo> vchLinkInfo;
    for(const std::pair<uint256, CLink>& link : m_Links)
    {
        if (link.second.nLinkState == 2) // completed link
        {
            if (link.second.RequestorFullObjectPath == vchFullObjectPath)
            {
                CLinkInfo linkInfo(link.second.RecipientFullObjectPath, link.second.RecipientPubKey, link.second.RequestorPubKey);
                vchLinkInfo.push_back(linkInfo);
            }
            else if (link.second.RecipientFullObjectPath == vchFullObjectPath)
            {
                CLinkInfo linkInfo(link.second.RequestorFullObjectPath, link.second.RequestorPubKey, link.second.RecipientPubKey);
                vchLinkInfo.push_back(linkInfo);
            }
        }
    }
    return vchLinkInfo;
}

int CLinkManager::IsMyMessage(const uint256& subjectID, const uint256& messageID, const int64_t& timestamp)
{
    std::vector<unsigned char> vchPubKey;
    if (GetLinkMessageInfo(subjectID, vchPubKey))
    {
        if (messageID != GetMessageID(vchPubKey, timestamp))
        {
            // Incorrect message id. Might be spoofed.
            return -100;
        }
        return 1;
    }
    return 0;
}

void CLinkManager::LoadLinkMessageInfo(const uint256& subjectID, const std::vector<unsigned char>& vchPubKey)
{
    if (m_LinkMessageInfo.count(subjectID) == 0)
        m_LinkMessageInfo[subjectID] = vchPubKey;
}

bool CLinkManager::GetLinkMessageInfo(const uint256& subjectID, std::vector<unsigned char>& vchPubKey)
{
    std::map<uint256, std::vector<unsigned char>>::iterator it = m_LinkMessageInfo.find(subjectID);
    if (it != m_LinkMessageInfo.end()) {
        vchPubKey = it->second;
        return true; // found subjectID
    }
    return false; // doesn't exist
}

uint256 GetLinkID(const CLinkRequest& request)
{
    std::vector<unsigned char> vchLinkPath = request.LinkPath();
    return Hash(vchLinkPath.begin(), vchLinkPath.end());
}

uint256 GetLinkID(const CLinkAccept& accept)
{
    std::vector<unsigned char> vchLinkPath = accept.LinkPath();
    return Hash(vchLinkPath.begin(), vchLinkPath.end());
}

uint256 GetLinkID(const std::string& account1, const std::string& account2)
{
    if (account1 != account2) {
        std::vector<unsigned char> vchSeparator = {':'};
        std::set<std::string> sorted;
        sorted.insert(account1);
        sorted.insert(account2);
        std::set<std::string>::iterator it = sorted.begin();
        std::vector<unsigned char> vchLink1 = vchFromString(*it);
        std::advance(it, 1);
        std::vector<unsigned char> vchLink2 = vchFromString(*it);
        vchLink1.insert(vchLink1.end(), vchSeparator.begin(), vchSeparator.end());
        vchLink1.insert(vchLink1.end(), vchLink2.begin(), vchLink2.end());
        return Hash(vchLink1.begin(), vchLink1.end());
    }
    return uint256();
}

uint256 GetMessageID(const std::vector<unsigned char>& vchPubKey, const int64_t& timestamp)
{
    CScript scriptMessage;
    scriptMessage << vchPubKey << timestamp;
    return Hash(scriptMessage.begin(), scriptMessage.end());
}

uint256 GetMessageID(const CKeyEd25519& key, const int64_t& timestamp)
{
    return GetMessageID(key.GetPubKeyBytes(), timestamp);
}
