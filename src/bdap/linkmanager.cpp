// Copyright (c) 2019 Duality Blockchain Solutions Developers
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

//#ifdef ENABLE_WALLET

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
            "    SharedLinkPubKey           = %s\n"
            "    LinkMessage                = %s\n"
            "    nHeightRequest             = %d\n"
            "    nExpireTimeRequest         = %d\n"
            "    txHashRequest              = %s\n"
            "    nHeightAccept              = %d\n"
            "    nExpireTimeAccept          = %d\n"
            "    txHashAccept               = %s\n"
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
            stringFromVch(SharedLinkPubKey),
            stringFromVch(LinkMessage),
            nHeightRequest,
            nExpireTimeRequest,
            txHashRequest.ToString(),
            nHeightAccept,
            nExpireTimeAccept,
            txHashAccept.ToString()
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

bool CLinkManager::IsLinkFromMe(const std::vector<unsigned char>& vchLinkPubKey)
{
    if (!pwalletMain)
        return false;

    CKeyID keyID(Hash160(vchLinkPubKey.begin(), vchLinkPubKey.end()));
    CKeyEd25519 keyOut;
    if (pwalletMain->GetDHTKey(keyID, keyOut))
        return true;

    return false;
}

bool CLinkManager::IsLinkForMe(const std::vector<unsigned char>& vchLinkPubKey, const std::vector<unsigned char>& vchSharedPubKey)
{
    if (!pwalletMain)
        return false;

    std::vector<std::vector<unsigned char>> vvchMyDHTPubKeys;
    if (!pwalletMain->GetDHTPubKeys(vvchMyDHTPubKeys))
        return false;

    if (vvchMyDHTPubKeys.size() == 0)
        return false;

    for (const std::vector<unsigned char>& vchMyDHTPubKey : vvchMyDHTPubKeys) {
        CKeyID keyID(Hash160(vchMyDHTPubKey.begin(), vchMyDHTPubKey.end()));
        CKeyEd25519 dhtKey;
        if (pwalletMain->GetDHTKey(keyID, dhtKey)) {
            std::vector<unsigned char> vchGetSharedPubKey = GetLinkSharedPubKey(dhtKey, vchLinkPubKey);
            if (vchGetSharedPubKey == vchSharedPubKey)
                return true;
        }
    }

    return false;
}

bool CLinkManager::GetLinkPrivateKey(const std::vector<unsigned char>& vchSenderPubKey, const std::vector<unsigned char>& vchSharedPubKey, std::array<char, 32>& sharedSeed, std::string& strErrorMessage)
{
    if (!pwalletMain)
        return false;

    std::vector<std::vector<unsigned char>> vvchDHTPubKeys;
    if (!pwalletMain->GetDHTPubKeys(vvchDHTPubKeys)) {
        strErrorMessage = "Error getting DHT key vector.";
        return false;
    }
    // loop through each account key to check if it matches the shared key
    for (const std::vector<unsigned char>& vchPubKey : vvchDHTPubKeys) {
        CDomainEntry entry;
        if (pDomainEntryDB->ReadDomainEntryPubKey(vchPubKey, entry)) {
            CKeyEd25519 dhtKey;
            CKeyID keyID(Hash160(vchPubKey.begin(), vchPubKey.end()));
            if (pwalletMain->GetDHTKey(keyID, dhtKey)) {
                if (vchSharedPubKey == GetLinkSharedPubKey(dhtKey, vchSenderPubKey)) {
                    sharedSeed = GetLinkSharedPrivateKey(dhtKey, vchSenderPubKey);
                    return true;
                }
            }
            else {
                strErrorMessage = strErrorMessage + "Error getting DHT private key.\n";
            }
        }
    }
    return false;
}

bool CLinkManager::FindLink(const uint256& id, CLink& link)
{
    if (m_Links.count(id) > 0) {
        link = m_Links.at(id);
        return true;
    }
    return false;
}

void CLinkManager::ProcessQueue()
{
    if (!pwalletMain)
        return;

    if (pwalletMain->IsLocked())
        return;

    while (!linkQueue.empty())
    {
        // TODO (BDAP): Do we need to lock the queue while processing?
        CLinkStorage storage = linkQueue.front();
        ProcessLink(storage);
        linkQueue.pop();
    }
}

bool CLinkManager::ListMyPendingRequests(std::vector<CLink>& vchLinks)
{
    for(const std::pair<uint256, CLink>& link : m_Links)
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

    for(const std::pair<uint256, CLink>& link : m_Links)
    {
        if (link.second.nLinkState == 1 && !link.second.fRequestFromMe) // pending accept
        {
            vchLinks.push_back(link.second);
        }
    }
    return true;
}

bool CLinkManager::ListMyCompleted(std::vector<CLink>& vchLinks)
{
    for(const std::pair<uint256, CLink>& link : m_Links)
    {
        if (link.second.nLinkState == 2) // completed link
        {
            vchLinks.push_back(link.second);
        }
    }
    return true;
}

bool CLinkManager::ProcessLink(const CLinkStorage& storage, const bool fStoreInQueueOnly)
{
    if (fStoreInQueueOnly || pwalletMain->IsLocked()) {
        linkQueue.push(storage);
        return true;
    }
    int nDataVersion = -1;
    if (!storage.Encrypted())
    {
        if (storage.nType == 1) // Clear text link request
        {
            std::vector<unsigned char> vchData = RemoveVersionFromLinkData(storage.vchRawData, nDataVersion);
            CLinkRequest link(vchData, storage.txHash);
            link.nHeight = storage.nHeight;
            link.txHash = storage.txHash;
            link.nExpireTime = storage.nExpireTime;
            CDomainEntry entry;
            if (GetDomainEntry(link.RequestorFullObjectPath, entry)) {
                if (SignatureProofIsValid(entry.GetWalletAddress(), link.RecipientFQDN(), link.SignatureProof)) {
                    bool fIsLinkFromMe = IsLinkFromMe(storage.vchLinkPubKey);
                    //bool fIsLinkForMe = IsLinkForMe(storage.vchLinkPubKey, storage.vchSharedPubKey);
                    LogPrintf("%s -- Link request from me found with a valid signature proof. Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(storage.vchLinkPubKey));
                    uint256 linkID = GetLinkID(link);
                    CLink record;
                    std::map<uint256, CLink>::iterator it = m_Links.find(linkID);
                    if (it != m_Links.end()) {
                        record = it->second;
                    }
                    record.LinkID = linkID;
                    record.fRequestFromMe = fIsLinkFromMe;
                    record.nLinkState = 1;
                    record.RequestorFullObjectPath = link.RequestorFullObjectPath;
                    record.RecipientFullObjectPath = link.RecipientFullObjectPath;
                    record.RequestorPubKey = link.RequestorPubKey;
                    record.SharedRequestPubKey = link.SharedPubKey;
                    record.LinkMessage = link.LinkMessage;
                    record.nHeightRequest = link.nHeight;
                    record.nExpireTimeRequest = link.nExpireTime;
                    record.txHashRequest = link.txHash;
                    LogPrintf("%s -- Added to map id = %s\n", __func__, linkID.ToString());
                    m_Links[linkID] = record;
                }
                else
                    LogPrintf("%s ***** Warning. Link request found with an invalid signature proof! Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(storage.vchLinkPubKey));
            }
            else {
                LogPrintf("%s -- Link request GetDomainEntry failed.\n", __func__);
                return false;
            }
        }
        else if (storage.nType == 2) // Clear text accept
        {
            std::vector<unsigned char> vchData = RemoveVersionFromLinkData(storage.vchRawData, nDataVersion);
            CLinkAccept link(vchData, storage.txHash);;
            link.nHeight = storage.nHeight;
            link.txHash = storage.txHash;
            link.nExpireTime = storage.nExpireTime;
            CDomainEntry entry;
            if (GetDomainEntry(link.RecipientFullObjectPath, entry)) {
                if (SignatureProofIsValid(entry.GetWalletAddress(), link.RequestorFQDN(), link.SignatureProof)) {
                    bool fIsLinkFromMe = IsLinkFromMe(storage.vchLinkPubKey);
                    //bool fIsLinkForMe = IsLinkForMe(storage.vchLinkPubKey, storage.vchSharedPubKey);
                    LogPrintf("%s -- Link accept from me found with a valid signature proof. Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(storage.vchLinkPubKey));
                    uint256 linkID = GetLinkID(link);
                    CLink record;
                    std::map<uint256, CLink>::iterator it = m_Links.find(linkID);
                    if (it != m_Links.end()) {
                        record = it->second;
                    }
                    record.LinkID = linkID;
                    record.fAcceptFromMe = fIsLinkFromMe;
                    record.nLinkState = 2;
                    record.RequestorFullObjectPath = link.RequestorFullObjectPath;
                    record.RecipientFullObjectPath = link.RecipientFullObjectPath;
                    record.RecipientPubKey = link.RecipientPubKey;
                    record.SharedLinkPubKey = link.SharedPubKey;
                    record.nHeightAccept = link.nHeight;
                    record.nExpireTimeAccept = link.nExpireTime;
                    record.txHashAccept = link.txHash;
                    LogPrintf("%s -- Added to map id = %s\n", __func__, linkID.ToString());
                    m_Links[linkID] = record;
                }
                else
                    LogPrintf("%s -- Warning! Link accept found with an invalid signature proof! Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(storage.vchLinkPubKey));
            }
            else {
                LogPrintf("%s -- Link accept GetDomainEntry failed.\n", __func__);
                return false;
            }
        }
    }
    else if (storage.Encrypted() && !pwalletMain->IsLocked())
    {
        bool fIsLinkFromMe = IsLinkFromMe(storage.vchLinkPubKey);
        bool fIsLinkForMe = IsLinkForMe(storage.vchLinkPubKey, storage.vchSharedPubKey);
        if (storage.nType == 1 && fIsLinkFromMe) // Encrypted link request from me
        {
            LogPrintf("%s -- Version 1 link request from me found! vchLinkPubKey = %s\n", __func__, stringFromVch(storage.vchLinkPubKey));
            CKeyEd25519 privDHTKey;
            CKeyID keyID(Hash160(storage.vchLinkPubKey.begin(), storage.vchLinkPubKey.end()));
            if (pwalletMain->GetDHTKey(keyID, privDHTKey)) {
                std::vector<unsigned char> vchData = RemoveVersionFromLinkData(storage.vchRawData, nDataVersion);
                std::string strMessage = "";
                std::vector<unsigned char> dataDecrypted;
                if (DecryptBDAPData(privDHTKey.GetPrivSeedBytes(), vchData, dataDecrypted, strMessage)) {
                    std::vector<unsigned char> vchData, vchHash;
                    CScript scriptData;
                    scriptData << OP_RETURN << dataDecrypted;
                    if (GetBDAPData(scriptData, vchData, vchHash)) {
                        CLinkRequest link(dataDecrypted, storage.txHash);
                        CDomainEntry entry;
                        if (!GetDomainEntry(link.RequestorFullObjectPath, entry)) {
                            LogPrintf("%s -- Failed to get link requestor %s\n", __func__, stringFromVch(link.RequestorFullObjectPath));
                            return false;
                        }
                        if (!SignatureProofIsValid(entry.GetWalletAddress(), link.RecipientFQDN(), link.SignatureProof)) {
                            LogPrintf("%s ***** Warning. Link request found with an invalid signature proof! Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(storage.vchLinkPubKey));
                            return false;
                        }
                        link.nHeight = storage.nHeight;
                        link.nExpireTime = storage.nExpireTime;
                        LogPrintf("%s -- DecryptBDAPData RequestorFQDN = %s, RecipientFQDN = %s, dataDecrypted size = %i\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), dataDecrypted.size());
                        uint256 linkID = GetLinkID(link);
                        CLink record;
                        std::map<uint256, CLink>::iterator it = m_Links.find(linkID);
                        if (it != m_Links.end()) {
                            record = it->second;
                        }
                        record.LinkID = linkID;
                        record.fRequestFromMe = fIsLinkFromMe;
                        record.nLinkState = 1;
                        record.RequestorFullObjectPath = link.RequestorFullObjectPath;
                        record.RecipientFullObjectPath = link.RecipientFullObjectPath;
                        record.RequestorPubKey = link.RequestorPubKey;
                        record.SharedRequestPubKey = link.SharedPubKey;
                        record.LinkMessage = link.LinkMessage;
                        record.nHeightRequest = link.nHeight;
                        record.nExpireTimeRequest = link.nExpireTime;
                        record.txHashRequest = link.txHash;
                        LogPrintf("%s -- Added to map id = %s\n", __func__, linkID.ToString());
                        m_Links[linkID] = record;
                        
                    }
                    else {
                        LogPrintf("%s -- Link request GetBDAPData failed.\n", __func__);
                        return false;
                    }
                }
                else {
                    LogPrintf("%s -- Link request DecryptBDAPData failed.\n", __func__);
                    return false;
                }
            }
            else {
                LogPrintf("%s -- Link request GetDHTKey failed.\n", __func__);
                return false;
            }
        }
        else if (storage.nType == 1 && !fIsLinkFromMe && fIsLinkForMe) // Encrypted link request for me
        {
            LogPrintf("%s -- Version 1 link request for me found! vchLinkPubKey = %s\n", __func__, stringFromVch(storage.vchLinkPubKey));
            CKeyEd25519 sharedDHTKey;
            std::array<char, 32> sharedSeed;
            std::string strErrorMessage;
            if (GetLinkPrivateKey(storage.vchLinkPubKey, storage.vchSharedPubKey, sharedSeed, strErrorMessage)) {
                CKeyEd25519 sharedKey(sharedSeed);
                std::vector<unsigned char> vchData = RemoveVersionFromLinkData(storage.vchRawData, nDataVersion);
                std::string strMessage = "";
                std::vector<unsigned char> dataDecrypted;
                if (DecryptBDAPData(sharedKey.GetPrivSeedBytes(), vchData, dataDecrypted, strMessage)) {
                    std::vector<unsigned char> vchData, vchHash;
                    CScript scriptData;
                    scriptData << OP_RETURN << dataDecrypted;
                    if (GetBDAPData(scriptData, vchData, vchHash)) {
                        CLinkRequest link(dataDecrypted, storage.txHash);
                        CDomainEntry entry;
                        if (!GetDomainEntry(link.RequestorFullObjectPath, entry)) {
                            LogPrintf("%s -- Failed to get link requestor %s\n", __func__, stringFromVch(link.RequestorFullObjectPath));
                            return false;
                        }
                        if (!SignatureProofIsValid(entry.GetWalletAddress(), link.RecipientFQDN(), link.SignatureProof)) {
                            LogPrintf("%s ***** Warning. Link request found with an invalid signature proof! Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(storage.vchLinkPubKey));
                            return false;
                        }
                        link.nHeight = storage.nHeight;
                        link.nExpireTime = storage.nExpireTime;
                        LogPrintf("%s -- DecryptBDAPData RequestorFQDN = %s, RecipientFQDN = %s, dataDecrypted size = %i\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), dataDecrypted.size());
                        uint256 linkID = GetLinkID(link);
                        CLink record;
                        std::map<uint256, CLink>::iterator it = m_Links.find(linkID);
                        if (it != m_Links.end()) {
                            record = it->second;
                        }
                        record.LinkID = linkID;
                        record.fRequestFromMe = fIsLinkFromMe;
                        record.nLinkState = 1;
                        record.RequestorFullObjectPath = link.RequestorFullObjectPath;
                        record.RecipientFullObjectPath = link.RecipientFullObjectPath;
                        record.RequestorPubKey = link.RequestorPubKey;
                        record.SharedRequestPubKey = link.SharedPubKey;
                        record.LinkMessage = link.LinkMessage;
                        record.nHeightRequest = link.nHeight;
                        record.nExpireTimeRequest = link.nExpireTime;
                        record.txHashRequest = link.txHash;
                        LogPrintf("%s -- Added to map id = %s\n", __func__, linkID.ToString());
                        m_Links[linkID] = record;
                    }
                    else {
                        LogPrintf("%s -- Link request GetBDAPData failed.\n", __func__);
                        return false;
                    }
                }
                else {
                    LogPrintf("%s -- Link request DecryptBDAPData failed.\n", __func__);
                    return false;
                }
            }
            else {
                LogPrintf("%s -- Link request GetLinkPrivateKey failed.\n", __func__);
                return false;
            }
        }
        else if (storage.nType == 2 && fIsLinkFromMe) // Link accept from me
        {
            LogPrintf("%s -- Version 1 encrypted link accept from me found! vchLinkPubKey = %s\n", __func__, stringFromVch(storage.vchLinkPubKey));
            CKeyEd25519 privDHTKey;
            CKeyID keyID(Hash160(storage.vchLinkPubKey.begin(), storage.vchLinkPubKey.end()));
            if (pwalletMain->GetDHTKey(keyID, privDHTKey)) {
                std::vector<unsigned char> vchData = RemoveVersionFromLinkData(storage.vchRawData, nDataVersion);
                std::string strMessage = "";
                std::vector<unsigned char> dataDecrypted;
                if (DecryptBDAPData(privDHTKey.GetPrivSeedBytes(), vchData, dataDecrypted, strMessage)) {
                    std::vector<unsigned char> vchData, vchHash;
                    CScript scriptData;
                    scriptData << OP_RETURN << dataDecrypted;
                    if (GetBDAPData(scriptData, vchData, vchHash)) {
                        CLinkAccept link(dataDecrypted, storage.txHash);
                        CDomainEntry entry;
                        if (!GetDomainEntry(link.RecipientFullObjectPath, entry)) {
                            LogPrintf("%s -- Failed to get link recipient %s\n", __func__, stringFromVch(link.RecipientFullObjectPath));
                            return false;
                        }
                        if (!SignatureProofIsValid(entry.GetWalletAddress(), link.RequestorFQDN(), link.SignatureProof)) {
                            LogPrintf("%s ***** Warning. Link accept found with an invalid signature proof! Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(storage.vchLinkPubKey));
                            return false;
                        }
                        link.nHeight = storage.nHeight;
                        link.nExpireTime = storage.nExpireTime;
                        LogPrintf("%s -- DecryptBDAPData RequestorFQDN = %s, RecipientFQDN = %s, dataDecrypted size = %i\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), dataDecrypted.size());
                        uint256 linkID = GetLinkID(link);
                        CLink record;
                        std::map<uint256, CLink>::iterator it = m_Links.find(linkID);
                        if (it != m_Links.end()) {
                            record = it->second;
                        }
                        record.LinkID = linkID;
                        record.fAcceptFromMe = fIsLinkFromMe;
                        record.nLinkState = 2;
                        record.RequestorFullObjectPath = link.RequestorFullObjectPath;
                        record.RecipientFullObjectPath = link.RecipientFullObjectPath;
                        record.RecipientPubKey = link.RecipientPubKey;
                        record.SharedLinkPubKey = link.SharedPubKey;
                        record.nHeightAccept = link.nHeight;
                        record.nExpireTimeAccept = link.nExpireTime;
                        record.txHashAccept = link.txHash;
                        LogPrintf("%s -- Added to map id = %s\n", __func__, linkID.ToString());
                        m_Links[linkID] = record;
                    }
                    else {
                        LogPrintf("%s -- Link accept GetBDAPData failed.\n", __func__);
                        return false;
                    }
                }
                else {
                    LogPrintf("%s -- Link accept DecryptBDAPData failed.\n", __func__);
                    return false;
                }
            }
            else {
                LogPrintf("%s -- Link accept GetDHTKey failed.\n", __func__);
                return false;
            }
        }
        else if (storage.nType == 2 && !fIsLinkFromMe && fIsLinkForMe) // Link accept for me
        {
            LogPrintf("%s -- Version 1 link accept for me found! vchLinkPubKey = %s\n", __func__, stringFromVch(storage.vchLinkPubKey));
            CKeyEd25519 sharedDHTKey;
            std::array<char, 32> sharedSeed;
            std::string strErrorMessage;
            if (GetLinkPrivateKey(storage.vchLinkPubKey, storage.vchSharedPubKey, sharedSeed, strErrorMessage)) {
                CKeyEd25519 sharedKey(sharedSeed);
                std::vector<unsigned char> vchData = RemoveVersionFromLinkData(storage.vchRawData, nDataVersion);
                std::string strMessage = "";
                std::vector<unsigned char> dataDecrypted;
                if (DecryptBDAPData(sharedKey.GetPrivSeedBytes(), vchData, dataDecrypted, strMessage)) {
                    std::vector<unsigned char> vchData, vchHash;
                    CScript scriptData;
                    scriptData << OP_RETURN << dataDecrypted;
                    if (GetBDAPData(scriptData, vchData, vchHash)) {
                        CLinkAccept link(dataDecrypted, storage.txHash);
                        CDomainEntry entry;
                        if (!GetDomainEntry(link.RecipientFullObjectPath, entry)) {
                            LogPrintf("%s -- Failed to get link recipient %s\n", __func__, stringFromVch(link.RecipientFullObjectPath));
                            return false;
                        }
                        if (!SignatureProofIsValid(entry.GetWalletAddress(), link.RequestorFQDN(), link.SignatureProof)) {
                            LogPrintf("%s ***** Warning. Link accept found with an invalid signature proof! Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(storage.vchLinkPubKey));
                            return false;
                        }
                        link.nHeight = storage.nHeight;
                        link.nExpireTime = storage.nExpireTime;
                        LogPrintf("%s -- DecryptBDAPData RequestorFQDN = %s, RecipientFQDN = %s, dataDecrypted size = %i\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), dataDecrypted.size());
                        uint256 linkID = GetLinkID(link);
                        CLink record;
                        std::map<uint256, CLink>::iterator it = m_Links.find(linkID);
                        if (it != m_Links.end()) {
                            record = it->second;
                        }
                        record.LinkID = linkID;
                        record.fAcceptFromMe = fIsLinkFromMe;
                        record.nLinkState = 2;
                        record.RequestorFullObjectPath = link.RequestorFullObjectPath;
                        record.RecipientFullObjectPath = link.RecipientFullObjectPath;
                        record.RecipientPubKey = link.RecipientPubKey;
                        record.SharedLinkPubKey = link.SharedPubKey;
                        record.nHeightAccept = link.nHeight;
                        record.nExpireTimeAccept = link.nExpireTime;
                        record.txHashAccept = link.txHash;
                        LogPrintf("%s -- Added to map id = %s\n", __func__, linkID.ToString());
                        m_Links[linkID] = record;
                    }
                    else {
                        LogPrintf("%s -- Link accept GetBDAPData failed.\n", __func__);
                        return false;
                    }
                }
                else {
                    LogPrintf("%s -- Link accept DecryptBDAPData failed.\n", __func__);
                    return false;
                }
            }
            else {
                LogPrintf("%s -- Link accept GetLinkPrivateKey failed.\n", __func__);
                return false;
            }
        }
        else
        {
            linkQueue.push(storage);
        }
    }
    return true;
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

//#endif // ENABLE_WALLET