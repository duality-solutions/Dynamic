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
        strErrorMessage = "Failed to get DHT key vector.";
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

void CLinkManager::ProcessQueue()
{
    if (!pwalletMain)
        return;

    if (pwalletMain->IsLocked())
        return;

    // make sure we are not stuck in an infinite loop
    size_t size = QueueSize();
    size_t counter = 0;
    LogPrintf("CLinkManager::%s -- Start links in queue = %d\n", __func__, size);
    while (!linkQueue.empty() && size > counter)
    {
        // TODO (BDAP): Do we need to lock the queue while processing?
        CLinkStorage storage = linkQueue.front();
        ProcessLink(storage);
        linkQueue.pop();
        counter++;
    }
    LogPrintf("CLinkManager::%s -- Finished links in queue = %d\n", __func__, QueueSize());
}

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

bool CLinkManager::ProcessLink(const CLinkStorage& storage, const bool fStoreInQueueOnly)
{

    if (!pwalletMain) {
        linkQueue.push(storage);
        return true;
    }

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
            LogPrint("bdap", "%s -- %s\n", __func__, link.ToString());
            link.nHeight = storage.nHeight;
            link.txHash = storage.txHash;
            link.nExpireTime = storage.nExpireTime;
            CDomainEntry entry;
            if (GetDomainEntry(link.RequestorFullObjectPath, entry)) {
                if (SignatureProofIsValid(entry.GetWalletAddress(), link.RecipientFQDN(), link.SignatureProof)) {
                    bool fIsLinkFromMe = IsLinkFromMe(storage.vchLinkPubKey);
                    LogPrint("bdap", "%s -- Link request from me found with a valid signature proof. Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(storage.vchLinkPubKey));
                    uint256 linkID = GetLinkID(link);
                    CLink record;
                    std::map<uint256, CLink>::iterator it = m_Links.find(linkID);
                    if (it != m_Links.end()) {
                        record = it->second;
                    }
                    record.LinkID = linkID;
                    record.fRequestFromMe = fIsLinkFromMe;
                    if (record.nHeightAccept > 0) {
                        record.nLinkState = 2;
                    }
                    else {
                        record.nLinkState = 1;
                    }
                    record.RequestorFullObjectPath = link.RequestorFullObjectPath;
                    record.RecipientFullObjectPath = link.RecipientFullObjectPath;
                    record.RequestorPubKey = link.RequestorPubKey;
                    record.SharedRequestPubKey = link.SharedPubKey;
                    record.LinkMessage = link.LinkMessage;
                    record.nHeightRequest = link.nHeight;
                    record.nExpireTimeRequest = link.nExpireTime;
                    record.txHashRequest = link.txHash;
                    record.RequestorWalletAddress = entry.WalletAddress;
                    if (record.SharedAcceptPubKey.size() > 0 && record.SharedRequestPubKey.size() > 0)
                    {
                        std::string strErrorMessage = "";
                        if (!GetMessageInfo(record, strErrorMessage))
                        {
                            LogPrintf("%s -- Error getting message info %s\n", __func__, strErrorMessage);
                        }
                        else
                        {
                            pwalletMain->WriteLinkMessageInfo(record.SubjectID, record.vchSecretPubKeyBytes);
                            m_LinkMessageInfo[record.SubjectID] = record.vchSecretPubKeyBytes;
                        }
                        //LogPrintf("%s -- link request = %s\n", __func__, record.ToString());
                    }
                    LogPrint("bdap", "%s -- Clear text link request added to map id = %s\n", __func__, linkID.ToString());
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
            CLinkAccept link(vchData, storage.txHash);
            LogPrint("bdap", "%s -- %s\n", __func__, link.ToString());
            link.nHeight = storage.nHeight;
            link.txHash = storage.txHash;
            link.nExpireTime = storage.nExpireTime;
            CDomainEntry entry;
            if (GetDomainEntry(link.RecipientFullObjectPath, entry)) {
                if (SignatureProofIsValid(entry.GetWalletAddress(), link.RequestorFQDN(), link.SignatureProof)) {
                    bool fIsLinkFromMe = IsLinkFromMe(storage.vchLinkPubKey);
                    //bool fIsLinkForMe = IsLinkForMe(storage.vchLinkPubKey, storage.vchSharedPubKey);
                    LogPrint("bdap", "%s -- Link accept from me found with a valid signature proof. Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(storage.vchLinkPubKey));
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
                    record.SharedAcceptPubKey = link.SharedPubKey;
                    record.nHeightAccept = link.nHeight;
                    record.nExpireTimeAccept = link.nExpireTime;
                    record.txHashAccept = link.txHash;
                    record.RecipientWalletAddress = entry.WalletAddress;
                    if (record.SharedAcceptPubKey.size() > 0 && record.SharedRequestPubKey.size() > 0)
                    {
                        std::string strErrorMessage = "";
                        if (!GetMessageInfo(record, strErrorMessage))
                        {
                            LogPrintf("%s -- Error getting message info %s\n", __func__, strErrorMessage);
                        }
                        else
                        {
                            pwalletMain->WriteLinkMessageInfo(record.SubjectID, record.vchSecretPubKeyBytes);
                            m_LinkMessageInfo[record.SubjectID] = record.vchSecretPubKeyBytes;
                        }
                        //LogPrintf("%s -- link accept = %s\n", __func__, record.ToString());
                    }
                    LogPrint("bdap", "%s -- Clear text accept added to map id = %s, %s\n", __func__, linkID.ToString(), record.ToString());
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
        if (!fIsLinkFromMe && !fIsLinkForMe) {
            // This happens if you lose your DHT private key but have the BDAP account link wallet private key.
            LogPrintf("%s -- ** Warning: Encrypted link received but can not process it: TxID = %s\n", __func__, storage.txHash.ToString());
            return false;
        }

        if (storage.nType == 1 && fIsLinkFromMe) // Encrypted link request from me
        {
            //LogPrintf("%s -- Version 1 link request from me found! vchLinkPubKey = %s\n", __func__, stringFromVch(storage.vchLinkPubKey));
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
                        LogPrint("bdap", "%s -- %s\n", __func__, link.ToString());
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
                        uint256 linkID = GetLinkID(link);
                        CLink record;
                        std::map<uint256, CLink>::iterator it = m_Links.find(linkID);
                        if (it != m_Links.end()) {
                            record = it->second;
                        }
                        record.LinkID = linkID;
                        record.fRequestFromMe = fIsLinkFromMe;
                        record.fAcceptFromMe =  (fIsLinkFromMe && fIsLinkForMe);
                        if (record.nHeightAccept > 0) {
                            record.nLinkState = 2;
                        }
                        else {
                            record.nLinkState = 1;
                        }
                        record.RequestorFullObjectPath = link.RequestorFullObjectPath;
                        record.RecipientFullObjectPath = link.RecipientFullObjectPath;
                        record.RequestorPubKey = link.RequestorPubKey;
                        record.SharedRequestPubKey = link.SharedPubKey;
                        record.LinkMessage = link.LinkMessage;
                        record.nHeightRequest = link.nHeight;
                        record.nExpireTimeRequest = link.nExpireTime;
                        record.txHashRequest = link.txHash;
                        record.RequestorWalletAddress = entry.WalletAddress;
                        if (record.SharedAcceptPubKey.size() > 0 && record.SharedRequestPubKey.size() > 0)
                        {
                            std::string strErrorMessage = "";
                            if (!GetMessageInfo(record, strErrorMessage))
                            {
                                LogPrintf("%s -- Error getting message info %s\n", __func__, strErrorMessage);
                            }
                            else
                            {
                                pwalletMain->WriteLinkMessageInfo(record.SubjectID, record.vchSecretPubKeyBytes);
                                m_LinkMessageInfo[record.SubjectID] = record.vchSecretPubKeyBytes;
                            }
                            //LogPrintf("%s -- link request = %s\n", __func__, record.ToString());
                        }
                        LogPrint("bdap", "%s -- Encrypted link request from me added to map id = %s\n%s\n", __func__, linkID.ToString(), record.ToString());
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
            //LogPrintf("%s -- Version 1 link request for me found! vchLinkPubKey = %s\n", __func__, stringFromVch(storage.vchLinkPubKey));
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
                        LogPrint("bdap", "%s -- %s\n", __func__, link.ToString());
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
                        uint256 linkID = GetLinkID(link);
                        CLink record;
                        std::map<uint256, CLink>::iterator it = m_Links.find(linkID);
                        if (it != m_Links.end()) {
                            record = it->second;
                        }

                        record.LinkID = linkID;
                        record.fRequestFromMe = fIsLinkFromMe;
                        if (record.nHeightAccept > 0) {
                            record.nLinkState = 2;
                        }
                        else {
                            record.nLinkState = 1;
                        }
                        record.RequestorFullObjectPath = link.RequestorFullObjectPath;
                        record.RecipientFullObjectPath = link.RecipientFullObjectPath;
                        record.RequestorPubKey = link.RequestorPubKey;
                        record.SharedRequestPubKey = link.SharedPubKey;
                        record.LinkMessage = link.LinkMessage;
                        record.nHeightRequest = link.nHeight;
                        record.nExpireTimeRequest = link.nExpireTime;
                        record.txHashRequest = link.txHash;
                        record.RequestorWalletAddress = entry.WalletAddress;
                        if (record.SharedAcceptPubKey.size() > 0 && record.SharedRequestPubKey.size() > 0)
                        {
                            std::string strErrorMessage = "";
                            if (!GetMessageInfo(record, strErrorMessage))
                            {
                                LogPrintf("%s -- Error getting message info %s\n", __func__, strErrorMessage);
                            }
                            else
                            {
                                pwalletMain->WriteLinkMessageInfo(record.SubjectID, record.vchSecretPubKeyBytes);
                                m_LinkMessageInfo[record.SubjectID] = record.vchSecretPubKeyBytes;
                            }
                            //LogPrintf("%s -- link request = %s\n", __func__, record.ToString());
                        }
                        LogPrint("bdap", "%s -- Encrypted link request for me added to map id = %s\n%s\n", __func__, linkID.ToString(), record.ToString());
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
            //LogPrintf("%s -- Version 1 encrypted link accept from me found! vchLinkPubKey = %s\n", __func__, stringFromVch(storage.vchLinkPubKey));
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
                        LogPrint("bdap", "%s -- %s\n", __func__, link.ToString());
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
                        uint256 linkID = GetLinkID(link);
                        CLink record;
                        std::map<uint256, CLink>::iterator it = m_Links.find(linkID);
                        if (it != m_Links.end()) {
                            record = it->second;
                        }
                        record.LinkID = linkID;
                        record.fRequestFromMe = (fIsLinkFromMe && fIsLinkForMe);
                        record.fAcceptFromMe = fIsLinkFromMe;
                        record.nLinkState = 2;
                        record.RequestorFullObjectPath = link.RequestorFullObjectPath;
                        record.RecipientFullObjectPath = link.RecipientFullObjectPath;
                        record.RecipientPubKey = link.RecipientPubKey;
                        record.SharedAcceptPubKey = link.SharedPubKey;
                        record.nHeightAccept = link.nHeight;
                        record.nExpireTimeAccept = link.nExpireTime;
                        record.txHashAccept = link.txHash;
                        record.RecipientWalletAddress = entry.WalletAddress;
                        if (record.SharedAcceptPubKey.size() > 0 && record.SharedRequestPubKey.size() > 0)
                        {
                            std::string strErrorMessage = "";
                            if (!GetMessageInfo(record, strErrorMessage))
                            {
                                LogPrintf("%s -- Error getting message info %s\n", __func__, strErrorMessage);
                            }
                            else
                            {
                                pwalletMain->WriteLinkMessageInfo(record.SubjectID, record.vchSecretPubKeyBytes);
                                m_LinkMessageInfo[record.SubjectID] = record.vchSecretPubKeyBytes;
                            }
                            //LogPrintf("%s -- accept request = %s\n", __func__, record.ToString());
                        }
                        LogPrint("bdap", "%s -- Encrypted link accept from me added to map id = %s\n%s\n", __func__, linkID.ToString(), record.ToString());
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
            //LogPrintf("%s -- Version 1 link accept for me found! vchLinkPubKey = %s\n", __func__, stringFromVch(storage.vchLinkPubKey));
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
                        LogPrint("bdap", "%s -- %s\n", __func__, link.ToString());
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
                        record.SharedAcceptPubKey = link.SharedPubKey;
                        record.nHeightAccept = link.nHeight;
                        record.nExpireTimeAccept = link.nExpireTime;
                        record.txHashAccept = link.txHash;
                        record.RecipientWalletAddress = entry.WalletAddress;
                        if (record.SharedAcceptPubKey.size() > 0 && record.SharedRequestPubKey.size() > 0)
                        {
                            std::string strErrorMessage = "";
                            if (!GetMessageInfo(record, strErrorMessage))
                            {
                                LogPrintf("%s -- Error getting message info %s\n", __func__, strErrorMessage);
                            }
                            else
                            {
                                pwalletMain->WriteLinkMessageInfo(record.SubjectID, record.vchSecretPubKeyBytes);
                                m_LinkMessageInfo[record.SubjectID] = record.vchSecretPubKeyBytes;
                            }
                            //LogPrintf("%s -- accept request = %s\n", __func__, record.ToString());
                        }
                        LogPrint("bdap", "%s -- Encrypted link accept for me added to map id = %s\n%s\n", __func__, linkID.ToString(), record.ToString());
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

bool GetSharedPrivateSeed(const CLink& link, std::array<char, 32>& seed, std::string& strErrorMessage)
{
    if (!pwalletMain)
        return false;

    if (link.nLinkState != 2)
        return false;

    //LogPrint("bdap", "%s -- %s\n", __func__, link.ToString());
    std::array<char, 32> sharedSeed1;
    std::array<char, 32> sharedSeed2;
    CDomainEntry entry;
    if (pDomainEntryDB->GetDomainEntryInfo(link.RecipientFullObjectPath, entry)) {
        if (link.fRequestFromMe) // Requestor
        {
            // first key exchange: requestor link pubkey + recipient account pubkey
            std::vector<unsigned char> vchRecipientPubKey = entry.DHTPublicKey;
            std::vector<unsigned char> vchRequestorPubKey = link.RequestorPubKey;
            CKeyEd25519 reqKey;
            CKeyID reqKeyID(Hash160(vchRequestorPubKey.begin(), vchRequestorPubKey.end()));
            if (pwalletMain->GetDHTKey(reqKeyID, reqKey)) {
                std::vector<unsigned char> vchGetLinkSharedPubKey = GetLinkSharedPubKey(reqKey, vchRecipientPubKey);
                if (link.SharedRequestPubKey == vchGetLinkSharedPubKey)
                {
                    sharedSeed1 = GetLinkSharedPrivateKey(reqKey, vchRecipientPubKey);
                }
                else
                {
                    strErrorMessage = strprintf("Requestor SharedRequestPubKey (%s) does not match derived shared request public key (%s).", 
                                            stringFromVch(link.SharedRequestPubKey), stringFromVch(vchGetLinkSharedPubKey));
                    return false;
                }
            }
            else {
                strErrorMessage = strprintf("Failed to get reqKey %s DHT private key.", stringFromVch(vchRequestorPubKey));
                return false;
            }
            // second key exchange: recipient link pubkey + requestor account pubkey
            CDomainEntry entryRequestor;
            if (pDomainEntryDB->GetDomainEntryInfo(link.RequestorFullObjectPath, entryRequestor))
            {
                std::vector<unsigned char> vchReqPubKey = entryRequestor.DHTPublicKey;
                std::vector<unsigned char> vchLinkPubKey = link.RecipientPubKey;
                CKeyEd25519 linkKey;
                CKeyID linkKeyID(Hash160(vchReqPubKey.begin(), vchReqPubKey.end()));
                if (pwalletMain->GetDHTKey(linkKeyID, linkKey)) {
                    std::vector<unsigned char> vchGetLinkSharedPubKey = GetLinkSharedPubKey(linkKey, vchLinkPubKey);
                    if (link.SharedAcceptPubKey == vchGetLinkSharedPubKey)
                    {
                        sharedSeed2 = GetLinkSharedPrivateKey(linkKey, vchLinkPubKey);
                    }
                    else
                    {
                        strErrorMessage = strprintf("Requestor SharedAcceptPubKey (%s) does not match derived shared link public key (%s).", 
                                                stringFromVch(link.SharedAcceptPubKey), stringFromVch(vchGetLinkSharedPubKey));
                        return false;
                    }
                }
                else {
                    strErrorMessage = strprintf("Failed to get requestor link Key %s DHT private key.", stringFromVch(vchLinkPubKey));
                    return false;
                }
            }
            else
            {
                strErrorMessage = strprintf("Can not find %s link requestor record.", stringFromVch(link.RequestorFullObjectPath));
                return false;
            }
        }
        else // Recipient
        {
            // first key exchange: requestor link pubkey + recipient account pubkey
            std::vector<unsigned char> vchRecipientPubKey = entry.DHTPublicKey;
            std::vector<unsigned char> vchRequestorPubKey = link.RequestorPubKey;
            CKeyEd25519 recKey;
            CKeyID recKeyID(Hash160(vchRecipientPubKey.begin(), vchRecipientPubKey.end()));
            if (pwalletMain->GetDHTKey(recKeyID, recKey))
            {
                std::vector<unsigned char> vchGetLinkSharedPubKey = GetLinkSharedPubKey(recKey, vchRequestorPubKey);
                if (link.SharedRequestPubKey == vchGetLinkSharedPubKey) {
                    sharedSeed1 = GetLinkSharedPrivateKey(recKey, vchRequestorPubKey);
                }
                else
                {
                    strErrorMessage = strprintf("Recipient SharedRequestPubKey (%s) does not match derived shared request public key (%s).",
                                        stringFromVch(link.SharedRequestPubKey), stringFromVch(vchGetLinkSharedPubKey));
                    return false;
                }
            }
            else {
                strErrorMessage = strprintf("Failed to get recKey %s DHT private key.", stringFromVch(vchRecipientPubKey));
                return false;
            }
            // second key exchange: recipient link pubkey + requestor account pubkey
            CDomainEntry entryRequestor;
            if (pDomainEntryDB->GetDomainEntryInfo(link.RequestorFullObjectPath, entryRequestor))
            {
                std::vector<unsigned char> vchLinkPubKey = link.RecipientPubKey;
                std::vector<unsigned char> vchReqPubKey = entryRequestor.DHTPublicKey;
                CKeyEd25519 linkKey;
                CKeyID linkKeyID(Hash160(vchLinkPubKey.begin(), vchLinkPubKey.end()));
                if (pwalletMain->GetDHTKey(linkKeyID, linkKey))
                {
                    std::vector<unsigned char> vchGetLinkSharedPubKey = GetLinkSharedPubKey(linkKey, vchReqPubKey);
                    if (link.SharedAcceptPubKey == vchGetLinkSharedPubKey) {
                        sharedSeed2 = GetLinkSharedPrivateKey(linkKey, vchReqPubKey);
                    }
                    else
                    {
                        strErrorMessage = strprintf("Recipient SharedAcceptPubKey (%s) does not match derived shared link public key (%s).",
                                            stringFromVch(link.SharedAcceptPubKey), stringFromVch(vchGetLinkSharedPubKey));
                        return false;
                    }
                }
                else {
                    strErrorMessage = strprintf("Failed to get recipient linkKey %s DHT private key.", stringFromVch(vchLinkPubKey));
                    return false;
                }
            }
            else
            {
                strErrorMessage = strprintf("Can not find %s link requestor record.", stringFromVch(link.RequestorFullObjectPath));
                return false;
            }
        }
    }
    else
    {
        strErrorMessage = strprintf("Can not find %s link recipient record.", stringFromVch(link.RecipientFullObjectPath));
        return false;
    }
    CKeyEd25519 sharedKey1(sharedSeed1);
    CKeyEd25519 sharedKey2(sharedSeed2);
    // third key exchange: shared link request pubkey + shared link accept pubkey
    // Only the link recipient and requestor can derive this secret key.
    // the third shared public key is not on the blockchain and should only be known by the participants.
    seed = GetLinkSharedPrivateKey(sharedKey1, sharedKey2.GetPubKey());
    return true;
}

bool GetMessageInfo(CLink& link, std::string& strErrorMessage)
{
    std::array<char, 32> seed;
    if (!GetSharedPrivateSeed(link, seed, strErrorMessage))
    {
        return false;
    }
    CKeyEd25519 key(seed);
    link.vchSecretPubKeyBytes = key.GetPubKeyBytes();
    link.SubjectID = Hash(link.vchSecretPubKeyBytes.begin(), link.vchSecretPubKeyBytes.end());
    return true;
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

//#endif // ENABLE_WALLET