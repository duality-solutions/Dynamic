// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_LINKMANAGER_H
#define DYNAMIC_BDAP_LINKMANAGER_H

#include "bdap/linkstorage.h"
#include "uint256.h"

#include <array>
#include <map>
#include <queue>
#include <string>
#include <vector>

class CKeyEd25519;
class CLinkRequest;
class CLinkAccept;

namespace BDAP {

    enum LinkState : std::uint8_t
    {
        unknown_state = 0,
        pending_state = 1,
        complete_state = 2,
        deleted_state = 3
    };
}

class CLink {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    uint256 LinkID;
    bool fRequestFromMe;
    bool fAcceptFromMe;
    uint8_t nLinkState;
    std::vector<unsigned char> RequestorFullObjectPath; // Requestor's BDAP object path
    std::vector<unsigned char> RecipientFullObjectPath; // Recipient's BDAP object path
    std::vector<unsigned char> RequestorPubKey; // ed25519 public key new/unique for this link
    std::vector<unsigned char> RecipientPubKey; // ed25519 public key new/unique for this link
    std::vector<unsigned char> SharedRequestPubKey; // ed25519 shared public key. RequestorPubKey + Recipient's BDAP DHT PubKey
    std::vector<unsigned char> SharedAcceptPubKey; // ed25519 shared public key. RecipientPubKey + RequestorPubKey's BDAP DHT PubKey
    std::vector<unsigned char> LinkMessage; // Link message to recipient
    std::vector<unsigned char> RequestorWalletAddress; // Requestor's BDAP wallet address
    std::vector<unsigned char> RecipientWalletAddress; // Recipient's BDAP wallet address

    uint64_t nHeightRequest;
    uint64_t nExpireTimeRequest;
    uint256 txHashRequest;

    uint64_t nHeightAccept;
    uint64_t nExpireTimeAccept;
    uint256 txHashAccept;

    
    uint256 SubjectID; // Used to tell when a VGP message is for this link
    std::vector<unsigned char> vchSecretPubKeyBytes; // Used to derive the VGP message id for this link

    CLink() {
        SetNull();
    }

    inline void SetNull()
    {
        nVersion = CLink::CURRENT_VERSION;
        LinkID.SetNull();
        nLinkState = 0;
        fRequestFromMe = false;
        fAcceptFromMe = false;
        RequestorFullObjectPath.clear();
        RecipientFullObjectPath.clear();
        RequestorPubKey.clear();
        RecipientPubKey.clear();
        SharedRequestPubKey.clear();
        SharedAcceptPubKey.clear();
        LinkMessage.clear();
        nHeightRequest = 0;
        nExpireTimeRequest = 0;
        txHashRequest.SetNull();
        nHeightAccept = 0;
        nExpireTimeAccept = 0;
        txHashAccept.SetNull();
        SubjectID.SetNull();
        vchSecretPubKeyBytes.clear();
        RequestorWalletAddress.clear();
        RecipientWalletAddress.clear();
    }

    inline friend bool operator==(const CLink &a, const CLink &b) {
        return (a.txHashRequest == b.txHashRequest && a.txHashAccept == b.txHashAccept);
    }

    inline friend bool operator!=(const CLink &a, const CLink &b) {
        return !(a == b);
    }

    inline CLink operator=(const CLink &b) {
        nVersion = b.nVersion;
        LinkID = b.LinkID;
        nLinkState = b.nLinkState;
        fRequestFromMe = b.fRequestFromMe;
        fAcceptFromMe = b.fAcceptFromMe;
        RequestorFullObjectPath = b.RequestorFullObjectPath;
        RecipientFullObjectPath = b.RecipientFullObjectPath;
        RequestorPubKey = b.RequestorPubKey;
        RecipientPubKey = b.RecipientPubKey;
        SharedRequestPubKey = b.SharedRequestPubKey;
        SharedAcceptPubKey = b.SharedAcceptPubKey;
        LinkMessage = b.LinkMessage;
        nHeightRequest = b.nHeightRequest;
        nExpireTimeRequest = b.nExpireTimeRequest;
        txHashRequest = b.txHashRequest;
        nHeightAccept = b.nHeightAccept;
        nExpireTimeAccept = b.nExpireTimeAccept;
        txHashAccept = b.txHashAccept;
        SubjectID = b.SubjectID;
        vchSecretPubKeyBytes = b.vchSecretPubKeyBytes;
        RequestorWalletAddress = b.RequestorWalletAddress;
        RecipientWalletAddress = b.RecipientWalletAddress;
        return *this;
    }
 
    std::string RequestorFQDN() const;
    std::string RecipientFQDN() const;
    std::string RequestorPubKeyString() const;
    std::string RecipientPubKeyString() const;

    inline bool IsNull() const { return (nLinkState == 0 && RequestorFullObjectPath.empty() && RecipientFullObjectPath.empty()); }

    std::string LinkState() const;
    std::string ToString() const;
};

class CLinkManager {
private:
    std::queue<CLinkStorage> linkQueue;
    std::map<uint256, CLink> m_Links;
    std::map<uint256, std::vector<unsigned char>> m_LinkMessageInfo;

public:
    CLinkManager() {
        SetNull();
    }

    inline void SetNull()
    {
        std::queue<CLinkStorage> emptyQueue;
        linkQueue = emptyQueue;
        m_Links.clear();
    }

    std::size_t QueueSize() const { return linkQueue.size(); }
    std::size_t LinkCount() const { return m_Links.size(); }

    bool ProcessLink(const CLinkStorage& storage, const bool fStoreInQueueOnly = false);
    void ProcessQueue();

    bool FindLink(const uint256& id, CLink& link);
    bool FindLinkBySubjectID(const uint256& subjectID, CLink& getLink);
    bool ListMyPendingRequests(std::vector<CLink>& vchLinks);
    bool ListMyPendingAccepts(std::vector<CLink>& vchLinks);
    bool ListMyCompleted(std::vector<CLink>& vchLinks);
    std::vector<CLinkInfo> GetCompletedLinkInfo(const std::vector<unsigned char>& vchFullObjectPath);
    int IsMyMessage(const uint256& subjectID, const uint256& messageID, const int64_t& timestamp);
    void LoadLinkMessageInfo(const uint256& subjectID, const std::vector<unsigned char>& vchPubKey);
    bool GetLinkMessageInfo(const uint256& subjectID, std::vector<unsigned char>& vchPubKey);
    bool GetAllMessagesByType(const std::vector<unsigned char> vchMessageType);

private:
    bool IsLinkFromMe(const std::vector<unsigned char>& vchLinkPubKey);
    bool IsLinkForMe(const std::vector<unsigned char>& vchLinkPubKey, const std::vector<unsigned char>& vchSharedPubKey);
    bool GetLinkPrivateKey(const std::vector<unsigned char>& vchSenderPubKey, const std::vector<unsigned char>& vchSharedPubKey, std::array<char, 32>& sharedSeed, std::string& strErrorMessage);
};

uint256 GetLinkID(const CLinkRequest& request);
uint256 GetLinkID(const CLinkAccept& accept);
uint256 GetLinkID(const std::string& account1, const std::string& account2);

bool GetSharedPrivateSeed(const CLink& link, std::array<char, 32>& seed, std::string& strErrorMessage);
bool GetMessageInfo(CLink& link, std::string& strErrorMessage);
uint256 GetMessageID(const std::vector<unsigned char>& vchPubKey, const int64_t& timestamp);
uint256 GetMessageID(const CKeyEd25519& key, const int64_t& timestamp);

extern CLinkManager* pLinkManager;

#endif // DYNAMIC_BDAP_LINKMANAGER_H