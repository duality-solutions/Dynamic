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

bool CLinkManager::ProcessLink(const CLinkStorage& storage, const bool fStoreInQueueOnly)
{
    if (fStoreInQueueOnly) {
        std::vector<unsigned char> vchPubKeys = storage.vchLinkPubKey;
        vchPubKeys.insert(vchPubKeys.end(), storage.vchSharedPubKey.begin(), storage.vchSharedPubKey.end());
        uint256 linkID = Hash(vchPubKeys.begin(), vchPubKeys.end());
        LogPrintf("%s -- Store in queue mode. Stored link %s\n", __func__, linkID.ToString());
        linkQueue.push_front(std::make_pair(linkID, storage));
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
                    LogPrintf("%s ***** Warning. Link request from me found with an invalid signature proof! Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(storage.vchLinkPubKey));
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
                if (SignatureProofIsValid(entry.GetWalletAddress(), link.RecipientFQDN(), link.SignatureProof)) {
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
                    LogPrintf("%s -- Warning! Link accept from me found with an invalid signature proof! Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(storage.vchLinkPubKey));
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
                LogPrintf("%s --  Encrypted data size = %i\n", __func__, vchData.size());
                std::string strMessage = "";
                std::vector<unsigned char> dataDecrypted;
                if (DecryptBDAPData(privDHTKey.GetPrivSeedBytes(), vchData, dataDecrypted, strMessage)) {
                    std::vector<unsigned char> vchData, vchHash;
                    CScript scriptData;
                    scriptData << OP_RETURN << dataDecrypted;
                    if (GetBDAPData(scriptData, vchData, vchHash)) {
                        CLinkRequest link(dataDecrypted, storage.txHash);
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
                LogPrintf("%s --  Encrypted data size = %i\n", __func__, vchData.size());
                std::string strMessage = "";
                std::vector<unsigned char> dataDecrypted;
                if (DecryptBDAPData(sharedKey.GetPrivSeedBytes(), vchData, dataDecrypted, strMessage)) {
                    std::vector<unsigned char> vchData, vchHash;
                    CScript scriptData;
                    scriptData << OP_RETURN << dataDecrypted;
                    if (GetBDAPData(scriptData, vchData, vchHash)) {
                        CLinkRequest link(dataDecrypted, storage.txHash);
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
                LogPrintf("%s --  Encrypted data size = %i\n", __func__, vchData.size());
                std::string strMessage = "";
                std::vector<unsigned char> dataDecrypted;
                if (DecryptBDAPData(privDHTKey.GetPrivSeedBytes(), vchData, dataDecrypted, strMessage)) {
                    std::vector<unsigned char> vchData, vchHash;
                    CScript scriptData;
                    scriptData << OP_RETURN << dataDecrypted;
                    if (GetBDAPData(scriptData, vchData, vchHash)) {
                        CLinkAccept link(dataDecrypted, storage.txHash);
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
                LogPrintf("%s --  Encrypted data size = %i\n", __func__, vchData.size());
                std::string strMessage = "";
                std::vector<unsigned char> dataDecrypted;
                if (DecryptBDAPData(sharedKey.GetPrivSeedBytes(), vchData, dataDecrypted, strMessage)) {
                    std::vector<unsigned char> vchData, vchHash;
                    CScript scriptData;
                    scriptData << OP_RETURN << dataDecrypted;
                    if (GetBDAPData(scriptData, vchData, vchHash)) {
                        CLinkAccept link(dataDecrypted, storage.txHash);
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
            std::vector<unsigned char> vchPubKeys = storage.vchLinkPubKey;
            vchPubKeys.insert(vchPubKeys.end(), storage.vchSharedPubKey.begin(), storage.vchSharedPubKey.end());
            uint256 linkID = Hash(vchPubKeys.begin(), vchPubKeys.end());
            LogPrintf("%s -- Wallet locked. Stored link %s and process when the wallet is unlocked.\n", __func__, linkID.ToString());
            linkQueue.push_front(std::make_pair(linkID, storage));
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

//#endif // ENABLE_WALLET