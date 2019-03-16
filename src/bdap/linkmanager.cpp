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

/*
// Check to see if this link request is from one of my BDAP accounts.
std::vector<unsigned char> vchLinkPubKey = vvchOpParameters[0];
{
    bool fIsLinkRequestFromMe = IsLinkFromMe(vchLinkPubKey);
    if (fIsLinkRequestFromMe) {
        int nOut;
        std::vector<unsigned char> vchData, vchHash;
        if (GetBDAPData(ptx, vchData, vchHash, nOut)) {
            int nVersion = GetLinkVersionFromData(vchData);
            if (nVersion == 0) {
                if (nVersion == 0) { // version 0 is public and unencrypted
                    CLinkRequest link(MakeTransactionRef(tx));
                    link.nHeight = pIndex->nHeight;
                    CDomainEntry entry;
                    if (GetDomainEntry(link.RequestorFullObjectPath, entry)) {
                        if (SignatureProofIsValid(entry.GetWalletAddress(), link.RecipientFQDN(), link.SignatureProof)) {
                            LogPrint("bdap", "%s -- Link request from me found with a valid signature proof. Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(vchLinkPubKey));
                            pLinkRequestDB->AddMyLinkRequest(link);
                        }
                        else
                            LogPrintf("%s ***** Warning. Link request from me found with an invalid signature proof! Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(vchLinkPubKey));
                    }
                }
            }
            else if (nVersion == 1) {
                bool fDecrypted = false;
                LogPrint("bdap", "%s -- Version 1 link request from me found! vchLinkPubKey = %s\n", __func__, stringFromVch(vchLinkPubKey));
                CKeyEd25519 privDHTKey;
                CKeyID keyID(Hash160(vchLinkPubKey.begin(), vchLinkPubKey.end()));
                if (GetDHTKey(keyID, privDHTKey)) {
                    int nDataVersion;
                    LogPrint("bdap", "%s --  Encrypted data size = %i\n", __func__, vchData.size());
                    vchData = RemoveVersionFromLinkData(vchData, nDataVersion);
                    if (nDataVersion == nVersion) {
                        std::string strMessage = "";
                        std::vector<unsigned char> dataDecrypted;
                        if (DecryptBDAPData(privDHTKey.GetPrivSeedBytes(), vchData, dataDecrypted, strMessage)) {
                            std::vector<unsigned char> vchData, vchHash;
                            CScript scriptData;
                            scriptData << OP_RETURN << dataDecrypted;
                            if (GetBDAPData(scriptData, vchData, vchHash)) {
                                CLinkRequest link;
                                link.UnserializeFromData(dataDecrypted, vchHash);
                                if (pIndex) {
                                    link.nHeight = pIndex->nHeight;
                                }
                                else {
                                    link.nHeight = chainActive.Height();
                                }
                                pLinkRequestDB->AddMyLinkRequest(link);
                                LogPrint("bdap", "%s -- DecryptBDAPData RequestorFQDN = %s, RecipientFQDN = %s, dataDecrypted size = %i\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), dataDecrypted.size());
                                fDecrypted = true;
                            }
                        }
                    }
                }
                if (!fDecrypted)
                    LogPrint("bdap", "%s -- Link request DecryptBDAPData failed.\n", __func__);
            }
        }
    }
}
// Check to see if this link request is for one of my BDAP accounts.
{
    std::vector<unsigned char> vchSharedPubKey = vvchOpParameters[1];
    bool fIsLinkRequestForMe = IsLinkForMe(vchLinkPubKey, vchSharedPubKey);
    if (fIsLinkRequestForMe) {
        int nOut;
        std::vector<unsigned char> vchData, vchHash;
        if (GetBDAPData(ptx, vchData, vchHash, nOut)) {
            int nVersion = GetLinkVersionFromData(vchData);
            if (nVersion == 0) {
                if (nVersion == 0) { // version 0 is public and unencrypted
                    CLinkRequest link(MakeTransactionRef(tx));
                    link.nHeight = pIndex->nHeight;
                    CDomainEntry entry;
                    if (GetDomainEntry(link.RequestorFullObjectPath, entry)) {
                        if (SignatureProofIsValid(entry.GetWalletAddress(), link.RecipientFQDN(), link.SignatureProof)) {
                            LogPrint("bdap", "%s -- Link request for me found with a valid signature proof. Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(vchLinkPubKey));
                            pLinkRequestDB->AddMyLinkRequest(link);
                        }
                        else
                            LogPrintf("%s -- ***** Alert. Link request for me found with an invalid signature proof! Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(vchLinkPubKey));
                    }
                }
            }
            else if (nVersion == 1) {
                bool fDecrypted = false;
                LogPrint("bdap", "%s -- Version 1 link request for me found! vchLinkPubKey = %s\n", __func__, stringFromVch(vchLinkPubKey));
                CKeyEd25519 sharedDHTKey;
                std::array<char, 32> sharedSeed;
                std::string strErrorMessage;
                if (GetLinkPrivateKey(vchLinkPubKey, vchSharedPubKey, sharedSeed, strErrorMessage)) {
                    CKeyEd25519 sharedKey(sharedSeed);
                    int nDataVersion;
                    LogPrint("bdap", "%s --  Encrypted data size = %i\n", __func__, vchData.size());
                    vchData = RemoveVersionFromLinkData(vchData, nDataVersion);
                    if (nDataVersion == nVersion) {
                        std::string strMessage = "";
                        std::vector<unsigned char> dataDecrypted;
                        if (DecryptBDAPData(sharedKey.GetPrivSeedBytes(), vchData, dataDecrypted, strMessage)) {
                            std::vector<unsigned char> vchData, vchHash;
                            CScript scriptData;
                            scriptData << OP_RETURN << dataDecrypted;
                            if (GetBDAPData(scriptData, vchData, vchHash)) {
                                CLinkRequest link;
                                link.UnserializeFromData(dataDecrypted, vchHash);
                                if (pIndex) {
                                    link.nHeight = pIndex->nHeight;
                                }
                                else {
                                    link.nHeight = chainActive.Height();
                                }
                                pLinkRequestDB->AddMyLinkRequest(link);
                                LogPrint("bdap", "%s -- DecryptBDAPData RequestorFQDN = %s, RecipientFQDN = %s, dataDecrypted size = %i\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), dataDecrypted.size());
                                fDecrypted = true;
                            }
                        }
                    }
                }
                if (!fDecrypted)
                    LogPrint("bdap", "%s -- Link request DecryptBDAPData failed.\n", __func__);
            }
        }
    }
}

// Check to see if this link accept is from one of my BDAP accounts.
std::vector<unsigned char> vchLinkPubKey = vvchOpParameters[0];
{
    bool fIsLinkAcceptFromMe = IsLinkFromMe(vchLinkPubKey);
    if (fIsLinkAcceptFromMe) {
        int nOut;
        std::vector<unsigned char> vchData, vchHash;
        if (GetBDAPData(ptx, vchData, vchHash, nOut)) {
            int nVersion = GetLinkVersionFromData(vchData);
            if (nVersion == 0) {
                if (nVersion == 0) { // version 0 is public and unencrypted
                    CLinkAccept link(MakeTransactionRef(tx));
                    link.nHeight = pIndex->nHeight;
                    CDomainEntry entry;
                    if (GetDomainEntry(link.RecipientFullObjectPath, entry)) {
                        if (SignatureProofIsValid(entry.GetWalletAddress(), link.RequestorFQDN(), link.SignatureProof)) {
                            LogPrint("bdap", "%s -- Link accept from me found with a valid signature proof. Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(vchLinkPubKey));
                            pLinkAcceptDB->AddMyLinkAccept(link);
                        }
                        else
                            LogPrintf("%s -- Warning! Link accept from me found with an invalid signature proof! Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(vchLinkPubKey));
                    }
                }
            }
            else if (nVersion == 1) {
                bool fDecrypted = false;
                LogPrint("bdap", "%s -- Version 1 encrypted link accept for me found! vchLinkPubKey = %s\n", __func__, stringFromVch(vchLinkPubKey));
                CKeyEd25519 privDHTKey;
                CKeyID keyID(Hash160(vchLinkPubKey.begin(), vchLinkPubKey.end()));
                if (GetDHTKey(keyID, privDHTKey)) {
                    int nDataVersion;
                    LogPrint("bdap", "%s --  Encrypted data size = %i\n", __func__, vchData.size());
                    vchData = RemoveVersionFromLinkData(vchData, nDataVersion);
                    if (nDataVersion == nVersion) {
                        std::string strMessage = "";
                        std::vector<unsigned char> dataDecrypted;
                        if (DecryptBDAPData(privDHTKey.GetPrivSeedBytes(), vchData, dataDecrypted, strMessage)) {
                            std::vector<unsigned char> vchData, vchHash;
                            CScript scriptData;
                            scriptData << OP_RETURN << dataDecrypted;
                            if (GetBDAPData(scriptData, vchData, vchHash)) {
                                CLinkAccept link;
                                link.UnserializeFromData(dataDecrypted, vchHash);
                                if (pIndex) {
                                    link.nHeight = pIndex->nHeight;
                                }
                                else {
                                    link.nHeight = chainActive.Height();
                                }
                                pLinkAcceptDB->AddMyLinkAccept(link);
                                LogPrint("bdap", "%s -- DecryptBDAPData RequestorFQDN = %s, RecipientFQDN = %s, dataDecrypted size = %i\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), dataDecrypted.size());
                                fDecrypted = true;
                            }
                        }
                    }
                }
                if (!fDecrypted)
                    LogPrint("bdap", "%s -- DecryptBDAPData failed.\n", __func__);
            }
        }
    }
}
// Check to see if this link accept is for one of my BDAP accounts.
{
    std::vector<unsigned char> vchSharedPubKey = vvchOpParameters[1];
    bool fIsLinkAcceptForMe = IsLinkForMe(vchLinkPubKey, vchSharedPubKey);
    if (fIsLinkAcceptForMe) {
        int nOut;
        std::vector<unsigned char> vchData, vchHash;
        if (GetBDAPData(ptx, vchData, vchHash, nOut)) {
            int nVersion = GetLinkVersionFromData(vchData);
            if (nVersion == 0) {
                if (nVersion == 0) { // version 0 is public and unencrypted
                    CLinkAccept link(MakeTransactionRef(tx));
                    link.nHeight = pIndex->nHeight;
                    CDomainEntry entry;
                    if (GetDomainEntry(link.RecipientFullObjectPath, entry)) {
                        if (SignatureProofIsValid(entry.GetWalletAddress(), link.RequestorFQDN(), link.SignatureProof)) {
                            LogPrint("bdap", "%s -- Link accept for me found with a valid signature proof. Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(vchLinkPubKey));
                            pLinkAcceptDB->AddMyLinkAccept(link);
                        }
                        else
                            LogPrintf("%s -- ***** Alert. Link accept for me found with an invalid signature proof! Link requestor = %s, recipient = %s, pubkey = %s\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), stringFromVch(vchLinkPubKey));
                    }
                }
            }
            else if (nVersion == 1) {
                bool fDecrypted = false;
                LogPrint("bdap", "%s -- Version 1 link request for me found! vchLinkPubKey = %s\n", __func__, stringFromVch(vchLinkPubKey));
                CKeyEd25519 sharedDHTKey;
                std::array<char, 32> sharedSeed;
                std::string strErrorMessage;
                if (GetLinkPrivateKey(vchLinkPubKey, vchSharedPubKey, sharedSeed, strErrorMessage)) {
                    CKeyEd25519 sharedKey(sharedSeed);
                    int nDataVersion;
                    LogPrint("bdap", "%s --  Encrypted data size = %i\n", __func__, vchData.size());
                    vchData = RemoveVersionFromLinkData(vchData, nDataVersion);
                    if (nDataVersion == nVersion) {
                        std::string strMessage = "";
                        std::vector<unsigned char> dataDecrypted;
                        if (DecryptBDAPData(sharedKey.GetPrivSeedBytes(), vchData, dataDecrypted, strMessage)) {
                            std::vector<unsigned char> vchData, vchHash;
                            CScript scriptData;
                            scriptData << OP_RETURN << dataDecrypted;
                            if (GetBDAPData(scriptData, vchData, vchHash)) {
                                CLinkAccept link;
                                link.UnserializeFromData(dataDecrypted, vchHash);
                                if (pIndex) {
                                    link.nHeight = pIndex->nHeight;
                                }
                                else {
                                    link.nHeight = chainActive.Height();
                                }
                                pLinkAcceptDB->AddMyLinkAccept(link);
                                LogPrint("bdap", "%s -- DecryptBDAPData RequestorFQDN = %s, RecipientFQDN = %s, dataDecrypted size = %i\n", __func__, link.RequestorFQDN(), link.RecipientFQDN(), dataDecrypted.size());
                                fDecrypted = true;
                            }
                        }
                    }
                }
                if (!fDecrypted)
                    LogPrint("bdap", "%s -- Link request DecryptBDAPData failed.\n", __func__);
            }
        }
    }
}
*/
//#endif // ENABLE_WALLET