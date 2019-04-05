// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/linking.h"

#include "base58.h"
#include "bdap/domainentry.h"
#include "bdap/utils.h"
#include "hash.h"
#include "key.h"
#include "pubkey.h"
#include "script/script.h"
#include "streams.h"
#include "txmempool.h"
#include "utilstrencodings.h" // For EncodeBase64
#include "validation.h" // For strMessageMagic

#include <algorithm> // For std::find

bool CLinkRequest::UnserializeFromTx(const CTransactionRef& tx) 
{
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nOut;
    if(!GetBDAPData(tx, vchData, vchHash, nOut))
    {
        SetNull();
        return false;
    }
    if(!UnserializeFromData(vchData))
    {
        return false;
    }
    return true;
}

void CLinkRequest::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsLinkRequest(SER_NETWORK, PROTOCOL_VERSION);
    dsLinkRequest << *this;
    vchData = std::vector<unsigned char>(dsLinkRequest.begin(), dsLinkRequest.end());
}

bool CLinkRequest::UnserializeFromData(const std::vector<unsigned char>& vchData) 
{
    try {
        CDataStream dsLinkRequest(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsLinkRequest >> *this;
    } catch (std::exception &e) {
        SetNull();
        return false;
    }
    return true;
}

bool CLinkRequest::ValidateValues(std::string& errorMessage)
{
    // check object requestor FQDN size
    if (RequestorFullObjectPath.size() > MAX_OBJECT_FULL_PATH_LENGTH) 
    {
        errorMessage = "Invalid BDAP link requestor FQDN. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }
    // check object recipient FQDN size
    if (RecipientFullObjectPath.size() > MAX_OBJECT_FULL_PATH_LENGTH) 
    {
        errorMessage = "Invalid BDAP link recipient FQDN. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }
    // check requestor pubkey size
    if (RequestorPubKey.size() != DHT_HEX_PUBLIC_KEY_LENGTH) 
    {
        errorMessage = "Invalid BDAP link requestor public key. DHT pubkey are " + std::to_string(DHT_HEX_PUBLIC_KEY_LENGTH) + " characters.";
        return false;
    }
    // check shared pubkey size
    if (SharedPubKey.size() != DHT_HEX_PUBLIC_KEY_LENGTH) 
    {
        errorMessage = "Invalid BDAP link shared public key. DHT pubkey are " + std::to_string(DHT_HEX_PUBLIC_KEY_LENGTH) + " characters.";
        return false;
    }
    // check link message size
    if (LinkMessage.size() > MAX_BDAP_LINK_MESSAGE) 
    {
        errorMessage = "Invalid invite message length. The maximum invite message length is " + std::to_string(MAX_BDAP_LINK_MESSAGE) + " characters.";
        return false;
    }
    // check signature proof size
    if (SignatureProof.size() > MAX_BDAP_SIGNATURE_PROOF) 
    {
        errorMessage = "Invalid signature proof length. The maximum signature proof length is " + std::to_string(MAX_BDAP_SIGNATURE_PROOF) + " characters.";
        return false;
    }
    return true;
}

std::string CLinkRequest::RequestorPubKeyString() const
{
    return stringFromVch(RequestorPubKey);
}

std::string CLinkRequest::SharedPubKeyString() const
{
    return stringFromVch(SharedPubKey);
}

std::string CLinkRequest::SignatureProofString() const
{
    return EncodeBase64(&SignatureProof[0], SignatureProof.size());
}

std::string CLinkRequest::RequestorFQDN() const
{
    return stringFromVch(RequestorFullObjectPath);
}

std::string CLinkRequest::RecipientFQDN() const
{
    return stringFromVch(RecipientFullObjectPath);
}

std::set<std::string> CLinkRequest::SortedAccounts() const
{
    std::set<std::string> sortedAccounts;
    sortedAccounts.insert(RequestorFQDN());
    sortedAccounts.insert(RecipientFQDN());
    return sortedAccounts;
}

bool CLinkRequest::Matches(const std::string& strRequestorFQDN, const std::string& strRecipientFQDN) const
{
    std::set<std::string> sortedAccounts;
    sortedAccounts.insert(strRequestorFQDN);
    sortedAccounts.insert(strRecipientFQDN);
    if (sortedAccounts == SortedAccounts())
        return true;

    return false;
}

CharString CLinkRequest::LinkPath() const
{
    std::vector<unsigned char> vchSeparator = {':'};
    std::set<std::string> sorted = SortedAccounts();
    std::set<std::string>::iterator it = sorted.begin();
    std::vector<unsigned char> vchLink1 = vchFromString(*it);
    std::advance(it, 1);
    std::vector<unsigned char> vchLink2 = vchFromString(*it);
    vchLink1.insert(vchLink1.end(), vchSeparator.begin(), vchSeparator.end());
    vchLink1.insert(vchLink1.end(), vchLink2.begin(), vchLink2.end());
    return vchLink1;
}

std::string CLinkRequest::LinkPathString() const
{
    return stringFromVch(LinkPath());
}

std::string CLinkRequest::ToString() const
{
    return strprintf(
        "CLinkRequest(\n"
        "    nVersion                   = %d\n"
        "    RequestorFullObjectPath    = %s\n"
        "    RecipientFullObjectPath    = %s\n"
        "    RequestorPubKey            = %s\n"
        "    SharedRequestPubKey        = %s\n"
        "    LinkMessage                = %s\n"
        "    SignatureProof             = %s\n"
        "    nHeight                    = %d\n"
        "    nExpireTime                = %d\n"
        "    txHash                     = %s\n"
        ")\n",
        nVersion,
        stringFromVch(RequestorFullObjectPath),
        stringFromVch(RecipientFullObjectPath),
        stringFromVch(RequestorPubKey),
        stringFromVch(SharedPubKey),
        stringFromVch(LinkMessage),
        CharVectorToHexString(SignatureProof),
        nHeight,
        nExpireTime,
        txHash.ToString()
    );
}

bool CLinkAccept::UnserializeFromTx(const CTransactionRef& tx) 
{
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nOut;
    if(!GetBDAPData(tx, vchData, vchHash, nOut))
    {
        SetNull();
        return false;
    }
    if(!UnserializeFromData(vchData))
    {
        return false;
    }
    return true;
}

void CLinkAccept::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsAcceptLink(SER_NETWORK, PROTOCOL_VERSION);
    dsAcceptLink << *this;
    vchData = std::vector<unsigned char>(dsAcceptLink.begin(), dsAcceptLink.end());
}

bool CLinkAccept::UnserializeFromData(const std::vector<unsigned char>& vchData) 
{
    try {
        CDataStream dsAcceptLink(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsAcceptLink >> *this;
    } catch (std::exception &e) {
        SetNull();
        return false;
    }
    return true;
}

bool CLinkAccept::ValidateValues(std::string& errorMessage)
{
    // check object requestor FQDN size
    if (RequestorFullObjectPath.size() > MAX_OBJECT_FULL_PATH_LENGTH) 
    {
        errorMessage = "Invalid BDAP link requestor FQDN. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }
    // check object recipient FQDN size
    if (RecipientFullObjectPath.size() > MAX_OBJECT_FULL_PATH_LENGTH) 
    {
        errorMessage = "Invalid BDAP link recipient FQDN. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }
    // check recipient pubkey size
    if (RecipientPubKey.size() != DHT_HEX_PUBLIC_KEY_LENGTH) 
    {
        errorMessage = "Invalid BDAP link recipient pubkey. DHT pubkey are " + std::to_string(DHT_HEX_PUBLIC_KEY_LENGTH) + " characters.";
        return false;
    }
    // check shared pubkey size
    if (SharedPubKey.size() != DHT_HEX_PUBLIC_KEY_LENGTH) 
    {
        errorMessage = "Invalid BDAP link shared pubkey. DHT pubkey are " + std::to_string(DHT_HEX_PUBLIC_KEY_LENGTH) + " characters.";
        return false;
    }
    // check signature proof size
    if (SignatureProof.size() > MAX_BDAP_SIGNATURE_PROOF) 
    {
        errorMessage = "Invalid signature proof length. The maximum signature proof length is " + std::to_string(MAX_BDAP_SIGNATURE_PROOF) + " characters.";
        return false;
    }
    return true;
}

std::string CLinkAccept::RecipientPubKeyString() const
{
    return stringFromVch(RecipientPubKey);
}

std::string CLinkAccept::SharedPubKeyString() const
{
    return stringFromVch(SharedPubKey);
}

std::string CLinkAccept::SignatureProofString() const
{
    return EncodeBase64(&SignatureProof[0], SignatureProof.size());
}

std::string CLinkAccept::RequestorFQDN() const
{
    return stringFromVch(RequestorFullObjectPath);
}

std::string CLinkAccept::RecipientFQDN() const
{
    return stringFromVch(RecipientFullObjectPath);
}

std::set<std::string> CLinkAccept::SortedAccounts() const
{
    std::set<std::string> sortedAccounts;
    sortedAccounts.insert(RequestorFQDN());
    sortedAccounts.insert(RecipientFQDN());
    return sortedAccounts;
}

bool CLinkAccept::Matches(const std::string& strRequestorFQDN, const std::string& strRecipientFQDN) const
{
    std::set<std::string> sortedAccounts;
    sortedAccounts.insert(strRequestorFQDN);
    sortedAccounts.insert(strRecipientFQDN);
    if (sortedAccounts == SortedAccounts())
        return true;

    return false;
}

CharString CLinkAccept::LinkPath() const
{
    std::vector<unsigned char> vchSeparator = {':'};
    std::set<std::string> sorted = SortedAccounts();
    std::set<std::string>::iterator it = sorted.begin();
    std::vector<unsigned char> vchLink1 = vchFromString(*it);
    std::advance(it, 1);
    std::vector<unsigned char> vchLink2 = vchFromString(*it);
    vchLink1.insert(vchLink1.end(), vchSeparator.begin(), vchSeparator.end());
    vchLink1.insert(vchLink1.end(), vchLink2.begin(), vchLink2.end());
    return vchLink1;
}

std::string CLinkAccept::LinkPathString() const
{
    return stringFromVch(LinkPath());
}

std::string CLinkAccept::ToString() const
{
    return strprintf(
        "CLinkAccept(\n"
        "    nVersion                   = %d\n"
        "    RequestorFullObjectPath    = %s\n"
        "    RecipientFullObjectPath    = %s\n"
        "    txLinkRequestHash          = %s\n"
        "    RecipientPubKey            = %s\n"
        "    SharedPubKey               = %s\n"
        "    SignatureProof             = %s\n"
        "    nHeight                    = %d\n"
        "    nExpireTime                = %d\n"
        "    txHash                     = %s\n"
        ")\n",
        nVersion,
        stringFromVch(RequestorFullObjectPath),
        stringFromVch(RecipientFullObjectPath),
        txLinkRequestHash.ToString(),
        stringFromVch(RecipientPubKey),
        stringFromVch(SharedPubKey),
        CharVectorToHexString(SignatureProof),
        nHeight,
        nExpireTime,
        txHash.ToString()
    );
}

CLinkDenyList::CLinkDenyList(const std::vector<unsigned char>& vchData)
{
    if (!UnserializeFromData(vchData))
        throw std::runtime_error("Failed to unserialize from data\n");
}

void CLinkDenyList::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsLinkDenyList(SER_NETWORK, PROTOCOL_VERSION);
    dsLinkDenyList << *this;
    vchData = std::vector<unsigned char>(dsLinkDenyList.begin(), dsLinkDenyList.end());
}

bool CLinkDenyList::UnserializeFromData(const std::vector<unsigned char>& vchData) 
{
    try {
        CDataStream dsLinkDenyList(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsLinkDenyList >> *this;
    } catch (std::exception& e) {
        SetNull();
        return false;
    }
    return true;
}

void CLinkDenyList::Add(const std::string& addAccount, const uint32_t timestamp)
{
    vDenyAccounts.push_back(addAccount);
    vTimestamps.push_back(timestamp);
}

bool CLinkDenyList::Find(const std::string& searchAccount)
{
    return std::find(vDenyAccounts.begin(), vDenyAccounts.end(), searchAccount) != vDenyAccounts.end();
}

bool CLinkDenyList::Remove(const std::string& account)
{
    if (!Find(account))
        return false;
    // TODO (BDAP): implement remove account from deny list.
    return true;
}
/** Checks if BDAP link request pubkey exists in the memory pool */
bool LinkPubKeyExistsInMemPool(const CTxMemPool& pool, const std::vector<unsigned char>& vchPubKey, const std::string& strOpType, std::string& errorMessage)
{
    for (const CTxMemPoolEntry& e : pool.mapTx) {
        const CTransactionRef& tx = e.GetSharedTx();
        for (const CTxOut& txOut : tx->vout) {
            if (IsBDAPDataOutput(txOut)) {
                std::vector<unsigned char> vchMemPoolPubKey;
                std::string strGetOpType;
                if (!ExtractOpTypeValue(txOut.scriptPubKey, strGetOpType, vchMemPoolPubKey))
                    continue;
                if (vchPubKey == vchMemPoolPubKey && strOpType == strGetOpType) {
                    errorMessage = "CheckIfExistsInMemPool: A BDAP link request public key " + stringFromVch(vchPubKey) + " transaction is already in the memory pool!";
                    return true;
                }
            }
        }
    }
    return false;
}

// Signs the FQDN with the input private key
bool CreateSignatureProof(const CKey& key, const std::string& strFQDN, std::vector<unsigned char>& vchSignatureProof)
{
    CHashWriter ss(SER_GETHASH, 0);
    ss << strMessageMagic;
    ss << strFQDN;

    if (!key.SignCompact(ss.GetHash(), vchSignatureProof)) {
        CDynamicAddress addr(key.GetPubKey().GetID());
        LogPrint("bdap", "%s -- Failed to sign BDAP account %s with the %s wallet address\n", __func__, strFQDN, addr.ToString());
        return false;
    }
    return true;
}

// Verifies that the signature for the FQDN matches the input address 
bool SignatureProofIsValid(const CDynamicAddress& addr,  const std::string& strFQDN, const std::vector<unsigned char>& vchSig)
{
    CHashWriter ss(SER_GETHASH, 0);
    ss << strMessageMagic;
    ss << strFQDN;

    CPubKey pubkey;
    if (!pubkey.RecoverCompact(ss.GetHash(), vchSig)) {
        LogPrint("bdap", "%s -- Failed to get pubkey from signature data. Address = %s, vchSig size = %u\n", __func__, addr.ToString(), vchSig.size());
        return false;
    }

    if (!(CDynamicAddress(pubkey.GetID()) == addr)) {
        CDynamicAddress sigAddress(pubkey.GetID());
        LogPrint("bdap", "%s -- Signature address %s does not match input address = %s\n", __func__, sigAddress.ToString(), addr.ToString());
        return false;
    }
    return true;
}