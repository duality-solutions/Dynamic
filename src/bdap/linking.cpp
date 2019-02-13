// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: license

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
    if(!UnserializeFromData(vchData, vchHash))
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

bool CLinkRequest::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsLinkRequest(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsLinkRequest >> *this;

        std::vector<unsigned char> vchLinkRequestData;
        Serialize(vchLinkRequestData);
        const uint256 &calculatedHash = Hash(vchLinkRequestData.begin(), vchLinkRequestData.end());
        const std::vector<unsigned char> &vchRandLinkRequest = vchFromValue(calculatedHash.GetHex());
        if(vchRandLinkRequest != vchHash)
        {
            SetNull();
            return false;
        }
        txHash = Hash(vchLinkRequestData.begin(), vchLinkRequestData.end());
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
    std::vector<unsigned char> vchLinkPath = RequestorFullObjectPath;
    vchLinkPath.insert(vchLinkPath.end(), vchSeparator.begin(), vchSeparator.end());
    vchLinkPath.insert(vchLinkPath.end(), RecipientFullObjectPath.begin(), RecipientFullObjectPath.end());
    return vchLinkPath;
}

std::string CLinkRequest::LinkPathString() const
{
    return stringFromVch(LinkPath());
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
    if(!UnserializeFromData(vchData, vchHash))
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

bool CLinkAccept::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsAcceptLink(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsAcceptLink >> *this;

        std::vector<unsigned char> vchAcceptLinkData;
        Serialize(vchAcceptLinkData);
        const uint256 &calculatedHash = Hash(vchAcceptLinkData.begin(), vchAcceptLinkData.end());
        const std::vector<unsigned char> &vchRandAcceptLink = vchFromValue(calculatedHash.GetHex());
        if(vchRandAcceptLink != vchHash)
        {
            SetNull();
            return false;
        }
        txHash = Hash(vchAcceptLinkData.begin(), vchAcceptLinkData.end());
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
    std::vector<unsigned char> vchLinkPath = RequestorFullObjectPath;
    vchLinkPath.insert(vchLinkPath.end(), vchSeparator.begin(), vchSeparator.end());
    vchLinkPath.insert(vchLinkPath.end(), RecipientFullObjectPath.begin(), RecipientFullObjectPath.end());
    return vchLinkPath;
}

std::string CLinkAccept::LinkPathString() const
{
    return stringFromVch(LinkPath());
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

// TODO: (BDAP) implement BDAP encryption by converting ed25519 private key to curve25519
/*
CharString GetEncryptedLinkMessage(const CLinkRequest& requestLink)
{

}

CharString GetEncryptedLinkMessage(const CLinkRequest& requestLink, const CLinkAccept& acceptLink)
{


}
*/