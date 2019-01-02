// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: license

#include "bdap/linking.h"

#include "bdap/domainentry.h"
#include "bdap/utils.h"
#include "hash.h"
#include "script/script.h"
#include "streams.h"
#include "txmempool.h"

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
    } catch (std::exception &e) {
        SetNull();
        return false;
    }
    return true;
}

bool CLinkRequest::ValidateValues(std::string& errorMessage)
{
    // check object requestor FQDN
    if (RequestorFullObjectPath.size() > MAX_OBJECT_FULL_PATH_LENGTH) 
    {
        errorMessage = "Invalid BDAP link requestor FQDN. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }
    // check object recipient FQDN
    if (RecipientFullObjectPath.size() > MAX_OBJECT_FULL_PATH_LENGTH) 
    {
        errorMessage = "Invalid BDAP link recipient FQDN. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }
    // check requestor pubkey
    if (RequestorPubKey.size() != DHT_HEX_PUBLIC_KEY_LENGTH) 
    {
        errorMessage = "Invalid BDAP link requestor public key. DHT pubkey are " + std::to_string(DHT_HEX_PUBLIC_KEY_LENGTH) + " characters.";
        return false;
    }
    // check shared pubkey
    if (SharedPubKey.size() != DHT_HEX_PUBLIC_KEY_LENGTH) 
    {
        errorMessage = "Invalid BDAP link shared public key. DHT pubkey are " + std::to_string(DHT_HEX_PUBLIC_KEY_LENGTH) + " characters.";
        return false;
    }
    // check requestor pubkey
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
    // TODO (bdap): test SignatureProof
    return true;
}

bool CLinkRequest::IsMyLinkRequest(const CTransactionRef& tx)
{
    return false;
}

std::string CLinkRequest::RequestorPubKeyString() const
{
    return stringFromVch(RequestorPubKey);
}

std::string CLinkRequest::SharedPubKeyString() const
{
    return stringFromVch(SharedPubKey);
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
    } catch (std::exception &e) {
        SetNull();
        return false;
    }
    return true;
}

bool CLinkAccept::ValidateValues(std::string& errorMessage)
{
    // check object requestor FQDN
    if (RequestorFullObjectPath.size() > MAX_OBJECT_FULL_PATH_LENGTH) 
    {
        errorMessage = "Invalid BDAP link requestor FQDN. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }
    // check object recipient FQDN
    if (RecipientFullObjectPath.size() > MAX_OBJECT_FULL_PATH_LENGTH) 
    {
        errorMessage = "Invalid BDAP link recipient FQDN. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }
    // check recipient pubkey
    if (RecipientPubKey.size() != DHT_HEX_PUBLIC_KEY_LENGTH) 
    {
        errorMessage = "Invalid BDAP link recipient pubkey. DHT pubkey are " + std::to_string(DHT_HEX_PUBLIC_KEY_LENGTH) + " characters.";
        return false;
    }
    // check shared pubkey
    if (SharedPubKey.size() != DHT_HEX_PUBLIC_KEY_LENGTH) 
    {
        errorMessage = "Invalid BDAP link shared pubkey. DHT pubkey are " + std::to_string(DHT_HEX_PUBLIC_KEY_LENGTH) + " characters.";
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

/** Checks if BDAP link request pubkey exists in the memory pool */
bool LinkRequestExistsInMemPool(const CTxMemPool& pool, const std::vector<unsigned char>& vchPubKey, std::string& errorMessage)
{
    for (const CTxMemPoolEntry& e : pool.mapTx) {
        const CTransactionRef& tx = e.GetSharedTx();
        for (const CTxOut& txOut : tx->vout) {
            if (IsBDAPDataOutput(txOut)) {
                std::vector<unsigned char> vchMemPoolPubKey;
                std::string strOpType;
                if (!ExtractOpTypeValue(txOut.scriptPubKey, strOpType, vchMemPoolPubKey))
                    continue;
                if (vchPubKey == vchMemPoolPubKey) {
                    errorMessage = "CheckIfExistsInMemPool: A BDAP link request public key " + stringFromVch(vchPubKey) + " transaction is already in the memory pool!";
                    return true;
                }
            }
        }
    }
    return false;
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