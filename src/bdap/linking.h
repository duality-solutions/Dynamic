// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_LINKING_H
#define DYNAMIC_BDAP_LINKING_H

#include "bdap.h"
#include "primitives/transaction.h"
#include "serialize.h"
#include "uint256.h"

class CDynamicAddress;
class CKey;
class CTxMemPool;
class CTransaction;

// Entry linking is a type of DAP binding operation.  This class is used to
// manage domain entry link requests. When linking entries, we want 
// to use stealth addresses so the linkage requests are not public.

// CLinkRequest are stored serilzed and encrypted in a BDAP OP_RETURN transaction.
// The link request recipient can decrypt the BDAP OP_RETURN transaction 
// and get the needed information to accept the link request
// It is used to bootstrap the linkage relationship with a new set of public keys

// OP_RETURN Format: std::vector<unsigned char> GetEncryptedMessage(Serialize(CLinkRequest))

namespace BDAP {

    enum LinkFilterType
    {
        BOTH = 0,
        REQUEST = 1,
        RECIPIENT = 2
    };
}

class CLinkRequest {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    CharString RequestorFullObjectPath; // Requestor's BDAP object path
    CharString RecipientFullObjectPath; // Recipient's BDAP object path
    CharString RequestorPubKey; // ed25519 public key new/unique for this link
    CharString SharedPubKey; // ed25519 shared public key. RequestorPubKey + Recipient's BDAP DHT PubKey
    CharString LinkMessage; // Link message to recipient
    CharString SignatureProof; // Requestor's BDAP account ownership proof by signing the recipient's object path with their wallet pub key.

    unsigned int nHeight;
    uint64_t nExpireTime;
    uint256 txHash;

    CLinkRequest() {
        SetNull();
    }

    CLinkRequest(const CTransactionRef& tx) {
        SetNull();
        UnserializeFromTx(tx);
        txHash = tx->GetHash();
    }

    CLinkRequest(const std::vector<unsigned char>& vchData, const uint256& hash) {
        SetNull();
        UnserializeFromData(vchData);
        txHash = hash;
    }

    inline void SetNull()
    {
        nVersion = CLinkRequest::CURRENT_VERSION;
        RequestorFullObjectPath.clear();
        RecipientFullObjectPath.clear();
        RequestorPubKey.clear();
        SharedPubKey.clear();
        LinkMessage.clear();
        SignatureProof.clear();
        nHeight = 0;
        nExpireTime = 0;
        txHash.SetNull();
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(RequestorFullObjectPath);
        READWRITE(RecipientFullObjectPath);
        READWRITE(RequestorPubKey);
        READWRITE(SharedPubKey);
        READWRITE(LinkMessage);
        READWRITE(SignatureProof);
        READWRITE(VARINT(nHeight));
        READWRITE(VARINT(nExpireTime));
        READWRITE(txHash);
    }

    inline friend bool operator==(const CLinkRequest &a, const CLinkRequest &b) {
        return (a.RequestorPubKey == b.RequestorPubKey && a.SharedPubKey == b.SharedPubKey && a.LinkMessage == b.LinkMessage);
    }

    inline friend bool operator!=(const CLinkRequest &a, const CLinkRequest &b) {
        return !(a == b);
    }

    inline CLinkRequest operator=(const CLinkRequest &b) {
        nVersion = b.nVersion;
        RequestorFullObjectPath = b.RequestorFullObjectPath;
        RecipientFullObjectPath = b.RecipientFullObjectPath;
        RequestorPubKey = b.RequestorPubKey;
        SharedPubKey = b.SharedPubKey;
        LinkMessage = b.LinkMessage;
        SignatureProof = b.SignatureProof;
        nHeight = b.nHeight;
        nExpireTime = b.nExpireTime;
        txHash = b.txHash;
        return *this;
    }
 
    inline bool IsNull() const { return (RequestorFullObjectPath.empty()); }
    bool UnserializeFromTx(const CTransactionRef& tx);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData);
    void Serialize(std::vector<unsigned char>& vchData);

    bool ValidateValues(std::string& errorMessage);
    std::string RequestorPubKeyString() const;
    std::string SharedPubKeyString() const;
    std::string SignatureProofString() const;
    std::string RequestorFQDN() const;
    std::string RecipientFQDN() const;
    std::set<std::string> SortedAccounts() const;
    bool Matches(const std::string& strRequestorFQDN, const std::string& strRecipientFQDN) const;
    CharString LinkPath() const;
    std::string LinkPathString() const;
    std::string ToString() const;

};

// OP_RETURN Format: std::vector<unsigned char> GetEncryptedMessage(Serialize(CLinkAccept))
class CLinkAccept {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    CharString RequestorFullObjectPath; // Requestor's BDAP object path
    CharString RecipientFullObjectPath; // Recipient's BDAP object path
    uint256 txLinkRequestHash; // transaction hash for the link request.
    CharString RecipientPubKey; // ed25519 public key new/unique for this link
    CharString SharedPubKey; // ed25519 shared public key using the requestor and recipient keys
    CharString SignatureProof; // Acceptor's BDAP account ownership proof by signing the requestor's object path with their wallet pub key.

    unsigned int nHeight;
    uint64_t nExpireTime;
    uint256 txHash;

    CLinkAccept() {
        SetNull();
    }

    CLinkAccept(const CTransactionRef& tx) {
        SetNull();
        UnserializeFromTx(tx);
        txHash = tx->GetHash();
    }

    CLinkAccept(const std::vector<unsigned char>& vchData, const uint256& hash) {
        SetNull();
        UnserializeFromData(vchData);
        txHash = hash;
    }

    inline void SetNull()
    {
        nVersion = CLinkAccept::CURRENT_VERSION;
        RequestorFullObjectPath.clear();
        RecipientFullObjectPath.clear();
        txLinkRequestHash.SetNull();
        RecipientPubKey.clear();
        SharedPubKey.clear();
        SignatureProof.clear();
        nHeight = 0;
        nExpireTime = 0;
        txHash.SetNull();
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(RequestorFullObjectPath);
        READWRITE(RecipientFullObjectPath);
        READWRITE(txLinkRequestHash);
        READWRITE(RecipientPubKey);
        READWRITE(SharedPubKey);
        READWRITE(SignatureProof);
        READWRITE(VARINT(nHeight));
        READWRITE(VARINT(nExpireTime));
        READWRITE(txHash);
    }

    inline friend bool operator==(const CLinkAccept &a, const CLinkAccept &b) {
        return (a.SignatureProof == b.SignatureProof && a.RecipientPubKey == b.RecipientPubKey && a.SharedPubKey == b.SharedPubKey);
    }

    inline friend bool operator!=(const CLinkAccept &a, const CLinkAccept &b) {
        return !(a == b);
    }

    inline CLinkAccept operator=(const CLinkAccept &b) {
        nVersion = b.nVersion;
        RequestorFullObjectPath = b.RequestorFullObjectPath;
        RecipientFullObjectPath = b.RecipientFullObjectPath;
        RecipientPubKey = b.RecipientPubKey;
        SharedPubKey = b.SharedPubKey;
        SignatureProof = b.SignatureProof;
        txLinkRequestHash = b.txLinkRequestHash;
        nHeight = b.nHeight;
        nExpireTime = b.nExpireTime;
        txHash = b.txHash;
        return *this;
    }
 
    inline bool IsNull() const { return (RequestorFullObjectPath.empty()); }
    bool UnserializeFromTx(const CTransactionRef& tx);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData);
    void Serialize(std::vector<unsigned char>& vchData);

    bool ValidateValues(std::string& errorMessage);
    std::string RecipientPubKeyString() const;
    std::string SharedPubKeyString() const;
    std::string SignatureProofString() const;
    std::string RequestorFQDN() const;
    std::string RecipientFQDN() const;
    std::set<std::string> SortedAccounts() const;
    bool Matches(const std::string& strRequestorFQDN, const std::string& strRecipientFQDN) const;
    CharString LinkPath() const;
    std::string LinkPathString() const;
    std::string ToString() const;

};

class CLinkDenyList
{
public:
    std::vector<std::string> vDenyAccounts;
    std::vector<uint32_t> vTimestamps;

    CLinkDenyList() {}
    CLinkDenyList(const std::vector<unsigned char>& vchData);

    inline void SetNull()
    {
        vDenyAccounts.clear();
        vTimestamps.clear();
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(vDenyAccounts);
        READWRITE(vTimestamps);
    }

    inline friend bool operator==(const CLinkDenyList& a, const CLinkDenyList& b) {
        return (a.vDenyAccounts == b.vDenyAccounts && a.vTimestamps == b.vTimestamps);
    }

    inline friend bool operator!=(const CLinkDenyList& a, const CLinkDenyList& b) {
        return !(a == b);
    }

    inline CLinkDenyList operator=(const CLinkDenyList& b) {
        vDenyAccounts = b.vDenyAccounts;
        vTimestamps = b.vTimestamps;
        return *this;
    }
 
    inline bool IsNull() const { return (vDenyAccounts.size() == 0); }

    void Add(const std::string& addAccount, const uint32_t timestamp);
    bool Find(const std::string& searchAccount);
    bool Remove(const std::string& account);

    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData);

};

bool LinkPubKeyExistsInMemPool(const CTxMemPool& pool, const std::vector<unsigned char>& vchPubKey, const std::string& strOpType, std::string& errorMessage);
bool CreateSignatureProof(const CKey& key, const std::string& strFQDN, std::vector<unsigned char>& vchSignatureProof);
bool SignatureProofIsValid(const CDynamicAddress& addr,  const std::string& strFQDN, const std::vector<unsigned char>& vchSig);

#endif // DYNAMIC_BDAP_LINKING_H