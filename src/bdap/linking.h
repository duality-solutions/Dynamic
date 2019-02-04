// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: license

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
    bool UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash);
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
        return *this;
    }
 
    inline bool IsNull() const { return (RequestorFullObjectPath.empty()); }
    bool UnserializeFromTx(const CTransactionRef& tx);
    bool UnserializeFromData(const std::vector<unsigned char> &vchData, const std::vector<unsigned char> &vchHash);
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

};

bool LinkPubKeyExistsInMemPool(const CTxMemPool& pool, const std::vector<unsigned char>& vchPubKey, const std::string& strOpType, std::string& errorMessage);
bool CreateSignatureProof(const CKey& key, const std::string& strFQDN, std::vector<unsigned char>& vchSignatureProof);
bool SignatureProofIsValid(const CDynamicAddress& addr,  const std::string& strFQDN, const std::vector<unsigned char>& vchSig);
// TODO (BDAP): Implement
CharString GetEncryptedRequestMessage(const CLinkRequest& requestLink); // stored in an OP_RETURN transaction
CharString GetEncryptedAcceptMessage(const CLinkRequest& requestLink, const CLinkAccept& acceptLink); // stored on BitTorrent DHT network

#endif // DYNAMIC_BDAP_LINKING_H