// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: license

#ifndef DYNAMIC_BDAP_LINKING_H
#define DYNAMIC_BDAP_LINKING_H

#include "bdap.h"
#include "primitives/transaction.h"
#include "serialize.h"
#include "uint256.h"

class CTransaction;


// Entry linking is a type of DAP binding operation.  This class is used to
// manage domain entry link requests. When linking entries, we want 
// to use stealth addresses so the linkage requests are not public.

// CRequestLink are stored serilzed and encrypted in a BDAP OP_RETURN transaction.
// The link request recipient can decrypt the BDAP OP_RETURN transaction 
// and get the needed information to accept the link request
// It is used to bootstrap the linkage relationship with a new set of public keys

// OP_RETURN Format:
// CAcceptLink.RecipientPublicKey.Encrypt(CRequestLink.SharedSymmetricPrivKey).ToHex() + space +
// CRequestLink.RequestorPublicKey.Encrypt(CRequestLink.SharedSymmetricPrivKey).ToHex() + spcae + 
// CRequestLink.SharedSymmetricPrivKey.Encrypt(Serialize(CRequestLink)).ToHex()
class CRequestLink {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    CharString RequestorFullObjectPath; // Requestor's BDAP object path
    CharString RecipientFullObjectPath; // Recipient's BDAP object path
    CharString RequestorPublicKey; // ed25519 public key new/unique for this link

    unsigned int nHeight;
    uint64_t nExpireTime;
    uint256 txHash;

    CRequestLink() {
        SetNull();
    }

    CRequestLink(const CTransactionRef& tx) {
        SetNull();
        UnserializeFromTx(tx);
    }

    inline void SetNull()
    {
        nVersion = CRequestLink::CURRENT_VERSION;
        RequestorFullObjectPath.clear();
        RecipientFullObjectPath.clear();
        RequestorPublicKey.clear();
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
        READWRITE(RequestorPublicKey);
        READWRITE(VARINT(nHeight));
        READWRITE(VARINT(nExpireTime));
        READWRITE(txHash);
    }

    inline friend bool operator==(const CRequestLink &a, const CRequestLink &b) {
        return (a.RequestorFullObjectPath == b.RequestorFullObjectPath && a.RecipientFullObjectPath == b.RecipientFullObjectPath 
                    && a.RequestorPublicKey == b.RequestorPublicKey);
    }

    inline friend bool operator!=(const CRequestLink &a, const CRequestLink &b) {
        return !(a == b);
    }

    inline CRequestLink operator=(const CRequestLink &b) {
        nVersion = b.nVersion;
        RequestorFullObjectPath = b.RequestorFullObjectPath;
        RecipientFullObjectPath = b.RecipientFullObjectPath;
        RequestorPublicKey = b.RequestorPublicKey;
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
    bool IsMyLinkRequest(const CTransactionRef& tx);
};

// CAcceptLink are stored serilzed and encrypted in a LibTorrent DHT key value pair entry
// Stored in the Torrent DHT for a limited time.
// This is only used when the link recipient wants to accepts the request.

// DHT Data Format (must be under 1000 bytes):
// CRequestLink.RequestorPublicKey.Encrypt(CRequestLink.SharedSymmetricPrivKey).ToHex() + spcae + 
// CAcceptLink.RecipientPublicKey.Encrypt(CRequestLink.SharedSymmetricPrivKey).ToHex() + space +
// CRequestLink.SharedSymmetricPrivKey.Encrypt(Serialize(CAcceptLink)).ToHex()
class CAcceptLink {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    CharString RequestorFullObjectPath; // Requestor's BDAP object path
    CharString RecipientFullObjectPath; // Recipient's BDAP object path
    uint256 txLinkRequestHash; // transaction hash for the link request.
    CharString RecipientPublicKey; // ed25519 public key new/unique for this link
    CharString SharedPublicKey; // ed25519 shared public key using the requestor and recipient keys

    unsigned int nHeight;
    uint64_t nExpireTime;
    uint256 txHash;

    CAcceptLink() {
        SetNull();
    }

    inline void SetNull()
    {
        nVersion = CAcceptLink::CURRENT_VERSION;
        RequestorFullObjectPath.clear();
        RecipientFullObjectPath.clear();
        txLinkRequestHash.SetNull();
        RecipientPublicKey.clear();
        SharedPublicKey.clear();
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
        READWRITE(RecipientPublicKey);
        READWRITE(SharedPublicKey);
        READWRITE(VARINT(nHeight));
        READWRITE(VARINT(nExpireTime));
        READWRITE(txHash);
    }

    inline friend bool operator==(const CAcceptLink &a, const CAcceptLink &b) {
        return (a.RequestorFullObjectPath == b.RequestorFullObjectPath && a.RecipientFullObjectPath == b.RecipientFullObjectPath
                    && a.txLinkRequestHash == b.txLinkRequestHash && a.RecipientPublicKey == b.RecipientPublicKey && a.SharedPublicKey == b.SharedPublicKey);
    }

    inline friend bool operator!=(const CAcceptLink &a, const CAcceptLink &b) {
        return !(a == b);
    }

    inline CAcceptLink operator=(const CAcceptLink &b) {
        nVersion = b.nVersion;
        RequestorFullObjectPath = b.RequestorFullObjectPath;
        RecipientFullObjectPath = b.RecipientFullObjectPath;
        RecipientPublicKey = b.RecipientPublicKey;
        SharedPublicKey = b.SharedPublicKey;
        txLinkRequestHash = b.txLinkRequestHash;
        return *this;
    }
 
    inline bool IsNull() const { return (RequestorFullObjectPath.empty()); }
    bool UnserializeFromData(const std::vector<unsigned char> &vchData, const std::vector<unsigned char> &vchHash);
    void Serialize(std::vector<unsigned char>& vchData);

    bool ValidateValues(std::string& errorMessage);
};

// TODO (BDAP): Implement
CharString GetEncryptedRequestMessage(const CRequestLink& requestLink); // stored in an OP_RETURN transaction
CharString GetEncryptedAcceptMessage(const CRequestLink& requestLink, const CAcceptLink& acceptLink); // stored on BitTorrent DHT network

#endif // DYNAMIC_BDAP_LINKING_H