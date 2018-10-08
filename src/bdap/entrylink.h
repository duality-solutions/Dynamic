// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_ENTRYLINK_H
#define DYNAMIC_BDAP_ENTRYLINK_H

#include "bdap.h"
#include "primitives/transaction.h"
#include "serialize.h"
#include "uint256.h"

/*
Entry linking is a DAP binding operation.  This class is used to
manage domain entry link requests. When linking entries, we want 
to use stealth addresses so the linkage requests are not public.
*/


class CEntryLink {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    CharString SenderFullPath;  // points to the link requestor's full domain entry path
    CharString SenderPublicKey;  // an unique public key for this link request
    CharString RecipientFullPath; // points to the link recipient's full domain entry path
    CharString RecipientPublicKey; // an unique public key for this link request
    unsigned int nHeight;
    uint64_t nExpireTime;
    uint256 txHash;

    CEntryLink() {
        SetNull();
    }

    CEntryLink(const CTransactionRef& tx) {
        SetNull();
        UnserializeFromTx(tx);
    }

    inline void SetNull()
    {
        nVersion = CEntryLink::CURRENT_VERSION;
        SenderFullPath.clear();
        SenderPublicKey.clear();
        RecipientFullPath.clear();
        RecipientPublicKey.clear();
        nHeight = 0;
        nExpireTime = 0;
        txHash.SetNull();
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(SenderFullPath);
        READWRITE(SenderPublicKey);
        READWRITE(RecipientFullPath);
        READWRITE(RecipientPublicKey);
        READWRITE(VARINT(nHeight));
        READWRITE(VARINT(nExpireTime));
        READWRITE(txHash);
    }

    inline friend bool operator==(const CEntryLink &a, const CEntryLink &b) {
        return (a.SenderFullPath == b.SenderFullPath && a.RecipientFullPath == b.RecipientFullPath && a.nHeight == b.nHeight);
    }

    inline friend bool operator!=(const CEntryLink &a, const CEntryLink &b) {
        return !(a == b);
    }

    inline CEntryLink operator=(const CEntryLink &b) {
        SenderFullPath = b.SenderFullPath;
        SenderFullPath = b.SenderFullPath;
        SenderPublicKey = b.SenderPublicKey;
        RecipientFullPath = b.RecipientFullPath;
        RecipientPublicKey = b.RecipientPublicKey;
        nHeight = b.nHeight;
        nExpireTime = b.nExpireTime;
        txHash = b.txHash;
        return *this;
    }
 
    inline bool IsNull() const { return (SenderFullPath.empty()); }
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char> &vchData, const std::vector<unsigned char> &vchHash);
    bool UnserializeFromTx(const CTransactionRef& tx);

    bool IsMyRequest(const CTransactionRef& tx);
    CharString SenderFileName(unsigned int nTime);
    CharString RecipientFileName(unsigned int nTime);
};

#endif // DYNAMIC_BDAP_ENTRYLINK_H