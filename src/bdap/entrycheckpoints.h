// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_ENTRYCHECKPOINT_H
#define DYNAMIC_BDAP_ENTRYCHECKPOINT_H

#include "bdap.h"
#include "bdap/domainentry.h"
#include "serialize.h"
#include "uint256.h"

class CTransaction;

class CEntryCheckpoints {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    CharString OwnerFullPath;  // name of the owner's full domain entry path
    vCheckPoints CheckPointHashes; // vector of checkpoints containing the entry's channel or sub-chain block height and hash
    unsigned int nHeight;
    //uint64_t nExpireTime; // not sure if needed.  checkpoint will expire when owner channel expires
    uint256 txHash;
    CDomainEntry* OwnerDomainEntry;

    CEntryCheckpoints() {
        SetNull();
    }

    CEntryCheckpoints(const CTransactionRef& tx) {
        SetNull();
        UnserializeFromTx(tx);
    }

    inline void SetNull()
    {
        nVersion = CEntryCheckpoints::CURRENT_VERSION;
        OwnerFullPath.clear();
        CheckPointHashes.clear();
        nHeight = 0;
        txHash.SetNull();
        OwnerDomainEntry = nullptr;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(OwnerFullPath);
        READWRITE(CheckPointHashes);
        READWRITE(VARINT(nHeight));
        READWRITE(txHash);
    }

    inline friend bool operator==(const CEntryCheckpoints& a, const CEntryCheckpoints& b) {
        return (a.OwnerFullPath == b.OwnerFullPath && a.CheckPointHashes == b.CheckPointHashes && a.nHeight == b.nHeight);
    }

    inline friend bool operator!=(const CEntryCheckpoints& a, const CEntryCheckpoints& b) {
        return !(a == b);
    }

    inline CEntryCheckpoints operator=(const CEntryCheckpoints& b) {
        OwnerFullPath = b.OwnerFullPath;

        for (unsigned int i = 0; i < b.CheckPointHashes.size(); i++)
            CheckPointHashes.push_back(b.CheckPointHashes[i]);

        nHeight = b.nHeight;
        txHash = b.txHash;
        return *this;
    }
 
    inline bool IsNull() const { return (OwnerFullPath.empty()); }
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash);
    bool UnserializeFromTx(const CTransactionRef& tx);

    bool ValidateValues(std::string& errorMessage);
    void AddCheckpoint(const uint32_t& height, const CharString& vchHash);
};

#endif // DYNAMIC_BDAP_ENTRYCHECKPOINT_H