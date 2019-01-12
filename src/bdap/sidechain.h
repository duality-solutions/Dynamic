// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_SIDECHAIN_H
#define DYNAMIC_BDAP_SIDECHAIN_H

#include "amount.h"
#include "bdap.h"
#include "bdap/domainentry.h"
#include "serialize.h"
#include "uint256.h"

class CTransaction;

enum ResourcePointerType {
    OTHER = 0,
    IPFS = 1,
    LIBTORRENT = 2,
    CLOUDSTORAGE = 3
};

class CSideChain {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    CharString OwnerFullPath;  // name of the owner's full domain entry path
    CharString Description;
    CharString ResourcePointer; // used to point to a domain shared resource like a stream (video, audio, file sharing), P2P storage (LibTorrent or IPFS network), or private cloud storage
    ResourcePointerType ResourceType;
    CAmount InitialTransactionFee;
    CAmount InitialRegistrationFeePerDay;
    CAmount InitialSupply;
    CAmount InitialBlockReward;
    unsigned int nTargetSpacing;
    unsigned int nHeight;
    uint64_t nExpireTime;

    uint256 txHash;

    CDomainEntry* OwnerDomainEntry;

    CSideChain() {
        SetNull();
    }

    CSideChain(const CTransactionRef& tx) {
        SetNull();
        UnserializeFromTx(tx);
    }

    inline void SetNull()
    {
        nVersion = CSideChain::CURRENT_VERSION;
        OwnerFullPath.clear();
        Description.clear();
        ResourcePointer.clear();
        ResourceType = ResourcePointerType::OTHER;
        InitialTransactionFee = 0;
        InitialRegistrationFeePerDay = 0;
        InitialSupply = 0;
        nTargetSpacing = 0;
        nHeight = 0;
        nExpireTime = 0;
        txHash.SetNull();
        OwnerDomainEntry = nullptr;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(OwnerFullPath);
        READWRITE(Description);
        READWRITE(ResourcePointer);
        //READWRITE(ResourceType); //TODO: (bdap) serialize this enum.
        READWRITE(InitialTransactionFee);
        READWRITE(InitialRegistrationFeePerDay);
        READWRITE(InitialSupply);
        READWRITE(VARINT(nTargetSpacing));
        READWRITE(VARINT(nHeight));
        READWRITE(VARINT(nExpireTime));
        READWRITE(txHash);
    }

    inline friend bool operator==(const CSideChain& a, const CSideChain& b) {
        return (a.OwnerFullPath == b.OwnerFullPath && a.ResourcePointer == b.ResourcePointer && a.Description == b.Description && a.nHeight == b.nHeight);
    }

    inline friend bool operator!=(const CSideChain& a, const CSideChain& b) {
        return !(a == b);
    }

    inline CSideChain operator=(const CSideChain& b) {
        OwnerFullPath = b.OwnerFullPath;
        Description = b.Description;
        ResourcePointer = b.ResourcePointer;
        ResourceType = b.ResourceType;
        InitialTransactionFee = b.InitialTransactionFee;
        InitialRegistrationFeePerDay = b.InitialRegistrationFeePerDay;
        InitialSupply = b.InitialSupply;
        nTargetSpacing = b.nTargetSpacing;
        nHeight = b.nHeight;
        nExpireTime = b.nExpireTime;
        txHash = b.txHash;
        return *this;
    }
 
    inline bool IsNull() const { return (OwnerFullPath.empty()); }
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash);
    bool UnserializeFromTx(const CTransactionRef& tx);

    bool ValidateValues(std::string& errorMessage);
};

#endif // DYNAMIC_BDAP_SIDECHAIN_H