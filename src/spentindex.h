// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2009-2017 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_SPENTINDEX_H
#define DYNAMIC_SPENTINDEX_H

#include "amount.h"
#include "uint256.h"

struct CSpentIndexKey {
    uint256 txid;
    unsigned int outputIndex;

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(txid);
        READWRITE(outputIndex);
    }

    CSpentIndexKey(uint256 t, unsigned int i) {
        txid = t;
        outputIndex = i;
    }

    CSpentIndexKey() {
        SetNull();
    }

    void SetNull() {
        txid.SetNull();
        outputIndex = 0;
    }

};

struct CSpentIndexValue {
    uint256 txid;
    unsigned int inputIndex;
    int blockHeight;
    CAmount satoshis;
    int addressType;
    uint160 addressHash;

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(txid);
        READWRITE(inputIndex);
        READWRITE(blockHeight);
        READWRITE(satoshis);
        READWRITE(addressType);
        READWRITE(addressHash);
    }

    CSpentIndexValue(uint256 t, unsigned int i, int h, CAmount s, int type, uint160 a) {
        txid = t;
        inputIndex = i;
        blockHeight = h;
        satoshis = s;
        addressType = type;
        addressHash = a;
    }

    CSpentIndexValue() {
        SetNull();
    }

    void SetNull() {
        txid.SetNull();
        inputIndex = 0;
        blockHeight = 0;
        satoshis = 0;
        addressType = 0;
        addressHash.SetNull();
    }

    bool IsNull() const {
        return txid.IsNull();
    }
};

struct CSpentIndexKeyCompare
{
    bool operator()(const CSpentIndexKey& a, const CSpentIndexKey& b) const {
        if (a.txid == b.txid) {
            return a.outputIndex < b.outputIndex;
        } else {
            return a.txid < b.txid;
        }
    }
};

#endif // DYNAMIC_SPENTINDEX_H
