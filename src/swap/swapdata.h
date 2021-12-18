// Copyright (c) 2021-present Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_SWAP_SWAPDATA_H
#define DYNAMIC_SWAP_SWAPDATA_H

#include "amount.h"
#include "primitives/transaction.h"
#include "serialize.h"
#include "uint256.h"

#include <string>
#include <vector>

class CSwapData {
public:
    static const int CURRENT_VERSION = 1;
    int nVersion;
    std::vector<unsigned char> vSwapData; // Swap address
    CAmount Amount; // Swap amount
    CAmount Fee; // Swap fee amount
    uint256 TxId;
    int nOut;
    int nHeight;

    CSwapData() {
        SetNull();
    }

    CSwapData(const std::vector<unsigned char>& vchData) {
        SetNull();
        UnserializeFromData(vchData);
    }

    CSwapData(const CTransactionRef& tx, const int& height);

    inline void SetNull()
    {
        nVersion = CSwapData::CURRENT_VERSION;
        vSwapData.clear();
        Amount = -1;
        Fee = 0;
        TxId = uint256();
        nOut = -1;
        nHeight = -1;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(vSwapData);
        READWRITE(Amount);
        READWRITE(Fee);
        READWRITE(TxId);
        READWRITE(VARINT(nOut));
        READWRITE(VARINT(nHeight));
    }

    inline friend bool operator==(const CSwapData& a, const CSwapData& b) {
        return (a.vSwapData == b.vSwapData && a.Amount == b.Amount);
    }

    inline friend bool operator!=(const CSwapData& a, const CSwapData& b) {
        return !(a == b);
    }

    inline CSwapData operator=(const CSwapData& b) {
        nVersion = b.nVersion;
        vSwapData = b.vSwapData;
        Amount = b.Amount;
        Fee = b.Fee;
        TxId = b.TxId;
        nOut = b.nOut;
        nHeight = b.nHeight;
        return *this;
    }

    inline friend bool operator<(const CSwapData& a, const CSwapData& b) {
        return (a.nHeight < b.nHeight);
    }

    inline friend bool operator>(const CSwapData& a, const CSwapData& b) {
        return (a.nHeight > b.nHeight);
    }

    inline bool IsNull() const { return (Amount == -1); }
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData);
    std::string Address() const;
    std::string ToString() const;
    std::vector<unsigned char> vchTxId() const;
    CAmount GetFee() const;

};

#endif // DYNAMIC_SWAP_SWAP_H
