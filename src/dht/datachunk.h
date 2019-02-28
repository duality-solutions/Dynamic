// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DHT_DATACHUNK_H
#define DYNAMIC_DHT_DATACHUNK_H

#include "serialize.h"

#include <string>
#include <vector>

static constexpr unsigned int DHT_DATA_MAX_CHUNK_SIZE = 450;

class CDataChunk
{
public:
    uint16_t nOrdinal;
    std::string strValue;
    uint16_t nPlacement;

    CDataChunk() {
        SetNull();
    }

    CDataChunk(const uint16_t ordinal, const std::string& value, const uint16_t placement) : nOrdinal(ordinal), strValue(value), nPlacement(placement) {}

    inline void SetNull()
    {
        nOrdinal = 0;
        strValue = "";
        nPlacement = 0;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(nOrdinal);
        READWRITE(strValue);
        READWRITE(nPlacement);
    }

    inline friend bool operator==(const CDataChunk& a, const CDataChunk& b) {
        return (a.nOrdinal == b.nOrdinal && a.strValue == b.strValue && a.nPlacement == b.nPlacement);
    }

    inline friend bool operator!=(const CDataChunk& a, const CDataChunk& b) {
        return !(a == b);
    }

    inline CDataChunk operator=(const CDataChunk& b) {
        nOrdinal = b.nOrdinal;
        strValue = b.strValue;
        nPlacement = b.nPlacement;
        return *this;
    }
 
    inline bool IsNull() const { return (nOrdinal == 0 && strValue == ""); }

    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash);

};

#endif // DYNAMIC_DHT_DATACHUNK_H
