// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DHT_DATACHUNK_H
#define DYNAMIC_DHT_DATACHUNK_H

#include "serialize.h"

#include <string>
#include <vector>

static constexpr unsigned int DHT_DATA_MAX_CHUNK_SIZE = 900;

class CDataChunk
{
public:
    uint16_t nOrdinal;
    uint16_t nPlacement;
    std::string Salt;
    std::vector<unsigned char> vchValue;

    CDataChunk() : nOrdinal(0), nPlacement(0), Salt("")
    { 
        vchValue.clear();
    }

    CDataChunk(const uint16_t ordinal, const uint16_t placement, const std::string& salt, const std::string& value) : 
                    nOrdinal(ordinal), nPlacement(placement), Salt(salt), vchValue(value.begin(), value.end()) {}

    CDataChunk(const uint16_t ordinal, const uint16_t placement, const std::string& salt, const std::vector<unsigned char>& vch) : 
                    nOrdinal(ordinal), nPlacement(placement), Salt(salt), vchValue(vch) {}

    inline void SetNull()
    {
        nOrdinal = 0;
        nPlacement = 0;
        Salt = "";
        vchValue.clear();
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(nOrdinal);
        READWRITE(nPlacement);
        READWRITE(Salt);
        READWRITE(vchValue);
    }

    inline friend bool operator==(const CDataChunk& a, const CDataChunk& b) {
        return (a.nOrdinal == b.nOrdinal && a.nPlacement == b.nPlacement && a.Salt == b.Salt && a.vchValue == b.vchValue);
    }

    inline friend bool operator!=(const CDataChunk& a, const CDataChunk& b) {
        return !(a == b);
    }

    inline CDataChunk operator=(const CDataChunk& b) {
        nOrdinal = b.nOrdinal;
        nPlacement = b.nPlacement;
        Salt = b.Salt;
        vchValue = b.vchValue;
        return *this;
    }
 
    inline bool IsNull() const { return (nOrdinal == 0 && nPlacement == 0 && Salt == "" && vchValue.size() == 0); }

    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData);

};

#endif // DYNAMIC_DHT_DATACHUNK_H
