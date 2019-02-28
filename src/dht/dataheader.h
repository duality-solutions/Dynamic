// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DHT_DATAHEADER_H
#define DYNAMIC_DHT_DATAHEADER_H

#include "serialize.h"

#include <string>
#include <vector>

namespace DHT {
    enum DataFormat : std::uint32_t {
      Undefined = 0,
      String = 1,
      BinaryBlob = 2,
      Rows = 3,
      PNG = 4,
      GIF = 5,
      JPEG = 6
    };
}

/****************************
Header format:
1) Version (uint16_t) 0 - 65535
2) Epoch Expire Time (uint32_t)
3) Number of fragments (uint16_t)
4) Size per fragment (uint16_t)
5) Format (uint16_t)
6) Index Location 

Example Header:
OpCode: avatar(24):0

- version 0 is unencrypted, version 1 and greater is VGP encrypted

**************************/

class CDataHeader
{
public:
    uint16_t nVersion;
    uint32_t nExpireTime;
    uint16_t nChunks;
    uint16_t nChunkSize;
    uint32_t nFormat;
    uint16_t nIndexLocation;

    CDataHeader() {
        SetNull();
    }

    CDataHeader(const uint16_t version, const uint32_t expireTime, const uint16_t chunks, const uint16_t chunkSize, const uint32_t format, const uint16_t indexLocation) :
                               nVersion(version), nExpireTime(expireTime), nChunks(chunks), nChunkSize(chunkSize), nFormat(format), nIndexLocation(indexLocation){}

    CDataHeader(const std::string strHex);

    inline void SetNull()
    {
        nVersion = 0;
        nExpireTime = 0;
        nChunks = 0;
        nChunkSize = 0;
        nFormat = 0;
        nIndexLocation = 0;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(nExpireTime);
        READWRITE(nChunks);
        READWRITE(nChunkSize);
        READWRITE(nFormat);
        READWRITE(nIndexLocation);
    }

    inline friend bool operator==(const CDataHeader& a, const CDataHeader& b) {
        return (a.nVersion == b.nVersion && a.nExpireTime == b.nExpireTime && a.nChunks == b.nChunks && 
                a.nChunkSize == b.nChunkSize && a.nFormat == b.nFormat && a.nIndexLocation == b.nIndexLocation);
    }

    inline friend bool operator!=(const CDataHeader& a, const CDataHeader& b) {
        return !(a == b);
    }

    inline CDataHeader operator=(const CDataHeader& b) {
        nVersion = b.nVersion;
        nExpireTime = b.nExpireTime;
        nChunks = b.nChunks;
        nChunkSize = b.nChunkSize;
        nFormat = b.nFormat;
        nIndexLocation = b.nIndexLocation;
        return *this;
    }
 
    inline bool IsNull() const { return (nVersion == 0 && nExpireTime == 0); }
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash);

    bool Encrypted() const { return (nVersion > 0); }

    std::string ToHex();
};

#endif // DYNAMIC_DHT_DATAHEADER_H
