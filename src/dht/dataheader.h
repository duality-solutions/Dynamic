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
      JPEG = 6,
      Null = 7
    };
}

/****************************
Record header format:
1) Version (uint16_t) 0 - 65535
2) Epoch Expire Time (uint32_t)
3) Number of fragments/chunks (uint16_t)
4) Size per fragment/chunks (uint16_t)
5) Format. See DHT::DataFormat enum above (uint16_t)
6) Index Location (uint16_t)
7) Unlock Epoch time (uint32_t)
8) Total unencrypted data size (uint32_t)
9) Record update timestamp (uint32_t)

- version 0 is unencrypted, version 1 and greater is VGP encrypted
- chunk 0 is always the header
- when epire = 0, the record doesn't expire

Record header DHT Salt format:
opcode[record]:chunk

Record header DHT salt example:
avatar[1]:0

**************************/

class CRecordHeader
{
public:
    uint16_t nVersion;
    uint32_t nExpireTime;
    uint16_t nChunks;
    uint16_t nChunkSize;
    uint32_t nFormat;
    uint16_t nIndexLocation;
    uint32_t nUnlockTime;
    uint32_t nDataSize;
    uint32_t nTimeStamp;
    std::string Salt;

    CRecordHeader() {
        SetNull();
    }

    CRecordHeader(const uint16_t version, const uint32_t expireTime, const uint16_t chunks, const uint16_t chunkSize, 
                        const uint32_t format, const uint16_t indexLocation, const uint32_t size, const uint32_t timestamp);

    CRecordHeader(const std::string& hex);

    bool LoadHex(const std::string& hex);

    inline void SetNull()
    {
        nVersion = 0;
        nExpireTime = 0;
        nChunks = 0;
        nChunkSize = 0;
        nFormat = 0;
        nIndexLocation = 0;
        nUnlockTime = 0;
        nDataSize = 0;
        nTimeStamp = 0;
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
        READWRITE(nUnlockTime);
        READWRITE(nDataSize);
        READWRITE(nTimeStamp);
    }

    inline friend bool operator==(const CRecordHeader& a, const CRecordHeader& b) {
        return (a.nVersion == b.nVersion && a.nExpireTime == b.nExpireTime && a.nChunks == b.nChunks && a.nChunkSize == b.nChunkSize && 
                a.nFormat == b.nFormat && a.nIndexLocation == b.nIndexLocation && a.nUnlockTime == b.nUnlockTime && a.nDataSize == b.nDataSize && a.nTimeStamp ==b.nTimeStamp);
    }

    inline friend bool operator!=(const CRecordHeader& a, const CRecordHeader& b) {
        return !(a == b);
    }

    inline CRecordHeader operator=(const CRecordHeader& b) {
        nVersion = b.nVersion;
        nExpireTime = b.nExpireTime;
        nChunks = b.nChunks;
        nChunkSize = b.nChunkSize;
        nFormat = b.nFormat;
        nIndexLocation = b.nIndexLocation;
        nUnlockTime = b.nUnlockTime;
        nDataSize = b.nDataSize;
        nTimeStamp = b.nTimeStamp;
        return *this;
    }
 
    inline bool IsNull() const { return (nFormat == 7); }
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData);

    bool Encrypted() const { return (nVersion > 0); }

    void SetHex();

    std::string HexValue() const { return strHex; }
    std::string ToString();

private:
    std::string strHex;
    std::string ToHex();
};

/****************************
Data set header format:
1) Version (uint16_t) 0 - 65535
2) Number of records (uint32_t)
3) Number of indexes (uint16_t)
4) Unlock epoch Time (uint32_t)
5) Last epoch update time  (uint32_t)

- index 0 is the root index needed to get the other indexes

Data set header DHT Salt format:
opcode:chunk

Data set header DHT salt example:
messages:0

**************************/

class CDataSetHeader
{
public:
    uint16_t nVersion;
    uint32_t nRecordCount;
    uint16_t nIndexCount;
    uint32_t nUnlockTime;
    uint32_t nLastUpdateTime;
    std::string Salt;

    CDataSetHeader() {
        SetNull();
    }

    CDataSetHeader(const uint16_t version, const uint32_t recordCount, const uint16_t indexCount, const uint32_t unlockTime, const uint32_t updateTime);

    CDataSetHeader(const std::string strHex);

    inline void SetNull()
    {
        nVersion = 0;
        nRecordCount = 0;
        nIndexCount = 0;
        nUnlockTime = 0;
        nLastUpdateTime = 0;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(nRecordCount);
        READWRITE(nIndexCount);
        READWRITE(nUnlockTime);
        READWRITE(nLastUpdateTime);
    }

    inline friend bool operator==(const CDataSetHeader& a, const CDataSetHeader& b) {
        return (a.nVersion == b.nVersion && a.nRecordCount == b.nRecordCount && a.nIndexCount == b.nIndexCount && a.nUnlockTime == b.nUnlockTime && a.nLastUpdateTime == b.nLastUpdateTime);
    }

    inline friend bool operator!=(const CDataSetHeader& a, const CDataSetHeader& b) {
        return !(a == b);
    }

    inline CDataSetHeader operator=(const CDataSetHeader& b) {
        nVersion = b.nVersion;
        nRecordCount = b.nRecordCount;
        nIndexCount = b.nIndexCount;
        nUnlockTime = b.nUnlockTime;
        nLastUpdateTime = b.nLastUpdateTime;
        return *this;
    }
 
    inline bool IsNull() const { return (nVersion == 0 && nRecordCount == 0 && nIndexCount == 0 && nUnlockTime == 0 && nLastUpdateTime == 0); }
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData);

    void SetHex();

    std::string HexValue() const { return strHex; }
    std::string ToString();

private:
    std::string strHex;
    std::string ToHex();
};
#endif // DYNAMIC_DHT_DATAHEADER_H
