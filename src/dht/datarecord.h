// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DHT_DATAENTRY_H
#define DYNAMIC_DHT_DATAENTRY_H

#include "serialize.h"

#include <string>
#include <vector>

#include "dht/datachunk.h"
#include "dht/dataheader.h"

namespace DHT {
    enum DataMode : std::uint8_t {
      Put = 1,
      Get = 2
    };
}

class CDataRecord
{
private:
    std::string strOperationCode;
    uint16_t nTotalSlots;
    DHT::DataMode nMode;

    std::vector<unsigned char> vchData;
    CRecordHeader dataHeader;
    std::vector<CDataChunk> vChunks;
    std::string strErrorMessage;
    std::vector<std::vector<unsigned char>> vPubKeys;
    bool fValid = false;

public:
    CDataRecord() {}

    CDataRecord(const std::string& opCode, const uint16_t slots, const std::vector<std::vector<unsigned char>>& pubkeys, const std::vector<unsigned char>& data,
                 const uint16_t version, const uint32_t expire, const DHT::DataFormat format);

    CDataRecord(const std::string& opCode, const uint16_t slots, const CRecordHeader& header, const std::vector<CDataChunk>& chunks, const std::vector<unsigned char>& privateKey);

    std::vector<unsigned char> vchOwnerFQDN;

    std::string OperationCode() const { return strOperationCode; }
    uint16_t TotalSlots() const { return nTotalSlots; }
    std::vector<unsigned char> RawData() const { return vchData; }
    CRecordHeader GetHeader() const { return dataHeader; }
    bool Encrypted() { return dataHeader.Encrypted(); }
    uint16_t Version() { return dataHeader.nVersion; }

    std::vector<CDataChunk> GetChunks() const { return vChunks; }
    std::string Value() const;
    std::string ErrorMessage() { return strErrorMessage; }
    DHT::DataMode Mode() const { return nMode; }
    std::string HeaderHex;
    bool HasError() const { return strErrorMessage.size() > 0; }
    bool Valid() const { return (fValid); }
private:
    bool InitPut();
    bool InitClear();
    bool InitGet(const std::vector<unsigned char>& privateKey);
};

class CDataRecordBuffer
{
public:
    CDataRecordBuffer(size_t size);
    void push_back(const CDataRecord& input);
    size_t size() const { return buffer.size(); }
    size_t position() const { return (record % capacity); }

private:
    std::vector<CDataRecord> buffer;
    size_t capacity;
    size_t record;
};

#endif // DYNAMIC_DHT_DATAENTRY_H
