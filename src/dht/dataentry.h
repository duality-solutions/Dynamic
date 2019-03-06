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

class CDataEntry
{
private:
    const std::string strOperationCode;
    const uint16_t nTotalSlots;
    const DHT::DataMode nMode;

    std::vector<unsigned char> vchData;
    CDataHeader dataHeader;
    std::vector<CDataChunk> vChunks;
    std::string strErrorMessage;
    std::vector<std::vector<unsigned char>> vPubKeys;
    

public:
    CDataEntry(const std::string& opCode, const uint16_t slots, const std::vector<std::vector<unsigned char>>& pubkeys, const std::vector<unsigned char>& data,
                 const uint16_t version, const uint32_t expire, const DHT::DataFormat format);

    CDataEntry(const std::string& opCode, const uint16_t slots, const CDataHeader& header, const std::vector<CDataChunk>& chunks, const std::vector<unsigned char>& privateKey);

    std::string OperationCode() const { return strOperationCode; }
    uint16_t TotalSlots() const { return nTotalSlots; }
    std::vector<unsigned char> RawData() const { return vchData; }
    CDataHeader GetHeader() { return dataHeader; }
    std::vector<CDataChunk> GetChunks() const { return vChunks; }
    std::string Value() const;
    std::string ErrorMessage() { return strErrorMessage; }
    DHT::DataMode Mode() const { return nMode; }
    std::string HeaderHex;
    bool HasError() const { return strErrorMessage.size() > 0; }
    bool Valid() const { return (dataHeader.nDataSize == vchData.size()); }
private:
    bool InitPut();
    bool InitGet(const std::vector<unsigned char>& privateKey);
};

#endif // DYNAMIC_DHT_DATAENTRY_H
