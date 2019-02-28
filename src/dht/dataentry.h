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

class CDataEntry
{
private:
    const std::string strOperationCode;
    const uint16_t nTotalSlots;
    const std::vector<std::vector<unsigned char>> vPubKeys;
    const std::vector<unsigned char> vchData;
    CDataHeader dataHeader;
    std::vector<CDataChunk> vChunks;
    std::string strErrorMessage;

public:
    CDataEntry(const std::string& opCode, const uint16_t slots, const std::vector<std::vector<unsigned char>>& pubkeys, const std::vector<unsigned char>& data, const uint16_t version, const DHT::DataFormat format);

    std::string OperationCode() const { return strOperationCode; }
    uint16_t TotalSlots() const { return nTotalSlots; }
    std::vector<std::vector<unsigned char>> PubKeys() const { return vPubKeys; }
    std::vector<unsigned char> RawData() const { return vchData; }

    CDataHeader GetHeader() { return dataHeader; }

    std::vector<CDataChunk> GetChunks() { return vChunks; }

    std::string ErrorMessage() { return strErrorMessage; }

private:
    bool Init();

};

#endif // DYNAMIC_DHT_DATAENTRY_H
