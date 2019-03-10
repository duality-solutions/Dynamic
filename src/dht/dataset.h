// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DHT_DATASET_H
#define DYNAMIC_DHT_DATASET_H

#include "dht/dataheader.h"

#include <string>
#include <vector>
/****************************
Data set format:
1) Version (uint16_t) 0 - 65535
2) Operation Code (Salt in the Kademlia DHT)
3) Maximum rumber of records allowed (uint32_t)
4) Maximum rumber of indexes allowed (uint16_t)

- record 0 is the index header needed to get all other indexes

Dataset DHT Salt format:
opcode[record_num]:chunk

Dataset record DHT salt example:
messages[1]:1 // gets the first record's first chuck

**************************/

class CDataSet
{
private:
    uint16_t nVersion;
    std::string strOperationCode;
    uint32_t nMaxRecords;
    uint16_t nMaxChunks;
    uint16_t nMaxIndexes;
    CDataSetHeader Header;
    std::string strErrorMessage;
    
public:
    CDataSet() {}

    CDataSet(const uint16_t version, const std::string& opCode, const uint32_t maxRecords, const uint16_t maxIndexes) : nVersion(version), strOperationCode(opCode), nMaxRecords(maxRecords), nMaxIndexes(maxIndexes) {}

    CDataSet(const std::string& opCode, const uint32_t max, const CDataSetHeader& header);

    std::string OperationCode() const { return strOperationCode; }
    uint32_t MaximumRecords() const { return nMaxRecords; }
    uint16_t MaximumChunks() const { return nMaxChunks; }
    uint16_t MaximumIndexes() const { return nMaxIndexes; }
    // Header info
    CDataSetHeader GetHeader() { return Header; }
    uint16_t Version() { return Header.nVersion; }
    uint32_t RecordCount() { return Header.nRecordCount; }
    uint16_t IndexCount() { return Header.nIndexCount; }
    uint32_t UnlockTime() { return Header.nUnlockTime; }
    uint32_t LastUpdateTime() { return Header.nLastUpdateTime; }

    std::string ErrorMessage() { return strErrorMessage; }

    std::string HeaderHex;
    bool HasError() const { return strErrorMessage.size() > 0; }

    // NewRecord
    // UpdateRecord
    // bool GetRecord()
    // DeleteRecord

private:

};

#endif // DYNAMIC_DHT_DATASET_H
