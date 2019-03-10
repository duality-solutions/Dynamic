// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DHT_DATASET_H
#define DYNAMIC_DHT_DATASET_H

#include "dht/dataheader.h"

#include <string>
#include <vector>

class CDataSet
{
private:
    std::string strOperationCode;
    uint32_t nMaxRecords;
    uint16_t nMaxIndexes;
    CDataSetHeader Header;
    std::string strErrorMessage;
    
public:
    CDataSet() {}

    CDataSet(const std::string& opCode, const uint32_t maxRecords, const uint16_t maxIndexes) : strOperationCode(opCode), nMaxRecords(maxRecords), nMaxIndexes(maxIndexes) {}

    CDataSet(const std::string& opCode, const uint32_t max, const CDataSetHeader& header);

    std::string OperationCode() const { return strOperationCode; }
    uint32_t MaximumRecords() const { return nMaxRecords; }
    uint16_t MaximumIndexes() const { return nMaxIndexes; }
    CDataSetHeader GetHeader() { return Header; }
    std::string ErrorMessage() { return strErrorMessage; }
    std::string HeaderHex;
    bool HasError() const { return strErrorMessage.size() > 0; }

private:

};

#endif // DYNAMIC_DHT_DATASET_H
