// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dht/dataheader.h"

#include "bdap/utils.h"
#include "hash.h"
#include "streams.h"
#include "uint256.h"

CDataHeader::CDataHeader(const std::string strHex)
{
    //TODO: parse hex into serialized data. Call UnserializeFromData
}

void CDataHeader::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsDataHeader(SER_NETWORK, PROTOCOL_VERSION);
    dsDataHeader << *this;
    vchData = std::vector<unsigned char>(dsDataHeader.begin(), dsDataHeader.end());
}

bool CDataHeader::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsDataHeader(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsDataHeader >> *this;

        std::vector<unsigned char> vchDataHeader;
        Serialize(vchDataHeader);
        const uint256& calculatedHash = Hash(vchDataHeader.begin(), vchDataHeader.end());
        const std::vector<unsigned char>& vchRandDataHeader = vchFromString(calculatedHash.GetHex());
        if(vchRandDataHeader != vchHash)
        {
            SetNull();
            return false;
        }
    } catch (std::exception& e) {
        SetNull();
        return false;
    }
    return true;
}

std::string CDataHeader::ToHex()
{
    std::vector<unsigned char> vchData;
    Serialize(vchData);
    return CharVectorToHexString(vchData);
}