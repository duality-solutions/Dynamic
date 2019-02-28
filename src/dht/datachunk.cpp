// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dht/datachunk.h"

#include "bdap/utils.h"
#include "hash.h"
#include "streams.h"
#include "uint256.h"


void CDataChunk::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsDataChunk(SER_NETWORK, PROTOCOL_VERSION);
    dsDataChunk << *this;
    vchData = std::vector<unsigned char>(dsDataChunk.begin(), dsDataChunk.end());
}

bool CDataChunk::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsDataChunk(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsDataChunk >> *this;

        std::vector<unsigned char> vchDataChunk;
        Serialize(vchDataChunk);
        const uint256& calculatedHash = Hash(vchDataChunk.begin(), vchDataChunk.end());
        const std::vector<unsigned char>& vchRandDataChunk = vchFromString(calculatedHash.GetHex());
        if(vchRandDataChunk != vchHash)
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