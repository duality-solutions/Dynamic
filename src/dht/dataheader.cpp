// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dht/dataheader.h"

#include "bdap/utils.h"
#include "hash.h"
#include "streams.h"
#include "tinyformat.h"
#include "uint256.h"
#include "utiltime.h"

CDataHeader::CDataHeader(const uint16_t version, const uint32_t expireTime, const uint16_t chunks, const uint16_t chunkSize, const uint32_t format, const uint16_t indexLocation) :
                               nVersion(version), nExpireTime(expireTime), nChunks(chunks), nChunkSize(chunkSize), nFormat(format), nIndexLocation(indexLocation)
{
    nUnlockTime = GetTime() + 30; // unlocks in 30 seconds
    strHex = ToHex();
}

CDataHeader::CDataHeader(const std::string strHex)
{
    std::vector<unsigned char> vchData = HexStringToCharVector(strHex);
    UnserializeFromData(vchData);
}

void CDataHeader::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsDataHeader(SER_NETWORK, PROTOCOL_VERSION);
    dsDataHeader << *this;
    vchData = std::vector<unsigned char>(dsDataHeader.begin(), dsDataHeader.end());
}

bool CDataHeader::UnserializeFromData(const std::vector<unsigned char>& vchData) 
{
    try {
        CDataStream dsDataHeader(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsDataHeader >> *this;
    } catch (std::exception& e) {
        SetNull();
        return false;
    }
    return true;
}

void CDataHeader::SetHex()
{
    std::vector<unsigned char> vchData;
    Serialize(vchData);
    strHex = CharVectorToHexString(vchData);
}

std::string CDataHeader::ToHex()
{
    std::vector<unsigned char> vchData;
    Serialize(vchData);
    return CharVectorToHexString(vchData);
}

std::string CDataHeader::ToString()
{
   return strprintf("CDataHeader(version=%u, encrypted=%s, expire=%u, chunks=%u, chunk_size=%u, format=%u, index_loc=%u, unlock_time=%u)\n", 
                                    nVersion, (nVersion > 0 ? "true": "false"), nExpireTime, nChunks, nChunkSize, nFormat, nIndexLocation, nUnlockTime);
}