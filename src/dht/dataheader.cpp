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

CRecordHeader::CRecordHeader(const uint16_t version, const uint32_t expireTime, const uint16_t chunks, const uint16_t chunkSize, const uint32_t format, const uint16_t indexLocation, const uint32_t size, const uint32_t timestamp) :
                               nVersion(version), nExpireTime(expireTime), nChunks(chunks), nChunkSize(chunkSize), nFormat(format), nIndexLocation(indexLocation), nDataSize(size), nTimeStamp(timestamp)
{
    nUnlockTime = GetTime() + 30; // unlocks in 30 seconds by default
    strHex = ToHex();
}

CRecordHeader::CRecordHeader(const std::string& hex)
{
    LoadHex(hex);
}

bool CRecordHeader::LoadHex(const std::string& hex)
{
    strHex = hex;
    std::vector<unsigned char> vchData = HexStringToCharVector(hex);
    return UnserializeFromData(vchData);
}

void CRecordHeader::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsDataHeader(SER_NETWORK, PROTOCOL_VERSION);
    dsDataHeader << *this;
    vchData = std::vector<unsigned char>(dsDataHeader.begin(), dsDataHeader.end());
}

bool CRecordHeader::UnserializeFromData(const std::vector<unsigned char>& vchData) 
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

void CRecordHeader::SetHex()
{
    std::vector<unsigned char> vchData;
    Serialize(vchData);
    strHex = CharVectorToHexString(vchData);
}

std::string CRecordHeader::ToHex()
{
    std::vector<unsigned char> vchData;
    Serialize(vchData);
    return CharVectorToHexString(vchData);
}

std::string CRecordHeader::ToString()
{
   return strprintf("CRecordHeader(version=%u, encrypted=%s, expire=%u, chunks=%u, chunk_size=%u, data_size=%u, format=%u, index_loc=%u, unlock_time=%u, time_stamp=%u)\n", 
                                    nVersion, (nVersion > 0 ? "true": "false"), nExpireTime, nChunks, nChunkSize, nDataSize, nFormat, nIndexLocation, nUnlockTime, nTimeStamp);
}

CDataSetHeader::CDataSetHeader(const uint16_t version, const uint32_t recordCount, const uint16_t indexCount, const uint32_t unlockTime, const uint32_t updateTime) : 
                                    nVersion(version), nRecordCount(recordCount), nIndexCount(indexCount), nUnlockTime(unlockTime), nLastUpdateTime(updateTime)
{
    nUnlockTime = GetTime() + 30; // unlocks in 30 seconds by default
    strHex = ToHex();
}

CDataSetHeader::CDataSetHeader(const std::string strHex)
{
    std::vector<unsigned char> vchData = HexStringToCharVector(strHex);
    UnserializeFromData(vchData);
}

void CDataSetHeader::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsDataHeader(SER_NETWORK, PROTOCOL_VERSION);
    dsDataHeader << *this;
    vchData = std::vector<unsigned char>(dsDataHeader.begin(), dsDataHeader.end());
}

bool CDataSetHeader::UnserializeFromData(const std::vector<unsigned char>& vchData) 
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

void CDataSetHeader::SetHex()
{
    std::vector<unsigned char> vchData;
    Serialize(vchData);
    strHex = CharVectorToHexString(vchData);
}

std::string CDataSetHeader::ToHex()
{
    std::vector<unsigned char> vchData;
    Serialize(vchData);
    return CharVectorToHexString(vchData);
}

std::string CDataSetHeader::ToString()
{
   return strprintf("CDataSetHeader(version=%u, records=%s, indexes=%u, unlock_time=%u, last_update=%u)\n", nVersion, nRecordCount, nIndexCount, nUnlockTime, nLastUpdateTime);
}