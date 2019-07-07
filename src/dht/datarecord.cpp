// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dht/datarecord.h"

#include "bdap/utils.h"
#include "bdap/vgp/include/encryption.h" // for VGP E2E encryption
#include "util.h"
#include "utilstrencodings.h"

#include <string>
#include <vector>

static bool Encrypt(const std::vector<std::vector<unsigned char>>& vvchPubKeys, const std::vector<unsigned char>& vchValue, std::vector<unsigned char>& vchEncrypted, std::string& strErrorMessage) try
{
    if (!EncryptBDAPData(vvchPubKeys, vchValue, vchEncrypted, strErrorMessage))
        return false;

    return true;
}
catch (std::bad_alloc const&)
{
    LogPrintf("%s -- catch std::bad_alloc\n", __func__);
    return false;
}

static bool Decrypt(const std::vector<unsigned char>& vchPrivSeedBytes, const std::vector<unsigned char>& vchData, std::vector<unsigned char>& vchDecrypted, std::string& strErrorMessage) try
{
    if (!DecryptBDAPData(vchPrivSeedBytes, vchData, vchDecrypted, strErrorMessage))
        return false;

    return true;
}
catch (std::bad_alloc const&)
{
    LogPrintf("%s -- catch std::bad_alloc\n", __func__);
    return false;
}

CDataRecord::CDataRecord(const std::string& opCode, const uint16_t slots, const std::vector<std::vector<unsigned char>>& pubkeys, 
                        const std::vector<unsigned char>& data, const uint16_t version, const uint32_t expire, const DHT::DataFormat format)
                        : strOperationCode(opCode), nTotalSlots(slots), nMode(DHT::DataMode::Put), vchData(data), vPubKeys(pubkeys)
{
    if (version > 0 && vPubKeys.size() == 0)
        throw std::runtime_error("Encrypted data entries require at least one public key.\n");
    if (version == 0 && vPubKeys.size() > 0)
        throw std::runtime_error("Clear text (Version 0) data entries do not need public keys.\n");

    dataHeader.nVersion = version;
    dataHeader.nExpireTime = expire;
    dataHeader.nFormat = (uint32_t)format;
    dataHeader.Salt = strOperationCode + ":" + std::to_string(0);
    dataHeader.nUnlockTime = GetTime() + 30; // default to unlocks in 30 seconds
    dataHeader.nTimeStamp = GetTime(); // unlocks in 30 seconds
    if (format != DHT::DataFormat::Null) {
        InitPut();
    }
    else {
        InitClear();
    }
    dataHeader.SetHex();
    HeaderHex = dataHeader.HexValue();
    fValid = true;
}

bool CDataRecord::InitPut()
{
    std::vector<unsigned char> vchRaw;
    if (dataHeader.nVersion == 1)
    {
        if (!Encrypt(vPubKeys, vchData, vchRaw, strErrorMessage))
        {
            LogPrintf("CDataRecord::%s -- Encrypt failed: %s\n", __func__, strErrorMessage);
            return false;
        }
    }
    else if (dataHeader.nVersion == 0)
    {
       vchRaw = vchData;
    }
    else
    {
        strErrorMessage = "Unknown and unsupported version.";
        return false;
    }
    if (vchRaw.size() > DHT_DATA_MAX_CHUNK_SIZE * nTotalSlots) {
        strErrorMessage = "Data is too large for this operation code.";
        return false;
    }

    if (vchRaw.size() > DHT_DATA_MAX_CHUNK_SIZE) {
        // We need to chunk into many pieces.
        uint32_t size = 0;
        uint16_t total_chunks = (vchRaw.size() / DHT_DATA_MAX_CHUNK_SIZE) + 1;
        for (unsigned int i = 0; i < total_chunks; i++) {
            uint16_t nPlacement = i + 1;
            std::vector<unsigned char>::const_iterator first = vchRaw.begin() + (i * DHT_DATA_MAX_CHUNK_SIZE);
            std::vector<unsigned char>::const_iterator last;
            if (i + 1 == total_chunks)
            {
                last = vchRaw.end();
            }
            else
            {
                last = vchRaw.begin() + (i * DHT_DATA_MAX_CHUNK_SIZE) + DHT_DATA_MAX_CHUNK_SIZE;
            }
            std::vector<unsigned char> vchChunk(first, last);
            std::string strSalt = strOperationCode + ":" + std::to_string(nPlacement);
            CDataChunk chunk(i, nPlacement, strSalt, vchChunk);
            vChunks.push_back(chunk);
            size = size + vchChunk.size();
        }
        dataHeader.nChunks = total_chunks;
        dataHeader.nChunkSize = DHT_DATA_MAX_CHUNK_SIZE;
        dataHeader.nIndexLocation = 0;
        dataHeader.nDataSize = size;
    }
    else
    {
        std::string strSalt = strOperationCode + ":" + std::to_string(1);
        CDataChunk chunk(1, 1, strSalt, vchRaw);
        vChunks.push_back(chunk);
        dataHeader.nChunks = 1;
        dataHeader.nChunkSize = vchRaw.size();
        dataHeader.nIndexLocation = 0;
        dataHeader.nDataSize = vchRaw.size();
    }
    return true;
}

bool CDataRecord::InitClear()
{
    std::string strNullValue = ZeroString();
    for(unsigned int i = 0; i < nTotalSlots; i++) {
        uint16_t nPlacement = i + 1;
        std::string strSalt = strOperationCode + ":" + std::to_string(nPlacement);
        CDataChunk chunk(i, nPlacement, strSalt, strNullValue);
        vChunks.push_back(chunk);
    }
    dataHeader.nChunks = nTotalSlots;
    dataHeader.nChunkSize = 0;
    dataHeader.nIndexLocation = 0;

    return true;
}

CDataRecord::CDataRecord(const std::string& opCode, const uint16_t slots, const CRecordHeader& header, const std::vector<CDataChunk>& chunks, const std::vector<unsigned char>& privateKey)
        : strOperationCode(opCode), nTotalSlots(slots),  nMode(DHT::DataMode::Get), dataHeader(header), vChunks(chunks)
{
    if (header.nVersion > 0 && privateKey.size() == 0)
        throw std::runtime_error("Decrypt entry requires a private key seed.\n");

    if (header.nChunks != chunks.size())
        throw std::runtime_error("Number of chunks in header mismatches the data.\n");

    if (!dataHeader.IsNull()) {
        if (InitGet(privateKey)) {
            fValid = true;
        }
    }
}

bool CDataRecord::InitGet(const std::vector<unsigned char>& privateKey)
{
    std::vector<unsigned char> vchChunks;
    for(unsigned int i = 0; i < dataHeader.nChunks; i++) {
        vchChunks.insert(vchChunks.end(), vChunks[i].vchValue.begin(), vChunks[i].vchValue.end());
    }
    std::vector<unsigned char> vchUnHexValue;
    std::string strChunk = stringFromVch(vchChunks);
    if (IsHex(strChunk)) {
        vchUnHexValue = ParseHex(strChunk);
    } else {
        vchUnHexValue = vchChunks;
    }
    if (vchUnHexValue.size() != dataHeader.nDataSize)
    {
        LogPrintf("CDataRecord::%s --Warning, data size in header (%d) mismatches the total size (%d) from all chunks (%d).\n", __func__, dataHeader.nDataSize, vchUnHexValue.size(), dataHeader.nChunks);
    }
    if (dataHeader.nVersion == 0) {
        vchData = vchChunks;
    }
    else if (dataHeader.nVersion == 1) {

        if (!Decrypt(privateKey, vchUnHexValue, vchData, strErrorMessage)) {
            return false;
        }
    }
    else {
        strErrorMessage = "Unsupported version.";
        return false;
    }
    return true;
}

std::string CDataRecord::Value() const
{
    return stringFromVch(vchData);
}

CDataRecordBuffer::CDataRecordBuffer(const size_t size) : capacity(size)
{
    buffer.resize(size);
    record = 0;
}

void CDataRecordBuffer::push_back(const CDataRecord& input)
{
    buffer[position()] = input;
    record++;
}