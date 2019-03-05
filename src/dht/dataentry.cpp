// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dht/dataentry.h"

#include "bdap/utils.h"
#include "bdap/vgp/include/encryption.h" // for VGP E2E encryption
#include "util.h"

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

CDataEntry::CDataEntry(const std::string& opCode, const uint16_t slots, const std::vector<std::vector<unsigned char>>& pubkeys, 
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
    InitPut();
    dataHeader.SetHex();
    HeaderHex = dataHeader.HexValue();
}

bool CDataEntry::InitPut()
{
    std::vector<unsigned char> vchRaw;
    if (dataHeader.nVersion == 1) {
        if (!Encrypt(vPubKeys, vchData, vchRaw, strErrorMessage)) {
            LogPrintf("CDataEntry::%s -- Encrypt failed: %s\n", __func__, strErrorMessage);
            return false;
        }
    }
    else if (dataHeader.nVersion == 0) {
       vchRaw = vchData;
    }
    else {
        strErrorMessage = "Unknown and unsupported version.";
        return false;
    }

    std::string strHexData = CharVectorToHexString(vchRaw);
    if (strHexData.size() > DHT_DATA_MAX_CHUNK_SIZE * nTotalSlots) {
        strErrorMessage = "Data is too large for this operation code.";
        return false;
    }
    if (strHexData.size() > DHT_DATA_MAX_CHUNK_SIZE) {
        // We need to chunk into many pieces.
        uint16_t total_chunks = (strHexData.size() / DHT_DATA_MAX_CHUNK_SIZE) + 1;
        for(unsigned int i = 0; i < total_chunks; i++) {
            uint16_t nPlacement = i + 1;
            std::string strHexChunk = strHexData.substr(i * DHT_DATA_MAX_CHUNK_SIZE, DHT_DATA_MAX_CHUNK_SIZE);
            std::string strSalt = strOperationCode + ":" + std::to_string(nPlacement);
            CDataChunk chunk(i, nPlacement, strSalt, strHexChunk);
            vChunks.push_back(chunk);
        }
        dataHeader.nChunks = total_chunks;
        dataHeader.nChunkSize = DHT_DATA_MAX_CHUNK_SIZE;
        dataHeader.nIndexLocation = 0;
    }
    else {
        std::string strSalt = strOperationCode + ":" + std::to_string(1);
        CDataChunk chunk(1, 1, strSalt, strHexData);
        vChunks.push_back(chunk);
        dataHeader.nChunks = 1;
        dataHeader.nChunkSize = strHexData.size();
        dataHeader.nIndexLocation = 0;
    }
    return true;
}

CDataEntry::CDataEntry(const std::string& opCode, const uint16_t slots, const CDataHeader& header, const std::vector<CDataChunk>& chunks, const std::vector<unsigned char>& privateKey)
        : strOperationCode(opCode), nTotalSlots(slots),  nMode(DHT::DataMode::Get), dataHeader(header), vChunks(chunks)
{
    if (header.nVersion > 0 && privateKey.size() == 0)
        throw std::runtime_error("Decrypt entry requires a private key seed.\n");
    if (header.nVersion == 0 && privateKey.size() > 0)
        throw std::runtime_error("Private key seed not required for clear text data\n");

    InitGet(privateKey);
}

bool CDataEntry::InitGet(const std::vector<unsigned char>& privateKey)
{
    std::string strHexChunks;
    for(unsigned int i = 0; i < dataHeader.nChunks; i++) {
        strHexChunks += vChunks[i].Value;
    }
    std::vector<unsigned char> vchUnHex = HexStringToCharVector(strHexChunks);
    if (dataHeader.nVersion == 0) {
        vchData = vchUnHex;
    }
    else if (dataHeader.nVersion == 1) {
        if (!Decrypt(privateKey, vchUnHex, vchData, strErrorMessage))
            return false;
    }
    else {
        strErrorMessage = "Unsupported version.";
        return false;
    }
    return true;
}

std::string CDataEntry::Value() const
{
    return stringFromVch(vchData);
}