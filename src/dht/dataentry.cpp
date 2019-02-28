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
    LogPrintf("%s -- value converted %s \n", __func__, stringFromVch(vchValue));
    //EncodeBase64
    if (!EncryptBDAPData(vvchPubKeys, vchValue, vchEncrypted, strErrorMessage))
        return false;

    if (vchEncrypted.size() > 1277) {
        LogPrintf("%s -- Ciphertext too large for one DHT entry. %u\n", __func__, vchEncrypted.size());
        return false;
    }
    LogPrintf("%s -- Ciphertext size %u\n", __func__, vchEncrypted.size());

    return true;
}
catch (std::bad_alloc const&)
{
    LogPrintf("%s -- catch std::bad_alloc\n", __func__);
    return false;
}
/*
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

*/

CDataEntry::CDataEntry(const std::string& opCode, const uint16_t slots, const std::vector<std::vector<unsigned char>>& pubkeys, const std::vector<unsigned char>& data, const uint16_t version, const DHT::DataFormat format)
    : strOperationCode(opCode), nTotalSlots(slots), vPubKeys(pubkeys), vchData(data)
{
    if (version > 0 && vPubKeys.size() == 0)
        throw std::runtime_error("Encrypted data entries require at least one public key.\n");
    if (version == 0 && vPubKeys.size() > 0)
        throw std::runtime_error("Clear text (Version 0) data entries do not need public keys.\n");

    dataHeader.nVersion = version;
    dataHeader.nFormat = (uint32_t)format;
    Init();
}

bool CDataEntry::Init()
{
    std::vector<unsigned char> vchRaw;
    if (dataHeader.nVersion == 1) {
        if (!Encrypt(vPubKeys, vchData, vchRaw, strErrorMessage)) {
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
            std::string strHexChunk = strHexData.substr(i * DHT_DATA_MAX_CHUNK_SIZE, DHT_DATA_MAX_CHUNK_SIZE);
            CDataChunk chunk(i, strHexChunk, i + 1);
            vChunks.push_back(chunk);
        }
        dataHeader.nChunks = 1;
        dataHeader.nChunkSize = strHexData.size();
        dataHeader.nIndexLocation = 0;
    }
    else {
        CDataChunk chunk(1, strHexData, 1);
        vChunks.push_back(chunk);
        dataHeader.nChunks = 1;
        dataHeader.nChunkSize = strHexData.size();
        dataHeader.nIndexLocation = 0;
    }
    return true;
}