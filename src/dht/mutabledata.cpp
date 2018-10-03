// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#include "dht/mutabledata.h"

#include "bdap/domainentry.h"
#include "hash.h"

#include <univalue.h>

#include <boost/thread.hpp>

CMutableDataDB *pMutableDataDB = NULL;

bool AddMutableData(const std::vector<unsigned char>& vchInfoHash,const  CMutableData& data)
{
    if (!pMutableDataDB) {
        return false;
    }
    if (!pMutableDataDB->AddMutableData(data)) {
        return false;
    }
    return true;
}

bool UpdateMutableData(const std::vector<unsigned char>& vchInfoHash,const  CMutableData& data)
{
    if (!pMutableDataDB) {
        return false;
    }
    if (!pMutableDataDB->UpdateMutableData(data)) {
        return false;
    }
    return true;
}

bool GetMutableData(const std::vector<unsigned char>& vchInfoHash, CMutableData& data)
{
    if (!pMutableDataDB || !pMutableDataDB->ReadMutableData(vchInfoHash, data)) {
        return false;
    }
    return !data.IsNull();
}

bool PutMutableData(const std::vector<unsigned char>& vchInfoHash, const CMutableData& data)
{
    if (!pMutableDataDB) {
        return false;
    }
    CMutableData readMutableData;
    if (pMutableDataDB->ReadMutableData(vchInfoHash, readMutableData)) {
        UpdateMutableData(vchInfoHash, data);
    }
    else {
        AddMutableData(vchInfoHash, data);
    }
    return !data.IsNull();
}

bool EraseMutableData(const std::vector<unsigned char>& vchInfoHash)
{
    if (!pMutableDataDB) {
        return false;
    }
    if (!pMutableDataDB->EraseMutableData(vchInfoHash)) {
        return false;
    }
    return true;
}

bool GetAllMutableData(std::vector<CMutableData>& vchMutableData)
{
    if (!pMutableDataDB) {
        return false;
    }
    if (!pMutableDataDB->ListMutableData(vchMutableData)) {
        return false;
    }
    return true;
}

void CMutableData::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsMutableData(SER_NETWORK, PROTOCOL_VERSION);
    dsMutableData << *this;
    vchData = std::vector<unsigned char>(dsMutableData.begin(), dsMutableData.end());
}

bool CMutableData::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsMutableData(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsMutableData >> *this;

        std::vector<unsigned char> vchMutableData;
        Serialize(vchMutableData);
        const uint256 &calculatedHash = Hash(vchMutableData.begin(), vchMutableData.end());
        const std::vector<unsigned char> &vchRandMutableData = vchFromValue(calculatedHash.GetHex());
        if(vchRandMutableData != vchHash)
        {
            SetNull();
            return false;
        }
    } catch (std::exception &e) {
        SetNull();
        return false;
    }
    return true;
}

std::string CMutableData::InfoHash() const
{
    return stringFromVch(vchInfoHash);
}

std::string CMutableData::PublicKey() const
{
    return stringFromVch(vchPublicKey);
}

std::string CMutableData::Signature() const
{
    return stringFromVch(vchSignature);
}

std::string CMutableData::Salt() const
{
    return stringFromVch(vchSalt);
}

std::string CMutableData::Value() const
{
    return stringFromVch(vchValue);
}

bool CMutableDataDB::AddMutableData(const CMutableData& data)
{
    bool writeState = false;
    {
        LOCK(cs_dht_entry);
        writeState = CDBWrapper::Write(make_pair(std::string("ih"), data.vchInfoHash), data);  // use info hash as key
    }
    return writeState;
}

bool CMutableDataDB::ReadMutableData(const std::vector<unsigned char>& vchInfoHash, CMutableData& data)
{
    LOCK(cs_dht_entry);
    return CDBWrapper::Read(make_pair(std::string("ih"), vchInfoHash), data);
}

bool CMutableDataDB::EraseMutableData(const std::vector<unsigned char>& vchInfoHash)
{
    LOCK(cs_dht_entry);
    return CDBWrapper::Erase(make_pair(std::string("ih"), vchInfoHash));
}

bool CMutableDataDB::UpdateMutableData(const CMutableData& data)
{
    LOCK(cs_dht_entry);

    if (!EraseMutableData(data.vchInfoHash))
        return false;

    bool writeState = false;
    writeState = CDBWrapper::Update(make_pair(std::string("ih"), data.vchInfoHash), data);
    return writeState;
}

bool CMutableDataDB::ListMutableData(std::vector<CMutableData>& vchMutableData)
{
    std::pair<std::string, CharString> infoHash;
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        CMutableData data;
        try {
            if (pcursor->GetKey(infoHash) && infoHash.first == "ih") {
                pcursor->GetValue(data);
                vchMutableData.push_back(data);
            }
            pcursor->Next();
        }
        catch (std::exception& e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}