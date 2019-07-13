// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dht/mutabledb.h"

#include "dht/mutable.h"
#include "util.h"

#include <univalue.h>

#include <boost/thread.hpp>

#include "map"

static std::map<std::vector<unsigned char>, CMutableData> mapDataStorage;

CMutableDataDB *pMutableDataDB = NULL;

bool AddLocalMutableData(const std::vector<unsigned char>& vchInfoHash,const  CMutableData& data)
{
    if (!pMutableDataDB) {
        return false;
    }
    if (!pMutableDataDB->AddMutableData(data)) {
        return false;
    }
    return true;
}

bool UpdateLocalMutableData(const std::vector<unsigned char>& vchInfoHash,const  CMutableData& data)
{
    if (!pMutableDataDB) {
        return false;
    }
    if (!pMutableDataDB->UpdateMutableData(data)) {
        return false;
    }
    return true;
}

bool GetLocalMutableData(const std::vector<unsigned char>& vchInfoHash, CMutableData& data)
{
    if (!pMutableDataDB || !pMutableDataDB->ReadMutableData(vchInfoHash, data)) {
        return false;
    }
    return !data.IsNull();
}

bool PutLocalMutableData(const std::vector<unsigned char>& vchInfoHash, const CMutableData& data)
{
    if (!pMutableDataDB) {
        return false;
    }
    CMutableData readMutableData;
    if (pMutableDataDB->ReadMutableData(vchInfoHash, readMutableData)) {
        UpdateLocalMutableData(vchInfoHash, data);
    }
    else {
        AddLocalMutableData(vchInfoHash, data);
    }
    return !data.IsNull();
}

bool EraseLocalMutableData(const std::vector<unsigned char>& vchInfoHash)
{
    if (!pMutableDataDB) {
        return false;
    }
    if (!pMutableDataDB->EraseMutableData(vchInfoHash)) {
        return false;
    }
    return true;
}

bool GetAllLocalMutableData(std::vector<CMutableData>& vchMutableData)
{
    if (!pMutableDataDB) {
        return false;
    }
    if (!pMutableDataDB->ListMutableData(vchMutableData)) {
        return false;
    }
    return true;
}

bool InitMemoryMap()
{
    if (!pMutableDataDB)
        return false;

    if (!pMutableDataDB->LoadMemoryMap())
        return false;

    return true;
}

bool SelectRandomMutableItem(CMutableData& randomItem)
{
    if (!pMutableDataDB)
        return false;

    if (!pMutableDataDB->SelectRandomMutableItem(randomItem))
        return false;

    return true;
}

bool CheckMutableItemDB()
{
    if (!pMutableDataDB)
        return false;

    return true;
}

bool CMutableDataDB::AddMutableData(const CMutableData& data)
{
    bool writeState = false;
    {
        LOCK(cs_dht_entry);
        writeState = CDBWrapper::Write(make_pair(std::string("ih"), data.vchInfoHash), data);  // use info hash as key
        if (count >= 0) {
            mapDataStorage[data.vchInfoHash] = data;
            count++;
        }
    }
    return writeState;
}

bool CMutableDataDB::ReadMutableData(const std::vector<unsigned char>& vchInfoHash, CMutableData& data)
{
    if (count >= 0) {
        data = mapDataStorage[vchInfoHash];
        if (!data.IsNull())
            return true;
    }

    LOCK(cs_dht_entry);
    return CDBWrapper::Read(make_pair(std::string("ih"), vchInfoHash), data);
}

bool CMutableDataDB::EraseMutableData(const std::vector<unsigned char>& vchInfoHash)
{
    LOCK(cs_dht_entry);
    if (count >= 0)
        mapDataStorage.erase(vchInfoHash);

    return CDBWrapper::Erase(make_pair(std::string("ih"), vchInfoHash));
}

bool CMutableDataDB::UpdateMutableData(const CMutableData& data)
{
    LOCK(cs_dht_entry);

    if (!EraseMutableData(data.vchInfoHash))
        return false;

    bool writeState = false;
    writeState = CDBWrapper::Update(make_pair(std::string("ih"), data.vchInfoHash), data);
    if (count >= 0)
        mapDataStorage[data.vchInfoHash] = data;

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

bool CMutableDataDB::LoadMemoryMap()
{
    std::pair<std::string, CharString> infoHash;
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    count = 0;
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        CMutableData data;
        try {
            if (pcursor->GetKey(infoHash) && infoHash.first == "ih") {
                pcursor->GetValue(data);
                mapDataStorage[infoHash.second] = data;
            }
            pcursor->Next();
            count++;
        }
        catch (std::exception& e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

bool CMutableDataDB::SelectRandomMutableItem(CMutableData& randomItem) {
    if (count < 1)
        return false;

    unsigned int nAdvance = 0;
    std::map<std::vector<unsigned char>, CMutableData>::iterator it = mapDataStorage.begin();
    if (count != 1) {
        nAdvance = RandomIntegerRange(0, count - 1);
    }
    std::advance(it, nAdvance);
    randomItem = it->second;
    return true;
}