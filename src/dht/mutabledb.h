// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#ifndef DYNAMIC_DHT_MUTABLE_DB_H
#define DYNAMIC_DHT_MUTABLE_DB_H

#include "dbwrapper.h"
#include "sync.h"

static CCriticalSection cs_dht_entry;

class CMutableData;

class CMutableDataDB : public CDBWrapper {
public:
    CMutableDataDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "dht", nCacheSize, fMemory, fWipe, obfuscate) {
    }

    bool AddMutableData(const CMutableData& data);
    bool UpdateMutableData(const CMutableData& data);
    bool ReadMutableData(const std::vector<unsigned char>& vchInfoHash, CMutableData& data);
    bool EraseMutableData(const std::vector<unsigned char>& vchInfoHash);
    bool ListMutableData(std::vector<CMutableData>& vchMutableData);
};

bool AddLocalMutableData(const std::vector<unsigned char>& vchInfoHash, const CMutableData& data);
bool UpdateLocalMutableData(const std::vector<unsigned char>& vchInfoHash, const CMutableData& data);
bool GetLocalMutableData(const std::vector<unsigned char>& vchInfoHash, CMutableData& data);
bool PutLocalMutableData(const std::vector<unsigned char>& vchInfoHash, const CMutableData& data);
bool GetAllLocalMutableData(std::vector<CMutableData>& vchMutableData);

extern CMutableDataDB* pMutableDataDB;

#endif // DYNAMIC_DHT_MUTABLE_DB_H
