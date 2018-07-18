// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DIRECTORYDB_H
#define DYNAMIC_DIRECTORYDB_H

#include "bdap/directory.h"
#include "dbwrapper.h"

static CCriticalSection cs_bdap_directory;

class CDirectoryDB : public CDBWrapper {
public:
    CDirectoryDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "bdap", nCacheSize, fMemory, fWipe, obfuscate) {
    }

    // Add, Read, Modify, ModifyRDN, Delete, List, Search, Bind, and Compare
    bool AddDirectory(const CDirectory& directory, const int op);
    void AddDirectoryIndex(const CDirectory& directory, const int op);
    bool ReadDirectory(const std::vector<unsigned char>& vchObjectPath, CDirectory& directory);
    bool ReadDirectoryAddress(const std::vector<unsigned char>& vchAddress, std::vector<unsigned char>& vchObjectPath);
    bool EraseDirectory(const std::vector<unsigned char>& vchObjectPath);
    bool EraseDirectoryAddress(const std::vector<unsigned char>& vchAddress);
    bool DirectoryExists(const std::vector<unsigned char>& vchObjectPath);
    bool DirectoryExistsAddress(const std::vector<unsigned char>& vchAddress);
    bool RemoveExpired(int& entriesRemoved);
    void WriteDirectoryIndex(const CDirectory& directory, const int op);
    void WriteDirectoryIndexHistory(const CDirectory& directory, const int op);
    bool UpdateDirectory(const std::vector<unsigned char>& vchObjectPath, CDirectory& directory);
    bool UpdateDirectoryAddress(const std::vector<unsigned char>& vchAddress, CDirectory& directory);
    bool CleanupLevelDB(int& nRemoved);
};

bool GetDirectory(const std::vector<unsigned char>& vchObjectPath, CDirectory& directory);
bool CheckDirectoryDB();
bool FlushLevelDB();
void CleanupLevelDB(int& nRemoved);

extern CDirectoryDB *pDirectoryDB;

#endif // DYNAMIC_DIRECTORYDB_H