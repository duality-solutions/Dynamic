// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DIRECTORYDB_H
#define DYNAMIC_DIRECTORYDB_H

#include "bdap/directory.h"
#include "dbwrapper.h"

class CDirectoryDB : public CDBWrapper {
public:
    CDirectoryDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "bdap", nCacheSize, fMemory, fWipe, obfuscate) {
    }

    // Add, Read, Modify, ModifyRDN, Delete, List, Search, Bind, and Compare
    bool AddDirectory(const CDirectory& directory, const int& op);

    void AddDirectoryIndex(const CDirectory& directory, const int& op);

    bool ReadDirectory();

    bool UpdateDirectory();

    bool ExpireDirectory();

    bool DirectoryExists();

    bool CleanupDirectoryDatabase();
    

    void WriteDirectoryIndex(const CDirectory& directory, const int& op);
    void WriteDirectoryIndexHistory(const CDirectory& directory, const int& op);
};

extern CDirectoryDB *pDirectoryDB;

#endif // DYNAMIC_DIRECTORYDB_H