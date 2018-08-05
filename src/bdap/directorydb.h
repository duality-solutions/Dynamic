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
    bool ListDirectories(const std::vector<unsigned char>& vchObjectLocation, const unsigned int nResultsPerPage, const unsigned int nPage, UniValue& oDirectoryList);
};

bool GetDirectory(const std::vector<unsigned char>& vchObjectPath, CDirectory& directory);
bool CheckDirectoryDB();
bool FlushLevelDB();
void CleanupLevelDB(int& nRemoved);
bool CheckNewDirectoryTxInputs(const CTransaction& tx, const CDirectory& directory, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                               const int op, std::string& errorMessage, bool fJustCheck);
bool CheckDeleteDirectoryTxInputs(const CTransaction& tx, const CDirectory& directory, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                  const int op, std::string& errorMessage, bool fJustCheck);
bool CheckActivateDirectoryTxInputs(const CTransaction& tx, const CDirectory& directory, const vchCharString& vvchOpParameters,
                                    const int op, std::string& errorMessage, bool fJustCheck);
bool CheckUpdateDirectoryTxInputs(const CTransaction& tx, const CDirectory& directory, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                  const int op, std::string& errorMessage, bool fJustCheck);
bool CheckMoveDirectoryTxInputs(const CTransaction& tx, const CDirectory& directory, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                const int op, std::string& errorMessage, bool fJustCheck);
bool CheckExecuteDirectoryTxInputs(const CTransaction& tx, const CDirectory& directory, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                   const int op, std::string& errorMessage, bool fJustCheck);
bool CheckBindDirectoryTxInputs(const CTransaction& tx, const CDirectory& directory, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                const int op, std::string& errorMessage, bool fJustCheck);
bool CheckRevokeDirectoryTxInputs(const CTransaction& tx, const CDirectory& directory, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                  const int op, std::string& errorMessage, bool fJustCheck);
bool CheckDirectoryTxInputs(const CCoinsViewCache& inputs, const CTransaction& tx, 
                            int op, const std::vector<std::vector<unsigned char> >& vvchArgs, bool fJustCheck, int nHeight, std::string& errorMessage, bool bSanityCheck);

extern CDirectoryDB *pDirectoryDB;

#endif // DYNAMIC_DIRECTORYDB_H