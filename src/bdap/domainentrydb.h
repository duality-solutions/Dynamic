// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_DOMAINENTRYDB_H
#define DYNAMIC_BDAP_DOMAINENTRYDB_H

#include "bdap/domainentry.h"
#include "dbwrapper.h"
#include "sync.h"

class CCoinsViewCache;

static CCriticalSection cs_bdap_entry;

class CDomainEntryDB : public CDBWrapper {
public:
    CDomainEntryDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "bdap-entries", nCacheSize, fMemory, fWipe, obfuscate) {
    }

    // Add, Read, Modify, ModifyRDN, Delete, List, Search, Bind, and Compare
    bool AddDomainEntry(const CDomainEntry& entry, const int op);
    void AddDomainEntryIndex(const CDomainEntry& entry, const int op);
    bool ReadDomainEntry(const std::vector<unsigned char>& vchObjectPath, CDomainEntry& entry);
    bool ReadDomainEntryPubKey(const std::vector<unsigned char>& vchPubKey, CDomainEntry& entry);
    bool EraseDomainEntry(const std::vector<unsigned char>& vchObjectPath);
    bool EraseDomainEntryPubKey(const std::vector<unsigned char>& vchPubKey);
    bool DomainEntryExists(const std::vector<unsigned char>& vchObjectPath);
    bool DomainEntryExistsPubKey(const std::vector<unsigned char>& vchPubKey);
    bool RemoveExpired(int& entriesRemoved);
    void WriteDomainEntryIndex(const CDomainEntry& entry, const int op);
    void WriteDomainEntryIndexHistory(const CDomainEntry& entry, const int op);
    bool UpdateDomainEntry(const std::vector<unsigned char>& vchObjectPath, const CDomainEntry& entry);
    bool CleanupLevelDB(int& nRemoved);
    bool ListDirectories(const std::vector<unsigned char>& vchObjectLocation, const unsigned int nResultsPerPage, const unsigned int nPage, UniValue& oDomainEntryList);
    bool GetDomainEntryInfo(const std::vector<unsigned char>& vchFullObjectPath, UniValue& oDomainEntryInfo);
    bool GetDomainEntryInfo(const std::vector<unsigned char>& vchFullObjectPath, CDomainEntry& entry);
};

bool GetDomainEntry(const std::vector<unsigned char>& vchObjectPath, CDomainEntry& entry);
bool GetDomainEntryPubKey(const std::vector<unsigned char>& vchPubKey, CDomainEntry& entry);
bool CheckDomainEntryDB();
bool FlushLevelDB();
void CleanupLevelDB(int& nRemoved);
bool CheckNewDomainEntryTxInputs(const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                               std::string& errorMessage, bool fJustCheck);
bool CheckDeleteDomainEntryTxInputs(const CTransaction& tx, const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                  std::string& errorMessage, bool fJustCheck);
bool CheckUpdateDomainEntryTxInputs(const CTransaction& tx, const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                  std::string& errorMessage, bool fJustCheck);
bool CheckMoveDomainEntryTxInputs(const CTransaction& tx, const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                std::string& errorMessage, bool fJustCheck);
bool CheckExecuteDomainEntryTxInputs(const CTransaction& tx, const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                   std::string& errorMessage, bool fJustCheck);
bool CheckBindDomainEntryTxInputs(const CTransaction& tx, const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                std::string& errorMessage, bool fJustCheck);
bool CheckRevokeDomainEntryTxInputs(const CTransaction& tx, const CDomainEntry& entry, const CScript& scriptOp, const vchCharString& vvchOpParameters,
                                  std::string& errorMessage, bool fJustCheck);
bool CheckDomainEntryTx(const CTransactionRef& tx, const CScript& scriptOp, const int& op1, const int& op2, const std::vector<std::vector<unsigned char> >& vvchArgs, 
                                bool fJustCheck, int nHeight, std::string& errorMessage, bool bSanityCheck);

extern CDomainEntryDB *pDomainEntryDB;

#endif // DYNAMIC_BDAP_DOMAINENTRYDB_H