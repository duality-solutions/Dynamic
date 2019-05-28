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

const BDAP::ObjectType DEFAULT_ACCOUNT_TYPE = BDAP::ObjectType::BDAP_DEFAULT_TYPE;

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
    bool ListDirectories(const std::vector<unsigned char>& vchObjectLocation, const unsigned int& nResultsPerPage, const unsigned int& nPage, UniValue& oDomainEntryList, const BDAP::ObjectType& accountType = DEFAULT_ACCOUNT_TYPE);
    bool GetDomainEntryInfo(const std::vector<unsigned char>& vchFullObjectPath, UniValue& oDomainEntryInfo);
    bool GetDomainEntryInfo(const std::vector<unsigned char>& vchFullObjectPath, CDomainEntry& entry);
};

bool GetDomainEntry(const std::vector<unsigned char>& vchObjectPath, CDomainEntry& entry);
bool GetDomainEntryPubKey(const std::vector<unsigned char>& vchPubKey, CDomainEntry& entry);
bool DomainEntryExists(const std::vector<unsigned char>& vchObjectPath);
bool DeleteDomainEntry(const CDomainEntry& entry);
bool CheckDomainEntryDB();
bool FlushLevelDB();
void CleanupLevelDB(int& nRemoved);
bool CheckDomainEntryTx(const CTransactionRef& tx, const CScript& scriptOp, const int& op1, const int& op2, const std::vector<std::vector<unsigned char> >& vvchArgs, 
                                const bool fJustCheck, const int& nHeight, const uint32_t& nBlockTime, const bool bSanityCheck, std::string& errorMessage);

extern CDomainEntryDB *pDomainEntryDB;

#endif // DYNAMIC_BDAP_DOMAINENTRYDB_H