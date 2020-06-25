// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_AUDITDB_H
#define DYNAMIC_BDAP_AUDITDB_H

#include "bdap/audit.h"
#include "dbwrapper.h"
#include "sync.h"

class CCoinsViewCache;
class UniValue;

static CCriticalSection cs_bdap_audit;

class CAuditDB : public CDBWrapper {
public:
    CAuditDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "bdap-audits", nCacheSize, fMemory, fWipe, obfuscate) {
    }
    bool AddAudit(const CAudit& audit, const int op);
    bool ReadAudit(const std::vector<unsigned char>& vchAudit, CAudit& audit);
    bool ReadAuditTxId(const std::vector<unsigned char>& vchTxId, CAudit& audit) ;
    bool EraseAuditTxId(const std::vector<unsigned char>& vchTxId);
    bool EraseAudit(const std::vector<unsigned char>& vchAudit);
    bool AuditExists(const std::vector<unsigned char>& vchAudit);
    bool GetAuditInfo(const std::vector<unsigned char>& vchAudit, UniValue& oAuditInfo);
    bool GetAuditInfo(const std::vector<unsigned char>& vchAudit, CAudit& audit);
};

bool GetAudit(const std::vector<unsigned char>& vchAudit, CAudit& audit);
bool GetAudit(const std::string& strAudit, CAudit& audit);
bool GetAuditTxId(const std::string& strTxId, CAudit& audit);
bool AuditExists(const std::vector<unsigned char>& vchAudit);
bool UndoAddAudit(const CAudit& audit);
bool CheckAuditDB();
bool FlushAuditLevelDB();
bool CheckAuditTx(const CTransactionRef& tx, const CScript& scriptOp, const int& op1, const int& op2, const std::vector<std::vector<unsigned char> >& vvchArgs, 
                                const bool fJustCheck, const int& nHeight, const uint32_t& nBlockTime, const bool bSanityCheck, std::string& errorMessage);

extern CAuditDB *pAuditDB;

#endif // DYNAMIC_BDAP_AUDITDB_H