// Copyright (c) 2020 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_CERTIFICATEDB_H
#define DYNAMIC_BDAP_CERTIFICATEDB_H

#include "bdap/x509certificate.h"
#include "dbwrapper.h"
#include "sync.h"

class CCoinsViewCache;
class UniValue;

static CCriticalSection cs_bdap_certificate;

class CCertificateDB : public CDBWrapper {
public:
    CCertificateDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "bdap-certificates", nCacheSize, fMemory, fWipe, obfuscate) {
    }
    bool AddCertificate(const CX509Certificate& certificate);
    bool ReadCertificateTxId(const std::vector<unsigned char>& vchTxId, CX509Certificate& certificate);
    bool ReadCertificateIssuerRootCA(const std::vector<unsigned char>& vchIssuer, CX509Certificate& certificate); 
    bool ReadCertificateSerialNumber(const uint64_t& nSerialNumber, CX509Certificate& certificate); 

    bool ReadCertificateSubjectDNRequest(const std::vector<unsigned char>& vchSubject, std::vector<CX509Certificate>& vCertificates, bool getAll = true);
    bool ReadCertificateIssuerDNRequest(const std::vector<unsigned char>& vchIssuer, std::vector<CX509Certificate>& vCertificates, bool getAll = true);
    bool ReadCertificateSubjectDNApprove(const std::vector<unsigned char>& vchSubject, std::vector<CX509Certificate>& vCertificates);
    bool ReadCertificateIssuerDNApprove(const std::vector<unsigned char>& vchSubject, std::vector<CX509Certificate>& vCertificates);

    bool EraseCertificateTxId(const std::vector<unsigned char>& vchTxId);
};

bool GetCertificateTxId(const std::string& strTxId, CX509Certificate& certificate);
bool GetCertificateSerialNumber(const std::string& strSerialNumber, CX509Certificate& certificate);
bool UndoAddCertificate(const CX509Certificate& certificate);
bool CheckCertificateDB();
bool FlushCertificateLevelDB();
bool CheckCertificateTx(const CTransactionRef& tx, const CScript& scriptOp, const int& op1, const int& op2, const std::vector<std::vector<unsigned char> >& vvchArgs, 
                                const bool fJustCheck, const int& nHeight, const uint32_t& nBlockTime, const bool bSanityCheck, std::string& errorMessage);

extern CCertificateDB *pCertificateDB;

#endif // DYNAMIC_BDAP_CERTIFICATEDB_H