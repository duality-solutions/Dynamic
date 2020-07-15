// Copyright (c) 2020 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_CERTIFICATEDB_H
#define DYNAMIC_BDAP_CERTIFICATEDB_H

#include "bdap/certificate.h"
#include "dbwrapper.h"
#include "sync.h"

class CCoinsViewCache;
class UniValue;

static CCriticalSection cs_bdap_certificate;

class CCertificateDB : public CDBWrapper {
public:
    CCertificateDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "bdap-certificates", nCacheSize, fMemory, fWipe, obfuscate) {
    }
    bool AddCertificate(const CCertificate& certificate, const int op);
    bool ReadCertificate(const std::vector<unsigned char>& vchCertificate, CCertificate& certificate);
    bool ReadCertificateTxId(const std::vector<unsigned char>& vchTxId, CCertificate& certificate);

    bool ReadCertificateSubjectDNRequest(const std::vector<unsigned char>& vchSubject, std::vector<CCertificate>& vCertificates, bool getAll = true);
    bool ReadCertificateIssuerDNRequest(const std::vector<unsigned char>& vchIssuer, std::vector<CCertificate>& vCertificates, bool getAll = true);
    bool ReadCertificateSubjectDNApprove(const std::vector<unsigned char>& vchSubject, std::vector<CCertificate>& vCertificates);
    bool ReadCertificateIssuerDNApprove(const std::vector<unsigned char>& vchSubject, std::vector<CCertificate>& vCertificates);

    bool EraseCertificateTxId(const std::vector<unsigned char>& vchTxId);
    bool EraseCertificate(const std::vector<unsigned char>& vchCertificate);
    bool CertificateExists(const std::vector<unsigned char>& vchCertificate);
    bool GetCertificateInfo(const std::vector<unsigned char>& vchCertificate, UniValue& oCertificateInfo);
    bool GetCertificateInfo(const std::vector<unsigned char>& vchCertificate, CCertificate& certificate);
};

bool GetCertificate(const std::vector<unsigned char>& vchCertificate, CCertificate& certificate);
bool GetCertificate(const std::string& strCertificate, CCertificate& certificate);
bool GetCertificateTxId(const std::string& strTxId, CCertificate& certificate);
bool CertificateExists(const std::vector<unsigned char>& vchCertificate);
bool UndoAddCertificate(const CCertificate& certificate);
bool CheckCertificateDB();
bool FlushCertificateLevelDB();
bool CheckCertificateTx(const CTransactionRef& tx, const CScript& scriptOp, const int& op1, const int& op2, const std::vector<std::vector<unsigned char> >& vvchArgs, 
                                const bool fJustCheck, const int& nHeight, const uint32_t& nBlockTime, const bool bSanityCheck, std::string& errorMessage);

extern CCertificateDB *pCertificateDB;

#endif // DYNAMIC_BDAP_CERTIFICATEDB_H