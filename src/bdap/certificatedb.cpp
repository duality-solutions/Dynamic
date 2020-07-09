// Copyright (c) 2020 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/certificatedb.h"

#include "amount.h"
#include "base58.h"
#include "bdap/domainentrydb.h"
#include "bdap/fees.h"
#include "bdap/utils.h"
#include "coins.h"
#include "utilmoneystr.h"
#include "utiltime.h"
#include "validation.h"
#include "validationinterface.h"

#include <univalue.h>

#include <boost/thread.hpp>

CCertificateDB *pCertificateDB = NULL;

bool GetCertificate(const std::vector<unsigned char>& vchCertificate, CCertificate& certificate)
{
    if (!pCertificateDB || !pCertificateDB->ReadCertificate(vchCertificate, certificate))
        return false;

    return !certificate.IsNull();
}

bool GetCertificate(const std::string& strCertificate, CCertificate& certificate)
{
    if (!pCertificateDB || !pCertificateDB->ReadCertificate(vchFromString(strCertificate), certificate))
        return false;

    return !certificate.IsNull();
}

bool GetCertificateTxId(const std::string& strTxId, CCertificate& certificate)
{
    if (!pCertificateDB || !pCertificateDB->ReadCertificateTxId(vchFromString(strTxId), certificate))
        return false;

    return !certificate.IsNull();
}

bool CertificateExists(const std::vector<unsigned char>& vchCertificate)
{
    if (!pCertificateDB)
        return false;

    return pCertificateDB->CertificateExists(vchCertificate);
}

bool UndoAddCertificate(const CCertificate& certificate)
{
    if (!pCertificateDB)
        return false;

    return pCertificateDB->EraseCertificateTxId(vchFromString(certificate.GetHash().ToString()));
}

bool CCertificateDB::AddCertificate(const CCertificate& certificate, const int op) 
{ 
    bool writeState = false;
    bool writeStateIssuerDN = false;
    bool writeStateSubjectDN = false;
    {
        LOCK(cs_bdap_certificate);

        //WIP Needs Review

        //WIP Indexes
        //SerialNumber?
        //Should NewCertificate set txHashRequest = txHashApprove if self-signed?
        writeStateSubjectDN = Write(make_pair(std::string("subjectdn"), certificate.Subject), vchFromString(certificate.txHashRequest.ToString()));

        if (certificate.Issuer.size() > 0) {
            writeStateIssuerDN = Write(make_pair(std::string("issuerdn"), certificate.Issuer), vchFromString(certificate.txHashRequest.ToString()));
        }
        else {
            writeStateIssuerDN = true;
        }

        if (certificate.IsApproved()){
            writeState = Write(make_pair(std::string("txapproveid"), vchFromString(certificate.txHashApprove.ToString())), certificate);
        }
        else {
            writeState = Write(make_pair(std::string("txrequestid"), vchFromString(certificate.txHashRequest.ToString())), certificate);
        }
    }

    return writeState && writeStateIssuerDN && writeStateSubjectDN;
}

bool CCertificateDB::ReadCertificateTxId(const std::vector<unsigned char>& vchTxId, CCertificate& certificate) 
{
    LOCK(cs_bdap_certificate);
    return CDBWrapper::Read(make_pair(std::string("txid"), vchTxId), certificate);
}


bool CCertificateDB::ReadCertificate(const std::vector<unsigned char>& vchCertificate, CCertificate& certificate) 
{
    std::vector<unsigned char> vchTxId;
    LOCK(cs_bdap_certificate);
    if (CDBWrapper::Read(make_pair(std::string("certificate"), vchCertificate), vchTxId)) {
        if (!ReadCertificateTxId(vchTxId, certificate))
            return false;
    } else {
        return false;
    }
    return true;
}

bool CCertificateDB::CertificateExists(const std::vector<unsigned char>& vchCertificate)
{
    LOCK(cs_bdap_certificate);
    return CDBWrapper::Exists(make_pair(std::string("certificate"), vchCertificate));
}

bool CCertificateDB::EraseCertificateTxId(const std::vector<unsigned char>& vchTxId)
{
    LOCK(cs_bdap_certificate);
    CCertificate certificate;

    return false;
}

bool CCertificateDB::EraseCertificate(const std::vector<unsigned char>& vchCertificate)
{
    LOCK(cs_bdap_certificate);
    return CDBWrapper::Erase(make_pair(std::string("certificate"), vchCertificate));
}

bool CCertificateDB::GetCertificateInfo(const std::vector<unsigned char>& vchCertificate, UniValue& oCertificateInfo)
{
    CCertificate certificate;
    if (!ReadCertificate(vchCertificate, certificate))
        return false;

    if (!BuildCertificateJson(certificate, oCertificateInfo))
        return false;  

    return true;
}

bool CCertificateDB::GetCertificateInfo(const std::vector<unsigned char>& vchCertificate, CCertificate& certificate)
{
    if (!ReadCertificate(vchCertificate, certificate))
        return false;

    return true;
}

bool CheckCertificateDB()
{
    if (!pCertificateDB)
        return false;

    return true;
}

bool FlushCertificateLevelDB() 
{
    {
        LOCK(cs_bdap_certificate);
        if (pCertificateDB != NULL)
        {
            if (!pCertificateDB->Flush()) {
                LogPrintf("Failed to flush Certificate BDAP database!");
                return false;
            }
        }
    }
    return true;
}

static bool CommonDataCheck(const CCertificate& certificate, const vchCharString& vvchOpParameters, std::string& errorMessage)
{
    if (certificate.IsNull()) {
        errorMessage = "CommonDataCheck failed! Certificate is null.";
        return false;
    }

    return false;
}

static bool CheckNewCertificateTxInputs(const CCertificate& certificate, const CScript& scriptOp, const vchCharString& vvchOpParameters, const uint256& txHash,
                               std::string& errorMessage, bool fJustCheck)
{
    if (!CommonDataCheck(certificate, vvchOpParameters, errorMessage))
        return error(errorMessage.c_str());

    if (fJustCheck)
        return true;


    return FlushCertificateLevelDB();
}

bool CheckCertificateTx(const CTransactionRef& tx, const CScript& scriptOp, const int& op1, const int& op2, const std::vector<std::vector<unsigned char> >& vvchArgs, 
                                const bool fJustCheck, const int& nHeight, const uint32_t& nBlockTime, const bool bSanityCheck, std::string& errorMessage) 
{
    if (tx->IsCoinBase() && !fJustCheck && !bSanityCheck) {
        LogPrintf("*Trying to add BDAP certificate in coinbase transaction, skipping...");
        return true;
    }

    LogPrint("bdap", "%s -- BDAP nHeight=%d, chainActive.Tip()=%d, op1=%s, op2=%s, hash=%s justcheck=%s\n", __func__, nHeight, chainActive.Tip()->nHeight, BDAPFromOp(op1).c_str(), BDAPFromOp(op2).c_str(), tx->GetHash().ToString().c_str(), fJustCheck ? "JUSTCHECK" : "BLOCK");

    // unserialize BDAP from txn, check if the certificate is valid and does not conflict with a previous certificate
    CCertificate certificate;
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;



    return false;
}
