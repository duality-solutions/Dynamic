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
    bool updateState = true;
    bool writeStateIssuerDN = false;
    bool writeStateSubjectDN = false;
    {
        LOCK(cs_bdap_certificate);

        std::string labelTxId;
        std::string labelSubjectDN;
        std::string labelIssuerDN;
        std::vector<unsigned char> vchTxHash;
        std::vector<unsigned char> vchTxHashRequest;

        if (certificate.IsApproved()){  //Approve
            vchTxHash = vchFromString(certificate.txHashApprove.ToString());
            vchTxHashRequest = vchFromString(certificate.txHashRequest.ToString());
            labelTxId = "txapproveid";
            labelSubjectDN = "subjectdnapprove";
            labelIssuerDN = "issuerdnapprove";
        }
        else { //Request
            vchTxHash = vchFromString(certificate.txHashRequest.ToString());
            labelTxId = "txrequestid";
            labelSubjectDN = "subjectdnrequest";
            labelIssuerDN = "issuerdnrequest";
        }

        //Subject
        std::vector<std::vector<unsigned char>> vvTxIdSubject;
        CDBWrapper::Read(make_pair(labelSubjectDN, certificate.Subject), vvTxIdSubject);
        vvTxIdSubject.push_back(vchTxHash);
        writeStateSubjectDN = Write(make_pair(labelSubjectDN, certificate.Subject), vvTxIdSubject);

        //Issuer
        std::vector<std::vector<unsigned char>> vvTxIdIssuer;
        CDBWrapper::Read(make_pair(labelIssuerDN, certificate.Issuer), vvTxIdIssuer);
        vvTxIdIssuer.push_back(vchTxHash);
        writeStateIssuerDN = Write(make_pair(labelIssuerDN, certificate.Issuer), vvTxIdIssuer);

        //Certificate
        writeState = Write(make_pair(labelTxId, vchTxHash), certificate);

        //if an approve (not self-signed), update the previous request certificate with txHashApprove
        if ((certificate.IsApproved()) && (!certificate.SelfSignedCertificate())) {
            CCertificate requestCertificate;
            if (ReadCertificateTxId(vchTxHashRequest, requestCertificate)) {
                requestCertificate.txHashApprove = certificate.txHashApprove;
                updateState = Write(make_pair(std::string("txrequestid"), vchTxHashRequest), requestCertificate);
            }
        }

    }
    return writeState && writeStateIssuerDN && writeStateSubjectDN && updateState;
}

bool CCertificateDB::ReadCertificateTxId(const std::vector<unsigned char>& vchTxId, CCertificate& certificate) 
{
    LOCK(cs_bdap_certificate);
    if(!(CDBWrapper::Read(make_pair(std::string("txrequestid"), vchTxId), certificate))) {
        return CDBWrapper::Read(make_pair(std::string("txapproveid"), vchTxId), certificate);
    }
    return true;
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

bool CCertificateDB::ReadCertificateSubjectDNRequest(const std::vector<unsigned char>& vchSubject, std::vector<CCertificate>& vCertificates, bool getAll) 
{
    LOCK(cs_bdap_certificate);
    std::vector<std::vector<unsigned char>> vvTxId;
    std::vector<std::vector<unsigned char>> vvTxIdApprove;
    bool readState = false;
    readState = CDBWrapper::Read(make_pair(std::string("subjectdnrequest"), vchSubject), vvTxId);

    if (readState) {
        for (const std::vector<unsigned char>& vchTxId : vvTxId) {
            CCertificate certificate;
            if (ReadCertificateTxId(vchTxId, certificate)) {
                if (getAll || (!certificate.IsApproved())) {
                    vCertificates.push_back(certificate);
                }
            }
        }
    }

    return (vCertificates.size() > 0);
}

bool CCertificateDB::ReadCertificateIssuerDNRequest(const std::vector<unsigned char>& vchIssuer, std::vector<CCertificate>& vCertificates, bool getAll) 
{
    LOCK(cs_bdap_certificate);
    std::vector<std::vector<unsigned char>> vvTxId;
    std::vector<std::vector<unsigned char>> vvTxIdApprove;
    bool readState = false;
    readState = CDBWrapper::Read(make_pair(std::string("issuerdnrequest"), vchIssuer), vvTxId);

    if (readState) {
        for (const std::vector<unsigned char>& vchTxId : vvTxId) {
            CCertificate certificate;
            if (ReadCertificateTxId(vchTxId, certificate)) {
                if (getAll || (!certificate.IsApproved())) {
                    vCertificates.push_back(certificate);
                }
            }
        }
    }

    return (vCertificates.size() > 0);
}

bool CCertificateDB::ReadCertificateSubjectDNApprove(const std::vector<unsigned char>& vchSubject, std::vector<CCertificate>& vCertificates) 
{
    LOCK(cs_bdap_certificate);
    std::vector<std::vector<unsigned char>> vvTxId;
    bool readState = false;
    readState = CDBWrapper::Read(make_pair(std::string("subjectdnapprove"), vchSubject), vvTxId);

    if (readState) {
        for (const std::vector<unsigned char>& vchTxId : vvTxId) {
            CCertificate certificate;
            if (ReadCertificateTxId(vchTxId, certificate)) {
                vCertificates.push_back(certificate);
            }
        }
    }

    return (vCertificates.size() > 0);
}

bool CCertificateDB::ReadCertificateIssuerDNApprove(const std::vector<unsigned char>& vchIssuer, std::vector<CCertificate>& vCertificates) 
{
    LOCK(cs_bdap_certificate);
    std::vector<std::vector<unsigned char>> vvTxId;
    bool readState = false;
    readState = CDBWrapper::Read(make_pair(std::string("issuerdnapprove"), vchIssuer), vvTxId);

    if (readState) {
        for (const std::vector<unsigned char>& vchTxId : vvTxId) {
            CCertificate certificate;
            if (ReadCertificateTxId(vchTxId, certificate)) {
                vCertificates.push_back(certificate);
            }
        }
    }

    return (vCertificates.size() > 0);
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

    if (!ReadCertificateTxId(vchTxId, certificate))
        return false;

    //subjectrequest
    std::vector<std::vector<unsigned char>> vvSubjectRequestTxId;
    CDBWrapper::Read(make_pair(std::string("subjectdnrequest"), certificate.Subject), vvSubjectRequestTxId);
    if (vvSubjectRequestTxId.size() == 1 && vvSubjectRequestTxId[0] == vchTxId) {
        CDBWrapper::Erase(make_pair(std::string("subjectdnrequest"), certificate.Subject));
    }
    else {
        std::vector<std::vector<unsigned char>> vvTxIdSubjectRequestNew;
        for (const std::vector<unsigned char>& txid : vvSubjectRequestTxId) {
            if (txid != vchTxId) {
                vvTxIdSubjectRequestNew.push_back(txid);
            }
        }
        Write(make_pair(std::string("subjectdnrequest"), certificate.Subject), vvTxIdSubjectRequestNew);
    }

    //issuerrequest
    std::vector<std::vector<unsigned char>> vvIssuerRequestTxId;
    CDBWrapper::Read(make_pair(std::string("issuerdnrequest"), certificate.Issuer), vvIssuerRequestTxId);
    if (vvIssuerRequestTxId.size() == 1 && vvIssuerRequestTxId[0] == vchTxId) {
        CDBWrapper::Erase(make_pair(std::string("issuerdnrequest"), certificate.Issuer));
    }
    else {
        std::vector<std::vector<unsigned char>> vvTxIdIssuerRequestNew;
        for (const std::vector<unsigned char>& txid : vvIssuerRequestTxId) {
            if (txid != vchTxId) {
                vvTxIdIssuerRequestNew.push_back(txid);
            }
        }
        Write(make_pair(std::string("issuerdnrequest"), certificate.Issuer), vvTxIdIssuerRequestNew);
    }

    //subjectapprove
    std::vector<std::vector<unsigned char>> vvSubjectApproveTxId;
    CDBWrapper::Read(make_pair(std::string("subjectdnapprove"), certificate.Subject), vvSubjectApproveTxId);
    if (vvSubjectApproveTxId.size() == 1 && vvSubjectApproveTxId[0] == vchTxId) {
        CDBWrapper::Erase(make_pair(std::string("subjectdnapprove"), certificate.Subject));
    }
    else {
        std::vector<std::vector<unsigned char>> vvTxIdSubjectApproveNew;
        for (const std::vector<unsigned char>& txid : vvSubjectApproveTxId) {
            if (txid != vchTxId) {
                vvTxIdSubjectApproveNew.push_back(txid);
            }
        }
        Write(make_pair(std::string("subjectdnapprove"), certificate.Subject), vvTxIdSubjectApproveNew);
    }

    //issuerapprove
    std::vector<std::vector<unsigned char>> vvIssuerApproveTxId;
    CDBWrapper::Read(make_pair(std::string("issuerdnapprove"), certificate.Issuer), vvIssuerApproveTxId);
    if (vvIssuerApproveTxId.size() == 1 && vvIssuerApproveTxId[0] == vchTxId) {
        CDBWrapper::Erase(make_pair(std::string("issuerdnapprove"), certificate.Issuer));
    }
    else {
        std::vector<std::vector<unsigned char>> vvTxIdIssuerApproveNew;
        for (const std::vector<unsigned char>& txid : vvIssuerApproveTxId) {
            if (txid != vchTxId) {
                vvTxIdIssuerApproveNew.push_back(txid);
            }
        }
        Write(make_pair(std::string("issuerdnapprove"), certificate.Issuer), vvTxIdIssuerApproveNew);
    }

    if (certificate.IsApproved()) {
        return CDBWrapper::Erase(make_pair(std::string("txapproveid"), vchTxId));
    }
    else {
        return CDBWrapper::Erase(make_pair(std::string("txrequestid"), vchTxId));
    }
    
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
