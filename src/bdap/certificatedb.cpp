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
#include "dht/ed25519.h"
#include "utilmoneystr.h"
#include "utiltime.h"
#include "validation.h"
#include "validationinterface.h"

#include <univalue.h>

#include <boost/thread.hpp>

CCertificateDB *pCertificateDB = NULL;

bool GetCertificateTxId(const std::string& strTxId, CX509Certificate& certificate)
{
    if (!pCertificateDB || !pCertificateDB->ReadCertificateTxId(vchFromString(strTxId), certificate))
        return false;

    return !certificate.IsNull();
}

bool GetCertificateSerialNumber(const std::string& strSerialNumber, CX509Certificate& certificate)
{
    uint64_t value;
    std::istringstream iss(strSerialNumber);
    iss >> value;

    if (!pCertificateDB || !pCertificateDB->ReadCertificateSerialNumber(value, certificate))
        return false;

    return !certificate.IsNull();
}

bool UndoAddCertificate(const CX509Certificate& certificate)
{
    if (!pCertificateDB)
        return false;

    if (certificate.IsApproved()) {
        return pCertificateDB->EraseCertificateTxId(vchFromString(certificate.txHashSigned.ToString()));
    }
    else {
        return pCertificateDB->EraseCertificateTxId(vchFromString(certificate.txHashRequest.ToString()));
    }

}

bool CCertificateDB::AddCertificate(const CX509Certificate& certificate) 
{ 
    bool writeState = false;
    bool writeStateCA = true;
    bool writeStateSerial = true;
    bool updateState = true;
    bool writeStateIssuerDN = true;
    bool writeStateSubjectDN = true;
    {
        LOCK(cs_bdap_certificate);

        std::string labelTxId;
        std::string labelSubjectDN;
        std::string labelIssuerDN;
        std::vector<unsigned char> vchTxHash;
        std::vector<unsigned char> vchTxHashRequest;

        if (certificate.IsRootCA){  //Root certificate
            vchTxHash = vchFromString(certificate.txHashSigned.ToString());
            labelTxId = "txrootcaid";
            //labelSubjectDN = "subjectdnrootca";
            //labelIssuerDN = "issuerdnrootca";
        }
        else if (certificate.IsApproved()){  //Approve
            vchTxHash = vchFromString(certificate.txHashSigned.ToString());
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

        if (!certificate.IsRootCA) {
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
        }

        //Certificate
        writeState = Write(make_pair(labelTxId, vchTxHash), certificate);

        //Serial Number
        if (certificate.SerialNumber != 0) {
            writeStateSerial = Write(make_pair(std::string("serialnumber"), certificate.SerialNumber), vchTxHash);
        }

        //if root certificate, index the issuer (subject=issuer). only one root certificate per bdap account
        if (certificate.IsRootCA){
            writeStateCA = Write(make_pair(std::string("issuerrootca"), certificate.Issuer), vchTxHash);
        }
        //if an approve (not self-signed), update the previous request certificate with txHashSigned
        else if ((certificate.IsApproved()) && (!certificate.SelfSignedX509Certificate())) {
            CX509Certificate requestCertificate;
            if (ReadCertificateTxId(vchTxHashRequest, requestCertificate)) {
                requestCertificate.txHashSigned = certificate.txHashSigned;
                updateState = Write(make_pair(std::string("txrequestid"), vchTxHashRequest), requestCertificate);
            }
        }

    }
    return writeState && writeStateIssuerDN && writeStateSubjectDN && updateState && writeStateCA && writeStateSerial;
}

bool CCertificateDB::ReadCertificateTxId(const std::vector<unsigned char>& vchTxId, CX509Certificate& certificate) 
{
    LOCK(cs_bdap_certificate);
    if(!(CDBWrapper::Read(make_pair(std::string("txrequestid"), vchTxId), certificate))) {
        if(!(CDBWrapper::Read(make_pair(std::string("txapproveid"), vchTxId), certificate))) {
            return CDBWrapper::Read(make_pair(std::string("txrootcaid"), vchTxId), certificate);
        }
    }
    return true;
}

bool CCertificateDB::ReadCertificateIssuerRootCA(const std::vector<unsigned char>& vchIssuer, CX509Certificate& certificate) 
{
    LOCK(cs_bdap_certificate);
    std::vector<unsigned char> vchTxId;
    bool readState = false;
    readState = CDBWrapper::Read(make_pair(std::string("issuerrootca"), vchIssuer), vchTxId);

    if (readState) {
        return ReadCertificateTxId(vchTxId, certificate);
    }
    else {
        return false;
    }
}

bool CCertificateDB::ReadCertificateSerialNumber(const uint64_t& nSerialNumber, CX509Certificate& certificate) 
{
    LOCK(cs_bdap_certificate);
    std::vector<unsigned char> vchTxId;
    bool readState = false;
    readState = CDBWrapper::Read(make_pair(std::string("serialnumber"), nSerialNumber), vchTxId);

    if (readState) {
        return ReadCertificateTxId(vchTxId, certificate);
    }
    else {
        return false;
    }
    return true;
}

bool CCertificateDB::ReadCertificateSubjectDNRequest(const std::vector<unsigned char>& vchSubject, std::vector<CX509Certificate>& vCertificates, bool getAll) 
{
    LOCK(cs_bdap_certificate);
    std::vector<std::vector<unsigned char>> vvTxId;
    std::vector<std::vector<unsigned char>> vvTxIdApprove;
    bool readState = false;
    readState = CDBWrapper::Read(make_pair(std::string("subjectdnrequest"), vchSubject), vvTxId);

    if (readState) {
        for (const std::vector<unsigned char>& vchTxId : vvTxId) {
            CX509Certificate certificate;
            if (ReadCertificateTxId(vchTxId, certificate)) {
                if (getAll || (!certificate.IsApproved())) {
                    vCertificates.push_back(certificate);
                }
            }
        }
    }

    return (vCertificates.size() > 0);
}

bool CCertificateDB::ReadCertificateIssuerDNRequest(const std::vector<unsigned char>& vchIssuer, std::vector<CX509Certificate>& vCertificates, bool getAll) 
{
    LOCK(cs_bdap_certificate);
    std::vector<std::vector<unsigned char>> vvTxId;
    std::vector<std::vector<unsigned char>> vvTxIdApprove;
    bool readState = false;
    readState = CDBWrapper::Read(make_pair(std::string("issuerdnrequest"), vchIssuer), vvTxId);

    if (readState) {
        for (const std::vector<unsigned char>& vchTxId : vvTxId) {
            CX509Certificate certificate;
            if (ReadCertificateTxId(vchTxId, certificate)) {
                if (getAll || (!certificate.IsApproved())) {
                    vCertificates.push_back(certificate);
                }
            }
        }
    }

    return (vCertificates.size() > 0);
}

bool CCertificateDB::ReadCertificateSubjectDNApprove(const std::vector<unsigned char>& vchSubject, std::vector<CX509Certificate>& vCertificates) 
{
    LOCK(cs_bdap_certificate);
    std::vector<std::vector<unsigned char>> vvTxId;
    bool readState = false;
    readState = CDBWrapper::Read(make_pair(std::string("subjectdnapprove"), vchSubject), vvTxId);

    if (readState) {
        for (const std::vector<unsigned char>& vchTxId : vvTxId) {
            CX509Certificate certificate;
            if (ReadCertificateTxId(vchTxId, certificate)) {
                vCertificates.push_back(certificate);
            }
        }
    }

    return (vCertificates.size() > 0);
}

bool CCertificateDB::ReadCertificateIssuerDNApprove(const std::vector<unsigned char>& vchIssuer, std::vector<CX509Certificate>& vCertificates) 
{
    LOCK(cs_bdap_certificate);
    std::vector<std::vector<unsigned char>> vvTxId;
    bool readState = false;
    readState = CDBWrapper::Read(make_pair(std::string("issuerdnapprove"), vchIssuer), vvTxId);

    if (readState) {
        for (const std::vector<unsigned char>& vchTxId : vvTxId) {
            CX509Certificate certificate;
            if (ReadCertificateTxId(vchTxId, certificate)) {
                vCertificates.push_back(certificate);
            }
        }
    }

    return (vCertificates.size() > 0);
}

bool CCertificateDB::EraseCertificateTxId(const std::vector<unsigned char>& vchTxId)
{
    LOCK(cs_bdap_certificate);
    CX509Certificate certificate;

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

    if (certificate.SerialNumber != 0) {
        CDBWrapper::Erase(make_pair(std::string("serialnumber"), certificate.SerialNumber));
    }

    if (certificate.IsRootCA) {
        return (CDBWrapper::Erase(make_pair(std::string("issuerrootca"), certificate.Issuer))) && (CDBWrapper::Erase(make_pair(std::string("txrootcaid"), vchTxId)));
    }
    else if (certificate.IsApproved()) {
        return CDBWrapper::Erase(make_pair(std::string("txapproveid"), vchTxId));
    }
    else {
        return CDBWrapper::Erase(make_pair(std::string("txrequestid"), vchTxId));
    }
    
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

static bool CommonDataCheck(const CX509Certificate& certificate, const vchCharString& vvchOpParameters, std::string& errorMessage)
{
    if (certificate.IsNull()) {
        errorMessage = "CommonDataCheck failed! Certificate is null.";
        return false;
    }

    if (!certificate.ValidateValues(errorMessage)) {
        errorMessage = "CommonDataCheck failed! Invalid certificate value. " + errorMessage;
        return false;
    }

    if (vvchOpParameters.size() > 6) {
        errorMessage = "CommonDataCheck failed! Too many parameters.";
        return false;
    }

    if (vvchOpParameters.size() < 5) {
        errorMessage = "CommonDataCheck failed! Not enough parameters.";
        return false;
    }

    if (certificate.Subject != vvchOpParameters[2]) {
        errorMessage = "CommonDataCheck failed! Script operation subject account parameter does not match subject account in certificate object.";
        return false;
    }

    if (certificate.Issuer != vvchOpParameters[4]) {
        errorMessage = "CommonDataCheck failed! Script operation issuer account parameter does not match issuer account in certificate object.";
        return false;
    }

    // check SubjectFQDN size
    if (vvchOpParameters.size() > 2 && vvchOpParameters[2].size() > MAX_OBJECT_FULL_PATH_LENGTH) {
        errorMessage = "CommonDataCheck failed! Subject FQDN is too large.";
        return false;
    }

    // check IssuerFQDN size
    if (vvchOpParameters.size() > 4 && vvchOpParameters[4].size() > MAX_OBJECT_FULL_PATH_LENGTH) {
        errorMessage = "CommonDataCheck failed! Issuer FQDN is too large.";
        return false;
    }

    // check subject pubkey size
    if (vvchOpParameters.size() > 3 && vvchOpParameters[3].size() > MAX_CERTIFICATE_KEY_LENGTH) {
        errorMessage = "CommonDataCheck failed! Subject PubKey is too large.";
        return false;
    }

    //check certificate pubkey size
    if (certificate.SubjectPublicKey.size() > MAX_CERTIFICATE_KEY_LENGTH) {
        errorMessage = "CommonDataCheck failed! Certificate PubKey is too large.";
        return false;
    }

    // if self-signed, pubkey of subject = issuer
    if (certificate.SelfSignedX509Certificate()) {
        if (vvchOpParameters[3] != vvchOpParameters[5]) {
            errorMessage = "CommonDataCheck failed! Self signed, but subject pubkey not equal to issuer pubkey.";
            return false;            
        }
    }

    // check if Months Valid is an accepted value
    uint32_t nMonthsValid;
    ParseUInt32(stringFromVch(vvchOpParameters[1]), &nMonthsValid);

    if (certificate.IsRootCA) {
        if (!(nMonthsValid > 0 && nMonthsValid <= MAX_CERTIFICATE_CA_MONTHS_VALID)) { // if NOT (nMonthsValid greater than 0 and less than or equal to 120)
            errorMessage = "CommonDataCheck failed! Months Valid is out of bounds.";
            return false;
        }
    }
    else {
        if (!(nMonthsValid > 0 && nMonthsValid <= MAX_CERTIFICATE_MONTHS_VALID)) { // if NOT (nMonthsValid greater than 0 and less than or equal to 12)
            errorMessage = "CommonDataCheck failed! Months Valid is out of bounds.";
            return false;
        }
    }

    // if approved or self signed, do additional checks
    if (certificate.IsApproved() || certificate.SelfSignedX509Certificate()) {

        // check issuer pubkey size
        if (vvchOpParameters.size() > 5 && vvchOpParameters[5].size() > MAX_CERTIFICATE_KEY_LENGTH) {
            errorMessage = "CommonDataCheck failed! Issuer PubKey is too large.";
            return false;
        }

    }

    return true;
}

static bool CheckNewCertificateTxInputs(const CX509Certificate& certificate, const CScript& scriptOp, const vchCharString& vvchOpParameters, const uint256& txHash,
                               const std::string& strOpType, std::string& errorMessage, bool fJustCheck)
{
    if (!CommonDataCheck(certificate, vvchOpParameters, errorMessage))
        return error(errorMessage.c_str());

    if (fJustCheck)
        return true;

    std::string strTxHashToUse;

    //confirm state of certificate matches strOpType
    if (strOpType == "bdap_new_certificate") {
        if (certificate.IsApproved()) {
            errorMessage = "CheckNewCertificateTxInputs: - Certificate approved but this is identified as a request op";
            return error(errorMessage.c_str());
        }
    }
    else if (strOpType == "bdap_approve_certificate") {
        if (!certificate.IsApproved()) {
            errorMessage = "CheckNewCertificateTxInputs: - Certificate not approved but this is identified as an approve op";
            return error(errorMessage.c_str());
        }
    }

    if (certificate.IsApproved()){  //Approve
        strTxHashToUse = certificate.txHashSigned.ToString();
    }
    else { //Request
        strTxHashToUse = certificate.txHashRequest.ToString();
    }

    CDomainEntry entrySubject;
    if (!GetDomainEntry(certificate.Subject, entrySubject)) {
        errorMessage = "CheckNewCertificateTxInputs: - Could not find specified certificate subject! " + stringFromVch(certificate.Subject);
        return error(errorMessage.c_str());
    }
   CharString vchSubjectPubKey = entrySubject.DHTPublicKey;

    if ( (!certificate.SelfSignedX509Certificate()) && (!certificate.IsApproved()) ) {
        //check subject signature (only if Request)
        if (!certificate.CheckSubjectSignature(EncodedPubKeyToBytes(vchSubjectPubKey))) { //test in rpc, should work
            errorMessage = "CheckNewCertificateTxInputs: - Could not validate subject signature. ";
            return error(errorMessage.c_str());
        }
    }

    //if approved check issuer signature and if not self signed check if request exists
    if (certificate.IsApproved()) {
        CDomainEntry entryIssuer;
        if (!GetDomainEntry(certificate.Issuer, entryIssuer)) {
            errorMessage = "CheckNewCertificateTxInputs: - Could not find specified certificate issuer! " + stringFromVch(certificate.Issuer);
            return error(errorMessage.c_str());
        }
        CDynamicAddress address = entryIssuer.GetWalletAddress();

        CharString vchIssuerPubKey = entryIssuer.DHTPublicKey;

        if (!certificate.CheckIssuerSignature(EncodedPubKeyToBytes(vchIssuerPubKey))) { //test in rpc, should work
            errorMessage = "CheckNewCertificateTxInputs: - Could not validate issuer signature. ";
            return error(errorMessage.c_str());
        }

        //if not self signed, check if request exists
        if (!certificate.SelfSignedX509Certificate()) {
            CX509Certificate certificateRequest;
            if (!GetCertificateTxId(certificate.txHashRequest.ToString(), certificateRequest)) {
                errorMessage = "CheckNewCertificateTxInputs: - Could not find previous request. ";
                return error(errorMessage.c_str());
            }
        }
    }

    //Check if Certificate already exists - should work w/Approve TXID occurring after Request
    CX509Certificate getCertificate;
    if (GetCertificateTxId(strTxHashToUse, getCertificate)) {
        if ((certificate.txHashSigned != txHash) && (certificate.txHashRequest != txHash)) { //check request and approve in case
            errorMessage = "CheckNewCertificateTxInputs: - The certificate " + txHash.ToString() + " already exists.  Add new certificate failed!";
            return error(errorMessage.c_str());
        } else {
            LogPrintf("%s -- Already have certificate %s in local database. Skipping add certificate step.\n", __func__, txHash.ToString());
            return true;
        }
    }    

    //make sure serial number doesn't already exist. 
    if (certificate.SerialNumber != 0) {
        CX509Certificate getCertificateSerial;
        if (GetCertificateSerialNumber(std::to_string(certificate.SerialNumber), getCertificateSerial)) {
            errorMessage = "CheckNewCertificateTxInputs: - The certificate serial number " + std::to_string(certificate.SerialNumber) + " already exists.  Add new certificate failed!";
            return error(errorMessage.c_str());
        }
    }

    if (!pCertificateDB) {
        errorMessage = "CheckNewCertificateTxInputs failed! Can not open LevelDB BDAP certificate database.";
        return error(errorMessage.c_str());
    }

    if (!pCertificateDB->AddCertificate(certificate)) {
        errorMessage = "CheckNewCertificateTxInputs failed! Error adding new certificate record to LevelDB.";
        pCertificateDB->EraseCertificateTxId(vchFromString(txHash.ToString())); 
        return error(errorMessage.c_str());
    }

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
    CX509Certificate certificate;
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nDataOut;
    bool bData = GetBDAPData(tx, vchData, vchHash, nDataOut);
    if(bData && !certificate.UnserializeFromTx(tx, nHeight))
    {
        errorMessage = ("UnserializeFromData data in tx failed!");
        LogPrintf("%s -- %s \n", __func__, errorMessage);
        return error(errorMessage.c_str());
    }
    const std::string strOperationType = GetBDAPOpTypeString(op1, op2);
    CAmount monthlyFee, oneTimeFee, depositFee;
    if (strOperationType == "bdap_new_certificate" || strOperationType == "bdap_approve_certificate") {
        if (!certificate.ValidatePEM(errorMessage))
            return false;

        if (!certificate.ValidateValues(errorMessage))
            return false;

        if (vvchArgs.size() > 6) {
            errorMessage = "Failed to get fees to add a certificate request";
            return false;
        }
        std::string strCount = stringFromVch(vvchArgs[1]);
        uint32_t nCount;
        ParseUInt32(stringFromVch(vvchArgs[1]), &nCount);

        if (strOperationType == "bdap_new_certificate") { //Request
            if (!GetBDAPFees(OP_BDAP_NEW, OP_BDAP_CERTIFICATE, BDAP::ObjectType::BDAP_CERTIFICATE, (uint16_t)nCount, monthlyFee, oneTimeFee, depositFee)) {
                errorMessage = "Failed to get fees to add a certificate request";
                return false;
            }
        }
        else { //Approve
            if (!GetBDAPFees(OP_BDAP_MODIFY, OP_BDAP_CERTIFICATE, BDAP::ObjectType::BDAP_CERTIFICATE, (uint16_t)nCount, monthlyFee, oneTimeFee, depositFee)) {
                errorMessage = "Failed to get fees to add a certificate appoval";
                return false;
            }

        }

        LogPrint("bdap", "%s -- nCount %d, oneTimeFee %d\n", __func__, nCount, FormatMoney(oneTimeFee));
        // extract amounts from tx.
        CAmount dataAmount, opAmount;
        if (!ExtractAmountsFromTx(tx, dataAmount, opAmount)) {
            errorMessage = "Unable to extract BDAP amounts from transaction";
            return false;
        }
        LogPrint("bdap", "%s -- dataAmount %d, opAmount %d\n", __func__, FormatMoney(dataAmount), FormatMoney(opAmount));
        if (monthlyFee > dataAmount) {
            LogPrintf("%s -- Invalid BDAP monthly registration fee amount for certificate. Monthly paid %d but should be %d\n", __func__, 
                                    FormatMoney(dataAmount), FormatMoney(monthlyFee));
            errorMessage = "Invalid BDAP monthly registration fee amount for certificate";
            return false;
        }
        else {
            LogPrint("bdap", "%s -- Valid BDAP monthly registration fee amount for certificate. Monthly paid %d, should be %d.\n", __func__, 
                                    FormatMoney(dataAmount), FormatMoney(monthlyFee));
        }
        if (depositFee > opAmount) {
            LogPrintf("%s -- Invalid BDAP deposit fee amount for certificate. Deposit paid %d but should be %d\n", __func__, 
                                    FormatMoney(opAmount), FormatMoney(depositFee));
            errorMessage = "Invalid BDAP deposit fee amount for certificate";
            return false;
        }
        else {
            LogPrint("bdap", "%s -- Valid BDAP deposit fee amount for certificate. Deposit paid %d, should be %d\n", __func__, 
                                    FormatMoney(opAmount), FormatMoney(depositFee));
        }

        return CheckNewCertificateTxInputs(certificate, scriptOp, vvchArgs, tx->GetHash(), strOperationType, errorMessage, fJustCheck);
    }

    return false;

}
