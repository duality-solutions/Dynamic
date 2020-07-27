// Copyright (c) 2020 Duality Blockchain Solutions Developers 
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/certificate.h"
#include "bdap/certificatedb.h"
#include "bdap/domainentry.h"
#include "bdap/domainentrydb.h"
#include "bdap/fees.h"
#include "bdap/utils.h"
#include "core_io.h" // needed for ScriptToAsmStr
#include "dynode-sync.h"
#include "dynodeman.h"
#include "rpc/protocol.h"
#include "rpc/server.h"
#include "primitives/transaction.h"
#include "spork.h"
#include "timedata.h"
#include "utilmoneystr.h"
#include "uint256.h"
#include "utilstrencodings.h"
#include "validation.h"
#include "wallet/wallet.h"

#include <libtorrent/ed25519.hpp>
#include <univalue.h>

extern void SendBDAPTransaction(const CScript& bdapDataScript, const CScript& bdapOPScript, CWalletTx& wtxNew, const CAmount& nDataAmount, const CAmount& nOpAmount, const bool fUseInstantSend);

static UniValue NewCertificate(const JSONRPCRequest& request)
{
#ifdef ENABLE_WALLET
    if (request.fHelp || (request.params.size() != 4 ))
        throw std::runtime_error(
            "certificate new \"subject\" ( \"issuer\" ) \"key_usage_array\" \n"
            "\nAdds an X.509 certificate to the blockchain.\n"
            "\nArguments:\n"
            "1. \"subject\"          (string, required)  BDAP account that created certificate\n"
            "2. \"issuer\"           (string, optional)  BDAP account that issued certificate\n"
            "3. \"key_usage_array\"  (string, required)  Descriptions of how this certificate will be used\n"
            "\nResult:\n"
            "{(json object)\n"
            " \"version\"                   (string, required)   Version \n"
            " \"signature_algorithm\"       (string, required)   Algorithm used to sign \n"
            " \"signature_hash_algorithm\"  (string, required)   Algorithm used to hash \n"
            " \"fingerprint\"               (string, required)   Fingerprint of certificate \n"
            " \"months_valid\"              (int, required)      How long certificate is valid \n"
            " \"subject\"                   (string, required)   BDAP account of subject \n"
            " \"subject_signature\"         (string, required)   Signature of subject \n"
            " \"issuer\"                    (string, required)   BDAP account of issuer \n"
            " \"public_key\"                (string, required)   Public Key of certificate \n"
            " \"signature_value\"           (string, optional)   Signature of approval \n"
            " \"approved\"                  (boolean, required)  Certificate approved \n"
            " \"serial_number\"             (string, required)   Unique serial number \n"
            " \"certificate_keyid\"         (string, required)   Key ID \n"
            " \"key_usage\"                 (string, required)   List of usages \n"
            " \"txid_request\"              (string, required)   Certificate request transaction id\n"
            " \"txid_approve\"              (string, optional)   Certificate approved transaction id  \n"
            " \"request_time\"              (int, required)      Time when request was made \n"
            " \"request_height\"            (int, required)      Block where request is stored \n"
            " \"valid_from\"                (int, optional)      Time when certificate is valid \n"
            " \"valid_until\"               (int, optional)      Time when certificate expires \n"
            " \"approve_height\"            (int, optional)      Block where approval is stored \n"
            "}\n"
            "\nExamples\n" +
           HelpExampleCli("certificate new", "\"subject\" (\"issuer\") \"key_usage_array\" ") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("certificate new", "\"subject\" (\"issuer\")  \"key_usage_array\" "));

    EnsureWalletIsUnlocked();

    CCertificate txCertificate;
    CharString vchSubjectFQDN;
    CharString vchIssuerFQDN;
    bool selfSign = false;
    int32_t nMonths = 12; // certificates last a year?

    //initialize Certificate with values we know
    txCertificate.SignatureAlgorithm = vchFromString("ed25519"); 
    txCertificate.SignatureHashAlgorithm = vchFromString("sha512"); 
    txCertificate.SerialNumber = GetTimeMillis();

    std::string strIssuer = request.params[2].get_str();
    if (strIssuer.size() > 0) {
        selfSign = false;
    }
    else {
        selfSign = true;
    }

    //ALSO considered self-sign if subject = issuer
    if (request.params[1].get_str() == request.params[2].get_str()) {
        selfSign = true;
    }

    //Handle Key Usage array [REQUIRED]
    std::string strKeyUsages = request.params[3].get_str();

    if (!(strKeyUsages.size() > 0)) {
        throw std::runtime_error("BDAP_CERTIFICATE_NEW_RPC_ERROR: Key usage cannot be empty");
    }

    if (strKeyUsages.find(",") > 0) {
        std::vector<std::string> vKeyUsages = SplitString(strKeyUsages, ',');
        for(const std::string& strKeyUsage : vKeyUsages)
            txCertificate.KeyUsage.push_back(vchFromString(TrimString(strKeyUsage)));
    } else {
        txCertificate.KeyUsage.push_back(vchFromValue(strKeyUsages));
    }

    //Handle SUBJECT [required]
    std::string strSubjectFQDN = request.params[1].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
    ToLowerCase(strSubjectFQDN);
    vchSubjectFQDN = vchFromString(strSubjectFQDN);
    // Check if name exists
    CDomainEntry subjectDomainEntry;
    if (!GetDomainEntry(vchSubjectFQDN, subjectDomainEntry))
        throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", strSubjectFQDN));

    txCertificate.Subject = subjectDomainEntry.vchFullObjectPath();
    
    //Get Subject BDAP user Public Key
    CharString vchSubjectPubKey = subjectDomainEntry.DHTPublicKey;
    CKeyEd25519 privSubjectDHTKey;
    std::vector<unsigned char> SubjectSecretKey;
    std::vector<unsigned char> SubjectPublicKey;

    //Generate Subject ed25519 key
    CKeyID vchSubjectPubKeyID = GetIdFromCharVector(vchSubjectPubKey);
    if (!pwalletMain->GetDHTKey(vchSubjectPubKeyID, privSubjectDHTKey))
        throw std::runtime_error("BDAP_CERTIFICATE_NEW_RPC_ERROR: Unable to retrieve DHT Key");

    SubjectSecretKey = privSubjectDHTKey.GetPrivKeyBytes();
    SubjectPublicKey = privSubjectDHTKey.GetPubKeyBytes();

    //Subject Signs
    if (!txCertificate.SignSubject(SubjectPublicKey, SubjectSecretKey)) {
        throw JSONRPCError(RPC_BDAP_INVALID_SIGNATURE, "Error Subject signing.");
    }

    if (!txCertificate.CheckSubjectSignature(SubjectPublicKey)) {
        throw JSONRPCError(RPC_BDAP_INVALID_SIGNATURE, "Subject Signature invalid.");
    }

    //CKeyEd25519 privIssuerDHTKey;
    std::vector<unsigned char> IssuerSecretKey;
    std::vector<unsigned char> PublicKey;

    //At minimum, subject owns these keys
    //generate ed25519 key for certificate
    CPubKey pubWalletKey; //won't be needing this
    CharString vchCertificatePubKey;
    CKeyEd25519 privCertificateKey;
    if (!pwalletMain->GetKeysFromPool(pubWalletKey, vchCertificatePubKey, true))
        throw JSONRPCError(RPC_WALLET_KEYPOOL_RAN_OUT, "Error: Keypool ran out, please call keypoolrefill first");

    CKeyID vchCertificatePubKeyID = GetIdFromCharVector(vchCertificatePubKey);
    if (!pwalletMain->GetDHTKey(vchCertificatePubKeyID, privCertificateKey))
        throw std::runtime_error("BDAP_SEND_LINK_RPC_ERROR: Unable to retrieve DHT Key");

    txCertificate.PublicKey = privCertificateKey.GetPubKeyBytes();

    txCertificate.MonthsValid = nMonths;

    if (selfSign) { //Self Sign

        //Issuer = Subject
        txCertificate.Issuer = txCertificate.Subject;

        //Issuer Signs
        if (!txCertificate.SignIssuer(SubjectPublicKey, SubjectSecretKey)) {
            throw JSONRPCError(RPC_BDAP_INVALID_SIGNATURE, "Error Issuer signing.");
        }

        if (!txCertificate.CheckIssuerSignature(SubjectPublicKey)) {
            throw JSONRPCError(RPC_BDAP_INVALID_SIGNATURE, "Issuer Signature invalid.");
        }

    } //end Self Sign
    else {
        //Handle ISSUER [optional]
        std::string strIssuerFQDN = request.params[2].get_str() + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN;
        ToLowerCase(strIssuerFQDN);
        vchIssuerFQDN = vchFromString(strIssuerFQDN);
        // Check if name exists
        CDomainEntry issuerDomainEntry;
        if (!GetDomainEntry(vchIssuerFQDN, issuerDomainEntry))
            throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", strIssuerFQDN));

        txCertificate.Issuer = issuerDomainEntry.vchFullObjectPath();
    } //end ISSUER

    std::string strMessage;

    //Validate BDAP values
    if (!txCertificate.ValidateValues(strMessage))
        throw JSONRPCError(RPC_BDAP_CERTIFICATE_INVALID, strprintf("Invalid certificate transaction. %s", strMessage));

    // Create BDAP operation script
    // OP_BDAP_CERTIFICATE
    // BDAP_CERTIFICATE
    std::vector<unsigned char> vchMonths = vchFromString(std::to_string(nMonths));

    //Only send PubKeys of BDAP accounts
    CScript scriptPubKey;
    if (selfSign) { 
        scriptPubKey << CScript::EncodeOP_N(OP_BDAP_MODIFY) << CScript::EncodeOP_N(OP_BDAP_CERTIFICATE) 
                 << vchMonths << vchSubjectFQDN << SubjectPublicKey << vchSubjectFQDN << SubjectPublicKey << OP_2DROP << OP_2DROP << OP_2DROP << OP_DROP; 
    } else { //NEW CERTIFICATE REQUEST
        scriptPubKey << CScript::EncodeOP_N(OP_BDAP_NEW) << CScript::EncodeOP_N(OP_BDAP_CERTIFICATE) 
                 << vchMonths << vchSubjectFQDN << SubjectPublicKey << vchIssuerFQDN << OP_2DROP << OP_2DROP << OP_2DROP; 
    }

    CKeyID keyWalletID = privSubjectDHTKey.GetID();
    CDynamicAddress walletAddress = CDynamicAddress(keyWalletID);

    CScript scriptDestination;
    scriptDestination = GetScriptForDestination(walletAddress.Get());
    scriptPubKey += scriptDestination;

    // Create BDAP OP_RETURN script
    CharString data;
    txCertificate.Serialize(data);
    CScript scriptData;
    scriptData << OP_RETURN << data;

    // Get BDAP Fees
    BDAP::ObjectType bdapType = BDAP::ObjectType::BDAP_CERTIFICATE;
    CAmount monthlyFee, oneTimeFee, depositFee;

    if (selfSign) {
        if (!GetBDAPFees(OP_BDAP_MODIFY, OP_BDAP_CERTIFICATE, bdapType, nMonths, monthlyFee, oneTimeFee, depositFee))
            throw JSONRPCError(RPC_BDAP_FEE_UNKNOWN, strprintf("Error calculating BDAP fees."));
    }
    else {
        if (!GetBDAPFees(OP_BDAP_NEW, OP_BDAP_CERTIFICATE, bdapType, nMonths, monthlyFee, oneTimeFee, depositFee))
            throw JSONRPCError(RPC_BDAP_FEE_UNKNOWN, strprintf("Error calculating BDAP fees."));
    }

    CAmount curBalance = pwalletMain->GetBalance() + pwalletMain->GetBDAPDynamicAmount();
    if (monthlyFee + oneTimeFee + depositFee > curBalance)
        throw JSONRPCError(RPC_WALLET_INSUFFICIENT_FUNDS, strprintf("Insufficient funds for BDAP transaction. %s DYN required.", FormatMoney(monthlyFee + oneTimeFee + depositFee)));

    bool fUseInstantSend = false;
    // Send the transaction
    CWalletTx wtx;

    SendBDAPTransaction(scriptData, scriptPubKey, wtx, monthlyFee, oneTimeFee + depositFee, fUseInstantSend);

    if (selfSign) {
        txCertificate.txHashApprove = wtx.GetHash();
    }
    else {
        txCertificate.txHashRequest = wtx.GetHash();
    }

    UniValue oCertificateTransaction(UniValue::VOBJ);
    BuildCertificateJson(txCertificate, oCertificateTransaction);

    return oCertificateTransaction;
#else
    throw JSONRPCError(RPC_WALLET_ERROR, strprintf("New certificate transaction is not available when the wallet is disabled."));
#endif
} //NewCertificate


static UniValue ApproveCertificate(const JSONRPCRequest& request)
{
#ifdef ENABLE_WALLET
    if (request.fHelp || (request.params.size() < 2 || request.params.size() > 2))
        throw std::runtime_error(
            "certificate approve \"txid\" \n"
            "\nApprove an X.509 certificate request\n"
            "\nArguments:\n"
            "1. \"txid\"             (string, required)  Transaction ID of certificate to approve\n"
            "\nResult:\n"
            "{(json object)\n"
            " \"version\"                   (string, required)   Version \n"
            " \"signature_algorithm\"       (string, required)   Algorithm used to sign \n"
            " \"signature_hash_algorithm\"  (string, required)   Algorithm used to hash \n"
            " \"fingerprint\"               (string, required)   Fingerprint of certificate \n"
            " \"months_valid\"              (int, required)      How long certificate is valid \n"
            " \"subject\"                   (string, required)   BDAP account of subject \n"
            " \"subject_signature\"         (string, required)   Signature of subject \n"
            " \"issuer\"                    (string, required)   BDAP account of issuer \n"
            " \"public_key\"                (string, required)   Public Key of certificate \n"
            " \"signature_value\"           (string, optional)   Signature of approval \n"
            " \"approved\"                  (boolean, required)  Certificate approved \n"
            " \"serial_number\"             (string, required)   Unique serial number \n"
            " \"certificate_keyid\"         (string, required)   Key ID \n"
            " \"key_usage\"                 (string, required)   List of usages \n"
            " \"txid_request\"              (string, required)   Certificate request transaction id\n"
            " \"txid_approve\"              (string, optional)   Certificate approved transaction id  \n"
            " \"request_time\"              (int, required)      Time when request was made \n"
            " \"request_height\"            (int, required)      Block where request is stored \n"
            " \"valid_from\"                (int, optional)      Time when certificate is valid \n"
            " \"valid_until\"               (int, optional)      Time when certificate expires \n"
            " \"approve_height\"            (int, optional)      Block where approval is stored \n"
            "}\n"
            "\nExamples\n" +
           HelpExampleCli("certificate approve", "\"txid\" ") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("certificate approve", "\"txid\" "));

    EnsureWalletIsUnlocked();

    std::vector<unsigned char> vchTxId;
    std::string parameter1 = request.params[1].get_str();
    vchTxId = vchFromString(parameter1);

    CCertificate txCertificate;
    int32_t nMonths = 12; // certificates last a year?

    //Retrieve certificate from CertificateDB
    bool readCertificateState = false;
    readCertificateState = pCertificateDB->ReadCertificateTxId(vchTxId,txCertificate);
    if (!readCertificateState) {
        throw JSONRPCError(RPC_DATABASE_ERROR, "Unable to retrieve certificate from CertificateDB");
    }

    //Check if certificate already approved
    if (txCertificate.IsApproved())
        throw JSONRPCError(RPC_BDAP_ERROR, "Certificate already approved");

    std::vector<unsigned char> vchIssuer;
    std::vector<unsigned char> vchSubject;
    CDomainEntry issuerDomainEntry;
    CDomainEntry subjectDomainEntry;

    vchIssuer = txCertificate.Issuer;
    if (!GetDomainEntry(vchIssuer, issuerDomainEntry))
        throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", stringFromVch(vchIssuer)));

    //need subject BDAP for BDAP operation script
    vchSubject = txCertificate.Subject;
    if (!GetDomainEntry(vchSubject, subjectDomainEntry))
        throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", stringFromVch(vchSubject)));
    CharString vchSubjectPubKey = subjectDomainEntry.DHTPublicKey;

    //Get Issuer BDAP user Public Key
    CharString vchIssuerPubKey = issuerDomainEntry.DHTPublicKey;
    CKeyEd25519 privIssuerDHTKey;
    std::vector<unsigned char> IssuerSecretKey;
    std::vector<unsigned char> IssuerPublicKey;

    CKeyID vchIssuerPubKeyID = GetIdFromCharVector(vchIssuerPubKey);

    //Check if I'm the correct account to approve
    //look for issuer public key in the wallet
    if (!pwalletMain->HaveDHTKey(vchIssuerPubKeyID))
        throw std::runtime_error("BDAP_CERTIFICATE_APPROVE_RPC_ERROR: Issuer public key not found in wallet");

    //Get Issuer ed25519 key
    if (!pwalletMain->GetDHTKey(vchIssuerPubKeyID, privIssuerDHTKey))
        throw std::runtime_error("BDAP_CERTIFICATE_APPROVE_RPC_ERROR: Unable to retrieve DHT Key");

    IssuerSecretKey = privIssuerDHTKey.GetPrivKeyBytes();
    IssuerPublicKey = privIssuerDHTKey.GetPubKeyBytes();

    //Issuer Signs
    if (!txCertificate.SignIssuer(IssuerPublicKey, IssuerSecretKey)) {
        throw JSONRPCError(RPC_BDAP_INVALID_SIGNATURE, "Error Issuer signing.");
    }

    if (!txCertificate.CheckIssuerSignature(IssuerPublicKey)) {
        throw JSONRPCError(RPC_BDAP_INVALID_SIGNATURE, "Issuer Signature invalid.");
        }

    std::string strMessage;

    //Validate BDAP values
    if (!txCertificate.ValidateValues(strMessage))
        throw JSONRPCError(RPC_BDAP_CERTIFICATE_INVALID, strprintf("Invalid certificate transaction. %s", strMessage));

    // Create BDAP operation script
    // OP_BDAP_CERTIFICATE
    // BDAP_CERTIFICATE
    std::vector<unsigned char> vchMonths = vchFromString(std::to_string(nMonths));

    //Only send PubKeys of BDAP accounts
    CScript scriptPubKey;
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_MODIFY) << CScript::EncodeOP_N(OP_BDAP_CERTIFICATE) 
                << vchMonths << vchSubject << vchSubjectPubKey << vchIssuer << vchIssuerPubKey << OP_2DROP << OP_2DROP << OP_2DROP << OP_DROP; 

    CDynamicAddress walletAddress = subjectDomainEntry.GetWalletAddress();

    CScript scriptDestination;
    scriptDestination = GetScriptForDestination(walletAddress.Get());
    scriptPubKey += scriptDestination;

    // Create BDAP OP_RETURN script
    CharString data;
    txCertificate.Serialize(data);
    CScript scriptData;
    scriptData << OP_RETURN << data;

    // Get BDAP Fees
    BDAP::ObjectType bdapType = BDAP::ObjectType::BDAP_CERTIFICATE;
    CAmount monthlyFee, oneTimeFee, depositFee;

    if (!GetBDAPFees(OP_BDAP_MODIFY, OP_BDAP_CERTIFICATE, bdapType, nMonths, monthlyFee, oneTimeFee, depositFee))
        throw JSONRPCError(RPC_BDAP_FEE_UNKNOWN, strprintf("Error calculating BDAP fees."));

    CAmount curBalance = pwalletMain->GetBalance() + pwalletMain->GetBDAPDynamicAmount();
    if (monthlyFee + oneTimeFee + depositFee > curBalance)
        throw JSONRPCError(RPC_WALLET_INSUFFICIENT_FUNDS, strprintf("Insufficient funds for BDAP transaction. %s DYN required.", FormatMoney(monthlyFee + oneTimeFee + depositFee)));

    bool fUseInstantSend = false;
    // Send the transaction
    CWalletTx wtx;
    SendBDAPTransaction(scriptData, scriptPubKey, wtx, monthlyFee, oneTimeFee + depositFee, fUseInstantSend);

    txCertificate.txHashApprove = wtx.GetHash();

    UniValue oCertificateTransaction(UniValue::VOBJ);
    BuildCertificateJson(txCertificate, oCertificateTransaction);

    return oCertificateTransaction;
#else
    throw JSONRPCError(RPC_WALLET_ERROR, strprintf("Approve certificate transaction is not available when the wallet is disabled."));
#endif
} //ApproveCertificate

static UniValue ViewCertificate(const JSONRPCRequest& request)
{
    if (request.fHelp || (request.params.size() < 2 || request.params.size() > 4))
        throw std::runtime_error(
            "certificate view \"txid\" or \n"
            "certificate view (\"subject\") (\"issuer\") (\"pending\") \n"
            "\nView X.509 certificates from blockchain\n"
            "\nArguments:\n"
            "1. \"txid\"                    (string, required)   Transaction ID of certificate\n"
            "      or\n"
            "1. \"subject\"                 (string, optional)   BDAP account of subject\n"
            "2. \"issuer\"                  (string, optional)   BDAP account of issuer\n"
            "3. \"pending\"                 (boolean, optional)  retrieve pending only (default = false)\n"
            "\nResult:\n"
            "{(json object)\n"
            " \"version\"                   (string, required)   Version \n"
            " \"signature_algorithm\"       (string, required)   Algorithm used to sign \n"
            " \"signature_hash_algorithm\"  (string, required)   Algorithm used to hash \n"
            " \"fingerprint\"               (string, required)   Fingerprint of certificate \n"
            " \"months_valid\"              (int, required)      How long certificate is valid \n"
            " \"subject\"                   (string, required)   BDAP account of subject \n"
            " \"subject_signature\"         (string, required)   Signature of subject \n"
            " \"issuer\"                    (string, required)   BDAP account of issuer \n"
            " \"public_key\"                (string, required)   Public Key of certificate \n"
            " \"signature_value\"           (string, optional)   Signature of approval \n"
            " \"approved\"                  (boolean, required)  Certificate approved \n"
            " \"serial_number\"             (string, required)   Unique serial number \n"
            " \"certificate_keyid\"         (string, required)   Key ID \n"
            " \"key_usage\"                 (string, required)   List of usages \n"
            " \"txid_request\"              (string, required)   Certificate request transaction id\n"
            " \"txid_approve\"              (string, optional)   Certificate approved transaction id  \n"
            " \"request_time\"              (int, required)      Time when request was made \n"
            " \"request_height\"            (int, required)      Block where request is stored \n"
            " \"valid_from\"                (int, optional)      Time when certificate is valid \n"
            " \"valid_until\"               (int, optional)      Time when certificate expires \n"
            " \"approve_height\"            (int, optional)      Block where approval is stored \n"
            "}\n"
            "\nExamples\n" +
           HelpExampleCli("certificate view", "\"txid\" ") +
           "\nAs a JSON-RPC call\n" + 
           HelpExampleRpc("certificate view", "\"txid\" "));

    //txid only
    std::vector<unsigned char> vchTxId;
    std::string parameterTxId = request.params[1].get_str();
    vchTxId = vchFromString(parameterTxId);
    bool readCertificateState = false;

    CCertificate certificate;
    UniValue oCertificateTransaction(UniValue::VOBJ);

    readCertificateState = pCertificateDB->ReadCertificateTxId(vchTxId,certificate);
    if (readCertificateState) {
        BuildCertificateJson(certificate, oCertificateTransaction);
        return oCertificateTransaction;
    }

    //if can't find txid, use BDAP logic
    UniValue oCertificateLists(UniValue::VARR);
    UniValue oCertificateList(UniValue::VOBJ);

    bool subjectDetected = false;
    bool issuerDetected = false;
    bool getAll = true;

    std::string subject = "";
    std::string issuer = "";
    std::string pending = "";

    std::vector<unsigned char> vchSubjectFQDN;
    std::vector<unsigned char> vchIssuerFQDN;

    //Subject
    if (request.params.size() > 1) {
        subject = request.params[1].get_str();
        ToLowerCase(subject);
        if (subject.size() > 0) {
            subjectDetected = true;
            vchSubjectFQDN = vchFromString(subject + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN);

            CDomainEntry subjectDomainEntry;
            if (!GetDomainEntry(vchSubjectFQDN, subjectDomainEntry))
                throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", subject));
        }
    }

    //Issuer
    if (request.params.size() > 2) {
        issuer = request.params[2].get_str();
        ToLowerCase(issuer);
        if (issuer.size() > 0) {
            issuerDetected = true;
            vchIssuerFQDN = vchFromString(issuer + "@" + DEFAULT_PUBLIC_OU + "." + DEFAULT_PUBLIC_DOMAIN);

            CDomainEntry issuerDomainEntry;
            if (!GetDomainEntry(vchIssuerFQDN, issuerDomainEntry))
                throw JSONRPCError(RPC_BDAP_ACCOUNT_NOT_FOUND, strprintf("%s account not found.", issuer));
        }
    }

    //Pending
    if (request.params.size() > 3) {
        pending = request.params[3].get_str();
        ToLowerCase(pending);
        if (pending.size() > 0)
            if (pending == "true")
                getAll = false;
    }

    std::vector<CCertificate> vCertificates;
    if ((subjectDetected) && (issuerDetected)) {
        if (getAll) {
            pCertificateDB->ReadCertificateSubjectDNApprove(vchSubjectFQDN, vCertificates);
        }
        else {
            pCertificateDB->ReadCertificateSubjectDNRequest(vchSubjectFQDN, vCertificates, false);
        }

        for (const CCertificate& certificate : vCertificates) {
            UniValue oCertificateList(UniValue::VOBJ);
            if (certificate.Issuer == vchIssuerFQDN) {
                BuildCertificateJson(certificate, oCertificateList);
                oCertificateLists.push_back(oCertificateList);
            }
        };
    }
    else if (subjectDetected) {
        if (getAll) {
            pCertificateDB->ReadCertificateSubjectDNApprove(vchSubjectFQDN, vCertificates);
        }
        else {
            pCertificateDB->ReadCertificateSubjectDNRequest(vchSubjectFQDN, vCertificates, false);
        }

        for (const CCertificate& certificate : vCertificates) {
            UniValue oCertificateList(UniValue::VOBJ);

            BuildCertificateJson(certificate, oCertificateList);
            oCertificateLists.push_back(oCertificateList);
        };
    }
    else if (issuerDetected) {
        if (getAll) {
            pCertificateDB->ReadCertificateIssuerDNApprove(vchIssuerFQDN, vCertificates);
        }
        else {
            pCertificateDB->ReadCertificateIssuerDNRequest(vchIssuerFQDN, vCertificates, false);
        }

        for (const CCertificate& certificate : vCertificates) {
            UniValue oCertificateList(UniValue::VOBJ);

            BuildCertificateJson(certificate, oCertificateList);
            oCertificateLists.push_back(oCertificateList);
        };
    }

    return oCertificateLists;

} //ViewCertificate

UniValue certificate_rpc(const JSONRPCRequest& request) 
{
    std::string strCommand;
    if (request.params.size() >= 1) {
        strCommand = request.params[0].get_str();
        ToLowerCase(strCommand);
    }
    else {
        throw std::runtime_error(
            "certificate \"command\"...\n"
            "\nAvailable commands:\n"
            "  new                - Create new X.509 certificate\n"
            "  approve            - Approve an X.509 certificate\n"
            "  view               - View X.509 certificate(s)\n"
            "\nExamples:\n"
            + HelpExampleCli("certificate new", "\"owner\" (\"issuer\") ") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("certificate new", "\"owner\" (\"issuer\") "));
    }
    if (strCommand == "new" || strCommand == "approve" || strCommand == "view") {
        if (!sporkManager.IsSporkActive(SPORK_32_BDAP_V2))
            throw JSONRPCError(RPC_BDAP_SPORK_INACTIVE, strprintf("Can not use certificate functionality until the BDAP version 2 spork is active."));
    }
    if (strCommand == "new") {
        return NewCertificate(request);
    }
    else if (strCommand == "approve") {
        return ApproveCertificate(request);
    }
    else if (strCommand == "view") {
        return ViewCertificate(request);
    }
    else {
        throw JSONRPCError(RPC_METHOD_NOT_FOUND, strprintf("%s is an unknown BDAP certificate method command.", strCommand));
    }
    return NullUniValue;
}

static const CRPCCommand commands[] =
{ //  category              name                     actor (function)               okSafe argNames
  //  --------------------- ------------------------ -----------------------        ------ --------------------
    { "bdap",               "certificate",           &certificate_rpc,               true,  {"command", "param1", "param2", "param3"} },
};

void RegisterCertificateRPCCommands(CRPCTable &t)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        t.appendCommand(commands[vcidx].name, &commands[vcidx]);
}
