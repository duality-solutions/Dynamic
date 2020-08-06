// Copyright (c) 2020 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/x509certificate.h"
#include "bdap/utils.h"
#include "hash.h"
#include "script/script.h"
#include "streams.h"
#include "uint256.h"
#include "validation.h"

#include "dht/ed25519.h"
#include <libtorrent/ed25519.hpp>
#include <univalue.h>

#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <openssl/evp.h>

#include <openssl/ec.h>

#include <openssl/pem.h>
#include <openssl/conf.h>
#include <openssl/x509v3.h>
#ifndef OPENSSL_NO_ENGINE
#include <openssl/engine.h>
#endif

int add_ext(X509 *cert, int nid, char *value);
int add_ext_req(STACK_OF(X509_EXTENSION) *sk, int nid, char *value); 
bool vchPEMfromX509(X509 *x509, std::vector<unsigned char>& vchPEM);
bool vchPEMfromX509req(X509_REQ *x509, std::vector<unsigned char>& vchPEM);

void CX509Certificate::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsEntryX509Certificate(SER_NETWORK, PROTOCOL_VERSION);
    dsEntryX509Certificate << *this;
    vchData = std::vector<unsigned char>(dsEntryX509Certificate.begin(), dsEntryX509Certificate.end());
}

bool CX509Certificate::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsEntryX509Certificate(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsEntryX509Certificate >> *this;

        std::vector<unsigned char> vchEntryLinkData;
        Serialize(vchEntryLinkData);
        const uint256 &calculatedHash = Hash(vchEntryLinkData.begin(), vchEntryLinkData.end());
        const std::vector<unsigned char> &vchRandEntryLink = vchFromValue(calculatedHash.GetHex());
        if(vchRandEntryLink != vchHash)
        {
            SetNull();
            return false;
        }
    } catch (std::exception &e) {
        SetNull();
        return false;
    }
    return true;
}

bool CX509Certificate::UnserializeFromTx(const CTransactionRef& tx, const unsigned int& height) 
{
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nOut;
    if(!GetBDAPData(tx, vchData, vchHash, nOut))
    {
        SetNull();
        return false;
    }
    if(!UnserializeFromData(vchData, vchHash))
    {
        return false;
    }

    //Distinguish between Request and Approve
    int op1, op2;
    std::vector<std::vector<unsigned char> > vvchBDAPArgs;
    CScript scriptOp;
    if (GetBDAPOpScript(tx, scriptOp, vvchBDAPArgs, op1, op2)) {
        std::string errorMessage;
        std::string strOpType = GetBDAPOpTypeString(op1, op2);
        if (strOpType == "bdap_new_certificate") {
            txHashRequest = tx->GetHash();
            nHeightRequest = height;
        }
        else if (strOpType == "bdap_approve_certificate") {
            txHashApprove = tx->GetHash();
            nHeightApprove = height;
        }
        //TODO: bdap_revoke_certificate?
    }

    return true;
}

std::string CX509Certificate::GetPubKeyHex() const
{
    std::vector<unsigned char> certPubKey = PublicKey;
    
    return ToHex(&certPubKey[0], certPubKey.size());
}

std::string CX509Certificate::GetSubjectSignature() const
{
    std::vector<unsigned char> subjectSig = SubjectSignature;

    return EncodeBase64(&subjectSig[0], subjectSig.size());
}

std::string CX509Certificate::GetIssuerSignature() const
{
    std::vector<unsigned char> issuerSig = IssuerSignature;

    return EncodeBase64(&issuerSig[0], issuerSig.size());
}

uint256 CX509Certificate::GetHash() const
{
    CDataStream dsX509Certificate(SER_NETWORK, PROTOCOL_VERSION);
    dsX509Certificate << *this;
    return Hash(dsX509Certificate.begin(), dsX509Certificate.end());
}

uint256 CX509Certificate::GetSubjectHash() const
{
    CDataStream dsX509Certificate(SER_NETWORK, PROTOCOL_VERSION);
    dsX509Certificate << Subject << PublicKey << SerialNumber << PEM;
    return Hash(dsX509Certificate.begin(), dsX509Certificate.end());
}

uint256 CX509Certificate::GetIssuerHash() const
{
    CDataStream dsX509Certificate(SER_NETWORK, PROTOCOL_VERSION);
    dsX509Certificate << MonthsValid << Subject << SubjectSignature << Issuer << PublicKey << SerialNumber << PEM;
    return Hash(dsX509Certificate.begin(), dsX509Certificate.end());
}

bool CX509Certificate::SignSubject(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchPrivKey)
{
    std::vector<unsigned char> msg = vchFromString(GetSubjectHash().ToString());
    std::vector<unsigned char> sig(64);

    libtorrent::ed25519_sign(&sig[0], &msg[0], msg.size(), &vchPubKey[0], &vchPrivKey[0]);
    SubjectSignature = sig;

    return true;
}

bool CX509Certificate::SignIssuer(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchPrivKey)
{
    std::vector<unsigned char> msg = vchFromString(GetIssuerHash().ToString());
    std::vector<unsigned char> sig(64);

    libtorrent::ed25519_sign(&sig[0], &msg[0], msg.size(), &vchPubKey[0], &vchPrivKey[0]);
    IssuerSignature = sig;

    return true;
}

bool CX509Certificate::CheckSubjectSignature(const std::vector<unsigned char>& vchPubKey) const
{
    std::vector<unsigned char> msg = vchFromString(GetSubjectHash().ToString());

    if (!libtorrent::ed25519_verify(&SubjectSignature[0], &msg[0], msg.size(), &vchPubKey[0])) {
        return false;
    }

    return true;
}

bool CX509Certificate::CheckIssuerSignature(const std::vector<unsigned char>& vchPubKey) const
{
    std::vector<unsigned char> msg = vchFromString(GetIssuerHash().ToString());

    if (!libtorrent::ed25519_verify(&IssuerSignature[0], &msg[0], msg.size(), &vchPubKey[0])) {
        return false;
    }

    return true;
}

int add_ext(X509 *cert, int nid, char *value)
{
    X509_EXTENSION *ex;
    X509V3_CTX ctx;
    /* This sets the 'context' of the extensions. */
    /* No configuration database */
    X509V3_set_ctx_nodb(&ctx);
    /* Issuer and subject certs: both the target since it is self signed,
        * no request and no CRL
        */
    X509V3_set_ctx(&ctx, cert, cert, NULL, NULL, 0);
    ex = X509V3_EXT_conf_nid(NULL, &ctx, nid, value);
    if (!ex)
        return 0;

    X509_add_ext(cert,ex,-1);
    X509_EXTENSION_free(ex);
    return 1;
}

int add_ext_req(STACK_OF(X509_EXTENSION) *sk, int nid, char *value) 
{
    X509_EXTENSION *ex;
    ex = X509V3_EXT_conf_nid(NULL, NULL, nid, value);
    if (!ex)
        return 0;
    sk_X509_EXTENSION_push(sk, ex);
    return 1;
}


bool vchPEMfromX509(X509 *x509, std::vector<unsigned char>& vchPEM)
{
    int rc = 0;
    unsigned long err = 0;

    std::unique_ptr<BIO, decltype(&::BIO_free)> bio(BIO_new(BIO_s_mem()), ::BIO_free);

    rc = PEM_write_bio_X509(bio.get(), x509);
    err = ERR_get_error();

    if (rc != 1)
    {
        return false;
    }

    BUF_MEM *mem = NULL;
    BIO_get_mem_ptr(bio.get(), &mem);
    err = ERR_get_error();

    if (!mem || !mem->data || !mem->length)
    {
        return false;
    }

    std::string pem(mem->data, mem->length);
    vchPEM = vchFromString(pem);
    return true;
}

bool vchPEMfromX509req(X509_REQ *x509, std::vector<unsigned char>& vchPEM)
{
    int rc = 0;
    unsigned long err = 0;

    std::unique_ptr<BIO, decltype(&::BIO_free)> bio(BIO_new(BIO_s_mem()), ::BIO_free);

    rc = PEM_write_bio_X509_REQ(bio.get(), x509);
    err = ERR_get_error();

    if (rc != 1)
    {
        return false;
    }

    BUF_MEM *mem = NULL;
    BIO_get_mem_ptr(bio.get(), &mem);
    err = ERR_get_error();

    if (!mem || !mem->data || !mem->length)
    {
        return false;
    }

    std::string pem(mem->data, mem->length);
    vchPEM = vchFromString(pem);
    return true;
}

bool CX509Certificate::X509RequestSign(const std::vector<unsigned char>& vchSubjectPrivKey)  //Pass PrivKeyBytes
{
    X509_REQ *certificate;
    X509_NAME *subjectName=NULL;
	EVP_PKEY* pubkeyEd25519;
	EVP_PKEY* privkeyEd25519;

    STACK_OF(X509_EXTENSION) *exts = NULL;

	pubkeyEd25519=EVP_PKEY_new();
    privkeyEd25519=EVP_PKEY_new();

	pubkeyEd25519 = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, NULL, &PublicKey[0], 32);
	privkeyEd25519 = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, NULL, &vchSubjectPrivKey[0], 32);

    if ((certificate=X509_REQ_new()) == NULL)
        return false;

    X509_REQ_set_version(certificate,2);
    X509_REQ_set_pubkey(certificate,pubkeyEd25519);

    subjectName=X509_REQ_get_subject_name(certificate);

    X509_NAME_add_entry_by_txt(subjectName, "C",  MBSTRING_ASC,
                            (unsigned char *)"US", -1, -1, 0);
    X509_NAME_add_entry_by_txt(subjectName, "O",  MBSTRING_ASC,
                            (unsigned char *)DEFAULT_ORGANIZATION_NAME.c_str(), -1, -1, 0);
    X509_NAME_add_entry_by_txt(subjectName, "CN", MBSTRING_ASC,
                            (unsigned char *)(stringFromVch(Subject).c_str()), -1, -1, 0);

	exts = sk_X509_EXTENSION_new_null();

    char* keyUsage = strdup("critical,keyCertSign,cRLSign");

	add_ext_req(exts, NID_key_usage, keyUsage);

    X509_REQ_add_extensions(certificate, exts);

    if (!X509_REQ_sign(certificate,privkeyEd25519,EVP_md_null()))
        return false;

    std::vector<unsigned char> vchPEM;
    if (!vchPEMfromX509req(certificate,vchPEM))
        return false;

    PEM = vchPEM; //Store PEM in certificate object

    sk_X509_EXTENSION_pop_free(exts, X509_EXTENSION_free);
    EVP_PKEY_free(pubkeyEd25519);
    EVP_PKEY_free(privkeyEd25519);
    X509_REQ_free(certificate);

    return true;

} //X509RequestSign


bool CX509Certificate::X509SelfSign(const std::vector<unsigned char>& vchSubjectPrivKey)  //Pass PrivKeyBytes
{
    X509 *certificate;
    X509_NAME *subjectName=NULL;
	EVP_PKEY* pubkeyEd25519;
	EVP_PKEY* privkeyEd25519;

	pubkeyEd25519=EVP_PKEY_new();
    privkeyEd25519=EVP_PKEY_new();

	pubkeyEd25519 = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, NULL, &PublicKey[0], 32);
	privkeyEd25519 = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, NULL, &vchSubjectPrivKey[0], 32);

    if ((certificate=X509_new()) == NULL)
        return false;

    X509_set_version(certificate,2);
    ASN1_INTEGER_set(X509_get_serialNumber(certificate), SerialNumber);
    X509_gmtime_adj(X509_get_notBefore(certificate),(long)0);
    X509_gmtime_adj(X509_get_notAfter(certificate),(long)AddMonthsToBlockTime(0,MonthsValid));
    X509_set_pubkey(certificate,pubkeyEd25519);

    subjectName=X509_get_subject_name(certificate);

    X509_NAME_add_entry_by_txt(subjectName, "C",  MBSTRING_ASC,
                            (unsigned char *)"US", -1, -1, 0);
    X509_NAME_add_entry_by_txt(subjectName, "O",  MBSTRING_ASC,
                            (unsigned char *)DEFAULT_ORGANIZATION_NAME.c_str(), -1, -1, 0);
    X509_NAME_add_entry_by_txt(subjectName, "CN", MBSTRING_ASC,
                            (unsigned char *)(stringFromVch(Subject).c_str()), -1, -1, 0);

    //self signed so subject=issuer
    X509_set_issuer_name(certificate,subjectName);

    char* basicConstraints = strdup("critical,CA:TRUE"); 
    char* keyUsage = strdup("critical,keyCertSign,cRLSign");
    char* keyIdentifier = strdup("hash");

    add_ext(certificate, NID_basic_constraints, basicConstraints);
    add_ext(certificate, NID_key_usage, keyUsage);
    add_ext(certificate, NID_subject_key_identifier, keyIdentifier);

    if (!X509_sign(certificate,privkeyEd25519,EVP_md_null()))
        return false;

    std::vector<unsigned char> vchPEM;
    if (!vchPEMfromX509(certificate,vchPEM))
        return false;

    PEM = vchPEM; //Store PEM in certificate object

    EVP_PKEY_free(pubkeyEd25519);
    EVP_PKEY_free(privkeyEd25519);
    X509_free(certificate);

    return true;

} //X509SelfSign

bool CX509Certificate::X509ApproveSign(const std::vector<unsigned char>& vchIssuerPrivKey)  //Pass PrivKeyBytes
{
    X509 *certificate;
    X509_NAME *subjectName=NULL;
    X509_NAME *issuerName=NULL;
	EVP_PKEY* pubkeyEd25519;
	EVP_PKEY* privkeyEd25519;

	pubkeyEd25519=EVP_PKEY_new();
    privkeyEd25519=EVP_PKEY_new();

	pubkeyEd25519 = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, NULL, &PublicKey[0], 32);
	privkeyEd25519 = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, NULL, &vchIssuerPrivKey[0], 32);

    if ((certificate=X509_new()) == NULL)
        return false;

    X509_set_version(certificate,2);
    ASN1_INTEGER_set(X509_get_serialNumber(certificate), SerialNumber);
    X509_gmtime_adj(X509_get_notBefore(certificate),(long)0);
    X509_gmtime_adj(X509_get_notAfter(certificate),(long)AddMonthsToBlockTime(0,MonthsValid));
    X509_set_pubkey(certificate,pubkeyEd25519);

    subjectName=X509_get_subject_name(certificate);
    issuerName=X509_get_issuer_name(certificate);

    X509_NAME_add_entry_by_txt(subjectName, "C",  MBSTRING_ASC,
                            (unsigned char *)"US", -1, -1, 0);
    X509_NAME_add_entry_by_txt(subjectName, "O",  MBSTRING_ASC,
                            (unsigned char *)DEFAULT_ORGANIZATION_NAME.c_str(), -1, -1, 0);
    X509_NAME_add_entry_by_txt(subjectName, "CN", MBSTRING_ASC,
                            (unsigned char *)(stringFromVch(Subject).c_str()), -1, -1, 0);

    X509_NAME_add_entry_by_txt(issuerName, "C",  MBSTRING_ASC,
                            (unsigned char *)"US", -1, -1, 0);
    X509_NAME_add_entry_by_txt(issuerName, "O",  MBSTRING_ASC,
                            (unsigned char *)DEFAULT_ORGANIZATION_NAME.c_str(), -1, -1, 0);
    X509_NAME_add_entry_by_txt(issuerName, "CN", MBSTRING_ASC,
                            (unsigned char *)(stringFromVch(Issuer).c_str()), -1, -1, 0);

    //self signed so subject=issuer
    X509_set_issuer_name(certificate,issuerName);

    char* basicConstraints = strdup("critical,CA:TRUE"); 
    char* keyUsage = strdup("critical,keyCertSign,cRLSign");
    char* keyIdentifier = strdup("hash");

    add_ext(certificate, NID_basic_constraints, basicConstraints);
    add_ext(certificate, NID_key_usage, keyUsage);
    add_ext(certificate, NID_subject_key_identifier, keyIdentifier);

    if (!X509_sign(certificate,privkeyEd25519,EVP_md_null()))
        return false;

    std::vector<unsigned char> vchPEM;
    if (!vchPEMfromX509(certificate,vchPEM))
        return false;

    PEM = vchPEM; //Store PEM in certificate object

    EVP_PKEY_free(pubkeyEd25519);
    EVP_PKEY_free(privkeyEd25519);
    X509_free(certificate);

    return true;

} //X509ApproveSign


bool CX509Certificate::ValidateValues(std::string& errorMessage) const
{
    //Check that Subject exists
    if (Subject.size() == 0)
    {
        errorMessage = "Subject cannot be empty.";
        return false;
    }

    //Check that Subject signature exists
    if (SubjectSignature.size() == 0)
    {
        errorMessage = "Subject Signature cannot be empty.";
        return false;
    }

    //Check that PublicKey exists
    if (PublicKey.size() == 0)
    {
        errorMessage = "Public Key cannot be empty.";
        return false;
    }

    //Check that PEM exists
    if (PEM.size() == 0)
    {
        errorMessage = "PEM cannot be empty.";
        return false;
    }

    if (PEM.size() > MAX_CERTIFICATE_PEM_LENGTH)
    {
        errorMessage = "Invalid PEM size. Can not have more than " + std::to_string(MAX_CERTIFICATE_PEM_LENGTH) + " characters.";
        return false;
    }

    // check subject owner path
    if (Subject.size() > MAX_OBJECT_FULL_PATH_LENGTH) 
    {
        errorMessage = "Invalid Subject full path name. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }

    // check SubjectSignature
    if (SubjectSignature.size() > MAX_CERTIFICATE_SIGNATURE_LENGTH) 
    {
        errorMessage = "Invalid SubjectSignature. Can not have more than " + std::to_string(MAX_CERTIFICATE_SIGNATURE_LENGTH) + " characters.";
        return false;
    }

    // check issuer owner path
    if (Issuer.size() > MAX_OBJECT_FULL_PATH_LENGTH) 
    {
        errorMessage = "Invalid Issuer full path name. Can not have more than " + std::to_string(MAX_OBJECT_FULL_PATH_LENGTH) + " characters.";
        return false;
    }

    // check PublicKey
    if (PublicKey.size() > MAX_CERTIFICATE_KEY_LENGTH) 
    {
        errorMessage = "Invalid PublicKey. Can not have more than " + std::to_string(MAX_CERTIFICATE_KEY_LENGTH) + " characters.";
        return false;
    }

    // check IssuerSignature
    if (IssuerSignature.size() > MAX_CERTIFICATE_SIGNATURE_LENGTH) 
    {
        errorMessage = "Invalid IssuerSignature. Can not have more than " + std::to_string(MAX_CERTIFICATE_SIGNATURE_LENGTH) + " characters.";
        return false;
    }

    return true;
}

std::string CX509Certificate::ToString() const
{
     return strprintf(
        "CCertificate(\n"
        "    nVersion                 = %d\n"
        "    Months Valid             = %d\n"
        "    Subject                  = %s\n"
        "    Subject Signature        = %s\n"
        "    PublicKey                = %s\n"
        "    Issuer                   = %s\n"
        "    Issuer Signature         = %s\n"
        "    Serial Number            = %d\n"
        "    Self Signed              = %s\n"
        "    Approved                 = %s\n"
        "    Request TxId             = %s\n"
        "    Approve TxId             = %s\n"
        ")\n",
        nVersion,
        MonthsValid,
        stringFromVch(Subject),
        GetSubjectSignature(),
        GetPubKeyHex(),
        stringFromVch(Issuer),
        GetIssuerSignature(),
        SerialNumber,
        SelfSignedX509Certificate() ? "True" : "False",
        IsApproved() ? "True" : "False",
        txHashRequest.GetHex(),
        txHashApprove.GetHex()
        );
}


bool BuildX509CertificateJson(const CX509Certificate& certificate, UniValue& oCertificate)
{
    int64_t nTime = 0;
    int64_t nApproveTime = 0;

    oCertificate.push_back(Pair("version", std::to_string(certificate.nVersion)));

    oCertificate.push_back(Pair("months_valid", std::to_string(certificate.MonthsValid)));
    oCertificate.push_back(Pair("subject", stringFromVch(certificate.Subject)));
    oCertificate.push_back(Pair("subject_signature", certificate.GetSubjectSignature()));
    oCertificate.push_back(Pair("public_key", certificate.GetPubKeyHex()));
    oCertificate.push_back(Pair("issuer", stringFromVch(certificate.Issuer)));
    oCertificate.push_back(Pair("issuer_signature", certificate.GetIssuerSignature()));
    oCertificate.push_back(Pair("approved", certificate.IsApproved() ? "True" : "False"));
    oCertificate.push_back(Pair("serial_number", std::to_string(certificate.SerialNumber)));

    oCertificate.push_back(Pair("pem", stringFromVch(certificate.PEM)));

    oCertificate.push_back(Pair("txid_request", certificate.txHashRequest.GetHex()));
    oCertificate.push_back(Pair("txid_approve", certificate.txHashApprove.GetHex()));
    if ((unsigned int)chainActive.Height() >= certificate.nHeightRequest) {
        CBlockIndex *pindex = chainActive[certificate.nHeightRequest];
        if (pindex) {
            nTime = pindex->GetBlockTime();
        }
    }
    oCertificate.push_back(Pair("request_time", nTime));
    oCertificate.push_back(Pair("request_height", std::to_string(certificate.nHeightRequest)));

    if (certificate.nHeightApprove != 0) {
        if ((unsigned int)chainActive.Height() >= certificate.nHeightApprove) {
            CBlockIndex *pindex = chainActive[certificate.nHeightApprove];
            if (pindex) {
                nApproveTime = pindex->GetBlockTime();
            }
        }
        oCertificate.push_back(Pair("valid_from", nApproveTime));
        oCertificate.push_back(Pair("valid_until", AddMonthsToBlockTime(nApproveTime,certificate.MonthsValid)));
        oCertificate.push_back(Pair("approve_height", std::to_string(certificate.nHeightApprove)));
    }
    
    return true;
}