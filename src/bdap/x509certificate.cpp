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

#include "util.h"

#include "dht/ed25519.h"
#include <libtorrent/ed25519.hpp>
#include <univalue.h>

#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <openssl/evp.h>

#include <openssl/asn1.h>

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
        else if (strOpType == "bdap_rootca_certificate") {
            txHashRootCA = tx->GetHash();
            nHeightRootCA = height;
        }
        //TODO: bdap_revoke_certificate?
    }

    return true;
}

std::string CX509Certificate::GetPubKeyHex() const
{
    std::vector<unsigned char> certPubKey = SubjectPublicKey;
    
    return ToHex(&certPubKey[0], certPubKey.size());
}

std::string CX509Certificate::GetPubKeyBase64() const
{
    std::vector<unsigned char> certPubKey = SubjectPublicKey;
    
    return EncodeBase64(&certPubKey[0], certPubKey.size());
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
    dsX509Certificate << Subject << SubjectPublicKey << SerialNumber << PEM;
    return Hash(dsX509Certificate.begin(), dsX509Certificate.end());
}

uint256 CX509Certificate::GetIssuerHash() const
{
    CDataStream dsX509Certificate(SER_NETWORK, PROTOCOL_VERSION);
    dsX509Certificate << MonthsValid << Subject << SubjectSignature << Issuer << SubjectPublicKey << SerialNumber << PEM;
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
    OpenSSL_add_all_algorithms();
    OpenSSL_add_all_digests();
    ERR_load_BIO_strings();
    ERR_load_crypto_strings();



    X509_REQ *certificate;
    X509_NAME *subjectName=NULL;
	EVP_PKEY* pubkeyEd25519;
	EVP_PKEY* privkeyEd25519;

    STACK_OF(X509_EXTENSION) *exts = NULL;

	pubkeyEd25519=EVP_PKEY_new();
    privkeyEd25519=EVP_PKEY_new();

	pubkeyEd25519 = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, NULL, &SubjectPublicKey[0], 32);
	privkeyEd25519 = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, NULL, &vchSubjectPrivKey[0], 32);

    if ((certificate=X509_REQ_new()) == NULL)
        return false;

    X509_REQ_set_version(certificate,2);
    X509_REQ_set_pubkey(certificate,privkeyEd25519);

    subjectName=X509_REQ_get_subject_name(certificate);

    X509_NAME_add_entry_by_txt(subjectName, "C",  MBSTRING_ASC,
                            (unsigned char *)"US", -1, -1, 0);
    X509_NAME_add_entry_by_txt(subjectName, "O",  MBSTRING_ASC,
                            (unsigned char *)DEFAULT_ORGANIZATION_NAME.c_str(), -1, -1, 0);
    X509_NAME_add_entry_by_txt(subjectName, "CN", MBSTRING_ASC,
                            (unsigned char *)(stringFromVch(Subject).c_str()), -1, -1, 0);

	exts = sk_X509_EXTENSION_new_null();

    char* keyUsage = strdup("digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment"); //critical,keyCertSign,cRLSign

	add_ext_req(exts, NID_key_usage, keyUsage);

    X509_REQ_add_extensions(certificate, exts);

    if (!X509_REQ_sign(certificate,privkeyEd25519,EVP_md_null()))
        return false;

EVP_PKEY *pkey = X509_REQ_get_pubkey(certificate);
int signVerify = X509_REQ_verify(certificate,pkey);
LogPrintf("DEBUGGER %s - signVerify: [%d]\n",__func__,signVerify);


//  void X509_get0_signature(const ASN1_BIT_STRING **psig,
//                           const X509_ALGOR **palg,
//                           const X509 *x);

//digestverify test begin--------------------------------------------------------------------------------------------------------

// const ASN1_BIT_STRING *psig;
// const X509_ALGOR *palg;

// X509_REQ_get0_signature(certificate, (&psig), (&palg));

// // const ASN1_ITEM *it = ASN1_ITEM_rptr(X509_REQ_INFO);
// // unsigned char *buf_in = NULL;
// // size_t inl = 0;
// //inl = ASN1_item_i2d(certificate->req_info, &buf_in, it);

// unsigned char* tbs = NULL;

// i2d_re_X509_REQ_tbs(certificate, &tbs);
// int tbs_size = i2d_re_X509_REQ_tbs(certificate, &tbs);

// int outResult = 0;



// LogPrintf("DEBUGGER %s - made it here 1\n",__func__);
// EVP_MD_CTX* ctx = EVP_MD_CTX_new();
// LogPrintf("DEBUGGER %s - made it here 2\n",__func__);
// EVP_PKEY_CTX* ppctx = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, NULL);
// LogPrintf("DEBUGGER %s - made it here 3\n",__func__);
// int result = EVP_DigestVerifyInit(ctx, &ppctx, NULL, NULL, pkey);
// if (result == 1) {
// LogPrintf("DEBUGGER %s - made it here 4a, tbs_size: [%d]\n",__func__, tbs_size);
// LogPrintf("DEBUGGER %s - made it here 4b, psig length: [%d]\n",__func__,(psig)->length);
// LogPrintf("DEBUGGER %s - made it here 4c, psig null: [%s]\n",__func__,psig->data ? "True" : "False" );
// LogPrintf("DEBUGGER %s - made it here 4d, psig data: [%s]\n",__func__,EncodeBase64(psig->data,(psig)->length));

// // result = EVP_DigestVerifyUpdate(ctx, (tbs), tbs_size);
// // LogPrintf("DEBUGGER %s - EVP_DigestVerifyUpdate result: [%d]\n",__func__,result);
// //     if (result != 1)
// //         LogPrintf("DEBBGUGER %s - EVP_DigestVerifyUpdate geterror: [%s]\n",__func__,ERR_error_string(ERR_get_error(),NULL));


// //     int result3 = EVP_DigestVerifyFinal(ctx,(psig)->data, (psig)->length);
// //     LogPrintf("DEBUGGER %s - result3: [%d]\n",__func__,result3);
// //     if (result3 != 1)
// //         LogPrintf("DEBBGUGER %s - EVP_DigestVerifyFinal geterror: [%s]\n",__func__,ERR_error_string(ERR_get_error(),NULL));

//     //const unsigned char* sigret = NULL;
//     //result = EVP_DigestVerify(ctx, sigret, 64, tbs, 64);
//     result = EVP_DigestVerify(ctx, (psig)->data, (psig)->length, (tbs), tbs_size);
//     LogPrintf("DEBUGGER %s - EVP_DigestVerify result: [%d]\n",__func__,result);

//     if (result == 1) {
//         // passed signature verification
//         outResult = 1;
//     } else {
//         // failed signature verification
//         LogPrintf("DEBBGUGER %s - geterror: [%s]\n",__func__,ERR_error_string(ERR_get_error(),NULL));

//     }
// }

// LogPrintf("DEBUGGER %s - made it here 5, outResult: [%d]\n",__func__,outResult);

//digestverify test end--------------------------------------------------------------------------------------------------------


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
    OpenSSL_add_all_algorithms();
    OpenSSL_add_all_digests();
    ERR_load_BIO_strings();
    ERR_load_crypto_strings();

    X509 *certificate;
    X509_NAME *subjectName=NULL;
	EVP_PKEY* pubkeyEd25519;
	EVP_PKEY* privkeyEd25519;

	pubkeyEd25519=EVP_PKEY_new();
    privkeyEd25519=EVP_PKEY_new();

	pubkeyEd25519 = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, NULL, &SubjectPublicKey[0], 32);
	privkeyEd25519 = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, NULL, &vchSubjectPrivKey[0], 32);

    if ((certificate=X509_new()) == NULL)
        return false;

    X509_set_version(certificate,2);
    ASN1_INTEGER_set(X509_get_serialNumber(certificate), SerialNumber);
    X509_gmtime_adj(X509_get_notBefore(certificate),(long)0);
    X509_gmtime_adj(X509_get_notAfter(certificate),(long)AddMonthsToBlockTime(0,MonthsValid));
    X509_set_pubkey(certificate,privkeyEd25519);

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
    char* authKeyIdentifier = strdup("keyid,issuer");

    add_ext(certificate, NID_basic_constraints, basicConstraints);
    add_ext(certificate, NID_key_usage, keyUsage);
    add_ext(certificate, NID_subject_key_identifier, keyIdentifier);
    add_ext(certificate, NID_authority_key_identifier, authKeyIdentifier);
     int resultverifyx509;

     EVP_PKEY *pkey = X509_get_pubkey(certificate);


    // result = X509_verify(certificate, pkey);


    // LogPrintf("DEBUGGER %s - result before sign: [%d]\n",__func__,result);
    // LogPrintf("DEBBGUGER %s - geterror: [%s]\n",__func__,ERR_error_string(ERR_get_error(),NULL));


    if (!X509_sign(certificate,privkeyEd25519,EVP_md_null()))
        return false;

    resultverifyx509 = X509_verify(certificate, pkey);

    LogPrintf("DEBUGGER %s - result after sign: [%d]\n",__func__,resultverifyx509);
    LogPrintf("DEBBGUGER %s - geterror: [%s]\n",__func__,ERR_error_string(ERR_get_error(),NULL));

//digestverify test begin--------------------------------------------------------------------------------------------------------

// const ASN1_BIT_STRING *psig;
// const X509_ALGOR *palg;

// X509_get0_signature((&psig), (&palg), certificate);

// // const ASN1_ITEM *it = ASN1_ITEM_rptr(X509_REQ_INFO);
// // unsigned char *buf_in = NULL;
// // size_t inl = 0;
// //inl = ASN1_item_i2d(certificate->req_info, &buf_in, it);

// unsigned char* tbs = NULL;

// i2d_re_X509_tbs(certificate, &tbs);
// int tbs_size = i2d_re_X509_tbs(certificate, &tbs);

// int outResult = 0;

// //EVP_PKEY *pkey = X509_get_pubkey(certificate);


// LogPrintf("DEBUGGER %s - made it here 1\n",__func__);
// EVP_MD_CTX* ctx = EVP_MD_CTX_new();
// LogPrintf("DEBUGGER %s - made it here 2\n",__func__);
// EVP_PKEY_CTX* ppctx = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, NULL);
// LogPrintf("DEBUGGER %s - made it here 3\n",__func__);
// int result = EVP_DigestVerifyInit(ctx, &ppctx, NULL, NULL, pkey);
// if (result == 1) {
// LogPrintf("DEBUGGER %s - made it here 4a, tbs_size: [%d]\n",__func__, tbs_size);
// LogPrintf("DEBUGGER %s - made it here 4b, psig length: [%d]\n",__func__,(psig)->length);
// LogPrintf("DEBUGGER %s - made it here 4c, psig null: [%s]\n",__func__,psig->data ? "True" : "False" );
// LogPrintf("DEBUGGER %s - made it here 4d, psig data: [%s]\n",__func__,EncodeBase64(psig->data,(psig)->length));

// // result = EVP_DigestVerifyUpdate(ctx, (tbs), tbs_size);
// // LogPrintf("DEBUGGER %s - EVP_DigestVerifyUpdate result: [%d]\n",__func__,result);
// //     if (result != 1)
// //         LogPrintf("DEBBGUGER %s - EVP_DigestVerifyUpdate geterror: [%s]\n",__func__,ERR_error_string(ERR_get_error(),NULL));


// //     int result3 = EVP_DigestVerifyFinal(ctx,(psig)->data, (psig)->length);
// //     LogPrintf("DEBUGGER %s - result3: [%d]\n",__func__,result3);
// //     if (result3 != 1)
// //         LogPrintf("DEBBGUGER %s - EVP_DigestVerifyFinal geterror: [%s]\n",__func__,ERR_error_string(ERR_get_error(),NULL));

//     //const unsigned char* sigret = NULL;
//     //result = EVP_DigestVerify(ctx, sigret, 64, tbs, 64);
//     result = EVP_DigestVerify(ctx, (psig)->data, (psig)->length, (tbs), tbs_size);
//     LogPrintf("DEBUGGER %s - EVP_DigestVerify result: [%d]\n",__func__,result);

//     if (result == 1) {
//         // passed signature verification
//         outResult = 1;
//     } else {
//         // failed signature verification
//         LogPrintf("DEBBGUGER %s - geterror: [%s]\n",__func__,ERR_error_string(ERR_get_error(),NULL));

//     }
// }

// LogPrintf("DEBUGGER %s - made it here 5, outResult: [%d]\n",__func__,outResult);
//digestverify test end--------------------------------------------------------------------------------------------------------




    std::vector<unsigned char> vchPEM;
    if (!vchPEMfromX509(certificate,vchPEM))
        return false;

    PEM = vchPEM; //Store PEM in certificate object

    EVP_PKEY_free(pubkeyEd25519);
    EVP_PKEY_free(privkeyEd25519);
    //EVP_PKEY_free(pkey);
    X509_free(certificate);

    return true;

} //X509SelfSign


bool CX509Certificate::X509TestApproveSign(const std::vector<unsigned char>& vchSubjectPrivKey, const std::vector<unsigned char>& vchIssuerPrivKey)  //Pass PrivKeyBytes
{
    //create root CA certificate w/issuer info
    OpenSSL_add_all_algorithms();
    OpenSSL_add_all_digests();
    ERR_load_BIO_strings();
    ERR_load_crypto_strings();

    X509 *certificateCA;
    X509_NAME *subjectName=NULL;
    X509_NAME *issuerName=NULL;
	EVP_PKEY* subjectprivkeyEd25519;
	EVP_PKEY* issuerprivkeyEd25519;

	subjectprivkeyEd25519=EVP_PKEY_new();
    issuerprivkeyEd25519=EVP_PKEY_new();

	subjectprivkeyEd25519 = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, NULL, &vchSubjectPrivKey[0], 32);
	issuerprivkeyEd25519 = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, NULL, &vchIssuerPrivKey[0], 32);

    if ((certificateCA=X509_new()) == NULL)
        return false;

    X509_set_version(certificateCA,2);
    ASN1_INTEGER_set(X509_get_serialNumber(certificateCA), SerialNumber);
    X509_gmtime_adj(X509_get_notBefore(certificateCA),(long)0);
    X509_gmtime_adj(X509_get_notAfter(certificateCA),(long)AddMonthsToBlockTime(0,MonthsValid));
    X509_set_pubkey(certificateCA,issuerprivkeyEd25519);

    subjectName=X509_get_subject_name(certificateCA);

    X509_NAME_add_entry_by_txt(subjectName, "C",  MBSTRING_ASC,
                            (unsigned char *)"US", -1, -1, 0);
    X509_NAME_add_entry_by_txt(subjectName, "O",  MBSTRING_ASC,
                            (unsigned char *)DEFAULT_ORGANIZATION_NAME.c_str(), -1, -1, 0);
    X509_NAME_add_entry_by_txt(subjectName, "CN", MBSTRING_ASC,
                            (unsigned char *)(stringFromVch(Issuer).c_str()), -1, -1, 0);

    //self signed so subject=issuer
    X509_set_issuer_name(certificateCA,subjectName);

    char* basicConstraints = strdup("critical,CA:TRUE"); 
    char* keyUsage = strdup("critical,keyCertSign,cRLSign");
    char* keyIdentifier = strdup("hash");
    char* authKeyIdentifier = strdup("keyid,issuer");

    add_ext(certificateCA, NID_basic_constraints, basicConstraints);
    add_ext(certificateCA, NID_key_usage, keyUsage);
    add_ext(certificateCA, NID_subject_key_identifier, keyIdentifier);
    add_ext(certificateCA, NID_authority_key_identifier, authKeyIdentifier);
     int resultverifyx509;

     EVP_PKEY *pkey = X509_get_pubkey(certificateCA);

    if (!X509_sign(certificateCA,issuerprivkeyEd25519,EVP_md_null()))
        return false;


    resultverifyx509 = X509_verify(certificateCA, pkey);

    LogPrintf("DEBUGGER %s - result after CA sign: [%d]\n",__func__,resultverifyx509);

    FILE * x509CAFile;
    x509CAFile = fopen("CA.pem", "wb");

    //PEM_write_PrivateKey(stdout,pkey,NULL,NULL,0,NULL, NULL);
    PEM_write_X509(x509CAFile,certificateCA);
    fclose(x509CAFile);


//request portion

    X509_REQ *certificateREQ;
    X509_NAME *subjectNameREQ=NULL;

    if ((certificateREQ=X509_REQ_new()) == NULL)
        return false;

    X509_REQ_set_version(certificateREQ,2);
    X509_REQ_set_pubkey(certificateREQ,subjectprivkeyEd25519);

    subjectNameREQ=X509_REQ_get_subject_name(certificateREQ);

    X509_NAME_add_entry_by_txt(subjectNameREQ, "C",  MBSTRING_ASC,
                            (unsigned char *)"US", -1, -1, 0);
    X509_NAME_add_entry_by_txt(subjectNameREQ, "O",  MBSTRING_ASC,
                            (unsigned char *)DEFAULT_ORGANIZATION_NAME.c_str(), -1, -1, 0);
    X509_NAME_add_entry_by_txt(subjectNameREQ, "CN", MBSTRING_ASC,
                            (unsigned char *)(stringFromVch(Subject).c_str()), -1, -1, 0);


    STACK_OF(X509_EXTENSION) *exts = NULL;

	exts = sk_X509_EXTENSION_new_null();

    char* keyUsageREQ = strdup("digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment"); //critical,keyCertSign,cRLSign

	add_ext_req(exts, NID_key_usage, keyUsageREQ);

    X509_REQ_add_extensions(certificateREQ, exts);

     int resultverifyx509REQ;

     EVP_PKEY *pkeyREQ = X509_REQ_get_pubkey(certificateREQ);

    if (!X509_REQ_sign(certificateREQ,subjectprivkeyEd25519,EVP_md_null()))
        return false;

    resultverifyx509REQ = X509_REQ_verify(certificateREQ, pkeyREQ);

    LogPrintf("DEBUGGER %s - result after REQ sign: [%d]\n",__func__,resultverifyx509REQ);

    FILE * x509REQFile;
    x509REQFile = fopen("subject.csr", "wb");

    //PEM_write_PrivateKey(stdout,pkey,NULL,NULL,0,NULL, NULL);
    PEM_write_X509_REQ(x509REQFile,certificateREQ);
    fclose(x509REQFile);


//signed cert portion
    X509 *certificateCRT;

    if ((certificateCRT=X509_new()) == NULL)
        return false;

    X509_set_version(certificateCRT,2);
    ASN1_INTEGER_set(X509_get_serialNumber(certificateCRT), GetTimeMillis());
    X509_gmtime_adj(X509_get_notBefore(certificateCRT),(long)0);
    X509_gmtime_adj(X509_get_notAfter(certificateCRT),(long)AddMonthsToBlockTime(0,MonthsValid));
    X509_set_pubkey(certificateCRT,subjectprivkeyEd25519);

    subjectName=X509_get_subject_name(certificateCRT);
    issuerName=X509_get_issuer_name(certificateCRT);

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

    X509_set_issuer_name(certificateCRT,issuerName);

    //char* basicConstraintsCRT = strdup("critical,CA:FALSE"); 
    //char* keyUsageCRT = strdup("critical");
    //char* keyIdentifierCRT = strdup("hash");
    //char* authKeyIdentifierCRT = strdup("keyid:always"); //keyid,issuer

    //add_ext(certificateCRT, NID_basic_constraints, basicConstraintsCRT);
    //add_ext(certificateCRT, NID_key_usage, keyUsageCRT);
    //add_ext(certificateCRT, NID_subject_key_identifier, keyIdentifierCRT);
    //add_ext(certificateCRT, NID_authority_key_identifier, authKeyIdentifierCRT);

    X509_EXTENSION *ex;
    X509V3_CTX v3ctx;
    X509V3_set_ctx(&v3ctx, certificateCA, certificateCRT, 0, 0, 0); 

    ex = X509V3_EXT_conf_nid(0, &v3ctx, NID_basic_constraints, "critical,CA:FALSE");
    X509_add_ext(certificateCRT, ex, -1);

    ex = X509V3_EXT_conf_nid(0, &v3ctx, NID_key_usage, "digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment");
    X509_add_ext(certificateCRT, ex, -1);

    ex = X509V3_EXT_conf_nid(0, &v3ctx, NID_authority_key_identifier, "keyid:always");
    X509_add_ext(certificateCRT, ex, -1);
    X509_EXTENSION_free(ex);

     int resultverifyx509CRT;

     //EVP_PKEY *pkeyCRT = X509_get_pubkey(certificateCRT);

    if (!X509_sign(certificateCRT,issuerprivkeyEd25519,EVP_md_null()))
        return false;

    //resultverifyx509CRT = X509_verify(certificateCRT, pkeyCRT);

    X509_STORE_CTX *ctx2 = X509_STORE_CTX_new();
    X509_STORE *store = X509_STORE_new();
    X509_STORE_add_cert(store, certificateCA);
    X509_STORE_CTX_init(ctx2,store,certificateCRT, NULL);
    resultverifyx509CRT  = X509_verify_cert(ctx2);

    LogPrintf("DEBUGGER %s - result after CRT sign: [%d]\n",__func__,resultverifyx509CRT);

    FILE * x509CRTFile;
    x509CRTFile = fopen("subject.crt", "wb");

    //PEM_write_PrivateKey(stdout,pkey,NULL,NULL,0,NULL, NULL);
    PEM_write_X509(x509CRTFile,certificateCRT);
    fclose(x509CRTFile);


    X509_STORE_CTX_free(ctx2);
    X509_STORE_free(store);
    EVP_PKEY_free(subjectprivkeyEd25519);
    EVP_PKEY_free(issuerprivkeyEd25519);
    EVP_PKEY_free(pkey);
    //EVP_PKEY_free(pkeyCRT);
    EVP_PKEY_free(pkeyREQ);
    X509_free(certificateCA);
    X509_free(certificateCRT);
    X509_REQ_free(certificateREQ);

    return true;    




} //X509TestApproveSign



bool CX509Certificate::X509ApproveSign(const std::vector<unsigned char>& vchIssuerPrivKey)  //Pass PrivKeyBytes
{
    X509 *certificate;
    X509_NAME *subjectName=NULL;
    X509_NAME *issuerName=NULL;
	EVP_PKEY* pubkeyEd25519;
	EVP_PKEY* privkeyEd25519;

	pubkeyEd25519=EVP_PKEY_new();
    privkeyEd25519=EVP_PKEY_new();

	pubkeyEd25519 = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, NULL, &SubjectPublicKey[0], 32);
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

std::string CX509Certificate::GetPEMSubject() const {
  
    X509 *certRetrieve = NULL;
    BIO *certbio = NULL;

    std::string pem = stringFromVch(PEM);
    std::string outputString = "";

    certbio = BIO_new_mem_buf(pem.c_str(), -1);

    if (!(certRetrieve = PEM_read_bio_X509(certbio, NULL, NULL, NULL)))
        return "";

    char line[2000+1];
    X509_NAME_oneline(X509_get_subject_name(certRetrieve), line, 2000 ); // convert
    line[2000] = '\0'; // set paranoid terminator in case DN is exactly MAX_DN_SIZE long

    outputString = line;

    pem = "";

    X509_free(certRetrieve);
    BIO_free(certbio);

    return outputString;
}

std::string CX509Certificate::GetReqPEMSubject() const {
  
    X509_REQ *certRetrieve = NULL;
    BIO *certbio = NULL;

    std::string pem = stringFromVch(PEM);
    std::string outputString = "";

    certbio = BIO_new_mem_buf(pem.c_str(), -1);

    if (!(certRetrieve = PEM_read_bio_X509_REQ(certbio, NULL, NULL, NULL)))
        return "";

    char line[2000+1];
    X509_NAME_oneline(X509_REQ_get_subject_name(certRetrieve), line, 2000 ); // convert
    line[2000] = '\0'; // set paranoid terminator in case DN is exactly MAX_DN_SIZE long

    outputString = line;

    pem = "";

    X509_REQ_free(certRetrieve);
    BIO_free(certbio);

    return outputString;
}

std::string CX509Certificate::GetPEMIssuer() const {
  
    X509 *certRetrieve = NULL;
    BIO *certbio = NULL;

    std::string pem = stringFromVch(PEM);
    std::string outputString = "";

    certbio = BIO_new_mem_buf(pem.c_str(), -1);

    if (!(certRetrieve = PEM_read_bio_X509(certbio, NULL, NULL, NULL)))
        return "";

    char line[2000+1];
    X509_NAME_oneline(X509_get_issuer_name(certRetrieve), line, 2000 ); // convert
    line[2000] = '\0'; // set paranoid terminator in case DN is exactly MAX_DN_SIZE long

    outputString = line;

    pem = "";

    X509_free(certRetrieve);
    BIO_free(certbio);

    return outputString;
}

std::string CX509Certificate::GetPEMPubKey() const {
  
    X509 *certRetrieve = NULL;
    BIO *certbio = NULL;

    std::unique_ptr<BIO, decltype(&::BIO_free)> output_bio(BIO_new(BIO_s_mem()), ::BIO_free);

    BIO_reset(output_bio.get());

    std::string pem = stringFromVch(PEM);
    std::string outputString = "";

    certbio = BIO_new_mem_buf(pem.c_str(), -1);

    if (!(certRetrieve = PEM_read_bio_X509(certbio, NULL, NULL, NULL)))
        return "";

    EVP_PKEY *pkey = X509_get_pubkey(certRetrieve);
    PEM_write_bio_PUBKEY(output_bio.get(), pkey);

    BUF_MEM *mem = NULL;
    BIO_get_mem_ptr(output_bio.get(), &mem);

    if (!mem || !mem->data || !mem->length)
    {
        return "";
    }

    std::string pubkey(mem->data, mem->length);

    outputString = pubkey;    

    X509_free(certRetrieve);
    BIO_free(certbio);

    return outputString;
}

std::string CX509Certificate::GetReqPEMPubKey() const {
  
    X509_REQ *certRetrieve = NULL;
    BIO *certbio = NULL;

    std::unique_ptr<BIO, decltype(&::BIO_free)> output_bio(BIO_new(BIO_s_mem()), ::BIO_free);

    BIO_reset(output_bio.get());

    std::string pem = stringFromVch(PEM);
    std::string outputString = "";

    certbio = BIO_new_mem_buf(pem.c_str(), -1);

    if (!(certRetrieve = PEM_read_bio_X509_REQ(certbio, NULL, NULL, NULL)))
        return "";

    EVP_PKEY *pkey = X509_REQ_get_pubkey(certRetrieve);
    PEM_write_bio_PUBKEY(output_bio.get(), pkey);

    BUF_MEM *mem = NULL;
    BIO_get_mem_ptr(output_bio.get(), &mem);

    if (!mem || !mem->data || !mem->length)
    {
        return "";
    }

    std::string pubkey(mem->data, mem->length);

    outputString = pubkey;    

    X509_REQ_free(certRetrieve);
    BIO_free(certbio);

    return outputString;
}

std::string CX509Certificate::GetPEMSerialNumber() const {
    static constexpr unsigned int SERIAL_NUM_LEN = 1000;
    char serial_number[SERIAL_NUM_LEN+1];
    X509 *certRetrieve = NULL;
    BIO *certbio = NULL;
    std::string pem = stringFromVch(PEM);
    std::string outputString = "";

    certbio = BIO_new_mem_buf(pem.c_str(), -1);

    if (!(certRetrieve = PEM_read_bio_X509(certbio, NULL, NULL, NULL)))
        return "";

    ASN1_INTEGER *serial = X509_get_serialNumber(certRetrieve);

    BIGNUM *bn = ASN1_INTEGER_to_BN(serial, NULL);
    if (!bn) {
        return "";
    }

    char *tmp = BN_bn2dec(bn);
    if (!tmp) {
        fprintf(stderr, "unable to convert BN to decimal string.\n");
        BN_free(bn);
        return "";
    }

    if (strlen(tmp) >= SERIAL_NUM_LEN) {
        BN_free(bn);
        OPENSSL_free(tmp);
        return "";
    }

    strncpy(serial_number, tmp, SERIAL_NUM_LEN);
    outputString = serial_number;
    BN_free(bn);
    OPENSSL_free(tmp);
    X509_free(certRetrieve);
    BIO_free(certbio);

    return outputString;
}

bool CX509Certificate::ValidatePEMSignature(std::string& errorMessage) const
{
    OpenSSL_add_all_algorithms();
    OpenSSL_add_all_digests();
    ERR_load_BIO_strings();
    ERR_load_crypto_strings();



	EVP_PKEY* pubkeyEd25519;
	pubkeyEd25519=EVP_PKEY_new();

    X509 *certRetrieve = NULL;
    BIO *certbio = NULL;
    std::string pem = stringFromVch(PEM);
    std::string outputString = "";

    certbio = BIO_new_mem_buf(pem.c_str(), -1);

    if (!(certRetrieve = PEM_read_bio_X509(certbio, NULL, NULL, NULL))) {
        errorMessage = "Cannot retrieve X509 certificate";
        return false;
    }

	pubkeyEd25519 = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, NULL, &SubjectPublicKey[0], 32);

    EVP_PKEY *pkey = X509_get_pubkey(certRetrieve);

    // int resulttest = X509_verify(certRetrieve, pkey);

    // LogPrintf("DEBUGGER %s - resulttest: [%d]\n",__func__,resulttest);


    int outResult = 0;

//test begin-------------------------------------------------------------------------------------------------
 X509_STORE_CTX *ctx2 = X509_STORE_CTX_new();
 X509_STORE *store = X509_STORE_new();
 X509_STORE_add_cert(store, certRetrieve);
 X509_STORE_CTX_init(ctx2,store,certRetrieve, NULL);
int status  = X509_verify_cert(ctx2);

LogPrintf("DEBUGGER %s - status [%d]\n",__func__,status);

X509_STORE_free(store);
X509_STORE_CTX_free(ctx2);

LogPrintf("DEBUGGER %s - made it here 1\n",__func__);
EVP_MD_CTX* ctx = EVP_MD_CTX_new();
LogPrintf("DEBUGGER %s - made it here 2\n",__func__);
EVP_PKEY_CTX* ppctx = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, NULL);
LogPrintf("DEBUGGER %s - made it here 3\n",__func__);
int result = EVP_DigestVerifyInit(ctx, &ppctx, NULL, NULL, pkey);
if (result == 1) {
LogPrintf("DEBUGGER %s - made it here 4\n",__func__);
    const unsigned char* sigret = NULL;
    const unsigned char* tbs = NULL;
    result = EVP_DigestVerify(ctx, sigret, 64, tbs, 64);
    //result = EVP_DigestVerify(ctx, certRetrieve->signature.data, certRetrieve->signature.length, tbs, 64);
    if (result == 1) {
        // passed signature verification
        outResult = 1;
    } else {
        // failed signature verification
    }
}

LogPrintf("DEBUGGER %s - made it here 5\n",__func__);


EVP_MD_CTX_free(ctx);
EVP_PKEY_CTX_free(ppctx);

//test end---------------------------------------------------------------------------------------------------

//test2 begin-------------------------------------------------------------------------------------------------
//   unsigned char* encMessage;
//   size_t encMessageLength;
//   Base64Decode(signatureBase64, &encMessage, &encMessageLength);

//   bool authentic = false;
//   EVP_PKEY* pubKey  = EVP_PKEY_new();
//   EVP_PKEY_assign_RSA(pubKey, rsa);
//   EVP_MD_CTX* m_RSAVerifyCtx = EVP_MD_CTX_create();
//   if (EVP_DigestVerifyInit(m_RSAVerifyCtx,NULL, EVP_sha256(),NULL,pubKey)<=0) {
//     return false;
//   }
//   if (EVP_DigestVerifyUpdate(m_RSAVerifyCtx, Msg, MsgLen) <= 0) {
//     return false;
//   }
//   int AuthStatus = EVP_DigestVerifyFinal(m_RSAVerifyCtx, encMessage, encMessageLength);
//   if (AuthStatus==1) {
//     authentic = true;
//     EVP_MD_CTX_cleanup(m_RSAVerifyCtx);
//     return true;
//   } else if(AuthStatus==0){
//     authentic = false;
//     EVP_MD_CTX_cleanup(m_RSAVerifyCtx);
//     return true;
//   } else{
//     authentic = false;
//     EVP_MD_CTX_cleanup(m_RSAVerifyCtx);
//     return false;
//   }

//test2 end---------------------------------------------------------------------------------------------------




    LogPrintf("DEBUGGER %s - outResult: [%d]\n",__func__,outResult);

    if (outResult != 1) {
        errorMessage = "PEM Certificate cannot be verified";
        return false;
    }

    EVP_PKEY_free(pubkeyEd25519);
    EVP_PKEY_free(pkey);
    X509_free(certRetrieve);
    BIO_free(certbio);

    return true;
}

bool CX509Certificate::ValidatePEM(std::string& errorMessage) const
{
    std::string certSubject = stringFromVch(Subject);
    std::string certIssuer = stringFromVch(Issuer);
    std::string certPublickKey = GetPubKeyBase64();
    std::string certSerialNumber = std::to_string(SerialNumber);
    std::string BDAPprefix = "/C=US/O=Duality Blockchain Solutions/CN=";

    if (!IsApproved()) { //REQUEST
        std::string PEMSubject = GetReqPEMSubject();
        std::size_t foundSubject = PEMSubject.find(certSubject);
        if (foundSubject == std::string::npos)
        {
            errorMessage = "Certificate subject does not match PEM";
            return false;
        }

        std::string PEMPubKey = GetReqPEMPubKey();
        std::size_t foundPubKey = PEMPubKey.find(certPublickKey);
        if (foundPubKey == std::string::npos)
        {
            errorMessage = "Certificate PublicKey does not match PEM";
            return false;
        }
    }
    else { //APPROVE
        std::string PEMPubKey = GetPEMPubKey();
        std::size_t foundPubKey = PEMPubKey.find(certPublickKey);
        if (foundPubKey == std::string::npos)
        {
            errorMessage = "Certificate PublicKey does not match PEM";
            return false;
        }

        std::string PEMSerialNumber = GetPEMSerialNumber();
        if (PEMSerialNumber != certSerialNumber)
        {
            errorMessage = "Certificate SerialNumber does not match PEM";
            return false;
        }

        std::string PEMSubject = GetPEMSubject();
        std::size_t foundSubject = PEMSubject.find(certSubject);
        if (foundSubject == std::string::npos)
        {
            errorMessage = "Certificate subject does not match PEM";
            return false;
        }

        std::string PEMIssuer = GetPEMIssuer();
        std::size_t foundIssuer = PEMIssuer.find(certIssuer);
        if (foundIssuer == std::string::npos)
        {
            errorMessage = "Certificate Issuer does not match PEM";
            return false;
        }

        // if (!ValidatePEMSignature(errorMessage)) {
        //     return false;
        // }
    }

    return true;
} //ValidatePEM

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
    if (SubjectPublicKey.size() == 0)
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
    if (SubjectPublicKey.size() > MAX_CERTIFICATE_KEY_LENGTH) 
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

    // check ExternalVerificationFile
    if (ExternalVerificationFile.size() > MAX_CERTIFICATE_FILENAME) 
    {
        errorMessage = "Invalid ExternalVerificationFile. Can not have more than " + std::to_string(MAX_CERTIFICATE_FILENAME) + " characters.";
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