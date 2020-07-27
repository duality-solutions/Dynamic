// Copyright (c) 2020 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/x509.h"
#include "bdap/certificate.h"
#include "chain.h"
#include "validation.h"

#include <stdio.h>
#include <stdlib.h>

#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <openssl/evp.h>

#include <openssl/pem.h>
#include <openssl/conf.h>
#include <openssl/x509v3.h>
#ifndef OPENSSL_NO_ENGINE
#include <openssl/engine.h>
#endif

int mkcert(X509 **x509p, EVP_PKEY **pkeyp, int bits, int serial, int days,const CCertificate& certificate);
int add_ext(X509 *cert, int nid, char *value);

static void callback(int p, int n, void *arg)
	{
	char c='B';

	if (p == 0) c='.';
	if (p == 1) c='+';
	if (p == 2) c='*';
	if (p == 3) c='\n';
	fputc(c,stderr);
	}

int mkcert(X509 **x509p, EVP_PKEY **pkeyp, int bits, int serial, int days,const CCertificate& certificate)
	{
	X509 *x;
	EVP_PKEY *pk;
	RSA *rsa;
	X509_NAME *name=NULL;

    int64_t nApproveTime = 0;

	if ((pkeyp == NULL) || (*pkeyp == NULL))
		{
		if ((pk=EVP_PKEY_new()) == NULL)
			{
			abort(); 
			return(0);
			}
		}
	else
		pk= *pkeyp;

	if ((x509p == NULL) || (*x509p == NULL))
		{
		if ((x=X509_new()) == NULL)
			goto err;
		}
	else
		x= *x509p;

	rsa=RSA_generate_key(bits,RSA_F4,callback,NULL);
	if (!EVP_PKEY_assign_RSA(pk,rsa))
		{
		abort();
		goto err;
		}
	rsa=NULL;

	X509_set_version(x,2);
	//ASN1_INTEGER_set(X509_get_serialNumber(x),serial);
	//X509_gmtime_adj(X509_get_notBefore(x),0);
	//X509_gmtime_adj(X509_get_notAfter(x),(long)60*60*24*days);

    if ((unsigned int)chainActive.Height() >= certificate.nHeightApprove) {
        CBlockIndex *pindex = chainActive[certificate.nHeightApprove];
        if (pindex) {
            nApproveTime = pindex->GetBlockTime();
        }
    }

	ASN1_INTEGER_set(X509_get_serialNumber(x),certificate.SerialNumber);
	X509_gmtime_adj(X509_get_notBefore(x),(long)nApproveTime);
	X509_gmtime_adj(X509_get_notAfter(x),(long)AddMonthsToBlockTime(nApproveTime,certificate.MonthsValid));

	X509_set_pubkey(x,pk);

	name=X509_get_subject_name(x);

	/* This function creates and adds the entry, working out the
	 * correct string type and performing checks on its length.
	 * Normally we'd check the return value for errors...
	 */
	// X509_NAME_add_entry_by_txt(name,"C",
	// 			MBSTRING_ASC, "UK", -1, -1, 0);
	// X509_NAME_add_entry_by_txt(name,"CN",
	// 			MBSTRING_ASC, "OpenSSL Group", -1, -1, 0);

    X509_NAME_add_entry_by_txt(name, "C",  MBSTRING_ASC,
                            (unsigned char *)"US", -1, -1, 0);
    X509_NAME_add_entry_by_txt(name, "O",  MBSTRING_ASC,
                            (unsigned char *)"Duality Blockchain Solutions", -1, -1, 0);
    //X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
    //                        C"localhost", -1, -1, 0);



    X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
                            (unsigned char *)(stringFromVch(certificate.Subject).c_str()), -1, -1, 0);


	/* Its self signed so set the issuer name to be the same as the
 	 * subject.
	 */
	X509_set_issuer_name(x,name);

	/* Add various extensions: standard extensions */
	add_ext(x, NID_basic_constraints, "critical,CA:TRUE");
	add_ext(x, NID_key_usage, "critical,keyCertSign,cRLSign");

	add_ext(x, NID_subject_key_identifier, "hash");

	/* Some Netscape specific extensions */
	add_ext(x, NID_netscape_cert_type, "sslCA");

	add_ext(x, NID_netscape_comment, "example comment extension");


#ifdef CUSTOM_EXT
	/* Maybe even add our own extension based on existing */
	{
		int nid;
		nid = OBJ_create("1.2.3.4", "MyAlias", "My Test Alias Extension");
		X509V3_EXT_add_alias(nid, NID_netscape_comment);
		add_ext(x, nid, "example comment alias");
	}
#endif
	
	if (!X509_sign(x,pk,EVP_md5()))
		goto err;

	*x509p=x;
	*pkeyp=pk;
	return(1);
err:
	return(0);
	}

/* Add extension using V3 code: we can set the config file as NULL
 * because we wont reference any other sections.
 */

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

bool ExportX509Certificate(const CCertificate& certificate, const std::string& filename) {

	BIO *bio_err;
	X509 *x509=NULL;
	EVP_PKEY *pkey=NULL;

	CRYPTO_mem_ctrl(CRYPTO_MEM_CHECK_ON);

	//bio_err=BIO_new_fp(stderr, BIO_NOCLOSE);

	mkcert(&x509,&pkey,512,0,365,certificate);

	//RSA_print_fp(stdout,pkey->pkey.rsa,0);
	//X509_print_fp(stdout,x509);

    FILE * x509File;
    x509File = fopen("x509.pem", "wb");

	//PEM_write_PrivateKey(stdout,pkey,NULL,NULL,0,NULL, NULL);
	PEM_write_X509(x509File,x509);

    fclose(x509File);
	X509_free(x509);
	EVP_PKEY_free(pkey);

#ifndef OPENSSL_NO_ENGINE
	ENGINE_cleanup();
#endif
	CRYPTO_cleanup_all_ex_data();

	//CRYPTO_mem_leaks(bio_err);
	BIO_free(bio_err);

    return true;


}